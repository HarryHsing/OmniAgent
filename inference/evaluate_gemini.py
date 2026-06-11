#!/usr/bin/env python3
# Evaluate Gemini on Video-MME dataset

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
import uuid
import random
from functools import partial
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import ray
import requests
from dotenv import load_dotenv

from agent_system.environments.env_manager import VideoEnvironmentManager
# Import your existing environment
from agent_system.environments.env_package.video_env import build_video_envs

# ========= Configuration =========
load_dotenv()

# Get API key from environment variables
API_KEY = os.getenv("OSS_ACCESS_KEY")
if not API_KEY:
    API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not API_KEY:
    print("⛔️ Please set OSS_ACCESS_KEY or DASHSCOPE_API_KEY in .env or environment variables")
    sys.exit(1)

# API configuration
BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")

# Default model
DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_PROCESSOR_PATH = os.getenv("PROCESSOR_PATH", "/path/to/Qwen2.5-Omni-7B")
DEFAULT_DATASET_PATH = os.getenv("DATASET_PATH", "/path/to/Video-MME.json")
DEFAULT_VIDEO_PREFIX = os.getenv("VIDEO_PREFIX", "/path/to/videomme")

# Default hyperparameters
DEFAULT_MAX_STEPS = 32
DEFAULT_MAX_FRAMES_LEN = 8
DEFAULT_MAX_AUDIO_LEN = 15.0
DEFAULT_MAX_CLIP_LEN = 10.0
DEFAULT_MODE = "OmniAgent"

# ======================== Deep Optimization: Environment-Aligned Cleaning Logic ========================
def robust_clean(text: str) -> str:
    s = text.strip()
    # 1. Prefer extracting content inside Markdown code blocks (for Gemini's habit of wrapping in ```json)
    code_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', s, re.DOTALL)
    if code_block:
        s = code_block.group(1).strip()
    else:
        # 2. If no code block found, look for the outermost {}
        json_match = re.search(r'(\{.*\})', s, re.DOTALL)
        if json_match:
            s = json_match.group(1).strip()
    
    # 3. Verify it meets the strict requirements of the environment code
    try:
        # Must be valid JSON with curly braces as first and last chars (aligned with environment check)
        json.loads(s)
        if s.startswith("{") and s.endswith("}"):
            # 4. Compress to single line (aligned with the environment's "ONE single line" requirement)
            return "".join(s.splitlines())
    except:
        pass
    return text # On failure, return as-is and let the environment raise standard error logs
# =============================================================================

# ========= DashScope Call (Native Protocol & Streaming Logic) =========
def gemini_call(messages: List[Dict[str, Any]], 
                model: str = DEFAULT_MODEL, 
                retries: int = 10, 
                interval: float = 0.5,
                think_level: str = "high") -> str: # Added think_level parameter
                
    """
    Call Gemini via DashScope Native Protocol with Thinking and streaming support.
    """
    
    api_content = []
    system_instruction = None 
    
    for msg in messages:
        if msg['role'] == "system":
            system_text = ""
            if isinstance(msg['content'], str):
                system_text = msg['content']
            elif isinstance(msg['content'], list):
                system_text = "".join([item.get('text', '') for item in msg['content'] if item.get('type') == 'text'])
            
            system_instruction = {"parts": [{"text": system_text}]}
            continue 
            
        role = "user" if msg['role'] == "user" else "model"
        parts = []
        
        content_items = msg['content'] if isinstance(msg['content'], list) else [{"type": "text", "text": msg['content']}]
        
        for item in content_items:
            item_type = item.get('type')
            if item_type == 'text':
                parts.append({"text": item.get('text', '')})
            elif item_type in ['video', 'video_url']:
                url = item.get('video') or (item.get('video_url', {}).get('url') if isinstance(item.get('video_url'), dict) else None)
                if url:
                    parts.append({"fileData": {"mimeType": "video/mp4", "fileUri": url}})
            elif item_type in ['audio', 'audio_url']:
                url = item.get('audio') or (item.get('audio_url', {}).get('url') if isinstance(item.get('audio_url'), dict) else None)
                if url:
                    parts.append({"fileData": {"mimeType": "audio/mp3", "fileUri": url}})
            elif item_type in ['image', 'image_url']:
                url = item.get('image') or (item.get('image_url', {}).get('url') if isinstance(item.get('image_url'), dict) else None)
                if url:
                    parts.append({"fileData": {"mimeType": "image/jpeg", "fileUri": url}})
        
        if parts:
            api_content.append({"role": role, "parts": parts})

    payload = {
        "model": model,
        "dashscope_extend_params": {
            "using_native_protocol": True,
            'X-DashScope-DataInspection': '{"input":"disable","output":"disable"}'
        },
        "generationConfig": {
            "maxOutputTokens": 1024*32,
            "thinkingConfig": {
                "includeThoughts": True,
                # "thinkingBudget": -1,
                "thinkingLevel": think_level # high, low
            }
        },
        "stream": True,
        "contents": api_content,
    }

    if system_instruction:
        payload["system_instruction"] = system_instruction

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    for attempt in range(retries):
        try:
            response = requests.post(BASE_URL, json=payload, headers=headers, timeout=300, stream=True)
            
            if response.status_code == 429:
                time.sleep(interval)
                continue

            response.raise_for_status()

            stream_output = ""
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        decoded_line = decoded_line[len("data: "):]
                        try:
                            json_line = json.loads(decoded_line)
                            if "candidates" in json_line and len(json_line["candidates"]) > 0:
                                for candidate in json_line["candidates"]:
                                    if "content" in candidate and "parts" in candidate["content"]:
                                        for part in candidate["content"]["parts"]:
                                            if not part.get("thought", False):
                                                stream_output += part.get("text", "")
                        except json.JSONDecodeError:
                            continue

            # --- Diagnostic logic ---
            final_res = stream_output.strip()
            if not final_res and attempt < retries - 1:
                logging.warning(f"⚠️ Empty output! [{response.raise_for_status()}] Retrying...")
                # If blocked for safety reasons, try simplifying the prompt or skip
                time.sleep(interval)
                continue

            return stream_output

        except requests.exceptions.Timeout:
            logging.warning("Timeout Error. Retrying...")
            continue
        except Exception as e:
            logging.warning("Submission Error: %s", e)
            time.sleep(interval)
    
    raise RuntimeError("Gemini native call failed repeatedly")


# ========= Video Analysis Tools =========
def get_video_info(video_path: str) -> Dict[str, Any]:
    try:
        cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "default=nw=1", video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip().split('=')[1])
        
        cmd = ["ffprobe", "-v", "quiet", "-show_entries", "stream=r_frame_rate", "-select_streams", "v:0", "-of", "default=nw=1", video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        fps_str = result.stdout.strip().split('=')[1]
        fps = eval(fps_str) if '/' in fps_str else float(fps_str)
        
        cmd = ["ffprobe", "-v", "quiet", "-show_entries", "stream=codec_type", "-of", "default=nw=1", video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        has_audio = "audio" in result.stdout
        
        return {"duration": duration, "fps": fps, "has_audio": has_audio}
    except Exception as e:
        return {"duration": 10.0, "fps": 30.0, "has_audio": True}

# ========= Configuration Classes =========
class Config:
    def __init__(self, max_steps=DEFAULT_MAX_STEPS, max_frames_len=DEFAULT_MAX_FRAMES_LEN, max_audio_len=DEFAULT_MAX_AUDIO_LEN, 
                 max_clip_len=DEFAULT_MAX_CLIP_LEN, mode=DEFAULT_MODE):
        self.env = EnvConfig(max_steps, max_frames_len, max_audio_len, max_clip_len, mode)

class EnvConfig:
    def __init__(self, max_steps=DEFAULT_MAX_STEPS, max_frames_len=DEFAULT_MAX_FRAMES_LEN, max_audio_len=DEFAULT_MAX_AUDIO_LEN, 
                 max_clip_len=DEFAULT_MAX_CLIP_LEN, mode=DEFAULT_MODE):
        self.env_name = "video_env"
        self.seed = 42
        self.max_steps = max_steps
        self.rollout = RolloutConfig()
        self.video_star = VideoStarConfig(max_frames_len, max_audio_len, max_clip_len, mode)

class RolloutConfig:
    def __init__(self): self.n = 1

class VideoStarConfig:
    def __init__(self, max_frames_len=DEFAULT_MAX_FRAMES_LEN, max_audio_len=DEFAULT_MAX_AUDIO_LEN, max_clip_len=DEFAULT_MAX_CLIP_LEN, mode=DEFAULT_MODE):
        self.max_frames_len = max_frames_len
        self.max_audio_len = max_audio_len
        self.max_clip_len = max_clip_len
        self.mode = mode

# ========= Agent Implementation =========
class GeminiVideoAgent:
    def __init__(self, config, model=DEFAULT_MODEL, processor_path=DEFAULT_PROCESSOR_PATH):
        self.config = config
        self.model = model
        self.processor_path = processor_path
        self._build_environment()

    def _build_environment(self):
        val_envs = build_video_envs(
            seed=self.config.env.seed + 1000, env_num=1, group_n=1,
            max_frames_len=self.config.env.video_star.max_frames_len,
            max_audio_len=self.config.env.video_star.max_audio_len,
            max_clip_len=self.config.env.video_star.max_clip_len,
            max_steps=self.config.env.max_steps,
            processor_path=self.processor_path,
            max_prompt_len=102400, max_response_len=4096, is_train=False
        )
        projection_f = partial(lambda actions, *args, **kwargs: (actions, [True] * len(actions)))
        self.val_envs = VideoEnvironmentManager(val_envs, projection_f, self.config)

    def run_episode(self, video_data_batch: Dict[str, Any], question_id: str, think_level: str = "high") -> str: # Added think_level parameter
        obs, _ = self.val_envs.reset(video_data_batch)
        traj_id, step_logs, total_reward, step_count, done = str(uuid.uuid4()), [], 0.0, 0, False

        while not done:
            raw_input = obs['text'][0]
            # Here raw_input is the complete history up to the current point
            # print(f"Step {step_count}-INPUT: {raw_input}")
            output = gemini_call(raw_input, model=self.model, think_level=think_level) 
            # print(f"Step {step_count}-OUTPUT: {output}")
            # print(f"Step {step_count}")

            # --- Optionally enable robust cleaning based on toggle ---
            if os.getenv("ENABLE_ROBUST_CLEAN", "false").lower() in ("1", "true"):
                output = robust_clean(output) # Call the regex-based cleaning function

            next_obs, rewards, dones, extra_info = self.val_envs.step([output])
            step_reward = float(rewards[0])
            total_reward += step_reward
            step_count += 1
            done = bool(dones[0])

            step_logs.append({
                "question_id": question_id, "traj_id": traj_id, "step": step_count,
                "raw_input": raw_input, "output": output, "step_reward": step_reward,
                "done": dones[0], "extra_info": extra_info[0],
            })
            obs = next_obs

        for log in step_logs: log["episode_reward"] = total_reward
        return {
            "final_observation": obs, "total_reward": total_reward, "steps": step_count,
            "traj_id": traj_id, "step_logs": step_logs, "done": dones[0], 
            "win": extra_info[0]["won"], "extra_info": extra_info[0],
        }

# ========= Utilities =========
def generate_results_filename(model_name: str, max_steps: int, max_frames_len: int,
                             max_audio_len: float, max_clip_len: float, mode: str) -> str:
    sanitized_model = model_name.replace("/", "-")
    filename = f"Video-MME_{sanitized_model}_steps{max_steps}_frames{max_frames_len}_audio{max_audio_len}_clip{max_clip_len}_{mode}.json"
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", filename)

def evaluate_dataset(dataset_path: str, video_prefix: str, results_path: str, model_name: str,
                   max_steps: int, max_frames_len: int, max_audio_len: float, max_clip_len: float, mode: str,
                   max_samples: int = None, sft_attempts: int = 1, processor_path: str = DEFAULT_PROCESSOR_PATH):
    with open(dataset_path, 'r') as f: dataset = json.load(f)
    if max_samples: dataset = dataset[:max_samples]
    
    agent = GeminiVideoAgent(
        Config(max_steps, max_frames_len, max_audio_len, max_clip_len, mode),
        model=model_name,
        processor_path=processor_path,
    )
    step_log_path = results_path.replace(".json", "_steps.jsonl")
    step_log_f = open(step_log_path, "a", encoding="utf-8")
    
    results, processed_indices = [], set()
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r') as f: results = json.load(f)
            # processed_indices = {r['index'] for r in results}
            processed_indices = {
                str(r.get('index') if r.get('index') is not None else r.get('qid'))
                for r in results 
                if r.get('index') is not None or r.get('qid') is not None
            }
        except: pass

    for sample in dataset:
    # Unified identifier extraction
        current_id = str(sample.get('index') if sample.get('index') is not None else sample.get('qid'))
        print(f"Processing {current_id}...")
        if current_id in processed_indices:
            continue
        
        relative_video_path = sample.get('video_path') if sample.get('video_path') is not None else sample.get('video')
        video_path = os.path.join(video_prefix, relative_video_path.lstrip('./')) if video_prefix else relative_video_path
        if not os.path.exists(video_path): continue

        info = get_video_info(video_path)
        question = sample.get('question') if sample.get('question') is not None else sample.get('problem')
        question_type = sample['question_type']
        if question_type == "tr":
            question_type = "TR"
        if question_type == "multiple-choice":
            question_type = "MCQ"
        if question_type == "size":
            question_type = "SIZE"
        batch = {
            'video': [video_path], 'question_type': [question_type], 'question': [question],
            'answer': [sample['answer']], 'options': [sample['options']], 'fps': [info['fps']],
            'duration_seconds': [info['duration']], 'has_audio': [info['has_audio']],
        }
        
        best_result = None
        for outer in range(3):
            traj_results = []
            max_try = sft_attempts if sft_attempts != -1 else (len(sample['options'])+1 if question_type=="MCQ" else 5)
            for attempt in range(1, max_try + 1):
                try:
                    res = agent.run_episode(batch, f"v_{current_id}_{outer}_{attempt}")
                    traj_results.append(res)
                    if res["win"]:
                        best_result = res
                        break
                    if not best_result or res["total_reward"] > best_result["total_reward"]: 
                        best_result = res
                except Exception: continue
            # if best_result and best_result["win"]: break
            if best_result is not None:
                break
        
        if not best_result: continue
        processed_indices.add(current_id)

        for res in traj_results:
            for log in res["step_logs"]: 
                step_log_f.write(json.dumps(log, ensure_ascii=False, default=str) + "\n")
        step_log_f.flush()

        # --- Core improvement: align output format with the provided Example ---
        results.append({
            "index": current_id,
            "question": question,
            "answer": sample['answer'],
            "options": sample['options'],
            "output": str(best_result['final_observation']), # Contains the complete history (text list)
            "total_reward": best_result['total_reward'],
            "steps": best_result['steps'],
            "duration": sample.get('duration', 'No'),
            "attempts": len(traj_results),
            "done": str(best_result["done"]),
            "win": best_result["win"],
            "extra_info": best_result['extra_info'] # Contains video metadata, mode, and other details
        })
        
        with open(results_path + ".tmp", 'w') as f: 
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        os.replace(results_path + ".tmp", results_path)

    step_log_f.close()
    return results

def connect_to_ray():
    addr = os.environ.get("RAY_ADDRESS")
    
    # Add random delay of 0~3 seconds to spread concurrent load
    if addr:
        time.sleep(random.uniform(0, 3))
    
    # Original logic
    try:
        ray.init(
            address="auto" if addr else None, 
            local_mode=not addr, 
            namespace="omni_eval_ws"
        )
    except Exception as e:
        # If still failing, wait and retry once more
        print(f"Ray connection failed, retrying... Error: {e}")
        time.sleep(5)
        ray.init(address="auto" if addr else None, namespace="omni_eval_ws")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--video_prefix", type=str, default=DEFAULT_VIDEO_PREFIX)
    parser.add_argument("--processor_path", type=str, default=DEFAULT_PROCESSOR_PATH)
    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)  # test-time scaling: try 12, 22, 32, 42, 52
    parser.add_argument("--max_frames_len", type=int, default=DEFAULT_MAX_FRAMES_LEN)
    parser.add_argument("--max_audio_len", type=float, default=DEFAULT_MAX_AUDIO_LEN)
    parser.add_argument("--max_clip_len", type=float, default=DEFAULT_MAX_CLIP_LEN)
    parser.add_argument("--mode", type=str, default=DEFAULT_MODE)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--sft_attempts", type=int, default=1)
    
    args = parser.parse_args()
    if not args.results_path:
        args.results_path = generate_results_filename(args.model, args.max_steps, args.max_frames_len, args.max_audio_len, args.max_clip_len, args.mode)
    
    if os.path.dirname(args.results_path): os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
    
    # Call at program entry point
    connect_to_ray()

    evaluate_dataset(args.dataset_path, args.video_prefix, args.results_path, args.model,
                     args.max_steps, args.max_frames_len, args.max_audio_len, args.max_clip_len, args.mode,
                     args.max_samples, args.sft_attempts, args.processor_path)

if __name__ == "__main__":
    main()
