#!/usr/bin/env python3
"""
Single-sample agentic inference: multi-turn OTA loop with Gemini + VideoEnv.

Usage:
    python inference/single_agentic_sample_inference_gemini_v2.py \
        --video_path assets/example_video_mcq.mp4 \
        --question 'Who or what lauds "Immigrant Diaries" as "A SURE FIRE HIT", according to the video?' \
        --options "A. Remote Goat.\\nB. The New York Times.\\nC. Variety.\\nD. IndieWire." \
        --answer "A" \
        --processor_path /path/to/Qwen2.5-Omni-7B

Requires DASHSCOPE_API_KEY (or OSS_ACCESS_KEY) in env or .env file.
Requires a local Qwen2.5-Omni-7B processor for token counting.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
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
    print("⛔️ Please set OSS_ACCESS_KEY or DASHSCOPE_API_KEY in .env")
    sys.exit(1)

# API configuration - Native protocol endpoint
BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")

# Default model
DEFAULT_MODEL = "vertex_ai.gemini-3.1-pro-preview"

# ========= DashScope Call (Native Protocol & Unified Logic) =========
import base64

def _file_part(url: str, mime: str) -> dict:
    """Convert a local file path to inline base64, or keep as fileUri for remote URLs."""
    if url.startswith(("http://", "https://", "oss://")):
        return {"fileData": {"mimeType": mime, "fileUri": url}}
    with open(url, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return {"inlineData": {"mimeType": mime, "data": data}}

def gemini_call(messages: List[Dict[str, Any]],
                model: str = DEFAULT_MODEL, 
                retries: int = 100, 
                interval: float = 0.5) -> str:
    """
    Call Gemini via Native Protocol.
    Handles both raw environment format (video/audio) and generic format (video_url/audio_url) in this layer.
    """

    from pydantic import BaseModel, Field
    from typing import List, Union, Literal, Annotated, Optional

        # --- Sub-action definitions: each action only has its own required fields ---
    class Action(BaseModel):
        type: Literal["get_frames", "get_audio", "get_clip", "answer"] = Field(
            ..., 
            description="The tool to use. 'get_frames' needs 'timestamps'. 'get_audio'/'get_clip' need 'start'/'end'. 'answer' needs 'content'."
        )
        timestamps: Optional[List[float]] = Field(
            None, 
            description="Required ONLY for 'get_frames'. List of seconds."
        )
        start: Optional[float] = Field(
            None, 
            description="Required for 'get_audio' or 'get_clip'. Start time in seconds."
        )
        end: Optional[float] = Field(
            None, 
            description="Required for 'get_audio' or 'get_clip'. End time in seconds."
        )
        content: Optional[str] = Field(
            None, 
            description="Required ONLY for 'answer'. Final answer text or option."
        )

    class AgentResponse(BaseModel):
        observation: str = Field(
            ..., 
            description="Detailed log of visual/auditory evidence found in this or previous steps."
        )
        think: str = Field(
            ..., 
            description="Reasoning about the evidence and planning the next action."
        )
        action: Action

    action_schema = AgentResponse.model_json_schema()
    print("action_schema: ", action_schema)

    native_contents = []
    system_instruction = None

    for msg in messages:
        # 1. Extract System Prompt as top-level field
        if msg['role'] == "system":
            system_text = ""
            if isinstance(msg['content'], str):
                system_text = msg['content']
            elif isinstance(msg['content'], list):
                system_text = "".join([item.get('text', '') for item in msg['content'] if item.get('type') == 'text'])
            system_instruction = {"parts": [{"text": system_text}]}
            continue

        # 2. Role mapping: assistant -> model
        role = "user" if msg['role'] == "user" else "model"
        parts = []
        
        content_items = msg['content'] if isinstance(msg['content'], list) else [{"type": "text", "text": msg['content']}]
        
        for item in content_items:
            item_type = item.get('type')
            
            if item_type == 'text':
                parts.append({"text": item.get('text', '')})
            
            # --- Core merge logic: directly recognize raw environment keys or intermediate keys ---
            elif item_type in ['video', 'video_url']:
                # Compatible with environment format item['video'] and possible item['video_url']['url']
                url = item.get('video') or (item.get('video_url', {}).get('url') if isinstance(item.get('video_url'), dict) else None)
                if url:
                    parts.append(_file_part(url, "video/mp4"))
            
            elif item_type in ['audio', 'audio_url']:
                url = item.get('audio') or (item.get('audio_url', {}).get('url') if isinstance(item.get('audio_url'), dict) else None)
                if url:
                    parts.append(_file_part(url, "audio/mp3"))
            
            elif item_type in ['image', 'image_url']:
                url = item.get('image') or (item.get('image_url', {}).get('url') if isinstance(item.get('image_url'), dict) else None)
                if url:
                    parts.append(_file_part(url, "image/jpeg"))
        
        if parts:
            native_contents.append({"role": role, "parts": parts})

    print(f"system_instruction: [{system_instruction}]")
    print(f"native_contents: [{native_contents}]")
    # Build Native protocol payload
    payload = {
        "model": model,
        "dashscope_extend_params": {
            "using_native_protocol": True,
            'X-DashScope-DataInspection': '{"input":"disable","output":"disable"}'
        },
        "generationConfig": {
            "maxOutputTokens": 32768,
            # "responseMimeType": "application/json",  # Must be set to application/json
            # "responseSchema": action_schema,         # Pass in the defined Schema
            "thinkingConfig": {
                "includeThoughts": True,
                # "thinkingBudget": -1,
                "thinkingLevel": "high"
            }
        },
        "stream": True,
        "contents": native_contents,
    }

    if system_instruction:
        payload["system_instruction"] = system_instruction

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    QPS_LIMIT_SLEEP = 5

    for i in range(retries):
        try:
            stream_output = ""
            resp = requests.post(BASE_URL, json=payload, headers=headers, timeout=300, stream=True)
            
            if resp.status_code == 429:
                time.sleep(QPS_LIMIT_SLEEP + random.uniform(0, 5))
                continue

            resp.raise_for_status()

            for line in resp.iter_lines():
                if line:
                    # print(f"line-{line}")
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        try:
                            data = json.loads(decoded_line[6:])
                            if "candidates" in data and len(data["candidates"]) > 0:
                                for candidate in data["candidates"]:
                                    if "content" in candidate and "parts" in candidate["content"]:
                                        for part in candidate["content"]["parts"]:
                                            # Filter thinking process, keep only final response text
                                            if not part.get("thought", False):
                                                stream_output += part.get("text", "")
                        except json.JSONDecodeError:
                            continue

            # Core improvement: double safety net
            try:
                # Parse possibly multi-line JSON string and convert to single-line
                data = json.loads(stream_output)
                return json.dumps(data, ensure_ascii=False) # Force single-line JSON without indentation
            except json.JSONDecodeError:
                # If parsing fails (very rare), return as-is or handle the error
                return stream_output.replace("\n", " ").strip()            

        except Exception as e:
            logging.warning("API error (%d/%d): %s", i+1, retries, e)
            print(resp)
            if i < retries - 1:
                time.sleep(interval)
    
    raise RuntimeError("Gemini native call failed repeatedly")

# ========= Video Analysis Tools (Keep details) =========
def get_video_info(video_path: str) -> Dict[str, Any]:
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

# ========= Configuration Classes (Keep details) =========
class Config:
    def __init__(self, max_steps=10, max_frames_len=5, max_audio_len=10.0, max_clip_len=10.0, mode="OmniAgent"):
        self.env = EnvConfig(max_steps, max_frames_len, max_audio_len, max_clip_len, mode)

class EnvConfig:
    def __init__(self, max_steps=10, max_frames_len=5, max_audio_len=10.0, max_clip_len=10.0, mode="OmniAgent"):
        self.env_name = "video_env"
        self.seed = 42
        self.max_steps = max_steps
        self.rollout = RolloutConfig()
        self.video_star = VideoStarConfig(max_frames_len, max_audio_len, max_clip_len, mode)

class RolloutConfig:
    def __init__(self): self.n = 1

class VideoStarConfig:
    def __init__(self, max_frames_len=5, max_audio_len=10.0, max_clip_len=10.0, mode="OmniAgent"):
        self.max_frames_len = max_frames_len
        self.max_audio_len = max_audio_len
        self.max_clip_len = max_clip_len
        self.mode = mode

# ========= Agent Implementation =========
class GeminiVideoAgent:
    def __init__(self, config, model=DEFAULT_MODEL):
        self.config = config
        self.model = model
        self._build_environment()

    def _build_environment(self):
        val_envs = build_video_envs(
            seed=self.config.env.seed + 1000, env_num=1, group_n=1,
            max_frames_len=self.config.env.video_star.max_frames_len,
            max_audio_len=self.config.env.video_star.max_audio_len,
            max_clip_len=self.config.env.video_star.max_clip_len,
            max_steps=self.config.env.max_steps,
            processor_path=os.getenv("PROCESSOR_PATH", "/path/to/Qwen2.5-Omni-7B"),
            max_prompt_len=102400, max_response_len=4096, is_train=False
        )
        projection_f = partial(lambda actions, *args, **kwargs: (actions, [True] * len(actions)))
        self.val_envs = VideoEnvironmentManager(val_envs, projection_f, self.config)

    def run_episode(self, video_data_batch: Dict[str, Any]) -> Dict[str, Any]:
        obs, infos = self.val_envs.reset(video_data_batch)
        done, step_count, total_reward = False, 0, 0.0
        
        print(f"Question: {video_data_batch['question'][0]}")

        while not done:
            # Extract environment messages directly; gemini_call handles format compatibility
            messages = obs['text'][0]

            print(f"Step {step_count+1} - Calling Gemini...")
            response = gemini_call(messages, model=self.model, retries=500)
            print(f"Step {step_count+1} - Agent response: {response}\n")
            
            next_obs, rewards, dones, extra_info = self.val_envs.step([response])
            
            obs = next_obs
            done = bool(dones[0])
            total_reward += float(rewards[0])
            step_count += 1
        
        return {"final_observation": obs, "total_reward": total_reward, "steps": step_count}

def main():
    parser = argparse.ArgumentParser(description="Single-sample agentic inference with Gemini")
    parser.add_argument("--video_path", type=str, default=None,
                        help="Path to video file. Falls back to VIDEO_PREFIX + SAMPLE_VIDEO_PATH env vars")
    parser.add_argument("--question", type=str, default="What is the relationship between the woman and the monster in the video?")
    parser.add_argument("--answer", type=str, default="F")
    parser.add_argument("--options", type=str, default="A. enemy\nB. friend\nF. stranger",
                        help="Options separated by \\n")
    parser.add_argument("--question_type", type=str, default="MCQ", choices=["MCQ", "TR", "FF", "NUM", "SIZE"])
    parser.add_argument("--max_steps", type=int, default=32)  # test-time scaling: try 12, 22, 32, 42, 52
    parser.add_argument("--max_frames_len", type=int, default=32)
    parser.add_argument("--max_audio_len", type=float, default=120)
    parser.add_argument("--max_clip_len", type=float, default=60)
    parser.add_argument("--mode", type=str, default="OmniAgent")
    parser.add_argument("--processor_path", type=str, default=None,
                        help="Path to Qwen2.5-Omni-7B processor. Falls back to PROCESSOR_PATH env var")
    args = parser.parse_args()

    if args.processor_path:
        os.environ["PROCESSOR_PATH"] = args.processor_path

    if args.video_path:
        SAMPLE_URL = args.video_path
    else:
        VIDEO_PREFIX = os.getenv("VIDEO_PREFIX", "/path/to/Video-Holmes")
        video_path = os.getenv("SAMPLE_VIDEO_PATH", "./videos_cropped/example.mp4")
        SAMPLE_URL = os.path.join(VIDEO_PREFIX, video_path)

    from agent_system.environments.env_package.oss_reader import OssReader
    oss_reader = OssReader()
    os.environ["USE_OSS_IN_VIDEOENV"] = os.getenv("USE_OSS_IN_VIDEOENV", str(oss_reader.enabled))
    os.environ["USE_DYNAMIC_STEP"] = "True"
    os.environ["DELETE_MEDIA_WHEN_ERROR"] = "True"

    video_info = get_video_info(SAMPLE_URL)
    agent = GeminiVideoAgent(Config(
        max_steps=args.max_steps, max_frames_len=args.max_frames_len,
        max_audio_len=args.max_audio_len, max_clip_len=args.max_clip_len, mode=args.mode,
    ))

    options_list = [opt.strip() for opt in args.options.replace("\\n", "\n").split("\n") if opt.strip()]
    video_data_batch = {
        'video': [SAMPLE_URL], 'question_type': [args.question_type],
        'question': [args.question],
        'answer': [args.answer], 'options': [options_list],
        'fps': [video_info['fps']], 'duration_seconds': [video_info['duration']],
        'has_audio': [video_info['has_audio']],
    }

    result = agent.run_episode(video_data_batch)
    print(f"\n=== Result: Reward {result['total_reward']}, Steps {result['steps']} ===")

if __name__ == "__main__":
    main()
