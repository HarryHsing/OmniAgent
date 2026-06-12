import argparse
import json
import os
import time
import ray
import sys
import uuid
import random
import numpy as np
from enum import Enum
from pathlib import Path
from functools import partial

# Ensure evaluate_gemini classes and functions can be found
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from evaluate_gemini import (
    GeminiVideoAgent, Config, get_video_info, gemini_call
)
from agent_system.environments.env_package.video_env import get_global_processor

# ================== JSON Compatibility Handling ==================
class MyEncoder(json.JSONEncoder):
    """Handle NumPy types, compatible with NumPy 1.x and 2.x"""
    def default(self, obj):
        # Use base class checks to cover all precision levels of floats and integers
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)): # Compatible with numpy bool and python bool
            return bool(obj)
        elif isinstance(obj, Enum):
            return obj.name
        return super(MyEncoder, self).default(obj)

# ================== Path and Filename Logic ==================
def extract_dataset_name(dataset_path):
    return os.path.splitext(os.path.basename(dataset_path))[0]

def generate_results_filename(dataset_path, model_name, think_level, max_steps, max_frames_len, 
                             max_audio_len, max_clip_len, mode, num_processes):
    dataset_name = extract_dataset_name(dataset_path)
    sanitized_model = model_name.replace("/", "-")
    
    # 1. Resource allocation mode
    is_rand = os.getenv("RANDOM_RESOURCE_SFT", "false").lower() in ("1", "true", "yes")
    is_dyn  = os.getenv("USE_DYNAMIC_STEP", "false").lower() in ("1", "true", "t", "yes", "y")
    
    # 2. Export/processing toggles (new)
    # RC: Whether regex-based robust cleaning is enabled
    is_rc   = os.getenv("ENABLE_ROBUST_CLEAN", "false").lower() in ("1", "true")

    
    # Build suffix tag
    tag = ""
    if is_rand: tag += "-RAND"
    if is_dyn:  tag += "-DYN"
    if is_rc:   tag += "-RC" # Robust Clean enabled tag
    
    res_str = f"s{max_steps}_f{max_frames_len}_a{max_audio_len}_c{max_clip_len}"
    
    # Assemble final filename
    # Example: Video-MME_gemini-3-pro_high_s32_f8_a15_c10_Omni_RAND-DYN-RC-FT_p16_v3.json
    return f"{dataset_name}_{sanitized_model}_{think_level}_{res_str}_{mode}{tag}_p{num_processes}_v3.json"
    
# ================== Ray Worker Actor ==================
@ray.remote(num_cpus=1)
class EvaluationWorker:
    def __init__(self, config_args):
        # Internally use the Config class from evaluate_gemini.py
        self.config = Config(
            max_steps=config_args.max_steps,
            max_frames_len=config_args.max_frames_len,
            max_audio_len=config_args.max_audio_len,
            max_clip_len=config_args.max_clip_len,
            mode=config_args.mode
        )
        # Initialize Agent (includes build_video_envs internally)
        self.agent = GeminiVideoAgent(self.config, model=config_args.model, processor_path=config_args.processor_path)
        self.video_prefix = config_args.video_prefix
        self.sft_attempts = config_args.sft_attempts
        self.think_level = config_args.think_level # 1. Store this parameter

    def process_sample(self, sample):
        try:
            # 1. Unified ID extraction (aligned with evaluate_gemini logic)
            current_id = str(sample.get('index') if sample.get('index') is not None else sample.get('qid'))
            
            # 2. Video path processing
            relative_video_path = sample.get('video_path') or sample.get('video')
            video_path = os.path.join(self.video_prefix, relative_video_path.lstrip('./')) if self.video_prefix else relative_video_path
            
            if not os.path.exists(video_path):
                return None, f"Video not found: {video_path}"

            # 3. Extract metadata and map Question Type
            info = get_video_info(video_path)
            question = sample.get('question') or sample.get('problem')
            # question_type = sample.get('question_type')
            
            # 1. Get raw field (handle None or missing cases)
            raw_question_type = sample.get("question_type")

            # 2. Extract prefix and convert to lowercase
            #    e.g. "multiple-choice_1" -> "multiple-choice"
            prefix = str(raw_question_type).split("_", 1)[0].lower()

            # 3. Define mapping table (keys in lowercase)
            q_mapping = {
                "tr": "TR", 
                "multiple-choice": "MCQ", 
                "size": "SIZE",
                "num": "NUM",
                "ff": "FF"
            }

            # 4. Map and finalize question_type, output in uppercase
            question_type = q_mapping.get(prefix, prefix).upper()

            batch = {
                'video': [video_path], 'question_type': [question_type], 'question': [question],
                'answer': [sample['answer']], 'options': [sample['options']], 'fps': [info['fps']],
                'duration_seconds': [info['duration']], 'has_audio': [info['has_audio']],
            }

            # 4. Retry logic (1 outer * N sft_attempts)
            best_result = None
            traj_results = []
            max_try = self.sft_attempts if self.sft_attempts != -1 else (len(sample['options'])+1 if question_type=="MCQ" else 5)
            
            found_win = False
            for outer in range(1):
                for attempt in range(1, max_try + 1):
                    # Call run_episode (aligned with original script)
                    res = self.agent.run_episode(batch, f"v_{current_id}_{outer}_{attempt}", self.think_level)
                    traj_results.append(res)
                    if res["win"]:
                        best_result = res
                        found_win = True
                        break
                    if not best_result or res["total_reward"] > best_result["total_reward"]: 
                        best_result = res
                if found_win: break

            if best_result:
                # 5. Build result dict (format fully aligned with evaluate_gemini)
                output_entry = {
                    "index": current_id,
                    "question": question,
                    "answer": sample['answer'],
                    "options": sample['options'],
                    "output": str(best_result['final_observation']),
                    "total_reward": best_result['total_reward'],
                    "steps": best_result['steps'],
                    "duration": sample.get('duration', 'No'),
                    "attempts": len(traj_results),
                    "done": str(best_result["done"]),
                    "win": best_result["win"],
                    "extra_info": best_result['extra_info']
                }
                
                # Get all trajectory logs
                all_logs = []
                for res in traj_results:
                    all_logs.extend(res["step_logs"])
                
                return (output_entry, all_logs), None
        except Exception as e:
            return None, str(e)
        return None, "Unknown error"

# ================== Main Orchestrator ==================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--video_prefix", type=str, default="")
    parser.add_argument("--processor_path", type=str, default=os.getenv("PROCESSOR_PATH", "/path/to/Qwen2.5-Omni-7B"))
    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("--model", type=str, default="gemini-3-pro-preview")
    parser.add_argument("--think_level", type=str, default="high")
    parser.add_argument("--num_processes", type=int, default=30)
    parser.add_argument("--max_steps", type=int, default=32)  # test-time scaling: try 12, 22, 32, 42, 52
    parser.add_argument("--max_frames_len", type=int, default=32)
    parser.add_argument("--max_audio_len", type=float, default=120.0)
    parser.add_argument("--max_clip_len", type=float, default=60.0)
    parser.add_argument("--mode", type=str, default="OmniAgent")
    parser.add_argument("--sft_attempts", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    # 1. Auto-generate path logic (based on previous approach)
    if not args.results_path:
        BASE_DIR = "./results"
        filename = generate_results_filename(args.dataset_path, args.model, args.think_level, args.max_steps, args.max_frames_len, 
                                            args.max_audio_len, args.max_clip_len, args.mode, args.num_processes)
        args.results_path = os.path.join(BASE_DIR, filename)
    
    args.results_path = os.path.abspath(args.results_path)
    os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
    step_log_path = args.results_path.replace(".json", "_steps.jsonl")

    # 2. Initialize Ray (must specify namespace to share GlobalProcessor)
    # Use .rayignore to prevent working_dir sync from hanging
    ray.init(address="auto", namespace="omni_eval_ws")
    print(f"✅ Connected to Ray cluster. Resources: {ray.cluster_resources()}")

    # 3. Pre-initialize GlobalProcessor
    print(f"🚀 Pre-initializing GlobalProcessor...")
    get_global_processor(args.processor_path)

    # 4. Load data with checkpoint resume support
    with open(args.dataset_path, 'r') as f:
        dataset = json.load(f)
    if args.max_samples: dataset = dataset[:args.max_samples]

    results = []
    processed_ids = set()
    if os.path.exists(args.results_path):
        with open(args.results_path, 'r') as f:
            try:
                results = json.load(f)
                # Unified ID check (aligned with original script)
                processed_ids = {str(r.get('index') if r.get('index') is not None else r.get('qid')) for r in results}
            except: pass

    to_process = [s for s in dataset if str(s.get('index') or s.get('qid')) not in processed_ids]
    print(f"📊 Total samples: {len(dataset)}, Already done: {len(processed_ids)}, Remaining: {len(to_process)}")

    if not to_process:
        print("🎉 All samples already processed.")
        return

    # 5. Start dynamic Actor Pool
    # Reserve 2 CPUs for management process and GlobalProcessor stability
    available_cpus = int(ray.cluster_resources().get("CPU", 1))
    num_workers = min(args.num_processes, available_cpus - 2)
    print(f"👷 Starting {num_workers} Workers...")
    workers = [EvaluationWorker.remote(args) for _ in range(num_workers)]

    # 6. Dynamic scheduling loop
    idle_workers = list(range(num_workers))
    active_tasks = {} # {ref: (worker_idx, sample_data)}
    
    f_step = open(step_log_path, "a", encoding="utf-8")
    start_time = time.time()

    while to_process or active_tasks:
        # Dispatch tasks
        while to_process and idle_workers:
            w_idx = idle_workers.pop()
            sample = to_process.pop(0)
            ref = workers[w_idx].process_sample.remote(sample)
            active_tasks[ref] = (w_idx, sample)

        # Collect results
        if active_tasks:
            done_refs, _ = ray.wait(list(active_tasks.keys()), timeout=1.0)
            for ref in done_refs:
                w_idx, sample_data = active_tasks.pop(ref)
                idle_workers.append(w_idx)
                
                try:
                    res, err = ray.get(ref)
                    if res:
                        entry, logs = res
                        results.append(entry)
                        # Write detailed step logs
                        for l in logs:
                            f_step.write(json.dumps(l, ensure_ascii=False, default=str) + "\n")
                        f_step.flush()
                        
                        # Save main results in real-time (using custom Encoder)
                        with open(args.results_path + ".tmp", 'w') as f:
                            json.dump(results, f, indent=2, ensure_ascii=False, cls=MyEncoder)
                        os.replace(args.results_path + ".tmp", args.results_path)
                        
                        elapsed = time.time() - start_time
                        print(f"✨ [DONE] {entry['index']} | Progress: {len(results)}/{len(dataset)} | Time: {elapsed:.1f}s")
                    else:
                        print(f"❌ [FAILED] {sample_data.get('index') or sample_data.get('qid')}: {err}")
                except Exception as e:
                    print(f"⚠️ [CRITICAL] Worker Error: {e}")

    f_step.close()
    print(f"🏁 Evaluation finished in {time.time()-start_time:.2f}s")

if __name__ == "__main__":
    main()
