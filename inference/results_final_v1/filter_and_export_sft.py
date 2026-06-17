import json
import re
import os
import argparse
from collections import OrderedDict

# Define error categories
FFMPEG_ERRORS = {"FFMPEG_AUDIO_FAIL", "FFMPEG_CLIP_FAIL", "FFMPEG_FRAME_FAIL"}
STRATEGY_LIMITS = {"STEP_LIMIT_REACHED", "TOKEN_LIMIT_EXCEEDED"}

def fix_frames_text(text):
    """When num=1, change 'Frames' to 'Frame'"""
    if not isinstance(text, str): return text
    return re.sub(r'Frames ([\d\.]+s-[\d\.]+s) \(num=1\)', r'Frame \1 (num=1)', text)

def replace_oss_path(path_str, base_path):
    """
    Enhanced path replacement logic:
    1. Capture all path content after agentic_tmp
    2. Automatically truncate OSS signature parameters (?Expires=...)
    """
    if not isinstance(path_str, str) or "agentic_tmp" not in path_str:
        return path_str
    
    # Match content after agentic_tmp/ until encountering ? or # or whitespace/end
    pattern = r'agentic_tmp/([^?#\s]+)'
    match = re.search(pattern, path_str)
    if match:
        relative_path = match.group(1)
        # Use base_path to construct local path
        return os.path.join(base_path, relative_path).replace("\\", "/")
    return path_str

def process_sft_data(input_path, output_dir, max_steps, base_path):
    if not os.path.exists(input_path):
        print(f"Error: Input file not found {input_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    all_trajectories = OrderedDict()
    sample_to_trajs = OrderedDict()

    # --- 1. Read and analyze trajectories (with deduplication logic) ---
    print(f"Reading and analyzing trajectories: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                step_data = json.loads(line)
                t_id = step_data['traj_id']
                if t_id not in all_trajectories:
                    all_trajectories[t_id] = []
                all_trajectories[t_id].append(step_data)
                
                info = step_data.get('extra_info', {})
                triplet = (info.get('question', ''), str(info.get('answer', '')), info.get('video', ''))
                
                if triplet not in sample_to_trajs:
                    sample_to_trajs[triplet] = []
                if t_id not in sample_to_trajs[triplet]:
                    sample_to_trajs[triplet].append(t_id)
            except: pass

    # Deduplication logic
    kept_traj_ids = set()
    for triplet, t_ids in sample_to_trajs.items():
        best_t_id = None
        for tid in reversed(t_ids):
            if all_trajectories[tid][-1].get('extra_info', {}).get('won'):
                best_t_id = tid
                break
        kept_traj_ids.add(best_t_id if best_t_id else t_ids[-1])

    # Statistics variables
    stats = {
        "ffmpeg_err": 0,
        "step_limit": 0,
        "token_limit": 0,
        "win_count": 0,
        "fail_count": 0,
        "overstep_count": 0
    }

    output_sft_path = os.path.join(output_dir, os.path.basename(input_path).replace(".jsonl", "_final_sft.jsonl"))

    # --- 2. Deep cleaning and path localization ---
    print(f"Performing full-field deep cleaning (including extra_info and video/audio)...")
    with open(output_sft_path, 'w', encoding='utf-8') as f_out:
        for t_id in all_trajectories:
            if t_id not in kept_traj_ids:
                continue

            steps = all_trajectories[t_id]
            is_won = steps[-1].get('extra_info', {}).get('won', False)
            
            cleaned_steps = []
            for s in steps:
                # Check error code
                err = s.get('extra_info', {}).get('error_code')
                if err:
                    if err in FFMPEG_ERRORS: stats["ffmpeg_err"] += 1
                    elif err == "STEP_LIMIT_REACHED": stats["step_limit"] += 1
                    elif err == "TOKEN_LIMIT_EXCEEDED": stats["token_limit"] += 1
                    continue 

                # Deep Fix A: Replace metadata paths in extra_info (fixes KeyError in recomputation script)
                if 'extra_info' in s:
                    for media_key in ['video', 'image', 'audio']:
                        if media_key in s['extra_info']:
                            s['extra_info'][media_key] = replace_oss_path(s['extra_info'][media_key], base_path)
                
                # Deep Fix B: Process all modality fields in raw_input
                if 'raw_input' in s:
                    for role_content in s['raw_input']:
                        for content in role_content.get('content', []):
                            ctype = content.get('type')
                            # 1. Text fields
                            if ctype == 'text':
                                content['text'] = fix_frames_text(content['text'])
                                content['text'] = replace_oss_path(content['text'], base_path)
                            # 2. Multimodal fields (image/video/audio)
                            elif ctype in ['image', 'video', 'audio']:
                                if ctype in content:
                                    content[ctype] = replace_oss_path(content[ctype], base_path)
                                    
                cleaned_steps.append(s)

            if not cleaned_steps: continue

            # Only keep successful SFT trajectories
            if is_won:
                if len(steps) <= max_steps:
                    for s in cleaned_steps:
                        f_out.write(json.dumps(s, ensure_ascii=False) + "\n")
                    stats["win_count"] += 1
                else:
                    stats["overstep_count"] += 1
            else:
                stats["fail_count"] += 1

    # --- 3. Print report ---
    print(f"\n" + "="*50)
    print(f"Enhanced Cleaning and Classification Statistics Report")
    print(f"="*50)
    print(f"Unique samples: {len(sample_to_trajs)}")
    print(f"Original total trajectories: {len(all_trajectories)}")
    print(f"-"*50)
    print(f"[Env Error] FFMPEG Fail Turns:          {stats['ffmpeg_err']}")
    print(f"[Strategy Limit] Step Limit Turns:      {stats['step_limit']}")
    print(f"[Strategy Limit] Token Limit Turns:     {stats['token_limit']}")
    print(f"-"*50)
    print(f"Final Kept SFT Trajectories (Win):      {stats['win_count']}")
    print(f"Discarded (Failed/Overstep):            {stats['fail_count'] + stats['overstep_count']}")
    print(f"="*50)
    print(f"Output: {output_sft_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=50)
    args = parser.parse_args()
    process_sft_data(args.input, args.out_dir, args.max_steps, args.base_path)



'''
Example:
python filter_and_export_sft.py \
    --input "/path/to/train_steps.jsonl" \
    --out_dir "/path/to/results_final_v1" \
    --base_path "/path/to/video_env_tmp" \
    --max_steps 50
'''
