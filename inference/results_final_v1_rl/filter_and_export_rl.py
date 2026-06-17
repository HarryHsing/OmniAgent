import json
import re
import os
import argparse
import statistics
from collections import OrderedDict, Counter

# Define error categories
FFMPEG_ERRORS = {"FFMPEG_AUDIO_FAIL", "FFMPEG_CLIP_FAIL", "FFMPEG_FRAME_FAIL"}
# Exclude TOKEN_LIMIT, keep STEP_LIMIT as negative examples for RL
EXCLUDE_ERRORS = FFMPEG_ERRORS | {"TOKEN_LIMIT_EXCEEDED"}

def fix_frames_text(text):
    if not isinstance(text, str): return text
    return re.sub(r'Frames ([\d\.]+s-[\d\.]+s) \(num=1\)', r'Frame \1 (num=1)', text)

def replace_oss_path(path_str, base_path):
    if not isinstance(path_str, str) or "agentic_tmp" not in path_str:
        return path_str
    pattern = r'agentic_tmp/([^?#\s]+)'
    match = re.search(pattern, path_str)
    if match:
        relative_path = match.group(1)
        return os.path.join(base_path, relative_path).replace("\\", "/")
    return path_str

def process_rl_negative_data(input_path, output_dir, base_path, suffix):
    if not os.path.exists(input_path):
        print(f"Error: Input file not found {input_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    all_trajectories = OrderedDict()
    
    # Statistics containers
    stats = {
        "total_trajs": 0,
        "skipped_won": 0,
        "skipped_env_err": 0,
        "kept_negatives": 0,
        "fail_reasons": Counter(),
        "q_types": Counter(),
        "durations": [],
        "sources": Counter()
    }

    print(f"Loading trajectory data: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                step_data = json.loads(line)
                t_id = step_data['traj_id']
                if t_id not in all_trajectories:
                    all_trajectories[t_id] = []
                all_trajectories[t_id].append(step_data)
            except: pass

    output_rl_path = os.path.join(output_dir, os.path.basename(input_path).replace(".jsonl", "_rl_negatives.jsonl"))

    print(f"Filtering failed samples and generating RL dataset (Suffix: {suffix})...")
    
    with open(output_rl_path, 'w', encoding='utf-8') as f_out:
        for t_id, steps in all_trajectories.items():
            stats["total_trajs"] += 1
            last_step = steps[-1]
            first_step = steps[0]
            extra = last_step.get('extra_info', {})
            
            is_won = extra.get('won', False)
            error_code = extra.get('error_code')

            # 1. Exclude successful samples
            if is_won:
                stats["skipped_won"] += 1
                continue

            # 2. Exclude environment and token limit errors
            if error_code in EXCLUDE_ERRORS:
                stats["skipped_env_err"] += 1
                continue
            
            # Record failure reason
            reason = error_code if error_code else "LOGIC_WRONG_ANSWER"
            stats["fail_reasons"][reason] += 1

            # 3. Extract Prompt
            prompt_content = []
            if 'raw_input' in first_step:
                for msg in first_step['raw_input']:
                    if msg.get('role') == 'user':
                        cleaned_msg_content = []
                        for c in msg.get('content', []):
                            if c['type'] == 'text':
                                c['text'] = fix_frames_text(c['text'])
                                c['text'] = replace_oss_path(c['text'], base_path)
                            elif c['type'] in ['video', 'image', 'audio']:
                                if c['type'] in c:
                                    c[c['type']] = replace_oss_path(c[c['type']], base_path)
                            cleaned_msg_content.append(c)
                        prompt_content.append({"content": cleaned_msg_content, "role": "user"})

            # Handle question_type with suffix
            raw_q_type = extra.get("question_type", "")
            final_q_type = f"{raw_q_type}_{suffix}" if suffix else raw_q_type

            # 4. Assemble RL format
            rl_item = {
                "prompt": [{"content": "", "role": "user"}],
                "question_type": final_q_type,
                "origin_question_type": extra.get("origin_question_type", ""),
                "question": extra.get("question", ""),
                "answer": str(extra.get("answer", [])),
                "options": extra.get("options", []),
                "video": replace_oss_path(extra.get("video", ""), base_path),
                "fps": extra.get("fps", 0.0),
                "duration_seconds": extra.get("duration_seconds", 0.0),
                "has_audio": extra.get("has_audio", False),
                "data_source": extra.get("data_source", "agent"),
                "ability": "agent",
                "extra_info": {
                    "traj_id": t_id,
                    "error_reason": reason
                }
            }

            f_out.write(json.dumps(rl_item, ensure_ascii=False) + "\n")
            
            # Statistics data collection
            stats["kept_negatives"] += 1
            stats["q_types"][final_q_type] += 1
            stats["sources"][extra.get("data_source", "unknown")] += 1
            if isinstance(extra.get("duration_seconds"), (int, float)):
                stats["durations"].append(extra.get("duration_seconds"))

    # --- Print report ---
    print("\n" + "="*60)
    print("RL Negative Sample Dataset Analysis Report")
    print("="*60)
    print(f"Overall Trajectory Processing:")
    print(f"   - Total raw trajectories:     {stats['total_trajs']}")
    print(f"   - Excluded success (for SFT): {stats['skipped_won']}")
    print(f"   - Excluded env errors:        {stats['skipped_env_err']}")
    print(f"   - Final kept negatives:       {stats['kept_negatives']}")

    print(f"\nFail Reasons Distribution:")
    for r, count in stats["fail_reasons"].items():
        print(f"   - {r}: {count}")

    if stats["durations"]:
        print(f"\nVideo Duration Statistics:")
        print(f"   - Mean Duration:   {statistics.mean(stats['durations']):.2f}s")
        print(f"   - Max Duration:    {max(stats['durations']):.2f}s")
        print(f"   - Min Duration:    {min(stats['durations']):.2f}s")
        print(f"   - Median:          {statistics.median(stats['durations']):.2f}s")

    print(f"\nQuestion Type Distribution (Top 10):")
    for qt, count in stats["q_types"].most_common(10):
        print(f"   - {qt}: {count}")

    print(f"\nData Source Distribution:")
    for src, count in stats["sources"].items():
        print(f"   - {src}: {count}")

    print("="*60)
    print(f"Output file: {output_rl_path}\n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--suffix", type=str, default="", help="Dataset name, appended to question_type")
    args = parser.parse_args()
    
    print("-"*100)
    process_rl_negative_data(args.input, args.out_dir, args.base_path, args.suffix)
    print("-"*100)
    print("\n")


'''
Example:
python filter_and_export_rl.py \
    --input "/path/to/train_steps.jsonl" \
    --out_dir "/path/to/results_final_v1_rl" \
    --suffix DATASET \
    --base_path "/path/to/video_env_tmp"
'''
