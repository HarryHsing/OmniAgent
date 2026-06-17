#!/usr/bin/env python3
import json
import argparse
import sys
from pathlib import Path
from typing import Iterable, Set, Dict

THRESH = 1.0

def iter_jsonl_files(paths: Iterable[str]):
    for p in paths:
        path = Path(p).expanduser()
        if path.is_dir():
            yield from path.rglob("*.jsonl")
        else:
            yield path

def make_question_id(step_info: dict) -> str:
    """
    Generate a unique identifier for a question.
    Priority: index > (video + question) > (video + question + options)
    """
    # Option 1: use the existing index directly if available
    if "index" in step_info:
        return str(step_info["index"])

    # Option 2: combine video + question
    video = step_info.get("video", "")
    question = step_info.get("question", "")

    if not video:
        return ""  # invalid data

    # If the same video has multiple questions, must include question
    if question:
        # Option 2a: direct concatenation (recommended)
        return f"{video}||{question}"
    else:
        # If there is no question field, can only use video (assumes one question per video)
        return video

def read_jsonl_last_step(fpath: Path) -> Dict[str, float]:
    """
    Read JSONL, group by traj_uid and take the last step, use question unique ID as key.
    Returns {question_id: reward}
    """
    traj_map: Dict[str, dict] = {}
    
    with open(fpath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] {fpath}:{line_num} invalid JSON ({e})", file=sys.stderr)
                continue
            
            traj_uid = obj.get("traj_uid") or obj.get("uid")
            if not traj_uid:
                continue
            
            cur_step = obj.get("cur_step", 0)
            
            if traj_uid not in traj_map or cur_step > traj_map[traj_uid].get("cur_step", -1):
                traj_map[traj_uid] = obj
    
    result = {}
    for traj_uid, obj in traj_map.items():
        step_info = obj.get("step_info", {})
        
        # Generate unique identifier
        qid = make_question_id(step_info)
        if not qid:
            print(f"[warn] {fpath} traj {traj_uid} cannot generate question ID", file=sys.stderr)
            continue

        # Extract reward
        reward = obj.get("reward", 0.0)
        if reward == 0.0:
            reward = step_info.get("reward", 0.0)
        
        result[qid] = float(reward)
    
    return result

def main(args):
    union_all: Set[str] = set()
    union_ok: Set[str] = set()
    
    per_file_all = []
    per_file_ok = []
    
    file_results = []
    
    n_files = 0
    for fpath in iter_jsonl_files(args.inputs):
        n_files += 1
        try:
            qid_reward = read_jsonl_last_step(fpath)
        except Exception as e:
            print(f"[warn] skip {fpath} ({e})", file=sys.stderr)
            continue
        
        file_all = set(qid_reward.keys())
        file_ok = {q for q, r in qid_reward.items() if r >= THRESH}
        
        per_file_all.append(file_all)
        per_file_ok.append(file_ok)
        
        union_all |= file_all
        union_ok |= file_ok
        
        file_results.append((fpath.name, qid_reward))
        
        acc = len(file_ok) / len(file_all) * 100 if file_all else 0
        print(f"[file {n_files}] {fpath.name}: {len(file_ok)}/{len(file_all)} correct ({acc:.1f}%)")
    
    # Union statistics
    total = len(union_all)
    correct = len(union_ok)
    acc = correct / total if total else 0.0
    
    print(f"\n{'='*60}")
    print(f"Scanned {n_files} jsonl files")
    print(f"Unique questions (union): {total}")
    print(f"Correct questions (reward ≥ {THRESH}): {correct}")
    print(f"Union accuracy: {acc:.4%}" if total else "Union accuracy: N/A")
    
    # Intersection statistics
    if per_file_all:
        inter_all = set.intersection(*per_file_all)
        inter_ok = set.intersection(*per_file_ok)
        if inter_all:
            both_acc = len(inter_ok) / len(inter_all)
            print(f"\nQuestions present in EVERY file: {len(inter_all)}")
            print(f"Questions correct in EVERY file: {len(inter_ok)}")
            print(f"All-correct ratio: {both_acc:.4%}")
            
            inconsistent = len(inter_all) - len(inter_ok)
            inconsistent_rate = inconsistent / len(inter_all)
            print(f"Inconsistent questions: {inconsistent} ({inconsistent_rate:.2%})")
        else:
            print("\nNo common questions across all files")
    
    # Detailed difference analysis
    if args.show_diff and len(file_results) == 2:
        print(f"\n{'='*60}")
        print("Detailed inconsistency analysis:")
        fname1, res1 = file_results[0]
        fname2, res2 = file_results[1]
        
        common = set(res1.keys()) & set(res2.keys())
        
        diff_questions = [
            q for q in common
            if (res1[q] >= THRESH) != (res2[q] >= THRESH)
        ]
        
        print(f"Found {len(diff_questions)} inconsistent questions (out of {len(common)} common)")
        
        if diff_questions:
            print(f"\nShowing first {min(10, len(diff_questions))} examples:")
            for qid in diff_questions[:10]:
                # Try to parse question ID
                if "||" in qid:
                    video, question = qid.split("||", 1)
                    video_name = Path(video).name
                    q_preview = question[:60] + "..." if len(question) > 60 else question
                else:
                    video_name = Path(qid).name
                    q_preview = "(no question text)"
                
                r1, r2 = res1[qid], res2[qid]
                status1 = "✓" if r1 >= THRESH else "✗"
                status2 = "✓" if r2 >= THRESH else "✗"
                
                print(f"  Video: {video_name}")
                print(f"  Question: {q_preview}")
                print(f"    {fname1}: {status1} (reward={r1:.2f})")
                print(f"    {fname2}: {status2} (reward={r2:.2f})")
                print()
    
    print(f"Total unique questions: {total}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count union & intersection accuracy of JSONL trajectory files"
    )
    parser.add_argument("inputs", nargs="+",
                        help="JSONL files / directories / patterns")
    parser.add_argument("--thresh", type=float, default=1.0,
                        help="Reward threshold for correctness (default: 1.0)")
    parser.add_argument("--show-diff", action="store_true",
                        help="Show detailed inconsistency analysis for 2 files")
    args = parser.parse_args()
    THRESH = args.thresh
    main(args)


'''
# 1. Basic statistics (multiple files)
python count_correct_union.py results/*.jsonl

# 2. Detailed comparison of two files
python count_correct_union.py --show-diff \
  results/run1.jsonl \
  results/run2.jsonl

# 3. Custom threshold
python count_correct_union.py --thresh 0.5 results/*.jsonl

python count_correct_union.py \
  /path/to/file1.jsonl \
  /another/path/*.jsonl \
  ./local_results/file3.jsonl

python ./log_rollout/count_union_acc.py  \
  /path/to/log_rollout_val/run_a/0.jsonl \
  /path/to/log_rollout_val/run_b/0.jsonl


'''
