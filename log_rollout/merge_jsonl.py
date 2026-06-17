#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
merge_jsonl.py

Usage
-----
Simple concatenation:
    python log_rollout/merge_jsonl.py  \
        /path/to/log_rollout_val/run_part1/0.jsonl \
        /path/to/log_rollout_val/run_part2/0.jsonl \
        --out /path/to/log_rollout_val/merged_run/0.jsonl

Deduplication (keep only the first occurrence of each traj_uid):
    python merge_jsonl.py  a.jsonl  b.jsonl  --out merged.jsonl  --dedup
"""

import argparse, json

def parse_args():
    p = argparse.ArgumentParser(description="Merge two .jsonl logs.")
    p.add_argument("file_a", help="first  .jsonl path")
    p.add_argument("file_b", help="second .jsonl path")
    p.add_argument("--out", required=True, help="output .jsonl file")
    p.add_argument("--dedup", action="store_true",
                   help="deduplicate by traj_uid / traj_id")
    return p.parse_args()

def stream_lines(path):
    """Yield (raw_line, obj) line by line; obj=None if parsing fails"""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                obj = None
            yield line.rstrip("\n"), obj

def main():
    args = parse_args()

    seen_tids = set()
    kept_cnt  = 0

    with open(args.out, "w", encoding="utf-8") as fout:
        for path in (args.file_a, args.file_b):
            for raw, obj in stream_lines(path):
                if args.dedup and obj is not None:
                    tid = obj.get("traj_uid") or obj.get("traj_id")
                    # Skip deduplication if neither field exists
                    if tid is not None:
                        if tid in seen_tids:
                            continue
                        seen_tids.add(tid)

                fout.write(raw + "\n")
                kept_cnt += 1

    print(f"Written {kept_cnt} lines to {args.out}")
    if args.dedup:
        print(f"Unique traj_uid / traj_id: {len(seen_tids)}")

if __name__ == "__main__":
    main()
