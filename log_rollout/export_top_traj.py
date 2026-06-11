#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
export_top_traj.py

Example
-------
Keep the first 1000 trajectories:
    python log_rollout/export_top_traj.py  \
        /path/to/log_rollout_val/run_name/0.jsonl \
        --bottom 2565 \
        --out /path/to/log_rollout_val/run_name/0_bottom2565.jsonl

Keep the last 500 trajectories:
    python export_top_traj.py  input.jsonl  --bottom 500  --out last500.jsonl
"""

import argparse, json, sys
from collections import defaultdict

def parse_args():
    p = argparse.ArgumentParser(description="Extract top / bottom N trajectories from a jsonl log.")
    p.add_argument("log_path", help="Input .jsonl log file")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--top",    type=int, help="keep the first N trajectories")
    g.add_argument("--bottom", type=int, help="keep the last  N trajectories")
    p.add_argument("--out", required=True, help="output .jsonl path")
    return p.parse_args()

def main():
    args = parse_args()

    # 1) First pass: record which traj_uid each line belongs to
    traj_order  = []                # trajectory appearance order
    traj_lines  = defaultdict(list) # tid -> [raw_line1, raw_line2, ...]
    with open(args.log_path, "r", encoding="utf-8") as fh:
        for ln in fh:
            if not ln.strip():
                continue
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue
            tid = rec.get("traj_uid") or rec.get("traj_id")
            if tid is None:          # skip lines without an id
                continue
            if tid not in traj_lines:
                traj_order.append(tid)
            traj_lines[tid].append(ln.rstrip("\n"))

    if not traj_order:
        print("No valid trajectory found, exit.")
        sys.exit(1)

    # 2) Determine which trajectories to keep
    if args.top:
        keep_tids = set(traj_order[: min(args.top, len(traj_order))])
    else:  # bottom
        keep_tids = set(traj_order[-min(args.bottom, len(traj_order)):])

    print(f"Total traj = {len(traj_order)}, keep = {len(keep_tids)}")

    # 3) Write output
    with open(args.out, "w", encoding="utf-8") as fout:
        kept_cnt = 0
        for tid in traj_order:
            if tid in keep_tids:
                for line in traj_lines[tid]:
                    fout.write(line + "\n")
                kept_cnt += 1
        print(f"Written {kept_cnt} trajectories to {args.out}")

if __name__ == "__main__":
    main()
