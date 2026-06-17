#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
count_jsonl_stats.py  (enhanced)

Compute statistics for *.jsonl files in a given directory:
  1) Line count (record count)
  2) Unique traj_id count
  3) Video duration stats: mean / min / max (based on extra_info.duration_seconds)
  4) Question type (question_type) distribution and percentages

Filtering:
  --include FILE : only count files listed (txt, one filename per line)
  --exclude FILE : exclude files listed (txt, one filename per line)
  --recursive    : search sub-directories recursively
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple, List


def load_name_list(path: Path) -> Set[str]:
    with path.open() as f:
        return {ln.strip() for ln in f if ln.strip()}


def file_stats(path: Path) -> Tuple[int, int, float, float, float, Dict[str, int]]:
    """
    Returns:
      n_lines,
      n_traj,
      total_video_duration_sec,
      min_dur,
      max_dur,
      qtype_counter (dict)
    """
    n_lines = 0
    traj_seen: Set[str] = set()
    qtype_counter: Counter[str] = Counter()
    total_dur = 0.0
    min_dur = float("inf")
    max_dur = 0.0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            traj_id = rec.get("traj_id")
            if traj_id and traj_id not in traj_seen:
                traj_seen.add(traj_id)

                ei = rec.get("extra_info", {})
                dur = ei.get("duration_seconds")
                if isinstance(dur, (int, float)):
                    total_dur += dur
                    min_dur = min(min_dur, dur)
                    max_dur = max(max_dur, dur)

                qtype = ei.get("question_type") or rec.get("question_type")
                if qtype:
                    qtype_counter[qtype] += 1

    if min_dur == float("inf"):
        min_dur = 0.0
        print(f"[WARN] {path.name} has no video duration")
    return n_lines, len(traj_seen), total_dur, min_dur, max_dur, qtype_counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory that contains jsonl files")
    ap.add_argument("--include", help="txt file: list of filenames to include")
    ap.add_argument("--exclude", help="txt file: list of filenames to exclude")
    ap.add_argument("--recursive", action="store_true", help="search sub-folders")
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        sys.exit(f"[ERROR] {root} is not a directory")

    # filter lists
    include_set = load_name_list(Path(args.include)) if args.include else set()
    exclude_set = load_name_list(Path(args.exclude)) if args.exclude else set()
    if include_set and exclude_set:
        sys.exit("[ERROR] --include and --exclude cannot be used together")

    pattern = "**/*.jsonl" if args.recursive else "*.jsonl"
    files = sorted(root.glob(pattern))

    # global accumulators
    g_lines = g_traj = 0
    g_total_dur = g_min_dur = 0.0
    g_max_dur = 0.0
    g_qtype_counter: Counter[str] = Counter()
    g_files = 0

    for fp in files:
        if include_set and fp.name not in include_set:
            continue
        if exclude_set and fp.name in exclude_set:
            continue

        g_files += 1
        n_lines, n_traj, tot_dur, min_d, max_d, q_counter = file_stats(fp)

        g_lines += n_lines
        g_traj += n_traj
        g_total_dur += tot_dur
        g_min_dur = min(g_min_dur or min_d, min_d) if n_traj else g_min_dur
        g_max_dur = max(g_max_dur, max_d)
        g_qtype_counter.update(q_counter)

        # per-file brief
        avg_dur = (tot_dur / n_traj) if n_traj else 0
        qtypes_str = ", ".join(f"{k}:{v}" for k, v in q_counter.items())
        print(f"{fp.name:<60}  lines:{n_lines:>6}  traj:{n_traj:>5}  "
              f"avg_dur:{avg_dur:6.1f}s  qtypes[{qtypes_str}]")

    # ----- global summary -----
    print("-" * 120)
    print(f"TOTAL files   : {g_files}")
    print(f"TOTAL lines   : {g_lines}")
    print(f"TOTAL traj    : {g_traj}")

    if g_traj:
        g_avg_dur = g_total_dur / g_traj
        print(f"Video duration: avg {g_avg_dur:.1f}s   min {g_min_dur:.1f}s   max {g_max_dur:.1f}s")

    if g_qtype_counter:
        print("\nQuestion-type distribution:")
        total_qt = sum(g_qtype_counter.values())
        for k, v in g_qtype_counter.most_common():
            pct = v * 100 / total_qt
            print(f"  {k:<10}: {v:>6}  ({pct:5.1f}%)")

if __name__ == "__main__":
    main()


'''
python inference/filter_train/count_jsonl_stats.py \
       --dir /path/to/filter_train \
       --exclude /path/to/filter_train/exclude_list.txt
'''
