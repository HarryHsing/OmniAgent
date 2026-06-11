#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purge (or just list) samples that satisfy BOTH:
  1) attempts < max_try      (MCQ: len(options)+1, else 5)
  2) win / won == False

from every process*.json result file (and its step log) under --root.

Back-up files as *.bak before modification.
Provide --dry-run / -n to only report without changing files.

Additionally, collect the task names (directory prefix before '_gemini')
and list unique tasks that need to be re-run.

python purge_incomplete_samples.py --root /path/to/inference/tmp --dry-run

"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple

# ----------------- Configuration -----------------
MODEL_SPLITTER = "_gemini"   # split directory name by this; keep left part
# -------------------------------------------------

# ---------- helpers ----------
def calc_max_try(sample: Dict[str, Any]) -> int:
    qtype = sample.get("extra_info", {}).get("question_type") or sample.get("question_type")
    options = sample.get("options")
    if qtype == "MCQ":
        return (len(options) if options else 0) + 1
    return 5

def should_drop(sample: Dict[str, Any]) -> bool:
    attempts = int(sample.get("attempts", 0))
    max_try  = calc_max_try(sample)
    win_flag = sample.get("win")
    if win_flag is None:
        win_flag = sample.get("extra_info", {}).get("won")
    return attempts < max_try and (win_flag is False)

def backup(fp: Path):
    bak = fp.with_suffix(fp.suffix + ".bak")
    if not bak.exists():
        bak.write_bytes(fp.read_bytes())

def atomic_write(fp: Path, text: str):
    tmp = fp.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(fp)

def get_task_name_from_path(fp: Path) -> str:
    """
    Derive task name from parent directory.
    If directory contains '_gemini', strip from that part to the end.
    """
    dir_name = fp.parent.name
    if MODEL_SPLITTER in dir_name:
        return dir_name.split(MODEL_SPLITTER)[0]
    return dir_name

# ---------- per-file processing ----------
# ---------- per-file processing ----------

from typing import Tuple
def purge_one_file(json_fp: Path, dry: bool) -> Tuple[int, int, Set[str]]:
    """
    Returns (removed_cnt, total_cnt, tasks_set)
    """
    try:
        samples: List[Dict[str, Any]] = json.loads(json_fp.read_text())
    except Exception as e:
        print(f"[ERROR] read {json_fp}: {e}")
        return 0, 0, set()

    total_cnt = len(samples)
    keep, drop_idx = [], set()
    for s in samples:
        if should_drop(s):
            drop_idx.add(s["index"])
        else:
            keep.append(s)

    removed = len(drop_idx)
    tasks = {get_task_name_from_path(json_fp)} if removed else set()

    if removed and not dry:
        # backup & overwrite result json
        backup(json_fp)
        atomic_write(json_fp, json.dumps(keep, indent=2, ensure_ascii=False))

        # step log handling
        step_fp = json_fp.with_suffix("_steps.jsonl")
        if step_fp.exists():
            backup(step_fp)
            kept_lines = []
            with step_fp.open("r", encoding="utf-8") as fin:
                for line in fin:
                    try:
                        rec = json.loads(line)
                        qid = rec.get("question_id", "")
                        m = re.search(r"_(\d+)_", qid)
                        if m and int(m.group(1)) in drop_idx:
                            continue
                    except Exception:
                        pass
                    kept_lines.append(line)
            atomic_write(step_fp, "".join(kept_lines))

    return removed, total_cnt, tasks

# ---------- traversal ----------
def purge_directory(root: Path, dry: bool):
    total_removed = 0
    total_samples = 0
    files_affected = 0
    files_scanned  = 0
    task_set: Set[str] = set()

    for fp in sorted(root.rglob("process*.json")):
        files_scanned += 1
        removed, total, tasks = purge_one_file(fp, dry)
        total_samples += total
        if removed:
            files_affected += 1
            total_removed += removed
            task_set.update(tasks)
            action = "WOULD purge" if dry else "Purged"
            print(f"{fp.relative_to(root)}: {action} {removed} samples")

    # ---- summary ----
    print("\n====== Summary ======")
    print(f"Files scanned   : {files_scanned}")
    print(f"Files affected  : {files_affected}")
    verb = "would be" if dry else "were"
    print(f"Samples total   : {total_samples}")
    print(f"Samples that {verb} removed: {total_removed}")
    print("=====================")

    # ---- task list ----
    if task_set:
        print(f"\nTasks to rerun ({len(task_set)}):")
        for t in sorted(task_set):
            print(f"  - {t}")
    else:
        print("\nNo tasks need to be rerun.")

    if dry:
        print("\n(Dry-run: no files modified.)")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory containing process*.json")
    ap.add_argument("-n", "--dry-run", action="store_true", help="Only report, don't modify files")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"Root {root} not found.")
        return
    purge_directory(root, dry=args.dry_run)

if __name__ == "__main__":
    main()
