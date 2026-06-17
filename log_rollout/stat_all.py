#!/usr/bin/env python3
"""
stat_all.py
──────────────────────────────────────────────────────────────
Three tables:

A) record-level  valid / total   (all records accumulated directly)
B) step-level    average traj-valid-ratio
       - Within each file: a trajectory is invalid if any step is invalid
       - ratio_file(step) = valid_traj(step) / total_traj(step)
       - When averaging across files, only count files that contain the step
C) file-level    per-file:
       - traj-valid-ratio
       - wins / total and accuracy

Usage:
    python stat_all.py [-r] [--max-step N]  /path/to/folder
"""

import os, sys, glob, json, argparse, re
from collections import defaultdict

# ───────────────── argparse ─────────────────
parser = argparse.ArgumentParser(description="Three kinds of valid-ratio statistics")
parser.add_argument("log_dir", help="directory with *.jsonl (recursively searched)")
parser.add_argument("--max-step", type=int, default=None,
                    help="only show steps ≤ this value")
parser.add_argument("-r", "--render", action="store_true",
                    help="draw ASCII bars for table-A")
args = parser.parse_args()

log_dir   = args.log_dir
max_step  = args.max_step
do_render = args.render

if not os.path.isdir(log_dir):
    print(f"Error: {log_dir} is not a directory")
    sys.exit(1)

jsonl_paths = glob.glob(os.path.join(log_dir, "**", "*.jsonl"), recursive=True)
file_cnt = len(jsonl_paths)
if file_cnt == 0:
    print("No .jsonl files found.")
    sys.exit(0)

print(f"Found {file_cnt} jsonl files, start scanning...\n")

# ═════════════ Statistics containers ═════════════
# A) record-level
rec_valid_cnt = defaultdict(int)   # step -> valid record #
rec_total_cnt = defaultdict(int)   # step -> total record #

# B) per-file step ratio
file_step_ratio_list = []          # list[dict(step->ratio)]

# C) file-level rows: (fname, valid_traj, total_traj, valid_ratio,
#                      wins, accuracy)
file_level_rows = []

# ═════════════ Iterate over files ═════════════
for jp in jsonl_paths:
    traj_overall_valid = defaultdict(lambda: True)   # traj -> bool
    traj_steps         = defaultdict(set)            # traj -> {steps}
    traj_last_rec      = {}                          # traj -> latest record

    try:
        with open(jp, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                step = rec.get("cur_step")
                if step is None or (max_step is not None and step > max_step):
                    continue

                traj = rec.get("traj_uid") or rec.get("traj_id")
                if traj is None:
                    continue

                # —— A) record-level —— #
                is_valid = rec.get("step_info", {}).get("is_action_valid")
                if is_valid is None:
                    is_valid = True

                rec_total_cnt[step] += 1
                if is_valid:
                    rec_valid_cnt[step] += 1

                # —— trajectory info —— #
                traj_steps[traj].add(step)
                if not is_valid:
                    traj_overall_valid[traj] = False

                # Save the latest step record (keep the one with max step)
                last_rec = traj_last_rec.get(traj)
                if last_rec is None or step > last_rec.get("cur_step", -1):
                    traj_last_rec[traj] = rec

    except Exception as e:
        print(f"[WARN] reading {jp}: {e}")
        continue

    # ---------- C) file-level ----------
    total_traj = len(traj_steps)
    valid_traj = sum(1 for t in traj_steps if traj_overall_valid[t])
    ratio_file = valid_traj / total_traj if total_traj else 0.0

    # wins / accuracy
    wins = sum(
        1 for rec in traj_last_rec.values()
        if bool(rec.get("step_info", {}).get("won"))
    )
    acc = wins / total_traj if total_traj else 0.0

    file_level_rows.append((
        os.path.basename(jp),
        valid_traj, total_traj, ratio_file,
        wins, acc
    ))

    # ---------- B) step-level ratio for the file ----------
    step_total_traj = defaultdict(int)
    step_valid_traj = defaultdict(int)
    for traj, steps in traj_steps.items():
        ok = traj_overall_valid[traj]
        for s in steps:
            step_total_traj[s] += 1
            if ok:
                step_valid_traj[s] += 1
    ratio_dict = {s: step_valid_traj[s] / step_total_traj[s]
                  for s in step_total_traj}          # only steps present
    file_step_ratio_list.append(ratio_dict)

# ═════════════ Aggregate Table-B ═════════════
step_sum = defaultdict(float)
step_file_num = defaultdict(int)   # number of files contributing to this step

for d in file_step_ratio_list:
    for s, r in d.items():
        step_sum[s]      += r
        step_file_num[s] += 1

avg_step_ratio = {s: step_sum[s] / step_file_num[s] for s in step_sum}

# ═════════════ Print results ═════════════
def sorted_steps(keys):
    return sorted(k for k in keys if (max_step is None or k <= max_step))

# --- Table A ---
print("Table-A  Global record-level valid ratio")
print("Step |  Valid/Total |  Ratio%")
print("-------------------------------")
for s in sorted_steps(rec_total_cnt):
    v, t = rec_valid_cnt[s], rec_total_cnt[s]
    ratio = 100 * v / t if t else 0.0
    bar = ""
    if do_render:
        bar_len = int(ratio // 2)
        bar = " |" + "█" * bar_len
    print(f"{s:4d} | {v:6d}/{t:<6d} | {ratio:6.2f}%{bar}")

# --- Table B ---
print("\nTable-B  Per-file traj-valid-ratio averaged (only files containing the step)")
print("(trajectory invalid if ANY step invalid)")
print("Step |  Avg ratio% | #FilesContributed")
print("---------------------------------------")
for s in sorted_steps(avg_step_ratio):
    print(f"{s:4d} |   {avg_step_ratio[s]*100:6.2f}% | {step_file_num[s]:18d}")

# --- Table C ---
print("\nTable-C  File-level statistics (one line per jsonl)")
print("File                               | Valid/Total | Val%  | Wins/Total | Acc% ")
print("----------------------------------------------------------------------------")

def nat_key(fname: str):
    """Natural sort key: prefer trailing digits, fall back to string"""
    base = os.path.splitext(fname)[0]
    m = re.match(r"(\d+)$", base)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)$", base)
    return int(m.group(1)) if m else base

for row in sorted(file_level_rows, key=lambda x: nat_key(x[0])):
    fname, v_cnt, t_cnt, v_ratio, wins, acc = row
    print(f"{fname:<34s}| {v_cnt:5d}/{t_cnt:<5d} | {v_ratio*100:6.2f}% | "
          f"{wins:4d}/{t_cnt:<5d} | {acc*100:6.2f}%")
