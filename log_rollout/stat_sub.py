#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
import argparse
import re
import math
import statistics
import os
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# 1. Helper classes and parsing logic
# ---------------------------------------------------------------------
class Aggregator:
    def __init__(self):
        self.values = []
    def add(self, v):
        if v is not None: self.values.append(v)
    def stat(self):
        if not self.values: return None
        return {
            "cnt": len(self.values),
            "min": round(min(self.values), 2),
            "max": round(max(self.values), 2),
            "mean": round(statistics.mean(self.values), 4),
            "median": round(statistics.median(self.values), 2),
            "p90": round(pct(self.values, 90), 2)
        }

def pct(lst, p):
    if not lst: return 0.0
    sl = sorted(lst)
    idx = max(0, min(len(sl) - 1, int(round(p / 100 * (len(sl) - 1)))))
    return sl[idx]

def parse_action_with_err(raw: Optional[str]) -> Tuple[dict, str]:
    if not raw: return {}, "no_output"
    s = str(raw).strip()
    try:
        if s.startswith("```"):
            s = re.sub(r"^```[a-zA-Z]*\n", "", s, 1).split("```", 1)[0]
        start = s.find("{")
        if start == -1: return {}, "no_brace"
        depth = end = 0
        for i, ch in enumerate(s[start:], start):
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1; break
        if not end: return {}, "unbalanced_braces"
        obj = json.loads(s[start:end])
        return obj, ""
    except Exception: return {}, "json_decode_err"

# ---------------------------------------------------------------------
# 2. VSI-Bench metadata support
# ---------------------------------------------------------------------
DEFAULT_VSI_META = [
    p.strip()
    for p in re.split(r"[:;]", os.getenv("VSI_META_PATHS", ""))
    if p.strip()
]
VSI_TABLE = {}

def load_vsi_table(candidates):
    for p in candidates:
        if p and os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    for line in fh:
                        rec = json.loads(line)
                        q, v, o = rec.get("question","").strip(), rec.get("video","").strip(), rec.get("origin_question_type")
                        if q and v and o: VSI_TABLE[(q, v)] = o
                print(f"[INFO] Loaded VSI meta: {p}", file=sys.stderr)
                return p
            except: continue
    return None

# ---------------------------------------------------------------------
# 3. Main analysis logic
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Merged Stat Script (VSI + Evidence + Confidence + Range Support + Duration Stats)")
    parser.add_argument("log_path")
    parser.add_argument("-t", "--top", type=int, help="First N trajectories")
    parser.add_argument("-b", "--bottom", type=int, help="Last N trajectories")
    parser.add_argument("-r", "--range", nargs=2, type=int, help="Range of trajectories [START END] inclusive, 1-based (e.g. 301 600)")
    parser.add_argument("--qtype", help="Comma separated question types")
    parser.add_argument("--vsi-meta")
    parser.add_argument("--interval", type=int, default=5, help="Minute interval for duration stats (default: 5)")
    args = parser.parse_args()

    # VSI Meta Setup
    is_vsi_bench = "VSI" in args.log_path
    vsi_meta_list = [args.vsi_meta] + DEFAULT_VSI_META if args.vsi_meta else DEFAULT_VSI_META
    vsi_meta_source = load_vsi_table(vsi_meta_list) if is_vsi_bench else None

    # Step 1: Read and group by Trajectory
    traj_steps = defaultdict(list)
    traj_order = []
    traj_qtype = {}
    traj_extra = {}

    with open(args.log_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                tid = rec.get("traj_uid") or rec.get("traj_id") or rec.get("uid")
                step = rec.get("cur_step")
                if tid is None or step is None: continue
                if tid not in traj_steps: traj_order.append(tid)
                traj_steps[tid].append(rec)
                
                if tid not in traj_qtype:
                    qt = rec.get("step_info", {}).get("question_type") or rec.get("question_type")
                    if qt: traj_qtype[tid] = qt
                if is_vsi_bench and tid not in traj_extra:
                    qtxt = rec.get("step_info", {}).get("question") or rec.get("question", "")
                    vpth = rec.get("step_info", {}).get("video") or rec.get("video", "")
                    traj_extra[tid] = (qtxt.strip(), vpth.strip())
            except: continue

    # Step 2: Filter
    if args.qtype:
        q_set = {x.strip() for x in args.qtype.split(',')}
        traj_order = [tid for tid in traj_order if traj_qtype.get(tid) in q_set]

    if args.range:
        start, end = args.range
        traj_order = traj_order[max(0, start-1) : end]
        print(f"[INFO] Applied range filter: [{start}, {end}], Total items: {len(traj_order)}")
    elif args.top:
        traj_order = traj_order[:args.top]
    elif args.bottom:
        traj_order = traj_order[-args.bottom:]

    if not traj_order:
        print("No trajectories found matching criteria."); return

    # Step 3: Statistics containers
    total_wins = 0
    total_reward = 0.0
    steps_list = []
    valid_traj_cnt = 0
    valid_traj_wins = 0
    
    all_step_errors = Counter()
    final_step_errors = Counter()
    first_invalid_dist = Counter()
    
    # Evidence statistics
    single_frames = Aggregator(); single_images = Aggregator()
    single_clip = Aggregator();   single_audio = Aggregator()
    per_traj_evidence = defaultdict(lambda: {"f": 0, "i": 0, "c": 0.0, "a": 0.0})
    
    # Action distribution
    action_type_all = Counter()
    action_type_valid = Counter()
    parse_fail_cnt = 0
    parse_err_detail = Counter()

    # Confidence statistics
    conf_buckets = defaultdict(lambda: {"total": 0, "wins": 0})

    # Bucketing and VSI statistics
    step_buckets = defaultdict(lambda: {"total": 0, "wins": 0, "valid": 0, "reward": 0.0})
    # Duration bucketing, based on args.interval
    duration_buckets = defaultdict(lambda: {"total": 0, "wins": 0, "reward": 0.0, "turns": 0})
    interval_seconds = args.interval * 60
    
    otype_total, otype_wins, otype_reward = Counter(), Counter(), defaultdict(float)

    # Step 4: Iterate over Trajectories
    for tid in traj_order:
        steps = sorted(traj_steps[tid], key=lambda x: x.get("cur_step", 0))
        final_rec = steps[-1]
        
        info = final_rec.get("step_info", {})
        won = bool(info.get("won"))
        reward = float(info.get("reward", 0.0))
        steps_len = final_rec.get("cur_step", 0) + 1
        
        # Get duration
        duration = info.get("duration_seconds") or final_rec.get("video_duration")
        
        total_wins += 1 if won else 0
        total_reward += reward
        steps_list.append(steps_len)
        
        traj_is_valid = True
        first_invalid = None

        for r in steps:
            s_idx = r.get("cur_step", 0)
            s_info = r.get("step_info", {})
            is_valid_flag = s_info.get("is_action_valid", True)
            err_code = s_info.get("error_code") or r.get("error_code")
            
            if err_code: all_step_errors[err_code] += 1
            
            obj, p_err = parse_action_with_err(r.get("output"))
            
            if r == final_rec:
                conf = obj.get("confidence")
                if conf is not None:
                    try:
                        c_val = round(float(conf), 2)
                        conf_buckets[c_val]["total"] += 1
                        if won: conf_buckets[c_val]["wins"] += 1
                    except: pass

            if p_err:
                parse_fail_cnt += 1
                parse_err_detail[p_err] += 1
                if traj_is_valid: 
                    traj_is_valid = False
                    first_invalid = s_idx
                continue

            act_obj = obj.get("action") if isinstance(obj.get("action"), dict) else obj
            atype = act_obj.get("type", "UNKNOWN") if isinstance(act_obj, dict) else "UNKNOWN"
            action_type_all[atype] += 1

            if is_valid_flag and not err_code:
                action_type_valid[atype] += 1
                try:
                    if atype == "get_frames":
                        val = int(act_obj.get("num", len(act_obj.get("timestamps", []))))
                        single_frames.add(val); per_traj_evidence[tid]["f"] += val
                    elif atype == "get_images":
                        val = int(act_obj.get("num", 1))
                        single_images.add(val); per_traj_evidence[tid]["i"] += val
                    elif atype in ["get_clip", "get_audio"]:
                        val = max(0.0, float(act_obj.get("end", 0)) - float(act_obj.get("start", 0)))
                        if atype == "get_clip":
                            single_clip.add(val); per_traj_evidence[tid]["c"] += val
                        else:
                            single_audio.add(val); per_traj_evidence[tid]["a"] += val
                except: pass
            else:
                if traj_is_valid:
                    traj_is_valid = False
                    first_invalid = s_idx

        if traj_is_valid:
            valid_traj_cnt += 1
            if won: valid_traj_wins += 1
        else:
            if first_invalid is not None:
                first_invalid_dist[first_invalid] += 1

        f_err = info.get("error_code") or final_rec.get("error_code")
        if f_err: final_step_errors[f_err] += 1
        
        # Step Buckets
        sb = step_buckets[steps_len]
        sb["total"] += 1; sb["wins"] += 1 if won else 0; sb["reward"] += reward
        if traj_is_valid: sb["valid"] += 1

        # Duration Buckets (Dynamic Interval)
        if duration is not None:
            d_idx = int(float(duration) // interval_seconds)
            db = duration_buckets[d_idx]
            db["total"] += 1
            db["wins"] += 1 if won else 0
            db["reward"] += reward
            db["turns"] += steps_len

        if is_vsi_bench:
            q, v = traj_extra.get(tid, ("",""))
            ot = info.get("origin_question_type") or VSI_TABLE.get((q,v), "UNKNOWN")
            otype_total[ot] += 1
            otype_wins[ot] += 1 if won else 0
            otype_reward[ot] += reward

    # ---------------------------------------------------------------------
    # 5. Output report
    # ---------------------------------------------------------------------
    line = "=" * 100
    print(line)
    print(f"REPORT: {args.log_path} | Count: {len(traj_order)}")
    if args.range: print(f"Selected Range: {args.range[0]} to {args.range[1]} (Inclusive)")
    print(line)

    print(f"Overall Accuracy                     : {100*total_wins/len(traj_order):.2f}% ({total_wins}/{len(traj_order)})")
    print(f"Average Final Reward                 : {total_reward/len(traj_order):.4f}")
    if is_vsi_bench and otype_total:
        macro = sum([100*otype_wins[k]/otype_total[k] for k in otype_total]) / len(otype_total)
        print(f"[VSI-Bench] Macro Accuracy           : {macro:.2f}%")

    print(f"Trajectory Validity Ratio            : {100*valid_traj_cnt/len(traj_order):.2f}% ({valid_traj_cnt}/{len(traj_order)})")
    print(f"Accuracy on VALID Trajectories       : {(100*valid_traj_wins/valid_traj_cnt if valid_traj_cnt else 0):.2f}%")
    print(f"Steps (Mean/Median/Std)              : {statistics.mean(steps_list):.2f} / {statistics.median(steps_list)} / {(statistics.stdev(steps_list) if len(steps_list)>1 else 0):.2f}")

    # Stats display: Stats by Duration
    print(f"\nStats Grouped by Video Duration ({args.interval}-minute intervals):")
    if duration_buckets:
        print(f"{'Duration Range':>15} | {'Acc%':>7} | {'Wins/Total':>11} | {'AvgR':>6} | {'AvgTurns':>8}")
        print("-" * 68)
        for d_idx in sorted(duration_buckets.keys()):
            db = duration_buckets[d_idx]
            lower = d_idx * args.interval
            upper = (d_idx + 1) * args.interval
            range_str = f"{lower}-{upper} min"
            acc = 100 * db["wins"] / db["total"]
            avg_r = db["reward"] / db["total"]
            avg_t = db["turns"] / db["total"]
            print(f"{range_str:>15} | {acc:7.2f}% | {db['wins']:3d}/{db['total']:<7d} | {avg_r:6.3f} | {avg_t:8.2f}")
    else:
        print("  No duration metadata found.")

    print("\nAccuracy by Final Turn Confidence (Calibration):")
    if conf_buckets:
        print(f"{'Conf Level':>10} | {'Acc%':>7} | {'Wins/Total':>12}")
        print("-" * 35)
        for c in sorted(conf_buckets.keys(), reverse=True):
            b = conf_buckets[c]
            acc = 100 * b["wins"] / b["total"]
            print(f"{c:10.2f} | {acc:7.2f}% | {b['wins']:>4d} / {b['total']:<4d}")
    else:
        print("  No confidence field found in model outputs.")

    print("\nEvidence Usage (Cumulative Per Trajectory):")
    def get_cum_str(data):
        if not data: return "N/A"
        return f"mean={statistics.mean(data):.2f}, med={statistics.median(data):.2f}, p90={pct(data, 90):.2f}, max={max(data):.2f}"
    print(f"  frames         : {get_cum_str([v['f'] for v in per_traj_evidence.values()])}")
    print(f"  clip(s)        : {get_cum_str([v['c'] for v in per_traj_evidence.values()])}")
    print(f"  audio(s)       : {get_cum_str([v['a'] for v in per_traj_evidence.values()])}")

    print("\nEvidence Usage (Per Valid Action):")
    for lbl, agg in [("frames", single_frames), ("images", single_images), ("clip", single_clip), ("audio", single_audio)]:
        s = agg.stat()
        if s: print(f"  {lbl:<7}: mean={s['mean']}, median={s['median']}, max={s['max']}, cnt={s['cnt']}")

    print("\nAction Distribution (Step-level):")
    print(f"{'Action Type':<18} | {'Total':<8} | {'Valid':<8} | {'Success%':<8}")
    for atype in sorted(action_type_all, key=lambda x: -action_type_all[x]):
        tot = action_type_all[atype]
        val = action_type_valid[atype]
        print(f"{atype:<18} | {tot:<8} | {val:<8} | {100*val/tot:>7.1f}%")

    print("\nStats Grouped by Steps:")
    print(f"{'Steps':>6} | {'Acc%':>7} | {'Wins/Total':>11} | {'Valid':>6} | {'AvgR':>6}")
    for s in sorted(step_buckets.keys()):
        b = step_buckets[s]
        print(f"{s:6d} | {100*b['wins']/b['total']:7.2f}% | {b['wins']:3d}/{b['total']:<7d} | {b['valid']:6d} | {b['reward']/b['total']:6.3f}")

    print("\nError Distribution:")
    print(f"  Parse Failures: {parse_fail_cnt}")
    for k, v in parse_err_detail.items(): print(f"    - {k}: {v}")
    print("  Final Step Error Codes (Termination):")
    for ec, cnt in final_step_errors.most_common():
        print(f"    - {ec:<24}: {cnt:5d} ({100*cnt/len(traj_order):.2f}%)")
    print("  All Step Errors (Full Process):")
    for ec, cnt in all_step_errors.most_common(10):
        print(f"    - {ec:<24}: {cnt:5d}")

    if is_vsi_bench and otype_total:
        print("\n[VSI-Bench: Per-Subtask]")
        for k in sorted(otype_total):
            print(f"  {k:<30}: Acc {100*otype_wins[k]/otype_total[k]:6.2f}% ({otype_wins[k]}/{otype_total[k]}), AvgR {otype_reward[k]/otype_total[k]:.4f}")

    if vsi_meta_source: print(f"\n[INFO] VSI meta source: {vsi_meta_source}")
    print(line)

if __name__ == "__main__":
    main()

# Usage example:
# python stat_sub.py path/to/log.jsonl --interval 10
