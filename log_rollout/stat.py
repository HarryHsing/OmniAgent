#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unified_benchmark_stat.py
--------------------------------------------------------------------
Features:
1. Trajectory-level statistics: Accuracy, Reward, Step distribution (Mean/Var/Median/Coverage)
2. VSI-Bench support: Maps origin_question_type, computes Macro-Avg
3. Action-level statistics: Action type distribution (Valid vs Invalid), parse error breakdown
4. Evidence usage statistics: get_frames/clip/audio usage count and duration (Per-Action & Per-Trajectory)
5. Filtering: Supports --qtype whitelist, --top / --bottom sampling
--------------------------------------------------------------------
"""

import json
import sys
import argparse
import re
import math
import os
import statistics
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# 1. Helper classes and parsing functions
# ---------------------------------------------------------------------

class Aggregator:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.min = math.inf
        self.max = -math.inf
        self.values = []

    def add(self, v):
        self.count += 1
        self.total += v
        self.min = min(self.min, v)
        self.max = max(self.max, v)
        self.values.append(v)

    def stat(self):
        if self.count == 0: return None
        return {
            "cnt": self.count,
            "min": round(self.min, 4),
            "max": round(self.max, 4),
            "mean": round(self.total / self.count, 4),
            "median": round(statistics.median(self.values), 4),
            "p90": round(self.get_pct(90), 4)
        }

    def get_pct(self, p):
        if not self.values: return 0.0
        sorted_vals = sorted(self.values)
        k = max(0, min(len(sorted_vals)-1, int(round(p/100*(len(sorted_vals)-1)))))
        return sorted_vals[k]

def parse_action_json(raw: Optional[str]) -> Tuple[Dict, str]:
    """Extract JSON action block from model output"""
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
        if isinstance(obj, dict) and isinstance(obj.get("action"), dict):
            return obj, ""
        return {}, "not_action_dict"
    except Exception: return {}, "json_decode_err"

# ---------------------------------------------------------------------
# 2. VSI-Bench metadata support
# ---------------------------------------------------------------------
DEFAULT_VSI_META = [
    p.strip()
    for p in re.split(r"[:;]", os.getenv("VSI_META_PATHS", ""))
    if p.strip()
]
VSI_TABLE = {} # (question, video) -> origin_question_type

def load_vsi_table(candidates):
    global VSI_TABLE
    meta_found = None
    for p in candidates:
        if p and os.path.isfile(p):
            meta_found = p
            break
    if not meta_found:
        print(f"[WARN] No VSI meta found in {candidates}", file=sys.stderr)
        return None
    
    try:
        with open(meta_found, "r", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                q = rec.get("question", "").strip()
                v = rec.get("video", "").strip()
                o = rec.get("origin_question_type")
                if q and v and o: VSI_TABLE[(q, v)] = o
        print(f"[INFO] Loaded VSI meta from {meta_found}", file=sys.stderr)
    except Exception as e:
        print(f"[WARN] Failed to load meta: {e}", file=sys.stderr)
    return meta_found

# ---------------------------------------------------------------------
# 3. Main program logic
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Unified Trajectory & Action Analysis Script")
    parser.add_argument("log_path", help="Path to log.jsonl")
    parser.add_argument("-t", "--top", type=int)
    parser.add_argument("-b", "--bottom", type=int)
    parser.add_argument("--qtype", help="Comma separated question types")
    parser.add_argument("--vsi-meta", help="VSI meta paths separated by colon")
    args = parser.parse_args()

    # Parameter validation
    if args.top and args.bottom: parser.error("--top and --bottom are exclusive")
    qtype_set = {t.strip() for t in args.qtype.split(',')} if args.qtype else set()

    # VSI Meta initialization
    vsi_meta_list = list(DEFAULT_VSI_META)
    if args.vsi_meta:
        vsi_meta_list = [p.strip() for p in re.split(r"[:;]", args.vsi_meta) if p.strip()] + vsi_meta_list
    is_vsi_bench = "VSI-Bench" in args.log_path
    vsi_meta_ok = load_vsi_table(vsi_meta_list) if is_vsi_bench else None

    # ---------------------------------------------------------------------
    # Data loading and initial filtering
    # ---------------------------------------------------------------------
    traj_data = defaultdict(list)
    traj_order = []
    traj_qtype = {}
    traj_vsi_info = {}

    with open(args.log_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                rec = json.loads(line)
            except: continue
            
            tid = rec.get("traj_uid") or rec.get("traj_id") or rec.get("uid")
            step = rec.get("cur_step")
            if tid is None or step is None: continue

            if tid not in traj_data:
                traj_order.append(tid)
            traj_data[tid].append(rec)

            if tid not in traj_qtype:
                qt = rec.get("step_info", {}).get("question_type") or rec.get("question_type")
                if qt: traj_qtype[tid] = qt
            
            if is_vsi_bench and tid not in traj_vsi_info:
                qtxt = rec.get("step_info", {}).get("question") or rec.get("question", "")
                vpth = rec.get("step_info", {}).get("video") or rec.get("video", "")
                traj_vsi_info[tid] = (qtxt.strip(), vpth.strip())

    # Apply filter (qtype)
    if qtype_set:
        traj_order = [tid for tid in traj_order if traj_qtype.get(tid) in qtype_set]
    
    # Apply filter (top/bottom)
    if args.top:
        keep_ids = set(traj_order[:args.top])
    elif args.bottom:
        keep_ids = set(traj_order[-args.bottom:])
    else:
        keep_ids = set(traj_order)

    # ---------------------------------------------------------------------
    # Core statistics loop
    # ---------------------------------------------------------------------
    # Trajectory-level variables
    total_traj = 0
    total_wins = 0
    final_reward_sum = 0.0
    steps_list = []
    
    # Validity statistics
    traj_valid_cnt = 0
    step_limit_traj_cnt = 0
    valid_traj_wins = 0
    valid_traj_reward_sum = 0.0
    
    # Detailed statistics
    first_invalid_step_cnt = defaultdict(int)
    final_error_counts = Counter()
    final_valid_flag_cnt = {"valid": 0, "invalid": 0}
    step_buckets = defaultdict(lambda: {"total": 0, "wins": 0, "valid_last": 0, "reward_sum": 0.0})
    
    # VSI-Bench specific
    otype_stats = defaultdict(lambda: {"total": 0, "wins": 0, "reward_sum": 0.0})

    # Action-level variables (analyse_actions logic)
    action_type_all = Counter()
    action_type_valid = Counter()
    action_type_invalid = Counter()
    parse_errors = Counter()
    
    # Evidence statistics
    g_frames = Aggregator()
    g_clip = Aggregator()
    g_audio = Aggregator()
    per_traj_evidence = defaultdict(lambda: {"frames": 0, "clip": 0.0, "audio": 0.0})

    for tid in traj_order:
        if tid not in keep_ids: continue
        steps = sorted(traj_data[tid], key=lambda x: x.get("cur_step", 0))
        if not steps: continue
        
        total_traj += 1
        final_rec = steps[-1]
        
        # Trajectory basic info
        won = bool(final_rec.get("step_info", {}).get("won"))
        num_steps = final_rec.get("cur_step", 0) + 1
        reward = final_rec.get("step_info", {}).get("reward", 0.0)
        
        total_wins += 1 if won else 0
        final_reward_sum += reward
        steps_list.append(num_steps)

        # Step bucketing
        sb = step_buckets[num_steps]
        sb["total"] += 1
        sb["wins"] += 1 if won else 0
        sb["reward_sum"] += reward

        # Trajectory validity check (Traj-level validity)
        overall_valid = True
        traj_first_invalid = None
        
        # Iterate over each step (Step-level Analysis)
        for rec in steps:
            sidx = rec.get("cur_step", 0)
            # Get validity flag
            is_step_valid = rec.get("is_action_valid")
            if is_step_valid is None:
                is_step_valid = rec.get("step_info", {}).get("is_action_valid", True)
            
            err_code = rec.get("error_code") or rec.get("step_info", {}).get("error_code", "")
            
            # Action parsing
            obj, p_err = parse_action_json(rec.get("output"))
            if p_err:
                parse_errors[p_err] += 1
                is_step_valid = False
            else:
                act = obj["action"]
                atype = act.get("type", "UNKNOWN")
                action_type_all[atype] += 1
                
                # If current step is valid, compute evidence statistics
                if is_step_valid and not err_code:
                    action_type_valid[atype] += 1
                    try:
                        if atype == "get_frames":
                            n = len(act.get("timestamps", []))
                            g_frames.add(n); per_traj_evidence[tid]["frames"] += n
                        elif atype == "get_clip":
                            d = max(0.0, float(act.get("end", 0)) - float(act.get("start", 0)))
                            g_clip.add(d); per_traj_evidence[tid]["clip"] += d
                        elif atype == "get_audio":
                            d = max(0.0, float(act.get("end", 0)) - float(act.get("start", 0)))
                            g_audio.add(d); per_traj_evidence[tid]["audio"] += d
                    except: pass
                else:
                    action_type_invalid[atype] += 1

            # Update trajectory-level validity flag
            if (not is_step_valid) or (err_code != ""):
                if overall_valid:
                    overall_valid = False
                    traj_first_invalid = sidx

        # Trajectory result finalization
        if overall_valid:
            traj_valid_cnt += 1
            if won: valid_traj_wins += 1
            valid_traj_reward_sum += reward
        else:
            if traj_first_invalid is not None:
                first_invalid_step_cnt[traj_first_invalid] += 1
        
        # Final step status
        f_err = final_rec.get("error_code") or final_rec.get("step_info", {}).get("error_code", "")
        if f_err:
            final_error_counts[f_err] += 1
            if f_err == "STEP_LIMIT_REACHED": step_limit_traj_cnt += 1
        
        f_valid_flag = final_rec.get("is_action_valid")
        if f_valid_flag is None: f_valid_flag = final_rec.get("step_info", {}).get("is_action_valid", True)
        
        if f_valid_flag and not f_err:
            final_valid_flag_cnt["valid"] += 1
            sb["valid_last"] += 1
        else:
            final_valid_flag_cnt["invalid"] += 1

        # VSI-Bench Origin Type statistics
        if is_vsi_bench:
            qtxt, vpth = traj_vsi_info.get(tid, ("", ""))
            otype = final_rec.get("step_info", {}).get("origin_question_type")
            if otype is None: otype = VSI_TABLE.get((qtxt, vpth), "UNKNOWN")
            otype_stats[otype]["total"] += 1
            otype_stats[otype]["wins"] += 1 if won else 0
            otype_stats[otype]["reward_sum"] += reward

    # ---------------------------------------------------------------------
    # Compute and print report
    # ---------------------------------------------------------------------
    if total_traj == 0:
        print("No trajectories match the filters."); return

    line_sep = "=" * 100
    print(line_sep)
    tag = []
    if qtype_set: tag.append(f"qtype={','.join(sorted(qtype_set))}")
    if args.top: tag.append(f"top{args.top}")
    if args.bottom: tag.append(f"bottom{args.bottom}")
    print(f"REPORT FOR: {args.log_path} ({' / '.join(tag) if tag else 'ALL'})")
    print(f"Total Trajectories: {total_traj}")
    print(line_sep)

    # 1. Trajectory accuracy and reward
    overall_acc = 100 * total_wins / total_traj
    avg_reward = final_reward_sum / total_traj
    print(f"Overall Accuracy          : {overall_acc:.2f}% ({total_wins}/{total_traj})")
    print(f"Average Final Reward      : {avg_reward:.4f}")
    
    # 2. VSI-Bench macro average
    if is_vsi_bench and otype_stats:
        m_acc_list = [(v["wins"]/v["total"]) for v in otype_stats.values()]
        m_rew_list = [(v["reward_sum"]/v["total"]) for v in otype_stats.values()]
        print(f"[VSI] Macro Accuracy      : {100*sum(m_acc_list)/len(m_acc_list):.2f}%")
        print(f"[VSI] Macro Avg Reward    : {sum(m_rew_list)/len(m_rew_list):.4f}")

    # 3. Validity statistics
    valid_ratio = 100 * traj_valid_cnt / total_traj
    non_sl_total = total_traj - step_limit_traj_cnt
    valid_ratio_excl = 100 * traj_valid_cnt / non_sl_total if non_sl_total > 0 else 0
    print(f"Traj-level Valid Ratio    : {valid_ratio:.2f}% ({traj_valid_cnt}/{total_traj})")
    print(f"Valid Ratio (excl SL)     : {valid_ratio_excl:.2f}% (SL_count={step_limit_traj_cnt})")
    print(f"Acc on Valid Trajectories : {100*valid_traj_wins/(traj_valid_cnt or 1):.2f}%")

    # 4. Step statistics
    mean_s = statistics.mean(steps_list)
    std_s = statistics.stdev(steps_list) if len(steps_list) > 1 else 0
    med_s = statistics.median(steps_list)
    cov_cnt = sum(1 for s in steps_list if s <= (mean_s + std_s))
    print(f"Steps (Mean/Std/Median)   : {mean_s:.2f} / {std_s:.2f} / {med_s:.2f}")
    print(f"Steps Coverage (μ+σ)      : {100*cov_cnt/total_traj:.2f}% ({cov_cnt}/{total_traj})")

    # 5. Action distribution (Valid/Invalid)
    print("\nAction Distribution (Step-level):")
    print(f"{'Action Type':<18} | {'Total':<8} | {'Valid':<8} | {'Invalid':<8} | {'Success%':<8}")
    print("-" * 65)
    for atype in sorted(action_type_all, key=lambda x: -action_type_all[x]):
        tot = action_type_all[atype]
        val = action_type_valid[atype]
        inv = action_type_invalid[atype]
        ratio = 100 * val / tot if tot > 0 else 0
        print(f"{atype:<18} | {tot:<8} | {val:<8} | {inv:<8} | {ratio:>7.1f}%")

    # 6. Evidence statistics (Per-Trajectory Cumulative)
    print("\nEvidence Usage (Cumulative Per Trajectory):")
    p_ev = per_traj_evidence.values()
    if p_ev:
        f_list = [v["frames"] for v in p_ev]
        c_list = [v["clip"] for v in p_ev]
        a_list = [v["audio"] for v in p_ev]
        def fmt_ev(name, data):
            if not data: return f"  {name:<8}: N/A"
            return f"  {name:<8}: mean={statistics.mean(data):.2f}, med={statistics.median(data):.2f}, p90={round(statistics.quantiles(data, n=10)[8], 2) if len(data)>1 else data[0]:.2f}, max={max(data):.2f}"
        print(fmt_ev("frames", f_list))
        print(fmt_ev("clip(s)", c_list))
        print(fmt_ev("audio(s)", a_list))

    # 7. Evidence statistics (Per-Action Single)
    print("\nEvidence Usage (Per Single Action):")
    print(f"  frames  : {g_frames.stat()}")
    print(f"  clip(s) : {g_clip.stat()}")
    print(f"  audio(s): {g_audio.stat()}")

    # 8. Error breakdown
    print("\nParse Errors (Step-level):")
    for k, v in parse_errors.items(): print(f"  {k:<20}: {v}")
    
    print("\nFinal Step Error Distribution:")
    for ec, cnt in final_error_counts.most_common():
        print(f"  {ec:<24}: {cnt:5d} ({100*cnt/total_traj:6.2f}%)")

    # 9. Bucketed table
    print("\nStats Grouped by Steps:")
    print(f"{'Steps':>5} | {'Acc%':>7} | {'Wins/Total':>11} | {'Share%':>7} | {'ValidL':>6} | {'AvgR':>6}")
    print("-" * 60)
    for s in sorted(step_buckets.keys()):
        b = step_buckets[s]
        acc = 100 * b["wins"] / b["total"]
        share = 100 * b["total"] / total_traj
        print(f"{s:5d} | {acc:7.2f}% | {b['wins']:3d}/{b['total']:<7d} | {share:7.2f}% | {b['valid_last']:6d} | {b['reward_sum']/b['total']:6.3f}")

    if vsi_meta_ok: print(f"\n[INFO] VSI Meta used: {vsi_meta_ok}")
    print(line_sep)

if __name__ == "__main__":
    main()
