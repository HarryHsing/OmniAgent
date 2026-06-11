#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, math, sys, logging, csv, statistics
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

# ------------------------------------------------------------
def _parse_action_with_err(raw: Optional[str]) -> Tuple[Dict, str]:
    if not raw: return {}, "no_output"
    s = str(raw).strip()
    try:
        if s.startswith("```"):
            import re
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

def _get_val(rec, key, default=None):
    if key in rec: return rec[key]
    if isinstance(rec.get("step_info"), dict):
        return rec["step_info"].get(key, default)
    return default

def pct(lst, p):
    if not lst: return 0.0
    k = max(0, min(len(lst) - 1, int(round(p / 100 * (len(lst) - 1)))))
    return sorted(lst)[k]

class Agg:
    def __init__(self): self.c=0; self.t=0.0; self.mn=math.inf; self.mx=-math.inf
    def add(self,v):
        self.c+=1; self.t+=v; self.mn=min(self.mn,v); self.mx=max(self.mx,v)
    def stat(self):
        return None if self.c==0 else dict(cnt=self.c, min=round(self.mn,2), max=round(self.mx,2), mean=round(self.t/self.c,2))

# ------------------------------------------------------------
def analyse(path, keep_ids):
    stats = {
        "total": 0, "valid": 0, "invalid": 0, "parse_fail": 0,
        "reached_limit_fail": 0,
        "type_all": Counter(), "type_valid": Counter(), "type_invalid": Counter(),
        "err_code_cnt": Counter(), "parse_err_detail": Counter(),
        "g_frames": Agg(), "g_clip": Agg(), "g_audio": Agg(),
        "per_traj": defaultdict(lambda: {"frames": 0, "clip": 0.0, "audio": 0.0})
    }

    with open(path, 'r', encoding='utf-8') as fh:
        for ln in fh:
            try: r = json.loads(ln)
            except: continue
            
            tid = r.get("traj_uid") or r.get("traj_id") or r.get("uid")
            if tid not in keep_ids: continue
            
            stats["total"] += 1
            cur_step = r.get("cur_step", 0)
            max_steps = _get_val(r, "max_steps", 80)
            is_valid_flag = _get_val(r, "is_action_valid", True)
            err_code = _get_val(r, "error_code")
            
            obj, p_err = _parse_action_with_err(r.get("output"))
            
            if p_err:
                stats["parse_fail"] += 1
                stats["parse_err_detail"][p_err] += 1
                if cur_step >= max_steps: stats["reached_limit_fail"] += 1
                continue
            
            act = obj["action"]
            atype = act.get("type", "UNKNOWN")
            stats["type_all"][atype] += 1

            if not is_valid_flag or (err_code and err_code != ""):
                stats["invalid"] += 1
                stats["type_invalid"][atype] += 1
                if err_code: stats["err_code_cnt"][err_code] += 1
                if cur_step >= max_steps: stats["reached_limit_fail"] += 1
                continue
            
            stats["valid"] += 1
            stats["type_valid"][atype] += 1
            
            try:
                if atype == "get_frames":
                    n = len(act.get("timestamps", []))
                    stats["g_frames"].add(n); stats["per_traj"][tid]["frames"] += n
                elif atype == "get_clip":
                    d = max(0.0, float(act.get("end",0)) - float(act.get("start",0)))
                    stats["g_clip"].add(d); stats["per_traj"][tid]["clip"] += d
                elif atype == "get_audio":
                    d = max(0.0, float(act.get("end",0)) - float(act.get("start",0)))
                    stats["g_audio"].add(d); stats["per_traj"][tid]["audio"] += d
            except: pass

    return stats

def print_report(st, desc):
    print("="*100)
    print(f"REPORT: {desc}")
    print("="*100)
    
    # 1. Line count statistics
    chk_sum = st['valid'] + st['invalid'] + st['parse_fail']
    print(f"Line Summary:")
    print(f"  Total Lines : {st['total']}")
    print(f"  Valid       : {st['valid']}")
    print(f"  Invalid     : {st['invalid']}")
    print(f"  Parse Fail  : {st['parse_fail']}")
    print(f"  Check Sum   : {chk_sum} ({'MATCH' if chk_sum==st['total'] else 'MISMATCH!'})")
    print(f"\nFailures at Max Steps (step >= max_steps): {st['reached_limit_fail']}")

    # 2. Error details
    print("\n[1] Parse Errors:")
    for k, v in st["parse_err_detail"].items(): print(f"  - {k:<20}: {v}")
    print("\n[2] Execution Errors (error_code):")
    for code, count in st["err_code_cnt"].most_common(): print(f"  - {code:<20}: {count}")

    # 3. Cumulative evidence statistics per video (Per-trajectory)
    print("\n[3] Per-trajectory valid evidence usage (Cumulative per video):")
    per = st["per_traj"]
    if per:
        f_list = [v["frames"] for v in per.values()]
        c_list = [v["clip"] for v in per.values()]
        a_list = [v["audio"] for v in per.values()]

        def get_dist(data):
            if not data: return "N/A"
            return {"mean": round(statistics.mean(data), 2), "median": round(statistics.median(data), 2),
                    "p90": round(pct(data, 90), 2), "max": round(max(data), 2)}
        print(f"  frames  : {get_dist(f_list)}")
        print(f"  clip(s) : {get_dist(c_list)}")
        print(f"  audio(s): {get_dist(a_list)}")

    # 4. Per-action statistics (Per-action)
    print("\n[4] Evidence Stats (Per Valid Action granularity):")
    def fmt_agg(agg_name, unit=""):
        s = st[agg_name].stat()
        if not s: return "N/A"
        return f"mean={s['mean']}, max={s['max']}, total_cnt={s['cnt']} {unit}"
    print(f"  Frames : {fmt_agg('g_frames')}")
    print(f"  Clips  : {fmt_agg('g_clip', 'sec')}")
    print(f"  Audio  : {fmt_agg('g_audio', 'sec')}")

    # 5. Action distribution
    print("\n[5] Action Type Distribution (Valid / Invalid):")
    all_types = set(st["type_valid"].keys()) | set(st["type_invalid"].keys())
    for t in sorted(all_types):
        print(f"  - {t:<15}: Valid={st['type_valid'][t]:<5} Invalid={st['type_invalid'][t]:<5}")

# ------------------------------------------------------------
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("file")
    pa.add_argument("--top", type=int)
    args = pa.parse_args()

    all_ids = []
    with open(args.file, 'r') as f:
        for line in f:
            try:
                d = json.loads(line)
                tid = d.get("traj_uid") or d.get("traj_id") or d.get("uid")
                if tid and tid not in all_ids: all_ids.append(tid)
            except: continue
    
    target_ids = all_ids[:args.top] if args.top else all_ids
    res = analyse(args.file, set(target_ids))
    print_report(res, args.file)
