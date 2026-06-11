#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# stat_iter.py  –  batch success-rate (wins / trajectories) + avg-reward
#                  special handling for OmniAgent VSI-Bench
# ---------------------------------------------------------------------------

import os
import sys
import fnmatch
import argparse
import re
import multiprocessing as mp
from collections import defaultdict

try:
    import ujson as json
except ImportError:
    import json

# ---------------------------------------------------------------------------
# ───── 1.  Multiple candidate VSI-Bench metadata files  ──────────────────────────────────────
# ---------------------------------------------------------------------------
DEFAULT_VSI_META = [
    p.strip()
    for p in re.split(r"[:;]", os.getenv("VSI_META_PATHS", ""))
    if p.strip()
]

VSI_TABLE   = {}
VSI_LOADED  = False
VSI_META_OK = None


def choose_vsi_meta(candidates):
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def load_vsi_bench_table(candidates):
    """Load VSI metadata into the global lookup table (one-time)."""
    global VSI_LOADED, VSI_TABLE, VSI_META_OK
    if VSI_LOADED:
        return

    meta = choose_vsi_meta(candidates)
    VSI_META_OK = meta
    if meta is None:
        print("[WARN] no valid VSI-Bench meta found", file=sys.stderr)
        VSI_LOADED = True
        return

    try:
        with open(meta, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                q = rec.get("question", "").strip()
                v = rec.get("video", "").strip()
                o = rec.get("origin_question_type")
                if q and v and o:
                    VSI_TABLE[(q, v)] = o
        print(f"[INFO] loaded VSI-Bench meta ({len(VSI_TABLE)} items) ← {meta}",
              file=sys.stderr)
    except Exception as e:
        print(f"[WARN] fail to load VSI meta {meta}: {e}", file=sys.stderr)
        VSI_TABLE = {}

    VSI_LOADED = True


# ---------------------------------------------------------------------------
# ───── 2.  Parse a single jsonl file  ─────────────────────────────────────────────
# ---------------------------------------------------------------------------
def analyse_file(args):
    """
    Returns:
      path, wins, total, reward_sum,
      vsi_macro_acc, vsi_macro_rew,
      vsi_detail: None or {otype:(wins,total,avg_rew)}
    """
    (path, top_n, bottom_n, avg_n, qtype_set, vsi_meta) = args
    is_vsi = "VSI" in path
    if is_vsi:
        load_vsi_bench_table(vsi_meta)

    traj_last   = {}
    first_seen  = {}
    traj_qtype  = {}
    traj_extra  = {}      # tid -> (q, v, origin)

    try:
        with open(path, "r", encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue

                tid  = rec.get("traj_uid") or rec.get("traj_id")
                step = rec.get("cur_step")
                if tid is None or step is None:
                    continue

                if tid not in first_seen:
                    first_seen[tid] = idx

                s_info = rec.get("step_info", {})
                won    = bool(s_info.get("won", False))
                reward = float(s_info.get("reward", 0.0))
                qtype  = s_info.get("question_type")

                if tid not in traj_qtype and qtype is not None:
                    traj_qtype[tid] = qtype

                if is_vsi and tid not in traj_extra:
                    qtxt  = s_info.get("question") or rec.get("question", "")
                    vpath = s_info.get("video")    or rec.get("video", "")
                    otype = s_info.get("origin_question_type")
                    if otype is None and VSI_TABLE:
                        otype = VSI_TABLE.get((qtxt.strip(), vpath.strip()))
                    traj_extra[tid] = (qtxt, vpath, otype or "UNKNOWN")

                if tid not in traj_last or step > traj_last[tid][0]:
                    traj_last[tid] = (step, won, reward)

    except Exception as e:
        print(f"[WARN] skip {path}: {e}", file=sys.stderr)
        return (path, 0, 0, 0.0, None, None, None)

    # Question type filter
    tids = [t for t in traj_last
            if not qtype_set or traj_qtype.get(t) in qtype_set]
    if not tids:
        return (path, 0, 0, 0.0, None, None, None)

    # Sampling
    if top_n:
        tids.sort(key=lambda t: first_seen[t]);   sel = set(tids[:top_n])
    elif bottom_n:
        tids.sort(key=lambda t: first_seen[t]);   sel = set(tids[-bottom_n:])
    elif avg_n:
        tids.sort(key=lambda t: first_seen[t])
        k = len(tids)
        if k <= avg_n:
            sel = set(tids)
        else:
            idxs = {int(i * k / avg_n) for i in range(avg_n)}
            sel = {tids[i] for i in sorted(idxs)}
    else:
        sel = set(tids)

    # Basic statistics
    wins = total = 0
    reward_sum = 0.0
    for tid in sel:
        _s, w, r = traj_last[tid]
        total += 1
        wins  += 1 if w else 0
        reward_sum += r

    if not is_vsi:
        return (path, wins, total, reward_sum, None, None, None)

    # VSI-Bench secondary aggregation
    by_type = defaultdict(lambda: [0, 0, 0.0])   # otype -> wins,total,rsum
    for tid in sel:
        _q, _v, otype = traj_extra[tid]
        _s, w, r = traj_last[tid]
        by_type[otype][1] += 1
        by_type[otype][0] += 1 if w else 0
        by_type[otype][2] += r

    accs, rews = [], []
    detail = {}
    for o, (w_t, t_t, r_t) in by_type.items():
        if t_t == 0:
            continue
        accs.append(100.0 * w_t / t_t)
        avg_rew = r_t / t_t
        rews.append(avg_rew)
        detail[o] = (w_t, t_t, avg_rew)

    vsi_macro_acc = sum(accs) / len(accs) if accs else 0.0
    vsi_macro_rew = sum(rews) / len(rews) if rews else 0.0
    return (path, wins, total, reward_sum, vsi_macro_acc, vsi_macro_rew, detail)


# ---------------------------------------------------------------------------
# 3.  Recursively find files
# ---------------------------------------------------------------------------
def find_files(root, pattern):
    for dp, _, files in os.walk(root):
        for fn in files:
            if fnmatch.fnmatch(fn, pattern):
                yield os.path.join(dp, fn)


# ---------------------------------------------------------------------------
# 4.  Sorting helpers (key fix)
# ---------------------------------------------------------------------------
STEP_RE = re.compile(r"_step[_\-]?(\d+)\b", re.IGNORECASE)
def step_key(p):           # regular step
    m = STEP_RE.search(p)
    return int(m.group(1)) if m else (1 << 60)

# Improved regex: match directory names containing auto_eval, extract task name and step number
# e.g.: auto_eval_auto2_VideoMME-Short-Long_1-1_step_100
# group(1) -> VideoMME-Short-Long_1-1, group(2) -> 100
BENCH_STEP_RE = re.compile(
    r"auto_eval_.*?_(.+?)[_\-]step[_\-]?(\d+)", re.IGNORECASE)

def bench_step_key(p):
    # Use the last directory component for matching
    dirname = os.path.basename(os.path.dirname(p))
    m = BENCH_STEP_RE.search(dirname)
    if m:
        # Return (task name string, integer step) for tuple sorting
        return (m.group(1), int(m.group(2)))
    # Fallback: if no match, try matching step in the filename
    m_alt = STEP_RE.search(p)
    if m_alt:
        return ("~", int(m_alt.group(1)))
    return ("~", 1 << 60)


# ---------------------------------------------------------------------------
# 5.  Main program
# ---------------------------------------------------------------------------
def main():
    pa = argparse.ArgumentParser(
        description="Batch success-rate & avg-reward statistics (VSI-Bench aware)")
    pa.add_argument("root")
    pa.add_argument("--pattern", default="0.jsonl")
    pa.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    pa.add_argument("--sort", choices=["name", "acc", "step", "agent_step"],
                    default="name")

    g = pa.add_mutually_exclusive_group()
    g.add_argument("--top", type=int)
    g.add_argument("--bottom", type=int)
    g.add_argument("--avg", type=int)

    pa.add_argument("--qtype", metavar="T[,T...]")
    pa.add_argument("--vsi-meta", metavar="P1[:P2...]", help="candidate meta paths")

    args = pa.parse_args()

    root_dir = os.path.abspath(args.root)
    if not os.path.isdir(root_dir):
        sys.exit(f"{root_dir} is not dir")

    vsi_meta_list = list(DEFAULT_VSI_META)
    if args.vsi_meta:
        parts = re.split(r"[:;]", args.vsi_meta)
        vsi_meta_list = [p.strip() for p in parts if p.strip()] + vsi_meta_list

    paths = list(find_files(root_dir, args.pattern))
    if not paths:
        sys.exit("no matched files")

    print(f"Found {len(paths)} files, workers={args.workers}")

    qset = {t.strip() for t in args.qtype.split(",")} if args.qtype else set()

    jobs = [(p, args.top or 0, args.bottom or 0, args.avg or 0, qset, vsi_meta_list)
            for p in paths]

    with mp.Pool(args.workers) as pool:
        stats = pool.map(analyse_file, jobs)

    # ---------------- Aggregate / Sort ----------------
    rows = []
    overall_macro_acc = 0.0
    file_cnt          = 0
    details_map       = {}

    for rec in stats:
        p, w, t, rsum, vsi_acc, vsi_rew, detail = rec
        if "VSI-Bench" in p and vsi_acc is not None:
            acc   = vsi_acc
            avg_r = vsi_rew
        else:
            if t == 0:
                continue
            acc   = 100.0 * w / t
            avg_r = rsum / t
        rows.append([p, acc, avg_r, w, t])
        details_map[p] = detail
        overall_macro_acc += acc
        file_cnt += 1

    if args.sort == "name":
        rows.sort(key=lambda x: x[0])
    elif args.sort == "acc":
        rows.sort(key=lambda x: -x[1])
    elif args.sort == "step":
        rows.sort(key=lambda x: (step_key(x[0]), x[0]))
    elif args.sort == "agent_step":
        rows.sort(key=lambda x: (*bench_step_key(x[0]), x[0]))

    # ---------------- Output ----------------
    print("\n" + "-" * 120)
    header = f'{"Idx":>4} │ {"Rew":>8} │ {"Win%":>7} │ {"Wins/Total":>11} │ Relative-Path'
    print(header)
    print("-" * 120)

    for i, (p, acc, avg_r, w, t) in enumerate(rows, 1):
        rel = os.path.relpath(p, root_dir)
        ratio = f"{w}/{t}" if t else "-"
        print(f"{i:4d} │ {avg_r:7.3f} │ {acc:6.2f}% │ {ratio:^11} │ {rel}")

        det = details_map.get(p)
        # if det:                   # VSI-Bench details
        #     parts = [f"{k}:{v[1]}({v[2]:.3f})" for k, v in sorted(det.items())]
        #     print("     └─", "  ".join(parts))

    print("-" * 120)
    overall = overall_macro_acc / file_cnt if file_cnt else 0.0
    print(f"OVERALL file-level macro-avg success-rate: {overall:.2f}%")
    if VSI_META_OK:
        print(f"[INFO] VSI meta from: {VSI_META_OK}")
    print("-" * 120)


if __name__ == "__main__":
    main()



'''
python ./log_rollout/stat_iter.py ./log_rollout/log_rollout_val/verl_omni_eval --pattern 0.jsonl --workers 32 --sort agent_step --qtype NUM,MCQ,SIZE
'''
