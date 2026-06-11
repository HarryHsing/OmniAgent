#!/usr/bin/env python3
# check_adv_explosion.py
# Usage: python check_adv_explosion.py run_log.jsonl

import json, sys
import torch
from collections import defaultdict

eps = 1e-6
THRESH = 1e5          # "explosion" threshold


# =============== Extract reward based on log format ===============
def get_scalar_reward(sample_json):
    """
    Adjust the field names here according to your own JSON structure.
    If reward is spread across token_level_scores, compute the sum.
    """
    for key in ("reward", "score", "raw_score"):
        if key in sample_json:
            return float(sample_json[key])

    if "token_level_scores" in sample_json:
        return float(sum(sample_json["token_level_scores"]))

    raise KeyError("No reward field found in sample")


# =============== Old implementation (with BUG) =======================
def compute_advantage_old(rewards, prompt_ids, traj_ids):
    """
    Reproduce old implementation: std = torch.std(torch.tensor([lst]))
    """
    B = len(rewards)
    rewards_t = torch.tensor(rewards, dtype=torch.float32)   # (B,)
    id2scores = defaultdict(list)
    id2mean, id2std = {}, {}
    seen_pairs = set()

    # Collect scores per prompt (index)
    for i in range(B):
        key  = prompt_ids[i]
        pair = (key, traj_ids[i])
        if pair in seen_pairs:
            continue
        id2scores[key].append(rewards_t[i])
        seen_pairs.add(pair)

    # Compute (buggy) mean & std
    for k, lst in id2scores.items():
        if len(lst) == 1:
            id2mean[k] = torch.tensor(0.0)
            id2std[k]  = torch.tensor(1.0)
        else:
            id2mean[k] = torch.mean(torch.tensor(lst))
            # BUG: extra brackets → std becomes 0
            id2std[k]  = torch.std(torch.tensor([lst]))

    # Normalize
    adv = torch.empty_like(rewards_t)
    for i in range(B):
        key = prompt_ids[i]
        adv[i] = (rewards_t[i] - id2mean[key]) / (id2std[key] + eps)

    return adv


# =============== Fixed implementation ===============================
def compute_advantage_fixed(rewards, prompt_ids, traj_ids, clip_val=5.0):
    B = len(rewards)
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    id2scores = defaultdict(list)
    id2mean, id2std = {}, {}
    seen_pairs = set()

    for i in range(B):
        key  = prompt_ids[i]
        pair = (key, traj_ids[i])
        if pair in seen_pairs:
            continue
        id2scores[key].append(rewards_t[i])
        seen_pairs.add(pair)

    for k, lst in id2scores.items():
        if len(lst) == 1:
            id2mean[k] = torch.tensor(0.0)
            id2std[k]  = torch.tensor(1.0)
        else:
            t = torch.stack(lst)
            id2mean[k] = t.mean()
            id2std[k]  = t.std(unbiased=False)

    adv = torch.empty_like(rewards_t)
    for i in range(B):
        key = prompt_ids[i]
        adv[i] = (rewards_t[i] - id2mean[key]) / (id2std[key] + eps)

    # Safety: optional hard clipping
    adv = adv.clamp(-clip_val, clip_val)
    return adv


# =============== Main function ================================
def main(path):
    rewards, prompt_ids, traj_ids = [], [], []

    with open(path, "r") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                js = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[Warn] line {line_no}: JSON decode error → {e}")
                continue

            try:
                r = get_scalar_reward(js)
            except KeyError:
                continue

            rewards.append(r)
            prompt_ids.append(int(js.get("prompt_id", js.get("index", 0))))
            traj_ids.append(int(js.get("traj_idx", line_no)))  # default to line no

    if not rewards:
        print("No usable samples found.")
        return

    # ---------- old ----------
    adv_old = compute_advantage_old(rewards, prompt_ids, traj_ids)
    max_abs_old = adv_old.abs().max().item()
    print(f"[OLD] max|adv| = {max_abs_old:,.1f}")

    if max_abs_old >= THRESH:
        big_idx = torch.nonzero(adv_old.abs() >= THRESH).view(-1).tolist()
        print(f"Found {len(big_idx)} exploding advantages (|adv| ≥ {THRESH}):")
        for i in big_idx:
            print(f"  sample#{i}: reward={rewards[i]}, prompt={prompt_ids[i]}, "
                  f"traj={traj_ids[i]}, adv={adv_old[i].item():.1f}")
    else:
        print("No exploding values under old implementation.")

    # ---------- fixed ----------
    adv_new = compute_advantage_fixed(rewards, prompt_ids, traj_ids)
    max_abs_new = adv_new.abs().max().item()
    print(f"[FIX] max|adv| = {max_abs_new:.2f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_adv_explosion.py <run_log.jsonl>")
        sys.exit(1)
    main(sys.argv[1])


'''
python log_rollout/check_explode.py /path/to/log_rollout_train/run_name/1.jsonl
'''
