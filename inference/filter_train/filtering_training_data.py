'''
1. win
2. total reward > xx.
2. oss path
3. steps >= 2
4. good logic (by qwen-max reward)
'''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trajectory_filter.py   –   v1.2

• First apply free filtering (win / reward / step / optional OSS);
• Only call Qwen-plus for "good logic" scoring when free filtering passes;
• OSS public URL -> oss://bucket/path;
• Display progress bar (tqdm).

Requirements:
    pip install requests python-dotenv tqdm
    Set DASHSCOPE_API_KEY in environment variables
"""

import os, sys, json, re, time, ast, requests
from pathlib import Path
from typing import Dict, List, Any
from urllib.parse import urlparse
from tqdm import tqdm
from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# DashScope (Qwen) utilities
load_dotenv()
DASH_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
API_KEY  = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OSS_ACCESS_KEY")
if not API_KEY:
    raise RuntimeError("Please set DASHSCOPE_API_KEY in environment variables")

def gpt_api(prompt: str,
            model_name: str = "qwen-plus",
            stream: bool = False,
            max_try: int = 5,
            timeout: int = 20) -> str | None:
    """DashScope call with retry; returns None on total failure."""
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    payload = {"model": model_name, "stream": stream, "messages": msgs}
    headers = {
        "Content-Type": "application/json",
        "Authorization": API_KEY,
        "X-DashScope-DataInspection": '{"input":"disable","output":"disable"}',
    }

    for att in range(1, max_try + 1):
        try:
            rsp = requests.post(DASH_URL,
                                headers=headers,
                                data=json.dumps(payload),
                                timeout=timeout,
                                stream=False)
            if rsp.status_code != 200:
                raise RuntimeError(f"HTTP {rsp.status_code}: {rsp.text}")
            return rsp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[WARN] gpt_api retry {att}/{max_try}: {e}", file=sys.stderr)
            time.sleep(1 + 0.5 * att)

    print("[ERROR] gpt_api failed after max retries", file=sys.stderr)
    return None

# --------------------------------------------------------------------------- #
# config
MIN_REWARD   = 0.5
MIN_STEPS    = 2
LOGIC_THR    = 0.5
REQUIRE_WIN  = True
REQUIRE_OSS  = False
CHECK_LOGIC  = False

Traj     = List[dict]
TrajDict = Dict[str, Traj]

# --------------------------------------------------------------------------- #
# OSS URL normalisation
OSS_RE = re.compile(r"^(?P<b>[a-zA-Z0-9\-]+)\.oss-[^.]+\.aliyuncs\.com$", re.I)

def to_oss(url: str) -> str:
    if not isinstance(url, str):
        return url
    try:
        p = urlparse(url)
    except Exception:
        return url
    m = OSS_RE.match(p.netloc)
    if not m:
        return url
    bucket = m.group("b")
    return f"oss://{bucket}/{p.path.lstrip('/')}"

def recurse_clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: recurse_clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [recurse_clean(v) for v in o]
    if isinstance(o, str):
        return to_oss(o)
    return o

def normalize_oss(trajs: TrajDict) -> None:
    for rs in trajs.values():
        for i, r in enumerate(rs):
            rs[i] = recurse_clean(r)

# --------------------------------------------------------------------------- #
# cheap filter
def cheap_pass(traj: Traj) -> bool:
    if REQUIRE_WIN and not any(r.get("won") or r.get("win") or r.get("extra_info", {}).get("won") or r.get("extra_info", {}).get("win") for r in traj):
        return False
    if max(r.get("episode_reward", 0) for r in traj) <= MIN_REWARD:
        return False
    if max(r.get("step", 0) for r in traj) < MIN_STEPS:
        return False
    if REQUIRE_OSS:
        if not any(OSS_RE.search(r.get("extra_info", {}).get("video", "") or "") for r in traj):
            return False
    return True

# --------------------------------------------------------------------------- #
# expensive logic scoring
def logic_score(traj: Traj, retry_times: int = 5) -> float:
    final = max(traj, key=lambda r: r.get("step", 0))

    won_flag = bool(final.get("won") or final.get("extra_info", {}).get("won"))
    user_in  = json.dumps(final.get("raw_input", ""), ensure_ascii=False)
    ans    = final.get("output", "")
    extra    = json.dumps(final.get("extra_info", {}), ensure_ascii=False)

    prompt = f"""
You are a senior evaluator of reasoning quality.

Evaluate ONLY the reasoning, not factual accuracy (won tells accuracy).

Return a SINGLE integer 0-5.

5 – Outstanding reasoning
    • Lists or paraphrases all key evidence from the input.
    • Walks through each inference step explicitly and sequentially.
    • No hidden leaps or unstated assumptions.
    • Actively rules out alternative answers / counter-arguments.
    • Conclusion follows inevitably from the evidence; presentation is clear.

4 – Strong, mostly complete reasoning
    • Main argument is clear and evidence-based.
    • At most one minor gap or implicit premise.
    • Most key evidence is cited or explained.
    • Conclusion is well supported.

3 – Moderate reasoning
    • Contains several reasoning steps but omits some important links or evidence.
    • Noticeable gaps or assumptions, yet the general line of thought is visible.
    • Support for the conclusion is partial but not purely guessed.

2 – Weak reasoning
    • Scattered hints of logic; little coherent chain of thought.
    • Large leaps or heavy reliance on unstated assumptions.
    • Evidence is sparse or loosely connected to the conclusion.

1 – Minimal reasoning
    • Virtually no logical explanation; answer appears largely guessed.
    • Evidence or justification is superficial or irrelevant.

0 – No reasoning / invalid
    • No reasoning provided, or reasoning is nonsensical / self-contradictory.
    • Or the final answer is demonstrably wrong (won = False).

won: {won_flag}

User's last input:
{user_in}

Extra info (question / correct answer / options):
{extra}

Assistant FINAL ANSWER: {ans}

Score (0-5):
""".strip()

    for t in range(retry_times):
        resp = gpt_api(prompt)
        if resp is None:
            continue
        try:
            val = float(ast.literal_eval(resp))
            if 0 <= val <= 5:
                # print(f"[INFO] prompt {prompt}")
                print(f"[INFO] logic_score {val:.2f}")
                return val / 5.0
        except Exception:
            pass
        print(f"[WARN] logic_score parse retry {t+1}/{retry_times}", file=sys.stderr)
        time.sleep(1)
    return 0.0

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def filter_trajs(trajs: TrajDict) -> TrajDict:
    """
    First apply local filtering; then compute logic_score for those that pass.
    Returns the kept TrajDict (each record has logic_score attached and is sorted by step).
    """
    kept: TrajDict = {}
    iterator = tqdm(trajs.items(), desc="Filtering trajectories", unit="traj")

    for tid, recs in iterator:
        if not cheap_pass(recs):
            continue

        logic_val = None
        if CHECK_LOGIC:
            logic_val = logic_score(recs)
            iterator.set_postfix({"logic": f"{logic_val:.2f}"})
            if logic_val < LOGIC_THR:
                continue

        # Write logic_score to all step records of this trajectory
        for rec in recs:
            rec["logic_score"] = logic_val

        # Sort by step to ensure consistent output order
        recs_sorted = sorted(recs, key=lambda r: r.get("step", 0))
        kept[tid] = recs_sorted

    return kept

# --------------------------------------------------------------------------- #
def load_jsonl(p: Path) -> TrajDict:
    trajs: TrajDict = {}
    with p.open(encoding="utf-8") as fh:
        for i, ln in enumerate(fh, 1):
            if not ln.strip():
                continue
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                print(f"[WARN] bad json at line {i}", file=sys.stderr)
                continue
            tid = rec.get("traj_id")
            if tid:
                trajs.setdefault(tid, []).append(rec)
    return trajs

def save_jsonl(trajs: TrajDict, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for recs in trajs.values():
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    input_folder = os.getenv("INPUT_RESULTS_DIR", "/path/to/inference/results")
    output_folder = os.getenv("OUTPUT_FILTER_DIR", "/path/to/inference/filter_train")
    src_file = os.getenv("SRC_FILE", "train_steps.jsonl")

    inp  = Path(input_folder) / src_file
    stem, ext = os.path.splitext(src_file)
    out  = Path(output_folder) / f"{stem}.filtered{ext}"

    # ---------------------------------------------------------------------- #
    trajs = load_jsonl(inp)
    print(f"[INFO] Loaded trajectories: {len(trajs)}")

    kept = filter_trajs(trajs)
    print(f"[INFO] Kept trajectories : {len(kept)}")

    clean_oss = True
    if clean_oss:
        normalize_oss(kept)

    save_jsonl(kept, out)
    print(f"[INFO] Saved filtered file to {out}")
