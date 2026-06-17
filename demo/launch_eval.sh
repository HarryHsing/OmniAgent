#!/usr/bin/env bash
set -eo pipefail

MODEL_PATH="${MODEL_PATH:-}"
DATASET_JSONL="${DATASET_JSONL:-}"
GPU_IDS="${GPU_IDS:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-./eval_output/auto}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
MAX_STEPS="${MAX_STEPS:-32}"          # test-time scaling: try 12, 22, 32, 42, 52
MAX_FRAMES_LEN="${MAX_FRAMES_LEN:-60}"
MAX_AUDIO_LEN="${MAX_AUDIO_LEN:-300}"
MAX_CLIP_LEN="${MAX_CLIP_LEN:-60}"
MODE="${MODE:-OmniAgent}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
USE_TITO="${USE_TITO:-false}"
USE_DYNAMIC_STEP="${USE_DYNAMIC_STEP:-true}"
OMNIAGENT_QUIET_LOGS="${OMNIAGENT_QUIET_LOGS:-true}"
PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export OMNIAGENT_QUIET_LOGS
export PYTHONWARNINGS
export TOKENIZERS_PARALLELISM
export PYTHONUNBUFFERED=1

if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is required."
  echo "Example: MODEL_PATH=checkpoints/OmniAgent-RL-7B DATASET_JSONL=/path/to/dataset.jsonl bash demo/launch_eval.sh"
  exit 1
fi

if [[ -z "${DATASET_JSONL}" ]]; then
  echo "DATASET_JSONL is required."
  echo "Example: MODEL_PATH=checkpoints/OmniAgent-RL-7B DATASET_JSONL=/path/to/dataset.jsonl bash demo/launch_eval.sh"
  exit 1
fi

if [[ ! -f "${DATASET_JSONL}" ]]; then
  echo "DATASET_JSONL not found: ${DATASET_JSONL}"
  exit 1
fi

RESOLVED_OUTPUT_DIR="${OUTPUT_DIR}"
if [[ "$(basename "${OUTPUT_DIR}")" == "auto" ]]; then
  export MODEL_PATH DATASET_JSONL OUTPUT_DIR MAX_SAMPLES MAX_STEPS MAX_FRAMES_LEN MAX_AUDIO_LEN MAX_CLIP_LEN MODE USE_DYNAMIC_STEP USE_TITO
  RESOLVED_OUTPUT_DIR="$(python - <<'PY'
from datetime import datetime
from pathlib import Path
import os

def slug(text: str) -> str:
    cleaned = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            cleaned.append(ch)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_.") or "run"

output_dir = Path(os.environ["OUTPUT_DIR"])
model_path = os.environ["MODEL_PATH"]
dataset_jsonl = os.environ["DATASET_JSONL"]
max_samples = os.environ["MAX_SAMPLES"]
max_steps = os.environ["MAX_STEPS"]
max_frames_len = os.environ["MAX_FRAMES_LEN"]
max_audio_len = os.environ["MAX_AUDIO_LEN"]
max_clip_len = os.environ["MAX_CLIP_LEN"]
mode = os.environ["MODE"]
use_dynamic_step = os.environ["USE_DYNAMIC_STEP"].lower()
use_tito = os.environ["USE_TITO"].lower()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = slug(Path(model_path).name)
dataset_name = slug(Path(dataset_jsonl).stem)
sample_tag = "full" if str(max_samples) == "-1" else f"n{slug(max_samples)}"
cfg_tag = f"s{max_steps}_f{max_frames_len}_a{max_audio_len}_c{max_clip_len}_{slug(mode)}"
flag_tag = f"dyn{1 if use_dynamic_step in ('1','true','t','yes','y') else 0}_tito{1 if use_tito in ('1','true','t','yes','y') else 0}"
print(output_dir.parent / f"{dataset_name}__{model_name}__{sample_tag}__{cfg_tag}__{flag_tag}__{timestamp}")
PY
)"
fi

ARGS=(
  --model_path "${MODEL_PATH}"
  --dataset_jsonl "${DATASET_JSONL}"
  --output_dir "${RESOLVED_OUTPUT_DIR}"
  --gpu_ids "${GPU_IDS}"
  --max_samples "${MAX_SAMPLES}"
  --max_steps "${MAX_STEPS}"
  --max_frames_len "${MAX_FRAMES_LEN}"
  --max_audio_len "${MAX_AUDIO_LEN}"
  --max_clip_len "${MAX_CLIP_LEN}"
  --mode "${MODE}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --top_k "${TOP_K}"
)

if [[ "${USE_TITO,,}" == "true" || "${USE_TITO}" == "1" ]]; then
  ARGS+=(--use_tito)
fi

if [[ "${USE_DYNAMIC_STEP,,}" == "false" || "${USE_DYNAMIC_STEP}" == "0" ]]; then
  ARGS+=(--no_dynamic_step)
else
  ARGS+=(--dynamic_step)
fi

mkdir -p "${RESOLVED_OUTPUT_DIR}"
RAW_LOG="${RESOLVED_OUTPUT_DIR}/run.log"
PROGRESS_LOG="${RESOLVED_OUTPUT_DIR}/progress.log"

{
  echo "Running evaluation..."
  echo "  Model: ${MODEL_PATH}"
  echo "  Dataset: ${DATASET_JSONL}"
  echo "  GPUs: ${GPU_IDS}"
  echo "  Output: ${RESOLVED_OUTPUT_DIR}"
  echo ""

  python -u demo/omniagent_eval.py "${ARGS[@]}" "$@"

  echo ""
  echo "Token Summary..."
  python - <<'PY' "${RESOLVED_OUTPUT_DIR}"
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
summary_path = run_dir / "summary.json"
results_path = run_dir / "results.jsonl"
if not summary_path.exists():
    print("  token_status: summary.json missing")
    raise SystemExit(0)

summary = json.loads(summary_path.read_text(encoding="utf-8"))
fields = [
    "avg_prompt_total_tokens",
    "avg_prompt_nontext_tokens",
    "avg_full_sequence_tokens",
]
has_summary_fields = all(k in summary for k in fields)
has_result_fields = False
if results_path.exists():
    with results_path.open("r", encoding="utf-8") as f:
        first = next((line.strip() for line in f if line.strip()), "")
    if first:
        obj = json.loads(first)
        has_result_fields = any(k in obj for k in ["total_prompt_total_tokens", "prompt_total_tokens"])

print(f"  token_fields_present: {'yes' if (has_summary_fields or has_result_fields) else 'no'}")
for key in fields:
    value = summary.get(key, None)
    if value is None:
        print(f"  {key}: <missing>")
    else:
        print(f"  {key}: {float(value):.2f}")
PY
} 2>&1 | tee "${RAW_LOG}" | grep --line-buffered -E \
'^(Running evaluation\.\.\.|  Model:|  Dataset:|  GPUs:|  Output:|$|\[[0-9]+/[0-9]+\] sample=|\[progress\] completed=|Summary$|Artifacts$|  results_jsonl:|  summary_json:|  summary_csv:|  sample_csv:|  breakdowns_json:|Generating latency report\.\.\.|Saved latency report to |  Report:|Token Summary\.\.\.|  token_fields_present:|  avg_prompt_total_tokens:|  avg_prompt_nontext_tokens:|  avg_full_sequence_tokens:)' \
| tee "${PROGRESS_LOG}"

echo ""
echo "Logs"
echo "  Raw: ${RAW_LOG}"
echo "  Progress: ${PROGRESS_LOG}"
