#!/usr/bin/env bash
set -e

MODEL_PATH_ARG=""
VIDEO_PATH_ARG=""
EXTRA_ARGS=()

POSITIONAL_ARGS=()
while [[ "$#" -gt 0 && "$1" != -* && "${#POSITIONAL_ARGS[@]}" -lt 2 ]]; do
  POSITIONAL_ARGS+=("$1")
  shift
done
if [[ "${#POSITIONAL_ARGS[@]}" -eq 2 ]]; then
  MODEL_PATH_ARG="${POSITIONAL_ARGS[0]}"
  VIDEO_PATH_ARG="${POSITIONAL_ARGS[1]}"
elif [[ "${#POSITIONAL_ARGS[@]}" -eq 1 ]]; then
  if [[ -n "${MODEL_PATH:-}" && -z "${VIDEO_PATH:-}" ]]; then
    VIDEO_PATH_ARG="${POSITIONAL_ARGS[0]}"
  else
    MODEL_PATH_ARG="${POSITIONAL_ARGS[0]}"
  fi
fi
if [[ "$#" -gt 0 ]]; then
  EXTRA_ARGS=("$@")
fi

MODEL_PATH="${MODEL_PATH:-${MODEL_PATH_ARG:-}}"
VIDEO_PATH="${VIDEO_PATH:-${VIDEO_PATH_ARG:-}}"
QUESTION="${QUESTION:-Which of the following features/items is not discussed in the video in relation to the tomb?}"
ANSWER="${ANSWER:-C}"
QUESTION_TYPE="${QUESTION_TYPE:-MCQ}"
OPTIONS="${OPTIONS:-A. Inkstone\nB. Niche\nC. Jade\nD. Sacrificial table}"
OUTPUT_JSON="${OUTPUT_JSON:-./inference_output/latest_run.json}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is required."
  echo "Example: bash demo/launch_inference.sh /path/to/model /path/to/video.mp4"
  exit 1
fi

if [[ -z "${VIDEO_PATH}" ]]; then
  echo "VIDEO_PATH is required."
  echo "Example: bash demo/launch_inference.sh /path/to/model /path/to/video.mp4"
  exit 1
fi

if [[ ! -f "${VIDEO_PATH}" ]]; then
  echo "VIDEO_PATH not found: ${VIDEO_PATH}"
  exit 1
fi

echo "Running standalone inference..."
echo "  Model: ${MODEL_PATH}"
echo "  Video: ${VIDEO_PATH}"
echo "  Output: ${OUTPUT_JSON}"
echo ""

python demo/omniagent_inference.py \
  --model_path "${MODEL_PATH}" \
  --video_path "${VIDEO_PATH}" \
  --question "${QUESTION}" \
  --answer "${ANSWER}" \
  --question_type "${QUESTION_TYPE}" \
  --options "${OPTIONS}" \
  --output_json "${OUTPUT_JSON}" \
  "${EXTRA_ARGS[@]}"
