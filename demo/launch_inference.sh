#!/usr/bin/env bash
set -e

MODEL_PATH_ARG="${1:-}"
VIDEO_PATH_ARG="${2:-}"
EXTRA_ARGS=()
if [[ -n "${MODEL_PATH_ARG}" ]]; then
  shift
fi
if [[ -n "${VIDEO_PATH_ARG}" ]]; then
  shift
fi
if [[ "$#" -gt 0 ]]; then
  EXTRA_ARGS=("$@")
fi

MODEL_PATH="${MODEL_PATH:-${MODEL_PATH_ARG:-/path/to/model}}"
VIDEO_PATH="${VIDEO_PATH:-${VIDEO_PATH_ARG:-/path/to/video.mp4}}"
QUESTION="${QUESTION:-Which of the following features/items is not discussed in the video in relation to the tomb?}"
ANSWER="${ANSWER:-C}"
QUESTION_TYPE="${QUESTION_TYPE:-MCQ}"
OPTIONS="${OPTIONS:-A. Inkstone\nB. Niche\nC. Jade\nD. Sacrificial table}"
OUTPUT_JSON="${OUTPUT_JSON:-./inference_output/latest_run.json}"

cd "$(dirname "$0")"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is required."
  echo "Example: bash launch_inference.sh /path/to/model /path/to/video.mp4"
  exit 1
fi

if [[ -z "${VIDEO_PATH}" ]]; then
  echo "VIDEO_PATH is required."
  echo "Example: bash launch_inference.sh /path/to/model /path/to/video.mp4"
  exit 1
fi

echo "Running standalone inference..."
echo "  Model: ${MODEL_PATH}"
echo "  Video: ${VIDEO_PATH}"
echo "  Output: ${OUTPUT_JSON}"
echo ""

python omniagent_inference.py \
  --model_path "${MODEL_PATH}" \
  --video_path "${VIDEO_PATH}" \
  --question "${QUESTION}" \
  --answer "${ANSWER}" \
  --question_type "${QUESTION_TYPE}" \
  --options "${OPTIONS}" \
  --output_json "${OUTPUT_JSON}" \
  "${EXTRA_ARGS[@]}"
