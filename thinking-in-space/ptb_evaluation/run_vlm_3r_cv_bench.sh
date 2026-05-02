#!/usr/bin/env bash
set -euo pipefail

# Example usage:
#   bash ptb_evaluation/run_vlm_3r_cv_bench.sh
#
# Override variables inline if needed, e.g.
#   CUDA_VISIBLE_DEVICES=0 MODEL_PATH=... MODEL_BASE=... bash ptb_evaluation/run_vlm_3r_cv_bench.sh

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

MODEL_PATH="${MODEL_PATH:-Journey9ni/vlm-3r-llava-qwen2-lora}"
MODEL_BASE="${MODEL_BASE:-lmms-lab/LLaVA-NeXT-Video-7B-Qwen2}"
OUTPUT_DIR="${OUTPUT_DIR:-ptb_evaluation/outputs/$(TZ="Asia/Shanghai" date "+%Y%m%d")}"
OUTPUT_NAME="${OUTPUT_NAME:-vlm_3r_cv_bench}"
LIMIT="${LIMIT:-}"

CMD=(
  python ptb_evaluation/eval_cv_bench.py
  --model-path "$MODEL_PATH"
  --model-base "$MODEL_BASE"
  --conv-mode qwen_1_5
  --output-dir "$OUTPUT_DIR"
  --output-name "$OUTPUT_NAME"
  --device cuda
)

if [[ -n "$LIMIT" ]]; then
  CMD+=(--limit "$LIMIT")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
