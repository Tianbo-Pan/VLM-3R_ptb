#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export LMMS_EVAL_LAUNCHER="${LMMS_EVAL_LAUNCHER:-accelerate}"

NUM_PROCESSES="${NUM_PROCESSES:-2}"
PRETRAINED="${PRETRAINED:-Journey9ni/vlm-3r-llava-qwen2-lora}"
MODEL_BASE="${MODEL_BASE:-lmms-lab/LLaVA-NeXT-Video-7B-Qwen2}"
CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_1_5}"
MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-32}"

STAGE1_SCORING_MODE="${STAGE1_SCORING_MODE:-question_cosine}"
SEMANTIC_NEG_RATIO="${SEMANTIC_NEG_RATIO:-0.3}"
SEMANTIC_NEG_CORRUPT_MODE="${SEMANTIC_NEG_CORRUPT_MODE:-zero}"
ALPHA="${ALPHA:-1.0}"
CANDIDATE_MODE="${CANDIDATE_MODE:-label}"

RUN_SUFFIX="${RUN_SUFFIX:-vlm_3r_semantic_vcd_vsibench}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/$(TZ="America/New_York" date "+%Y%m%d")/vsibench_semantic_vcd}"

MODEL_ARGS="pretrained=${PRETRAINED},model_base=${MODEL_BASE},conv_template=${CONV_TEMPLATE},max_frames_num=${MAX_FRAMES_NUM},stage1_scoring_mode=${STAGE1_SCORING_MODE},semantic_neg_ratio=${SEMANTIC_NEG_RATIO},semantic_neg_corrupt_mode=${SEMANTIC_NEG_CORRUPT_MODE},alpha=${ALPHA},candidate_mode=${CANDIDATE_MODE}"

echo "Running VSIBench with semantic VCD combined branch"
echo "model_args: ${MODEL_ARGS}"
echo "output_path: ${OUTPUT_ROOT}"

accelerate launch \
    --num_processes="${NUM_PROCESSES}" \
    -m lmms_eval \
    --model vlm_3r_semantic_vcd \
    --model_args "${MODEL_ARGS}" \
    --tasks vsibench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "${RUN_SUFFIX}" \
    --output_path "${OUTPUT_ROOT}"
