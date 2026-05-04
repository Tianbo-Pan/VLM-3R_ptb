#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export LMMS_EVAL_LAUNCHER="${LMMS_EVAL_LAUNCHER:-accelerate}"

# Full evaluation should use the whole VSIBench test split.
unset LMMS_EVAL_PER_TYPE_LIMIT
unset LMMS_EVAL_SAMPLE_SEED

NUM_PROCESSES="${NUM_PROCESSES:-8}"
PRETRAINED="${PRETRAINED:-Journey9ni/vlm-3r-llava-qwen2-lora}"
MODEL_BASE="${MODEL_BASE:-lmms-lab/LLaVA-NeXT-Video-7B-Qwen2}"
CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_1_5}"
MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-32}"

BRANCH_MODE="${BRANCH_MODE:-pairwise}"
CONTRAST_MODE="${CONTRAST_MODE:-pairwise}"
CONTRAST_ALPHAS="${CONTRAST_ALPHAS:-[0.8]}"
BETA="${BETA:-0.02}"
APPEND_NEWLINE="${APPEND_NEWLINE:-true}"

STAGE1_TOPK="${STAGE1_TOPK:-16}"
STAGE1_SCORING_MODE="${STAGE1_SCORING_MODE:-question_cosine}"
STAGE1_FINE_SCALE="${STAGE1_FINE_SCALE:-1.0}"
SEMANTIC_NEG_RATIO="${SEMANTIC_NEG_RATIO:-0.3}"
SEMANTIC_NEG_CORRUPT_MODE="${SEMANTIC_NEG_CORRUPT_MODE:-frame_mean}"

RUN_SUFFIX="${RUN_SUFFIX:-vlm_3r_gen_vcd_vsibench}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/$(TZ="America/New_York" date "+%Y%m%d")/vsibench_gen_vcd}"

MODEL_ARGS="pretrained=${PRETRAINED},model_base=${MODEL_BASE},conv_template=${CONV_TEMPLATE},max_frames_num=${MAX_FRAMES_NUM},branch_mode=${BRANCH_MODE},contrast_mode=${CONTRAST_MODE},contrast_alphas=${CONTRAST_ALPHAS},beta=${BETA},append_newline=${APPEND_NEWLINE},stage1_topk=${STAGE1_TOPK},stage1_scoring_mode=${STAGE1_SCORING_MODE},stage1_fine_scale=${STAGE1_FINE_SCALE},semantic_neg_ratio=${SEMANTIC_NEG_RATIO},semantic_neg_corrupt_mode=${SEMANTIC_NEG_CORRUPT_MODE}"

echo "Running full VSIBench evaluation with generation-time VCD"
echo "model: vlm_3r_gen_vcd"
echo "model_args: ${MODEL_ARGS}"
echo "output_path: ${OUTPUT_ROOT}"

accelerate launch \
    --num_processes="${NUM_PROCESSES}" \
    -m lmms_eval \
    --model vlm_3r_gen_vcd \
    --model_args "${MODEL_ARGS}" \
    --tasks vsibench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "${RUN_SUFFIX}" \
    --output_path "${OUTPUT_ROOT}"

