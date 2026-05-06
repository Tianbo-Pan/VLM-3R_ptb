#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}/thinking-in-space"

# export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export LMMS_EVAL_LAUNCHER="${LMMS_EVAL_LAUNCHER:-accelerate}"

NUM_PROCESSES="${NUM_PROCESSES:-4}"
PRETRAINED="${PRETRAINED:-Journey9ni/vlm-3r-llava-qwen2-lora-vstibench}"
MODEL_BASE="${MODEL_BASE:-lmms-lab/LLaVA-NeXT-Video-7B-Qwen2}"
CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_1_5}"
MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-32}"

FINE_TOPK="${FINE_TOPK:-16}"
FINE_RATIO="${FINE_RATIO:-none}"
SELECTION_SCOPE="${SELECTION_SCOPE:-per_frame}"
SCORING_MODE="${SCORING_MODE:-fusion_2d3d}"
FINE_SCALE="${FINE_SCALE:-1.0}"
FUSION_2D_WEIGHT="${FUSION_2D_WEIGHT:-1.0}"
FUSION_3D_WEIGHT="${FUSION_3D_WEIGHT:-1.0}"
INCLUDE_COARSE="${INCLUDE_COARSE:-True}"
APPEND_NEWLINE="${APPEND_NEWLINE:-True}"

RUN_SUFFIX="${RUN_SUFFIX:-vlm_3r_fusion_guided_patch_selection_vstibench}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/$(TZ="America/New_York" date "+%Y%m%d")/vstibench_fusion_guided_patch_selection}"

MODEL_ARGS="pretrained=${PRETRAINED},model_base=${MODEL_BASE},conv_template=${CONV_TEMPLATE},max_frames_num=${MAX_FRAMES_NUM},fine_topk=${FINE_TOPK},fine_ratio=${FINE_RATIO},selection_scope=${SELECTION_SCOPE},scoring_mode=${SCORING_MODE},fine_scale=${FINE_SCALE},fusion_2d_weight=${FUSION_2D_WEIGHT},fusion_3d_weight=${FUSION_3D_WEIGHT},include_coarse=${INCLUDE_COARSE},append_newline=${APPEND_NEWLINE}"

echo "Running VSTIBench with VLM-3R fusion-guided patch-selection inference"
echo "model_args: ${MODEL_ARGS}"
echo "output_path: ${OUTPUT_ROOT}"

accelerate launch \
    --num_processes="${NUM_PROCESSES}" \
    -m lmms_eval \
    --model vlm_3r_patch_selection \
    --model_args "${MODEL_ARGS}" \
    --tasks vstibench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "${RUN_SUFFIX}" \
    --output_path "${OUTPUT_ROOT}"
