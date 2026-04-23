#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="1,3"
export LMMS_EVAL_LAUNCHER="${LMMS_EVAL_LAUNCHER:-accelerate}"

NUM_PROCESSES="2"
PRETRAINED="${PRETRAINED:-Journey9ni/vlm-3r-llava-qwen2-lora}"
MODEL_BASE="${MODEL_BASE:-lmms-lab/LLaVA-NeXT-Video-7B-Qwen2}"
CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_1_5}"
MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-32}"

CD_GUIDANCE_SCALE="${CD_GUIDANCE_SCALE:-1.0}"
CD_FUSION_WEAK_RATIO="${CD_FUSION_WEAK_RATIO:-0.35}"
CD_FUSION_DROP_RATE="${CD_FUSION_DROP_RATE:-0.05}"
CD_FUSION_NOISE_STD="${CD_FUSION_NOISE_STD:-0.005}"

RUN_SUFFIX="${RUN_SUFFIX:-vlm_3r_7b_qwen2_lora_fusion_vcd}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/$(TZ="America/New_York" date "+%Y%m%d")/vsibench_fusion_vcd}"

MODEL_ARGS="pretrained=${PRETRAINED},model_base=${MODEL_BASE},conv_template=${CONV_TEMPLATE},max_frames_num=${MAX_FRAMES_NUM},feature_cd_mode=fusion,cd_guidance_scale=${CD_GUIDANCE_SCALE},cd_fusion_weak_ratio=${CD_FUSION_WEAK_RATIO},cd_fusion_drop_rate=${CD_FUSION_DROP_RATE},cd_fusion_noise_std=${CD_FUSION_NOISE_STD}"

echo "Running VSIBench with fusion feature-level VCD"
echo "model_args: ${MODEL_ARGS}"
echo "output_path: ${OUTPUT_ROOT}"

accelerate launch \
    --num_processes=1 \
    -m lmms_eval \
    --model vlm_3r \
    --model_args "${MODEL_ARGS}" \
    --tasks vsibench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "${RUN_SUFFIX}" \
    --output_path "${OUTPUT_ROOT}"
