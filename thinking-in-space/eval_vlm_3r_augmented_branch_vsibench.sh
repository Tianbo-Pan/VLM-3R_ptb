#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export LMMS_EVAL_LAUNCHER="${LMMS_EVAL_LAUNCHER:-accelerate}"

NUM_PROCESSES="${NUM_PROCESSES:-4}"
PRETRAINED="${PRETRAINED:-Journey9ni/vlm-3r-llava-qwen2-lora}"
MODEL_BASE="${MODEL_BASE:-lmms-lab/LLaVA-NeXT-Video-7B-Qwen2}"
CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_1_5}"
MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-32}"

PATCH_TOPK="${PATCH_TOPK:-16}"
PATCH_RATIO="${PATCH_RATIO:-0.01}"
SELECTION_SCOPE="${SELECTION_SCOPE:-global}"
SCORING_MODE="${SCORING_MODE:-question_cosine}"
INJECTION_MODE="${INJECTION_MODE:-inplace_boost_coarse}"
BOOST_FACTOR="${BOOST_FACTOR:-2.0}"
BACKGROUND_DECAY="${BACKGROUND_DECAY:-0.98}"
FINE_SCALE="${FINE_SCALE:-1.0}"
INCLUDE_COARSE="${INCLUDE_COARSE:-True}"
APPEND_NEWLINE="${APPEND_NEWLINE:-True}"
FUSION_2D_WEIGHT="${FUSION_2D_WEIGHT:-1.0}"
FUSION_3D_WEIGHT="${FUSION_3D_WEIGHT:-1.0}"

RUN_SUFFIX="${RUN_SUFFIX:-vlm_3r_augmented_branch_vsibench}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/$(TZ="America/New_York" date "+%Y%m%d")/vsibench_augmented_branch}"
COMPARE_TO_BASELINE="${COMPARE_TO_BASELINE:-true}"
BASELINE_JSON="/local_home/pantianbo/projects/vision_reasoning/VLM-3R/thinking-in-space/logs/20260417/vsibench/0417_0322_vlm_3r_7b_qwen2_lora_vlm_3r_model_args_70e1b2/vsibench.json"


MODEL_ARGS="pretrained=${PRETRAINED},model_base=${MODEL_BASE},conv_template=${CONV_TEMPLATE},max_frames_num=${MAX_FRAMES_NUM},patch_topk=${PATCH_TOPK},patch_ratio=${PATCH_RATIO},selection_scope=${SELECTION_SCOPE},scoring_mode=${SCORING_MODE},injection_mode=${INJECTION_MODE},boost_factor=${BOOST_FACTOR},background_decay=${BACKGROUND_DECAY},fine_scale=${FINE_SCALE},include_coarse=${INCLUDE_COARSE},append_newline=${APPEND_NEWLINE},fusion_2d_weight=${FUSION_2D_WEIGHT},fusion_3d_weight=${FUSION_3D_WEIGHT}"

echo "Running VSIBench evaluation with augmentation branch"
echo "model: vlm_3r_augmented_branch"
echo "model_args: ${MODEL_ARGS}"
echo "output_path: ${OUTPUT_ROOT}"
if [[ -n "${LMMS_EVAL_PER_TYPE_LIMIT:-}" ]]; then
    echo "LMMS_EVAL_PER_TYPE_LIMIT: ${LMMS_EVAL_PER_TYPE_LIMIT}"
    echo "LMMS_EVAL_SAMPLE_SEED: ${LMMS_EVAL_SAMPLE_SEED:-<unset>}"
else
    echo "LMMS_EVAL_PER_TYPE_LIMIT: <full split>"
fi

accelerate launch \
    --num_processes="${NUM_PROCESSES}" \
    -m lmms_eval \
    --model vlm_3r_augmented_branch \
    --model_args "${MODEL_ARGS}" \
    --tasks vsibench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "${RUN_SUFFIX}" \
    --output_path "${OUTPUT_ROOT}"

latest_run_dir="$(ls -td "${OUTPUT_ROOT}"/*/ 2>/dev/null | head -n 1 || true)"
current_json="$(find "${latest_run_dir}" -maxdepth 1 -type f -name 'vsibench*.json' | sort | tail -n 1 || true)"

if [[ -z "${current_json}" || ! -f "${current_json}" ]]; then
    echo "Failed to locate current VSIBench json under ${latest_run_dir}"
    exit 1
fi

echo
echo "Current VSIBench json: ${current_json}"

if [[ "${COMPARE_TO_BASELINE}" == "true" ]]; then
    if [[ -n "${BASELINE_JSON}" && -f "${BASELINE_JSON}" ]]; then
        echo
        echo "Comparing current run against baseline..."
        python tools/compare_vsibench_runs.py \
            --baseline_json "${BASELINE_JSON}" \
            --current_json "${current_json}"
    else
        echo
        echo "Skipping baseline comparison because BASELINE_JSON is empty or missing."
        echo "BASELINE_JSON=${BASELINE_JSON:-<unset>}"
    fi
fi
