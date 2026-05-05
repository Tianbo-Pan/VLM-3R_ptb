#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export LMMS_EVAL_LAUNCHER="${LMMS_EVAL_LAUNCHER:-accelerate}"

unset LMMS_EVAL_PER_TYPE_LIMIT
unset LMMS_EVAL_SAMPLE_SEED

NUM_PROCESSES="${NUM_PROCESSES:-8}"
PRETRAINED="${PRETRAINED:-Journey9ni/vlm-3r-llava-qwen2-lora}"
MODEL_BASE="${MODEL_BASE:-lmms-lab/LLaVA-NeXT-Video-7B-Qwen2}"
CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_1_5}"
MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-32}"

BRANCH_MODE="${BRANCH_MODE:-pairwise}"
CONTRAST_MODE="${CONTRAST_MODE:-pairwise}"
CONTRAST_ALPHAS="${CONTRAST_ALPHAS:-[1.2]}"
BETA="${BETA:-0.005}"
APPEND_NEWLINE="${APPEND_NEWLINE:-true}"

PATCH_WARP_RATIO="${PATCH_WARP_RATIO:-0.3}"
PATCH_WARP_SELECTION_MODE="${PATCH_WARP_SELECTION_MODE:-question_cosine}"
# PATCH_WARP_SELECTION_SCOPE="${PATCH_WARP_SELECTION_SCOPE:-per_frame}"
PATCH_WARP_SELECTION_SCOPE="${PATCH_WARP_SELECTION_SCOPE:-global}"
# PATCH_WARP_SELECTION_MODE="${PATCH_WARP_SELECTION_MODE:-fusion_2d3d}"
PATCH_WARP_SHIFT_SIZE="${PATCH_WARP_SHIFT_SIZE:-2}"
PATCH_WARP_MIX_RATIO="${PATCH_WARP_MIX_RATIO:-0.5}"
PATCH_WARP_FUSION_2D_WEIGHT="${PATCH_WARP_FUSION_2D_WEIGHT:-0.4}"
PATCH_WARP_FUSION_3D_WEIGHT="${PATCH_WARP_FUSION_3D_WEIGHT:-1.0}"

RUN_SUFFIX="${RUN_SUFFIX:-vlm_3r_gen_vcd_patch_warp_vsibench}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/$(TZ="America/New_York" date "+%Y%m%d")/vsibench_gen_vcd_patch_warp}"
COMPARE_TO_BASELINE="${COMPARE_TO_BASELINE:-true}"
BASELINE_JSON="${BASELINE_JSON:-/root/autodl-tmp/projects/VLM-3R/logs/20260501/vsibench/0501_1035_vlm_3r_7b_qwen2_lora_vlm_3r_model_args_70e1b2/vsibench.json}"

MODEL_ARGS="pretrained=${PRETRAINED},model_base=${MODEL_BASE},conv_template=${CONV_TEMPLATE},max_frames_num=${MAX_FRAMES_NUM},branch_mode=${BRANCH_MODE},contrast_mode=${CONTRAST_MODE},contrast_alphas=${CONTRAST_ALPHAS},beta=${BETA},append_newline=${APPEND_NEWLINE},patch_warp_ratio=${PATCH_WARP_RATIO},patch_warp_selection_mode=${PATCH_WARP_SELECTION_MODE},patch_warp_selection_scope=${PATCH_WARP_SELECTION_SCOPE},patch_warp_shift_size=${PATCH_WARP_SHIFT_SIZE},patch_warp_mix_ratio=${PATCH_WARP_MIX_RATIO},patch_warp_fusion_2d_weight=${PATCH_WARP_FUSION_2D_WEIGHT},patch_warp_fusion_3d_weight=${PATCH_WARP_FUSION_3D_WEIGHT}"

echo "Running full VSIBench evaluation with patch-warp generation-time VCD"
echo "model: vlm_3r_gen_vcd_patch_warp"
echo "model_args: ${MODEL_ARGS}"
echo "output_path: ${OUTPUT_ROOT}"

accelerate launch \
    --num_processes="${NUM_PROCESSES}" \
    -m lmms_eval \
    --model vlm_3r_gen_vcd_patch_warp \
    --model_args "${MODEL_ARGS}" \
    --tasks vsibench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "${RUN_SUFFIX}" \
    --output_path "${OUTPUT_ROOT}"

if [[ "${COMPARE_TO_BASELINE}" == "true" ]]; then
    latest_run_dir="$(ls -td "${OUTPUT_ROOT}"/*/ 2>/dev/null | head -n 1 || true)"
    if [[ -n "${latest_run_dir}" && -f "${latest_run_dir}/vsibench.json" && -f "${BASELINE_JSON}" ]]; then
        echo
        echo "Comparing current run against baseline..."
        python tools/compare_vsibench_runs.py \
            --baseline_json "${BASELINE_JSON}" \
            --current_json "${latest_run_dir}/vsibench.json"
    else
        echo
        echo "Skipping baseline comparison because current vsibench.json or BASELINE_JSON was not found."
        echo "BASELINE_JSON=${BASELINE_JSON}"
        echo "latest_run_dir=${latest_run_dir}"
    fi
fi
