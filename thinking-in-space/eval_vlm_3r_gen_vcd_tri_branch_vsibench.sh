#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export LMMS_EVAL_LAUNCHER="${LMMS_EVAL_LAUNCHER:-accelerate}"

if ! command -v "${LMMS_EVAL_LAUNCHER}" >/dev/null 2>&1; then
    if [[ -x "/root/miniconda3/envs/vsibench/bin/accelerate" ]]; then
        export LMMS_EVAL_LAUNCHER="/root/miniconda3/envs/vsibench/bin/accelerate"
    fi
fi

unset LMMS_EVAL_PER_TYPE_LIMIT
unset LMMS_EVAL_SAMPLE_SEED

NUM_PROCESSES="${NUM_PROCESSES:-8}"
PRETRAINED="${PRETRAINED:-Journey9ni/vlm-3r-llava-qwen2-lora}"
MODEL_BASE="${MODEL_BASE:-lmms-lab/LLaVA-NeXT-Video-7B-Qwen2}"
CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_1_5}"
MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-32}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

BRANCH_MODE="${BRANCH_MODE:-tri}"
CONTRAST_MODE="${CONTRAST_MODE:-tri_rectified}"
CONTRAST_ALPHAS="${CONTRAST_ALPHAS:-1.2:1.2}"
BETA="${BETA:-0.003}"
REFERENCE_MODE="${REFERENCE_MODE:-original}"
APPEND_NEWLINE="${APPEND_NEWLINE:-true}"

PATCH_WARP_RATIO="${PATCH_WARP_RATIO:-0.3}"
PATCH_WARP_SELECTION_MODE="${PATCH_WARP_SELECTION_MODE:-question_cosine_x_norm}"
PATCH_WARP_SELECTION_SCOPE="${PATCH_WARP_SELECTION_SCOPE:-per_frame}"
PATCH_WARP_SHIFT_SIZE="${PATCH_WARP_SHIFT_SIZE:-1}"
PATCH_WARP_MIX_RATIO="${PATCH_WARP_MIX_RATIO:-0.5}"
PATCH_WARP_FUSION_2D_WEIGHT="${PATCH_WARP_FUSION_2D_WEIGHT:-0.4}"
PATCH_WARP_FUSION_3D_WEIGHT="${PATCH_WARP_FUSION_3D_WEIGHT:-1.0}"

AUG_PATCH_TOPK="${AUG_PATCH_TOPK:-16}"
AUG_PATCH_RATIO="${AUG_PATCH_RATIO:-0.03}"
AUG_SELECTION_SCOPE="${AUG_SELECTION_SCOPE:-per_frame}"
AUG_SCORING_MODE="${AUG_SCORING_MODE:-question_cosine_x_norm}"
AUG_INJECTION_MODE="${AUG_INJECTION_MODE:-inplace_boost_coarse_tail}"
AUG_BOOST_FACTOR="${AUG_BOOST_FACTOR:-2}"
AUG_BACKGROUND_DECAY="${AUG_BACKGROUND_DECAY:-0.98}"
AUG_FINE_SCALE="${AUG_FINE_SCALE:-1.0}"
AUG_INCLUDE_COARSE="${AUG_INCLUDE_COARSE:-true}"
AUG_APPEND_NEWLINE="${AUG_APPEND_NEWLINE:-true}"
AUG_FUSION_2D_WEIGHT="${AUG_FUSION_2D_WEIGHT:-1.0}"
AUG_FUSION_3D_WEIGHT="${AUG_FUSION_3D_WEIGHT:-1.0}"

RUN_SUFFIX="${RUN_SUFFIX:-vlm_3r_gen_vcd_tri_branch_vsibench}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/$(TZ="America/New_York" date "+%Y%m%d")/vsibench_gen_vcd_tri_branch}"
COMPARE_TO_BASELINE="${COMPARE_TO_BASELINE:-true}"
BASELINE_JSON="${BASELINE_JSON:-/root/autodl-tmp/projects/VLM-3R/thinking-in-space/logs/20260501/vsibench/0501_1035_vlm_3r_7b_qwen2_lora_vlm_3r_model_args_70e1b2/vsibench.json}"

MODEL_ARGS="pretrained=${PRETRAINED},model_base=${MODEL_BASE},conv_template=${CONV_TEMPLATE},max_frames_num=${MAX_FRAMES_NUM},attn_implementation=${ATTN_IMPLEMENTATION},branch_mode=${BRANCH_MODE},contrast_mode=${CONTRAST_MODE},contrast_alphas=${CONTRAST_ALPHAS},beta=${BETA},reference_mode=${REFERENCE_MODE},append_newline=${APPEND_NEWLINE},patch_warp_ratio=${PATCH_WARP_RATIO},patch_warp_selection_mode=${PATCH_WARP_SELECTION_MODE},patch_warp_selection_scope=${PATCH_WARP_SELECTION_SCOPE},patch_warp_shift_size=${PATCH_WARP_SHIFT_SIZE},patch_warp_mix_ratio=${PATCH_WARP_MIX_RATIO},patch_warp_fusion_2d_weight=${PATCH_WARP_FUSION_2D_WEIGHT},patch_warp_fusion_3d_weight=${PATCH_WARP_FUSION_3D_WEIGHT},aug_patch_topk=${AUG_PATCH_TOPK},aug_patch_ratio=${AUG_PATCH_RATIO},aug_selection_scope=${AUG_SELECTION_SCOPE},aug_scoring_mode=${AUG_SCORING_MODE},aug_injection_mode=${AUG_INJECTION_MODE},aug_boost_factor=${AUG_BOOST_FACTOR},aug_background_decay=${AUG_BACKGROUND_DECAY},aug_fine_scale=${AUG_FINE_SCALE},aug_include_coarse=${AUG_INCLUDE_COARSE},aug_append_newline=${AUG_APPEND_NEWLINE},aug_fusion_2d_weight=${AUG_FUSION_2D_WEIGHT},aug_fusion_3d_weight=${AUG_FUSION_3D_WEIGHT}"

echo "Running full VSIBench evaluation with tri-branch generation-time VCD"
echo "model: vlm_3r_gen_vcd_tri_branch"
echo "model_args: ${MODEL_ARGS}"
echo "output_path: ${OUTPUT_ROOT}"

"${LMMS_EVAL_LAUNCHER}" launch \
    --num_processes="${NUM_PROCESSES}" \
    -m lmms_eval \
    --model vlm_3r_gen_vcd_tri_branch \
    --model_args "${MODEL_ARGS}" \
    --tasks vsibench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "${RUN_SUFFIX}" \
    --output_path "${OUTPUT_ROOT}"

latest_run_dir="$(ls -td "${OUTPUT_ROOT}"/*/ 2>/dev/null | head -n 1 || true)"
current_json=""
if [[ -n "${latest_run_dir}" && -f "${latest_run_dir}/vsibench.json" ]]; then
    current_json="${latest_run_dir}/vsibench.json"
fi

if [[ -n "${current_json}" ]]; then
    echo
    echo "===== Per-question-type sample counts in current run ====="
    python - "$current_json" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

log_path = Path(sys.argv[1])
data = json.loads(log_path.read_text())
logs = data.get("logs", [])
counter = Counter(item.get("doc", {}).get("question_type") for item in logs)

for question_type, count in sorted(counter.items()):
    print(f"{question_type}\t{count}")
print(f"total\t{sum(counter.values())}")
PY
fi

if [[ "${COMPARE_TO_BASELINE}" == "true" ]]; then
    if [[ -n "${current_json}" && -n "${BASELINE_JSON}" && -f "${BASELINE_JSON}" ]]; then
        echo
        echo "Comparing current run against baseline..."
        python tools/compare_vsibench_runs.py \
            --baseline_json "${BASELINE_JSON}" \
            --current_json "${current_json}"
        echo
        echo "Per-question-type change summary..."
        python tools/summarize_vsibench_changes_by_type.py \
            --baseline_json "${BASELINE_JSON}" \
            --current_json "${current_json}"
    else
        echo
        echo "Skipping baseline comparison because current vsibench.json or BASELINE_JSON was not found."
        echo "BASELINE_JSON=${BASELINE_JSON}"
        echo "current_json=${current_json}"
    fi
fi
