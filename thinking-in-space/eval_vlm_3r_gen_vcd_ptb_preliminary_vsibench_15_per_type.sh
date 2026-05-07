#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LMMS_EVAL_LAUNCHER="${LMMS_EVAL_LAUNCHER:-accelerate}"
export LMMS_EVAL_PER_TYPE_LIMIT="${LMMS_EVAL_PER_TYPE_LIMIT:-15}"
export LMMS_EVAL_SAMPLE_SEED="${LMMS_EVAL_SAMPLE_SEED:-42}"

NUM_PROCESSES="${NUM_PROCESSES:-1}"
PRETRAINED="${PRETRAINED:-Journey9ni/vlm-3r-llava-qwen2-lora}"
MODEL_BASE="${MODEL_BASE:-lmms-lab/LLaVA-NeXT-Video-7B-Qwen2}"
CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_1_5}"
MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-32}"

BRANCH_MODE="${BRANCH_MODE:-pairwise}"
CONTRAST_MODE="${CONTRAST_MODE:-pairwise}"
CONTRAST_ALPHAS="${CONTRAST_ALPHAS:-[2]}"
BETA="${BETA:-0.005}"
APPEND_NEWLINE="${APPEND_NEWLINE:-true}"

PTB_PRELIMINARY_RATIO="${PTB_PRELIMINARY_RATIO:-0.3}"
PTB_PRELIMINARY_QUERY_MODE="${PTB_PRELIMINARY_QUERY_MODE:-question_cosine}"
PTB_PRELIMINARY_SELECTION_SCOPE="${PTB_PRELIMINARY_SELECTION_SCOPE:-per_frame}"
PTB_PRELIMINARY_SHIFT_SIZE="${PTB_PRELIMINARY_SHIFT_SIZE:-1}"
PTB_PRELIMINARY_MIX_RATIO="${PTB_PRELIMINARY_MIX_RATIO:-0.2}"

RUN_SUFFIX="${RUN_SUFFIX:-vlm_3r_gen_vcd_ptb_preliminary_${LMMS_EVAL_PER_TYPE_LIMIT}_per_type}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/$(TZ="America/New_York" date "+%Y%m%d")/vsibench_gen_vcd_ptb_preliminary_${LMMS_EVAL_PER_TYPE_LIMIT}_per_type}"
COMPARE_TO_BASELINE="${COMPARE_TO_BASELINE:-true}"
BASELINE_JSON="${BASELINE_JSON:-/root/autodl-tmp/projects/VLM-3R/thinking-in-space/logs/20260505/vsibench_15_per_type/0505_0504_vlm_3r_7b_qwen2_lora_15_per_type_vlm_3r_model_args_70e1b2/vsibench.json}"


MODEL_ARGS="pretrained=${PRETRAINED},model_base=${MODEL_BASE},conv_template=${CONV_TEMPLATE},max_frames_num=${MAX_FRAMES_NUM},branch_mode=${BRANCH_MODE},contrast_mode=${CONTRAST_MODE},contrast_alphas=${CONTRAST_ALPHAS},beta=${BETA},append_newline=${APPEND_NEWLINE},ptb_preliminary_ratio=${PTB_PRELIMINARY_RATIO},ptb_preliminary_query_mode=${PTB_PRELIMINARY_QUERY_MODE},ptb_preliminary_selection_scope=${PTB_PRELIMINARY_SELECTION_SCOPE},ptb_preliminary_shift_size=${PTB_PRELIMINARY_SHIFT_SIZE},ptb_preliminary_mix_ratio=${PTB_PRELIMINARY_MIX_RATIO}"

echo "Running preliminary VSIBench per-type evaluation for 2D-query/3D-perturb VCD"
echo "model: vlm_3r_gen_vcd_ptb_preliminary"
echo "model_args: ${MODEL_ARGS}"
echo "per_type_limit: ${LMMS_EVAL_PER_TYPE_LIMIT}"
echo "sample_seed: ${LMMS_EVAL_SAMPLE_SEED}"
echo "output_path: ${OUTPUT_ROOT}"

accelerate launch \
    --num_processes="${NUM_PROCESSES}" \
    -m lmms_eval \
    --model vlm_3r_gen_vcd_ptb_preliminary \
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
