#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source /etc/network_turbo >/dev/null 2>&1 || true

if [[ -f "/root/miniconda3/etc/profile.d/conda.sh" ]]; then
    # Optional convenience: if the caller has not activated an env yet, try the common one.
    # shellcheck disable=SC1091
    source /root/miniconda3/etc/profile.d/conda.sh
    if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
        conda activate vsibench >/dev/null 2>&1 || true
    fi
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NUM_PROCESSES="${NUM_PROCESSES:-2}"
export LMMS_EVAL_PER_TYPE_LIMIT="${LMMS_EVAL_PER_TYPE_LIMIT:-15}"
export LMMS_EVAL_SAMPLE_SEED="${LMMS_EVAL_SAMPLE_SEED:-42}"
export COMPARE_TO_BASELINE="${COMPARE_TO_BASELINE:-false}"

OUTPUT_BASE="${OUTPUT_BASE:-logs/$(TZ="America/New_York" date "+%Y%m%d")/vsibench_gen_vcd_tri_branch_aggressive_${LMMS_EVAL_PER_TYPE_LIMIT}_per_type}"
SUMMARY_TSV="${SUMMARY_TSV:-${OUTPUT_BASE}/summary.tsv}"
BASELINE_JSON="${BASELINE_JSON:-${SCRIPT_DIR}/logs/20260505/vsibench_15_per_type/0505_0504_vlm_3r_7b_qwen2_lora_15_per_type_vlm_3r_model_args_70e1b2/vsibench.json}"
PATCH_BEST_JSON="${PATCH_BEST_JSON:-${SCRIPT_DIR}/logs/20260505/vsibench_gen_vcd_patch_warp_15_per_type/0505_2018_vlm_3r_gen_vcd_patch_warp_15_per_type_vlm_3r_gen_vcd_patch_warp_model_args_2e2026/vsibench.json}"

mkdir -p "${OUTPUT_BASE}"
printf "config\taccuracy\tavg_metric\tjson_path\n" > "${SUMMARY_TSV}"

summarize_json() {
    local json_path="$1"
    python - "$json_path" <<'PY'
import json
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
obj = json.loads(p.read_text())
logs = obj["logs"]
acc = sum(float(item["doc"].get("accuracy", 0)) for item in logs) / max(len(logs), 1)
vals = []
for item in logs:
    vs = item.get("vsibench_score", {})
    if "accuracy" in vs:
        vals.append(float(vs["accuracy"]))
    else:
        mk = next((k for k in vs if "MRA" in k), None)
        vals.append(float(vs[mk]) if mk else 0.0)
print(f"{acc:.4f}\t{sum(vals) / max(len(vals), 1):.4f}")
PY
}

compare_against() {
    local baseline_json="$1"
    local current_json="$2"
    if [[ -f "${baseline_json}" ]]; then
        python "${SCRIPT_DIR}/tools/compare_vsibench_runs.py" \
            --baseline_json "${baseline_json}" \
            --current_json "${current_json}" || true
        python "${SCRIPT_DIR}/tools/summarize_vsibench_changes_by_type.py" \
            --baseline_json "${baseline_json}" \
            --current_json "${current_json}" || true
    fi
}

run_cfg() {
    local name="$1"
    shift
    echo
    echo "========== Running ${name} =========="
    env \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
        NUM_PROCESSES="${NUM_PROCESSES}" \
        LMMS_EVAL_PER_TYPE_LIMIT="${LMMS_EVAL_PER_TYPE_LIMIT}" \
        LMMS_EVAL_SAMPLE_SEED="${LMMS_EVAL_SAMPLE_SEED}" \
        COMPARE_TO_BASELINE="${COMPARE_TO_BASELINE}" \
        OUTPUT_ROOT="${OUTPUT_BASE}/${name}" \
        RUN_SUFFIX="${name}" \
        "$@" \
        "${SCRIPT_DIR}/eval_vlm_3r_gen_vcd_tri_branch_vsibench_15_per_type.sh"

    local latest_run_dir
    latest_run_dir="$(ls -td "${SCRIPT_DIR}/${OUTPUT_BASE}/${name}"/*/ 2>/dev/null | head -n 1 || true)"
    if [[ -z "${latest_run_dir}" || ! -f "${latest_run_dir}/vsibench.json" ]]; then
        echo "No vsibench.json found for ${name}" >&2
        return 1
    fi

    local json_path="${latest_run_dir}/vsibench.json"
    local metrics
    metrics="$(summarize_json "${json_path}")"
    local accuracy avg_metric
    accuracy="$(echo "${metrics}" | cut -f1)"
    avg_metric="$(echo "${metrics}" | cut -f2)"
    printf "%s\t%s\t%s\t%s\n" "${name}" "${accuracy}" "${avg_metric}" "${json_path}" | tee -a "${SUMMARY_TSV}"

    echo
    echo "----- vs baseline -----"
    compare_against "${BASELINE_JSON}" "${json_path}"
    echo
    echo "----- vs patch-warp best -----"
    compare_against "${PATCH_BEST_JSON}" "${json_path}"
}

# Best candidates found so far: xnorm-guided degraded/augmented selection + tail augmentation.
run_cfg tri_patchlike_xnorm_tail \
    CONTRAST_MODE=tri_rectified \
    CONTRAST_ALPHAS=1.0:1.2 \
    BETA=0.003 \
    REFERENCE_MODE=original \
    PATCH_WARP_RATIO=0.30 \
    PATCH_WARP_MIX_RATIO=0.50 \
    PATCH_WARP_SELECTION_SCOPE=per_frame \
    PATCH_WARP_SELECTION_MODE=question_cosine_x_norm \
    AUG_PATCH_RATIO=0.03 \
    AUG_SELECTION_SCOPE=per_frame \
    AUG_SCORING_MODE=question_cosine_x_norm \
    AUG_INJECTION_MODE=inplace_boost_coarse_tail \
    AUG_BOOST_FACTOR=2.0

run_cfg tri_patchlike_xnorm_tail_balanced \
    CONTRAST_MODE=tri_rectified \
    CONTRAST_ALPHAS=1.2:1.2 \
    BETA=0.003 \
    REFERENCE_MODE=original \
    PATCH_WARP_RATIO=0.30 \
    PATCH_WARP_MIX_RATIO=0.50 \
    PATCH_WARP_SELECTION_SCOPE=per_frame \
    PATCH_WARP_SELECTION_MODE=question_cosine_x_norm \
    AUG_PATCH_RATIO=0.03 \
    AUG_SELECTION_SCOPE=per_frame \
    AUG_SCORING_MODE=question_cosine_x_norm \
    AUG_INJECTION_MODE=inplace_boost_coarse_tail \
    AUG_BOOST_FACTOR=2.0

run_cfg tri_patchlike_xnorm_tail_strongaug \
    CONTRAST_MODE=tri_rectified \
    CONTRAST_ALPHAS=1.2:1.2 \
    BETA=0.003 \
    REFERENCE_MODE=original \
    PATCH_WARP_RATIO=0.30 \
    PATCH_WARP_MIX_RATIO=0.50 \
    PATCH_WARP_SELECTION_SCOPE=per_frame \
    PATCH_WARP_SELECTION_MODE=question_cosine_x_norm \
    AUG_PATCH_RATIO=0.05 \
    AUG_SELECTION_SCOPE=per_frame \
    AUG_SCORING_MODE=question_cosine_x_norm \
    AUG_INJECTION_MODE=inplace_boost_coarse_tail \
    AUG_BOOST_FACTOR=2.5

run_cfg tri_patchlike_xnorm_tail_maxref \
    CONTRAST_MODE=tri_rectified \
    CONTRAST_ALPHAS=1.2:1.2 \
    BETA=0.003 \
    REFERENCE_MODE=max_original_augmented \
    PATCH_WARP_RATIO=0.30 \
    PATCH_WARP_MIX_RATIO=0.50 \
    PATCH_WARP_SELECTION_SCOPE=per_frame \
    PATCH_WARP_SELECTION_MODE=question_cosine_x_norm \
    AUG_PATCH_RATIO=0.03 \
    AUG_SELECTION_SCOPE=per_frame \
    AUG_SCORING_MODE=question_cosine_x_norm \
    AUG_INJECTION_MODE=inplace_boost_coarse_tail \
    AUG_BOOST_FACTOR=2.0

echo
echo "Sweep summary saved to ${SUMMARY_TSV}"
