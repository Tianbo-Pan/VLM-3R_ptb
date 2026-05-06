#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}/thinking-in-space"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export LMMS_EVAL_LAUNCHER="${LMMS_EVAL_LAUNCHER:-accelerate}"
export LMMS_EVAL_PER_TYPE_LIMIT="${LMMS_EVAL_PER_TYPE_LIMIT:-5}"
export LMMS_EVAL_SAMPLE_SEED="${LMMS_EVAL_SAMPLE_SEED:-42}"

NUM_PROCESSES="${NUM_PROCESSES:-1}"
PRETRAINED="${PRETRAINED:-Journey9ni/vlm-3r-llava-qwen2-lora}"
MODEL_BASE="${MODEL_BASE:-lmms-lab/LLaVA-NeXT-Video-7B-Qwen2}"
CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_1_5}"
MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-32}"

FINE_TOPK="${FINE_TOPK:-16}"
FINE_RATIO="${FINE_RATIO:-none}"
SELECTION_SCOPE="${SELECTION_SCOPE:-per_frame}"
SCORING_MODE="${SCORING_MODE:-question_cosine}"
FINE_SCALE="${FINE_SCALE:-1}"
INCLUDE_COARSE="${INCLUDE_COARSE:-True}"
APPEND_NEWLINE="${APPEND_NEWLINE:-True}"
COARSE_MODE="${COARSE_MODE:-full}"
COARSE_CONTEXT_RADIUS="${COARSE_CONTEXT_RADIUS:-0}"
COARSE_CONTEXT_TOPK="${COARSE_CONTEXT_TOPK:-none}"
COARSE_CONTEXT_SCALE="${COARSE_CONTEXT_SCALE:-1.0}"
CONTEXTUAL_COARSE_FIRST="${CONTEXTUAL_COARSE_FIRST:-True}"

RUN_SUFFIX="${RUN_SUFFIX:-vlm_3r_patch_selection_${LMMS_EVAL_PER_TYPE_LIMIT}_per_type}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/$(TZ="America/New_York" date "+%Y%m%d")/vsibench_patch_selection_${LMMS_EVAL_PER_TYPE_LIMIT}_per_type}"
BASELINE_LOG_JSON="${BASELINE_LOG_JSON:-}"

MODEL_ARGS="pretrained=${PRETRAINED},model_base=${MODEL_BASE},conv_template=${CONV_TEMPLATE},max_frames_num=${MAX_FRAMES_NUM},fine_topk=${FINE_TOPK},fine_ratio=${FINE_RATIO},selection_scope=${SELECTION_SCOPE},scoring_mode=${SCORING_MODE},fine_scale=${FINE_SCALE},include_coarse=${INCLUDE_COARSE},append_newline=${APPEND_NEWLINE},coarse_mode=${COARSE_MODE},coarse_context_radius=${COARSE_CONTEXT_RADIUS},coarse_context_topk=${COARSE_CONTEXT_TOPK},coarse_context_scale=${COARSE_CONTEXT_SCALE},contextual_coarse_first=${CONTEXTUAL_COARSE_FIRST}"

echo "Running sampled VSIBench patch-selection evaluation"
echo "model_args: ${MODEL_ARGS}"
echo "LMMS_EVAL_PER_TYPE_LIMIT: ${LMMS_EVAL_PER_TYPE_LIMIT}"
echo "LMMS_EVAL_SAMPLE_SEED: ${LMMS_EVAL_SAMPLE_SEED}"
echo "output_path: ${OUTPUT_ROOT}"
echo "baseline_log: ${BASELINE_LOG_JSON:-<none>}"

accelerate launch \
    --num_processes="${NUM_PROCESSES}" \
    -m lmms_eval \
    --model vlm_3r_patch_selection \
    --model_args "${MODEL_ARGS}" \
    --tasks vsibench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "${RUN_SUFFIX}" \
    --output_path "${OUTPUT_ROOT}"

LATEST_RUN="$(find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
LOG_JSON="$(find "${LATEST_RUN}" -maxdepth 1 -type f -name 'vsibench*.json' | sort | tail -n 1)"

if [[ -z "${LOG_JSON:-}" || ! -f "${LOG_JSON}" ]]; then
    echo "Cannot find patch-selection log file under: ${LATEST_RUN}"
    exit 1
fi

echo
echo "===== Patch-selection per-question-type sample counts ====="
python - "${LOG_JSON}" <<'PY'
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

if [[ -z "${BASELINE_LOG_JSON:-}" || ! -f "${BASELINE_LOG_JSON}" ]]; then
    echo
    echo "Baseline log not found at: ${BASELINE_LOG_JSON:-<none>}"
    echo "Patch-selection log saved at: ${LOG_JSON}"
    exit 0
fi

echo
echo "===== Patch-selection vs baseline ====="
python - "${BASELINE_LOG_JSON}" "${LOG_JSON}" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

baseline_path = Path(sys.argv[1])
current_path = Path(sys.argv[2])

MCA_QUESTION_TYPES = {
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
}
NA_QUESTION_TYPES = {
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
}


def load_log(path: Path):
    data = json.loads(path.read_text())
    logs = data.get("logs", [])
    by_id = {}
    per_type = defaultdict(list)
    for item in logs:
        doc = item.get("doc", {})
        qtype = doc.get("question_type")
        doc_id = doc.get("id", item.get("doc_id"))
        if qtype in MCA_QUESTION_TYPES:
            metric_name = "accuracy"
        elif qtype in NA_QUESTION_TYPES:
            metric_name = "MRA:.5:.95:.05"
        else:
            metric_name = None
        metric_value = float(doc.get(metric_name, 0.0)) if metric_name is not None else None
        record = {
            "id": doc_id,
            "question_type": qtype,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "prediction": doc.get("prediction"),
        }
        by_id[doc_id] = record
        per_type[qtype].append(record)
    return logs, by_id, per_type


def summarize_per_type(per_type):
    summary = {}
    direction_scores = []
    for qtype, records in per_type.items():
        if not records:
            continue
        score = sum(r["metric_value"] for r in records) / len(records)
        summary[qtype] = score
        if qtype in {
            "object_rel_direction_easy",
            "object_rel_direction_medium",
            "object_rel_direction_hard",
        }:
            direction_scores.append(score)
    if direction_scores:
        summary["object_rel_direction_accuracy"] = sum(direction_scores) / len(direction_scores)
    overall = sum(summary.values()) / len(summary) if summary else 0.0
    return summary, overall


baseline_logs, baseline_by_id, baseline_per_type = load_log(baseline_path)
current_logs, current_by_id, current_per_type = load_log(current_path)
baseline_summary, baseline_overall = summarize_per_type(baseline_per_type)
current_summary, current_overall = summarize_per_type(current_per_type)

shared_ids = sorted(set(baseline_by_id) & set(current_by_id))
print(f"baseline log: {baseline_path}")
print(f"current log:  {current_path}")
print(f"baseline samples: {len(baseline_logs)}")
print(f"current samples:  {len(current_logs)}")
print(f"shared samples:   {len(shared_ids)}")
print()
print("Per-type metric comparison (baseline%, current%, delta%)")
for qtype in sorted(set(baseline_summary) | set(current_summary)):
    base = baseline_summary.get(qtype)
    cur = current_summary.get(qtype)
    if base is None or cur is None:
        print(f"{qtype}\t{base}\t{cur}\tN/A")
    else:
        print(f"{qtype}\t{base * 100:.3f}\t{cur * 100:.3f}\t{(cur - base) * 100:+.3f}")

print()
print(f"Overall (baseline -> current): {baseline_overall * 100:.3f} -> {current_overall * 100:.3f} ({(current_overall - baseline_overall) * 100:+.3f})")
PY

echo
echo "Patch-selection log saved at: ${LOG_JSON}"
