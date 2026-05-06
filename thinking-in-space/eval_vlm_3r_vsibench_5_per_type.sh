#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"  # 可在命令前覆盖，例如 CUDA_VISIBLE_DEVICES=0 bash ...
export LMMS_EVAL_LAUNCHER="accelerate"
export LMMS_EVAL_PER_TYPE_LIMIT="${LMMS_EVAL_PER_TYPE_LIMIT:-15}"
export LMMS_EVAL_SAMPLE_SEED="${LMMS_EVAL_SAMPLE_SEED:-42}"

OUTPUT_ROOT="logs/$(TZ="America/New_York" date "+%Y%m%d")/vsibench_${LMMS_EVAL_PER_TYPE_LIMIT}_per_type"

accelerate launch \
    --num_processes=4 \
    -m lmms_eval \
    --model vlm_3r \
    --model_args pretrained=Journey9ni/vlm-3r-llava-qwen2-lora,model_base=lmms-lab/LLaVA-NeXT-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=32 \
    --tasks vsibench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "vlm_3r_7b_qwen2_lora_${LMMS_EVAL_PER_TYPE_LIMIT}_per_type" \
    --output_path "$OUTPUT_ROOT"

LATEST_RUN="$(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
LOG_JSON="$(find "$LATEST_RUN" -maxdepth 1 -type f -name 'vsibench*.json' | sort | tail -n 1)"

if [[ -z "${LOG_JSON:-}" || ! -f "$LOG_JSON" ]]; then
    echo "Cannot find log file under: $LATEST_RUN"
    exit 1
fi

echo
echo "===== Per-question-type sample counts ====="
python - "$LOG_JSON" <<'PY'
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

echo
echo "Full log saved at: $LOG_JSON"
