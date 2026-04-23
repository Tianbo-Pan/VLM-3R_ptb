#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"  # 可在命令前覆盖，例如 CUDA_VISIBLE_DEVICES=0 bash ...
export LMMS_EVAL_LAUNCHER="accelerate"

OUTPUT_ROOT="logs/$(TZ="America/New_York" date "+%Y%m%d")/vsibench_first10"

accelerate launch \
    --num_processes=1 \
    -m lmms_eval \
    --model vlm_3r \
    --model_args pretrained=Journey9ni/vlm-3r-llava-qwen2-lora,model_base=lmms-lab/LLaVA-NeXT-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=32 \
    --tasks vsibench \
    --batch_size 1 \
    --limit 10 \
    --log_samples \
    --log_samples_suffix vlm_3r_7b_qwen2_lora_first10 \
    --output_path "$OUTPUT_ROOT"

LATEST_RUN="$(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
LOG_JSON="$LATEST_RUN/vsibench_10.json"

if [[ ! -f "$LOG_JSON" ]]; then
    echo "Cannot find log file: $LOG_JSON"
    exit 1
fi

echo
echo "===== First 10 vsibench cases ====="
python - "$LOG_JSON" <<'PY'
import json
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
data = json.loads(log_path.read_text())
logs = data.get("logs", [])

score_keys = ["accuracy", "MRA:.5:.95:.05", "success_rate"]
meta_keys = {"mc_answer", "video_path", "scene_name", "question_type", "question", "ground_truth", "options", "id", "prediction", "dataset"}

for i, item in enumerate(logs[:10], 1):
    score_obj = item.get("vsibench_score", {})
    score_name = None
    score_value = None
    for key in score_keys:
        if key in score_obj:
            score_name = key
            score_value = score_obj[key]
            break
    if score_name is None:
        extra = [k for k in score_obj.keys() if k not in meta_keys]
        if extra:
            score_name = extra[0]
            score_value = score_obj[extra[0]]
        else:
            score_name = "score"
            score_value = "N/A"

    print(f"[{i}] doc_id={item.get('doc_id')}  id={score_obj.get('id', item.get('doc', {}).get('id'))}")
    print(f"question_type: {score_obj.get('question_type', item.get('doc', {}).get('question_type'))}")
    print(f"question     : {score_obj.get('question', item.get('doc', {}).get('question'))}")
    print(f"gt / pred    : {score_obj.get('ground_truth', item.get('target'))} / {score_obj.get('prediction', item.get('filtered_resps', ['']))[0]}")
    print(f"{score_name:<13}: {score_value}")
    print("-" * 80)
PY

echo "Full log saved at: $LOG_JSON"
