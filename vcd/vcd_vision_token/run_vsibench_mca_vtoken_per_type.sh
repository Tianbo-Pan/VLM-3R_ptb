#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)_vsibench_mca_vtoken_5_per_type}"
OUTPUT_ROOT="${OUTPUT_ROOT:-vcd/vcd_vision_token/results}"
SOURCE_LOG_JSON="${SOURCE_LOG_JSON:-}"
PER_TYPE_LIMIT="${PER_TYPE_LIMIT:-5}"

CMD=(python vcd/vcd_vision_token/vsibench_mca_vcd.py \
  --run_name "$RUN_NAME" \
  --output_root "$OUTPUT_ROOT" \
  --per_type_limit "$PER_TYPE_LIMIT")

if [[ -n "$SOURCE_LOG_JSON" ]]; then
  CMD+=(--source_log_json "$SOURCE_LOG_JSON")
fi

"${CMD[@]}"
