#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-vcd}"
mkdir -p "$MPLCONFIGDIR"

PER_TYPE_LIMIT="${PER_TYPE_LIMIT:-5}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
SOURCE_LOG_JSON="${SOURCE_LOG_JSON:-thinking-in-space/logs/20260417/vsibench_5_per_type/0417_2133_vlm_3r_7b_qwen2_lora_5_per_type_vlm_3r_model_args_70e1b2/vsibench.json}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)_vsibench_mca_vcd_${PER_TYPE_LIMIT}_per_type}"

python -m vcd.vsibench_mca_vcd \
  --source_log_json "$SOURCE_LOG_JSON" \
  --output_root vcd/results \
  --run_name "$RUN_NAME" \
  --per_type_limit "$PER_TYPE_LIMIT" \
  --sample_seed "$SAMPLE_SEED"
