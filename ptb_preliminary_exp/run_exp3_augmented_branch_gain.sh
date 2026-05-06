#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

python -m ptb_preliminary_exp.exp3_augmented_branch_gain \
  --model-path "${MODEL_PATH:-Journey9ni/vlm-3r-llava-qwen2-lora}" \
  --model-base "${MODEL_BASE:-lmms-lab/LLaVA-NeXT-Video-7B-Qwen2}" \
  --overwrite \
  --mm-spatial-pool-stride "${MM_SPATIAL_POOL_STRIDE:-2}" \
  --mm-spatial-pool-mode "${MM_SPATIAL_POOL_MODE:-average}" \
  --mm-newline-position "${MM_NEWLINE_POSITION:-grid}" \
  --num-frames "${NUM_FRAMES:-32}" \
  --max-cases "${MAX_CASES:-20}" \
  --scoring-mode "${SCORING_MODE:-question_cosine}" \
  --score-key "${SCORE_KEY:-avg_logprob}" \
  --patch-ratio "${PATCH_RATIO:-0.3}" \
  --selection-scope "${SELECTION_SCOPE:-per_frame}" \
  --fine-scale "${FINE_SCALE:-1.0}" \
  --injection-mode "${INJECTION_MODE:-tail_append}" \
  --coarse-alpha "${COARSE_ALPHA:-0.5}" \
  --coarse-agg-mode "${COARSE_AGG_MODE:-score_weighted}" \
  ${PRESERVE_COARSE_NORM:+--preserve-coarse-norm} \
  --append-newline \
  --output-root "${OUTPUT_ROOT:-ptb_preliminary_exp/outputs/augmented_branch_gain_query_patch_selection}"
