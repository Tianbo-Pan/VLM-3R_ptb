#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

# --scoring-mode "${SCORING_MODE:-question_cosine}"
# --output-root ptb_preliminary_exp/outputs/targeted_vs_random_ablation_query


python -m ptb_preliminary_exp.exp2_targeted_vs_random_ablation \
  --model-path "${MODEL_PATH:-Journey9ni/vlm-3r-llava-qwen2-lora}" \
  --model-base "${MODEL_BASE:-lmms-lab/LLaVA-NeXT-Video-7B-Qwen2}" \
  --overwrite \
  --mm-spatial-pool-stride "${MM_SPATIAL_POOL_STRIDE:-2}" \
  --mm-spatial-pool-mode "${MM_SPATIAL_POOL_MODE:-average}" \
  --mm-newline-position "${MM_NEWLINE_POSITION:-grid}" \
  --num-frames "${NUM_FRAMES:-32}" \
  --max-cases "${MAX_CASES:-20}" \
  --scoring-mode "${SCORING_MODE:-fusion_3d}" \
  --score-key "${SCORE_KEY:-avg_logprob}" \
  --patch-ratio "${PATCH_RATIO:-0.3}" \
  --ablation-mode "${ABLATION_MODE:-zero}" \
  --save-case-visuals \
  --output-root "${OUTPUT_ROOT:-ptb_preliminary_exp/outputs/targeted_vs_random_ablation_3d}"

