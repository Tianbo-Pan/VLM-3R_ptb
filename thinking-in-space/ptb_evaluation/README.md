# PTB Evaluation

This folder contains standalone evaluation utilities for VLM-3R.

## CV-Bench

Run:

```bash
cd /local_home/pantianbo/projects/vision_reasoning/VLM-3R/thinking-in-space
bash ptb_evaluation/run_vlm_3r_cv_bench.sh
```

Or directly:

```bash
python ptb_evaluation/eval_cv_bench.py \
  --model-path Journey9ni/vlm-3r-llava-qwen2-lora \
  --model-base lmms-lab/LLaVA-NeXT-Video-7B-Qwen2 \
  --conv-mode qwen_1_5 \
  --output-dir ptb_evaluation/outputs/$(TZ="Asia/Shanghai" date "+%Y%m%d") \
  --output-name vlm_3r_cv_bench
```

Outputs:

- `*_config.json`: run config
- `*_metrics.json`: aggregated CV-Bench metrics
- `*.jsonl`: per-sample predictions

Current implementation is standalone and does **not** yet register CV-Bench into `lmms_eval/tasks`.
