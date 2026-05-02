# VCD vision-token stage analysis

This directory evaluates how different **stage-wise vision token constructions** affect option/token logits on VSIBench MCA questions.

## Stages
- `stage0_coarse_only`: baseline coarse pooled visual tokens only
- `stage0_semantic_negative_coarse`: select the top semantic patches on the original encoded features by a per-frame ratio (default 50%), corrupt them before pooling, then use coarse-only tokens as a negative branch
- `stage1_semantic_fine`: coarse + question-conditioned semantic fine patches
- `stage2_fusion_guided_fine`: coarse + fusion-guided fine patches
- `stage3_joint_semantic_fusion`: coarse + jointly scored semantic/fusion fine patches

## Main script
- `vsibench_mca_vcd.py`: batch evaluation and token-logit plotting across the four stages.

## Output
- `results/<run_name>/results.json`
- `results/<run_name>/summary.json`
- `results/<run_name>/summary.csv`
- `results/<run_name>/plots/<question_type>/doc_*.png`

## Run
```bash
bash vcd/vcd_vision_token/run_vsibench_mca_vtoken_per_type.sh
```
