# Patch-warp generation-time VCD

This package contains a lightweight **local patch-shift** weak branch for VLM-3R generation-time VCD.

## Negative branch

- start from fused `encoded_video_features`
- score important patches with one of:
  - `question_cosine`
  - `question_cosine_x_norm`
  - `feature_norm`
  - `fusion_2d3d`
  - `fusion_2d`
  - `fusion_3d`
- select top `patch_warp_ratio`
- replace each selected patch with a local neighbor mix

## Main knobs

- `patch_warp_selection_mode`
- `patch_warp_ratio`
- `patch_warp_shift_size`
- `patch_warp_mix_ratio`
- `patch_warp_fusion_2d_weight`
- `patch_warp_fusion_3d_weight`

## LMMS eval entrypoint

Use:

```bash
bash thinking-in-space/eval_vlm_3r_gen_vcd_patch_warp_vsibench.sh
```
