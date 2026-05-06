# ptb_preliminary_exp

用于做 **preliminary evidence**：验证 VLM-3R 在部分 spatial reasoning case 上是否存在
“不充分依赖关键视觉证据”的现象。

当前包含两个脚本：

1. `exp1_counterfactual_consistency.py`
   - 原始视频 vs `no_vision / mismatched_video / frame_shuffle`
   - 统计 same-answer rate
2. `exp2_targeted_vs_random_ablation.py`
   - 用 `ptb_patch_selection` 的 patch score 做 targeted / random / low-score ablation
   - 统计 GT margin drop、flip rate、selective reliance gap

## 运行要点

- 默认只跑 `ptb_test` manifest 中带 `options_map` 的 MCA 题。
- 所有决策默认都走 **option-letter scoring**，减少自由生成噪声。
- 输出：
  - 每题 `case_result.json`
  - aggregate json
  - 简单统计图（png）

