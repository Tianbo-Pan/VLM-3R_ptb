# ptb_patch_selection

方案A的一个最小实现：  
**先保留全局 coarse token，再从 fusion 后的高分 patch token 中追加 fine token**。

## 当前实现

- 输入：VLM-3R fusion 后但尚未 pooling 的 video features，形状通常是 `[F, 729, D]`
- coarse 分支：
  - 复用现有 `get_2dPool()` 和 token packing 逻辑
- fine 分支：
  - 在 fusion 后 patch token 上做 query-conditioned 打分
  - 默认打分方式：`question_cosine`
    - 用问题文本 token embedding 的均值作为 query
    - 与每个 patch token 做 cosine similarity
  - 每帧保留 `top-k`
- 新增 fusion-guided fine 分支：
  - 可用 `scoring_mode=fusion_2d3d`
  - 不再依赖 question query
  - 直接利用 **2D/3D fusion 过程中** 的 patch importance：
    - `2D importance`：fusion 前后 token 变化量 `||fused - visual_only||`
    - `3D importance`：cross-attention 中 spatial patch token 被 2D patch 查询到的强度
  - 最终分数默认是 `2D + 3D` 的加权平均
- 最终输入：
  - `coarse_tokens + selected_fine_tokens`
- demo 会额外输出：
  - `patch_selection_vis/frame_XXXX_topk.png`
  - `patch_selection_vis/patch_selection_manifest.json`
  - 其中会把每帧选中的 top-k patch 画在原始视频帧上

## 主要接口

- `build_selective_patch_video_features(...)`
  - 只构造视频 token 与元信息
- `generate_with_selective_patch_pooling(...)`
  - 直接生成回答
- `build_fusion_guided_patch_video_features(...)`
  - 新版：使用 fusion importance 构造 fine patch
- `generate_with_fusion_guided_patch_pooling(...)`
  - 新版：直接走 fusion-guided patch selection 推理

## 设计取舍

- 不改动主干模型，不需要训练
- 不破坏原始 fusion 过程
- 选择发生在 **fusion 之后、pooling 之前**
- 当前版本是最小实现：
  - 还没有 temporal consistency
  - 还没有 2D/3D 联合 mask
  - 还没有 SECOND 风格 multi-stage CD

## 一个简单调用示例

```python
from ptb_patch_selection import generate_with_selective_patch_pooling

output_ids, metadata = generate_with_selective_patch_pooling(
    model,
    input_ids=input_ids,
    images=video,
    attention_mask=attention_masks,
    modalities="video",
    fine_topk=16,
    scoring_mode="question_cosine",
    fine_scale=1.0,
    include_coarse=True,
    return_metadata=True,
    max_new_tokens=256,
    do_sample=False,
    temperature=0.0,
)
```

fusion-guided 版本示例：

```python
output_ids, metadata = generate_with_selective_patch_pooling(
    model,
    input_ids=input_ids,
    images=video,
    attention_mask=attention_masks,
    modalities="video",
    fine_topk=16,
    scoring_mode="fusion_2d3d",
    fusion_2d_weight=1.0,
    fusion_3d_weight=1.0,
    include_coarse=True,
    return_metadata=True,
    max_new_tokens=256,
    do_sample=False,
    temperature=0.0,
)
```
