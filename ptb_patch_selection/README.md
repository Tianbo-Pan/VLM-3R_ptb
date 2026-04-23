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
