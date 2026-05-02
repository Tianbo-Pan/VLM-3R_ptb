# VCD、SECOND 与 VLM-3R Spatial Reasoning 讨论记录

## 1. 背景

本记录整理了围绕 `VLM-3R` 仓库中 VCD 相关实现、论文脉络，以及如何用 **training-free** 方法提升 spatial reasoning 能力的一次连续讨论，重点聚焦：

- `vcd/vcd_vision_token` 里的 VCD 到底是什么
- VCD 是否只能用于 option/logit 选择，还是也适用于 generate
- 对 spatial reasoning，哪些 VCD 系方法最值得迁移到 `VLM-3R`
- 为什么当前 repo 还没有真正做到 **generation-time VCD + adaptive plausibility constraint**
- 如果 VSIBench 题目要求“只输出一个字母/数字”，这会如何影响 generation-time VCD 设计

---

## 2. 当前仓库里的 VCD 是什么

### 2.1 `vcd/vcd_vision_token` 中的 VCD本质

从代码实现看，当前目录下的 VCD 本质上是：

> **推理时的视觉对比重打分（contrastive re-scoring）**

核心做法是：

1. 构造 **原始视觉分支** `stage0_coarse_only`
2. 构造 **负分支** `stage0_semantic_negative_coarse`
   - 先按问题语义对 patch token 打分
   - 选择每帧最相关的一部分 patch
   - 将这些 patch 置零或替换为 frame mean
   - 再做 coarse pooling
3. 对每个候选答案分别计算：
   - 原始分支的 `sequence_logprob`
   - 负分支的 `sequence_logprob`
4. 用下面公式组合：

\[
(1+\alpha)\cdot s_{orig} - \alpha \cdot s_{neg}
\]

因此它的本质是：

> 如果某个答案在“关键视觉证据被破坏后”仍然得分很高，那么这个答案可能更多依赖语言偏置或伪视觉线索，因此应该被压低。

### 2.2 当前实现的核心特点

- 这不是训练方法
- 这是 **推理时方法**
- 它现在更接近 **候选答案重打分 / rerank**
- 它目前主要服务于 VSIBench MCA（多选）评估

---

## 3. VCD 只能用于 option 选择吗？对 generate 有用吗？

### 3.1 方法层面：有用

从 VCD / SECOND 这类论文角度看，这类方法本质上是 **decoding-time** 方法，因此并不局限于多选题。

它理论上可以用于：

- 多选题
- 开放问答
- 数值生成
- 简短空间答案生成

### 3.2 但当前 repo 的语义 VCD 实现主要是 MCA rerank

当前 `vcd/vcd_vision_token` 和 `thinking-in-space/lmms_eval/models/vlm_3r_semantic_vcd.py` 这条线，本质上还是：

- 已知候选集合
- 对每个 candidate 串算分
- 再做 VCD 式重排

所以它适合：

- `A/B/C/D` 这种 MCA
- 已枚举答案集合的数值候选

但它**还不等于**真正的 generate-time VCD。

### 3.3 对 VSIBench 距离数值题的判断

如果距离题是：

- 候选数值集合已知：**VCD 很适合**
- 自由生成任意数值：**VCD 可以帮助，但不一定直接解决精确回归问题**

原因是距离估计还依赖：

- 尺度感
- 深度/遮挡线索
- 几何关系
- 相机先验

VCD更像是：

> 减少 hallucination 和语言偏置，而不是直接给模型补上“绝对量纲能力”。

---

## 4. VCD 相关文章调研后的总体判断

围绕 VCD 主线，重点讨论了以下工作（按思想脉络归纳）：

### 4.1 基础母体

- **Contrastive Decoding (ACL 2023)**  
  提出 expert–amateur 对比与 **plausibility constraint**

- **VCD / Visual Contrastive Decoding (CVPR 2024)**  
  将图像扰动引入 LVLM hallucination 缓解

### 4.2 2024：更聪明的负分支

- **LCD**：language prior contrast
- **ICD**：instruction contrast

### 4.3 2025：更结构化、更局部化的 contrast

- **SID**：选择性 token contrast
- **DAMO**：layer-wise consistency / overthinking 缓解
- **SECOND**：selective + contrastive，多尺度视觉证据整合
- **RVCD**：retrieval-based visual contrastive decoding
- **MFCD**：频域 contrastive decoding
- **MaskCD**：image-head masked contrastive decoding

### 4.4 2026：更强的 object-aligned contrast

- **Object-Aligned VCD / Mask What Matters**

---

## 5. 哪些技术最适合 training-free 提升 VLM-3R 的 spatial reasoning

讨论后的核心结论是：

### 5.1 最推荐的路线

> **SECOND 式 selective tokens**  
> + **object/head/fusion-aware 的负分支构造**  
> + **generation-time token-level contrastive decoding**  
> + **adaptive plausibility constraint**

### 5.2 原因

空间推理问题往往不是“整图看不清”，而是：

- 没抓住关键对象
- 没抓住关键局部区域
- 没抓住 3D / fusion 线索
- 被语言先验在最后几步带偏

因此，相比于“整图加噪/模糊”的原始 VCD，更值得做的是：

- **semantic top-k patch negative**
- **fusion-guided negative**
- **camera / patch token 选择性弱化**
- **image head / spatial head masking**
- **coarse-to-fine selective decoding**

### 5.3 对 VLM-3R 最有价值的几个方向

1. **SECOND 风格 selective multi-scale decoding**
2. **object-aligned / token-aligned negative branch**
3. **3D / fusion-aware contrast**
4. **加回 adaptive plausibility constraint**
5. **对距离题尝试 MFCD（频域分支）**

---

## 6. 为什么说 contrast 更应该发生在生成时，而不只是候选重打分

这是整个讨论中的核心观点之一。

### 6.1 候选重打分的局限

rerank 的前提是：

> 正确答案已经在候选集合里，或者已经被模型生成出来。

但 open-ended spatial QA / distance estimation 中，很多错误发生在：

- 第一个数字 token
- 第一个方向词
- 第一个关键关系词

例如：

- `left` vs `right`
- `2` vs `5`
- `front` vs `behind`

如果模型在很早的步骤已经走错了轨迹，那么事后 rerank 往往无能为力。

### 6.2 generation-time VCD 的价值

generation-time VCD 的优势在于：

- 它能 **逐步干预 token 选择**
- 它能在 early step 压低“依赖错误视觉证据/语言偏置”的 token
- 它能改变搜索轨迹，而不是事后评价完整答案

所以对于：

- 开放式空间问答
- 距离估计
- 简短数值/方向输出

generation-time contrast 往往更重要。

---

## 7. 为什么当前 repo 还没有真正做到 generation-time VCD

### 7.1 `semantic_vcd` 线：本质是 MCA rerank

当前 `vlm_3r_semantic_vcd.py` 的逻辑是：

- MCA 且有 options 时：
  - 构造 coarse / negative branch
  - 对每个选项算 `sequence_logprob`
  - 再做 VCD 重排
- 其他情况：
  - 直接回退到普通生成

所以这条线：

- 不是 open-ended generation-time VCD
- 没有逐 token contrast
- 没有 `V_head`

### 7.2 `feature_cd` 线：进入了 generate，但不是 dual-branch logit decoding

当前 `two_d_feature_cd.py` / `three_d_feature_cd.py` / `fusion_feature_cd.py` 做的是：

1. 分别构造 `orig_video_features` 和 `neg_video_features`
2. 先在特征空间合成：

\[
guided = orig + \lambda(orig - neg)
\]

3. 然后把这个 **单一 guided feature** 送入 `generate()`

这条路线的问题是：

- 对比发生在特征层，而不是 token logit 层
- 原始分支与负分支在生成前就被合并了
- 后续生成时只剩一个分支
- 因而无法在每个 step 上：
  - 同时拿到 `logits_orig` 和 `logits_neg`
  - 构造 plausibility set
  - 再做 CD/VCD 风格 token-level contrast

### 7.3 更深层原因

我认为当前 repo 之所以还没做到，不只是“还没写”，而是当前工程路线天然偏向两个简化版：

#### 简化版1：把 VCD 当成 sequence scorer

优点：

- 非常适合 MCA
- 实现简单

缺点：

- 无法改变生成轨迹
- 不适合真正 open-ended decoding

#### 简化版2：把 VCD 当成 feature guidance

优点：

- 能直接接入 `generate()`
- 工程侵入低

缺点：

- 不是完整 dual-branch decoding
- 没有 adaptive plausibility constraint

---

## 8. 什么是 adaptive plausibility constraint，为什么它重要

CD 和 VCD 的核心不是单纯做：

\[
(1+\alpha)l_{orig} - \alpha l_{neg}
\]

更关键的是：

> **不能对整个词表盲目减分**

而要先基于原始 expert 分支构造一个“可信候选集合”。

VCD 论文中常见形式是：

\[
\mathcal V_{head}(y_{<t})=\{y_t: p_{orig}(y_t|y_{<t}) \ge \beta \max_w p_{orig}(w|y_{<t})\}
\]

然后只在这个集合上做 contrast。

### 8.1 没有这个约束会怎样

没有 plausibility constraint 时，contrast 可能：

- 把正确 token 也减掉
- 把生成推向很奇怪的 token
- 对短答案尤其危险

### 8.2 为什么对 spatial generation 特别重要

空间任务的关键 token 往往是：

- `A/B/C/D`
- `1/2/3/4`
- `left/right`
- `front/behind`

这类 token：

- 数量少
- 决策关键
- 很容易被错误对比放大

因此更需要：

- 先用原始分支定义“合理答案空间”
- 再在这个集合中做 contrast

---

## 9. 可能的改进方式

### 9.1 最小改动：generate N candidates + VCD rerank

适合快速验证。

做法：

1. 先生成多个候选答案
2. 用当前 `score_candidate_with_video_features()` 分别算 orig / neg 分数
3. 用 VCD sequence score rerank

优点：

- 改动小
- 马上可用于数值题

缺点：

- 仍然不是真正逐 token decoding

### 9.2 推荐改法：dual-branch generation-time VCD

增加一个真正的双分支 decoder，例如：

- `llava/model/feature_cd/dual_branch_cd.py`

核心流程：

1. 维护两套分支：
   - `orig branch`
   - `neg branch`
2. 每步分别得到：
   - `logits_orig`
   - `logits_neg`
3. 用原始分支构造：

\[
V_t = \{y: p_{orig}(y) \ge \beta \max p_{orig}\}
\]

4. 只在 `V_t` 上做：

\[
logits_{cd} = (1+\alpha)logits_{orig} - \alpha logits_{neg}
\]

5. 从 `logits_cd` 中采样/贪心
6. 生成的 token 同时回灌到两条分支

### 9.3 更适合 VLM-3R 的自适应版本

#### 自适应 `beta`

按原始分支熵/置信度动态调整：

- 高置信时：更严格
- 高不确定时：更宽松

#### 自适应 `alpha`

按 token 类型调整：

- function word：低 `alpha`
- 空间关系词 / 数字词：高 `alpha`
- EOS：适中

#### 仅在 answer span 强化 contrast

例如：

- reasoning 段弱对比
- 最终答案段强对比

---

## 10. VSIBench 只要求输出一个字母/数字，会影响 generation-time VCD 吗？

### 10.1 会有影响，但总体是正面影响

如果题目要求：

- “只输出选项字母”
- “只输出一个数字”

那么 generation-time VCD 会从“长序列生成纠偏”退化成：

> **single-step / few-step constrained contrastive decoding**

这反而更稳定。

### 10.2 为什么反而更适合做 generation-time VCD

因为输出空间非常小，例如：

- `{A, B, C, D}`
- `{1, 2, 3, 4}`

这时最合理的做法是：

1. 先定义允许答案 token 集
2. 再和 expert-based plausibility set 取交集
3. 只在这个集合内做 contrast

例如：

\[
V_t = AllowedAnswerTokens \cap PlausibleTokensFromOrig
\]

### 10.3 对这种题，generation-time VCD 的意义会发生变化

对于这类任务，它的价值不再主要是：

- 改变长回答的搜索轨迹

而是：

- 在首个关键 answer token 上做更干净、更受约束的对比判别

### 10.4 一个重要判断

如果题目要求：

> “Answer with the option's letter directly.”

那么直接比较：

- `A`
- `B`
- `C`

通常会比比较整条 candidate string：

- `"A. table"`
- `"B. tv"`
- `"C. radiator"`

更纯净，因为后者会把对象文本本身的语言偏置也引入评分。

---

## 11. 总结结论

### 11.1 关于当前 repo

当前 repo 已经有：

- MCA rerank 版 semantic VCD
- feature-guidance 版 generate-time CD

但还没有真正实现：

> **带 dual-branch token-level logits、且带 adaptive plausibility constraint 的 generation-time semantic/fusion VCD**

### 11.2 关于 spatial reasoning 的最优方向

如果目标是 **training-free 提升 VLM-3R 的 spatial reasoning**，最值得走的路线是：

> **SECOND 风格 selective multi-scale evidence**  
> + **object/token/fusion-aware negative branch**  
> + **generation-time token-level contrast**  
> + **adaptive plausibility constraint**

### 11.3 关于 VSIBench 的字母/数字答案

对这类任务：

- generation-time VCD **不会失效**
- 反而更适合变成：
  - **single-step / few-step constrained contrastive decoding**
  - **answer-space-masked VCD**

也就是说，最自然的落地形态不是长句子生成，而是：

> **在非常小的答案 token 空间里，做受约束的 VCD 判别。**

---

## 12. 后续建议

建议后续按以下顺序推进：

1. **实现 answer-token masked semantic VCD**
   - 先支持字母/数字题
2. **实现 dual-branch generation-time decoder**
   - 保留 orig / neg 两套 logits
3. **加入 adaptive plausibility constraint**
   - 先做固定 `beta`
   - 再做 entropy-aware `beta`
4. **扩展 negative branch**
   - semantic top-k
   - fusion-guided
   - 3D dropout
   - object-aligned masking
5. **做 VSIBench ablation**
   - letter-only scoring
   - candidate-string scoring
   - rerank vs step-wise VCD

---

## 13. 参考链接

- Contrastive Decoding, ACL 2023  
  https://aclanthology.org/2023.acl-long.687/

- Visual Contrastive Decoding, CVPR 2024  
  https://openaccess.thecvf.com/content/CVPR2024/html/Leng_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_through_Visual_Contrastive_CVPR_2024_paper.html

- Language Contrastive Decoding, Findings ACL 2024  
  https://aclanthology.org/2024.findings-acl.359/

- Instruction Contrastive Decoding, Findings ACL 2024  
  https://aclanthology.org/2024.findings-acl.937/

- Self-Introspective Decoding (SID), ICLR 2025  
  https://openreview.net/forum?id=rsZwwjYHuD

- DAMO, ICLR 2025  
  https://openreview.net/forum?id=JUr0YOMvZA

- SECOND, ICML 2025  
  https://proceedings.mlr.press/v267/park25c.html

- RVCD, Findings ACL 2025  
  https://aclanthology.org/2025.findings-acl.430/

- MaskCD, Findings EMNLP 2025  
  https://aclanthology.org/2025.findings-emnlp.1025/

- Object-Aligned VCD / Mask What Matters, EACL SRW 2026  
  https://aclanthology.org/2026.eacl-srw.2/
