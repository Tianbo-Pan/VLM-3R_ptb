# VCD utilities

这个目录统一放置 VCD 相关代码与输出结果。

## 代码
- `option_demo_utils.py`: 原 `playground/demo/phase1_option_vcd_utils.py` 的核心实现，已迁移到这里。
- `methods.py`: 2D feature degradation / weak-fusion / spatial feature degradation 的分支构造逻辑。
- `vsibench_mca_vcd.py`: 在 VSIBench 多个 MCA question types 上批量跑 baseline、2D-feature VCD、weak-fusion VCD、spatial-noise VCD，并输出**按单题组织的 token-level logits 可视化**。
- `run_vsibench_mca_vcd_per_type.sh`: 对齐 `thinking-in-space/eval_vlm_3r_vsibench_5_per_type.sh` 风格的启动脚本。

## 可视化输出
默认输出到：
- `vcd/results/<run_name>/results.json`
- `vcd/results/<run_name>/summary.json`
- `vcd/results/<run_name>/summary.csv`
- `vcd/results/<run_name>/plots/<question_type>/doc_*.png`

每张 `doc_*.png` 对应一道题：
- 同一种 `question_type` 的题都放在同一个文件夹下
- 一张图里按行展示 `baseline / two_d_feature_vcd / weak_fusion_vcd / spatial_noise_vcd`
- 每行都展示**所有选项的全部 token logits**
- 正确选项会用绿色背景标出，标题里也会显示预测结果与 GT

## 运行示例
```bash
bash vcd/run_vsibench_mca_vcd_per_type.sh
```

如果你已经有其他 `vsibench.json` 采样日志，也可以覆盖：
```bash
SOURCE_LOG_JSON=/path/to/vsibench.json bash vcd/run_vsibench_mca_vcd_per_type.sh
```
