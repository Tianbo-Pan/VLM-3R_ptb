#!/bin/bash

# 设置工作目录
cd "$(dirname "$0")/../.."

# 执行 Python 模块
python -m src.tasks.vstibench.get_camera_displacement_qa \
    --split_path datasets/ScanNet/Tasks/Benchmark/scannetv2_val.txt \
    --split_type val \
    --processed_data_path data/processed_data/ScanNet \
    --output_dir data/processed_data/ScanNet/qa/vstibench \
    --dataset scannet \
    --question_template VSTI_CAMERA_DISPLACEMENT_TEMPLATE \
    --question_type camera_displacement \
    --output_filename_prefix qa_camera_displacement \
    --num_subsample 3 \
    --num_workers 64 