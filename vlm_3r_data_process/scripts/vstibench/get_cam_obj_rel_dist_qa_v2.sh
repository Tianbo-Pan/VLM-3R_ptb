#!/bin/bash

# 设置工作目录
cd "$(dirname "$0")/../.."

# 执行 Python 模块
python -m src.tasks.vstibench.get_cam_obj_rel_dist_qa_v2 \
    --split_path datasets/ScanNet/Tasks/Benchmark/scannetv2_val.txt \
    --split_type val \
    --processed_data_path data/processed_data/ScanNet \
    --output_dir data/processed_data/ScanNet/qa/vstibench \
    --dataset scannet \
    --question_template VSTI_CAMERA_OBJ_REL_DIST_TEMPLATE_V2 \
    --question_type camera_obj_rel_dist_v2 \
    --output_filename_prefix qa_camera_obj_rel_dist_v2 \
    --num_subsample 100 \
    --num_workers 64 