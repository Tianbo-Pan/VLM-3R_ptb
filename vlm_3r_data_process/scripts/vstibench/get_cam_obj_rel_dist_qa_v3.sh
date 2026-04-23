#!/bin/bash

# 设置工作目录
cd "$(dirname "$0")/../.."

# 执行 Python 模块
python -m src.tasks.vstibench.get_cam_obj_rel_dist_qa_v3 \
    --split_path datasets/ScanNet/Tasks/Benchmark/scannetv2_val.txt \
    --split_type val \
    --processed_data_path data/processed_data/ScanNet \
    --output_dir data/processed_data/ScanNet/qa/vstibench \
    --dataset scannet \
    --question_template VSTI_CAMERA_OBJ_REL_DIST_TEMPLATE_V3 \
    --question_type camera_obj_rel_dist_v3 \
    --output_filename_prefix qa_camera_obj_rel_dist_v3 \
    --num_subsample 5 \
    --num_workers 64