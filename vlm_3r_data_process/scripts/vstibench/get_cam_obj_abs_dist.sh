#!/bin/bash

# 设置工作目录
cd "$(dirname "$0")/../.."

# 执行 Python 模块 (val split)
python -m src.tasks.vstibench.get_cam_obj_abs_dist_qa \
    --split_path datasets/ScanNet/Tasks/Benchmark/scannetv2_val.txt \
    --split_type val \
    --processed_data_path data/processed_data/ScanNet \
    --output_dir data/processed_data/ScanNet/qa/vstibench \
    --dataset scannet \
    --question_template VSTI_CAMERA_OBJ_DIST_TEMPLATE \
    --question_type camera_obj_abs_dist \
    --output_filename_prefix qa_camera_obj_abs_dist \
    --num_subsample 3 \
    --num_workers 64

# # 执行 Python 模块 (train split)
# python -m src.tasks.vstibench.get_cam_obj_abs_dist_qa \
#     --split_path datasets/ScanNet/Tasks/Benchmark/scannetv2_train.txt \
#     --split_type train \
#     --processed_data_path data/processed_data/ScanNet \
#     --output_dir data/processed_data/ScanNet/qa/vstibench \
#     --dataset scannet \
#     --question_template VSTI_CAM_OBJ_ABS_DIST_TEMPLATE \
#     --question_type cam_obj_abs_dist \
#     --output_filename_prefix qa_cam_obj_abs_dist \
#     --num_subsample 2000 \
#     --num_workers 64

# # format qa
# python -m src.format_qa \
#     --input_path data/processed_data/ScanNet/qa/vstibench/val/qa_camera_obj_abs_dist.json \
#     --output_path data/processed_data/ScanNet/formatted_qa/vstibench/val/qa_camera_obj_abs_dist.json
