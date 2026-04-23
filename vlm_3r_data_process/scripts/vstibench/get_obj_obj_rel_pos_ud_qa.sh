#!/bin/bash

# 设置工作目录
cd "$(dirname "$0")/../.."

# 执行 Python 模块
python -m src.tasks.vstibench.get_obj_obj_rel_pos_ud_qa \
    --split_path datasets/ScanNet/Tasks/Benchmark/scannetv2_val.txt \
    --split_type val \
    --processed_data_path data/processed_data/ScanNet \
    --output_dir data/processed_data/ScanNet/qa/vstibench \
    --dataset scannet \
    --question_template VSTI_OBJ_OBJ_REL_POS_UD_TEMPLATE \
    --question_type obj_obj_relative_pos_ud \
    --output_filename_prefix qa_obj_obj_relative_pos_ud \
    --num_subsample 6 \
    --num_workers 64 

# # 执行 Python 模块 (Up/Down)
# python -m src.tasks.vstibench.get_obj_obj_rel_pos_ud_qa \
#     --split_path datasets/ScanNet/Tasks/Benchmark/scannetv2_train.txt \
#     --split_type train \
#     --processed_data_path data/processed_data/ScanNet \
#     --output_dir data/processed_data/ScanNet/qa/vstibench \
#     --dataset scannet \
#     --question_template VSTI_OBJ_OBJ_REL_POS_UD_TEMPLATE \
#     --question_type obj_obj_relative_pos_ud \
#     --output_filename_prefix qa_obj_obj_relative_pos_ud \
#     --num_subsample 120 \
#     --num_workers 64 