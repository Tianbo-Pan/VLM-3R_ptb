#!/bin/bash

# 设置工作目录
cd "$(dirname "$0")/../.."

# 执行 Python 模块
python -m src.tasks.vstibench.get_obj_obj_rel_pos_lr_qa \
    --split_path datasets/ScanNet/Tasks/Benchmark/scannetv2_val.txt \
    --split_type val \
    --processed_data_path data/processed_data/ScanNet \
    --output_dir data/processed_data/ScanNet/qa/vstibench \
    --dataset scannet \
    --question_template VSTI_OBJ_OBJ_REL_POS_LR_TEMPLATE \
    --question_type obj_obj_relative_pos_lr \
    --output_filename_prefix qa_obj_obj_relative_pos_lr \
    --num_subsample 3 \
    --num_workers 64 

# # 执行 Python 模块
# python -m src.tasks.vstibench.get_obj_obj_rel_pos_lr_qa \
#     --split_path datasets/ScanNet/Tasks/Benchmark/scannetv2_train.txt \
#     --split_type train \
#     --processed_data_path data/processed_data/ScanNet \
#     --output_dir data/processed_data/ScanNet/qa/vstibench \
#     --dataset scannet \
#     --question_template VSTI_OBJ_OBJ_REL_POS_LR_TEMPLATE \
#     --question_type obj_obj_relative_pos_lr \
#     --output_filename_prefix qa_obj_obj_relative_pos_lr \
#     --num_subsample 60 \
#     --num_workers 64 

# # format qa
# python -m src.format_qa \
#     --input_path data/processed_data/ScanNet/qa/vstibench/val/qa_obj_obj_relative_pos_lr.json \
#     --output_path data/processed_data/ScanNet/formatted_qa/vstibench/val/qa_obj_obj_relative_pos_lr.json
