#!/bin/bash

# 设置工作目录
cd "$(dirname "$0")/../.."

# 执行 Python 模块
python -m src.tasks.vstibench.get_obj_obj_rel_pos_nf_qa \
    --split_path datasets/ScanNet/Tasks/Benchmark/scannetv2_val.txt \
    --split_type val \
    --processed_data_path data/processed_data/ScanNet \
    --output_dir data/processed_data/ScanNet/qa/vstibench \
    --dataset scannet \
    --question_template VSTI_OBJ_OBJ_REL_POS_NF_TEMPLATE \
    --question_type obj_obj_relative_pos_nf \
    --output_filename_prefix qa_obj_obj_relative_pos_nf \
    --num_subsample 6 \
    --num_workers 64 

# # 执行 Python 模块 (Near/Far)
# python -m src.tasks.vstibench.get_obj_obj_rel_pos_nf_qa \
#     --split_path datasets/ScanNet/Tasks/Benchmark/scannetv2_train.txt \
#     --split_type train \
#     --processed_data_path data/processed_data/ScanNet \
#     --output_dir data/processed_data/ScanNet/qa/vstibench \
#     --dataset scannet \
#     --question_template VSTI_OBJ_OBJ_REL_POS_NF_TEMPLATE \
#     --question_type obj_obj_relative_pos_nf \
#     --output_filename_prefix qa_obj_obj_relative_pos_nf \
#     --num_subsample 120 \
#     --num_workers 64 

python -m src.format_qa \
    --input_path data/processed_data/ScanNet/qa/vstibench/val/qa_obj_obj_relative_pos_nf.json \
    --output_path data/processed_data/ScanNet/formatted_qa/vstibench/val/qa_obj_obj_relative_pos_nf.json
