#!/bin/bash


python -m src.tasks.vstibench.get_camera_movement_direction_qa_v3 \
    --split_path datasets/ScanNet/Tasks/Benchmark/scannetv2_val.txt \
    --split_type val \
    --processed_data_path data/processed_data/ScanNet \
    --output_dir data/processed_data/ScanNet/qa/vstibench \
    --dataset scannet \
    --question_template VSTI_CAMERA_MOVEMENT_DIRECTION_TEMPLATE_V3 \
    --question_type camera_movement_direction \
    --output_filename_prefix qa_camera_movement_direction_v3 \
    --num_subsample 1 \
    --num_workers 64 