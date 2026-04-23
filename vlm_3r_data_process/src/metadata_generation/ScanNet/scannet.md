# 1. preprocess

This section outlines the initial steps required to prepare the raw ScanNet data for further processing. It involves converting the raw scan data into usable formats like point clouds with labels and extracting video frames.

## 1.1. process scannet200 to get ply with label and instance id

This step processes the raw ScanNet scans (using the ScanNet200 benchmark annotations) to generate `.ply` files. These files contain the 3D point cloud data for each scene, along with semantic labels (what each point represents, e.g., 'wall', 'chair') and instance IDs (which specific object instance a point belongs to). This is crucial for tasks involving 3D scene understanding.

```bash
cd datasets/ScanNet/BenchmarkScripts/ScanNet200/
python preprocess_scannet200.py \
        --dataset_root ../../../../data/raw_data/scannet/scans \
        --output_root ../../../../data/processed_data/ScanNet/point_cloud \
        --label_map_file ../../../../data/raw_data/scannet/scannetv2-labels.combined.tsv \
        --num_workers 4
cd ../../../../
```

## 1.2. export video

Here, we extract video sequences from the raw ScanNet sensor streams (`.sens` files). Each scan originally contains RGB-D video data. This script converts it into standard video files (e.g., `.mp4` or image sequences) at a specified resolution and frame rate. These videos are useful for 2D analysis or multi-modal tasks combining 2D and 3D data.

```bash
python src/ScanNet/preprocess/export_video.py \
    --scans_dir data/raw_data/scannet/scans \
    --output_dir data/processed_data/ScanNet/videos \
    --train_val_splits_path datasets/ScanNet/Tasks/Benchmark \
    --width 640 \
    --height 480 \
    --fps 24 \
    --frame_skip 1 \
    --max_workers 32
```

## 1.3. sample frame data and camera intrinsics

From the extracted videos or raw sensor data, this step samples individual frames. Along with the image data for each sampled frame, it also extracts the corresponding camera pose (position and orientation) and camera intrinsics (focal length, principal point). This information is vital for linking the 2D frame content back to the 3D scene structure.

```bash
python -m src.ScanNet.preprocess.export_sampled_frames \
    --scans_dir data/raw_data/scannet/scans \
    --output_dir data/processed_data/ScanNet \
    --train_val_splits_path datasets/ScanNet/Tasks/Benchmark \
    --num_frames 32 \
    --max_workers 64 \
    --image_size 480 640
```

# 2. get scene metadata and frame metadata

After preprocessing, this section focuses on consolidating metadata about the scenes and the sampled frames into structured JSON files. This metadata facilitates easier data loading and querying.

## 2.1. get scene metadata

This step gathers high-level information about each ScanNet scene. It typically includes details like the scene ID, paths to the processed point cloud and video files, object counts per category, and potentially other scene-level statistics derived from the processed data. Separate files are generated for the training and validation splits.

### 2.1.1 train

```bash
python -m src.ScanNet.scannet_metadata \
    --input_dir data/processed_data/ScanNet/point_cloud/train \
    --scene_list_file datasets/ScanNet/Tasks/Benchmark/scannetv2_train.txt \
    --save_dir data/processed_data/ScanNet/metadata/train \
    --output_filename scannet_metadata_train.json \
    --label_mapping_path data/raw_data/scannet/scannetv2-labels.combined.tsv \
    --video_dir ScanNet/videos/train \
    --num_workers 64 \
    --overwrite
```

### 2.1.2 val

```bash
python -m src.ScanNet.scannet_metadata \
    --input_dir data/processed_data/ScanNet/point_cloud/val \
    --scene_list_file datasets/ScanNet/Tasks/Benchmark/scannetv2_val.txt \
    --save_dir data/processed_data/ScanNet/metadata/val \
    --output_filename scannet_metadata_val.json \
    --label_mapping_path data/raw_data/scannet/scannetv2-labels.combined.tsv \
    --video_dir ScanNet/videos/val \
    --num_workers 64 \
    --overwrite
```

## 2.2. get frame metadata

Similar to scene metadata, this step compiles detailed information for each sampled frame. This typically includes the frame ID, the scene it belongs to, the path to the image file, the corresponding camera pose, and camera intrinsics. This allows for efficient access to specific frames and their associated spatial information. Again, separate files are generated for training and validation splits.

### 2.2.1 train

```bash
python -m src.ScanNet.scannet_frame_metadata \
    --processed_dir "data/processed_data/ScanNet" \
    --scene_list_file "datasets/ScanNet/Tasks/Benchmark/scannetv2_train.txt" \
    --save_dir "data/processed_data/ScanNet/metadata/train" \
    --output_filename "scannet_frame_metadata_train.json" \
    --num_workers "64" \
    --overwrite \
    --random_seed "42"
```

### 2.2.2 val

```bash
python -m src.ScanNet.scannet_frame_metadata \
    --processed_dir "data/processed_data/ScanNet" \
    --scene_list_file "datasets/ScanNet/Tasks/Benchmark/scannetv2_val.txt" \
    --save_dir "data/processed_data/ScanNet/metadata/val" \
    --output_filename "scannet_frame_metadata_val.json" \
    --num_workers "64" \
    --overwrite \
    --random_seed "42"
```

# 3. generate qa

This section details the process of generating Question-Answering (QA) pairs based on the processed ScanNet data, likely for training vision-language models or evaluating scene understanding capabilities.

```bash
# This script likely uses the metadata and potentially the 3D/2D data
# to automatically generate questions about the scenes and their corresponding answers.
# --num_subsample might indicate using only a subset of scenes/frames.
# --meta_info_path points to a file containing necessary information for QA generation.
python -m stage2_data.tasks.all_generate \
    --split_path ScanNet/Tasks/Benchmark/scannetv2_train.txt \
    --num_subsample 10000 \
    --meta_info_path data/scannet_vsibench/all_meta_info.json \
    --output_dir data/scannet_vsibench \
    --tag 03_25_2025 \
    --dataset scannet \
    --base_dir stage2_data/tasks
```

## format qa

After generation, the QA data might need reformatting into a specific structure (e.g., a standardized JSON format) suitable for model training or evaluation frameworks.

```bash
# This script takes the raw generated QA data and formats it.
# The --tag likely corresponds to the generation run.
python -m stage2_data.format_qa \
--input_dir data/scannet_vsibench/03_26 \
--output_dir data/scannet_vsibench \
--tag 03_26_2025
```
