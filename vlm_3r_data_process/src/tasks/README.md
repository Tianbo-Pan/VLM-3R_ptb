# Stage 2 QA Generation Tasks

This directory contains Python scripts for generating Question-Answering (QA) pairs for various spatio-temporal reasoning tasks based on scene metadata.

## Structure

-   `base_qa_generator.py`: Defines the `BaseQAGenerator` abstract base class, which handles common functionalities like argument parsing, data loading, scene iteration, subsampling, and saving results.
-   `vsibench/`: Directory containing task scripts inheriting from `BaseQAGenerator` for VSI (Video Scene Instance) benchmark tasks (Static scene understanding). Examples include:
    -   `get_obj_count_qa.py`: Counts objects of a specific category.
    -   `get_obj_abs_distance_qa.py`: Calculates absolute distance between two unique objects.
    -   `get_obj_rel_distance_qa.py`: Determines which object (from choices) is closest to a target object.
    -   `get_obj_rel_direction_v1_qa.py`/`v2`/`v3`: Determines relative direction between three unique objects based on different spatial partitions (front-left/right, back-left/right; left/right/back; left/right).
    -   `get_obj_size_qa.py`: Estimates the size (longest dimension) of a unique object.
    -   `get_room_size_qa.py`: Estimates the area of the room.
    -   `get_spatial_temporal_appearance_order_qa.py`: Determines the order in which objects first appear in the video.
-   `vsdrbench/`: Directory containing task scripts inheriting from `BaseQAGenerator` for VSDR (Video Scene Dynamic Reasoning) benchmark tasks. Examples include:
    -   `get_camera_displacement_qa.py`: Estimates the Euclidean distance the camera traveled between two frames.
    -   `get_cam_obj_abs_dist_qa.py`: Estimates the distance between the camera and a specific object instance in a given frame (using sampled points).
    -   `get_cam_obj_dist_change_qa.py`: Estimates the change in distance between the camera and an object instance over a time interval (using sampled points).
    -   `get_cam_obj_rel_dir_qa.py`: Determines the direction of an object relative to the camera's viewing direction in a frame (using bounding box checks for ambiguity).
    -   `get_cam_obj_rel_dist_qa.py`: Determines which object (from unique instances) is closest to the camera in a frame (using sampled points).
    -   `get_occlusion_qa.py` *(New)*: Asks which object (from choices) is occluded by a specified object in a given frame, using 2D bounding box overlap and 3D distance heuristics.
    -   `get_cam_motion_trans_qa.py` *(New/Modified Logic)*: Identifies sequences where the camera exhibits **consistent translational motion** (Forward, Backward, Left, Right, Up, or Down) relative to its **initial orientation**. It first checks for directional consistency during the sequence before determining the overall primary direction based on accumulated displacement projections.
    -   `get_cam_motion_rot_qa.py` *(New/Modified Logic)*: Identifies sequences where the camera exhibits **consistent rotational motion** (Turning Left/Right, Tilting Up/Down, or Rolling Left/Right) relative to its **initial orientation**. It first checks for consistency in the primary axis of rotation during the sequence before determining the overall rotation type and direction based on the total relative rotation. (*Requires scipy*).
-   `all_generate.py`: A script to run multiple QA generation tasks sequentially (can be modified to run specific tasks).

## Usage

Each task script can be run independently from the command line.

**Common Arguments (Handled by BaseQAGenerator):**

-   `--split_path`: Path to the scene split file (e.g., `../data/splits/scannet_10.txt`).
-   `--processed_data_path`: Base directory containing processed data (metadata, color, depth, point clouds, etc.). Paths to specific data types are derived from this.
-   `--dataset`: Name of the dataset (e.g., `scannetpp`, `replica`, `scannet`). Used to construct data paths.
-   `--split_type`: Type of the data split (e.g., `train`, `val`, `test`). Used to construct data paths.
-   `--output_dir`: Directory to save the output QA JSON file (e.g., `../data/qa_pairs`). The script will create a subdirectory based on `split_type`.
-   `--question_template`: Name of the question template constant from `src.question_templates`. (Usually set by default within the script).
-   `--num_subsample`: Number of questions to subsample per scene (default: 6, can be overridden by task or in `all_generate.py`).
-   `--num_workers`: Number of parallel processes to use for scene processing (default: 1).

**Running a Single Task:**

Navigate to the project's root directory or ensure `src` is in your Python path. Use `python -m <module_path>` to run scripts as modules.

```bash
# Example: Run object counting QA generation (VSI benchmark)
python -m src.tasks.vsibench.get_obj_count_qa \
    --processed_data_path data/processed/scannetpp \
    --dataset scannetpp \
    --split_type val \
    --split_path data/splits/scannetpp_val.txt \
    --output_dir data/qa_output \
    --num_workers 4

# Example: Run consistent camera translation QA generation (VSDR benchmark)
python -m src.tasks.vsdrbench.get_cam_motion_trans_qa \
    --processed_data_path data/processed/replica \
    --dataset replica \
    --split_type train \
    --split_path data/splits/replica_train.txt \
    --output_dir data/qa_output \
    --num_workers 8
```

Each script will generate a JSON file named like `<output_filename_prefix>.json` (defined in the script's get_default_args) inside the `output_dir/split_type/` directory.

**Running All Task:**

Modify and run `all_generate.py` to execute multiple generation scripts. Ensure you provide the necessary common arguments and that the desired task scripts are included in its `script_configs` dictionary.

```bash
python -m src.tasks.all_generate \
    --split_path datasets/ScanNet/Tasks/Benchmark/scannetv2_train.txt \
    --split_type train \
    --processed_data_path data/processed_data/ScanNet \
    --dataset scannet \
    --num_subsample 10000 \
    --output_dir data/qa_output \
    --num_workers 64 # Add other arguments as needed
```