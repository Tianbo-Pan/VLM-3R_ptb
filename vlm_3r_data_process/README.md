# VLM 3R Data Processing

This document outlines the steps for processing VLM 3R data, focusing on datasets with ground truth annotations.

## Table of Contents

- [Quick Start: ScanNet Preprocessing Workflow](#quick-start-scannet-preprocessing-workflow)
- [1. Datasets with Ground Truth Annotations](#1-datasets-with-ground-truth-annotations)
- [2. Data Preprocessing Details](#2-data-preprocessing-details)
- [3. Downstream Task Generation (QA)](#3-downstream-task-generation-qa)

## Quick Start: ScanNet Preprocessing Workflow

We provide a complete, end-to-end workflow for processing the ScanNet dataset, from raw data to the final QA pairs. This includes detailed steps and command-line examples for each stage of the process. For the full guide, please refer to the documentation at [`src/metadata_generation/ScanNet/scannet.md`](./src/metadata_generation/ScanNet/scannet.md).

## 1. Datasets with Ground Truth Annotations

### Downloading Raw Data

Follow the instructions from the respective repositories to download the raw data:

* **ScanNet:** Download data to `data/raw_data/scannet`. Follow instructions at [https://github.com/ScanNet/ScanNet](https://github.com/ScanNet/ScanNet).
* **ScanNet++:** Download data to `data/raw_data/scannetpp`. Follow instructions at [https://github.com/scannetpp/scannetpp](https://github.com/scannetpp/scannetpp).
* **ARKitScenes:** Download data to `data/raw_data/arkitscenes`. Follow instructions at [https://github.com/apple/ARKitScenes](https://github.com/apple/ARKitScenes).

## 2. Data Preprocessing Details

This section outlines the general pipeline for preprocessing 3D scene data for tasks. The goal is to extract structured scene-level and frame-level metadata from raw inputs, which can then be used to generate diverse question-answering (QA) datasets.

### Input Data Requirements

The preprocessing pipeline requires the following types of data for each scene:

1.  **Point Cloud (PCD):** A 3D point cloud representation of the scene (e.g., from `.ply` or `.pcd` files). Each point should ideally be associated with:
    * Semantic label ID (indicating the object category).
    * Instance label ID (distinguishing individual objects of the same category).
    * Coordinates (x, y, z).
    * Color (R, G, B).
2.  **Video:** A video recording capturing a traversal through the scene (e.g., `.mp4` format).
3.  **Sampled Frame Data:** A collection of frames sampled from the video or reconstruction sequence, including:
    * **Color Images:** RGB images corresponding to sampled viewpoints.
    * **Depth Maps:** Per-pixel depth information for each sampled frame.
    * **Instance Masks:** Segmentation masks identifying object instances within each frame.
    * **Camera Poses:** The 6DoF camera pose (rotation and translation, typically as a 4x4 matrix) for each sampled frame, defining its position and orientation in the world coordinate system (often aligned with the point cloud).
4.  **Camera Intrinsics:** The intrinsic parameters of the camera used to capture the video/frames (focal length `fx`, `fy`; principal point `cx`, `cy`).

*(Note: Specific scripts or tools might be needed to generate the Sampled Frame Data from the raw Video and Point Cloud if not already available.)*

### Preprocessing Pipeline

The core preprocessing involves generating two types of metadata files:

1.  **Scene Metadata Generation:**
    * Processes the input Point Cloud.
    * Extracts scene-wide information such as:
        * Overall scene dimensions (e.g., room area).
        * Scene center coordinates.
        * Counts of different object categories present.
        * 3D bounding boxes (including center, size, and orientation) for each object instance.
    * Typically saves this information in a structured format like JSON (e.g., `scene_metadata.json`).

2.  **Frame Metadata Generation:**
    * Processes the Sampled Frame Data (Color, Depth, Masks, Poses) and Camera Intrinsics.
    * Extracts frame-specific information such as:
        * Camera pose for each frame.
        * 2D bounding boxes for object instances visible in each frame.
    * Typically saves this information, grouped by scene, in a structured format like JSON (e.g., `frame_metadata.json`).

### Output Metadata Format

The specific structure of the output metadata JSON files, based on the current implementation (e.g., `scannet_metadata.py`, `scannet_frame_metadata.py`), is as follows:

1.  **Scene Metadata (`scannet_metadata.json`):**
    A JSON file containing a dictionary where keys are scene IDs (e.g., "scene0000_00"). Each scene ID maps to a dictionary with the following structure:
    ```json
    {
      "scene_id": { // e.g., "scene0000_00"
        "video_path": "relative/or/absolute/path/to/scene_id.mp4",
        "dataset": "scannet", // Or other dataset identifier
        "room_size": 123.45, // Area in square units (calculated from point cloud)
        "room_center": [x, y, z], // List of floats for center coordinates
        "object_counts": {
          "category_name": count,
          // ...
        },
        "object_bboxes": {
          "category_name": [
            {
               "centroid": [cx, cy, cz], // Center coordinates
               "axesLengths": [lx, ly, lz], // Length of axes
               "normalizedAxes": [ // 3x3 rotation matrix as flattened list or list of lists
                 r11, r12, r13,
                 r21, r22, r23,
                 r31, r32, r33
               ],
               "min": [min_x, min_y, min_z], // Minimum corner of the axis-aligned bounding box
               "max": [max_x, max_y, max_z], // Maximum corner of the axis-aligned bounding box
               "instance_id": N // Instance ID from the dataset
            },
            // ... other instances of this category
          ],
          // ... other categories
        }
      },
      // ... other scene IDs
    }
    ```

2.  **Frame Metadata (`scannet_frame_metadata.json`):**
    A JSON file containing a dictionary where keys are scene IDs. Each scene ID maps to a dictionary with frame-level information:
    ```json
    {
      "scene_id": { // e.g., "scene0000_00"
        "camera_intrinsics": {
          "fx": float,
          "fy": float,
          "cx": float,
          "cy": float
        },
        "img_width": 640, // Image width used during processing
        "img_height": 480, // Image height used during processing
        "frames": [
          {
            "frame_id": 0, // Integer frame index/number from sampled data
            "file_path_color": "color/scene_id/000000.jpg", // Relative path to color image within processed dir
            "file_path_depth": "depth/scene_id/000000.png", // Relative path to depth map within processed dir
            "camera_pose_camera_to_world": [ // 4x4 matrix as list of lists
              [r11, r12, r13, tx],
              [r21, r22, r23, ty],
              [r31, r32, r33, tz],
              [0.0, 0.0, 0.0, 1.0]
            ],
            "bboxes_2d": [
              {
                "instance_id": 0, // Instance ID from the dataset
                "bbox_2d": [xmin, ymin, xmax, ymax] // Integer pixel coordinates
              },
              // ... other detected instances in this frame
            ]
          },
          // ... other frames in this scene
        ]
      },
      // ... other scene IDs
    }
    ```

## 3. Downstream Task Generation (QA)

This section details the Question-Answering (QA) tasks. We first introduce VSI-Bench, a benchmark from a published paper, and then describe our implemented VSTIbench tasks.

### VSI-Bench Task Details (from Paper "Thinking in Space")

This section details the QA tasks in the **VSI-Bench** benchmark from "Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces". The benchmark has over 5,000 QA pairs from 288 real-world indoor videos, categorized into Configurational, Measurement Estimation, and Spatiotemporal tasks. We extend our gratitude to the VSI-Bench team for providing the original data generation scripts, which served as a significant reference for our work.

**Task Summary Table (VSI-Bench)**

| Task Name           | Task Category     | Answer Type     |
| :------------------ | :---------------- | :-------------- |
| Object Count        | Configurational   | Numerical       |
| Relative Distance   | Configurational   | Multiple Choice |
| Relative Direction  | Configurational   | Multiple Choice |
| Route Plan          | Configurational   | Multiple Choice |
| Object Size         | Measurement       | Numerical       |
| Absolute Distance   | Measurement       | Numerical       |
| Room Size           | Measurement       | Numerical       |
| Appearance Order    | Spatiotemporal    | Multiple Choice |

The generated scene and frame metadata serve as input for generating these QA pairs.

#### Configurational Tasks (VSI-Bench)

These tasks test a model's understanding of the spatial layout and relationships.

* **Object Count:** Asks for the total number of instances of a specific object category (e.g., "How many chairs are there?").
    * **QA Generation (`get_obj_count_qa.py`):** This task generates multiple-choice questions based on `scene_metadata.json`. It iterates through the `object_counts` for each object type in the scene. For categories with more than one instance, it formulates a question. The correct answer is the actual count from the metadata. The other three distractor options are generated by adding small, random offsets to the correct answer, creating a four-choice question.
* **Relative Distance:** Asks which of four candidate objects is closest to a target object (e.g., "...which of these objects...is the closest to the printer?").
    * **QA Generation (`get_obj_rel_distance_v1_qa.py`, `...v2_qa.py`, `...v3_qa.py`):** This task generates multiple-choice questions asking which of several candidate objects is closest to a "primary" object.
        *   **Object Selection:** The "primary" object (the reference for distance) is chosen from unique object categories (with only one instance in the scene) to ensure clarity. Candidate objects are selected from the remaining objects. V1 provides 4 choices, V2 provides 3, and V3 provides 2.
        *   **Distance Calculation:** The script calculates the minimum Euclidean distance between the 3D bounding boxes of the primary object and each candidate object.
        *   **Ambiguity Filtering:** To create clear questions, the script filters out cases where: 1) The distances from any two candidate objects to the primary object are too close to each other (e.g., within 15-30 cm, depending on room size), making them hard to distinguish. 2) The closest candidate is already touching or extremely close to the primary object (less than 15 cm).
        *   **Answer:** The category of the candidate object closest to the primary object is the correct answer.
* **Relative Direction:** Asks for the direction of a 'querying object' relative to an observer's position and orientation (e.g., "If I am standing by the refrigerator and facing the sofa, is the kettle to my left, right, or back?").
    * **QA Generation (`get_obj_rel_direction_v1_qa.py`, `...v2_qa.py`, `...v3_qa.py`):** This task generates multiple-choice questions of the form, "If I am standing at object A and facing object B, in which direction is object C?"
        *   **Object Selection:** The script selects three **distinct** and **unique** (instance count of 1) objects from the scene to serve as the observer's position (A), the orientation reference (B), and the queried target (C).
        *   **Direction Calculation:** The script performs calculations on a 2D plane. It sets object A as the origin and defines the vector from A to B as "forward". It then calculates the angle between this "forward" vector and the vector from A to C.
        *   **Answer Options:** Based on the angle, the script categorizes object C into different directions. There are several versions:
            *   **V1:** Four diagonal categories (front-left, front-right, back-left, back-right).
            *   **V2:** Three main directions (left, right, back).
            *   **V3:** Two relative directions (left, right).
        *   **Ambiguity Filtering:** To ensure clarity, the script excludes cases where the objects are inappropriately spaced (too close or too far) or where the target object C lies on the boundary between two direction categories (e.g., exactly to the left).
* **Route Plan:** Asks the model to fill in missing turn commands in a navigation sequence (e.g., "...Fill in this route: 1. Go forward until the washing machine 2. [?]...").
    * **QA Generation:** (Implementation in progress)

#### Measurement Estimation Tasks (VSI-Bench)

These tasks require estimating quantitative spatial properties.

* **Object Size:** Asks for the length of the longest dimension of an object in centimeters (e.g., "What is the length of the longest dimension...of the refrigerator in centimeters?").
    * **QA Generation (`get_obj_size_qa.py`):** This task generates multiple-choice questions about the size of unique objects (instance count of 1) in the scene.
        *   **Answer Calculation:** The script reads the three `axesLengths` of the object's 3D bounding box from the metadata, finds the maximum value, and converts it from meters to centimeters to serve as the correct answer.
        *   **Option Generation:** The other three distractor options are created by generating random numbers within a certain percentage range (e.g., 40% to 180%) of the correct answer, forming a four-choice question.
* **Absolute Distance:** Asks for the Euclidean distance between two objects in meters (e.g., "...what is the distance between the bed and the sofa in meters?").
    * **QA Generation (`get_obj_abs_distance_qa.py`):** This task generates multiple-choice questions about the absolute distance between pairs of unique objects (instance count of 1) in the scene.
        *   **Object Selection & Distance Calculation:** The script iterates through all unique objects in the scene and calculates the minimum Euclidean distance between the 3D bounding boxes for each pair. It skips pairs that are closer than 0.2 meters.
        *   **Answer & Options:** The calculated distance (in meters, rounded to one decimal place) is the correct answer. The other three distractor options are created by generating random numbers near the correct answer.
* **Room Size:** Asks for the estimated area of the room in square meters (e.g., "What is the size of this room (in square meters)?").
    * **QA Generation (`get_room_size_qa.py`):** This task generates one multiple-choice question per scene about its total area.
        *   **Answer Calculation:** The script directly reads the `room_size` value from `scene_metadata.json` and rounds it to one decimal place to use as the correct answer.
        *   **Option Generation:** The other three distractor options are created by generating random numbers near the correct answer.

#### Spatiotemporal Task (VSI-Bench)

This task tests processing spatial information over time.

* **Appearance Order:** Asks for the order in which four object categories first appear in the video (e.g., "What will be the first-time appearance order of the following categories...: basket, printer, refrigerator, kettle?").
    * **QA Generation (`get_spatial_temporal_appearance_order_qa.py`):** This task generates multiple-choice questions about the first-appearance order of four object categories in the video.
        *   **First Appearance Time Calculation:** The script first iterates through the time-sorted `frame_metadata.json` to find the first frame number where each object instance appears. It then maps the instance ID to its category using `scene_metadata.json` to determine the first appearance frame number (and converts it to seconds) for each **category**.
        *   **Object Selection & Ambiguity Filtering:** The script randomly selects four categories from all that appear in the scene. To ensure a clear answer, it filters out combinations where the first appearance times are too close (e.g., less than 1 second apart).
        *   **Answer & Option Generation:**
            *   The **correct answer** is the list of the four categories sorted in ascending order of their first appearance time.
            *   The four categories are mentioned in a **random order** in the question to avoid giving clues.
            *   **Distractor options** are generated by creating other random permutations of the four categories.

### VSTIbench

The generated `scene_metadata.json` and `frame_metadata.json` files serve as the primary input for the various task-specific QA generation scripts located in the `src/tasks` directory. Each script reads the metadata and formulates questions and answers related to its specific focus.

**Task Summary Table (VSTIbench)**

| Task Name                                    | Task Level     | Answer Type                     |
| :------------------------------------------- | :------------- | :----------------------------   |
| Camera-Object Absolute Distance QA           | Frame-Level    | Numerical                       |
| Camera-Object Relative Distance QA (V1, V2, V3)| Frame-Level    | Multiple Choice                 |
| Object-Object Relative Position QA (Near/Far, Left/Right, Up/Down)| Frame-Level    | Multiple Choice                 |
| Camera Displacement QA                       | Sequence-Level | Numerical                       |
| Camera Movement Direction QA (V1, V2, V3)    | Sequence-Level | Multiple Choice                 |

#### Frame-Level QA Tasks

These tasks generate questions based on the information available within a single frame or the scene's point cloud.

* **Camera-Object Absolute Distance QA (`get_cam_obj_abs_dist_qa.py`)**: Generates frame-level questions asking for the approximate Euclidean distance (in meters) between the camera's position in a specific frame and the **closest point on the 3D bounding box** of a target unique object instance, expecting a **numerical answer**.
    * **Method:** Calculates the 3D Euclidean distance between the camera's world coordinates (from frame pose) and the closest point on the 3D bounding box of the object (derived from scene metadata).
    * *Example Question (Template: `VSTI_CAMERA_OBJ_DIST_TEMPLATE`)*: "What is the approximate distance (in meters) between the camera (or the person filming) and the nearest point of the nightstand in frame 2 of 32?"
* **Camera-Object Relative Distance QA (V1, V2, V3)**: Generates frame-level multiple-choice questions asking which of several candidate object instances is closest to the camera in a specific frame, measuring from the closest point of each object. This task comes in three variations, differing by the number of choices provided:
    * **V1 (`get_cam_obj_rel_dist_qa_v1.py`)**: 4 multiple-choice options. (Template: `VSTI_CAMERA_OBJ_REL_DIST_TEMPLATE_V1`)
    * **V2 (`get_cam_obj_rel_dist_qa_v2.py`)**: 3 multiple-choice options. (Template: `VSTI_CAMERA_OBJ_REL_DIST_TEMPLATE_V2`)
    * **V3 (`get_cam_obj_rel_dist_qa_v3.py`)**: 2 multiple-choice options. (Template: `VSTI_CAMERA_OBJ_REL_DIST_TEMPLATE_V3`)
    * **Method:** Calculates the 3D Euclidean distance between the camera's position (from frame pose) and the **closest point on the 3D bounding box** of each candidate object instance (derived from scene metadata). The candidate object with the minimum distance to the camera is the correct answer.
    * *Example Question (V2, 3 options)*: "Measuring from the closest point of each object, which of these objects (chair, sofa, lamp) is the closest to the camera in frame 10 of 32?"
    * *Example Options:* A. chair, B. sofa, C. lamp
    * *Example Answer:* A (if the chair is the closest to the camera among the options)
* **Object-Object Relative Position QA (Near/Far, Left/Right, Up/Down)**: Generates frame-level, 2-choice multiple-choice questions asking for the spatial relationship between two unique object instances within a frame. This task is split into three separate QA generation scripts, one for each spatial dimension:
    * **Near/Far (`get_obj_obj_rel_pos_nf_qa.py`)**: Asks if object A is nearer or farther than object B, relative to the camera. (Template: `VSTI_OBJ_OBJ_REL_POS_NF_TEMPLATE`)
    * **Left/Right (`get_obj_obj_rel_pos_lr_qa.py`)**: Asks if object A is to the left or right of object B. (Template: `VSTI_OBJ_OBJ_REL_POS_LR_TEMPLATE`)
    * **Up/Down (`get_obj_obj_rel_pos_ud_qa.py`)**: Asks if object A is up or down relative to object B. (Template: `VSTI_OBJ_OBJ_REL_POS_UD_TEMPLATE`)
    * **Method:** Transforms the 3D bounding box vertices of both unique objects into the camera's coordinate system for the given frame.
        *   For Near/Far: Compares the Z-coordinates (depth). If object A's maximum Z is less than object B's minimum Z (by a threshold), A is 'Near'. If object A's minimum Z is greater than object B's maximum Z (by a threshold), A is 'Far'.
        *   For Left/Right: Compares the X-coordinates. If object A's maximum X is less than object B's minimum X (by a threshold), A is 'Left'. If object A's minimum X is greater than object B's maximum X (by a threshold), A is 'Right'.
        *   For Up/Down: Compares the Y-coordinates (+Y is typically Down). If object A's maximum Y is less than object B's minimum Y (by a threshold), A is 'Up' (less Y). If object A's minimum Y is greater than object B's maximum Y (by a threshold), A is 'Down' (more Y).
        *   Uses only pairs where one object is entirely nearer/farther, left/right, or up/down than the other.
    * *Example Questions*:
        *   (Near/Far): "In frame 5 of 32, relative to the camera, is the chair Near or Far compared to the table?"
        *   (Left/Right): "In frame 12 of 32, relative to the sofa, is the lamp to the Left or Right?"
        *   (Up/Down): "In frame 20 of 32, relative to the desk, is the monitor Up or Down?"
    * *Example Options (for one question type, e.g., Near/Far):* A. Near, B. Far
    * *Example Answer (for the Near/Far example):* A (if the chair was entirely closer to the camera than the table)

#### Sequence-Level QA Tasks

These tasks generate questions based on information aggregated across a sequence of frames.

* **Camera Displacement QA (`get_camera_displacement_qa.py`)**: Generates sequence-level questions asking for the approximate Euclidean distance the camera traveled between two specified frames, expecting a **numerical answer**.
    * **Method:** Calculates the Euclidean distance between the camera's world positions (translation components) at the start and end frames of the specified sequence using their camera_pose_camera_to_world metadata.
    * *Example Question:* "Approximately how far (in meters) did the camera move between frame 10 and frame 30 of 32?"
    * *Example Answer:* 1.8 (if the calculated distance between camera positions was 1.8 meters)
* **Camera Movement Direction QA (V1, V2, V3)**: Generates sequence-level multiple-choice questions asking about the primary direction of the camera's translation (movement) during a specified sequence, relative to its starting orientation. This task is split into three variations based on the number of choices:
    * **V1 (`get_camera_movement_direction_qa_v1.py`)**: 4 multiple-choice options (Forward, Backward, Left, Right). (Template: `VSTI_CAMERA_MOVEMENT_DIRECTION_TEMPLATE_V1`)
    * **V2 (`get_camera_movement_direction_qa_v2.py`)**: 3 multiple-choice options. (Template: `VSTI_CAMERA_MOVEMENT_DIRECTION_TEMPLATE_V2`)
    * **V3 (`get_camera_movement_direction_qa_v3.py`)**: 2 multiple-choice options. (Template: `VSTI_CAMERA_MOVEMENT_DIRECTION_TEMPLATE_V3`)
    * **Method:** Compares the camera poses (`camera_pose_camera_to_world`) of the **start and end frames** for a given sequence. It calculates the displacement vector between these two points, transforms this vector into the starting frame's local coordinate system, and then determines the primary direction of movement (e.g., Forward, Backward, Left, Right). Note that this method considers only the net displacement, not the full path taken between the frames.
    * *Example Question (V2, 3 options):* "During the sequence from frame 5 to frame 20 of 32, what was the primary direction of the camera's movement?"
    * *Example Options:* A. Forward, B. Left, C. Right
    * *Example Answer:* A (if the camera primarily moved forward)