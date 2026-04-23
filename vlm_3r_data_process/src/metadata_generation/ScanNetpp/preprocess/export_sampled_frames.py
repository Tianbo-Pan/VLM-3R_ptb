# 1. get scene list from split_path
# 2. for each scene, get sampled frames from rendered_data_dir
# 3. save sampled frames to output_dir

import os
import shutil
import argparse
from tqdm import tqdm
import numpy as np
import json
import logging
import cv2 # Add OpenCV import

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Helper function to save numpy matrix to a text file
def save_matrix_to_file(matrix, filename):
    """Saves a numpy matrix to a text file."""
    with open(filename, 'w') as f:
        if matrix is not None:
            for row in matrix:
                f.write(' '.join(map(str, row)) + '\n')
        else:
            f.write('') # Write empty file if matrix is None

def save_point_cloud_to_ply(points, colors, filename, downsample_factor=0.1):
    """Saves a point cloud (with optional colors) to a PLY file."""
    if downsample_factor < 1.0:
        num_points = points.shape[0]
        indices = np.random.choice(num_points, size=int(num_points * downsample_factor), replace=False)
        points = points[indices]
        if colors is not None:
            colors = colors[indices]
    num_points = points.shape[0]
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {num_points}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if colors is not None:
        # Check color data type and range
        if colors.dtype == np.uint8:
            # Assume colors are already in 0-255 range
            pass
        elif colors.dtype == np.float32 or colors.dtype == np.float64:
             # Assume colors are in 0.0-1.0 range, scale to 0-255
             logging.info("Detected float colors, scaling to 0-255 for PLY.")
             colors = (colors * 255).clip(0, 255).astype(np.uint8)
        else:
            logging.warning(f"Unsupported color data type {colors.dtype}. Attempting to cast to uint8.")
            try:
                colors = colors.astype(np.uint8)
            except ValueError:
                logging.error("Failed to cast colors to uint8. Saving without color.")
                colors = None # Fallback to saving without color

    if colors is not None:
        if colors.shape[0] != num_points or colors.shape[1] != 3:
             logging.error(f"Color array shape mismatch ({colors.shape}). Expected ({num_points}, 3). Saving without color.")
             colors = None # Ensure colors match points or discard
        else:
            header.extend([
                "property uchar red",
                "property uchar green",
                "property uchar blue",
            ])
    header.append("end_header")

    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True) # Ensure directory exists
        with open(filename, 'w') as f:
            f.write("\n".join(header) + "\n")
            for i in range(num_points):
                line = f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}"
                if colors is not None:
                    # Colors should be uint8 0-255 at this point
                    line += f" {colors[i, 0]} {colors[i, 1]} {colors[i, 2]}"
                f.write(line + "\n")
        logging.info(f"Saved point cloud to {filename}")
    except Exception as e:
        logging.error(f"Failed to save PLY file {filename}: {e}")


def verify_reprojection(depth_path, intrinsic_path, output_ply_path, color_path=None, depth_scale=1000.0):
    """
    Loads a depth image and intrinsics, performs reprojection to a 3D point cloud,
    and saves it as a PLY file. Optionally includes color.

    Args:
        depth_path (str): Path to the depth image file (e.g., PNG).
        intrinsic_path (str): Path to the intrinsic matrix file (.txt).
        output_ply_path (str): Path where the output PLY file will be saved.
        color_path (str, optional): Path to the corresponding color image file. Defaults to None.
        depth_scale (float, optional): Scale factor to convert depth values to meters.
                                       Assumes depth is stored in millimeters if scale is 1000.0. Defaults to 1000.0.
    """
    logging.info(f"Starting reprojection verification for depth: {depth_path}")

    # Load depth image
    try:
        # Use cv2.IMREAD_ANYDEPTH for potentially higher bit depth images, fall back to unchanged
        try:
            depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        except: # If ANYDEPTH is not supported or fails, try standard UNCHANGED
             logging.warning("cv2.IMREAD_ANYDEPTH failed, trying cv2.IMREAD_UNCHANGED.")
             depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if depth_image is None:
            raise ValueError("Failed to load depth image.")
        # Convert to float for calculations, handle potential different input types
        depth_image = depth_image.astype(np.float32)
        logging.info(f"Loaded depth image {depth_path} with shape {depth_image.shape}, dtype after load: {depth_image.dtype}")
    except Exception as e:
        logging.error(f"Error loading depth image {depth_path}: {e}")
        return

    # Load intrinsic matrix
    try:
        K = np.loadtxt(intrinsic_path)
        if K.shape == (3, 3):
             fx = K[0, 0]
             fy = K[1, 1]
             cx = K[0, 2]
             cy = K[1, 2]
        elif K.shape == (4, 4): # Assume top-left 3x3 is the intrinsic matrix
             fx = K[0, 0]
             fy = K[1, 1]
             cx = K[0, 2]
             cy = K[1, 2]
             logging.warning(f"Loaded 4x4 matrix from {intrinsic_path}, using top-left 3x3 as intrinsics.")
        else:
             raise ValueError(f"Invalid intrinsic matrix shape: {K.shape}. Expected 3x3 or 4x4.")
        logging.info(f"Loaded intrinsics from {intrinsic_path}: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    except Exception as e:
        logging.error(f"Error loading intrinsic matrix {intrinsic_path}: {e}")
        return

    # Load color image if path provided
    colors_rgb = None
    color_image = None
    if color_path:
        try:
            color_image = cv2.imread(color_path)
            if color_image is None:
                raise ValueError("Failed to load color image.")
            if color_image.shape[:2] != depth_image.shape[:2]:
                 logging.warning(f"Color image {color_path} shape {color_image.shape[:2]} "
                                f"doesn't match depth image shape {depth_image.shape[:2]}. Color will not be used.")
                 color_image = None # Ignore color if shapes don't match
            else:
                 logging.info(f"Loaded color image {color_path} with shape {color_image.shape}")
        except Exception as e:
            logging.warning(f"Warning: Error loading color image {color_path}: {e}. Proceeding without color.")
            color_image = None

    # Perform reprojection
    height, width = depth_image.shape[:2]
    u = np.arange(width)
    v = np.arange(height)
    u, v = np.meshgrid(u, v)

    # Filter out invalid depth pixels (depth <= 0)
    valid_depth_mask = depth_image > 0
    depth_values = depth_image[valid_depth_mask]
    u_coords = u[valid_depth_mask]
    v_coords = v[valid_depth_mask]

    if depth_values.size == 0:
        logging.warning(f"No valid depth values (depth > 0) found in {depth_path}. Cannot create point cloud.")
        return

    # Convert depth to meters
    Z = depth_values / depth_scale
    # Calculate X and Y in camera coordinates
    X = (u_coords - cx) * Z / fx
    Y = (v_coords - cy) * Z / fy

    points = np.vstack((X, Y, Z)).T
    logging.info(f"Generated {points.shape[0]} points.")

    # Get corresponding colors if color image is valid
    if color_image is not None:
        # Extract colors (BGR format from cv2.imread)
        colors_bgr = color_image[valid_depth_mask]
        # Convert BGR to RGB for saving (PLY standard usually expects RGB)
        colors_rgb = colors_bgr[:, :, ::-1].copy() # Create RGB copy
        # colors_rgb should be uint8 [0-255] for PLY saving helper
        logging.info("Extracted corresponding colors (RGB).")


    # Ensure output directory exists (handled within save_point_cloud_to_ply)

    # Save to PLY
    try:
        save_point_cloud_to_ply(points, colors_rgb, output_ply_path)
    except Exception as e:
        # Error logging is handled within save_point_cloud_to_ply
        pass

def verify_scene_reprojection(scene_id, output_dir, split_name, verification_output_dir, depth_scale=1000.0):
    """
    Verifies the exported data for a scene by reprojecting all its sampled frames
    into a single world-coordinate point cloud.

    Args:
        scene_id (str): The ID of the scene to verify (e.g., 'scene0000_00').
        output_dir (str): The base directory where exported data was saved
                           (containing color/, depth/, pose/, intrinsic/ subdirs).
        split_name (str): The name of the split (e.g., 'train', 'val').
        verification_output_dir (str): Base directory to save the verification PLY file.
        depth_scale (float, optional): Scale factor to convert depth values to meters. Defaults to 1000.0.
    """
    logging.info(f"Starting verification for scene {scene_id} in split {split_name}")

    # Construct paths to the exported data for the scene
    scene_color_dir = os.path.join(output_dir, 'color', split_name, scene_id)
    scene_depth_dir = os.path.join(output_dir, 'depth', split_name, scene_id)
    scene_pose_dir = os.path.join(output_dir, 'pose', split_name, scene_id)
    intrinsic_path = os.path.join(output_dir, 'intrinsic', split_name, f'intrinsics_{scene_id}.txt')
    ply_output_path = os.path.join(verification_output_dir, split_name, f'{scene_id}_combined_pcd.ply')

    os.makedirs(os.path.dirname(ply_output_path), exist_ok=True)

    # Check if essential directories/files exist
    if not os.path.isdir(scene_depth_dir):
        logging.error(f"Exported depth directory not found: {scene_depth_dir}. Cannot verify.")
        return
    if not os.path.isdir(scene_pose_dir):
        logging.error(f"Exported pose directory not found: {scene_pose_dir}. Cannot verify.")
        return
    if not os.path.isfile(intrinsic_path):
        logging.error(f"Intrinsic file not found: {intrinsic_path}. Cannot verify.")
        return
    has_color = os.path.isdir(scene_color_dir)
    if not has_color:
        logging.warning(f"Color directory not found: {scene_color_dir}. Proceeding without color.")

    # Load intrinsic matrix
    try:
        K = np.loadtxt(intrinsic_path)
        if K.shape == (3, 3):
             fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        elif K.shape == (4, 4):
             fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
             logging.warning(f"Loaded 4x4 matrix from {intrinsic_path}, using top-left 3x3 as intrinsics.")
        else:
             raise ValueError(f"Invalid intrinsic matrix shape: {K.shape}. Expected 3x3 or 4x4.")
        logging.info(f"Loaded intrinsics for scene {scene_id}: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    except Exception as e:
        logging.error(f"Error loading intrinsic matrix {intrinsic_path}: {e}")
        return

    all_points_world = []
    all_colors_rgb = []

    # Iterate through frames (using depth files as reference)
    frame_files = sorted([f for f in os.listdir(scene_depth_dir) if f.endswith('.png')]) # Assuming depth is PNG
    if not frame_files:
        logging.warning(f"No depth files found in {scene_depth_dir}. Nothing to verify.")
        return

    logging.info(f"Processing {len(frame_files)} frames for scene {scene_id}...")

    for depth_filename in tqdm(frame_files, desc=f"Verifying {scene_id}"):
        frame_basename = os.path.splitext(depth_filename)[0]
        depth_path = os.path.join(scene_depth_dir, depth_filename)
        pose_path = os.path.join(scene_pose_dir, f'{frame_basename}.txt')
        # Assuming color has the same basename but .jpg extension (adjust if needed)
        color_path = os.path.join(scene_color_dir, f'{frame_basename}.jpg') if has_color else None

        # --- Load frame data ---
        # Load depth
        try:
            depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            if depth_image is None: raise ValueError("imread failed")
            depth_image = depth_image.astype(np.float32)
        except Exception as e:
            logging.warning(f"Skipping frame {frame_basename}: Error loading depth {depth_path}: {e}")
            continue

        # Load pose (Extrinsic: Camera-to-World)
        try:
            pose_matrix = np.loadtxt(pose_path)
            if pose_matrix.shape != (4, 4):
                raise ValueError(f"Invalid pose matrix shape {pose_matrix.shape}, expected (4, 4)")
        except Exception as e:
            logging.warning(f"Skipping frame {frame_basename}: Error loading pose {pose_path}: {e}")
            continue

        # Load color (optional)
        color_image_bgr = None
        if color_path and os.path.exists(color_path):
            try:
                color_image_bgr = cv2.imread(color_path)
                if color_image_bgr is None: raise ValueError("imread failed")
                if color_image_bgr.shape[:2] != depth_image.shape[:2]:
                    logging.warning(f"Color/depth shape mismatch for frame {frame_basename}. Ignoring color for this frame.")
                    color_image_bgr = None
            except Exception as e:
                logging.warning(f"Error loading color {color_path} for frame {frame_basename}: {e}. Ignoring color for this frame.")
                color_image_bgr = None
        # --- Reprojection --- 
        height, width = depth_image.shape[:2]
        u = np.arange(width)
        v = np.arange(height)
        u, v = np.meshgrid(u, v)

        valid_depth_mask = depth_image > 0
        depth_values = depth_image[valid_depth_mask]
        u_coords = u[valid_depth_mask]
        v_coords = v[valid_depth_mask]

        if depth_values.size == 0:
            # logging.debug(f"No valid depth for frame {frame_basename}") # Too verbose potentially
            continue

        Z_cam = depth_values / depth_scale
        X_cam = (u_coords - cx) * Z_cam / fx
        Y_cam = (v_coords - cy) * Z_cam / fy

        points_cam = np.vstack((X_cam, Y_cam, Z_cam)).T # Shape: (N, 3)

        # --- Transform to World Coordinates --- 
        points_cam_h = np.hstack((points_cam, np.ones((points_cam.shape[0], 1)))) # Shape: (N, 4)
        points_world_h = (pose_matrix @ points_cam_h.T).T # Shape: (N, 4)
        points_world = points_world_h[:, :3] # Shape: (N, 3)

        all_points_world.append(points_world)

        # --- Get Colors --- 
        if color_image_bgr is not None:
            colors_bgr_valid = color_image_bgr[valid_depth_mask]
            # Convert BGR to RGB
            colors_rgb_valid = colors_bgr_valid[:, ::-1].copy()
            all_colors_rgb.append(colors_rgb_valid)
        elif has_color: # If color dir exists but this frame failed loading/matching
             # Add placeholder (e.g., gray) if we want points even if color fails for some frames
             # Or, ensure all_colors_rgb has same number of entries as all_points_world if needed later
             # For simplicity now, only append if color is successfully loaded
             pass

    # --- Combine and Save --- 
    if not all_points_world:
        logging.warning(f"No points generated for scene {scene_id}. Cannot save PLY.")
        return

    final_points = np.concatenate(all_points_world, axis=0)
    final_colors = None
    if all_colors_rgb:
        # Check if number of color arrays matches number of point arrays processed
        # This basic check assumes that if color dir existed, we expect color for *every* frame with valid points
        # A more robust approach might pad missing colors or handle inconsistencies differently
        if len(all_colors_rgb) == len(all_points_world):
             try:
                final_colors = np.concatenate(all_colors_rgb, axis=0)
                if final_colors.shape[0] != final_points.shape[0]:
                     logging.error("Internal error: Mismatch between concatenated points and colors count. Saving without color.")
                     final_colors = None
             except ValueError as e:
                 logging.error(f"Error concatenating colors: {e}. Saving without color.")
                 final_colors = None
        else:
            logging.warning(f"Inconsistent number of color arrays ({len(all_colors_rgb)}) vs point arrays ({len(all_points_world)}). Saving without color.")


    logging.info(f"Saving combined point cloud for scene {scene_id} ({final_points.shape[0]} points) to {ply_output_path}")
    save_point_cloud_to_ply(final_points, final_colors, ply_output_path)

def export_sampled_frames(split_path, raw_data_dir, rendered_data_dir, output_dir, num_frames_to_sample, split_name):
    """
    Exports sampled frames (color, depth), poses, and intrinsics for scenes listed in the split file,
    matching the ScanNet output structure. Only samples from the 'train' split defined in train_test_lists.json.

    Args:
        split_path (str): Path to the file containing the list of scene IDs (one per line).
        raw_data_dir (str): Path to the directory containing raw scene data (including train_test_lists.json).
        rendered_data_dir (str): Path to the directory containing rendered data for all scenes.
                                Expected structure: rendered_data_dir/scene_id/dslr/...
        output_dir (str): Path to the base directory where output split folders will be saved.
                          Output structure: output_dir/[color|depth|pose|intrinsic]/split_name/scene_id/...
        num_frames_to_sample (int): Number of frames to uniformly sample per scene from the train list.
        split_name (str): Name of the split (e.g., 'train', 'val').
    """
    # Ensure base output directory exists (split directories created later)
    os.makedirs(output_dir, exist_ok=True)

    # Read scene list
    try:
        with open(split_path, 'r') as f:
            scenes = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Split file not found at {split_path}")
        return

    print(f"Found {len(scenes)} scenes in {split_path}")

    # Process each scene
    for scene_id in tqdm(scenes, desc="Exporting sampled frames"):
        logging.info(f"Processing scene: {scene_id}")

        # Define source directories
        camera_dir = os.path.join(rendered_data_dir, scene_id, 'dslr', 'camera')
        depth_dir = os.path.join(rendered_data_dir, scene_id, 'dslr', 'render_depth')
        color_dir = os.path.join(rendered_data_dir, scene_id, 'dslr', 'rgb_resized_undistorted')

        # Check if all source directories exist
        if not os.path.exists(camera_dir) or not os.path.exists(depth_dir) or not os.path.exists(color_dir):
            print(f"Warning: Missing source directory (camera, depth, or color) for scene {scene_id}. Skipping.")
            continue

        # --- Read train list from train_test_lists.json ---
        split_json_path = os.path.join(raw_data_dir, scene_id, 'dslr', 'train_test_lists.json')
        if not os.path.exists(split_json_path):
            logging.error(f"Train list JSON not found for scene {scene_id} at {split_json_path}. Skipping scene.")
            continue
        try:
            with open(split_json_path, 'r') as f:
                # Assuming the train list contains filenames like '000000.jpg', '000001.jpg', etc.
                train_list_filenames = json.load(f)['train']
                # Extract base names without extension for matching across types
                train_list_basenames = set(os.path.splitext(f)[0] for f in train_list_filenames)
            if not train_list_basenames:
                 logging.warning(f"Train list in {split_json_path} is empty for scene {scene_id}. Skipping scene.")
                 continue
            logging.info(f"Found {len(train_list_basenames)} frames in train list for scene {scene_id}.")
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON for scene {scene_id} at {split_json_path}. Skipping scene.")
            continue
        except KeyError:
             logging.error(f"'train' key not found in JSON {split_json_path} for scene {scene_id}. Skipping scene.")
             continue
        except Exception as e:
            logging.error(f"Error reading train list JSON for scene {scene_id}: {e}. Skipping scene.")
            continue
        # ----------------------------------------------------

        # Create output directories for the current scene following ScanNet structure
        scene_color_dir = os.path.join(output_dir, 'color', split_name, scene_id)
        scene_depth_dir = os.path.join(output_dir, 'depth', split_name, scene_id)
        scene_pose_dir = os.path.join(output_dir, 'pose', split_name, scene_id)
        # Intrinsic dir is top-level within split, not scene-specific dir needed here
        intrinsic_base_output_dir = os.path.join(output_dir, 'intrinsic', split_name)

        # Create directories if they don't exist
        os.makedirs(scene_color_dir, exist_ok=True)
        os.makedirs(scene_depth_dir, exist_ok=True)
        os.makedirs(scene_pose_dir, exist_ok=True)
        os.makedirs(intrinsic_base_output_dir, exist_ok=True) # Ensure top-level intrinsic dir exists

        # sample frames
        # 1. get all frames from one of the source dirs (assuming consistent naming and count)
        try:
            # Get all files first
            all_color_files = sorted([f for f in os.listdir(color_dir) if os.path.isfile(os.path.join(color_dir, f))])
            all_depth_files = sorted([f for f in os.listdir(depth_dir) if os.path.isfile(os.path.join(depth_dir, f))])
            all_camera_files = sorted([f for f in os.listdir(camera_dir) if os.path.isfile(os.path.join(camera_dir, f))])

            # Filter based on train_list_basenames
            color_files = [f for f in all_color_files if os.path.splitext(f)[0] in train_list_basenames]
            depth_files = [f for f in all_depth_files if os.path.splitext(f)[0] in train_list_basenames]
            camera_files = [f for f in all_camera_files if os.path.splitext(f)[0] in train_list_basenames]

        except FileNotFoundError as e:
             logging.warning(f"Error listing files in source directory for scene {scene_id}: {e}. Skipping.")
             continue

        # Basic check for consistency AFTER filtering
        if not (len(color_files) == len(depth_files) == len(camera_files)):
            logging.warning(f"Inconsistent number of *training* files in color/depth/camera directories for scene {scene_id} after filtering. Skipping.")
            logging.warning(f"Counts: color={len(color_files)}, depth={len(depth_files)}, camera={len(camera_files)}")
            continue

        total_train_frames = len(color_files) # Now using the count of train frames
        if total_train_frames == 0:
             logging.warning(f"No training frames found (or matched) for scene {scene_id}. Skipping.")
             continue

        # uniformly sample num_frames_to_sample frames from the *training* frames
        if total_train_frames <= num_frames_to_sample:
            indices = np.arange(total_train_frames)
            logging.warning(f"Total train frames ({total_train_frames}) <= num_frames_to_sample ({num_frames_to_sample}) for scene {scene_id}. Exporting all {total_train_frames} train frames.")
        else:
            indices = np.linspace(0, total_train_frames - 1, num_frames_to_sample, dtype=int)
            indices = np.unique(indices) # Ensure uniqueness

        logging.info(f"Sampling {len(indices)} frames for scene {scene_id} from train list.")

        sampled_color_files = [color_files[i] for i in indices]
        sampled_depth_files = [depth_files[i] for i in indices]
        sampled_camera_files = [camera_files[i] for i in indices] # Assumes camera files match by index/name stem

        # save sampled frames and camera parameters
        intrinsic_saved = False
        for i, frame_idx in enumerate(indices):
            color_fname = sampled_color_files[i]
            depth_fname = sampled_depth_files[i]
            camera_fname = sampled_camera_files[i] # e.g., 000000.npz
            frame_basename = os.path.splitext(camera_fname)[0] # e.g., 000000

            # Copy Color
            src_color_path = os.path.join(color_dir, color_fname)
            dst_color_path = os.path.join(scene_color_dir, color_fname) # Keep original filename (likely .jpg)
            try:
                shutil.copy2(src_color_path, dst_color_path) # copy2 preserves metadata
            except Exception as e:
                logging.warning(f"Warning: Failed to copy color file {src_color_path} to {dst_color_path}: {e}")

            # Copy Depth
            src_depth_path = os.path.join(depth_dir, depth_fname)
            dst_depth_path = os.path.join(scene_depth_dir, depth_fname) # Keep original filename (likely .png)
            try:
                shutil.copy2(src_depth_path, dst_depth_path)
            except Exception as e:
                logging.warning(f"Warning: Failed to copy depth file {src_depth_path} to {dst_depth_path}: {e}")

            # Process Camera File (Intrinsic + Pose)
            src_camera_path = os.path.join(camera_dir, camera_fname)
            try:
                camera_data = np.load(src_camera_path)
                intrinsic = camera_data.get('intrinsic')
                extrinsic = camera_data.get('extrinsic') # This is the camera pose (camera_to_world)

                if extrinsic is None:
                     logging.warning(f"Warning: 'extrinsic' (pose) data not found in {src_camera_path}. Skipping pose saving for this frame.")
                else:
                    # Save Pose (Extrinsic) as .txt
                    pose_filename = os.path.join(scene_pose_dir, f'{frame_basename}.txt')
                    save_matrix_to_file(extrinsic, pose_filename)

                # Save Intrinsic only for the first frame
                if not intrinsic_saved:
                    if intrinsic is None:
                        logging.warning(f"Warning: 'intrinsic' data not found in the first sampled camera file {src_camera_path} for scene {scene_id}. Cannot save scene intrinsic.")
                    else:
                        # Intrinsic filename now includes scene_id for clarity, saved in the split's intrinsic dir
                        intrinsic_filename = os.path.join(intrinsic_base_output_dir, f'intrinsic_color_{scene_id}.txt') # Modified filename convention
                        save_matrix_to_file(intrinsic, intrinsic_filename)
                        intrinsic_saved = True # Mark as saved
                        logging.info(f"Saved intrinsic for scene {scene_id} to {intrinsic_filename}") # Log intrinsic save

            except FileNotFoundError:
                logging.warning(f"Warning: Camera file not found: {src_camera_path}. Skipping pose/intrinsic saving for this frame.")
            except Exception as e:
                logging.warning(f"Warning: Error processing camera file {src_camera_path}: {e}. Skipping pose/intrinsic saving for this frame.")

    print(f"Finished exporting sampled frames for split '{split_name}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export sampled frames (color, depth, pose, intrinsics) based on a scene list, matching ScanNet structure.")
    parser.add_argument('--split_path', type=str, required=True,
                        help='Path to the file containing the list of scene IDs (e.g., train.txt, val.txt).')
    parser.add_argument('--raw_data_dir', type=str, required=True,
                        help='Path to the directory containing raw scene data (needs dslr/train_test_lists.json).')
    parser.add_argument('--rendered_data_dir', type=str, required=True,
                        help='Path to the directory containing rendered scene data (e.g., /path/to/ScanNetPP/data).')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the base directory where output folders (color, depth, pose, intrinsic) will be saved.')
    parser.add_argument('--num_frames_to_sample', type=int, required=False, default=32,
                        help='Number of frames to uniformly sample per scene from the train list (default: 32).')
    parser.add_argument('--verify_scene_id', type=str, required=False, default=None,
                        help='If provided, run verification for this specific scene ID after export.')
    parser.add_argument('--verification_output_dir', type=str, required=False, default='verification_output',
                        help='Directory to save verification point clouds (default: verification_output).')
    parser.add_argument('--split_name', type=str, required=True,
                        help='Name of the split being processed (e.g., train, val). This determines the subfolder within output_dir.')
    args = parser.parse_args()

    # Simple check: Extract split name from file path if not provided? Or require it explicitly.
    # Current implementation requires explicit --split_name.

    # export_sampled_frames(args.split_path, args.raw_data_dir, args.rendered_data_dir, args.output_dir, args.num_frames_to_sample, args.split_name)

    # --- Optional Verification Step --- 
    if args.verify_scene_id:
        logging.info(f"--- Running Post-Export Verification for Scene: {args.verify_scene_id} ---")
        verify_scene_reprojection(
            scene_id=args.verify_scene_id,
            output_dir=args.output_dir, # Use the same output dir where data was exported
            split_name=args.split_name, # Use the same split name
            verification_output_dir=args.verification_output_dir
            # depth_scale can be passed if needed, defaults to 1000.0
        )
        logging.info(f"--- Verification Finished for Scene: {args.verify_scene_id} ---")
    else:
        logging.info("Skipping verification step (no --verify_scene_id provided).")
