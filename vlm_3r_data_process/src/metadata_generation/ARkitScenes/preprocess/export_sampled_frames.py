import argparse
import json
from pathlib import Path
import numpy as np
import shutil
from tqdm import tqdm
import os
import cv2
import glob # For finding files
from src.metadata_generation.ScanNetpp.preprocess.export_sampled_frames import verify_scene_reprojection

# --- 尝试从ARKitScenes仓库导入相关工具 ---
# 假设此脚本位于ARKitScenes仓库的某个子目录，或者ARKitScenes的路径已添加到PYTHONPATH
try:
    # 这里的路径可能需要根据您的脚本实际存放位置进行调整
    # 例如，如果您的脚本和 threedod 目录同级：
    # from threedod.benchmark_scripts.utils.tenFpsDataLoader import TrajStringToMatrix
    # from threedod.benchmark_scripts.utils.rotation import convert_angle_axis_to_matrix3

    # 作为一个更通用的例子，我们直接从 ARKitScenes/threedod/benchmark_scripts/utils/ 目录导入
    # 您可能需要将该目录添加到 sys.path
    import sys
    current_script_path = Path(os.path.abspath(__file__))
    vlm_3r_root = current_script_path.parent.parent.parent.parent.parent
    candidate_dirs = [
        vlm_3r_root / "datasets" / "ARKitScenes" / "threedod" / "benchmark_scripts",
        vlm_3r_root.parent / "ARKitScenes" / "threedod" / "benchmark_scripts",
        Path("/local_home/pantianbo/projects/vision_reasoning/ARKitScenes/threedod/benchmark_scripts"),
    ]

    arkit_scripts_base_dir = None
    for candidate in candidate_dirs:
        if candidate.exists():
            arkit_scripts_base_dir = candidate
            break

    if arkit_scripts_base_dir is None:
        raise ImportError(f"Could not find ARKitScenes benchmark_scripts in candidates: {candidate_dirs}")

    sys.path.append(str(arkit_scripts_base_dir))
    from utils.tenFpsDataLoader import TrajStringToMatrix # Import from 'utils' package
    # from utils.rotation import convert_angle_axis_to_matrix3 # TrajStringToMatrix 内部已使用
except ImportError:
    print("Error: Could not import ARKitScenes utility functions.")
    print("Please ensure that the 'threedod/benchmark_scripts/utils' directory from the ARKitScenes repository")
    print("is in your PYTHONPATH or accessible from your script's location.")
    print("You might need to copy 'tenFpsDataLoader.py' and 'rotation.py' to a location")
    print("where your Python interpreter can find them, or adjust the import paths.")
    exit(1)
# --- ARKitScenes 工具导入结束 ---


def resolve_scene_frames_root(scene_id: str, raw_data_scene_path: Path) -> Path:
    """
    Supports both of the following layouts:
    1. {split}/{scene_id}/lowres_wide
    2. {split}/{scene_id}/{scene_id}_frames/lowres_wide
    """
    candidates = [
        raw_data_scene_path / f"{scene_id}_frames",
        raw_data_scene_path,
    ]
    for candidate in candidates:
        if (candidate / "lowres_wide").exists():
            return candidate
    return raw_data_scene_path / f"{scene_id}_frames"


def parse_pincam_file(pincam_path):
    """
    Parses an ARKitScenes .pincam file.
    Format: width height fx fy cx cy
    Returns a dictionary with intrinsics and original dimensions.
    """
    if not pincam_path.exists():
        return None
    with open(pincam_path, 'r') as f:
        line = f.readline().strip()
        parts = [float(p) for p in line.split()]
        if len(parts) == 6:
            return {
                'width': parts[0],
                'height': parts[1],
                'fx': parts[2],
                'fy': parts[3],
                'cx': parts[4],
                'cy': parts[5]
            }
    return None

def get_arkit_intrinsics_matrix(pincam_data, target_new_width=None, target_new_height=None):
    """
    Creates a 3x3 intrinsics matrix from parsed .pincam data.
    Scales intrinsics if target_new_width and target_new_height are provided.
    """
    if pincam_data is None:
        return None

    original_fx = pincam_data['fx']
    original_fy = pincam_data['fy']
    original_cx = pincam_data['cx']
    original_cy = pincam_data['cy']
    original_width = pincam_data['width']
    original_height = pincam_data['height']

    current_fx, current_fy, current_cx, current_cy = original_fx, original_fy, original_cx, original_cy

    if target_new_width is not None and target_new_height is not None and \
       (original_width != target_new_width or original_height != target_new_height):
        print(f"Scaling intrinsics from {original_width}x{original_height} to {target_new_width}x{target_new_height}")
        if original_width > 0 and original_height > 0:
            scale_x = target_new_width / original_width
            scale_y = target_new_height / original_height
            current_fx = original_fx * scale_x
            current_fy = original_fy * scale_y
            current_cx = original_cx * scale_x
            current_cy = original_cy * scale_y
        else:
            print(f"Warning: Original camera dimensions ({original_width}x{original_height}) are invalid for scaling. Intrinsics not scaled.")

    intrinsics_matrix = np.array([
        [current_fx, 0, current_cx],
        [0, current_fy, current_cy],
        [0, 0, 1]
    ])
    return intrinsics_matrix

def save_matrix_to_txt(matrix, output_path):
    """Saves a NumPy matrix to a text file."""
    np.savetxt(output_path, matrix)

def find_closest_pose(target_timestamp_str, poses_from_traj):
    """
    Finds the closest pose in poses_from_traj (dict mapping timestamp_str to pose_matrix)
    to the target_timestamp_str.
    A simple nearest match is used here. More sophisticated interpolation could be added.
    """
    target_ts = float(target_timestamp_str)
    
    # Check for exact match first (keys in poses_from_traj are already formatted to .3f)
    if target_timestamp_str in poses_from_traj:
        return poses_from_traj[target_timestamp_str]

    # If no exact match, find the numerically closest
    available_timestamps = np.array([float(ts_str) for ts_str in poses_from_traj.keys()])
    closest_idx = (np.abs(available_timestamps - target_ts)).argmin()
    closest_ts_str = f"{available_timestamps[closest_idx]:.3f}"
    
    # Add a tolerance check if needed, e.g., if closest_ts_str is too far from target_ts
    if abs(available_timestamps[closest_idx] - target_ts) > 0.05: # 50ms tolerance
        print(f"Warning: No pose found within 50ms for timestamp {target_timestamp_str}. Closest is {closest_ts_str}.")
        return None
        
    return poses_from_traj[closest_ts_str]


def process_arkit_scene(scene_id, raw_data_scene_path, output_base_path, split_name, num_frames_to_sample, target_image_size):
    """
    Processes a single ARKitScenes scene.
    Ensures that only valid frames are processed and counted towards num_frames_to_sample.
    """
    print(f"\nProcessing ARKitScene: {scene_id}")

    frames_root = resolve_scene_frames_root(scene_id, raw_data_scene_path)
    rgb_src_dir = frames_root / "lowres_wide"
    depth_src_dir = frames_root / "lowres_depth"
    pose_src_file = frames_root / "lowres_wide.traj"
    intrinsic_src_dir = frames_root / "lowres_wide_intrinsics"

    if not rgb_src_dir.exists():
        print(f"RGB source directory not found for scene {scene_id} at {rgb_src_dir}. Skipping scene.")
        return

    # Create output directories
    split_name_map = {"Training": "train", "Validation": "val"}
    split_name_mapped = split_name_map.get(split_name, split_name.lower()) # More robust split mapping
    color_out_dir = output_base_path / "color" / split_name_mapped / scene_id
    depth_out_dir = output_base_path / "depth" / split_name_mapped / scene_id
    pose_out_dir = output_base_path / "pose" / split_name_mapped / scene_id
    intrinsic_split_dir = output_base_path / "intrinsic" / split_name_mapped

    color_out_dir.mkdir(parents=True, exist_ok=True)
    # Depth, Pose, Intrinsic directories are created if their respective sources exist and are processed.
    # No need to create them upfront if the source data might be missing for the whole scene.

    # Load all poses from the .traj file
    poses_from_traj = {}
    if pose_src_file.exists():
        with open(pose_src_file, 'r') as f:
            traj_lines = f.readlines()
        for line in traj_lines:
            if line.strip():
                tokens = line.split()
                traj_timestamp_str = f"{float(tokens[0]):.3f}"
                _, pose_matrix_c2w = TrajStringToMatrix(line)
                poses_from_traj[traj_timestamp_str] = pose_matrix_c2w
        if not poses_from_traj:
            print(f"Warning: Pose file {pose_src_file} was read, but no poses were loaded. Pose processing will fail for all frames.")
    else:
        print(f"Warning: Pose (.traj) file not found for scene {scene_id} at {pose_src_file}. Pose processing will be skipped.")
        # If poses are essential, consider returning here or setting a flag.

    # Get list of all RGB frames
    all_rgb_frame_files = sorted(list(rgb_src_dir.glob(f"{scene_id}_*.png")))
    if not all_rgb_frame_files:
        print(f"No RGB frames found in {rgb_src_dir} for scene {scene_id} despite directory existing. Skipping scene.")
        return
    print(f"Found {len(all_rgb_frame_files)} total RGB frames for scene {scene_id}.")

    # --- Intrinsics Handling: Must be successful for the scene to be processed ---
    scene_intrinsic_matrix = None
    scene_intrinsic_saved = False
    if intrinsic_src_dir.exists():
        intrinsic_split_dir.mkdir(parents=True, exist_ok=True) # Create now as we intend to save here
        output_intrinsic_file = intrinsic_split_dir / f"intrinsics_{scene_id}.txt"
        
        # Try to get intrinsics from the first available RGB frame.
        first_rgb_for_intrinsic = all_rgb_frame_files[0]
        first_frame_id_from_rgb = first_rgb_for_intrinsic.stem.split('_')[-1]
        
        representative_pincam_path = None
        # Search for a pincam file matching the first RGB frame's timestamp (with some tolerance)
        # This could be made more robust by checking a few initial frames if the first one fails.
        try:
            # Format from lowres_wide_intrinsics is [video_id]_[timestamp].pincam
            # Timestamp might have slight float variations.
            ts_float = float(first_frame_id_from_rgb)
            pincam_options = [
                intrinsic_src_dir / f"{scene_id}_{ts_float:.3f}.pincam", # Exact match formatted
                intrinsic_src_dir / f"{scene_id}_{first_frame_id_from_rgb}.pincam", # Original string from filename
                intrinsic_src_dir / f"{scene_id}_{ts_float - 0.001:.3f}.pincam",
                intrinsic_src_dir / f"{scene_id}_{ts_float + 0.001:.3f}.pincam",
                # Broader search if needed: list(intrinsic_src_dir.glob(f"{scene_id}_{int(ts_float)}*.pincam"))
            ]
            for p_opt in pincam_options:
                if p_opt.exists():
                    representative_pincam_path = p_opt
                    break
            
            if not representative_pincam_path: # Try a slightly broader search by stem
                potential_pincams = list(intrinsic_src_dir.glob(f"{first_rgb_for_intrinsic.stem.rsplit('_', 1)[0]}_*.pincam"))
                if potential_pincams:
                    # Find closest timestamp among these if multiple exist
                    # This is a simplified placeholder for more advanced matching if needed
                    representative_pincam_path = min(potential_pincams, key=lambda p: abs(float(p.stem.split('_')[-1]) - ts_float) if p.stem.split('_')[-1].replace('.', '', 1).isdigit() else float('inf'))


        except ValueError:
             print(f"Warning: Could not parse timestamp {first_frame_id_from_rgb} from first RGB for intrinsics. Trying glob.")
             # Fallback to glob if timestamp parsing fails
             potential_pincams = list(intrinsic_src_dir.glob(f"{scene_id}_*.pincam"))
             if potential_pincams:
                 representative_pincam_path = potential_pincams[0] # Pick first one as a guess
                 print(f"Fallback: using {representative_pincam_path.name} for intrinsics.")


        if representative_pincam_path and representative_pincam_path.exists():
            first_intrinsic_data = parse_pincam_file(representative_pincam_path)
            if first_intrinsic_data:
                target_w, target_h = (target_image_size[1], target_image_size[0]) if target_image_size else (None, None)
                scene_intrinsic_matrix = get_arkit_intrinsics_matrix(first_intrinsic_data,
                                                                  target_new_width=target_w,
                                                                  target_new_height=target_h)
                if scene_intrinsic_matrix is not None:
                    save_matrix_to_txt(scene_intrinsic_matrix, output_intrinsic_file)
                    print(f"Saved scene intrinsics to {output_intrinsic_file}")
                    scene_intrinsic_saved = True
                else:
                    print(f"Warning: Could not generate scaled intrinsics matrix for {scene_id} from {representative_pincam_path.name}.")
            else:
                print(f"Warning: Failed to parse representative pincam file {representative_pincam_path} for {scene_id}.")
        else:
            print(f"Warning: No representative .pincam file found for scene {scene_id} in {intrinsic_src_dir} based on first frame. Glob results: {list(intrinsic_src_dir.glob(f'{scene_id}_*.pincam'))[:5]}")
    else:
        print(f"Warning: Intrinsic source directory {intrinsic_src_dir} not found. Cannot process scene {scene_id}.")

    if not scene_intrinsic_saved:
        print(f"CRITICAL: Failed to obtain and save valid intrinsics for scene {scene_id}. Skipping frame processing for this scene.")
        return
    # --- End Intrinsics ---

    # Create other output directories now that intrinsics are confirmed
    # These are created only if their respective source data might lead to output.
    # Color output directory (color_out_dir) was already created if intrinsics were saved.
    # Depth and Pose directories will be created if frames are actually processed for them.


    # --- First Pass: Validate all available frames ---
    all_rgb_frame_files = sorted(list(rgb_src_dir.glob(f"{scene_id}_*.png")))
    if not all_rgb_frame_files:
        print(f"No RGB frames found in {rgb_src_dir} for scene {scene_id} despite directory existing. Skipping scene processing.")
        return

    valid_frame_candidates = []
    print(f"Validating all {len(all_rgb_frame_files)} available frames for scene {scene_id}...")
    for rgb_frame_path in tqdm(all_rgb_frame_files, desc=f"Validating frames for scene {scene_id}", leave=False):
        frame_basename = rgb_frame_path.name
        frame_stem = rgb_frame_path.stem
        
        try:
            frame_timestamp_str = frame_stem.split('_')[-1]
            formatted_timestamp_str = f"{float(frame_timestamp_str):.3f}"
        except ValueError:
            # This warning was already present in the original loop, keeping it here.
            # print(f"Warning: Could not parse timestamp from frame {frame_basename} during validation. Skipping this frame candidate.")
            continue

        # 1. Validate Pose
        pose_is_available_for_frame = False
        pose_matrix_c2w_candidate = None
        if pose_src_file.exists() and poses_from_traj:
            pose_matrix_c2w_candidate = find_closest_pose(formatted_timestamp_str, poses_from_traj)
            if pose_matrix_c2w_candidate is not None:
                pose_is_available_for_frame = True
            # else: find_closest_pose prints its own warning if no suitable pose found
        elif not pose_src_file.exists() or not poses_from_traj: # Pose file missing or no poses loaded for the scene
            pose_is_available_for_frame = True # Valid in the sense that pose is not a blocker if not provided for scene

        if (pose_src_file.exists() and poses_from_traj) and not pose_is_available_for_frame:
            # print(f"Debug: Frame candidate {frame_basename} deemed invalid due to unavailable pose.")
            continue # Skip this candidate

        # 2. Validate RGB (Existence is given by all_rgb_frame_files)
        # Further checks like readability could be added here if necessary but adds I/O.
        # The main processing loop will handle cv2.imread errors.

        # 3. Validate Depth
        depth_is_available_for_frame_or_not_applicable = not depth_src_dir.exists()
        depth_frame_path_candidate = None
        if depth_src_dir.exists():
            _depth_path = depth_src_dir / frame_basename
            if _depth_path.exists():
                depth_is_available_for_frame_or_not_applicable = True
                depth_frame_path_candidate = _depth_path
            # else: This frame lacks a corresponding depth file

        if depth_src_dir.exists() and not depth_is_available_for_frame_or_not_applicable:
            # print(f"Debug: Frame candidate {frame_basename} deemed invalid due to missing depth file.")
            continue # Skip this candidate
        
        valid_frame_candidates.append({
            "rgb_path": rgb_frame_path,
            "frame_stem": frame_stem,
            "frame_basename": frame_basename,
            "timestamp_str": formatted_timestamp_str,
            "pose_matrix_c2w": pose_matrix_c2w_candidate, # Will be None if poses not applicable to scene
            "depth_path": depth_frame_path_candidate # Will be None if depth not applicable or not found
        })

    if not valid_frame_candidates:
        print(f"No valid frame candidates found for scene {scene_id} after validation pass. Skipping processing for this scene.")
        return
    print(f"Found {len(valid_frame_candidates)} valid frame candidates for scene {scene_id}.")

    # --- Sampling Logic ---
    selected_frames_to_process = []
    if num_frames_to_sample is not None and len(valid_frame_candidates) > num_frames_to_sample:
        indices = np.linspace(0, len(valid_frame_candidates) - 1, num_frames_to_sample, dtype=int)
        selected_frames_to_process = [valid_frame_candidates[i] for i in indices]
        print(f"Uniformly sampled {len(selected_frames_to_process)} frames from {len(valid_frame_candidates)} valid candidates.")
    else:
        selected_frames_to_process = valid_frame_candidates
        if num_frames_to_sample is not None:
            print(f"Using all {len(selected_frames_to_process)} valid candidates as count is <= requested {num_frames_to_sample}.")
        else:
            print(f"Using all {len(selected_frames_to_process)} valid candidates as num_frames_to_sample is not set.")
    
    if not selected_frames_to_process:
        print(f"No frames selected for processing for scene {scene_id}. This shouldn't happen if validation passed. Skipping.")
        return

    # --- Second Pass: Process and Save Selected Frames ---
    # Create output directories now that we know we have frames to process for them.
    if any(f["pose_matrix_c2w"] is not None for f in selected_frames_to_process) and pose_src_file.exists() and poses_from_traj:
        pose_out_dir.mkdir(parents=True, exist_ok=True)
    if any(f["depth_path"] is not None for f in selected_frames_to_process) and depth_src_dir.exists():
        depth_out_dir.mkdir(parents=True, exist_ok=True)
    # color_out_dir is already created


    processed_final_frames_count = 0
    for frame_info in tqdm(selected_frames_to_process, desc=f"Processing selected frames for scene {scene_id}"):
        rgb_frame_path = frame_info["rgb_path"]
        frame_stem = frame_info["frame_stem"]
        frame_basename = frame_info["frame_basename"]
        # formatted_timestamp_str = frame_info["timestamp_str"] # available if needed

        output_rgb_file = color_out_dir / frame_basename
        output_pose_file = pose_out_dir / f"{frame_stem}.txt" if (pose_src_file.exists() and poses_from_traj) else None
        output_depth_file = depth_out_dir / frame_basename if frame_info["depth_path"] else None

        # Flags for successful saving of components for the current frame
        rgb_saved = False
        pose_saved = False # Only relevant if poses are expected for the scene
        depth_saved = False # Only relevant if depth is expected for the scene

        # 1. Pose
        if frame_info["pose_matrix_c2w"] is not None and output_pose_file: # Pose exists for frame and scene expects poses
            try:
                save_matrix_to_txt(frame_info["pose_matrix_c2w"], output_pose_file)
                pose_saved = True
            except Exception as e:
                print(f"Warning: Failed to save pose for {frame_basename}: {e}. Skipping this frame.")
                continue # Skip to next frame in selected_frames_to_process
        
        # 2. RGB image (essential)
        try:
            color_image = cv2.imread(str(rgb_frame_path))
            if color_image is None:
                raise ValueError("cv2.imread for color returned None")
            
            if target_image_size:
                color_image_resized = cv2.resize(color_image, (target_image_size[1], target_image_size[0]), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(str(output_rgb_file), color_image_resized)
            else:
                cv2.imwrite(str(output_rgb_file), color_image)
            rgb_saved = True
        except Exception as e:
            print(f"Warning: Error processing/saving color image {rgb_frame_path}: {e}. Skipping this frame.")
            if pose_saved and output_pose_file: # Cleanup successfully saved pose
                output_pose_file.unlink(missing_ok=True)
            continue # Skip to next frame

        # 3. Depth image
        if frame_info["depth_path"] and output_depth_file: # Depth file exists for frame and scene expects depth
            try:
                depth_image_mm_uint16 = cv2.imread(str(frame_info["depth_path"]), cv2.IMREAD_UNCHANGED)
                if depth_image_mm_uint16 is None:
                    raise ValueError("cv2.imread for depth returned None")
                if np.all(depth_image_mm_uint16 == 0): # Check if depth image is all zeros
                    raise ValueError("Depth image is all zeros.")
                
                if target_image_size:
                    depth_image_resized = cv2.resize(depth_image_mm_uint16, (target_image_size[1], target_image_size[0]), interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite(str(output_depth_file), depth_image_resized)
                else:
                    cv2.imwrite(str(output_depth_file), depth_image_mm_uint16)
                depth_saved = True
            except Exception as e:
                print(f"Warning: Error processing/saving depth image {frame_info['depth_path']}: {e}. Skipping this frame.")
                if pose_saved and output_pose_file: output_pose_file.unlink(missing_ok=True)
                if rgb_saved: output_rgb_file.unlink(missing_ok=True) # rgb_saved is always true here
                continue # Skip to next frame

        # Frame fully processed if all *expected* components are saved
        # RGB must always be saved.
        # Pose must be saved IF poses are expected for the scene AND this frame had a valid pose.
        # Depth must be saved IF depth is expected for the scene AND this frame had a valid depth path.
        
        current_frame_fully_processed = rgb_saved # RGB is baseline
        
        if pose_src_file.exists() and poses_from_traj: # If poses are generally expected for the scene
            if frame_info["pose_matrix_c2w"] is not None: # And this frame was supposed to have a pose
                 current_frame_fully_processed = current_frame_fully_processed and pose_saved
            # If frame_info["pose_matrix_c2w"] is None here, it means pose was not found for this specific frame
            # even if expected for scene, so it's not a blocker for this frame's validity if RGB/Depth are OK.
            # However, our validation logic should ensure frame_info["pose_matrix_c2w"] is not None if poses are expected and it's a valid candidate.

        if depth_src_dir.exists(): # If depth is generally expected for the scene
            if frame_info["depth_path"] is not None: # And this frame was supposed to have depth
                current_frame_fully_processed = current_frame_fully_processed and depth_saved

        if current_frame_fully_processed:
            processed_final_frames_count += 1
        else:
            # This case should ideally not be reached if cleanup and continue logic is correct.
            # It implies some components were saved but the frame wasn't "fully processed" by its definition.
            print(f"Notice: Frame {frame_basename} completed processing steps but was not deemed fully processed. "
                  f"RGB: {rgb_saved}, Pose Expected/Saved: {frame_info['pose_matrix_c2w'] is not None}/{pose_saved}, "
                  f"Depth Expected/Saved: {frame_info['depth_path'] is not None}/{depth_saved}. Cleaning up partials.")
            # Ensure cleanup of all parts if not fully processed
            if output_pose_file and output_pose_file.exists(): output_pose_file.unlink(missing_ok=True)
            if output_rgb_file.exists(): output_rgb_file.unlink(missing_ok=True) # Should be true if rgb_saved
            if output_depth_file and output_depth_file.exists(): output_depth_file.unlink(missing_ok=True)


    # --- End Frame Processing Loop ---

    # Summary of processing for the scene
    if num_frames_to_sample is not None:
        # If sampling was intended
        target_count = min(num_frames_to_sample, len(valid_frame_candidates))
        if processed_final_frames_count < target_count :
            print(f"WARNING: For scene {scene_id}, processed and saved {processed_final_frames_count} frames, "
                  f"but aimed for {target_count} (sampled from {len(valid_frame_candidates)} valid candidates). "
                  f"Some selected frames may have failed during final processing.")
        elif processed_final_frames_count == 0 and len(selected_frames_to_process) > 0:
             print(f"WARNING: No frames could be finally processed and saved for scene {scene_id} out of {len(selected_frames_to_process)} selected candidates.")
        else:
            print(f"Successfully processed and saved {processed_final_frames_count} frames for scene {scene_id} (target after sampling: {target_count}).")

    else: # Processing all valid frames (no sampling or sampling resulted in all valid frames)
        if processed_final_frames_count < len(valid_frame_candidates):
             print(f"WARNING: For scene {scene_id}, found {len(valid_frame_candidates)} valid candidates, "
                   f"but only {processed_final_frames_count} were successfully processed and saved. Some failed in the final step.")
        elif processed_final_frames_count == 0 and len(valid_frame_candidates) > 0: # valid_frame_candidates == selected_frames_to_process here
             print(f"WARNING: No frames could be finally processed and saved for scene {scene_id} out of {len(valid_frame_candidates)} valid candidates.")
        else:
            print(f"Successfully processed and saved all {processed_final_frames_count} valid frames for scene {scene_id}.")
                
    print(f"Finished processing ARKitScene {scene_id}. Output for valid data at ~{output_base_path}/[color|depth|pose]/{split_name_mapped}/{scene_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Processes ARKitScenes raw data to extract and save per-frame RGB, depth, pose, and intrinsics."
    )
    parser.add_argument(
        "--arkit_data_root",
        type=str,
        required=True,
        help="Root directory of the downloaded ARKitScenes 'raw' dataset (e.g., '/path/to/arkitscenes_download/raw_data/arkitscenes/raw/'). Should contain Validation/Training subfolders."
    )
    parser.add_argument(
        "--scene_list_file",
        type=str,
        help="Path to a text file containing a list of scene IDs (video_ids) to process (one ID per line). If not provided, processes all scenes in the specified split."
    )
    parser.add_argument(
        "--scene_ids",
        nargs='*',
        help="List of specific scene IDs (video_ids) to process. Overrides --scene_list_file if both are provided."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="Validation", # Default to Validation as per user's context
        choices=["Training", "Validation"],
        help="Dataset split to process (Training or Validation). Default is Validation."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Base directory where the processed data will be saved."
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=32, # Process all frames by default, similar to original behavior for unspecified frame count
        help="Number of frames to uniformly sample per scene. Processes all if not set or if total frames < num_frames."
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        metavar=('HEIGHT', 'WIDTH'),
        default=[384, 512], # Example: [480, 640]. If None, uses original image size.
        help="Target image size (height width) to resize color and depth images to. If not set, uses original size."
    )
    args = parser.parse_args()

    arkit_data_root = Path(args.arkit_data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_name = args.split

    if not arkit_data_root.exists() or not (arkit_data_root / split_name).exists():
        print(f"Error: ARKitScenes data root '{arkit_data_root}' or split '{split_name}' directory does not exist.")
        return

    scene_ids_to_process = []
    if args.scene_ids:
        scene_ids_to_process = [sid.strip() for sid in args.scene_ids]
        print(f"Processing specified scene IDs: {scene_ids_to_process}")
    elif args.scene_list_file:
        try:
            with open(args.scene_list_file, 'r') as f:
                scene_ids_to_process = [line.strip() for line in f if line.strip()]
            print(f"Found {len(scene_ids_to_process)} scenes to process from {args.scene_list_file}")
        except FileNotFoundError:
            print(f"Error: Scene list file '{args.scene_list_file}' not found.")
            return
    else:
        # If no specific list or file, discover scenes in the split directory
        split_dir_path = arkit_data_root / split_name
        scene_ids_to_process = [d.name for d in split_dir_path.iterdir() if d.is_dir() and d.name.isdigit()]
        print(f"Found {len(scene_ids_to_process)} scenes in {split_dir_path}. Processing all.")


    if not scene_ids_to_process:
        print(f"No scene IDs to process for split '{split_name}'.")
        return

    for scene_id in scene_ids_to_process:
        raw_data_scene_path = arkit_data_root / split_name / scene_id
        if not raw_data_scene_path.is_dir():
            print(f"Warning: Scene directory {raw_data_scene_path} not found. Skipping scene {scene_id}.")
            continue
        process_arkit_scene(scene_id, raw_data_scene_path, output_dir, split_name, args.num_frames, args.image_size)

    print(f"\nAll specified scenes processed for split '{split_name}'.")
    print(f"Output data saved in: {output_dir}")

if __name__ == "__main__":
    main()
    # verify_scene_reprojection(
    #     scene_id="41069025",
    #     output_dir="data/processed_data/ARkitScenes",
    #     verification_output_dir="verification_output",
    #     split_name="val"
    # )
