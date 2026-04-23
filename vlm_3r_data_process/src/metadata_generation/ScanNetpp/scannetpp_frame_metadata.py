import argparse
import os
import numpy as np
import json
import tqdm
import logging
from PIL import Image
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

# Import base classes (Assuming they are accessible via PYTHONPATH or relative path)
# Adjust the import path based on your project structure if necessary
try:
    from src.base_processor import BaseProcessorConfig, AbstractSceneProcessor
except ImportError:
    # Simple fallback if the import fails, consider a more robust solution
    logging.error("Failed to import base processor classes. Ensure PYTHONPATH is set correctly.")
    # Define dummy base classes if needed for the script to run partially
    @dataclass
    class BaseProcessorConfig:
        save_dir: str = "output"
        output_filename: str = "metadata.json"
        num_workers: int = 4
        overwrite: bool = False
        random_seed: int = 42

    class AbstractSceneProcessor:
        def __init__(self, config):
            self.config = config
        def process_all_scenes(self):
            raise NotImplementedError
        def _load_scene_list(self):
            raise NotImplementedError
        def _process_single_scene(self, scene_id):
            raise NotImplementedError


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---

@dataclass
class ScanNetppFrameProcessorConfig(BaseProcessorConfig):
    """Configuration specific to ScanNet++ frame metadata processing."""
    # Required arguments first (no defaults)
    scene_list_file: str = "data/raw_data/scannetpp/splits/nvs_sem_train.txt"
    rendered_data_dir: str = "data/scannetpp_render"
    raw_data_dir: str = "data/raw_data/scannetpp/data"

    # Optional arguments with defaults (from subclass or overriding base)
    max_frames: Optional[int] = None
    img_height: Optional[int] = None
    img_width: Optional[int] = None
    split: str = 'train'
    def __post_init__(self):
        # Ensure save_dir exists (optional, can be done before running)
        # Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory set to: {self.save_dir}")

        # Update output filename to include split if using default name
        # Use the base class default name for comparison
        base_default_filename = "metadata.json" # Assuming this is BaseProcessorConfig's default
        if self.output_filename == base_default_filename:
             self.output_filename = f"scannetpp_frame_metadata_{self.split}.json"
             logger.info(f"Updated output filename to include split: {self.output_filename}")


# --- Helper Functions ---

def read_camera_npz(file_path: str) -> Dict[str, Any] | None:
    """Reads camera intrinsics and pose from a .npz file."""
    try:
        data = np.load(file_path)
        # ScanNet++ camera .npz structure needs confirmation. Assuming keys:
        # 'intrinsic': 3x3 K matrix
        # 'extrinsic': 4x4 C2W pose matrix
        # --- Intrinsics Extraction ---
        if 'intrinsic' not in data:
            logger.warning(f"Key 'intrinsic' not found in {file_path}")
            return None
        intrinsic_matrix = data['intrinsic']
        if intrinsic_matrix.shape[0] < 3 or intrinsic_matrix.shape[1] < 3:
             logger.warning(f"Invalid intrinsic matrix shape {intrinsic_matrix.shape} in {file_path}")
             return None
        intrinsics = {
            "fx": float(intrinsic_matrix[0, 0]),
            "fy": float(intrinsic_matrix[1, 1]),
            "cx": float(intrinsic_matrix[0, 2]),
            "cy": float(intrinsic_matrix[1, 2])
        }

        # --- Pose Extraction ---
        if 'extrinsic' not in data:
            logger.warning(f"Key 'extrinsic' not found in {file_path}")
            return None
        pose_matrix = data['extrinsic']
        if pose_matrix.shape != (4, 4):
            logger.warning(f"Invalid pose matrix shape {pose_matrix.shape} in {file_path}")
            return None

        # --- Image Size (Optional but useful) ---
        img_size = data.get('image_size') # Returns None if not found
        if img_size is not None and len(img_size) == 2:
             img_width, img_height = int(img_size[0]), int(img_size[1])
        else:
             img_width, img_height = None, None
             logger.debug(f"Key 'image_size' not found or invalid in {file_path}")


        return {
            "intrinsics": intrinsics,
            "pose_c2w": pose_matrix.tolist(), # Convert to list for JSON serialization
            "img_width": img_width,
            "img_height": img_height,
        }

    except FileNotFoundError:
        logger.warning(f"Camera file not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error reading camera npz file {file_path}: {e}")
        return None

# --- Processor Implementation ---

class ScanNetppFrameProcessor(AbstractSceneProcessor[ScanNetppFrameProcessorConfig]):
    """
    Processor for ScanNet++ dataset frame metadata. Reads data from rendered and raw directories.
    """
    def __init__(self, config: ScanNetppFrameProcessorConfig):
        super().__init__(config)
        # Validate essential directories
        if not os.path.isdir(self.config.rendered_data_dir):
             logger.warning(f"Rendered data directory not found: {self.config.rendered_data_dir}. This will likely cause errors.")
        if not os.path.isdir(self.config.raw_data_dir):
             logger.warning(f"Raw data directory not found: {self.config.raw_data_dir}. This will likely cause errors.")

    def _load_scene_list(self) -> List[str]:
        """Loads the list of scene IDs from the specified file."""
        logger.info(f"Loading scene list from: {self.config.scene_list_file}")
        try:
            with open(self.config.scene_list_file, "r") as f:
                scene_ids = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(scene_ids)} scene IDs.")
            # Optional: Filter scene_ids based on existence in rendered_data_dir?
            # valid_scene_ids = [sid for sid in scene_ids if os.path.isdir(os.path.join(self.config.rendered_data_dir, sid))]
            # if len(valid_scene_ids) < len(scene_ids):
            #     logger.warning(f"Found {len(valid_scene_ids)} scenes with data in {self.config.rendered_data_dir} out of {len(scene_ids)} listed.")
            # return valid_scene_ids
            return scene_ids
        except FileNotFoundError:
            logger.error(f"Scene list file not found: {self.config.scene_list_file}")
            return [] # Return empty list on error

    def _process_single_scene(self, scene_id: str) -> Dict[str, Any] | None:
        """Processes metadata for selected frames in a single scene."""
        logger.info(f"Processing scene: {scene_id}")
        scene_render_dir = os.path.join(self.config.rendered_data_dir, scene_id, 'dslr')
        scene_raw_dir = os.path.join(self.config.raw_data_dir, scene_id, 'dslr')

        # --- 1. Get Train/Test Split List ---
        split_path = os.path.join(scene_raw_dir, 'train_test_lists.json')
        if not os.path.exists(split_path):
            logger.error(f"Split list file not found for scene {scene_id}: {split_path}. Skipping scene.")
            return None
        try:
            with open(split_path, 'r') as f:
                split_data = json.load(f)
                if self.config.split not in split_data:
                     logger.error(f"Split '{self.config.split}' not found in {split_path} for scene {scene_id}. Available: {list(split_data.keys())}. Skipping scene.")
                     return None
                target_images = split_data['train'] # List of filenames like 'DSC01752.JPG'
            if not target_images:
                 logger.warning(f"No images found for split '{self.config.split}' in {split_path} for scene {scene_id}. Skipping scene.")
                 return None
            # Sort and potentially limit frames
            target_images.sort()
            total_frames_in_split = len(target_images)
            if self.config.max_frames is not None and self.config.max_frames > 0:
                num_to_sample = self.config.max_frames
                if total_frames_in_split <= num_to_sample:
                    # If total frames are less than or equal to desired samples, take all
                    logger.info(f"Requested {num_to_sample} frames, but only {total_frames_in_split} available for split '{self.config.split}'. Using all available frames.")
                    sampled_indices = np.arange(total_frames_in_split)
                else:
                    # Calculate indices for uniform sampling
                    sampled_indices = np.linspace(0, total_frames_in_split - 1, num_to_sample, dtype=int)
                    # Ensure unique indices if linspace produces duplicates
                    sampled_indices = np.unique(sampled_indices)
                    logger.info(f"Uniformly sampling {len(sampled_indices)} frames from {total_frames_in_split} available for split '{self.config.split}'.")
                # Select images using the sampled indices
                target_images = [target_images[i] for i in sampled_indices]

            # This log message now reflects the number of frames AFTER sampling
            logger.info(f"Processing {len(target_images)} frames for split '{self.config.split}' in scene {scene_id}.")
        except (json.JSONDecodeError, KeyError, Exception) as e:
             logger.error(f"Error reading or parsing split list {split_path}: {e}. Skipping scene {scene_id}.")
             return None

        # --- 2. Define Data Directories ---
        # Using scene_render_dir which points to .../<scene_id>/dslr/
        color_dir = os.path.join(scene_render_dir, 'rgb_resized_undistorted')
        depth_dir = os.path.join(scene_render_dir, 'render_depth')
        instance_dir = os.path.join(scene_render_dir, 'render_instance') # Note: Original comment had 'render_instance/render_instance'
        camera_dir = os.path.join(scene_render_dir, 'camera')

        # Check essential directories for the scene
        if not os.path.isdir(color_dir): logger.warning(f"Color dir not found: {color_dir}"); # Continue, frame checks will fail
        if not os.path.isdir(depth_dir): logger.warning(f"Depth dir not found: {depth_dir}");
        if not os.path.isdir(instance_dir): logger.warning(f"Instance dir not found: {instance_dir}");
        if not os.path.isdir(camera_dir): logger.warning(f"Camera dir not found: {camera_dir}");
        assert len(os.listdir(color_dir)) == len(os.listdir(depth_dir)) == len(os.listdir(instance_dir)) == len(os.listdir(camera_dir)), f"Number of frames in color, depth, instance, camera do not match for scene {scene_id}"

        # --- 3. Determine Image Dimensions ---
        img_width = self.config.img_width
        img_height = self.config.img_height
        first_image_processed = False
        first_intrinsics = None # Store intrinsics from the first successfully processed camera file

        # --- 4. Process Each Frame ---
        frame_data = []
        for image_filename in target_images:
            frame_id_base = os.path.splitext(image_filename)[0] # e.g., 'DSC01752'

            # Construct paths for this frame
            color_path_abs = os.path.join(color_dir, image_filename) # Assume JPG or similar from split list
            color_path_rel = os.path.join(scene_id, 'dslr', 'rgb_resized_undistorted', image_filename) # Relative path for metadata

            # Assume other files use the base name + standard extension
            depth_path_abs = os.path.join(depth_dir, f"{frame_id_base}.png")
            depth_path_rel = os.path.join(scene_id, 'dslr', 'render_depth', f"{frame_id_base}.png")

            instance_path_abs = os.path.join(instance_dir, f"{frame_id_base}.png")
            instance_path_rel = os.path.join(scene_id, 'dslr', 'render_instance', f"{frame_id_base}.png")

            camera_path_abs = os.path.join(camera_dir, f"{frame_id_base}.npz")
            # No relative camera path usually stored in metadata

            # --- 4a. Check File Existence ---
            if not os.path.exists(color_path_abs):
                logger.warning(f"Color file missing for frame {frame_id_base} in {scene_id}: {color_path_abs}. Skipping frame.")
                continue
            if not os.path.exists(depth_path_abs):
                logger.warning(f"Depth file missing for frame {frame_id_base} in {scene_id}: {depth_path_abs}. Skipping frame.")
                continue
            # Instance mask is optional? Check ScanNet++ documentation. Assume optional for now.
            if not os.path.exists(instance_path_abs):
                logger.debug(f"Instance mask file missing for frame {frame_id_base} in {scene_id}: {instance_path_abs}.")
                instance_path_rel = None # Indicate missing mask
            if not os.path.exists(camera_path_abs):
                 logger.warning(f"Camera file missing for frame {frame_id_base} in {scene_id}: {camera_path_abs}. Skipping frame.")
                 continue

            # --- 4b. Read Camera Data (Pose + Intrinsics for this frame) ---
            camera_info = read_camera_npz(camera_path_abs)
            if camera_info is None:
                logger.warning(f"Could not read camera info for frame {frame_id_base}, scene {scene_id}. Skipping frame.")
                continue

            frame_pose_c2w = camera_info["pose_c2w"]
            frame_intrinsics = camera_info["intrinsics"]

            # --- 4c. Get Image Dimensions (if not already set) ---
            if not first_image_processed:
                # Try getting dims from camera npz first
                npz_w, npz_h = camera_info.get("img_width"), camera_info.get("img_height")
                if npz_w and npz_h and img_width is None and img_height is None:
                     img_width = npz_w
                     img_height = npz_h
                     logger.info(f"Read image dimensions ({img_width}x{img_height}) from camera file {camera_path_abs}")
                # Fallback: Read from the actual color image if needed
                elif img_width is None or img_height is None:
                    try:
                        with Image.open(color_path_abs) as img:
                            if img_width is None: img_width = img.width
                            if img_height is None: img_height = img.height
                        logger.info(f"Read image dimensions ({img_width}x{img_height}) from image file {color_path_abs}")
                    except Exception as e:
                        logger.error(f"Could not read image dimensions from {color_path_abs}: {e}. Skipping scene (cannot determine dimensions).")
                        return None # Cannot proceed without image dimensions
                # Store the first valid intrinsics as the scene's representative intrinsics
                first_intrinsics = frame_intrinsics
                first_image_processed = True
            # Optional: Check if intrinsics are consistent across frames (they might not be for DSLR)
            # if frame_intrinsics != first_intrinsics:
            #     logger.warning(f"Intrinsics changed between frames in scene {scene_id}. Using first frame's intrinsics as scene default.")
            #     # Decide how to handle varying intrinsics - store per frame? Store first? Error?

            # --- 4d. Extract BBoxes (Optional) ---
            # TODO: Implement bbox extraction from instance mask if needed
            bboxes_2d = [] # Placeholder
            if instance_path_rel: # If mask exists
                try:
                    # Similar logic to ScanNet: open mask, find unique IDs, get coords, calculate bbox
                    with Image.open(instance_path_abs) as mask_img:
                         # Ensure mask matches expected dimensions if needed (e.g., img_width x img_height)
                         # mask_img = mask_img.resize((img_width, img_height), Image.NEAREST)
                         mask_np = np.array(mask_img)
                         if mask_np.ndim == 2:
                             instance_ids = np.unique(mask_np)
                             # Adjust filtering based on ScanNet++ instance format (0=background?)
                             valid_instance_ids = [inst_id for inst_id in instance_ids if inst_id > 0]

                             for inst_id in valid_instance_ids:
                                 coords = np.argwhere(mask_np == inst_id)
                                 if coords.size == 0: continue
                                 ymin, xmin = coords.min(axis=0)
                                 ymax, xmax = coords.max(axis=0)
                                 if xmax >= xmin and ymax >= ymin:
                                     # Adjust instance ID mapping if necessary (e.g., ScanNet++ might be 1-based)
                                     bboxes_2d.append({
                                         "instance_id": int(inst_id), # Keep original ID for now
                                         "bbox_2d": [int(xmin), int(ymin), int(xmax), int(ymax)],
                                     })
                         else:
                              logger.warning(f"Unexpected mask dimensions ({mask_np.ndim}) in {instance_path_abs} for frame {frame_id_base}")
                except Exception as e:
                    logger.exception(f"Error processing instance mask {instance_path_abs} for frame {frame_id_base}: {e}")
                    bboxes_2d = [] # Clear bboxes on error


            # --- 4e. Assemble Frame Information ---
            frame_info = {
                "frame_id": frame_id_base, # Use the string ID like 'DSC01752'
                "file_path_color": color_path_rel,
                "file_path_depth": depth_path_rel,
                "file_path_mask": instance_path_rel, # Can be None if mask was missing
                "camera_pose_camera_to_world": frame_pose_c2w,
                "bboxes_2d": bboxes_2d,
                # Optional: Store per-frame intrinsics if they vary significantly
                # "camera_intrinsics": frame_intrinsics
            }
            frame_data.append(frame_info)

        # --- 5. Final Scene Summary ---
        if not first_image_processed:
            logger.error(f"Could not process any frames successfully for scene {scene_id}. Skipping scene.")
            return None

        if img_width is None or img_height is None:
             logger.error(f"Could not determine image dimensions for scene {scene_id}. Skipping scene.")
             return None

        return {
            # Use intrinsics from the first processed frame as the representative for the scene
            "camera_intrinsics": first_intrinsics,
            "frames": frame_data,
            "img_width": img_width,
            "img_height": img_height,
            "split": self.config.split, # Record which split this metadata belongs to
        }


# --- Main Execution Logic ---

def main():
    parser = argparse.ArgumentParser(description="Process ScanNet++ frame metadata.")

    # Arguments for input data locations (Required)
    parser.add_argument("--scene_list_file", type=str, required=True,
                        help='Path to the text file listing scene IDs to process.')
    parser.add_argument("--rendered_data_dir", type=str, required=True,
                        help='Directory containing rendered data (color, depth, camera, instance) per scene.')
    parser.add_argument("--raw_data_dir", type=str, required=True,
                        help='Directory containing raw data (train/test lists) per scene.')

    # Arguments for processing control (Optional)
    parser.add_argument("--max_frames", type=int, default=None,
                        help='Maximum number of frames to process per scene (processes all if None).')
    parser.add_argument("--split", type=str, default='train', choices=['train', 'test', 'val'], # Adjust choices if needed
                        help='Which split list (train/test/val) to process from train_test_lists.json.')
    parser.add_argument('--img_height', type=int, default=None,
                        help='Specify image height. If None, read from first image/camera file.')
    parser.add_argument('--img_width', type=int, default=None,
                        help='Specify image width. If None, read from first image/camera file.')


    # Arguments for output and base processing behavior (Inherited defaults)
    parser.add_argument('--save_dir', type=str, default="output/ScanNetpp_metadata", # More specific default
                        help='Directory to save the output JSON metadata.')
    parser.add_argument('--output_filename', type=str, default="metadata.json", # Base default, will be updated by config
                        help='Base name for the output JSON file (split name will be added).')
    parser.add_argument('--num_workers', type=int, default=BaseProcessorConfig.num_workers,
                        help='Number of worker processes for parallel processing.')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Allow overwriting existing output file.')
    parser.add_argument('--random_seed', type=int, default=BaseProcessorConfig.random_seed,
                        help='Random seed for operations.')

    args = parser.parse_args()

    # Create config object using parsed arguments
    config = ScanNetppFrameProcessorConfig(
        scene_list_file=args.scene_list_file,
        rendered_data_dir=args.rendered_data_dir,
        raw_data_dir=args.raw_data_dir,
        max_frames=args.max_frames,
        split=args.split,
        img_height=args.img_height,
        img_width=args.img_width,
        # Base config args
        save_dir=args.save_dir,
        output_filename=args.output_filename,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
        random_seed=args.random_seed
    )
    # The __post_init__ in the config class will handle updating the output filename.

    # Initialize and run the processor
    # Note: Parallel processing logic is in the Base class, assuming it exists and works.
    # If BaseProcessorConfig/AbstractSceneProcessor are dummies, this will run sequentially.
    try:
        processor = ScanNetppFrameProcessor(config)
        processor.process_all_scenes() # This method handles parallel execution if implemented in base
        logger.info(f"Metadata generation complete. Output saved to {os.path.join(config.save_dir, config.output_filename)}")
    except NameError:
        logger.error("AbstractSceneProcessor or BaseProcessorConfig not properly defined/imported. Cannot run processor.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during processing: {e}")


if __name__ == "__main__":
    main()
    # Delete the old contents before adding the new structure
    # (The edit tool should replace the entire file content)
    pass # Placeholder, the edit tool handles file replacement

# Remove old functions if they exist after the edit replaces the file
# def parse_args(): ...
# def process_scene(...): ...
