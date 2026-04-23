import open3d as o3d
import numpy as np
import json
from collections import defaultdict
import os
# import alphashape # No longer needed
import tqdm # Keep for direct use if needed, base class uses it
import random
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any
import multiprocessing as mp # Keep for direct use if needed, base class handles setup
import argparse

# Import base classes
from src.base_processor import BaseProcessorConfig, AbstractSceneProcessor
# Import common utils for room metrics
from src.utils.common_utils import calculate_room_area, calculate_room_center 

logger = logging.getLogger(__name__)
# Base class or runner script handles basicConfig

# --- ARKitScenes Specific Helper Functions ---

# Helper function to calculate room area using alphashape - REMOVED
# def calculate_room_area(xyz: np.ndarray) -> float:
#     ...

# Helper function to compute 3D bounding box corners
def compute_box_3d(scale: List[float], transform: List[float], rotation_list: List[float]) -> np.ndarray:
    """
    Compute the 8 corners of a 3D bounding box given scale, transform (centroid), and rotation (9 floats).
    """
    try:
        scales = np.array(scale) / 2.0
        l, h, w = scales
        center = np.array(transform)
        rotation_matrix = np.array(rotation_list).reshape(3, 3)

        x_corners = [l, l, -l, -l, l, l, -l, -l]
        y_corners = [h, -h, -h, h, h, -h, -h, h]
        z_corners = [w, w, w, w, -w, -w, -w, -w]
        corners_3d = np.dot(rotation_matrix, np.vstack([x_corners, y_corners, z_corners]))

        corners_3d[0, :] += center[0]
        corners_3d[1, :] += center[1]
        corners_3d[2, :] += center[2]
        bbox3d_raw = np.transpose(corners_3d)
        return bbox3d_raw
    except Exception as e:
         logger.error(f"Error computing 3D box: {e}")
         # Return a degenerate box or raise error? Returning zeros for now.
         return np.zeros((8, 3))

# --- Configuration ---

@dataclass
class ARKitScenesProcessorConfig(BaseProcessorConfig):
    """Configuration specific to ARKitScenes metadata processing."""
    dataset_name: str = "ARkitScenes" # Dataset name
    root_dir: str = "data/raw_arkitscenes/raw" # Base directory for raw data (containing splits)
    metadata_csv: str = "data/raw_arkitscenes/raw/metadata.csv" # Path to metadata CSV
    annotations_dir: str = "data/raw_arkitscenes/raw" # Base directory for annotations (containing splits)
    split: str = "Training" # Data split to process ('Training' or 'Validation')
    
    # video_path_prefix: str = "arkitscenes/videos" # Optional prefix for relative video path
    # Inherits save_dir, output_filename, num_workers, overwrite, random_seed


# --- Processor Implementation ---

class ARKitScenesProcessor(AbstractSceneProcessor):
    """Processor for ARKitScenes dataset metadata."""
    def __init__(self, config: ARKitScenesProcessorConfig):
        super().__init__(config)
        self.config: ARKitScenesProcessorConfig # Type hint
        # Seed random number generators for reproducibility in scene sampling
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        logger.info(f"Initialized ARKitScenesProcessor for split '{self.config.split}'. Random seed: {self.config.random_seed}")

    def _load_scene_list(self) -> List[str]:
        """
        Loads metadata, filters scans based on config, samples one scan per scene, and returns scan IDs.
        """
        logger.info(f"Loading metadata from: {self.config.metadata_csv}")
        try:
            with open(self.config.metadata_csv, 'r') as f:
                # Use pandas for robust CSV parsing, handling potential quoting issues
                import pandas as pd
                metadatas_df = pd.read_csv(f)
        except FileNotFoundError:
            logger.error(f"Metadata file not found: {self.config.metadata_csv}")
            return []
        except Exception as e:
            logger.error(f"Error reading metadata CSV {self.config.metadata_csv}: {e}")
            return []
            
        # Ensure required columns exist
        required_cols = ['video_id', 'visit_id', 'fold']
        if not all(col in metadatas_df.columns for col in required_cols):
            logger.error(f"Metadata CSV missing one or more required columns: {required_cols}")
            return []

        # Filter by split
        metadatas_df = metadatas_df[metadatas_df['fold'] == self.config.split]
        # Filter out scans without a scene ID
        metadatas_df = metadatas_df[metadatas_df['visit_id'] != 'NA']
        
        logger.info(f"Found {len(metadatas_df)} scans matching split '{self.config.split}' before file checks.")

        scene_id2scans = defaultdict(list)
        processed_scan_ids = set()

        logger.info("Checking existence of required files for each scan...")
        # Iterate through the filtered DataFrame
        for _, row in tqdm.tqdm(metadatas_df.iterrows(), total=len(metadatas_df), desc="Checking files"):
            scan_id = str(row['video_id'])
            scene_id = str(row['visit_id'])
            
            # Construct paths based on the structure: {root_dir}/{split}/{scan_id}/
            base_scan_path_raw = os.path.join(self.config.root_dir, self.config.split, scan_id)
            base_scan_path_anno = os.path.join(self.config.annotations_dir, self.config.split, scan_id)
            
            # Check if the corresponding files exist (assuming structure from original script)
            mov_path = os.path.join(base_scan_path_raw, f"{scan_id}.mov")
            anno_path = os.path.join(base_scan_path_anno, f"{scan_id}_3dod_annotation.json")
            mesh_path = os.path.join(base_scan_path_anno, f"{scan_id}_3dod_mesh.ply")

            # if not os.path.exists(mov_path):
            #     logger.debug(f"MOV file not found for scan {scan_id}, skipping: {mov_path}")
            #     continue
            if not os.path.exists(anno_path):
                logger.debug(f"Annotation file not found for scan {scan_id}, skipping: {anno_path}")
                continue
            if not os.path.exists(mesh_path):
                 logger.debug(f"Mesh file not found for scan {scan_id}, skipping: {mesh_path}")
                 continue

            # If all files exist, add the scan to the scene
            scene_id2scans[scene_id].append(scan_id)
            processed_scan_ids.add(scan_id)
            
        logger.info(f"Found {len(scene_id2scans)} scenes with {len(processed_scan_ids)} valid scans for split {self.config.split} after file checks.")

        sampled_scans = []
        logger.info("Sampling one scan per scene...")
        # Use the seeded random generator for consistent sampling
        for scene_id, scan_ids in scene_id2scans.items():
            if scan_ids:
                sampled_scans.append(random.choice(scan_ids))
            else:
                 logger.warning(f"Scene {scene_id} had no valid scans after filtering, skipping sampling.")

        logger.info(f"Selected {len(sampled_scans)} scans for processing based on random sampling (1 per scene). Random seed: {self.config.random_seed}")
        return sampled_scans

    def _process_single_scene(self, scan_id: str) -> Dict[str, Any] | None:
        """Processes a single ARKitScenes scan (scene_id here is actually scan_id)."""
        try:
            base_anno_path = os.path.join(self.config.annotations_dir, self.config.split, scan_id)
            mesh_path = os.path.join(base_anno_path, f"{scan_id}_3dod_mesh.ply")
            anno_path = os.path.join(base_anno_path, f"{scan_id}_3dod_annotation.json")
            # Define a relative video path for the output JSON for portability
            low_split = 'train' if self.config.split == 'Training' else 'val'
            video_path_relative = os.path.join(self.config.dataset_name, "videos", low_split, f"{scan_id}.mp4") # Consistent relative path

            # Check files again just in case (though _load_scene_list should filter)
            if not os.path.exists(mesh_path) or not os.path.exists(anno_path):
                 logger.error(f"Mesh or annotation file missing for {scan_id} during processing. Should have been filtered. Skipping.")
                 return None
            # if not os.path.exists(video_path_raw):
            #      logger.warning(f"Video file {video_path_raw} missing for {scan_id}. Storing relative path anyway.")
                 # return None # Decide if missing video is critical

            # 1. Calculate Room Size and Center using common utils
            logger.debug(f"Loading mesh: {mesh_path}")
            try:
                ply_data = o3d.io.read_point_cloud(mesh_path)
                points = np.asarray(ply_data.points)
                if points.shape[0] == 0:
                     logger.warning(f"Mesh file {mesh_path} contains no points for scan {scan_id}. Cannot calculate room metrics. Skipping.")
                     return None # Skip if no points
                
                # Use common utility functions
                room_size = calculate_room_area(points) 
                room_center = calculate_room_center(points) 

            except Exception as e:
                 logger.error(f"Error calculating room metrics for {scan_id} from {mesh_path}: {e}", exc_info=True)
                 logger.warning("Setting room size and center to defaults (0.0 and [0,0,0]) due to error.")
                 room_size = 0.0
                 room_center = [0.0, 0.0, 0.0]
                 # Decide if this is critical: return None

            # 2. Process Annotations
            logger.debug(f"Loading annotations: {anno_path}")
            with open(anno_path, 'r') as f:
                video_anno = json.load(f)

            object_counts = defaultdict(int)
            object_bboxes = defaultdict(list)
            
            if 'data' not in video_anno or not isinstance(video_anno['data'], list):
                logger.warning(f"Annotation data format unexpected or missing for scan {scan_id} in {anno_path}. Skipping object processing.")
                # Decide if scene is still valid without object info: maybe return with empty objects?
                # return None # Or skip entirely
            else:
                for obj in video_anno['data']:
                    label = obj.get('label')
                    segments = obj.get('segments', {}).get('obbAligned', {})
                    
                    # Validate required fields for bounding box processing
                    required_keys = ['axesLengths', 'centroid', 'normalizedAxes']
                    if not label or not segments or not all(k in segments for k in required_keys):
                        logger.debug(f"Skipping object with incomplete data in scan {scan_id}: {obj.get('uid', 'N/A')}")
                        continue

                    object_counts[label] += 1
                    
                    scale = segments['axesLengths']
                    transform = segments['centroid']
                    rotation_list = segments['normalizedAxes'] # List of 9 floats

                    # Compute corners and min/max for simpler bbox representation if needed
                    vertexes = compute_box_3d(scale, transform, rotation_list)
                    if vertexes.shape != (8, 3):
                         logger.warning(f"Could not compute valid 3D box corners for object {obj.get('uid', 'N/A')} in {scan_id}. Skipping bbox info for this object.")
                         min_coords = [0.0, 0.0, 0.0]
                         max_coords = [0.0, 0.0, 0.0]
                    else:
                         min_coords = vertexes.min(axis=0).tolist()
                         max_coords = vertexes.max(axis=0).tolist()
                    
                    # Store the essential OBB parameters + min/max
                    object_bboxes[label].append({
                        'centroid': transform,
                        'axesLengths': scale,
                        'normalizedAxes': rotation_list, # Keep original list format
                        'min_corner': min_coords, # Use clearer key names
                        'max_corner': max_coords
                    })

            # Check if any objects were successfully processed
            if not object_bboxes:
                logger.warning(f"Scan {scan_id} has no valid annotated instances after processing. Skipping.")
                return None
            
            return {
                "video_path": video_path_relative, # Use relative path
                "scene_name": scan_id,
                "split": low_split,
                "dataset": self.config.dataset_name,
                "room_size": room_size,
                "room_center": room_center,
                "object_counts": dict(object_counts), # Convert defaultdict to dict
                "object_bboxes": dict(object_bboxes), # Convert defaultdict to dict
            }

        except FileNotFoundError as e:
            logger.error(f"File not found while processing {scan_id}: {e}")
            return None
        except json.JSONDecodeError as e:
             logger.error(f"Error decoding JSON annotation for {scan_id} from {anno_path}: {e}")
             return None
        except Exception as e:
            logger.error(f"Unexpected error processing scan {scan_id}: {e}", exc_info=True) # Log traceback
            return None

# --- Main Execution Logic ---

def main():
    parser = argparse.ArgumentParser(description="Process ARKitScenes dataset to generate metadata.")
    # Args for BaseProcessorConfig & ARKitScenesProcessorConfig (matching field names)
    parser.add_argument('--save_dir', type=str, default="stage2_data/ARkitScenes/processed_metadata", help='Directory to save the output JSON metadata.')
    parser.add_argument('--output_filename', type=str, default=None, help='Name of the output JSON file (default: arkitscenes_{split}_metadata_seed{seed}.json).')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for multiprocessing (overrides BaseProcessorConfig default).') # Added default like scannetpp
    parser.add_argument('--overwrite', action='store_true', default=False, help='Allow overwriting output file.') # Changed default to False
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed for sampling.') # Keep default 0 as previously
    # Args specific to ARKitScenesProcessorConfig
    parser.add_argument('--root_dir', type=str, default="data/raw_arkitscenes/raw", help='Root directory for raw data.') # Default from config class
    parser.add_argument('--metadata_csv', type=str, default="data/raw_arkitscenes/raw/metadata.csv", help='Metadata CSV path.') # Default from config class
    parser.add_argument('--annotations_dir', type=str, default="data/raw_arkitscenes/raw", help='Annotations directory.') # Default from config class
    parser.add_argument('--split', type=str, default="Training", choices=['Training', 'Validation'], help='Data split to process.')
    parser.add_argument('--dataset_name', type=str, default="ARkitScenes", help='Dataset name.') # Added
    args = parser.parse_args()

    # Construct default output filename if not provided
    output_filename = args.output_filename
    if output_filename is None:
        output_filename = f"arkitscenes_{args.split.lower()}_metadata_seed{args.random_seed}.json"

    # Create config object directly from args
    config = ARKitScenesProcessorConfig(
        save_dir=args.save_dir,
        output_filename=output_filename,
        root_dir=args.root_dir,
        metadata_csv=args.metadata_csv,
        annotations_dir=args.annotations_dir,
        split=args.split,
        num_workers=args.num_workers, # Pass directly
        overwrite=args.overwrite,     # Pass directly
        random_seed=args.random_seed, # Pass directly
        dataset_name=args.dataset_name # Pass directly
    )

    # # Override from args - NO LONGER NEEDED
    # if args.config_root_dir: config.root_dir = args.config_root_dir
    # if args.config_metadata_csv: config.metadata_csv = args.config_metadata_csv
    # if args.config_annotations_dir: config.annotations_dir = args.config_annotations_dir
    # if args.config_num_workers is not None: config.num_workers = args.config_num_workers
    # if args.config_overwrite: config.overwrite = True
    # # random_seed is handled directly above

    processor = ARKitScenesProcessor(config)
    processor.process_all_scenes()

if __name__ == "__main__":
    # Configure logging here if not handled by a higher-level script
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') # Added basicConfig like scannetpp
    main() 