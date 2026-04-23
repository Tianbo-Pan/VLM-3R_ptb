from typing import List, Dict, Tuple, Any
import numpy as np
import os
from src.utils.common_utils import calculate_room_area, calculate_room_center
import json
import multiprocessing as mp
import logging
from dataclasses import dataclass, field
from collections import defaultdict
import open3d as o3d
import tqdm
import random
import torch
import argparse

# Import base classes
from src.base_processor import BaseProcessorConfig, AbstractSceneProcessor

# Define ScanNet++ specific categories (keep this constant within the module)
SCANNETPP_INST_SEG_CATEGORY = ['table', 'door', 'ceiling lamp', 'cabinet', 'blinds', 'curtain', 'chair', 'storage cabinet', 'office chair', 'bookshelf', 
                               'whiteboard', 'window', 'box', 'monitor', 'shelf', 'heater', 'kitchen cabinet', 'sofa', 'bed', 'trash can', 'book', 'plant', 
                               'blanket', 'tv', 'computer tower', 'refrigerator', 'jacket', 'sink', 'bag', 'picture', 'pillow', 'towel', 'suitcase', 
                               'backpack', 'crate', 'keyboard', 'rack', 'toilet', 'printer', 'poster', 'painting', 'microwave', 'shoes', 'socket', 'bottle', 
                               'bucket', 'cushion', 'basket', 'shoe rack', 'telephone', 'file folder', 'laptop', 'plant pot', 'exhaust fan', 'cup', 
                               'coat hanger', 'light switch', 'speaker', 'table lamp', 'kettle', 'smoke detector', 'container', 'power strip', 'slippers', 
                               'paper bag', 'mouse', 'cutting board', 'toilet paper', 'paper towel', 'pot', 'clock', 'pan', 'tap', 'jar', 'soap dispenser', 
                               'binder', 'bowl', 'tissue box', 'whiteboard eraser', 'toilet brush', 'spray bottle', 'headphones', 'stapler', 'marker']

logger = logging.getLogger(__name__)
# Base class or runner script handles basicConfig

# --- Configuration ---

@dataclass
class ScanNetPPProcessorConfig(BaseProcessorConfig):
    """Configuration specific to ScanNet++ metadata processing."""
    input_dir: str = "data/raw_data/scannetpp/data"
    scene_list_file: str = "data/raw_data/scannetpp/splits/nvs_sem_train.txt"
    # Video path construction is handled within the processor
    # Inherits save_dir, output_filename, num_workers, overwrite, random_seed

# --- Processor Implementation ---

class ScanNetPPProcessor(AbstractSceneProcessor):
    """Processor for ScanNet++ dataset metadata."""
    def __init__(self, config: ScanNetPPProcessorConfig):
        super().__init__(config)
        self.config: ScanNetPPProcessorConfig # Type hint for clarity
        # Set seeds if needed (already handled in base_processor if called directly)
        # random.seed(self.config.random_seed)
        # np.random.seed(self.config.random_seed)
        # torch.manual_seed(self.config.random_seed)

    def _calculate_room_metrics(self, scene_id: str) -> Tuple[float | None, List[float] | None]:
        """Calculates room size and center for a ScanNet++ scene using common utils."""
        ply_path = os.path.join(self.config.input_dir, scene_id, "scans", "mesh_aligned_0.05_semantic.ply")
        if not os.path.exists(ply_path):
            logger.error(f"Mesh file {ply_path} not found for scene {scene_id}. Cannot calculate room metrics.")
            return None, None
        try:
            # Load points directly for common utils
            ply_data = o3d.io.read_point_cloud(ply_path)
            points = np.asarray(ply_data.points)
            if points.shape[0] == 0:
                logger.warning(f"Mesh file {ply_path} contains no points for scene {scene_id}. Cannot calculate room metrics.")
                return None, None
                
            # Use common utility functions
            room_size = calculate_room_area(points)
            room_center = calculate_room_center(points) # Use the common center calculation
            return room_size, room_center
        except Exception as e:
            logger.error(f"Error calculating room metrics for {scene_id} from {ply_path}: {e}", exc_info=True)
            return None, None

    def _load_scene_list(self) -> List[str]:
        """Loads the list of scene IDs from the specified file."""
        logger.info(f"Loading scene list from: {self.config.scene_list_file}")
        try:
            with open(self.config.scene_list_file, "r") as f:
                scene_ids = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(scene_ids)} scene IDs.")
            return scene_ids
        except FileNotFoundError:
            logger.error(f"Scene list file not found: {self.config.scene_list_file}")
            return []

    def _process_single_scene(self, scene_id: str) -> Dict[str, Any] | None:
        """Processes a single ScanNet++ scene."""
        seg_ann_path = os.path.join(self.config.input_dir, scene_id, "scans", "segments_anno.json")
        video_path_relative = os.path.join(self.config.dataset_name, "videos", self.config.split, f"{scene_id}.mp4") # Construct video path

        try:
            # Check essential files first
            if not os.path.exists(seg_ann_path):
                logger.error(f"Annotation file {seg_ann_path} not found for scene {scene_id}. Skipping.")
                return None
            # if not os.path.exists(video_path):
            #      logger.warning(f"Video file {video_path} not found for scene {scene_id}. Proceeding without video path guarantee.")
                 # Decide if this is critical: return None or continue

            # Load annotations
            with open(seg_ann_path, "r") as f:
                seg_ann = json.load(f)
            
            object_counts = defaultdict(int)
            object_bboxes = defaultdict(list) # Store OBBs as provided
            
            if 'segGroups' not in seg_ann or not isinstance(seg_ann['segGroups'], list):
                 logger.warning(f"'segGroups' not found or invalid format in {seg_ann_path}. Skipping scene {scene_id}.")
                 return None

            for obj in seg_ann['segGroups']:
                label = obj.get('label')
                obb = obj.get('obb') # Assuming 'obb' key exists and contains the bounding box info
                
                if label and obb and label in SCANNETPP_INST_SEG_CATEGORY:
                    object_counts[label] += 1
                    object_bboxes[label].append(obb) # Store the raw OBB data
            
            if not object_bboxes:
                logger.warning(f"Scene {scene_id} has no valid annotated instances after filtering. Skipping.")
                return None
            # elif len(object_bboxes) == 1:
            #     logger.info(f"Scene {scene_id} has only one instance type.") # Keep this logging if useful
            
            # Calculate room metrics
            room_size, room_center = self._calculate_room_metrics(scene_id)
            if room_size is None or room_center is None:
                logger.warning(f"Could not calculate room metrics for scene {scene_id}. Proceeding with defaults or skipping.")
                # Decide handling: return None or use defaults like 0.0 and [0,0,0]
                room_size = 0.0
                room_center = [0.0, 0.0, 0.0]
                # return None 

            return {
                "video_path": video_path_relative, # Use the constructed path
                "room_size": room_size,
                "dataset": self.config.dataset_name,
                "scene_name": scene_id,
                "split": self.config.split,
                "room_center": room_center,
                "object_counts": dict(object_counts), # Convert to dict for JSON
                "object_bboxes": dict(object_bboxes), # Convert to dict for JSON
            }

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {seg_ann_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error processing scene {scene_id}: {e}", exc_info=True)
            return None


# --- Main Execution Logic ---

def main():
    parser = argparse.ArgumentParser(description="Process ScanNet++ dataset to generate metadata.")
    # Arguments for BaseProcessorConfig & ScanNetPPProcessorConfig
    parser.add_argument('--input_dir', type=str, default="data/raw_data/scannetpp/data", help='Root directory for ScanNet++ data.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the output JSON metadata.')
    parser.add_argument('--output_filename', type=str, required=True, help='Name of the output JSON file (e.g., scannetpp_train_metadata.json).')
    parser.add_argument('--scene_list_file', type=str, default="data/raw_data/scannetpp/splits/nvs_sem_train.txt", help='File containing the list of scene IDs for the split.')
    parser.add_argument('--split', type=str, default="train", help='Split to process (train, val, test).')
    parser.add_argument('--dataset_name', type=str, default="ScanNetpp", help='Dataset name.')
    # General processing arguments (matching BaseProcessorConfig defaults)
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for multiprocessing.')
    parser.add_argument('--overwrite', action='store_true', default=False, help='Allow overwriting existing output file.')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility.')

    args = parser.parse_args()

    # Create config object directly from arguments
    config = ScanNetPPProcessorConfig(
        input_dir=args.input_dir,
        scene_list_file=args.scene_list_file,
        save_dir=args.save_dir,
        output_filename=args.output_filename,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
        random_seed=args.random_seed,
        split=args.split,
        dataset_name=args.dataset_name
    )

    # Seed setting is handled in base class init
    # random.seed(config.random_seed)
    # np.random.seed(config.random_seed)
    # torch.manual_seed(config.random_seed)

    processor = ScanNetPPProcessor(config)
    processor.process_all_scenes()

if __name__ == "__main__":
    # Configure logging here if not handled by a higher-level script
    # Example basic config (adjust level and format as needed):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()