import os
import json
import argparse
import tqdm
import random
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import logging # Added logging
import numpy as np

try:
    from plyfile import PlyData
    PLYFILE_AVAILABLE = True
except ImportError:
    PLYFILE_AVAILABLE = False
    # logger.warning("plyfile library not found. PLY loading functionality will be disabled.") # Logged later if needed

import src.question_templates as prompt_templates
from src.utils.common_utils import load_scene_list

# Configure logging
logger = logging.getLogger(__name__)

class BaseQAGenerator(ABC):
    """
    Base class for generating QA pairs for different tasks.

    Handles common functionalities like argument parsing, data loading,
    scene iteration, subsampling, ID assignment, and saving results.

    Subclasses must implement the `generate_scene_qa` method
    and define default arguments. It now supports loading separate
    scene-level and frame-level metadata files, deriving paths
    from processed_data_path, dataset, and split_type.
    """
    def __init__(self):
        self.args = self._parse_args()
        self.scene_list = load_scene_list(self.args.split_path)

        assert self.args.processed_data_path is not None, "processed_data_path must be provided"
        assert self.args.dataset is not None, "dataset must be provided"
        assert self.args.split_type is not None, "split_type must be provided"

        # Construct paths and load metadata if base path, dataset, and split are provided
        base_processed_path = self.args.processed_data_path
        dataset_name = self.args.dataset
        split_type = self.args.split_type
        dataset_name_lower = dataset_name.lower() # For consistent filename convention

        # Construct metadata directory path
        self.metadata_path = os.path.join(base_processed_path, 'metadata', split_type)
        assert os.path.exists(self.metadata_path), f"Metadata path {self.metadata_path} does not exist"
        # Construct specific metadata file paths
        scene_meta_filename = f"{dataset_name_lower}_metadata_{split_type}.json"
        frame_meta_filename = f"{dataset_name_lower}_frame_metadata_{split_type}.json"
        self.scene_meta_path = os.path.join(self.metadata_path, scene_meta_filename)
        self.frame_meta_path = os.path.join(self.metadata_path, frame_meta_filename)

        # Load annotations using constructed paths
        self.scene_annos = self._load_json(self.scene_meta_path)
        self.frame_annos = self._load_json(self.frame_meta_path)

        # Construct paths for other processed data types, including dataset name
        self.color_data_path = os.path.join(base_processed_path, 'color', split_type)
        self.depth_data_path = os.path.join(base_processed_path, 'depth', split_type)
        self.instance_data_path = os.path.join(base_processed_path, 'instance', split_type)
        self.intrinsic_data_path = os.path.join(base_processed_path, 'intrinsic', split_type)
        self.point_cloud_path = os.path.join(base_processed_path, 'point_cloud', split_type)
        self.pose_data_path = os.path.join(base_processed_path, 'pose', split_type)
        logger.info(f"Processed data paths constructed based on: {base_processed_path}, Dataset: {dataset_name}, Split: {split_type}")
        logger.info(f"Scene metadata path: {self.scene_meta_path}")
        logger.info(f"Frame metadata path: {self.frame_meta_path}")


        # Ensure annotations loaded successfully (or log warning if paths weren't constructed)
        if self.scene_meta_path and not self.scene_annos:
             logger.warning(f"Scene metadata from {self.scene_meta_path} is empty or failed to load.")
        if self.frame_meta_path and not self.frame_annos:
             logger.warning(f"Frame metadata from {self.frame_meta_path} is empty or failed to load.")

        self.question_template = None if self.args.question_template is None else getattr(prompt_templates, self.args.question_template)
        self.all_qa_list = []
        # Answer counts will be aggregated after parallel processing
        self.answer_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0} # Example for MC, adapt if needed
        self.option_letters = ['A', 'B', 'C', 'D'] # Example

    def _load_json(self, file_path):
        """Loads JSON data from a file."""
        if not file_path:
            logger.warning("No path provided for a metadata file.")
            return None
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Successfully loaded metadata from {file_path}")
            return data
        except FileNotFoundError:
            logger.error(f"Metadata file not found: {file_path}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred loading {file_path}: {e}")
            return None

    def _load_scene_ply(self, scene_name):
        """
        Loads the preprocessed PLY point cloud file for a given scene.

        Constructs the path, loads the file using plyfile, checks for the
        'vertex' element, and returns its data. Handles errors gracefully.

        Args:
            scene_name (str): The name of the scene.

        Returns:
            numpy.ndarray or None: The vertex data from the PLY file if successful,
                                   otherwise None.
        """
        if not PLYFILE_AVAILABLE:
            logger.error("Cannot load PLY file: 'plyfile' library is not installed. Please run 'pip install plyfile'.")
            return None

        if not self.point_cloud_path:
            logger.error(f"Scene {scene_name}: point_cloud_path is not set in the BaseQAGenerator instance. Cannot locate PLY file.")
            return None

        ply_path = os.path.join(self.point_cloud_path, f"{scene_name}.ply")

        if not os.path.exists(ply_path):
            logger.warning(f"Scene {scene_name}: Preprocessed PLY file not found at {ply_path}. Skipping.")
            return None

        try:
            plydata = PlyData.read(ply_path)
            if 'vertex' not in plydata:
                logger.error(f"Scene {scene_name}: PLY file {ply_path} does not contain a 'vertex' element. Skipping.")
                return None
            vertex_data = plydata['vertex'].data
            logger.debug(f"Scene {scene_name}: Successfully loaded vertex data from {ply_path}.")
            return vertex_data
        except FileNotFoundError:
             # This case is technically covered by os.path.exists, but kept for safety
             logger.warning(f"Scene {scene_name}: Could not find preprocessed PLY file at {ply_path} during read attempt. Skipping.")
             return None
        except Exception as e:
            logger.error(f"Scene {scene_name}: Error loading or reading PLY data from {ply_path}: {e}", exc_info=True)
            return None

    def _identify_unique_instances(self, scene_name, scene_info, vertex_data):
        """
        Identifies instances that are unique by category in scene metadata
        AND are present in the loaded PLY vertex data.

        Relies on 'object_counts' and 'object_bboxes' in scene_info, and
        'instance_id' within vertex_data.

        Args:
            scene_name (str): The name of the scene (for logging).
            scene_info (dict): Annotation information for the scene from scene meta.
            vertex_data (numpy.ndarray): Vertex data loaded from the PLY file,
                                         expected to have an 'instance_id' field.

        Returns:
            list[dict]: A list of dictionaries, each containing 'instance_id' (int)
                        and 'category_name' (str) for unique instances found in both
                        metadata and PLY data. Returns an empty list if requirements
                        are not met or no such instances are found.
        """
        if vertex_data is None:
            logger.warning(f"Scene {scene_name}: Cannot identify unique instances as vertex_data is None.")
            return []
        if 'instance_id' not in vertex_data.dtype.names:
            logger.error(f"Scene {scene_name}: Cannot identify unique instances as 'instance_id' field is missing in PLY vertex data.")
            return []

        # --- Get instance IDs present in the loaded PLY data (ignore <= 0) ---
        point_instance_ids = vertex_data['instance_id'].astype(int)
        valid_ply_mask = point_instance_ids > 0
        if not np.any(valid_ply_mask):
            logger.warning(f"Scene {scene_name}: No valid instance IDs (>0) found in the PLY file vertex data. Cannot find unique instances.")
            return []
        ply_instance_ids_set = set(point_instance_ids[valid_ply_mask])

        # --- Find unique instances based on metadata ---
        object_counts = scene_info.get('object_counts', {})
        object_bboxes = scene_info.get('object_bboxes', {})
        if not object_counts or not object_bboxes:
            logger.warning(f"Scene {scene_name}: Missing 'object_counts' or 'object_bboxes' in scene_info. Cannot identify unique instances from metadata.")
            return []

        unique_category_instance_info = []
        processed_instances = set() # Avoid duplicates if metadata is inconsistent

        for category_name, count in object_counts.items():
            if count == 1:
                # Found a category with only one instance according to metadata
                if category_name not in object_bboxes or not object_bboxes[category_name]:
                    logger.warning(f"Scene {scene_name}: Unique category '{category_name}' found in counts but missing/empty in bboxes metadata. Skipping this category.")
                    continue

                try:
                    instance_details = object_bboxes[category_name][0]
                    instance_id = instance_details.get('instance_id')

                    if instance_id is None:
                        logger.warning(f"Scene {scene_name}: Missing 'instance_id' key for unique category '{category_name}' in bboxes metadata. Skipping.")
                        continue
                    instance_id = int(instance_id) # Ensure integer type

                except (IndexError, KeyError, TypeError, ValueError) as e:
                    logger.warning(f"Scene {scene_name}: Error accessing instance details for unique category '{category_name}' from metadata: {e}. Skipping.")
                    continue

                if instance_id in processed_instances:
                    continue # Already processed this instance ID

                # --- Cross-reference with PLY data ---
                if instance_id in ply_instance_ids_set:
                    unique_category_instance_info.append({
                        "instance_id": instance_id,
                        "category_name": category_name
                    })
                    processed_instances.add(instance_id)
                # else: # Optional: Log if a unique instance from metadata is NOT in the PLY
                    # logger.debug(f"Scene {scene_name}: Instance ID {instance_id} for unique category '{category_name}' from metadata not found in PLY vertex data.")

        if not unique_category_instance_info:
            logger.info(f"Scene {scene_name}: No unique instances identified from metadata that also exist in the PLY data.")
            return []

        logger.debug(f"Scene {scene_name}: Found {len(unique_category_instance_info)} unique instances from metadata present in PLY: {unique_category_instance_info}")
        return unique_category_instance_info

    @abstractmethod
    def get_default_args(self):
        """Return a dictionary of default arguments for this task."""
        pass

    @abstractmethod
    def generate_scene_qa(self, scene_name, scene_info, frame_info_for_scene):
        """
        Generate QA pairs for a single scene, using both scene and frame info.

        Args:
            scene_name (str): The name of the scene.
            scene_info (dict): Annotation information for the scene from scene meta.
            frame_info_for_scene (dict): Annotation info for the scene from frame meta.

        Returns:
            list: A list of QA dictionaries for the scene.
        """
        pass

    def _parse_args(self):
        parser = argparse.ArgumentParser(description='Base QA Generator arg parser')

        # Common arguments
        parser.add_argument('--split_path', type=str, help='Path to the scene split file.')
        parser.add_argument('--split_type', type=str, help='Type of split to use (e.g., "train", "val", "test").')
        # Updated metadata arguments
        parser.add_argument('--processed_data_path', type=str, required=True, help='Path to the processed data directory (e.g., containing point clouds).')
        parser.add_argument('--output_dir', type=str, help='Directory to save the output QA JSON file.')
        parser.add_argument('--dataset', type=str, help='Name of the dataset.')
        parser.add_argument('--question_template', type=str, help='Name of the question template constant from question_templates.py.')
        parser.add_argument('--num_subsample', type=int, default=6, help='Max number of questions to generate per scene.') # Changed wording slightly
        parser.add_argument('--question_type', type=str, help='Identifier for the type of question being generated (e.g., "relative_depth").')
        parser.add_argument('--output_filename_prefix', type=str, help='Prefix for the output JSON filename (e.g., "qa_rel_depth").')
        parser.add_argument('--num_workers', type=int, default=1, help='Number of worker processes for parallel scene processing.')

        # Add default arguments from subclass using a temporary instance
        # This assumes subclasses have a simple __init__ or none at all before calling super()
        temp_instance_for_defaults = self.__class__.__new__(self.__class__)
        default_args = temp_instance_for_defaults.get_default_args()
        parser.set_defaults(**default_args)


        parsed_args = parser.parse_args()

        # Log the arguments being used
        logger.info("Using arguments:")
        for arg, value in vars(parsed_args).items():
            logger.info(f"  {arg}: {value}")

        return parsed_args

    def _process_scene(self, scene_name):
        """Processes a single scene to generate QA pairs."""
        scene_name = scene_name.strip()

        if self.scene_annos is not None and scene_name in self.scene_annos:
            scene_info = self.scene_annos[scene_name]
        else:
            logger.warning(f"Scene {scene_name} not found in scene meta (path: {self.scene_meta_path}). Skipping.")
            scene_info = {}
        
        if self.frame_annos is not None and scene_name in self.frame_annos:
            frame_info_for_scene = self.frame_annos[scene_name]
        else:
            logger.warning(f"Scene {scene_name} not found in frame meta (path: {self.frame_meta_path}). Skipping.")
            frame_info_for_scene = {}

        local_answer_counts = Counter()
        # Pass both scene and frame info to the generation method
        scene_qa_list = self.generate_scene_qa(scene_name, scene_info, frame_info_for_scene)

        # Subsample questions per scene if more were generated than requested
        if len(scene_qa_list) > self.args.num_subsample:
            scene_qa_list = random.sample(scene_qa_list, self.args.num_subsample)

        # Count MC answers for this scene's results (if applicable)
        mc_answer_key = 'mc_answer'
        for qa in scene_qa_list:
            if mc_answer_key in qa:
                mc_answer_value = qa[mc_answer_key]
                # Expecting a single letter string like "A", "B", etc.
                if isinstance(mc_answer_value, str) and mc_answer_value in self.option_letters:
                    local_answer_counts[mc_answer_value] += 1
                else:
                    logger.warning(
                        f"Scene {scene_name}: Unexpected format or value for mc_answer: '{mc_answer_value}' (type: {type(mc_answer_value)}). Expected one of {self.option_letters}. Counting raw value."
                    )
                    # Attempt to count the raw value anyway, converting to string if needed
                    local_answer_counts[str(mc_answer_value)] += 1

        return scene_qa_list, local_answer_counts

    def run(self):
        # Basic logging setup if not already configured by the user
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        aggregated_results = []
        total_answer_counts = Counter()

        # Filter scene list based on available annotations (ensure annotations were loaded)
        if not self.scene_annos and not self.frame_annos:
            logger.error("Scene and frame annotations not loaded. Cannot process scenes. Check metadata paths and files. Exiting.")
            return

        # Use ProcessPoolExecutor for parallel processing
        num_workers = max(1, self.args.num_workers) # Ensure at least 1 worker
        logger.info(f"Starting QA generation with {num_workers} worker(s) for {len(self.scene_list)} scenes...")

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Map scenes to the processing function
            futures = {executor.submit(self._process_scene, scene_name): scene_name for scene_name in self.scene_list}

            # Collect results as they complete, with progress bar
            for future in tqdm.tqdm(as_completed(futures), total=len(self.scene_list), desc="Processing Scenes"):
                scene_name = futures[future]
                try:
                    scene_qa_list, scene_answer_counts = future.result()
                    if scene_qa_list: # Only extend if results were generated
                        aggregated_results.extend(scene_qa_list)
                    total_answer_counts.update(scene_answer_counts)
                except Exception as exc:
                    logger.exception(f'Scene {scene_name} generated an exception: {exc}') # Log exception with traceback
                # else:
                #     logger.debug(f'Scene {scene_name} processed successfully.') # Use debug level

        # Re-assign IDs sequentially after collecting all results
        logger.info("Assigning final IDs...")
        for i, qa in enumerate(aggregated_results):
            qa["id"] = i # Assign a unique, sequential ID

        self.all_qa_list = aggregated_results
        # Convert Counter to dict for consistent output
        self.answer_counts = dict(sorted(total_answer_counts.items())) # Sort for consistent reporting


        logger.info(f"Total number of QA pairs generated: {len(self.all_qa_list)}")
        logger.info(f"Final MC Answer Distribution: {self.answer_counts}")
        self._save_results()

    def _save_results(self):
        output_filename = f"{self.args.output_filename_prefix}.json"
        qa_path = os.path.join(self.args.output_dir, self.args.split_type, output_filename)
        os.makedirs(os.path.dirname(qa_path), exist_ok=True)
        logger.info(f"Saving QA pairs to {qa_path}...")
        try:
            with open(qa_path, "w") as f:
                json.dump(self.all_qa_list, f, indent=4)
            logger.info(f"Successfully saved QA pairs to {qa_path}")
        except IOError as e:
            logger.error(f"Failed to write output file {qa_path}: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during saving: {e}")


# Example usage (in a subclass file):
# if __name__ == '__main__':
#     # Setup logging here if needed
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#     my_generator = MySpecificQAGenerator()
#     my_generator.run() 