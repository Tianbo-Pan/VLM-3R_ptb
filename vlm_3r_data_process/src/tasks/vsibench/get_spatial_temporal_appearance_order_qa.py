import random
import itertools
import copy
import logging

from src.utils.common_utils import from_options_to_mc_answer
from src.tasks.base_qa_generator import BaseQAGenerator

from src.utils import trajectory_utils

SAVE_MARGIN = 1

# Configure logging
logger = logging.getLogger(__name__)

class SpatialTemporalOrderQAGenerator(BaseQAGenerator):
    # Removed __init__ and _load_frame_metadata - frame info is passed by base runner

    def get_default_args(self):

        return {
            'split_path': "../data/splits/scannetpp_coreset_anno_filtered_0828.txt",
            'meta_info_path': "../data/meta_info/scannetpp_coreset_anno_filtered_0925.json",
            'frame_metadata_path': "../data/ScanNet/scannet_frame_metadata.json",
            'output_dir': "../data/qa_pairs",
            'tag': "scannet_10_test",
            'dataset': "scannetpp",
            'question_template': "OBJ_APPEARANCE_ORDER_TEMPLATE",
            'num_subsample': 6,
            'question_type': 'obj_appearance_order',
            'output_filename_prefix': 'qa_obj_appearance_order'
        }

    # Modified signature to match base class
    def generate_scene_qa(self, scene_name, scene_info, frame_info_for_scene):
        """Generate spatial-temporal appearance order QA pairs for a single scene."""
        scene_qa_list = []
        dataset_name = scene_info['dataset']
        video_path = scene_info['video_path']

        # Use frame_info_for_scene passed as argument
        if not frame_info_for_scene:
             logger.warning(f"Frame metadata not provided for scene {scene_name}. Skipping.")
             return []

        frames_list = frame_info_for_scene.get('frames', [])
        if not frames_list:
             logger.warning(f"No frames found in metadata for scene {scene_name}. Skipping.")
             return []

        # --- Generate instance_id_to_category mapping from scene_info ---
        # Based on scannetv2_utils.py snippet, assume scene_info['object_bbox'] structure
        instance_to_category_map = {}
        object_bboxes_info = scene_info.get('object_bboxes', {})
        if not object_bboxes_info:
             logger.warning(f"Could not find 'object_bboxes' in scene_info for {scene_name}. Cannot map instances to categories. Skipping.")
             return []

        for category_name, bbox_list in object_bboxes_info.items():
            for bbox_details in bbox_list:
                instance_id = bbox_details.get('instance_id')
                if instance_id is not None:
                    # Map instance_id (as string, consistent with previous attempt) to category_name
                    instance_to_category_map[str(instance_id)] = category_name
                else:
                    logger.warning(f"Missing 'instance_id' in bbox details for category {category_name} in scene {scene_name}.")

        if not instance_to_category_map:
            logger.error(f"Failed to generate instance_id_to_category map for {scene_name}. Skipping.")
            return []
        # --- End mapping generation ---


        valid_category_names = list(scene_info.get('object_counts', {}).keys())
        if not valid_category_names:
            logger.warning(f"No valid category names found in object_counts for {scene_name}. Skipping.")
            return []

        # Calculate appearance times using frame metadata and the generated map
        category_time_dict = calculate_first_appearance_time(
            dataset_name, frames_list, valid_category_names, instance_to_category_map
        )

        valid_timed_categories = list(category_time_dict.keys())
        if len(valid_timed_categories) < 4:
            # logger.warning(f"Scene {scene_name} has less than 4 categories with appearance time.") # Reduce verbosity
            return []
        
        candidate_cates_list = list(itertools.combinations(valid_timed_categories, 4))
        random.shuffle(candidate_cates_list)
        
        scene_qa_counter = 0
        for choice_categories_tuple in candidate_cates_list:
            choice_categories = list(choice_categories_tuple)
            # Check time difference constraint
            times_valid = True
            for i in range(4):
                for j in range(i + 1, 4):
                    if abs(category_time_dict[choice_categories[i]] - category_time_dict[choice_categories[j]]) <= SAVE_MARGIN:
                        times_valid = False
                        break
                if not times_valid:
                    break
            
            if not times_valid:
                continue

            # If times are valid, generate QA
            sorted_list = sorted(choice_categories, key=lambda x: category_time_dict[x])
            gt = ', '.join(sorted_list)
            
            # Generate 3 other options by shuffling
            options_set = {gt}
            attempts = 0
            shuffled_options = []
            while len(shuffled_options) < 3 and attempts < 100: # Limit attempts
                shuffled_cats = copy.deepcopy(choice_categories)
                random.shuffle(shuffled_cats)
                shuffled_str = ', '.join(shuffled_cats)
                if shuffled_str not in options_set:
                    shuffled_options.append(shuffled_str)
                    options_set.add(shuffled_str)
                attempts += 1
            
            if len(shuffled_options) < 3:
                # print(f"Warning: Could not generate 3 unique options for {scene_name}, {choice_categories}. Skipping.")
                continue # Cannot generate enough unique options

            all_options_for_mc = [gt] + shuffled_options
            
            # Use self.answer_counts and self.option_letters
            final_options, mc_answer, self.answer_counts = from_options_to_mc_answer(
                all_options_for_mc, gt, self.answer_counts, self.option_letters
            )

            # Shuffle the categories before formatting the question to avoid leaking order information
            question_mention_order = copy.deepcopy(choice_categories)
            random.shuffle(question_mention_order)

            qa = {
                # "id" will be assigned by the base class run method
                "scene_name": scene_name,
                'dataset': dataset_name,
                "question_type": self.args.question_type,
                "video_path": video_path,
                # "category" key seems less relevant here, maybe list the involved categories?
                "categories_involved": choice_categories, # Keep original list for reference
                "question": self.question_template.format(
                    choice_a=question_mention_order[0], choice_b=question_mention_order[1],
                    choice_c=question_mention_order[2], choice_d=question_mention_order[3]
                ),
                "options": [f"A. {final_options[0]}", f"B. {final_options[1]}", f"C. {final_options[2]}", f"D. {final_options[3]}"],
                "ground_truth": gt,
                "mc_answer": mc_answer
            }
            scene_qa_list.append(qa)
            scene_qa_counter += 1
            
            if scene_qa_counter >= self.args.num_subsample:
                break # Stop after generating enough QAs for the scene
        
        return scene_qa_list
        

def calculate_first_appearance_time(dataset_name, frames_list, valid_category_names, instance_to_category_map):
    """
    Calculates the first appearance time (in seconds) for each valid category
    based on bounding boxes in frame metadata.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'scannetpp') for FPS lookup.
        frames_list (list): List of frame dictionaries from scene metadata.
        valid_category_names (list): List of category names considered valid for this scene.
        instance_to_category_map (dict): Mapping from instance ID (str) to category name (str).

    Returns:
        dict: Dictionary mapping category names to their first appearance time in seconds.
    """
    category_time_dict = {}

    # Sort frames by frame_id to ensure chronological processing
    sorted_frames = sorted(frames_list, key=lambda x: x.get('frame_id', float('inf')))

    # Get FPS for time calculation
    try:
        fps = trajectory_utils.FRAME_CATEGORY_FPS_LOOKUP_TABLE[dataset_name]
    except KeyError:
        logger.error(f"FPS not found for dataset '{dataset_name}'. Cannot calculate time.")
        return {} # Cannot calculate time without FPS

    for frame_info in sorted_frames:
        frame_id = frame_info.get('frame_id')
        bboxes_2d = frame_info.get('bboxes_2d', [])

        if frame_id is None:
            logger.warning(f"Frame missing 'frame_id'. Skipping.")
            continue

        for bbox_info in bboxes_2d:
            # Instance ID from frame metadata is 0-based integer
            instance_id_int = bbox_info.get('instance_id')
            if instance_id_int is None:
                continue

            # Map instance ID (convert to string for lookup) to category name
            # Instance IDs in frame metadata are 0-based, while IDs in object_bbox might be different (e.g., 1-based?).
            # Need to confirm if the IDs match or need adjustment.
            # Assuming the instance IDs in frame metadata bboxes directly correspond to the instance IDs in scene_info['object_bbox'] *after* converting to string.
            # If frame metadata instance_id is 0-based and scene_info['object_bbox'] instance_id is 1-based, we might need:
            # category_name = instance_to_category_map.get(str(instance_id_int + 1))
            # For now, assume direct mapping with string conversion:
            category_name = instance_to_category_map.get(str(instance_id_int))

            if category_name and category_name in valid_category_names:
                # If this category hasn't been seen yet, record its first appearance time
                if category_name not in category_time_dict:
                    # Calculate time in seconds
                    time_sec = round(frame_id / fps)
                    category_time_dict[category_name] = time_sec

                    # Optimization: If all valid categories are found, we could potentially break early,
                    # but sorting frames first is generally safer and cleaner.

    return category_time_dict


if __name__ == '__main__':
    # Setup basic logging for the main execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    generator = SpatialTemporalOrderQAGenerator()
    generator.run()
