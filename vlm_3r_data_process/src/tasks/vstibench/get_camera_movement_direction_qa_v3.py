import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
import random
from pathlib import Path
import logging

try:
    from ..base_qa_generator import BaseQAGenerator
    from ...utils.common_utils import from_options_to_mc_answer # Added import
    # Assuming question_templates are loaded by BaseQAGenerator or available in its scope
except ImportError:
    # Fallback for running script directly or different structure
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from tasks.base_qa_generator import BaseQAGenerator
    from utils.common_utils import from_options_to_mc_answer # Added import

logger = logging.getLogger(__name__)

# Define direction options - simplified to 4 cardinal horizontal directions
DIRECTION_OPTIONS = {
    "forward": "Forward",
    "backward": "Backward",
    "left": "Left",
    "right": "Right",
}

class CameraMovementDirectionQAGeneratorV3(BaseQAGenerator):
    """
    Generates sequence-level questions (2 choices) about the primary direction of camera translation
    relative to its starting orientation. This is V3.
    """
    def __init__(self):
        super().__init__()
        self.direction_threshold = 0.5  # Threshold for detecting significant movement
        self.num_choices = 2 # V3: 2 choices

    def get_default_args(self):
        """Return default arguments specific to camera movement direction QA (V3 - 2 choices)."""
        return {
            'question_template_name': 'VSTI_CAMERA_MOVEMENT_DIRECTION_TEMPLATE_V3',
            'num_subsample': 5,  # Number of QA pairs to generate per scene
            'question_type': 'camera_movement_direction_v3',
            'output_filename_prefix': 'qa_camera_movement_direction_v3',
            'num_choices': 2 # Explicitly pass num_choices in args as well
        }

    def _get_camera_displacement_vector(self, start_pose: np.ndarray, end_pose: np.ndarray) -> np.ndarray:
        """Calculates the camera displacement vector in world coordinates."""
        start_pos = start_pose[:3, 3]
        end_pos = end_pose[:3, 3]
        return end_pos - start_pos

    def _transform_vector_to_camera_space(self, vector: np.ndarray, camera_pose: np.ndarray) -> np.ndarray:
        """Transforms a vector from world space to the camera's local coordinate system."""
        rotation_matrix_c2w = camera_pose[:3, :3]
        # To transform a world vector to camera space, we use R_wc.T which is R_cw
        rotation_matrix_w2c = rotation_matrix_c2w.T
        return np.dot(rotation_matrix_w2c, vector)

    def _determine_movement_direction(self, displacement_vector_camera_space: np.ndarray) -> Optional[str]:
        """
        Determines camera movement direction based on displacement in camera space.
        Only considers Forward, Backward, Left, Right based on dominant X or Z axis movement.
        Assumes camera_space_displacement is [Right/Left(dx), Down/Up(dy), Forward/Backward(dz)]
        based on +X Right, +Y Down, +Z Forward convention. dy is ignored.
        """
        dx = displacement_vector_camera_space[0]
        # dy = displacement_vector_camera_space[1] # Y-axis movement (Down/Up) is now ignored
        dz = displacement_vector_camera_space[2]

        mag_x = abs(dx)
        mag_z = abs(dz)

        significant_x_movement = mag_x > self.direction_threshold
        significant_z_movement = mag_z > self.direction_threshold

        best_direction = None

        # Determine dominant direction if both axes have significant movement
        if significant_x_movement and significant_z_movement:
            if mag_z >= mag_x: # Prioritize Z-axis (forward/backward) if its magnitude is greater or equal
                best_direction = "forward" if dz > 0 else "backward"
            else: # X-axis (left/right) is dominant
                best_direction = "right" if dx > 0 else "left"
        elif significant_z_movement: # Only Z-axis movement is significant
            best_direction = "forward" if dz > 0 else "backward"
        elif significant_x_movement: # Only X-axis movement is significant
            best_direction = "right" if dx > 0 else "left"
        
        return best_direction # Returns None if no significant movement along X or Z

    def _calculate_potential_qas(self, scene_name: str, frames: List[Dict]) -> List[Dict]:
        """Calculates all potential QA pairs for a scene based on frame pairs."""
        potential_qas = []
        if len(frames) < 2:
            logger.debug(f"Scene {scene_name}: Not enough frames ({len(frames)}) for movement QA.")
            return potential_qas

        for i in range(len(frames) - 1):
            for j in range(i + 1, len(frames)):
                start_frame_data = frames[i]
                end_frame_data = frames[j]
                
                start_frame_id = start_frame_data.get("frame_id")
                end_frame_id = end_frame_data.get("frame_id")

                if start_frame_id is None or end_frame_id is None:
                    logger.warning(f"Scene {scene_name}: Frame ID missing. Skipping pair.")
                    continue

                start_pose_list = start_frame_data.get("camera_pose_camera_to_world")
                end_pose_list = end_frame_data.get("camera_pose_camera_to_world")

                if not start_pose_list or not end_pose_list:
                    logger.debug(f"Scene {scene_name}, Frames {start_frame_id}-{end_frame_id}: Missing pose. Skipping.")
                    continue
                
                try:
                    start_pose = np.array(start_pose_list)
                    end_pose = np.array(end_pose_list)
                    if start_pose.shape != (4,4) or end_pose.shape != (4,4) or np.isnan(start_pose).any() or np.isnan(end_pose).any():
                        logger.warning(f"Scene {scene_name}, Frames {start_frame_id}-{end_frame_id}: Invalid pose. Skipping.")
                        continue
                except Exception as e:
                    logger.warning(f"Scene {scene_name}, Frames {start_frame_id}-{end_frame_id}: Error parsing pose: {e}. Skipping.")
                    continue
                
                displacement_world = self._get_camera_displacement_vector(start_pose, end_pose)
                camera_space_displacement = self._transform_vector_to_camera_space(displacement_world, start_pose)
                
                direction_key = self._determine_movement_direction(camera_space_displacement)
                
                if direction_key and direction_key in DIRECTION_OPTIONS:
                    potential_qas.append({
                        "start_frame_id": start_frame_id,
                        "end_frame_id": end_frame_id,
                        "start_rank": i, "end_rank": j, # Store ranks
                        "direction_key": direction_key,
                        "question_template_name": self.args.question_template_name 
                                                # Will be VSTI_CAMERA_MOVEMENT_DIRECTION_TEMPLATE_V3
                    })
                elif direction_key: # Should not happen if _determine_movement_direction is correct
                    logger.debug(f"Scene {scene_name}, Frames {start_frame_id}-{end_frame_id}: Dir '{direction_key}' not in DIRECTION_OPTIONS. Skipping.")
        
        return potential_qas

    def _format_qa_item(self, scene_name: str, scene_info: Dict, qa_candidate: Dict, num_frames: int) -> Optional[Dict]:
        """Formats a single QA item from a potential candidate."""
        
        # qa_candidate["question_template_name"] holds the name like 'VSTI_CAMERA_MOVEMENT_DIRECTION_TEMPLATE_V3'
        # We expect BaseQAGenerator to have loaded this template string into self.question_template (singular)
        question_template_str = self.question_template
        
        if not question_template_str:
            # Use self.args.question_template_name for the error message as it's the configured name
            logger.error(f"Question template string (self.question_template) not loaded for template name '{self.args.question_template_name}' in scene {scene_name}.")
            return None

        correct_direction_key = qa_candidate["direction_key"]
        correct_option_text = DIRECTION_OPTIONS[correct_direction_key]
        
        all_option_texts = list(DIRECTION_OPTIONS.values())
        if correct_option_text not in all_option_texts: # Should not happen
            logger.error(f"Correct option '{correct_option_text}' for key '{correct_direction_key}' not in DIRECTION_OPTIONS values. Scene: {scene_name}")
            return None

        wrong_option_texts = [opt for opt in all_option_texts if opt != correct_option_text]
        current_num_choices = self.num_choices 
        
        if len(wrong_option_texts) < current_num_choices -1:
            logger.warning(f"Scene {scene_name}, Dir {correct_direction_key}: Not enough unique wrong options ({len(wrong_option_texts)}) to make {current_num_choices} choices. Skipping.")
            return None

        try:
            if not hasattr(self, 'answer_counts') or not hasattr(self, 'option_letters'):
                 logger.error(f"Scene {scene_name}: Missing 'answer_counts' or 'option_letters' attribute. Ensure BaseQAGenerator initializes them properly.")
                 return None

            selected_wrong_options = random.sample(wrong_option_texts, current_num_choices - 1)
            candidate_options_texts = [correct_option_text] + selected_wrong_options

            options_display_texts, mc_answer_letter, updated_answer_counts = from_options_to_mc_answer(
                candidate_options_texts,
                correct_option_text,
                self.answer_counts,
                self.option_letters[:current_num_choices]
            )
            self.answer_counts = updated_answer_counts

        except ValueError as e: 
            logger.warning(f"Scene {scene_name}, Dir {correct_direction_key}: Could not generate {current_num_choices} options using from_options_to_mc_answer. Error: {e}. Skipping.")
            return None
        except AttributeError as e: 
            logger.error(f"Scene {scene_name}: AttributeError during from_options_to_mc_answer (likely missing self.answer_counts or self.option_letters correctly initialized by BaseQAGenerator): {e}. Skipping.")
            return None

        start_rank = qa_candidate["start_rank"]
        end_rank = qa_candidate["end_rank"]
        start_frame_description = f"frame {start_rank + 1}"
        end_frame_description = f"frame {end_rank + 1} of {num_frames}"

        format_args = {
            "start_frame_description": start_frame_description,
            "end_frame_description": end_frame_description
        }
        choice_letters_list = self.option_letters[:current_num_choices]
        for i, opt_text in enumerate(options_display_texts):
            format_args[f"choice_{choice_letters_list[i].lower()}"] = opt_text

        try:
            final_question = question_template_str.format(**format_args)
        except KeyError as e:
            logger.error(f"Scene {scene_name}: Missing key {e} for template '{self.args.question_template_name}'. Args: {format_args}. Skipping.")
            return None
        
        formatted_options_output = [f"{choice_letters_list[i]}. {opt_text}" for i, opt_text in enumerate(options_display_texts)]

        return {
            "dataset": self.args.dataset,
            "scene_name": scene_name,
            "video_path": scene_info.get("video_path"), 
            "frame_indices": [qa_candidate["start_frame_id"], qa_candidate["end_frame_id"]],
            "question_type": self.args.question_type, 
            "question": final_question,
            "options": formatted_options_output, 
            "ground_truth": correct_option_text,
            "mc_answer": mc_answer_letter,  
            "question_template": self.args.question_template_name
        }

    def generate_scene_qa(self, scene_name: str, scene_info: Dict, frame_info_for_scene: Dict) -> List[Dict]:
        """Generates all QA pairs for a given scene."""
        frames = frame_info_for_scene.get("frames", [])
        if not frames:
            logger.info(f"Scene {scene_name}: No frames in frame_info. Skipping.")
            return []
        # Sort frames by frame_id to ensure consistent start/end pairs
        frames.sort(key=lambda x: x.get('frame_id', 0))


        potential_qas = self._calculate_potential_qas(scene_name, frames)
        if not potential_qas:
            logger.info(f"Scene {scene_name}: No potential QAs from _calculate_potential_qas. Skipping.")
            return []
        
        num_frames_for_description = len(frames) # Total number of frames for description

        num_to_sample = min(len(potential_qas), self.args.num_subsample)
        sampled_qa_candidates = random.sample(potential_qas, num_to_sample) if len(potential_qas) > num_to_sample else potential_qas
        
        logger.info(f"Scene {scene_name}: Calculated {len(potential_qas)} potential QAs, sampling {len(sampled_qa_candidates)} for V3.")

        formatted_qa_items = []
        for qa_candidate in sampled_qa_candidates:
            # scene_info is needed for video_path
            formatted_item = self._format_qa_item(scene_name, scene_info, qa_candidate, num_frames_for_description)
            if formatted_item:
                formatted_qa_items.append(formatted_item)
        
        logger.info(f"Scene {scene_name}: Formatted {len(formatted_qa_items)} QA items for V3.")
        return formatted_qa_items

if __name__ == "__main__":
    # Setup basic logging if no handlers are configured
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s [%(levelname)s] - %(message)s')

    generator = CameraMovementDirectionQAGeneratorV3()
    # The `run` method in BaseQAGenerator handles loading data, iterating scenes,
    # calling generate_scene_qa, and saving results.
    # Ensure question_templates (e.g. VSTI_CAMERA_MOVEMENT_DIRECTION_TEMPLATE_V3) are defined
    # and loaded by the BaseQAGenerator or accessible to self.question_templates.
    generator.run() 