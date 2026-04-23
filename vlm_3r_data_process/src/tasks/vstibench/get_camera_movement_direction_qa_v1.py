import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
import random
from pathlib import Path
import logging

try:
    from ..base_qa_generator import BaseQAGenerator
    from ...utils.common_utils import from_options_to_mc_answer
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from tasks.base_qa_generator import BaseQAGenerator
    from utils.common_utils import from_options_to_mc_answer

logger = logging.getLogger(__name__)

# Define direction options - simplified to 4 cardinal horizontal directions
DIRECTION_OPTIONS = {
    "forward": "Forward",
    "backward": "Backward",
    "left": "Left",
    "right": "Right",
}

class CameraMovementDirectionQAGeneratorV1(BaseQAGenerator):
    """
    Generates sequence-level questions (4 choices) about the primary direction of camera translation
    relative to its starting orientation. This is V1.
    """
    def __init__(self):
        super().__init__()
        self.direction_threshold = 1.0
        self.num_choices = 4 # V1: 4 choices

    def get_default_args(self):
        """Return default arguments specific to camera movement direction QA (V1 - 4 choices)."""
        return {
            'question_template_name': 'VSTI_CAMERA_MOVEMENT_DIRECTION_TEMPLATE_V1',
            'num_subsample': 5,
            'question_type': 'camera_movement_direction_v1',
            'output_filename_prefix': 'qa_camera_movement_direction_v1',
            'num_choices': 4
        }

    def _get_camera_displacement_vector(self, start_pose: np.ndarray, end_pose: np.ndarray) -> np.ndarray:
        start_pos = start_pose[:3, 3]
        end_pos = end_pose[:3, 3]
        return end_pos - start_pos

    def _transform_vector_to_camera_space(self, vector: np.ndarray, camera_pose: np.ndarray) -> np.ndarray:
        rotation_matrix_c2w = camera_pose[:3, :3]
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
        potential_qas = []
        if len(frames) < 2: return potential_qas
        for i in range(len(frames) - 1):
            for j in range(i + 1, len(frames)):
                start_frame_data = frames[i]
                end_frame_data = frames[j]
                start_frame_id = start_frame_data.get("frame_id")
                end_frame_id = end_frame_data.get("frame_id")
                if start_frame_id is None or end_frame_id is None: continue
                start_pose_list = start_frame_data.get("camera_pose_camera_to_world")
                end_pose_list = end_frame_data.get("camera_pose_camera_to_world")
                if not start_pose_list or not end_pose_list: continue
                try:
                    start_pose = np.array(start_pose_list)
                    end_pose = np.array(end_pose_list)
                    if start_pose.shape != (4,4) or end_pose.shape != (4,4) or np.isnan(start_pose).any() or np.isnan(end_pose).any(): continue
                except Exception: continue
                displacement_world = self._get_camera_displacement_vector(start_pose, end_pose)
                camera_space_displacement = self._transform_vector_to_camera_space(displacement_world, start_pose)
                direction_key = self._determine_movement_direction(camera_space_displacement)
                if direction_key and direction_key in DIRECTION_OPTIONS:
                    potential_qas.append({
                        "start_frame_id": start_frame_id, "end_frame_id": end_frame_id,
                        "start_rank": i, "end_rank": j, # Store ranks
                        "direction_key": direction_key,
                        "question_template_name": self.args.question_template_name
                    })
        return potential_qas

    def _format_qa_item(self, scene_name: str, scene_info: Dict, qa_candidate: Dict, num_frames: int) -> Optional[Dict]:
        # qa_candidate["question_template_name"] holds the name like 'VSTI_CAMERA_MOVEMENT_DIRECTION_TEMPLATE_V1'
        # We expect BaseQAGenerator to have loaded this template string into self.question_template (singular)
        question_template_str = self.question_template
        
        if not question_template_str:
            # Use self.args.question_template_name for the error message as it's the configured name
            logger.error(f"Question template string (self.question_template) not loaded for template name '{self.args.question_template_name}' in scene {scene_name}.")
            return None

        correct_direction_key = qa_candidate["direction_key"]
        correct_option_text = DIRECTION_OPTIONS[correct_direction_key]
        all_option_texts = list(DIRECTION_OPTIONS.values())
        if correct_option_text not in all_option_texts:
            logger.error(f"Correct option '{correct_option_text}' for key '{correct_direction_key}' not in DIRECTION_OPTIONS values for scene {scene_name}.")
            return None
        
        wrong_option_texts = [opt for opt in all_option_texts if opt != correct_option_text]
        current_num_choices = self.num_choices
        
        if len(wrong_option_texts) < current_num_choices -1:
            logger.warning(f"Scene {scene_name}, Dir {correct_direction_key}: Not enough unique wrong options ({len(wrong_option_texts)}) to make {current_num_choices} choices. Skipping.")
            return None

        try:
            # from_options_to_mc_answer expects a list of all candidate options (including GT),
            # the ground truth text itself, current answer counts, and option letters.
            # It returns: shuffled_options_texts, mc_letter_answer, updated_answer_counts
            
            # Ensure self.answer_counts and self.option_letters are available (typically from BaseQAGenerator)
            if not hasattr(self, 'answer_counts') or not hasattr(self, 'option_letters'):
                 logger.error(f"Scene {scene_name}: Missing 'answer_counts' or 'option_letters' attribute. Ensure BaseQAGenerator initializes them properly.")
                 # Fallback initialization if critical, though base class should handle this.
                 # self.answer_counts = {chr(ord('A') + i): 0 for i in range(self.num_choices)} 
                 # self.option_letters = [chr(ord('A') + i) for i in range(self.num_choices)]
                 # For this fix, returning None to avoid further errors if not initialized.
                 return None

            # Construct the list of option texts to be passed to from_options_to_mc_answer
            # It should contain the correct option and a selection of wrong options.
            selected_wrong_options = random.sample(wrong_option_texts, current_num_choices - 1)
            candidate_options_texts = [correct_option_text] + selected_wrong_options
            # The from_options_to_mc_answer function will shuffle these internally.

            options_display_texts, mc_answer_letter, updated_answer_counts = from_options_to_mc_answer(
                candidate_options_texts, # This list will be shuffled by the function
                correct_option_text,    # The ground truth string
                self.answer_counts,     # The dict to track answer counts
                self.option_letters[:current_num_choices] # Relevant option letters e.g. [A, B, C, D]
            )
            self.answer_counts = updated_answer_counts # Update the instance's answer counts

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
            format_args[f"choice_{choice_letters_list[i].lower()}"] = opt_text # choice_a, choice_b etc.
        
        try:
            final_question = question_template_str.format(**format_args)
        except KeyError as e: 
            logger.error(f"Scene {scene_name}: Missing key {e} for template '{self.args.question_template_name}'. Args: {format_args}. Skipping.")
            return None
        
        # Prepare formatted options for the final QA item (e.g., "A. Forward", "B. Left")
        formatted_options_output = [f"{choice_letters_list[i]}. {opt_text}" for i, opt_text in enumerate(options_display_texts)]

        return {
            "dataset": self.args.dataset, "scene_name": scene_name,
            "video_path": scene_info.get("video_path"),
            "frame_indices": [qa_candidate["start_frame_id"], qa_candidate["end_frame_id"]],
            "question_type": self.args.question_type,
            "question": final_question, 
            "options": formatted_options_output, # Use the directly formatted list
            "ground_truth": correct_option_text, 
            "mc_answer": mc_answer_letter, # Directly use the letter from from_options_to_mc_answer
            "question_template": self.args.question_template_name, # Store the original template name used
            "start_frame_description": start_frame_description,
            "end_frame_description": end_frame_description
        }

    def generate_scene_qa(self, scene_name: str, scene_info: Dict, frame_info_for_scene: Dict) -> List[Dict]:
        frames = frame_info_for_scene.get("frames", [])
        if not frames: return []
        frames.sort(key=lambda x: x.get('frame_id', 0))
        potential_qas = self._calculate_potential_qas(scene_name, frames)
        if not potential_qas: return []
        num_frames_for_description = len(frames) # Total number of frames for description
        num_to_sample = min(len(potential_qas), self.args.num_subsample)
        sampled_qa_candidates = random.sample(potential_qas, num_to_sample) if len(potential_qas) > num_to_sample else potential_qas
        logger.info(f"Scene {scene_name}: Calculated {len(potential_qas)} potential QAs, sampling {len(sampled_qa_candidates)} for V1.")
        formatted_qa_items = []
        for qa_candidate in sampled_qa_candidates:
            formatted_item = self._format_qa_item(scene_name, scene_info, qa_candidate, num_frames_for_description)
            if formatted_item: formatted_qa_items.append(formatted_item)
        logger.info(f"Scene {scene_name}: Formatted {len(formatted_qa_items)} QA items for V1.")
        return formatted_qa_items

if __name__ == "__main__":
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    generator = CameraMovementDirectionQAGeneratorV1()
    generator.run() 