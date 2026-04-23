"""
The format of the qa file is as follows:
{
    "id": id,
    "scene_name": "xxxx",
    "question_type": "the question type a.k.a. the task at hand",
    "video_path": "path to the video",
    "question": "question",
    "options": "answer options",
    "ground truth": "either the float or string answer",
    "mc_answer": "letter answer",
}
"""

from src.tasks.base_qa_generator import BaseQAGenerator
from src.utils.common_utils import generate_multiple_choice


class RoomSizeQAGenerator(BaseQAGenerator):
    def get_default_args(self):
        return {
            'question_template': "ROOM_SIZE_TEMPLATE",
            'num_subsample': 1, # Only one question per scene for room size
            'question_type': 'room_size_estimation',
            'output_filename_prefix': 'qa_room_size'
        }

    def generate_scene_qa(self, scene_name, scene_info, frame_info_for_scene):
        """Generate room size QA pairs for a single scene."""
        scene_qa_list = []
        video_path = scene_info['video_path']

        ground_truth = round(scene_info['room_size'], 1)
        
        # Use self.answer_counts and self.option_letters from base class
        options, mc_answer, self.answer_counts = generate_multiple_choice(
            ground_truth, answer_counts=self.answer_counts, option_letters=self.option_letters
        )

        if mc_answer == "E": # Error from generate_multiple_choice
            return [] # Skip this scene if choices are ambiguous

        qa = {
            # "id" will be assigned by the base class run method
            "scene_name": scene_name,
            'dataset': self.args.dataset,
            "question_type": self.args.question_type,
            "video_path": video_path,
            "question": self.question_template, # Template doesn't require formatting
            "options": [f"A. {options[0]}", f"B. {options[1]}", f"C. {options[2]}", f"D. {options[3]}"],
            "ground_truth": ground_truth,
            "mc_answer": mc_answer
        }
        scene_qa_list.append(qa)
        
        # Note: Base class subsampling will apply, but since we return only 1 QA,
        # and default subsample is 1, it effectively keeps this single QA.
        return scene_qa_list


if __name__ == '__main__':
    generator = RoomSizeQAGenerator()
    generator.run()
