"""
The format of the qa file is as follows:
{
    "id": id,
    "scene_name": "xxxx",
    "question_type": "the question type a.k.a. the task at hand",
    "video_path": "path to the video",
    "category: "category of the object",
    "question": "question",
    "options": "answer options",
    "ground_truth": "either the float or string answer",
    "mc_answer": "letter answer",
}
"""

import random

from src.tasks.base_qa_generator import BaseQAGenerator
from src.utils.common_utils import from_options_to_mc_answer


def generate_choices(answer, answer_counts, option_letters):
    choices = [answer]
    
    # Create a set of possible offsets, excluding 0
    possible_offsets = set(range(-3, 4)) - {0}
    
    while len(choices) < 4:
        if not possible_offsets:
            # If we run out of offsets, extend the range
            possible_offsets = set(range(-5, 6)) - {0} - set(c - answer for c in choices)
        
        offset = random.choice(list(possible_offsets))
        new_choice = max(1, answer + offset)
        
        if new_choice not in choices:
            choices.append(new_choice)
            possible_offsets.remove(offset)
        else:
            # If the new_choice is already in choices, remove this offset
            possible_offsets.remove(offset)
    
    # Shuffle the choices
    # random.shuffle(choices)
    # correct_index = choices.index(answer)
    
    # options = ['A', 'B', 'C', 'D']
    # correct_option = options[correct_index]
    
    options, mc_answer, answer_counts = from_options_to_mc_answer(
        choices, answer, answer_counts, option_letters
    )
    
    return options, mc_answer, answer_counts


class ObjCountQAGenerator(BaseQAGenerator):
    def get_default_args(self):
        return {
            'question_template': "OBJ_COUNT_TEMPLATE",
            'num_subsample': 6,
            'question_type': 'object_counting',
            'output_filename_prefix': 'qa_obj_counting'
        }

    def generate_scene_qa(self, scene_name, scene_info, frame_info_for_scene):
        """Generate object count QA pairs for a single scene."""
        scene_qa_list = []
        video_path = scene_info['video_path']

        for category in scene_info['object_counts']:
            # skipping objects of count==1
            if scene_info['object_counts'][category] <= 1:
                continue

            ground_truth = scene_info['object_counts'][category]
            
            # Use self.answer_counts and self.option_letters from base class
            options, mc_answer, self.answer_counts = generate_choices(
                ground_truth, self.answer_counts, self.option_letters
            )
            
            qa = {
                # "id" will be assigned by the base class run method
                'dataset': self.args.dataset,
                "scene_name": scene_name,
                "question_type": self.args.question_type,
                "video_path": video_path,
                'category': category,
                "question": self.question_template.format(category=category),
                "options": [f"A. {options[0]}", f"B. {options[1]}", f"C. {options[2]}", f"D. {options[3]}"],
                "ground_truth": ground_truth,
                "mc_answer": mc_answer
            }
            scene_qa_list.append(qa)
            
        return scene_qa_list


if __name__ == '__main__':
    # Removed argparse logic, handled by BaseQAGenerator
    generator = ObjCountQAGenerator()
    generator.run()
