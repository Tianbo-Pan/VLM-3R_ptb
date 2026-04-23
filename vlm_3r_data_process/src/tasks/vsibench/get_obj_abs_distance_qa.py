"""
The format of the qa file is as follows:
{
    "id": id,
    "scene_name": "xxxx",
    "question_type": "the question type a.k.a. the task at hand",
    "video_path": "path to the video",
    "category, first object": "category of object 1",
    "category, second object": "category of object 2",
    "question": "question",
    "options": "answer options",
    "ground_truth": "either the float or string answer",
    "mc_answer": "letter answer",
}
"""

from src.utils.common_utils import cal_3d_bbox_distance_between_categories, generate_multiple_choice
from src.tasks.base_qa_generator import BaseQAGenerator


class ObjAbsDistanceQAGenerator(BaseQAGenerator):
    def get_default_args(self):
        return {
            'question_template': "OBJ_ABS_DISTANCE_TEMPLATE",
            'num_subsample': 6,
            'question_type': 'object_abs_distance',
            'output_filename_prefix': 'qa_obj_abs_distance'
        }

    def generate_scene_qa(self, scene_name, scene_info, frame_info_for_scene):
        """Generate absolute distance QA pairs for a single scene."""
        scene_qa_list = []
        video_path = scene_info['video_path']
        single_count_objects = []

        for first_object_category in scene_info['object_counts']:
            scene_bbox_info = scene_info['object_bboxes']
            if scene_info['object_counts'][first_object_category] == 1:
                if single_count_objects:
                    for second_object_category in single_count_objects:
                        ground_truth = round(
                            cal_3d_bbox_distance_between_categories(scene_bbox_info[first_object_category], scene_bbox_info[second_object_category]), 1
                        )

                        if ground_truth < 0.2:
                            continue # Skip if distance is too small or error occurred
                        
                        # Use self.answer_counts and self.option_letters from base class
                        options, mc_answer, self.answer_counts = generate_multiple_choice(
                            ground_truth, answer_counts=self.answer_counts, option_letters=self.option_letters
                        )

                        if mc_answer == "E": # Error from generate_multiple_choice
                            continue

                        qa = {
                            # "id" will be assigned by the base class run method
                            'dataset': self.args.dataset,
                            "scene_name": scene_name,
                            "question_type": self.args.question_type,
                            "video_path": video_path,
                            "category, first object": first_object_category,
                            "category, second object": second_object_category,
                            "question": self.question_template.format(
                                object1=first_object_category,
                                object2=second_object_category,
                            ),
                            "options": [f"A. {options[0]}", f"B. {options[1]}", f"C. {options[2]}", f"D. {options[3]}"],
                            "ground_truth": ground_truth,
                            "mc_answer": mc_answer
                        }
                        scene_qa_list.append(qa)

                single_count_objects.append(first_object_category)
                
        return scene_qa_list


if __name__ == '__main__':
    # Removed argparse logic, handled by BaseQAGenerator
    generator = ObjAbsDistanceQAGenerator()
    generator.run()
