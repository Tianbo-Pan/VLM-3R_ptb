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
    "ground truth": "either the float or string answer",
    "mc_answer": "letter answer",
}
"""
from src.utils.common_utils import generate_multiple_choice
from src.tasks.base_qa_generator import BaseQAGenerator


class ObjSizeQAGenerator(BaseQAGenerator):
    def get_default_args(self):
        return {
            'question_template': "OBJ_SIZE_ESTIMATE_TEMPLATE",
            'num_subsample': 6,
            'question_type': 'object_size_estimation',
            'output_filename_prefix': 'qa_obj_size'
        }

    def generate_scene_qa(self, scene_name, scene_info, frame_info_for_scene):
        """
        Generates size comparison QA pairs for objects within a single scene.
        """
        scene_qa_list = []
        video_path = scene_info['video_path']

        for category in scene_info['object_counts']:
            if scene_info['object_counts'][category] != 1:
                continue # Consider only single-count objects

            bbox = scene_info['object_bboxes'][category][0]
            scale = bbox['axesLengths']
            
            length = max(scale)
            length_cm = round(length * 100)  # convert to cm
            
            # Use self.answer_counts and self.option_letters from base class
            options, mc_answer, self.answer_counts = generate_multiple_choice(
                length_cm, lower_bound=0.4, upper_bound=1.8, decimals=0,
                answer_counts=self.answer_counts, option_letters=self.option_letters
            )

            if mc_answer == "E": # Error from generate_multiple_choice
                continue
            
            qa = {
                # "id" will be assigned by the base class run method
                "scene_name": scene_name,
                'dataset': self.args.dataset,
                'question_type': self.args.question_type,
                "video_path": video_path,
                'category': category,
                "question": self.question_template.format(category=category),
                "options": [f"A. {options[0]}", f"B. {options[1]}", f"C. {options[2]}", f"D. {options[3]}"],
                "ground_truth": length_cm,
                "mc_answer": mc_answer
            }
            scene_qa_list.append(qa)
            
        return scene_qa_list


if __name__ == '__main__':
    generator = ObjSizeQAGenerator()
    generator.run()
