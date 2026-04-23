"""
The format of the qa file is as follows:
{
    "id": id,
    "scene_name": "xxxx",
    "question_type": "the question type a.k.a. the task at hand",
    "video_path": "path to the video",
    "category, positioning object": "category of the positioning object, where the person should be",
    "category, orienting object": "category of orienting object, where the person should be facing",
    "category, querying object": "category of querying object, the object whose relative direction should be determined",
    "question": "question",
    "options": "answer options",
    "ground truth": "either the float or string answer",
    "mc_answer": "letter answer",
}
"""
import math
import random
import itertools
import numpy as np
import open3d as o3d

from src.utils.common_utils import from_options_to_mc_answer

from .get_obj_rel_direction_v1_qa import ambiguous

from src.tasks.base_qa_generator import BaseQAGenerator


def get_gt(positioning_obj, orienting_obj, querying_obj, answer_counts, option_letters):
    positioning_obj_pos = np.array([positioning_obj['centroid']])
    orienting_obj_pos = np.array([orienting_obj['centroid']])
    querying_obj_pos = np.array([querying_obj['centroid']])
    
    querying_bbox = o3d.geometry.OrientedBoundingBox(
            center=np.array(querying_obj["centroid"]).astype(np.float64).reshape(3, 1),
            R=np.array(querying_obj["normalizedAxes"]).astype(np.float64).reshape(3, 3).T,
            extent=np.array(querying_obj["axesLengths"]).astype(np.float64).reshape(3, 1)
        )
    
    vertices = np.asarray(querying_bbox.get_box_points())
    querying_points = np.concatenate([querying_obj_pos, vertices], axis=0)
    
    orienting_vec = orienting_obj_pos - positioning_obj_pos
    querying_vecs = querying_points - positioning_obj_pos
    
    def calculate_angle(v1, v2s):
        dot_products = (v1 * v2s).sum(axis=1)
        mag_v1 = np.linalg.norm(v1, axis=1)
        mag_v2s = np.linalg.norm(v2s, axis=1)
        
        angles = np.arccos(dot_products / (mag_v1 * mag_v2s))
        
        crs_products = np.cross(v1, v2s)
        
        angles = np.where(crs_products >= 0., angles, 2 * math.pi - angles)
        return np.degrees(angles)

    angles = calculate_angle(orienting_vec[:, :2], querying_vecs[:, :2])
    
    quadrants = np.digitize(angles, bins=[0, 135, 225, 360])
    quadrant_of_centroid = quadrants[0]
    quadrant_of_vertices = quadrants[1:]

    if (quadrant_of_centroid != quadrant_of_vertices).sum() > 2: # if more than two vertices falls into a different quadrant compared to the centroid, skip this sample
        return "ambiguous", None, None, answer_counts
    
    ambiguity_threshold = 10
    boundaries = np.array([0, 135, 225, 360])
    if np.abs(angles[0] - boundaries).min() < ambiguity_threshold:
        return "ambiguous", None, None, answer_counts

    # randomly shuffle direction options
    directions = ['left', 'right']

    if quadrants[0] == 1:
        gt = "left"
    elif quadrants[0] == 2:
        return "ambiguous", None, None, answer_counts
    elif quadrants[0] == 3:
        gt = "right"
    else:
        raise ValueError
    
    options, mc_anser, answer_counts = from_options_to_mc_answer(directions, gt, answer_counts, option_letters)

    return gt, mc_anser, options, answer_counts
    
    
class ObjRelDirV3QAGenerator(BaseQAGenerator):
    def __init__(self):
        # Override init for 2 options
        super().__init__()
        self.answer_counts = {'A': 0, 'B': 0}
        self.option_letters = ['A', 'B']

    def get_default_args(self):
        return {
            'question_template': "OBJ_REL_DIRECTION_V3_TEMPLATE",
            'num_subsample': 6,
            'question_type': 'object_rel_direction_v3',
            'output_filename_prefix': 'qa_obj_rel_direction_v3'
        }

    def generate_scene_qa(self, scene_name, scene_info, frame_info_for_scene):
        """Generate relative direction V3 QA pairs for a single scene."""
        scene_qa_list = []
        video_path = scene_info['video_path']

        single_count_objects = []
        for category in scene_info['object_counts']:
            if scene_info['object_counts'][category] == 1:
                single_count_objects.append(category)

        if len(single_count_objects) < 3:
            return []

        object_triples = list(itertools.combinations(single_count_objects, 3))

        if len(object_triples) > self.args.num_subsample:
             object_triples = random.sample(object_triples, self.args.num_subsample)

        dist_lookup = {} 
        for object_triple in object_triples:
            
            obj_bbox_info = scene_info['object_bboxes']
            if ambiguous(object_triple, dist_lookup, obj_bbox_info):
                continue

            object_triple_permutations = [
                (object_triple[0], object_triple[1], object_triple[2]),
                (object_triple[0], object_triple[2], object_triple[1]),
                (object_triple[1], object_triple[0], object_triple[2])
            ]
            
            for positioning_obj, orienting_obj, querying_obj in object_triple_permutations:
                
                gt, mc_answer, options, self.answer_counts = get_gt(
                    positioning_obj=obj_bbox_info[positioning_obj][0],
                    orienting_obj=obj_bbox_info[orienting_obj][0],
                    querying_obj=obj_bbox_info[querying_obj][0],
                    answer_counts=self.answer_counts,
                    option_letters=self.option_letters
                )
                
                if gt == "ambiguous":
                    continue

                qa = {
                    # "id" will be assigned by the base class run method
                    "scene_name": scene_name,
                    'dataset': self.args.dataset,
                    "video_path": video_path,
                    "question_type": self.args.question_type,
                    "category, positioning object": positioning_obj,
                    "category, orienting object": orienting_obj,
                    "category, querying object": querying_obj,
                    "question": self.question_template.format(
                        positioning_object=positioning_obj,
                        orienting_object=orienting_obj,
                        querying_object=querying_obj
                    ),
                    # Only 2 options for V3
                    "options": [f"A. {options[0]}", f"B. {options[1]}"], 
                    "ground_truth": gt,
                    "mc_answer": mc_answer
                }
                scene_qa_list.append(qa)
        
        return scene_qa_list

if __name__ == '__main__':
    generator = ObjRelDirV3QAGenerator()
    generator.run()
