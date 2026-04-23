import os
import logging
import numpy as np
import open3d as o3d
import random
import itertools

try:
    from ..base_qa_generator import BaseQAGenerator
    import question_templates as prompt_templates
    from ...utils.common_utils import load_scene_list, from_options_to_mc_answer, generate_multiple_choice
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from tasks.base_qa_generator import BaseQAGenerator
    import question_templates as prompt_templates
    from utils.common_utils import load_scene_list, from_options_to_mc_answer, generate_multiple_choice

logger = logging.getLogger(__name__)

class ObjectObjectRelativePositionNFQAGenerator(BaseQAGenerator):
    """
    Generates frame-level questions asking for the spatial relationship
    (Near/Far) between two objects that are unique
    by category within that specific frame.
    Camera coordinates assumed: +X Right, +Y Down, +Z Forward.
    """
    def __init__(self):
        super().__init__()
        self.options_nf = ['Near', 'Far'] # Near, Far (relative to camera)
        self.option_letters = ['A', 'B']
        self.answer_counts = {'nf': {'A': 0, 'B': 0}} # Simplified for NF only
        self.position_threshold = 0.15 # meters

    def get_default_args(self):
        """Return default arguments specific to object-object Near/Far relative position QA."""
        return {
            'question_template': 'VSTI_OBJ_OBJ_REL_POS_NF_TEMPLATE', # Near/Far template
            'num_subsample': 9, 
            'question_type_prefix': 'obj_obj_relative_pos_nf', # Specific to NF
            'output_filename_prefix': 'qa_obj_obj_relative_pos_nf', # Specific to NF
        }

    def _get_camera_pose_matrices(self, pose_list):
        """Extracts T_c2w and T_w2c from pose list."""
        try:
            T_c2w = np.array(pose_list)
            if T_c2w.shape != (4, 4) or np.isnan(T_c2w).any(): return None, None
            R_c2w = T_c2w[:3, :3]
            t_c2w = T_c2w[:3, 3]
            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ t_c2w
            T_w2c = np.identity(4)
            T_w2c[:3, :3] = R_w2c
            T_w2c[:3, 3] = t_w2c
            return T_c2w, T_w2c
        except Exception as e:
            logger.error(f"Error getting camera pose matrices: {e}", exc_info=True)
            return None, None

    def _get_object_bbox_info(self, object_bboxes_dict, category_name, instance_id):
        """Gets the full bounding box info of a specific object instance."""
        try:
            target_instance_id_str = str(instance_id)
            if category_name in object_bboxes_dict:
                for bbox_info in object_bboxes_dict[category_name]:
                    metadata_instance_id = bbox_info.get("instance_id")
                    if metadata_instance_id is None:
                        continue
                    metadata_instance_id_str = str(metadata_instance_id)
                    if metadata_instance_id_str == target_instance_id_str:
                        if "centroid" in bbox_info:
                             return bbox_info
                        else:
                            logger.warning(f"Bbox info for {category_name} instance {instance_id} missing 'centroid'.")
                            return None
            return None
        except Exception as e:
             logger.error(f"Error getting object bbox info for {category_name} instance {instance_id}: {e}", exc_info=True)
             return None

    def generate_scene_qa(self, scene_name, scene_info, frame_info_for_scene):
        """Generates frame-level object-object Near/Far relative position QA pairs."""
        scene_qa_list = []

        if not scene_info or 'object_bboxes' not in scene_info:
            logger.warning(f"Scene {scene_name}: Missing scene_info or object_bboxes. Skipping.")
            return []
        if not frame_info_for_scene or 'frames' not in frame_info_for_scene:
            logger.warning(f"Scene {scene_name}: Missing frame_info_for_scene or frames. Skipping.")
            return []
        object_bboxes_dict = scene_info.get('object_bboxes', {})

        all_frames_in_metadata = frame_info_for_scene.get('frames', [])
        original_total_num_frames = len(all_frames_in_metadata)
        if original_total_num_frames == 0:
            logger.info(f"Scene {scene_name}: No frames key or empty frames list in frame_info_for_scene. Skipping.")
            return []

        valid_frames = sorted([f for f in all_frames_in_metadata 
                               if f.get('camera_pose_camera_to_world') and f.get('bboxes_2d')],
                        key=lambda x: x['frame_id'])
        if not valid_frames:
            logger.warning(f"Scene {scene_name}: No frames with camera poses and 2D bboxes found after filtering. Skipping.")
            return []
        # num_frames for formatting will be original_total_num_frames

        instance_id_to_category = {}
        if isinstance(object_bboxes_dict, dict):
            for category, instances in object_bboxes_dict.items():
                if isinstance(instances, list):
                    for instance_info in instances:
                        if isinstance(instance_info, dict) and "instance_id" in instance_info:
                            instance_id_to_category[str(instance_info["instance_id"])] = category

        potential_qas = []
        for frame_index, frame_data in enumerate(valid_frames): # Iterate over valid_frames
            frame_id = frame_data['frame_id']
            camera_pose_list = frame_data.get('camera_pose_camera_to_world') # Checked by filter
            bboxes_2d = frame_data.get('bboxes_2d', []) # Checked by filter

            if not camera_pose_list or not bboxes_2d: continue 

            T_c2w, T_w2c = self._get_camera_pose_matrices(camera_pose_list)
            if T_c2w is None or T_w2c is None: continue

            visible_instances_this_frame = {}
            category_counts_this_frame = {}
            instance_details_map = {}

            for bbox_2d_info in bboxes_2d:
                instance_id = bbox_2d_info.get('instance_id')
                if instance_id is None: continue
                instance_id_str = str(instance_id) # Ensure consistency

                # Look up category_name using the precomputed map
                category_name = instance_id_to_category.get(instance_id_str)
                if category_name is None:
                    # logger.warning(f"Scene {scene_name}, Frame {frame_id}: Category name not found for instance ID {instance_id_str}. Skipping instance.")
                    continue # Skip if category cannot be determined

                if instance_id_str not in visible_instances_this_frame: # Process each visible instance only once per frame
                    # Get 3D bbox info using the correct category name and instance ID
                    obj_bbox_info = self._get_object_bbox_info(object_bboxes_dict, category_name, instance_id_str) # Use instance_id_str

                    if obj_bbox_info and "centroid" in obj_bbox_info:
                         visible_instances_this_frame[instance_id_str] = category_name # Use instance_id_str
                         instance_details_map[instance_id_str] = { # Use instance_id_str
                             "category_name": category_name,
                             "bbox_meta": obj_bbox_info
                         }
                         category_counts_this_frame[category_name] = category_counts_this_frame.get(category_name, 0) + 1
                     # Removed redundant else/warning, handled by None check above

            frame_specific_unique_instance_ids = [
                inst_id for inst_id, cat_name in visible_instances_this_frame.items()
                if category_counts_this_frame.get(cat_name) == 1
            ]

            if len(frame_specific_unique_instance_ids) >= 2:
                # Ensure we use the string instance IDs consistent with map keys
                for inst_id_A_str, inst_id_B_str in itertools.combinations(frame_specific_unique_instance_ids, 2):
                    details_A = instance_details_map[inst_id_A_str]
                    details_B = instance_details_map[inst_id_B_str]
                    cat_A = details_A["category_name"]
                    cat_B = details_B["category_name"]
                    bbox_A = details_A["bbox_meta"]
                    bbox_B = details_B["bbox_meta"]

                    try:
                        o3d_bbox_A = o3d.geometry.OrientedBoundingBox(
                            center=np.array(bbox_A["centroid"]).astype(np.float64),
                            R=np.array(bbox_A["normalizedAxes"]).astype(np.float64).reshape(3, 3).T,
                            extent=np.array(bbox_A["axesLengths"]).astype(np.float64)
                        )
                        o3d_bbox_B = o3d.geometry.OrientedBoundingBox(
                            center=np.array(bbox_B["centroid"]).astype(np.float64),
                            R=np.array(bbox_B["normalizedAxes"]).astype(np.float64).reshape(3, 3).T,
                            extent=np.array(bbox_B["axesLengths"]).astype(np.float64)
                        )
                        vertices_A_world = np.asarray(o3d_bbox_A.get_box_points())
                        vertices_B_world = np.asarray(o3d_bbox_B.get_box_points())
                    except Exception as e:
                        logger.warning(f"Frame {frame_id}: Error creating O3D bbox for {cat_A} or {cat_B}: {e}. Skipping pair.")
                        continue

                    vertices_A_cam_list = []
                    vertices_B_cam_list = []
                    try:
                        for v in vertices_A_world:
                            v_h = np.append(v, 1)
                            vertices_A_cam_list.append((T_w2c @ v_h)[:3])
                        for v in vertices_B_world:
                            v_h = np.append(v, 1)
                            vertices_B_cam_list.append((T_w2c @ v_h)[:3])
                        if not vertices_A_cam_list or not vertices_B_cam_list:
                            logger.warning(f"Frame {frame_id}: Failed to transform vertices for {cat_A} or {cat_B}. Skipping pair.")
                            continue
                        vertices_A_cam = np.array(vertices_A_cam_list)
                        vertices_B_cam = np.array(vertices_B_cam_list)
                    except Exception as e:
                        logger.error(f"Frame {frame_id}: Error transforming vertices for {cat_A}/{cat_B}: {e}")
                        continue

                    min_A = vertices_A_cam.min(axis=0)
                    max_A = vertices_A_cam.max(axis=0)
                    min_B = vertices_B_cam.min(axis=0)
                    max_B = vertices_B_cam.max(axis=0)

                    min_zA, max_zA = min_A[2], max_A[2]
                    min_zB, max_zB = min_B[2], max_B[2]

                    # Near/Far (Z-axis: +Z is Forward, smaller Z is nearer)
                    # A is entirely Near B if max_zA < min_zB (with margin)
                    if (min_zB - max_zA) > self.position_threshold:
                        gt_relation_nf = self.options_nf[0] # Near
                        potential_qas.append({
                            "frame_id": frame_id, "frame_rank": frame_index,
                            "obj_A_cat": cat_A, "obj_B_cat": cat_B,
                            "dimension": "nf", "options": self.options_nf,
                            "ground_truth": gt_relation_nf
                        })
                    # A is entirely Far from B if min_zA > max_zB (with margin)
                    elif (min_zA - max_zB) > self.position_threshold:
                        gt_relation_nf = self.options_nf[1] # Far
                        potential_qas.append({
                            "frame_id": frame_id, "frame_rank": frame_index,
                            "obj_A_cat": cat_A, "obj_B_cat": cat_B,
                            "dimension": "nf", "options": self.options_nf,
                            "ground_truth": gt_relation_nf
                        })
        
        # Subsample before detailed formatting if too many potential QAs
        if self.args.num_subsample > 0 and len(potential_qas) > self.args.num_subsample:
            potential_qas = random.sample(potential_qas, self.args.num_subsample)

        for qa_info in potential_qas: 
            frame_id = qa_info["frame_id"]
            frame_rank = qa_info["frame_rank"] # This rank is within valid_frames
            obj_A_cat = qa_info["obj_A_cat"]
            obj_B_cat = qa_info["obj_B_cat"]
            gt_relation = qa_info["ground_truth"]

            shuffled_options = self.options_nf[:]
            random.shuffle(shuffled_options)
            if gt_relation not in shuffled_options:
                logger.warning(f"Scene {scene_name}, Frame {frame_id}, Pair ({obj_A_cat}, {obj_B_cat}), Dim nf: GT '{gt_relation}' not in options. Skipping.")
                continue
            gt_index = shuffled_options.index(gt_relation)
            mc_answer = self.option_letters[gt_index]
            self.answer_counts["nf"][mc_answer] += 1
            formatted_options_out = [f"{self.option_letters[i]}. {opt}" for i, opt in enumerate(shuffled_options)]

            # Use original_total_num_frames for frame_description
            question_text = self.question_template.format(
                object_A_name=obj_A_cat,
                object_B_name=obj_B_cat,
                frame_description=f"frame {frame_rank + 1} of {original_total_num_frames}"
            )

            qa_item = {
                "dataset": self.args.dataset,
                "scene_name": scene_name,
                "video_path": scene_info.get("video_path"),
                "frame_indices": [frame_id],
                "question_type": f"{self.args.question_type_prefix}",
                "question": question_text,
                "options": formatted_options_out,
                "ground_truth": gt_relation,
                "mc_answer": mc_answer,
            }
            scene_qa_list.append(qa_item)
        
        logger.info(f"Scene {scene_name}: Generated {len(scene_qa_list)} obj-obj Near/Far QA pairs.")
        return scene_qa_list

if __name__ == '__main__':
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s [%(levelname)s] - %(message)s')

    generator = ObjectObjectRelativePositionNFQAGenerator() # Use new class name
    # Ensure the specific NF template is available or add a dummy one
    if not hasattr(prompt_templates, 'VSTI_OBJ_OBJ_REL_POS_NF_TEMPLATE'):
        prompt_templates.VSTI_OBJ_OBJ_REL_POS_NF_TEMPLATE = "In {frame_description}, relative to the camera, is {obj_A_name} [Near/Far] compared to {obj_B_name}?"
    # Remove LR and UD template checks as they are not used in this file
    generator.run() 