import json
import os
import open3d as o3d
import numpy as np

def load_json_metadata(json_path):
    """Loads the JSON metadata file."""
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    return metadata

if __name__ == "__main__":
    metadata_path = "data/processed_data/ScanNetpp/metadata/train/scannetpp_metadata_train.json"
    point_cloud_dir = "data/processed_data/ScanNetpp/point_cloud/train"

    metadata = load_json_metadata(metadata_path)
    print(f"Loaded metadata for {len(metadata)} scenes.")

    # We will add more functionality here
    for scene_id, scene_data in metadata.items():
        scene_name = scene_data["scene_name"]
        point_cloud_filename = f"{scene_name}.ply" # Assuming .ply format
        point_cloud_path = os.path.join(point_cloud_dir, point_cloud_filename)

        if not os.path.exists(point_cloud_path):
            print(f"Point cloud file not found for scene {scene_name}, skipping: {point_cloud_path}")
            continue

        print(f"Processing scene: {scene_name}")

        # Load point cloud
        try:
            pcd = o3d.io.read_point_cloud(point_cloud_path)
            if not pcd.has_points():
                print(f"Point cloud for scene {scene_name} is empty, skipping.")
                continue
        except Exception as e:
            print(f"Error loading point cloud for scene {scene_name}: {e}, skipping.")
            continue
        
        all_bboxes_for_scene = []
        if "object_bboxes" in scene_data:
            for obj_name, bboxes in scene_data["object_bboxes"].items():
                for bbox_data in bboxes:
                    centroid = np.array(bbox_data["centroid"])
                    axes_lengths = np.array(bbox_data["axesLengths"])
                    # The normalizedAxes represent the rotation matrix columns
                    # Open3D expects a 3x3 rotation matrix
                    rotation_matrix = np.array(bbox_data["normalizedAxes"]).reshape(3, 3).T # Transpose because Open3D expects row vectors for basis
                    
                    # Open3D OrientedBoundingBox uses extent (full side lengths)
                    # axesLengths seems to already be the full lengths based on typical definitions.
                    extent = axes_lengths

                    # Create OrientedBoundingBox
                    # The constructor takes center, R (rotation matrix), and extent.
                    obb = o3d.geometry.OrientedBoundingBox(centroid, rotation_matrix, extent)
                    obb.color = np.random.rand(3) # Assign a random color to each bbox
                    all_bboxes_for_scene.append(obb)
        
        if not all_bboxes_for_scene:
            print(f"No bounding boxes found for scene {scene_name}. Visualizing point cloud only.")
        
        # Visualize
        visualizables = [pcd] + all_bboxes_for_scene
        o3d.visualization.draw_geometries(visualizables,
                                          window_name=f"Scene: {scene_name}",
                                          width=800,
                                          height=600)
        
        # Limit to visualizing one scene for now to avoid opening too many windows
        # Remove or comment out this break to visualize all scenes
        print("Visualizing the first scene with point cloud and bounding boxes. Close the window to continue or stop the script.")
        break
