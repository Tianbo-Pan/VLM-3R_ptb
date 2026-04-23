import argparse
import json
import os
import cv2
import logging
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import functools

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Export video from ScanNet++ dataset.")
    parser.add_argument("--raw_data_dir", type=str, required=True, help="Path to the ScanNet++ dataset directory")
    parser.add_argument("--rendered_data_dir", type=str, required=True, help="Path to the ScanNet++ rendered data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the output video")
    parser.add_argument("--split_path", type=str, required=True, help="Path to the split file")
    parser.add_argument("--height", type=int, default=389, help="Height of the video")
    parser.add_argument("--width", type=int, default=584, help="Width of the video")
    parser.add_argument("--fps", type=int, default=3, help="FPS of the video")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers to use")
    return parser.parse_args()

def process_scene(scene_id, raw_data_dir, rendered_data_dir, output_dir, height, width, fps):
    logging.info(f"Processing scene: {scene_id}")
    # get split path
    split_json_path = os.path.join(raw_data_dir, scene_id, 'dslr', 'train_test_lists.json')
    if not os.path.exists(split_json_path):
        logging.error(f"Split JSON not found for scene {scene_id} at {split_json_path}")
        return

    try:
        with open(split_json_path, 'r') as f:
            train_list = json.load(f)['train']
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON for scene {scene_id} at {split_json_path}")
        return
    except Exception as e:
        logging.error(f"Error reading split JSON for scene {scene_id}: {e}")
        return

    # get image path
    image_dir = os.path.join(rendered_data_dir, scene_id, 'dslr', 'rgb_resized_undistorted')
    if not os.path.isdir(image_dir):
        logging.error(f"Image directory not found for scene {scene_id} at {image_dir}")
        return

    images_train_paths = sorted([os.path.join(image_dir, f) for f in train_list])

    # write video
    video_path = os.path.join(output_dir, f'{scene_id}.mp4')
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    if not writer.isOpened():
        logging.error(f"Failed to open video writer for {video_path}")
        return

    success_count = 0
    fail_count = 0
    for image_path in images_train_paths:
        if not os.path.exists(image_path):
            logging.warning(f"Image file not found: {image_path} (listed in train_list)")
            fail_count += 1
            continue
        try:
            img = cv2.imread(image_path)
            if img is None:
                logging.warning(f"Failed to read image: {image_path}")
                fail_count += 1
                continue
            # Ensure frame size matches video writer dimensions
            if img.shape[1] != width or img.shape[0] != height:
                img = cv2.resize(img, (width, height))
            writer.write(img)
            success_count += 1
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}")
            fail_count += 1

    writer.release()
    if success_count > 0:
        logging.info(f"Finished processing scene {scene_id}. Video saved to {video_path}. Frames written: {success_count}, Failed/Skipped: {fail_count}")
    else:
        logging.warning(f"No frames written for scene {scene_id}. Check image paths and logs.")
        # Optionally delete the empty video file
        try:
            os.remove(video_path)
            logging.info(f"Removed empty video file: {video_path}")
        except OSError as e:
            logging.error(f"Error removing empty video file {video_path}: {e}")


def main():
    args = parse_args()
    logging.info(f"Starting video export with args: {args}")

    # get split list
    try:
        with open(args.split_path, 'r') as f:
            split_list = f.readlines()
        split_list = [line.strip() for line in split_list if line.strip()]
        if not split_list:
            logging.error(f"Split file {args.split_path} is empty or contains no valid scene IDs.")
            return
    except FileNotFoundError:
        logging.error(f"Split file not found: {args.split_path}")
        return
    except Exception as e:
        logging.error(f"Error reading split file {args.split_path}: {e}")
        return

    # make output dir
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except OSError as e:
        logging.error(f"Failed to create output directory {args.output_dir}: {e}")
        return

    logging.info(f"Found {len(split_list)} scenes to process.")

    # Process scenes in parallel
    # Use functools.partial to create a function with fixed arguments
    process_func = functools.partial(
        process_scene,
        raw_data_dir=args.raw_data_dir,
        rendered_data_dir=args.rendered_data_dir,
        output_dir=args.output_dir,
        height=args.height,
        width=args.width,
        fps=args.fps
    )

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # Use tqdm to show progress
        list(tqdm(executor.map(process_func, split_list), total=len(split_list), desc="Processing scenes"))

    logging.info("Finished processing all scenes.")

if __name__ == "__main__":
    main()