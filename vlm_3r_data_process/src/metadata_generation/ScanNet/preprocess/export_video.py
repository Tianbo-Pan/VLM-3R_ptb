import argparse
import os
import sys
import cv2 # Requires opencv-python
import numpy as np
from tqdm import tqdm # Requires tqdm
import glob # Added for finding .sens files
import concurrent.futures # Added for parallel processing

# Assuming SensorData is in the same directory or PYTHONPATH is set
# If SensorData.py is elsewhere, adjust the import accordingly
try:
    # Assuming SensorData.py is in the same directory as the script using it or in the path
    # Adjust the path if SensorData is located differently, e.g., from ..SensorData import SensorData
    from .SensorData import SensorData
except ImportError as e:
    print(f"Error: Could not import SensorData class: {e}")
    print("Make sure SensorData.py is accessible (e.g., in the same directory or added to PYTHONPATH).")
    # Attempt import from parent directory if running from preprocess/scannet
    try:
        # This assumes the script is run from its directory and SensorData.py is one level up
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from SensorData import SensorData
        print("Successfully imported SensorData from parent directory.")
    except ImportError:
         # If SensorData.py is in the preprocess/scannet directory itself
         try:
             # This assumes the script and SensorData.py are in preprocess/scannet
             # No change needed if SensorData.py is in the same dir and PYTHONPATH is set correctly
             # Check current dir import again, maybe structure is flatter
             from SensorData import SensorData # Re-try import
             print("Successfully imported SensorData from current directory.")
         except ImportError:
             print("Error: SensorData class not found even after checking parent/current directories.")
             sys.exit(1)


# params
parser = argparse.ArgumentParser(description="Export color images from ScanNet .sens files to video.")
# data paths
# parser.add_argument('--filename', required=True, help='path to .sens file') # Removed
parser.add_argument('--scans_dir', required=True, help='path to the directory containing scene subdirectories (e.g., data/scannet/scans)')
# parser.add_argument('--output_video', required=True, help='path to output video file (e.g., output/video.mp4)') # Removed
parser.add_argument('--output_dir', required=True, help='path to the directory where output videos will be saved')
parser.add_argument('--train_val_splits_path', required=False, help='path to the directory containing train/val split files (e.g., scannetv2_train.txt, scannetv2_val.txt)') # Added
# video settings
parser.add_argument('--width', type=int, required=True, help='output video width')
parser.add_argument('--height', type=int, required=True, help='output video height')
parser.add_argument('--fps', type=int, default=30, help='output video frame rate (default: 30)')
parser.add_argument('--frame_skip', type=int, default=1, help='process every nth frame (default: 1)')
parser.add_argument('--codec', type=str, default='mp4v', help='video codec (default: mp4v for .mp4). Use "avc1" for H.264.')
parser.add_argument('--max_workers', type=int, default=None, help='Maximum number of processes to use for parallel processing (default: number of cores)')

def export_scene_video(sens_file_path, output_video_path, width, height, fps, frame_skip, codec):
    """Exports video for a single scene's .sens file."""
    print(f"Processing scene: {os.path.basename(os.path.dirname(sens_file_path))}")
    print(f'Loading {sens_file_path}...')
    try:
        sd = SensorData(sens_file_path)
    except NameError:
         print("Error: SensorData class failed to be imported.")
         return False # Indicate failure
    except Exception as e:
        print(f"Error loading SensorData from {sens_file_path}: {e}")
        return False # Indicate failure
    print(f'Loaded {len(sd.frames)} frames.')

    if not hasattr(sd, 'frames') or not sd.frames:
        print("Error: SensorData object does not contain frames or is empty.")
        return False

    # Check compression type - important for calling the right decompressor
    if not hasattr(sd, 'color_compression_type'):
         print("Error: SensorData object does not have 'color_compression_type' attribute.")
         print("Please check the SensorData class definition.")
         return False

    # Check if frames have the required decompress method
    if not hasattr(sd.frames[0], 'decompress_color'):
         print("Error: Frame objects in SensorData do not have the 'decompress_color' method.")
         print("Please check the SensorData class definition.")
         return False


    image_size = (width, height) # (width, height) for OpenCV
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, image_size)

    if not writer.isOpened():
        print(f"Error: Could not open video writer for {output_video_path}.")
        print("Check if the codec is supported and paths are correct.")
        return False

    print(f"Exporting video to {output_video_path} with resolution {image_size} and FPS {fps}...")

    frame_indices = range(0, len(sd.frames), frame_skip)
    processed_frames = 0
    skipped_frames = 0
    success = True # Track if processing completes without fatal errors

    try:
        for i in tqdm(frame_indices, desc="Processing frames", leave=False): # leave=False for nested loops
            try:
                frame = sd.frames[i]
                # Decompress color image using the type specified in SensorData header
                color_image_rgb = frame.decompress_color(sd.color_compression_type)

            except IndexError:
                 print(f"Warning: Frame index {i} out of bounds. Stopping scene processing.")
                 success = False # Consider this a failure for the scene
                 break
            except Exception as e:
                print(f"Warning: Could not decompress color image for frame index {i}: {e}. Skipping frame.")
                skipped_frames += 1
                continue

            if color_image_rgb is None:
                print(f"Warning: Decompressed null color image for frame index {i}. Skipping frame.")
                skipped_frames += 1
                continue

            # Ensure image is NumPy array and convert RGB to BGR for OpenCV
            if not isinstance(color_image_rgb, np.ndarray):
                 try:
                     color_image_rgb = np.array(color_image_rgb)
                 except Exception as e_conv:
                     print(f"Warning: Could not convert decompressed image to NumPy array for frame {i}: {e_conv}. Skipping frame.")
                     skipped_frames += 1
                     continue

            if len(color_image_rgb.shape) == 3 and color_image_rgb.shape[2] == 3:
                 color_image_bgr = cv2.cvtColor(color_image_rgb, cv2.COLOR_RGB2BGR)
            elif len(color_image_rgb.shape) == 2: # Grayscale
                 color_image_bgr = cv2.cvtColor(color_image_rgb, cv2.COLOR_GRAY2BGR)
            elif len(color_image_rgb.shape) == 3 and color_image_rgb.shape[2] == 4:
                 color_image_bgr = cv2.cvtColor(color_image_rgb, cv2.COLOR_RGBA2BGR)
            else:
                 print(f"Warning: Unexpected image format/shape ({color_image_rgb.shape}) for frame {i}. Skipping frame.")
                 skipped_frames += 1
                 continue

            # Resize image if necessary
            current_height, current_width = color_image_bgr.shape[:2]
            if current_width != width or current_height != height:
                 resized_image = cv2.resize(color_image_bgr, image_size, interpolation=cv2.INTER_AREA)
            else:
                 resized_image = color_image_bgr

            # Write frame
            writer.write(resized_image)
            processed_frames += 1

    except AttributeError as e:
         print(f"Error: Problem accessing frame data or methods for {sens_file_path}: {e}")
         print("Please check the SensorData class structure.")
         success = False # Indicate failure
    except Exception as e:
        print(f"An unexpected error occurred during processing for {sens_file_path}: {e}")
        success = False # Indicate failure
    finally:
        if writer.isOpened():
            writer.release()

    if success:
        print(f'Finished exporting video for scene.')
        print(f'Processed {processed_frames} frames.')
        if skipped_frames > 0:
            print(f'Skipped {skipped_frames} frames due to errors.')
        print(f'Video saved to: {output_video_path}')
    else:
        print(f"Failed to fully export video for scene {os.path.basename(os.path.dirname(sens_file_path))}.")
        # Clean up partially created video file if process failed significantly
        if os.path.exists(output_video_path) and processed_frames == 0: # Or some other condition
             print(f"Deleting potentially incomplete file: {output_video_path}")
             try:
                 os.remove(output_video_path)
             except OSError as e_del:
                 print(f"Error deleting file {output_video_path}: {e_del}")

    return success

def main():
    opt = parser.parse_args()
    print("Script Options:")
    print(opt)

    # Load train/val splits if path is provided
    train_scenes = set()
    val_scenes = set()
    if opt.train_val_splits_path:
        train_file_path = os.path.join(opt.train_val_splits_path, 'scannetv2_train.txt')
        val_file_path = os.path.join(opt.train_val_splits_path, 'scannetv2_val.txt')
        try:
            with open(train_file_path) as f:
                train_scenes = set(f.read().splitlines())
            print(f"Loaded {len(train_scenes)} train scenes.")
            with open(val_file_path) as f:
                val_scenes = set(f.read().splitlines())
            print(f"Loaded {len(val_scenes)} val scenes.")
        except FileNotFoundError:
            print(f"Warning: Train/Val split files not found in {opt.train_val_splits_path}. Videos will be saved directly in the output directory.")
            opt.train_val_splits_path = None # Disable split logic if files are missing
        except Exception as e:
            print(f"Warning: Error reading split files: {e}. Videos will be saved directly in the output directory.")
            opt.train_val_splits_path = None # Disable split logic on error


    # Ensure the base output directory and potential subdirectories exist
    output_dirs_to_create = [opt.output_dir]
    if opt.train_val_splits_path:
        output_dirs_to_create.append(os.path.join(opt.output_dir, 'train'))
        output_dirs_to_create.append(os.path.join(opt.output_dir, 'val'))

    for dir_path in output_dirs_to_create:
        if not os.path.exists(dir_path):
            print(f"Creating output directory: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)


    # Find all scene directories in the scans directory
    try:
        scene_dirs = [d.path for d in os.scandir(opt.scans_dir) if d.is_dir()]
    except FileNotFoundError:
        print(f"Error: Scans directory not found: {opt.scans_dir}")
        sys.exit(1)
    except Exception as e:
        print(f"Error scanning directory {opt.scans_dir}: {e}")
        sys.exit(1)

    if not scene_dirs:
        print(f"No subdirectories found in {opt.scans_dir}. Exiting.")
        sys.exit(0)

    print(f"Found {len(scene_dirs)} potential scene directories.")

    total_scenes = len(scene_dirs)
    results = [] # Store tuples of (scene_id, success_boolean)

    # Determine video extension based on codec
    # This is a basic guess; codecs might map to multiple containers
    extension = ".mp4" # Default
    if opt.codec.lower() in ['avc1', 'h264']:
        extension = ".mp4"
    elif opt.codec.lower() in ['mp4v']:
        extension = ".mp4"
    elif opt.codec.lower() in ['mjpg']:
        extension = ".avi" # MJPEG often uses AVI
    # Add more mappings if needed

    # Process each scene in parallel
    print(f"Starting parallel export using up to {opt.max_workers or os.cpu_count()} workers...")
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=opt.max_workers) as executor:
            futures_map = {}
            for scene_dir_path in scene_dirs:
                scene_id = os.path.basename(scene_dir_path)
                sens_file_path = os.path.join(scene_dir_path, f"{scene_id}.sens")

                # Determine output subdirectory based on splits
                output_subdir = ""
                if opt.train_val_splits_path:
                    if scene_id in train_scenes:
                        output_subdir = "train"
                    elif scene_id in val_scenes:
                        output_subdir = "val"
                    # else: scene is not in train/val splits, save to base output_dir or skip?
                    # Current logic: save to base dir if not in splits or if splits not provided.

                output_video_dir = os.path.join(opt.output_dir, output_subdir) # subdir is "" if not found in splits
                output_video_path = os.path.join(output_video_dir, f"{scene_id}{extension}")


                if not os.path.exists(sens_file_path):
                    print(f"Warning: .sens file not found for scene {scene_id} at expected path {sens_file_path}. Skipping.")
                    results.append((scene_id, False))
                    continue

                # Check if output video already exists, maybe add an overwrite flag later?
                if os.path.exists(output_video_path):
                    print(f"Output video {output_video_path} already exists. Skipping scene {scene_id}.")
                    # results.append((scene_id, True)) # Count as success or skip?
                    continue

                # Submit the task to the executor
                future = executor.submit(export_scene_video, sens_file_path, output_video_path, opt.width, opt.height, opt.fps, opt.frame_skip, opt.codec)
                futures_map[future] = scene_id

            # Process results as they complete, with a progress bar
            print("Processing Scenes... Press Ctrl+C to interrupt.") # Added note for user
            for future in tqdm(concurrent.futures.as_completed(futures_map), total=len(futures_map), desc="Processing Scenes"):
                scene_id = futures_map[future]
                try:
                    success = future.result() # Get the return value (True/False) from export_scene_video
                    results.append((scene_id, success))
                except Exception as exc:
                    print(f'\nScene {scene_id} generated an exception: {exc}')
                    results.append((scene_id, False))
                # Optionally print separator after each scene finishes if needed
                # print("-" * 30)
    except KeyboardInterrupt: # Add except block here
        print("\n\nInterrupted by user (Ctrl+C). Shutting down workers...")
        # Attempt to shut down the executor immediately.
        # For Python 3.9+, you can use cancel_futures=True for more aggressive cancellation:
        # executor.shutdown(wait=False, cancel_futures=True)
        # For compatibility, just use wait=False
        # Note: This might not immediately terminate the running child processes,
        # but it prevents new tasks from starting and signals existing ones.
        executor.shutdown(wait=False) # Shutdown without waiting for tasks to complete
        print("Executor shutdown initiated. Some processes might still be finishing.")
        # Optionally exit the script forcefully after shutdown attempt
        sys.exit(1) # Exit with a non-zero code to indicate interruption


    # Calculate final summary
    successful_exports = sum(1 for _, success in results if success)
    failed_exports = len(results) - successful_exports
    # Note: total_scenes might differ from len(results) if some were skipped before submission
    skipped_pre_submission = total_scenes - len(results)

    print("\n" + "=" * 30)
    print("Processing Summary:")
    print(f"Total scenes found: {total_scenes}")
    print(f"Scenes submitted for processing: {len(results)}")
    if skipped_pre_submission > 0:
        print(f"Scenes skipped before processing (e.g., .sens not found, output exists): {skipped_pre_submission}")
    print(f"Successfully exported: {successful_exports}")
    print(f"Failed during export: {failed_exports}")
    print("=" * 30)


if __name__ == '__main__':
    main() 