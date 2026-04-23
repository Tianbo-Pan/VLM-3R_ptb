import os
import subprocess
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_and_copy_videos(split_file_path, raw_data_dir, output_dir, split_name):
    """
    Reads a list of scene names from a split file, converts corresponding .mov videos
    to .mp4 format, and copies them to the output directory.

    Args:
        split_file_path (str): Path to the text file containing scene names.
        raw_data_dir (str): Path to the base directory containing raw scene folders.
        output_dir (str): Path to the directory where converted videos will be saved.
        split_name (str): Name of the split (e.g., 'train', 'val') to create a subfolder.
    """
    split_file = Path(split_file_path)
    raw_base = Path(raw_data_dir)
    output_base = Path(output_dir)
    output_split_dir = output_base / 'videos' / split_name

    if not split_file.exists():
        logging.error(f"Split file not found: {split_file}")
        return

    if not raw_base.is_dir():
        logging.error(f"Raw data directory not found: {raw_base}")
        return

    # Create the output directory if it doesn't exist
    output_split_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Ensured output directory exists: {output_split_dir}")

    try:
        with open(split_file, 'r') as f:
            scene_names = [line.strip() for line in f if line.strip()]
    except IOError as e:
        logging.error(f"Error reading split file {split_file}: {e}")
        return

    logging.info(f"Found {len(scene_names)} scenes in {split_file.name}")

    success_count = 0
    fail_count = 0

    for scene_name in tqdm(scene_names, desc=f"Processing {split_name} videos"):
        source_video_path = raw_base / scene_name / f"{scene_name}.mov"
        output_video_path = output_split_dir / f"{scene_name}.mp4"

        if not source_video_path.exists():
            logging.warning(f"Source video not found for scene '{scene_name}': {source_video_path}")
            fail_count += 1
            continue

        # Skip if the output file already exists
        if output_video_path.exists():
            logging.debug(f"Output file already exists, skipping: {output_video_path}")
            # Optionally, you might want to count this as success or track separately
            # success_count += 1
            continue

        # Use ffmpeg to convert and copy the video
        # -i: input file
        # -c:v copy -c:a copy: if no conversion is needed, just copy streams (faster, lossless)
        # -q:v 0: high quality variable bitrate for H.264 if re-encoding (adjust if needed)
        # -y: overwrite output files without asking
        # Use '-c:v libx264 -crf 23 -c:a aac -b:a 128k' for explicit re-encoding if needed
        # Added -loglevel error to suppress verbose ffmpeg output except errors
        command = [
            'ffmpeg',
            '-i', str(source_video_path),
            '-c:v', 'copy',  # Attempt to copy video stream without re-encoding
            '-c:a', 'copy',  # Attempt to copy audio stream without re-encoding
            '-y',
            '-loglevel', 'error', # Only show errors
            str(output_video_path)
        ]

        try:
            logging.debug(f"Running command: {' '.join(command)}")
            process = subprocess.run(command, check=True, capture_output=True, text=True)
            logging.debug(f"Successfully processed '{scene_name}'. Output: {output_video_path}")
            success_count += 1
        except subprocess.CalledProcessError as e:
            # If direct copy fails (e.g., incompatible codecs in mp4 container), try re-encoding
            logging.warning(f"Direct copy failed for '{scene_name}'. Attempting re-encoding. Error: {e.stderr}")
            command = [
                'ffmpeg',
                '-i', str(source_video_path),
                '-c:v', 'libx264', # Re-encode video to H.264
                '-crf', '23',      # Constant Rate Factor (lower is better quality, larger file)
                '-preset', 'medium', # Encoding speed vs. compression ratio
                '-c:a', 'aac',     # Re-encode audio to AAC
                '-b:a', '128k',    # Audio bitrate
                '-y',
                '-loglevel', 'error', # Only show errors
                str(output_video_path)
            ]
            try:
                logging.debug(f"Running re-encoding command: {' '.join(command)}")
                process = subprocess.run(command, check=True, capture_output=True, text=True)
                logging.info(f"Successfully re-encoded and processed '{scene_name}'. Output: {output_video_path}")
                success_count += 1
            except subprocess.CalledProcessError as e_reencode:
                 logging.error(f"Failed to convert video for scene '{scene_name}': {source_video_path}. Error: {e_reencode.stderr}")
                 fail_count += 1
                 # Clean up potentially corrupted output file
                 if output_video_path.exists():
                     try:
                         output_video_path.unlink()
                     except OSError as unlink_err:
                          logging.error(f"Failed to remove corrupted output file {output_video_path}: {unlink_err}")

    logging.info(f"Processing complete for {split_name}. Success: {success_count}, Failed: {fail_count}")

if __name__ == "__main__":
    # Define paths relative to the workspace root or use absolute paths
    workspace_root = Path(__file__).resolve().parents[3] # Assuming script is in src/metadata_generation/ARkitScenes
    split_file = workspace_root / "data/processed_data/ARkitScenes/splits/train.txt"
    raw_data_dir = workspace_root / "data/raw_data/arkitscenes/raw/Training"
    output_dir = workspace_root / "data/processed_data/ARkitScenes"

    convert_and_copy_videos(split_file, raw_data_dir, output_dir, 'train')

    # Example for validation split (if you have one)
    # val_split_file = workspace_root / "data/processed_data/ARkitScenes/splits/val.txt"
    # if val_split_file.exists():
    #     convert_and_copy_videos(val_split_file, raw_data_dir, output_dir, 'val')
    # else:
    #      logging.info("Validation split file not found, skipping validation set.")
