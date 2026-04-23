import argparse
import os
import zipfile
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm

def unzip_instance(args):
    zip_path, output_path = args
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_workers', type=int, 
                       default=max(1, multiprocessing.cpu_count() // 2),
                       help='number of parallel workers')
    args = parser.parse_args()

    scans_dir = os.path.join(args.input_dir, 'scans')
    scene_dirs = [d for d in os.listdir(scans_dir) if os.path.isdir(os.path.join(scans_dir, d))]

    # Create process arguments
    process_args = []
    for scene_dir in scene_dirs:
        zip_path = os.path.join(scans_dir, scene_dir, f'{scene_dir}_2d-instance-filt.zip')
        output_path = os.path.join(args.output_dir, scene_dir)
        process_args.append((zip_path, output_path))

    # Use process pool for parallel processing
    with Pool(processes=args.num_workers) as pool:
        list(tqdm(
            pool.imap(unzip_instance, process_args),
            total=len(scene_dirs),
            desc="Unzipping instance files"
        ))

if __name__ == '__main__':
    main()