import argparse
import os
import shutil
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm

def move_aggregation(args):
    aggregation_path, output_path = args
    shutil.copy(aggregation_path, output_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_workers', type=int, 
                       default=max(1, multiprocessing.cpu_count() // 2),
                       help='number of parallel workers')
    args = parser.parse_args()

    scene_dirs = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]

    # Create process arguments
    process_args = []
    for scene_dir in scene_dirs:
        aggregation_path = os.path.join(args.input_dir, scene_dir, f'{scene_dir}.aggregation.json')
        output_path = os.path.join(args.output_dir, scene_dir)
        process_args.append((aggregation_path, output_path))

    # Use process pool for parallel processing
    with Pool(processes=args.num_workers) as pool:
        list(tqdm(
            pool.imap(move_aggregation, process_args),
            total=len(scene_dirs),
            desc="Moving aggregation files"
        ))

if __name__ == '__main__':
    main()