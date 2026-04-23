# 1. unzip scannet
apt install unzip
pip install imageio pypng
1. sens
cd stage2_data/ScanNet/utils
python -m preprocess.scannet.preprocess --input_dir data/scannet --output_dir data/scannet_processed
2. instance
python preprocess/scannet/unzip_instance.py --input_dir data/scannet --output_dir data/scannet_processed
3. aggregation
python preprocess/scannet/move_aggregation.py --input_dir data/scannet --output_dir data/scannet_processed