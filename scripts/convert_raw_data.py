import sys
sys.path.append('./')
from src.dataset_converter.waymo_open import WaymoConverter
from src.dataset_converter.nvidia import NvidiaConverter
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, help="Path to the raw data", required=True)
    parser.add_argument("--output_dir", type=str, help="Path  where the extracted data will be saved", required=True)
    parser.add_argument("--dataset", type=str, help="Name of the dataset", choices=['nvidia', 'waymo_open'], required=True)
    parser.add_argument('--n_proc', default=1, help='Number of processes to spawn')
    parser.add_argument('--semantic_seg', action='store_true', help="infer the semantic segmention for all camera images")
    parser.add_argument('--instance_seg', action='store_true', help="infer the instance segmention for all camera images")
    parser.add_argument('--rec_surface', action='store_true', help="reconstruct the static background mesh")

    args = parser.parse_args()

    if args.dataset == 'nvidia':
        converter = NvidiaConverter(args)
    elif args.dataset == 'waymo_open':
        converter = WaymoConverter(args)
    
    converter.convert()
