# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import click
import os
import logging

import tqdm

import numpy as np

from src.py.common.nvidia_utils import parse_labels
from src.py.common.common import load_pc_dat, save_pkl
from src.py.dataset_converter import BaseNvidiaDataConverter

@click.command()
@click.option('--root-dir', type=str, help='Path to the preprocessed DSAI sequence to update', required=True)
@click.option('--parquet-file', type=str, help='Path to the parquet file to import NV labels from', required=True)
@click.option('--output-dir',
              type=str,
              help='Path to the output target directory - if missing, will update the original sequence in place',
              required=False,
              default=None)
@click.option("--index-digits", type=int, help="The number of integer digits to pad counters in output filenames to", default=6)
def dsai_import_labels(root_dir: str, parquet_file: str, output_dir: str, index_digits: int):
    ''' Imports the label-related properties of a preprocessed DSAI sequence, replacing existing labels

    Important: Importer currently only supports importing cuboid labels (following NV specs) from parquet files, and can therefore only be used along with NV data
    '''

    # Initialize the logger
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    logger.info(f"Importing from '{parquet_file}' into '{root_dir}'")

    # Parse lidar data for timestamp range
    lidar_dir = os.path.join(root_dir, 'lidar')
    assert os.path.exists(lidar_dir), f"Lidar folder {lidar_dir} doesn't exist"

    lidar_timestamps = np.load(os.path.join(lidar_dir, 'timestamps.npz'))['timestamps']

    # Parse the labels
    labels, frame_labels = parse_labels(parquet_file, lidar_timestamps[0], lidar_timestamps[-1],
                                        BaseNvidiaDataConverter.LABEL_STRING_TO_LABEL_ID,
                                        BaseNvidiaDataConverter.LABEL_STRINGS_UNCONDITIONALLY_DYNAMIC,
                                        BaseNvidiaDataConverter.LABEL_STRINGS_UNCONDITIONALLY_STATIC, logger)

    if not output_dir:
        output_dir = root_dir
    else:
        os.makedirs(output_dir, exist_ok=True)

    # Save the accumulated data / per-frame data
    save_pkl(labels, os.path.join(output_dir, 'labels.pkl'))
    save_pkl(frame_labels, os.path.join(output_dir, 'frame_labels.pkl'))

if __name__ == "__main__":
    dsai_import_labels()
