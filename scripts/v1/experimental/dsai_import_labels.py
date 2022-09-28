# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import click
import os
import logging
import multiprocessing

import numpy as np

from functools import partial

from src.py.common.nvidia_utils import LabelProcessor
from src.py.common.common import (load_pc_dat, load_pkl, save_pkl, save_pc_dat)


@click.command()
@click.option('--root-dir', type=str, help='Path to the preprocessed DSAI sequence to update', required=True)
@click.option('--parquet-file', type=str, help='Path to the parquet file to import NV labels from', required=True)
@click.option('--output-dir',
              type=str,
              help='Path to the output target directory - if missing, will update the original sequence in place',
              required=False,
              default=None)
@click.option("--index-digits",
              type=int,
              help="The number of integer digits to pad counters in output filenames to",
              default=6)
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
    labels, frame_labels = LabelProcessor.parse(parquet_file, lidar_timestamps[0], lidar_timestamps[-1], logger)

    if not output_dir:
        output_dir = root_dir
    else:
        os.makedirs(output_dir, exist_ok=True)

    # Save the accumulated data / per-frame data
    save_pkl(labels, os.path.join(output_dir, 'labels.pkl'))
    save_pkl(frame_labels, os.path.join(output_dir, 'frame_labels.pkl'))

    # Update dynamic flag of each lidar frame
    output_dir_lidar = os.path.join(output_dir, 'lidar')
    os.makedirs(output_dir_lidar, exist_ok=True)
    output_dir_labels = os.path.join(output_dir, 'labels')
    os.makedirs(output_dir_labels, exist_ok=True)

    with multiprocessing.Pool(
            processes=os.cpu_count(),
            # restart processes after this number of frames to free up potentially piled up resources
            maxtasksperchild=5) as pool:
        logger.info(
            f"Updating {len(lidar_timestamps)} lidar frames with new label data using {os.cpu_count()} worker processes"
        )
        pool.map(
            partial(update_lidar_frame_label_data_process,
                    lidar_dir=lidar_dir,
                    output_dir_lidar=output_dir_lidar,
                    output_dir_labels=output_dir_labels,
                    index_digits=index_digits,
                    labels=labels,
                    frame_labels=frame_labels), enumerate(lidar_timestamps))


def update_lidar_frame_label_data_process(args, lidar_dir, output_dir_lidar, output_dir_labels, index_digits, labels,
                                          frame_labels):
    ''' Per-frame execution in separate process '''

    # Decode current frame data to process
    continuos_frame_index, frame_timestamp = args[0], args[1]

    # Load original frame data
    pc_data = load_pc_dat(os.path.join(lidar_dir, f'{str(continuos_frame_index).zfill(index_digits)}.dat.xz'))
    meta_data = load_pkl(os.path.join(lidar_dir, f'{str(continuos_frame_index).zfill(index_digits)}.pkl'))

    # Construct world -> lidar transformation
    T_lidar_rig = meta_data['T_lidar_rig']
    T_rig_world = meta_data['T_rig_world']

    T_rig_lidar = np.linalg.inv(T_lidar_rig)
    T_world_rig = np.linalg.inv(T_rig_world)
    T_world_lidar = T_rig_lidar @ T_world_rig

    # Transform points from world to lidar
    xyz_world_homogeneous = np.row_stack([pc_data[:, 3:6].transpose(),
                                          np.ones(pc_data.shape[0], dtype=np.float32)])  # 4 x N
    xyz_lidar_homogeneous = T_world_lidar @ xyz_world_homogeneous  # 4 x N

    xyz = xyz_lidar_homogeneous[:3, :].transpose()  # N x 3

    # Compute dynamic flag / load current frame labels
    dynamic_flag, current_frame_labels = LabelProcessor.lidar_dynamic_flag(xyz, frame_timestamp, labels, frame_labels, skip_dynamic_flag=False)

    # Set point-cloud dynamic flag and serialize updated point-cloud
    pc_data[:, 8] = dynamic_flag
    save_pc_dat(os.path.join(output_dir_lidar, f'{str(continuos_frame_index).zfill(index_digits)}.dat.xz'), pc_data)

    # Serialize per-frame labels
    # Remark: it's currently simpler to serialize per lidar-frame labels for this timestamp here, as we perform frame subsampling as part of lidar processing.
    # However, in the future we might also incorporate camera data also, and this serialization might need to be relocated.
    save_pkl(current_frame_labels,
             os.path.join(output_dir_labels,
                          str(continuos_frame_index).zfill(index_digits) + '.pkl'))


if __name__ == "__main__":
    dsai_import_labels()
