# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import click
import os
import logging
import tqdm

import numpy as np
import point_cloud_utils as pcu
from point_cloud_utils import TriangleMesh

from src.py.common.common import load_pc_dat

@click.command()
@click.option('--root-dir', type=str, help='Path to the preprocessed sequence.', required=True)
@click.option('--output-dir', type=str, help='Path to the output folder sequence (if relative it is considered relative to the root-dir)', default='lidar/')
@click.option('--start-frame', type=click.IntRange(min=0, max_open=True), help='Initial lidar frame to be exported', default=0)
@click.option('--end-frame', type=click.IntRange(min=-1, max_open=True), help='End lidar frame to be exported', default=-1)
@click.option('--step-frame', type=click.IntRange(min=1, max_open=True), help='Step used to downsample the number of frames', default=1)
@click.option('--max-dist', type=float, help='Maximum distance for lidar pose.', default=100.0)

def export_ply_files(root_dir, output_dir, max_dist, start_frame, end_frame, step_frame):
    ''' Exports the point cloud data contained in the dat.xz files to the ply format with named attributes

    Args:
        root_dir (string): path to the root dir containing the dsai converted data
        output_dir (string): path to the output dir where the 
        max_dist (float): Filter out point with the distance greater than this distance from the ego vehicle (if negative, all points will be used)
        start_frame (int): Initial lidar frame to be exported
        end_frame (int): End lidar frame to be exported (-1 = use all frames up to the last)
        step_frame (int): Determines the temporal downsampling rate  (if 1 all frames will be used)
    ''' 

    # Initialize the logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Get all the files and filter them out based on the start, end, step
    lidar_dir = os.path.join(root_dir, 'lidar')
    assert os.path.exists(lidar_dir), f"Lidar folder {lidar_dir} doesn't exsist."

    files = sorted([os.path.join(lidar_dir, fname) for fname in os.listdir(lidar_dir) if (fname.endswith('.dat') or fname.endswith('.dat.xz'))])
    if end_frame < 0:
        end_frame = len(files) + 1
    files = files[start_frame:end_frame:step_frame]

    logger.info(f"Starting '.ply' export. {len(files)} files will be exported.")

    # initialize the output dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(root_dir, output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    

    for fname in tqdm.tqdm(files):

        pc_data = load_pc_dat(fname, allow_lookup_fallback=False)

        # Filter out the points based on the max dist if it is provided
        dist_mask = (pc_data[:,6] < max_dist) if max_dist > 0 else np.ones(pc_data.shape[0]).astype(bool)

        pc = TriangleMesh()
        pc.vertex_data.positions = pc_data[dist_mask, 3:6]
        pc.vertex_data.custom_attributes['dist'] = pc_data[dist_mask, 6]
        pc.vertex_data.custom_attributes['intensity'] = pc_data[dist_mask, 7]
        pc.vertex_data.custom_attributes['dynamic_flag'] = pc_data[dist_mask, 8]

        # Save the ply file
        pc.save(os.path.join(output_dir, os.path.splitext(os.path.basename(fname))[0] + '.ply'))


if __name__ == "__main__":
    export_ply_files()