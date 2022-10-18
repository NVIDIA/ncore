# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import click
import os 
import tqdm 

import numpy as np
from src.py.common.common import load_pc_dat, load_pkl
from src.py.common.visualization import LabelVisualizer

@click.command()
@click.option('--root-dir', type=str, help='Path to the preprocessed sequence.', required=True)
@click.option('--start-frame', type=click.IntRange(min=0, max_open=True), help='Initial camera frame to be use', default=0)
@click.option('--end-frame', type=click.IntRange(min=-1, max_open=True), help='End camera frame to be used', default=-1)
@click.option('--step-frame', type=click.IntRange(min=1, max_open=True), help='Step used to downsample the number of frames', default=1)
def visualize_labels(root_dir, start_frame, end_frame, step_frame):

    lidar_dir = os.path.join(root_dir, 'lidar')
    labels_dir = os.path.join(root_dir, 'labels')

    # Get all the files and filter them out based on the start, end, step
    lidar_files = sorted([os.path.join(lidar_dir, fname) for fname in os.listdir(lidar_dir) if (fname.endswith('.dat') or fname.endswith('.dat.xz'))])
    metadata_files = sorted([os.path.join(lidar_dir, fname) for fname in os.listdir(labels_dir) if fname.endswith('.pkl')])
    label_files = sorted([os.path.join(labels_dir, fname) for fname in os.listdir(labels_dir) if fname.endswith('.pkl')])
    
    assert (len(lidar_files) == len(label_files) == len(metadata_files)), "Number of lidar frames, metadata and label frames is not the same."

    if end_frame < 0:
        end_frame = len(lidar_files) + 1

    lidar_files = lidar_files[start_frame:end_frame:step_frame]
    label_files = label_files[start_frame:end_frame:step_frame]
    metadata_files = metadata_files[start_frame:end_frame:step_frame]

    for label_file, lidar_file, metadata_file in tqdm.tqdm(zip(label_files, lidar_files, metadata_files)):

        # Initialize the visualizer 
        viz = LabelVisualizer()
        
        frame_id = os.path.basename(lidar_file)

        # Load the metadata 
        meta_data = load_pkl(metadata_file)
        
        # Construct world -> lidar transformation
        T_lidar_rig = meta_data['T_lidar_rig']
        T_rig_world = meta_data['T_rig_world']

        T_rig_lidar = np.linalg.inv(T_lidar_rig)
        T_world_rig = np.linalg.inv(T_rig_world)
        T_world_lidar = T_rig_lidar @ T_world_rig

        # Import the point cloud and add it to the visualizer
        pc_data = load_pc_dat(lidar_file)     
        viz.add_pc(pc_data, T_world_lidar, frame_id)

        # Import the labels and add them to the viz
        labels = load_pkl(label_file)

        viz.add_labels(labels)

        # Show the point clouds
        viz.show()

if __name__ == "__main__":
    visualize_labels()
