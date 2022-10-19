# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import click
import logging
import tqdm

from pathlib import Path
from typing import Optional

import numpy as np

from src.py.data_converter.data import DataLoader, PointCloudSensor, CameraSensor, FrameTimepoint
from src.py.common.transformations import transform_point_cloud
from src.cpp.av_utils import rollingShutterProjection
from src.py.common.visualization import plot_points_on_image

@click.command()
@click.option('--root-dir', type=str, help='Path to the preprocessed sequence', required=True)
@click.option('--sensor-id', type=str, help='Sensor whose point cloud will be projected', required=True)
@click.option('--camera-id', type=str, help='Sensor to export ply files for', required=True)
@click.option('--start-frame',
              type=click.IntRange(min=0, max_open=True),
              help='Initial camera frame to be used',
              default=0)
@click.option('--end-frame',
              type=click.IntRange(min=-1, max_open=True),
              help='End camera frame to be used',
              default=-1)
@click.option('--step-frame',
              type=click.IntRange(min=1, max_open=True),
              help='Step used to downsample the number of frames',
              default=1)    
@click.option('--device',
              type=click.Choice(['gpu', 'cpu', 'both']),
              help='Device used for the computation. If gpu - projection will be done in pytorch.',
              default='cpu')


def dsai_project_pc_to_img(root_dir: str, sensor_id: str, camera_id: str, start_frame: int, end_frame: int,
                    step_frame: int, device: str):
    ''' Projects the point cloud to the camera image, comparing projection w. and w/o rolling shutter compensation  '''

    # Initialize the logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    loader = DataLoader(root_dir)
    pc_sensor = loader.get_sensor(sensor_id)
    assert isinstance(pc_sensor, PointCloudSensor), 'only point-cloud sensors supported as source sensor'
    cam_sensor = loader.get_sensor(camera_id)
    assert isinstance(cam_sensor, CameraSensor), 'only image sensors supported as the target sensor'

    # Get the camera frame indices from the index range
    indices = cam_sensor.get_frame_index_range(start_frame, end_frame, step_frame)
    logger.info(f"Starting the pc projection. {len(indices)} frames will be processed.")
    for frame_index in tqdm.tqdm(indices):

        # Get the camera timestamp and find the closes lidar frame
        cam_timestamp = cam_sensor.get_frame_timestamp_us(frame_index)
        pc_frame_index = pc_sensor.get_closest_frame(cam_timestamp)
        
        # Load the camera image and the point cloud
        img_frame = cam_sensor.get_frame_img(frame_index)
        pc = pc_sensor.get_frame_data(pc_frame_index, 'xyz_e')

        # Transform the point cloud to the world coordinate frame
        pc = transform_point_cloud(pc, pc_sensor.get_frame_T_sensor_world(pc_frame_index))
        
        # Get the camera metadata
        cam_metadata = cam_sensor.get_camera_model()

        T_world_sensor_start = cam_sensor.get_frame_T_world_sensor(frame_index, FrameTimepoint.START)
        T_world_sensor_end = cam_sensor.get_frame_T_world_sensor(frame_index, FrameTimepoint.END)
        cam_timestamp_start = cam_sensor.get_frame_timestamp_us(frame_index, FrameTimepoint.START )
        cam_timestamp_end = cam_sensor.get_frame_timestamp_us(frame_index, FrameTimepoint.END )

        T_world_sensor = np.vstack([T_world_sensor_start, T_world_sensor_end])
        cam_timestamps = np.vstack([cam_timestamp_start, cam_timestamp_end])
        pixel_coords_rs, trans_matrices_rs, valid_idx_rs = rollingShutterProjection(pc, cam_metadata, T_world_sensor, cam_timestamps)

        # Compute the distance to the points in the camera coordinate system
        transformed_points = (trans_matrices_rs[:,:3,:3] @ pc[valid_idx_rs,:,None] + trans_matrices_rs[:,:3,3:4]).squeeze(-1)
        dist_rs = np.linalg.norm(transformed_points,axis=1,keepdims=True)

        # Visualize the result 
        # plot_points_on_image(np.concatenate((pixel_coords[:,:2], dist),axis=1), img_frame, 
        #                                     "Projection without considering rolling shutter", point_size=4.0)
        plot_points_on_image(np.concatenate((pixel_coords_rs[:,:2], dist_rs),axis=1), img_frame, 
                                            "Projection with rolling shutter (c++ implementation)", point_size=4.0)


if __name__ == "__main__":
    dsai_project_pc_to_img()
