# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import logging

import click
import tqdm
import numpy as np

from dsai.impl.data.data3 import ShardDataLoader, PointCloudSensor, CameraSensor
from dsai.impl.data import types
from dsai.impl.common.transformations import transform_point_cloud
from dsai.impl.av_utils import rollingShutterProjection
from dsai.impl.common.visualization import plot_points_on_image
from dsai.impl.sensors.camera import CameraModel

@click.command()
@click.option('--shard-file-pattern',
              type=str,
              help='Data shard pattern to load (supports range expansion)',
              required=True)
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
              help='Device used for the computation. If gpu - projection will be done in pytorch on a gpu.',
              default='cpu')
def dsai_project_pc_to_img(shard_file_pattern: str, sensor_id: str, camera_id: str, start_frame: int, end_frame: int,
                           step_frame: int, device: str):
    ''' Projects the point cloud to the camera image, comparing projection w. and w/o rolling shutter compensation  '''

    # Initialize the logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    shards = ShardDataLoader.evaluate_shard_file_pattern(shard_file_pattern)
    loader = ShardDataLoader(shards)
    pc_sensor = loader.get_sensor(sensor_id)
    assert isinstance(pc_sensor, PointCloudSensor), 'only point-cloud sensors are supported as source sensor'
    cam_sensor = loader.get_sensor(camera_id)
    assert isinstance(cam_sensor, CameraSensor), 'only image sensors are supported as the target sensor'

    # Get the camera frame indices from the index range
    indices = cam_sensor.get_frame_index_range(start_frame, end_frame, step_frame)
    logger.info(f"Starting the pc projection. {len(indices)} frames will be processed.")
    for frame_index in tqdm.tqdm(indices):

        # Get the camera timestamp and find the closes lidar frame
        cam_timestamp = cam_sensor.get_frame_timestamp_us(frame_index)
        pc_frame_index = pc_sensor.get_closest_frame_index(cam_timestamp)

        # Load the camera image and the point cloud
        img_frame = cam_sensor.get_frame_image_array(frame_index)
        pc = pc_sensor.get_frame_data(pc_frame_index, 'xyz_e')

        # Transform the point cloud to the world coordinate frame
        pc = transform_point_cloud(pc, pc_sensor.get_frame_T_sensor_world(pc_frame_index))

        T_world_sensor_start = cam_sensor.get_frame_T_world_sensor(frame_index, types.FrameTimepoint.START)
        T_world_sensor_end = cam_sensor.get_frame_T_world_sensor(frame_index, types.FrameTimepoint.END)

        # Initialize the camera model
        cam_model_params = cam_sensor.get_camera_model_parameters()
        cam_model = CameraModel.from_parameters(cam_model_params)

        if device in ['gpu', 'both']:
            logger.info(f"Starting the projection with a torch GPU implementation.")

            pixel_coords_gpu, trans_matrices_gpu, valid_idx_gpu = cam_model.world_points_to_pixels_rolling_shutter(pc, T_world_sensor_start, T_world_sensor_end)

            pixel_coords_torch = pixel_coords_gpu.cpu().numpy()
            trans_matrices_torch = trans_matrices_gpu.cpu().numpy()
            valid_idx_torch = valid_idx_gpu.cpu().numpy()
            transformed_points = transform_point_cloud(pc[valid_idx_torch,None,:], trans_matrices_torch).squeeze(1)
            dist_rs = np.linalg.norm(transformed_points, axis=1, keepdims=True)

            plot_points_on_image(np.concatenate((pixel_coords_torch[:,:2], dist_rs),axis=1), img_frame,
                                            "Projection with rolling shutter (torch GPU implementation)", point_size=4.0)
        if device in ['cpu', 'both']:
            logger.info(f"Starting the projection with a c++ CPU implementation.")

            pixel_coords_rs, trans_matrices_rs, valid_idx_rs = rollingShutterProjection(pc, cam_model_params, np.stack([T_world_sensor_start, T_world_sensor_end]))

            # Compute the distance to the points in the camera coordinate system
            transformed_points = transform_point_cloud(pc[valid_idx_rs,None,:], trans_matrices_rs).squeeze(1)
            dist_rs = np.linalg.norm(transformed_points, axis=1, keepdims=True)

            plot_points_on_image(np.concatenate((pixel_coords_rs[:,:2], dist_rs),axis=1), img_frame,
                                                "Projection with rolling shutter (c++ implementation)", point_size=4.0)


if __name__ == "__main__":
    dsai_project_pc_to_img()
