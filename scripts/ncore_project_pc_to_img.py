# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import logging

import click
import tqdm
import numpy as np

from ncore.impl.data.data3 import ShardDataLoader, PointCloudSensor, CameraSensor
from ncore.impl.data import types
from ncore.impl.common.transformations import transform_point_cloud
from ncore.impl.common.visualization import plot_points_on_image
from ncore.impl.sensors.camera import CameraModel


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
@click.option('--end-frame', type=click.IntRange(min=-1, max_open=True), help='End camera frame to be used', default=-1)
@click.option('--step-frame',
              type=click.IntRange(min=1, max_open=True),
              help='Step used to downsample the number of frames',
              default=1)
@click.option('--device',
              type=click.Choice(['cuda', 'cpu']),
              help='Device used for the computation via torch',
              default='cuda')
def ncore_project_pc_to_img(shard_file_pattern: str, sensor_id: str, camera_id: str, start_frame: int, end_frame: int,
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

        # Initialize the camera model on requested device
        cam_model_params = cam_sensor.get_camera_model_parameters()
        cam_model = CameraModel.from_parameters(cam_model_params, device=device)

        logger.info(f"Starting the projection with torch implementation on device={device}")

        world_point_projections = cam_model.world_points_to_image_points_shutter_pose(pc,
                                                                                      T_world_sensor_start,
                                                                                      T_world_sensor_end,
                                                                                      return_valid_indices=True,
                                                                                      return_T_world_sensors=True)

        image_point_coords = world_point_projections.image_points.cpu().numpy()
        trans_matrices = world_point_projections.T_world_sensors.cpu().numpy()  # type: ignore
        valid_idx = world_point_projections.valid_indices.cpu().numpy()  # type: ignore
        transformed_points = transform_point_cloud(pc[valid_idx, None, :], trans_matrices).squeeze(1)
        dist_rs = np.linalg.norm(transformed_points, axis=1, keepdims=True)

        plot_points_on_image(np.concatenate((image_point_coords[:, :2], dist_rs), axis=1),
                             img_frame,
                             f'Projection with rolling shutter (torch implementation @ {device})',
                             point_size=4.0)


if __name__ == "__main__":
    ncore_project_pc_to_img()
