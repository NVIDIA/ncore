# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import click
import logging
import tqdm

from pathlib import Path
from typing import Optional

import numpy as np
from point_cloud_utils import TriangleMesh

from src.py.common.nvidia_utils import transform_point_cloud
from src.py.data_converter.data import DataLoader, LidarSensor, PointCloudSensor, padded_index_string


@click.command()
@click.option('--root-dir', type=str, help='Path to the preprocessed sequence', required=True)
@click.option('--sensor-id', type=str, help='Sensor to export ply files for', default='lidar_gt_top_p128_v4p5')
@click.option('--output-dir',
              type=str,
              help='Path to the output folder (will output into source folder if not provided)',
              default=None)
@click.option('--start-frame',
              type=click.IntRange(min=0, max_open=True),
              help='Initial frame to be exported',
              default=0)
@click.option('--end-frame',
              type=click.IntRange(min=-1, max_open=True),
              help='End frame to be exported',
              default=-1)
@click.option('--step-frame',
              type=click.IntRange(min=1, max_open=True),
              help='Step used to downsample the number of frames',
              default=1)
@click.option('--frame',
              type=click.Choice(['sensor', 'rig', 'world']),
              help='Frame to represent the point-cloud in',
              default='world')
def dsai_export_ply(root_dir: str, output_dir: Optional[str], sensor_id: str, start_frame: int, end_frame: int,
                    step_frame: int, frame: str):
    ''' Exports the point cloud data to the ply format with named attributes '''

    # Initialize the logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    loader = DataLoader(root_dir)
    sensor = loader.get_sensor(sensor_id)
    assert isinstance(sensor, PointCloudSensor), 'only point-cloud sensors supported'

    if not output_dir:
        output_path = sensor.get_sensor_dir()
    else:
        output_path = Path(output_dir)

    indices = sensor.get_frame_index_range(start_frame, end_frame, step_frame)
    logger.info(f"Starting '.ply' export. {len(indices)} files will be exported.")

    for frame_index in tqdm.tqdm(indices):

        # Setup target transformation
        if frame == 'sensor':
            T_sensor_target = np.identity(4)
        elif frame == 'rig':
            T_sensor_target = sensor.get_T_sensor_rig()
        elif frame == 'world':
            T_sensor_target = sensor.get_frame_T_sensor_world(frame_index)

        pc = TriangleMesh()
        pc.vertex_data.positions = transform_point_cloud(sensor.get_frame_data(frame_index, 'xyz_e'), T_sensor_target)
        pc.vertex_data.custom_attributes['xyz_s'] = transform_point_cloud(sensor.get_frame_data(frame_index, 'xyz_s'),
                                                                          T_sensor_target)
        if isinstance(sensor, LidarSensor):
            pc.vertex_data.custom_attributes['intensity'] = sensor.get_frame_data(frame_index, 'intensity')
            pc.vertex_data.custom_attributes['dynamic_flag'] = sensor.get_frame_data(frame_index, 'dynamic_flag')

            # Compute offset in "inverse" fashion to prevent wrapping around zero for uint64
            negative_offset_timestamp = (sensor.get_frame_timestamp_us(frame_index) -
                                         sensor.get_frame_data(frame_index, 'timestamp')).astype(np.int32)
            pc.vertex_data.custom_attributes['negative_offset_timestamp'] = negative_offset_timestamp

        # Save the ply file
        pc.save(str(output_path / (padded_index_string(frame_index) + '.ply')))


if __name__ == "__main__":
    dsai_export_ply()
