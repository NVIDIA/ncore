# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import click
import tqdm

from src.dsai_internal.common.visualization import LabelVisualizer
from src.dsai_internal.data.data import DataLoader, LidarSensor


@click.command()
@click.option('--root-dir', type=str, help='Path to the preprocessed sequence', required=True)
@click.option('--sensor-id', type=str, help='Sensor to visualize labels for', default='lidar_gt_top_p128_v4p5')
@click.option('--start-frame',
              type=click.IntRange(min=0, max_open=True),
              help='Initial frame to be visualized',
              default=0)
@click.option('--end-frame', type=click.IntRange(min=-1, max_open=True), help='End frame to be visualized', default=-1)
@click.option('--step-frame',
              type=click.IntRange(min=1, max_open=True),
              help='Step used to downsample the number of frames',
              default=1)
def dsai_visualize_labels(root_dir, sensor_id, start_frame, end_frame, step_frame):

    loader = DataLoader(root_dir)
    sensor = loader.get_sensor(sensor_id)
    assert isinstance(sensor, LidarSensor), 'only lidar sensors supported'

    for frame_index in tqdm.tqdm(sensor.get_frame_index_range(start_frame, end_frame, step_frame)):

        # Initialize the visualizer
        viz = LabelVisualizer()

        # Import the point cloud and add it to the visualizer
        viz.add_pc(sensor.get_frame_data(frame_index, 'xyz_e'), sensor.get_frame_data(frame_index, 'intensity'),
                   sensor.get_frame_data(frame_index, 'dynamic_flag'), sensor.get_frame_data(frame_index, 'timestamp'),
                   frame_index)

        viz.add_labels(sensor.get_frame_labels(frame_index))

        # Show the point clouds
        viz.show()


if __name__ == "__main__":
    dsai_visualize_labels()
