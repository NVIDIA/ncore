# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import click
import tqdm

from dsai.impl.common.visualization import LabelVisualizer
from dsai.impl.data.data3 import ShardDataLoader, LidarSensor


@click.command()
@click.option('--shard-file-pattern',
              type=str,
              help='Data shard pattern to load (supports range expansion)',
              required=True)
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
def dsai_visualize_labels(shard_file_pattern, sensor_id, start_frame, end_frame, step_frame):

    shards = ShardDataLoader.evaluate_shard_file_pattern(shard_file_pattern)
    loader = ShardDataLoader(shards)
    sensor = loader.get_sensor(sensor_id)
    assert isinstance(sensor, LidarSensor), 'only lidar sensors supported'

    for frame_index in tqdm.tqdm(sensor.get_frame_index_range(start_frame, end_frame, step_frame)):

        # Initialize the visualizer
        viz = LabelVisualizer()

        # Import the point cloud and add it to the visualizer
        viz.add_pc(frame_index,
                   sensor.get_frame_data(frame_index, 'xyz_e'), 
                   sensor.get_frame_data(frame_index, 'intensity'),
                   sensor.get_frame_data(frame_index, 'dynamic_flag'),
                   sensor.get_frame_data(frame_index, 'timestamp_us'),
                   sensor.get_frame_data(frame_index, 'semantic_class') if sensor.has_frame_data(frame_index, 'semantic_class') else None,
                   )

        viz.add_labels(sensor.get_frame_labels(frame_index))

        # Show the point clouds
        viz.show()


if __name__ == "__main__":
    dsai_visualize_labels()
