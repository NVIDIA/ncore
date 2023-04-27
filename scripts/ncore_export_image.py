# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import logging

from pathlib import Path

import click
import tqdm

from ncore.impl.data.data3 import ShardDataLoader, CameraSensor
from ncore.impl.data.util import padded_index_string


@click.command()
@click.option('--shard-file-pattern',
              type=str,
              help='Data shard pattern to load (supports range expansion)',
              required=True)
@click.option('--output-dir', type=str, help='Path to the output folder', required=True)
@click.option('--camera-id',
              type=str,
              help='Camera sensor to export image frames for',
              default='camera_front_wide_120fov')
@click.option('--start-frame',
              type=click.IntRange(min=0, max_open=True),
              help='Initial frame to be exported',
              default=0)
@click.option('--end-frame', type=click.IntRange(min=-1, max_open=True), help='End frame to be exported', default=-1)
@click.option('--step-frame',
              type=click.IntRange(min=1, max_open=True),
              help='Step used to downsample the number of frames',
              default=1)
def ncore_export_image(shard_file_pattern: str, output_dir: str, camera_id: str, start_frame: int, end_frame: int,
                      step_frame: int):
    ''' Exports image data to image files '''

    # Initialize the logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    shards = ShardDataLoader.evaluate_shard_file_pattern(shard_file_pattern)
    loader = ShardDataLoader(shards)
    assert isinstance(sensor := loader.get_sensor(camera_id), CameraSensor), 'only camera sensors supported'

    # Create output path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    indices = sensor.get_frame_index_range(start_frame, end_frame, step_frame)
    logger.info(f"Starting image export for '{camera_id}' into '{output_path}'. {len(indices)} files will be exported")

    for frame_index in tqdm.tqdm(indices):
        # Load encoded frame data
        image_data = sensor.get_frame_data(frame_index)

        # Store encoded frame data to file
        with open(
                output_path /
                Path(padded_index_string(frame_index)).with_suffix(f'.{image_data.get_encoded_image_format()}'),
                'wb') as f:
            f.write(image_data.get_encoded_image_data())

    logger.info(f"Exported {len(indices)} images")


if __name__ == "__main__":
    ncore_export_image()
