# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import logging

from typing import Tuple
from pathlib import Path

import click

from ncore.impl.deps.semantic_segmentation import run_semantic_segmentation
from ncore.impl.deps.instance_segmentation import run_instance_segmentation

from ncore.impl.data.types import EncodedImageHandle
from ncore.impl.data.data3 import ShardDataLoader, CameraSensor
from ncore.impl.data.util import INDEX_DIGITS


@click.command()
@click.option('--shard-file-pattern',
              type=str,
              help='Data shard pattern to load (supports range expansion)',
              required=True)
@click.option('--output-dir',
              type=str,
              help="Path to the output folder (will be prefixed by camera sensor id)",
              required=True)
@click.option('--camera-id',
              '-c',
              'camera_ids',
              multiple=True,
              type=str,
              help='Cameras to be used (multiple value option, all if not specified)',
              default=None)
@click.option('--semantic-seg', is_flag=True, default=False, help="Perform semantic segmentation")
@click.option('--instance-seg', is_flag=True, default=False, help="Perform instance segmentation")
@click.option('--start-frame',
              type=click.IntRange(min=0, max_open=True),
              help='Initial frame to be segmented',
              default=0)
@click.option('--end-frame', type=click.IntRange(min=-1, max_open=True), help='End frame to be exported', default=-1)
@click.option('--step-frame',
              type=click.IntRange(min=1, max_open=True),
              help='Step used to downsample the number of frames',
              default=1)
def ncore_extract_segmentation(shard_file_pattern: str, output_dir: str, camera_ids: list[str], semantic_seg: bool,
                               instance_seg: bool, start_frame: int, end_frame: int, step_frame: int):

    # Initialize the logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    shards = ShardDataLoader.evaluate_shard_file_pattern(shard_file_pattern)
    loader = ShardDataLoader(shards)

    if not camera_ids:
        camera_ids = loader.get_camera_ids()

    for camera_id in camera_ids:
        assert isinstance(camera_sensor := loader.get_sensor(camera_id), CameraSensor), 'only camera sensors supported'

        # set up output paths
        output_path = Path(output_dir) / camera_sensor.get_sensor_id()
        output_path.mkdir(parents=True, exist_ok=True)

        image_handles: list[Tuple[int, EncodedImageHandle]] = [
            (i, camera_sensor.get_frame_handle(i))
            for i in camera_sensor.get_frame_index_range(start_frame, end_frame, step_frame)
        ]

        if semantic_seg:
            logger.info(f'Running semantic segmentation on {len(image_handles)} images of camera {camera_id}')
            run_semantic_segmentation(image_handles, output_path, INDEX_DIGITS)

        if instance_seg:
            logger.info(f'Running instance segmentation on {len(image_handles)} images of camera {camera_id}')
            run_instance_segmentation(image_handles, output_path, INDEX_DIGITS)


if __name__ == "__main__":
    ncore_extract_segmentation(show_default=True)
