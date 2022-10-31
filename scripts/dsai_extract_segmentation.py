# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import click
import logging

from src.dsai_internal.deps.semantic_segmentation import run_semantic_segmentation
from src.dsai_internal.deps.instance_segmentation import run_instance_segmentation

from src.dsai_internal.data.data import DataLoader, CameraSensor, INDEX_DIGITS


@click.command()
@click.option('--root-dir', type=str, help="Path to the folder containing preprocessed data", required=True)
@click.option('--camera-sensor',
              '-c',
              'camera_sensors',
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
def dsai_extract_segmentation(root_dir: str, camera_sensors: list[str], semantic_seg: bool, instance_seg: bool,
                              start_frame: int, end_frame: int, step_frame: int):

    # Initialize the logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    loader = DataLoader(root_dir)

    if not camera_sensors:
        camera_sensors = loader.get_camera_sensor_ids()

    for camera_sensor in camera_sensors:
        sensor = loader.get_sensor(camera_sensor)
        assert isinstance(sensor, CameraSensor), 'only camera sensors supported'

        image_path_strings = [
             # segmentation functions expect file path strings
            str(sensor.get_frame_image_path(i))
            for i in sensor.get_frame_index_range(start_frame, end_frame, step_frame)
        ]

        if semantic_seg:
            logger.info(f'Running semantic segmentation on {len(image_path_strings)} images of camera {camera_sensor}')
            run_semantic_segmentation(image_path_strings, INDEX_DIGITS)

        if instance_seg:
            logger.info(f'Running instance segmentation on {len(image_path_strings)} images of camera {camera_sensor}')
            run_instance_segmentation(image_path_strings)


if __name__ == "__main__":
    dsai_extract_segmentation(show_default=True)
