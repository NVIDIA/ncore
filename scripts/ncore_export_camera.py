# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import logging

from pathlib import Path

import click
import tqdm
import cv2
import numpy as np

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
@click.option("--encode-video", is_flag=True, default=False, help="Encode video from frames")
@click.option("--encode-video-fps", type=int, default=30, help="Frame-rate for video encoding")
def ncore_export_camera(shard_file_pattern: str, output_dir: str, camera_id: str, start_frame: int, end_frame: int,
                        step_frame: int, encode_video: bool, encode_video_fps: int):
    ''' Exports camera frames to image files, and optionally encodes frames to a video file '''

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

    video_writer: cv2.VideoWriter | None = None
    video_path = None
    if encode_video:
        w, h = sensor.get_camera_model_parameters().resolution[:]
        video_writer = cv2.VideoWriter(str(video_path := (output_path / camera_id).with_suffix('.mp4')),
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       encode_video_fps,
                                       (int(w), int(h)))

    image_paths: list[str] = []
    for frame_index in tqdm.tqdm(indices):
        # Load encoded frame data
        image_data = sensor.get_frame_data(frame_index)

        # Store encoded frame data to file
        image_paths.append(
            str(output_path /
                Path(padded_index_string(frame_index)).with_suffix(f'.{image_data.get_encoded_image_format()}')))
        with open(image_paths[-1], 'wb') as f:
            f.write(image_data.get_encoded_image_data())

        if video_writer:
            image_rbg = np.asarray(image_data.get_decoded_image())
            image_bgr = image_rbg[..., ::-1]  # invert last dimension from RGB -> BGR (reverse RGB)

            video_writer.write(image_bgr)

    logger.info(f"Exported {len(indices)} images to {output_path}")

    if video_writer:
        video_writer.release()
        logger.info(f"Exported video to {video_path}")


if __name__ == "__main__":
    ncore_export_camera()
