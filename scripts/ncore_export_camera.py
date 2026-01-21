# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import logging

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import click
import cv2
import numpy as np
import tqdm

from ncore.impl.data.data3 import ShardDataLoader
from ncore.impl.data.v4.compat import SequenceLoaderProtocol, SequenceLoaderV3, SequenceLoaderV4
from ncore.impl.data.v4.components import SequenceComponentGroupsReader
from ncore.impl.data.util import padded_index_string


@dataclass(kw_only=True, slots=True, frozen=True)
class CLIBaseParams:
    """Parameters passed to non-command-based CLI part.

    Attributes:
        camera_id: ID of the camera sensor to visualize (e.g., 'camera_front_wide_120fov')
        output_dir: Path to the output folder
        start_frame: Optional starting frame index for export range
        stop_frame: Optional ending frame index (exclusive) for export range
        step_frame: Optional step size for downsampling frames
        encode_images: Whether to encode image files for frames
        timestamp_image_names: Whether to use timestamps for image filenames
        encode_video: Whether to encode a video of the frames
        encode_video_fps: Frame-rate for video encoding
    """

    camera_id: str
    output_dir: str
    start_frame: Optional[int]
    stop_frame: Optional[int]
    step_frame: Optional[int]
    encode_images: bool
    timestamp_image_names: bool
    encode_video: bool
    encode_video_fps: int


@click.group()
@click.option("--output-dir", type=str, help="Path to the output folder", required=True)
@click.option(
    "--camera-id", type=str, help="Camera sensor to export image frames for", default="camera_front_wide_120fov"
)
@click.option(
    "--start-frame", type=click.IntRange(min=0, max_open=True), help="Initial frame to be exported", default=None
)
@click.option(
    "--stop-frame", type=click.IntRange(min=0, max_open=True), help="Past-the-end frame to be exported", default=None
)
@click.option(
    "--step-frame",
    type=click.IntRange(min=1, max_open=True),
    help="Step used to downsample the number of frames",
    default=None,
)
@click.option("--encode-images/--no-encode-images", is_flag=True, default=True, help="Encode image files for frames")
@click.option(
    "--timestamp-image-names/--no-timestamp-image-names",
    is_flag=True,
    default=False,
    help="Store image with timestamp filenames or frame-index filenames",
)
@click.option("--encode-video", is_flag=True, default=False, help="Encode video of frames")
@click.option("--encode-video-fps", type=int, default=30, help="Frame-rate for video encoding")
@click.pass_context
def cli(ctx, **kwargs):
    """Exports camera frames to image files, and optionally encodes frames to a video file"""

    ctx.obj = CLIBaseParams(**kwargs)


@cli.command()
@click.option(
    "--shard-file-pattern", type=str, help="Data shard pattern to load (supports range expansion)", required=True
)
@click.pass_context
def v3(
    ctx,
    shard_file_pattern: str,
) -> None:
    params: CLIBaseParams = ctx.obj

    shards = ShardDataLoader.evaluate_shard_file_pattern(shard_file_pattern)
    loader = ShardDataLoader(shards)

    run(
        params,
        SequenceLoaderV3(
            loader,
        ),
    )


@cli.command()
@click.option(
    "component_groups",
    "--component-group",
    multiple=True,
    type=str,
    help="Data component group / sequence meta paths",
    required=True,
)
@click.option("--poses-component-group", type=str, help="Component group for 'poses'", default="default")
@click.option("--intrinsics-component-group", type=str, help="Component group for 'intrinsics'", default="default")
@click.option("--masks-component-group", type=str, help="Component group for 'masks'", default="default")
@click.option(
    "--cuboids-component-group",
    type=str,
    help="Component group for 'cuboids'",
    default="default",
)
@click.pass_context
def v4(
    ctx,
    component_groups: Tuple[str, ...],
    poses_component_group: str,
    intrinsics_component_group: str,
    masks_component_group: str,
    cuboids_component_group: str,
) -> None:
    params: CLIBaseParams = ctx.obj

    loader = SequenceComponentGroupsReader(
        [Path(group_path) for group_path in component_groups],
    )

    run(
        params,
        SequenceLoaderV4(
            loader,
            poses_component_group_name=poses_component_group,
            intrinsics_component_group_name=intrinsics_component_group,
            masks_component_group_name=masks_component_group,
            cuboids_component_group_name=cuboids_component_group,
        ),
    )


def run(params: CLIBaseParams, loader: SequenceLoaderProtocol) -> None:
    # Initialize the logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    sensor = loader.get_camera_sensor(params.camera_id)

    # Create output path
    output_path = Path(params.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    indices = sensor.get_frame_index_range(params.start_frame, params.stop_frame, params.step_frame)
    logger.info(
        f"Starting frame export for '{params.camera_id}' into '{output_path}'. {len(indices)} frames will be exported"
    )
    # Instantiate video encoder if requested
    video_writer: Optional[cv2.VideoWriter] = None
    video_path = None
    if params.encode_video:
        w, h = sensor.model_parameters.resolution[:]
        video_writer = cv2.VideoWriter(
            str(video_path := (output_path / params.camera_id).with_suffix(".mp4")),
            cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore
            params.encode_video_fps,
            (int(w), int(h)),
        )

    for frame_index in tqdm.tqdm(indices):
        # Load encoded frame data
        image_data = sensor.get_frame_data(frame_index)

        # Store encoded frame data to image files
        if params.encode_images:
            fname = (
                padded_index_string(frame_index)
                if not params.timestamp_image_names
                else str(sensor.get_frame_timestamp_us(frame_index))
            )

            with open(
                output_path / Path(fname).with_suffix(f".{image_data.get_encoded_image_format()}"),
                "wb",
            ) as f:
                f.write(image_data.get_encoded_image_data())

        # Encode frame to video
        if video_writer:
            image_rbg = np.asarray(image_data.get_decoded_image())
            image_bgr = image_rbg[..., ::-1]  # invert last dimension from RGB -> BGR (reverse RGB)

            video_writer.write(image_bgr)

    logger.info(f"Exported {len(indices)} images to {output_path}")

    if video_writer:
        video_writer.release()
        logger.info(f"Exported video to {video_path}")


if __name__ == "__main__":
    cli(show_default=True)
