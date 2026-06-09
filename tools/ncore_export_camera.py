# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import click
import cv2
import numpy as np
import tqdm

from ncore.impl.data.types import (
    ConcreteCameraModelParametersUnion,
    IdealPinholeCameraModelParameters,
    encode_camera_model_parameters,
)
from ncore.impl.data.util import padded_index_string
from ncore.impl.data.v4.compat import SequenceLoaderProtocol, SequenceLoaderV4
from ncore.impl.data.v4.components import SequenceComponentGroupsReader
from ncore.impl.sensors.camera import CameraModel
from ncore.impl.sensors.rectification import Rectificator


try:
    from .cli import OptionalStrParamType
except ImportError:
    from tools.cli import OptionalStrParamType


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
        rectify: Whether to rectify frames to an ideal pinhole camera before exporting
        rectify_target_fov_deg: Optional target full field of view [deg] of the rectified pinhole
        rectify_fov_factor: Multiplicative factor applied to the target (or natural) field of view
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
    rectify: bool
    rectify_target_fov_deg: Optional[float]
    rectify_fov_factor: float


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
@click.option(
    "--rectify/--no-rectify",
    is_flag=True,
    default=False,
    help="Rectify frames to an ideal (distortion-free) pinhole camera before exporting",
)
@click.option(
    "--rectify-target-fov-deg",
    type=click.FloatRange(min=0.0, max=180.0, min_open=True, max_open=True),
    default=None,
    help="Target full field of view [deg] of the rectified pinhole, wider or narrower than the "
    "inferred natural FOV (only used with --rectify). If omitted, the source's natural FOV is used.",
)
@click.option(
    "--rectify-fov-factor",
    type=click.FloatRange(min=0.0, min_open=True),
    default=1.0,
    help="Multiplicative factor applied to the (target or natural) field of view before rectifying; "
    ">1 widens, <1 narrows (only used with --rectify)",
)
@click.pass_context
def cli(ctx, **kwargs):
    """Exports camera frames to image files, and optionally encodes frames to a video file"""

    ctx.obj = CLIBaseParams(**kwargs)


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
@click.option(
    "--masks-component-group",
    type=OptionalStrParamType(),
    help="Component group for 'masks' (use 'none' to disable)",
    default="default",
)
@click.option(
    "--cuboids-component-group",
    type=OptionalStrParamType(),
    help="Component group for 'cuboids' (use 'none' to disable)",
    default="default",
)
@click.pass_context
def v4(
    ctx,
    component_groups: Tuple[str, ...],
    poses_component_group: str,
    intrinsics_component_group: str,
    masks_component_group: Optional[str],
    cuboids_component_group: Optional[str],
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


def _build_rectificator(
    params: CLIBaseParams, source_parameters: ConcreteCameraModelParametersUnion
) -> Tuple[Rectificator, IdealPinholeCameraModelParameters]:
    """Builds a Rectificator mapping the source camera to an ideal pinhole target

    Returns the Rectificator and the resulting ideal-pinhole target parameters.
    """

    # Target full field of view:
    # - an explicit --rectify-target-fov-deg is a single (isotropic, aspect-preserving)
    #   scalar FOV;
    # - otherwise the source's per-axis natural FOV is used;
    # - --rectify-fov-factor scales whichever of the two applies.
    target_fov: Union[float, np.ndarray]
    if params.rectify_target_fov_deg is not None:
        target_fov = float(np.deg2rad(params.rectify_target_fov_deg))
    else:
        target_fov = IdealPinholeCameraModelParameters.natural_fov(source_parameters)

    target_fov = target_fov * params.rectify_fov_factor

    target_parameters = IdealPinholeCameraModelParameters.from_source(source_parameters, target_fov=target_fov)

    source_model = CameraModel.from_parameters(source_parameters, device="cpu")
    target_model = CameraModel.from_parameters(target_parameters, device="cpu")
    return Rectificator(source_model, target_model), target_parameters


def run(params: CLIBaseParams, loader: SequenceLoaderProtocol) -> None:
    # Initialize the logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    sensor = loader.get_camera_sensor(params.camera_id)

    # Create output path
    output_path = Path(params.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Optionally set up rectification to an ideal pinhole target
    rectificator: Optional[Rectificator] = None
    rectified_resolution: Optional[Tuple[int, int]] = None
    if params.rectify:
        rectificator, target_parameters = _build_rectificator(params, sensor.model_parameters)
        rectified_resolution = (int(target_parameters.resolution[0]), int(target_parameters.resolution[1]))

        # Persist the rectified (ideal pinhole) intrinsics alongside the images
        intrinsics_path = output_path / f"{params.camera_id}.rectified_intrinsics.json"
        with open(intrinsics_path, "w") as f:
            json.dump(encode_camera_model_parameters(target_parameters), f, indent=2)
        logger.info(
            f"Rectifying '{params.camera_id}' to an ideal pinhole "
            f"(target_fov_deg={params.rectify_target_fov_deg}, fov_factor={params.rectify_fov_factor}). "
            f"Rectified intrinsics written to '{intrinsics_path}'"
        )

    indices = sensor.get_frame_index_range(params.start_frame, params.stop_frame, params.step_frame)
    logger.info(
        f"Starting frame export for '{params.camera_id}' into '{output_path}'. {len(indices)} frames will be exported"
    )
    # Instantiate video encoder if requested
    video_writer: Optional[cv2.VideoWriter] = None
    video_path = None
    if params.encode_video:
        if rectified_resolution is not None:
            w, h = rectified_resolution
        else:
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

        # Decode the image up front when rectifying (the pixels change, so the encoded
        # source bytes can no longer be passed through verbatim)
        rectified_rgb: Optional[np.ndarray] = None
        if rectificator is not None:
            source_rgb = np.asarray(image_data.get_decoded_image())
            rectified_rgb = rectificator.apply(source_rgb).cpu().numpy().astype(np.uint8)

        # Store frame data to image files
        if params.encode_images:
            fname = (
                padded_index_string(frame_index)
                if not params.timestamp_image_names
                else str(sensor.get_frame_timestamp_us(frame_index))
            )

            if rectified_rgb is not None:
                # Re-encode the rectified image (cv2 expects BGR)
                out_file = output_path / Path(fname).with_suffix(".png")
                cv2.imwrite(str(out_file), rectified_rgb[..., ::-1])
            else:
                with open(
                    output_path / Path(fname).with_suffix(f".{image_data.get_encoded_image_format()}"),
                    "wb",
                ) as f:
                    f.write(image_data.get_encoded_image_data())

        # Encode frame to video
        if video_writer:
            image_rgb = rectified_rgb if rectified_rgb is not None else np.asarray(image_data.get_decoded_image())
            image_bgr = image_rgb[..., ::-1]  # invert last dimension from RGB -> BGR (reverse RGB)

            video_writer.write(image_bgr)

    logger.info(f"Exported {len(indices)} images to {output_path}")

    if video_writer:
        video_writer.release()
        logger.info(f"Exported video to {video_path}")


if __name__ == "__main__":
    cli(show_default=True)
