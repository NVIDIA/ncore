# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import logging

from pathlib import Path
from typing import Optional, Tuple

import click
import numpy as np
import tqdm

from scipy.spatial.transform import Rotation as R

from ncore.impl.common.transformations import MotionCompensator, se3_inverse, transform_point_cloud
from ncore.impl.common.util import unpack_optional
from ncore.impl.data import types
from ncore.impl.data.util import padded_index_string
from ncore.impl.data.v4.compat import SequenceLoaderProtocol, SequenceLoaderV4
from ncore.impl.data.v4.components import SequenceComponentGroupsReader
from ncore.impl.sensors.camera import CameraModel
from ncore.impl.sensors.lidar import StructuredLidarModel

from .util import NPArrayParamType, OptionalStrParamType
from .visualization import plot_points_on_image


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def se3_matrix(se3_delta: np.ndarray) -> np.ndarray:
    """Create the corresponding 4x4 matrix for se3_delta parameters"""
    assert len(se3_delta) == 6
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(se3_delta[3:]).as_matrix()
    T[:3, 3] = se3_delta[:3]

    return T


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class CLIBaseParams:
    """Parameters passed to non-command-based CLI part"""

    sensor_id: str
    sensor_extrinsic_delta: np.ndarray
    sensor_return_index: int
    camera_id: str
    camera_extrinsic_delta: np.ndarray
    start_frame: Optional[int]
    stop_frame: Optional[int]
    step_frame: Optional[int]
    point_size: float
    device: str
    pose: str
    output_dir: str
    external_distortion: bool
    file_prefix: str
    file_suffix: str
    encode_images: bool
    enable_lidar_model: bool
    timestamp_image_names: bool
    open_consolidated: bool


@click.group()
@click.option("--sensor-id", type=str, help="Sensor whose point cloud will be projected", required=True)
@click.option(
    "--sensor-extrinsic-delta",
    help="Optional: 6d [transl,rot-vec]-encoded extrinsic delta of ray bundle sensor",
    type=NPArrayParamType(dim=(6,), dtype=np.float32),
    default="[0,0,0,0,0,0]",
)
@click.option(
    "--sensor-return-index",
    help="Optional: Return index of the ray bundle sensor",
    type=int,
    default=0,
)
@click.option("--camera-id", type=str, help="Camera sensor to project on", required=True)
@click.option(
    "--camera-extrinsic-delta",
    help="Optional: 6d [transl,rot-vec]-encoded extrinsic delta of camera sensor",
    type=NPArrayParamType(dim=(6,), dtype=np.float32),
    default="[0,0,0,0,0,0]",
)
@click.option(
    "--start-frame", type=click.IntRange(min=0, max_open=True), help="Initial camera frame to be used", default=None
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
@click.option("--open-consolidated/--no-open-consolidated", default=True, help="Pre-load shard meta-data?")
@click.option(
    "--point-size",
    type=click.FloatRange(min=0.0, max_open=True),
    default=4.0,
    help="Point size of rendering",
)
@click.option(
    "--device", type=click.Choice(["cuda", "cpu"]), help="Device used for the computation via torch", default="cuda"
)
@click.option(
    "--pose",
    type=click.Choice(["rolling-shutter", "mean", "start", "end"]),
    help="Per-pixel poses to use (rolling-shutter optimization, mean frame pose, start frame pose, end frame pose)",
    default="rolling-shutter",
)
@click.option("--output-dir", type=str, help="Path to the output folder if encoding images", required=False, default="")
@click.option(
    "--external-distortion/--no-external-distortion",
    is_flag=True,
    default=False,
    help="Allow / disallow external distortion",
)
@click.option("--file-prefix", type=str, help="Prefix to prepend to output files", required=False, default="")
@click.option("--file-suffix", type=str, help="Suffix to append to output files", required=False, default="")
@click.option("--encode-images/--no-encode-images", is_flag=True, default=False, help="Encode image files for frames")
@click.option(
    "--lidar-model/--no-lidar-model",
    "enable_lidar_model",
    is_flag=True,
    default=False,
    help="Use lidar-model for point cloud generation",
)
@click.option(
    "--timestamp-image-names/--no-timestamp-image-names",
    is_flag=True,
    default=False,
    help="Store image with timestamp filenames or frame-index filenames",
)
@click.pass_context
def cli(ctx, **kwargs) -> None:
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
        open_consolidated=params.open_consolidated,
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
    lidar_sensor = loader.get_lidar_sensor(params.sensor_id)
    cam_sensor = loader.get_camera_sensor(params.camera_id)

    # Get the camera frame indices from the index range
    indices = cam_sensor.get_frame_index_range(params.start_frame, params.stop_frame, params.step_frame)
    logger.info(f"Starting the pc projection. {len(indices)} frames will be processed.")

    msg = f"Camera torch model @ {params.device} | projection with {params.pose} poses"

    # Initialize the camera model on requested device
    cam_model_params = cam_sensor.model_parameters

    # Drop external distortion if not allowed
    if not params.external_distortion:
        cam_model_params = dataclasses.replace(cam_model_params, external_distortion_parameters=None)

    if cam_model_params.external_distortion_parameters is not None:
        msg += " | with external distortion"

    cam_model = CameraModel.from_parameters(cam_model_params, device=params.device)

    # Initialize motion compensator, and the lidar model if requested
    motion_compensator = MotionCompensator(lidar_sensor.pose_graph)
    lidar_model: Optional[StructuredLidarModel] = None
    if params.enable_lidar_model:
        lidar_model = StructuredLidarModel.maybe_from_parameters(lidar_sensor.model_parameters, device=params.device)

        assert lidar_model is not None, f"No structured lidar model available for lidar sensor {params.sensor_id}"

        msg += " | with structured lidar model"

    for frame_index in tqdm.tqdm(indices):
        # Get the camera timestamp and find the closes lidar frame (relative to center of camera frame timestamps)
        cam_timestamp_start_us = cam_sensor.get_frame_timestamp_us(frame_index, types.FrameTimepoint.START)
        cam_timestamp_end_us = cam_sensor.get_frame_timestamp_us(frame_index, types.FrameTimepoint.END)
        pc_frame_index = lidar_sensor.get_closest_frame_index(
            cam_timestamp_start_us + (cam_timestamp_end_us - cam_timestamp_start_us) // 2,
            relative_frame_time=0.5,
        )

        # Load the camera image and the point cloud
        img_frame = cam_sensor.get_frame_image_array(frame_index)

        if lidar_model is not None:
            ## Generate sensor points from model elements with length of the source data
            sensor_pc = (
                lidar_model.elements_to_sensor_points(
                    unpack_optional(
                        lidar_sensor.get_frame_ray_bundle_model_element(pc_frame_index),
                        msg=f"Lidar model elements not available for frame {pc_frame_index}",
                    ),
                    lidar_sensor.get_frame_ray_bundle_return_distance_m(
                        pc_frame_index, return_index=params.sensor_return_index
                    ),
                )
                .cpu()
                .numpy()
            )

            ## Perform motion-compensation
            pc = motion_compensator.motion_compensate_points(
                sensor_id=params.sensor_id,
                xyz_pointtime=sensor_pc,
                timestamp_us=lidar_sensor.get_frame_ray_bundle_timestamp_us(pc_frame_index),
                frame_start_timestamp_us=lidar_sensor.get_frame_timestamp_us(
                    pc_frame_index, types.FrameTimepoint.START
                ),
                frame_end_timestamp_us=lidar_sensor.get_frame_timestamp_us(pc_frame_index, types.FrameTimepoint.END),
            ).xyz_e_sensorend
        else:
            pc = lidar_sensor.get_frame_point_cloud(
                pc_frame_index,
                motion_compensation=True,
                with_start_points=False,
                return_index=params.sensor_return_index,
            ).xyz_m_end

        # Transform the point cloud to the world coordinate frame
        pc = transform_point_cloud(
            pc, lidar_sensor.get_frames_T_sensor_target("world", pc_frame_index, types.FrameTimepoint.END)
        )

        T_world_camera_start = se3_inverse(
            cam_sensor.get_frames_T_sensor_target("world", frame_index, types.FrameTimepoint.START)
        )
        T_world_camera_end = se3_inverse(
            cam_sensor.get_frames_T_sensor_target("world", frame_index, types.FrameTimepoint.END)
        )

        logger.info(msg)

        match params.pose:
            case "rolling-shutter":
                world_point_projections = cam_model.world_points_to_image_points_shutter_pose(
                    pc,
                    T_world_camera_start,
                    T_world_camera_end,
                    return_valid_indices=True,
                    return_T_world_sensors=True,
                )

            case "mean":
                world_point_projections = cam_model.world_points_to_image_points_mean_pose(
                    pc, T_world_camera_start, T_world_camera_end, return_valid_indices=True, return_T_world_sensors=True
                )

            case "start":
                world_point_projections = cam_model.world_points_to_image_points_static_pose(
                    pc, T_world_camera_start, return_valid_indices=True, return_T_world_sensors=True
                )

            case "end":
                world_point_projections = cam_model.world_points_to_image_points_static_pose(
                    pc, T_world_camera_end, return_valid_indices=True, return_T_world_sensors=True
                )

        image_point_coords = world_point_projections.image_points.cpu().numpy()
        trans_matrices = world_point_projections.T_world_sensors.cpu().numpy()  # type: ignore
        valid_idx = world_point_projections.valid_indices.cpu().numpy()  # type: ignore
        transformed_points = transform_point_cloud(pc[valid_idx, None, :], trans_matrices).squeeze(1)
        dist_rs = np.linalg.norm(transformed_points, axis=1, keepdims=True)

        save_path: Optional[Path] = None
        if params.encode_images:
            assert len(params.output_dir)

            output_path = Path(params.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            if params.timestamp_image_names:
                save_path = output_path / (params.file_prefix + str(cam_timestamp_end_us) + params.file_suffix + ".png")
            else:
                save_path = output_path / (
                    params.file_prefix + padded_index_string(frame_index) + params.file_suffix + ".png"
                )

        plot_points_on_image(
            np.concatenate((image_point_coords[:, :2], dist_rs), axis=1),
            img_frame,
            msg if not params.encode_images else "",
            point_size=params.point_size,
            show=not params.encode_images,
            save_path=str(save_path),
        )


if __name__ == "__main__":
    cli(show_default=True)
