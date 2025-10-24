# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import dataclasses
import logging

from pathlib import Path
from typing import Optional

import click
import numpy as np
import tqdm

from scipy.spatial.transform import Rotation as R

from ncore.impl.common.transformations import MotionCompensator, se3_inverse, transform_point_cloud
from ncore.impl.common.visualization import plot_points_on_image
from ncore.impl.data import types
from ncore.impl.data.data3 import CameraSensor, LidarSensor, PointCloudSensor, ShardDataLoader
from ncore.impl.data.util import padded_index_string
from ncore.impl.sensors.camera import CameraModel
from ncore.impl.sensors.lidar import StructuredLidarModel
from scripts.util import NPArrayParamType


def se3_matrix(se3_delta: np.ndarray) -> np.ndarray:
    """Create the corresponding 4x4 matrix for se3_delta parameters"""
    assert len(se3_delta) == 6
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(se3_delta[3:]).as_matrix()
    T[:3, 3] = se3_delta[:3]

    return T


@click.command()
@click.option(
    "--shard-file-pattern", type=str, help="Data shard pattern to load (supports range expansion)", required=True
)
@click.option("--sensor-id", type=str, help="Sensor whose point cloud will be projected", required=True)
@click.option(
    "--sensor-extrinsic-delta",
    help="Optional: 6d [transl,rot-vec]-encoded extrinsic delta of point-cloud sensor",
    type=NPArrayParamType(dim=(6,), dtype=np.float32),
    default="[0,0,0,0,0,0]",
)
@click.option("--camera-id", type=str, help="Sensor to export ply files for", required=True)
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
def ncore_project_pc_to_img(
    shard_file_pattern: str,
    sensor_id: str,
    sensor_extrinsic_delta: np.ndarray,
    camera_id: str,
    camera_extrinsic_delta: np.ndarray,
    start_frame: Optional[int],
    stop_frame: Optional[int],
    step_frame: Optional[int],
    point_size: float,
    device: str,
    pose: str,
    output_dir: str,
    external_distortion: bool,
    file_prefix: str,
    file_suffix: str,
    encode_images: bool,
    enable_lidar_model: bool,
    timestamp_image_names: bool,
):
    """Projects the point cloud to the camera image, comparing projection w. and w/o rolling shutter compensation"""

    # Initialize the logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    shards = ShardDataLoader.evaluate_shard_file_pattern(shard_file_pattern)
    loader = ShardDataLoader(shards)
    pc_sensor = loader.get_sensor(sensor_id)
    assert isinstance(pc_sensor, PointCloudSensor), "only point-cloud sensors are supported as source sensor"
    cam_sensor = loader.get_sensor(camera_id)
    assert isinstance(cam_sensor, CameraSensor), "only image sensors are supported as the target sensor"

    # Get the camera frame indices from the index range
    indices = cam_sensor.get_frame_index_range(start_frame, stop_frame, step_frame)
    logger.info(f"Starting the pc projection. {len(indices)} frames will be processed.")

    # Construct transformations
    T_sensor_rig = se3_matrix(sensor_extrinsic_delta) @ pc_sensor.get_T_sensor_rig()
    T_camera_rig = se3_matrix(camera_extrinsic_delta) @ cam_sensor.get_T_sensor_rig()

    poses = loader.get_poses()

    msg = f"Camera torch model @ {device} | projection with {pose} poses"

    # Initialize the camera model on requested device
    cam_model_params = cam_sensor.get_camera_model_parameters()

    # Drop external distortion if not allowed
    if not external_distortion:
        cam_model_params = dataclasses.replace(cam_model_params, external_distortion_parameters=None)

    if cam_model_params.external_distortion_parameters is not None:
        msg += " | with external distortion"

    cam_model = CameraModel.from_parameters(cam_model_params, device=device)

    # Initialize the lidar model if requested
    lidar_model: Optional[StructuredLidarModel] = None
    if enable_lidar_model:
        assert isinstance(pc_sensor, LidarSensor), "only lidar sensors are supported for lidar model"

        lidar_model = StructuredLidarModel.maybe_from_parameters(pc_sensor.get_lidar_model_parameters(), device=device)

        assert lidar_model is not None, f"No structured lidar model available for lidar sensor {sensor_id}"

        msg += " | with structured lidar model"

    for frame_index in tqdm.tqdm(indices):
        # Get the camera timestamp and find the closes lidar frame
        cam_timestamp = cam_sensor.get_frame_timestamp_us(frame_index)
        pc_frame_index = pc_sensor.get_closest_frame_index(cam_timestamp)

        # Load the camera image and the point cloud
        img_frame = cam_sensor.get_frame_image_array(frame_index)
        pc = pc_sensor.get_frame_data(pc_frame_index, "xyz_e")

        if lidar_model is not None:
            ## Generate sensor points from model elements with length of the source data
            sensor_pc = (
                lidar_model.elements_to_sensor_points(
                    pc_sensor.get_frame_data(pc_frame_index, "model_element"),
                    np.linalg.norm(pc - pc_sensor.get_frame_data(pc_frame_index, "xyz_s"), axis=1),
                )
                .cpu()
                .numpy()
            )

            ## Perform motion-compensation
            motion_compensator = MotionCompensator(
                T_sensor_rig=T_sensor_rig,
                T_rig_worlds=poses.T_rig_worlds,
                T_rig_worlds_timestamps_us=poses.T_rig_world_timestamps_us,
            )

            pc = motion_compensator.motion_compensate_points(
                xyz_pointtime=sensor_pc,
                timestamp_us=pc_sensor.get_frame_data(pc_frame_index, "timestamp_us"),
                frame_start_timestamp_us=pc_sensor.get_frame_timestamp_us(pc_frame_index, types.FrameTimepoint.START),
                frame_end_timestamp_us=pc_sensor.get_frame_timestamp_us(pc_frame_index, types.FrameTimepoint.END),
            ).xyz_e_sensorend

        # Transform the point cloud to the world coordinate frame
        pc = transform_point_cloud(pc, pc_sensor.get_frame_T_rig_world(pc_frame_index) @ T_sensor_rig)

        T_world_camera_start = se3_inverse(
            cam_sensor.get_frame_T_rig_world(frame_index, types.FrameTimepoint.START) @ T_camera_rig
        )
        T_world_camera_end = se3_inverse(
            cam_sensor.get_frame_T_rig_world(frame_index, types.FrameTimepoint.END) @ T_camera_rig
        )

        logger.info(msg)

        match pose:
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
        if encode_images:
            assert len(output_dir)

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            if timestamp_image_names:
                save_path = output_path / (file_prefix + str(cam_timestamp) + file_suffix + ".png")
            else:
                save_path = output_path / (file_prefix + padded_index_string(frame_index) + file_suffix + ".png")

        plot_points_on_image(
            np.concatenate((image_point_coords[:, :2], dist_rs), axis=1),
            img_frame,
            msg if not encode_images else "",
            point_size=point_size,
            show=not encode_images,
            save_path=str(save_path),
        )


if __name__ == "__main__":
    ncore_project_pc_to_img(show_default=True)
