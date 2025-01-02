# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import logging

from pathlib import Path
from typing import Optional

import click
import tqdm
import numpy as np

from scipy.spatial.transform import Rotation as R

from scripts.util import NPArrayParamType
from ncore.impl.data.data3 import ShardDataLoader, PointCloudSensor, CameraSensor
from ncore.impl.data import types
from ncore.impl.data.util import padded_index_string
from ncore.impl.common.transformations import se3_inverse, transform_point_cloud
from ncore.impl.common.visualization import plot_points_on_image
from ncore.impl.sensors.camera import CameraModel


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
    help="Per-pixel poses to use (rolling-shutter optimization, mean frame pose, start frame pose, end frame pose) ",
    default="rolling-shutter",
)
@click.option("--output-dir", type=str, help="Path to the output folder if encoding images", required=False, default="")
@click.option("--encode-images/--no-encode-images", is_flag=True, default=False, help="Encode image files for frames")
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
    encode_images: bool,
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

    for frame_index in tqdm.tqdm(indices):
        # Get the camera timestamp and find the closes lidar frame
        cam_timestamp = cam_sensor.get_frame_timestamp_us(frame_index)
        pc_frame_index = pc_sensor.get_closest_frame_index(cam_timestamp)

        # Load the camera image and the point cloud
        img_frame = cam_sensor.get_frame_image_array(frame_index)
        pc = pc_sensor.get_frame_data(pc_frame_index, "xyz_e")

        # Transform the point cloud to the world coordinate frame
        pc = transform_point_cloud(pc, pc_sensor.get_frame_T_rig_world(pc_frame_index) @ T_sensor_rig)

        T_world_camera_start = se3_inverse(
            cam_sensor.get_frame_T_rig_world(frame_index, types.FrameTimepoint.START) @ T_camera_rig
        )
        T_world_camera_end = se3_inverse(
            cam_sensor.get_frame_T_rig_world(frame_index, types.FrameTimepoint.END) @ T_camera_rig
        )

        # Initialize the camera model on requested device
        cam_model_params = cam_sensor.get_camera_model_parameters()
        cam_model = CameraModel.from_parameters(cam_model_params, device=device)

        logger.info(f"Starting the projection with torch implementation on device={device}")

        match pose:
            case "rolling-shutter":
                world_point_projections = cam_model.world_points_to_image_points_shutter_pose(
                    pc, T_world_camera_start, T_world_camera_end, return_valid_indices=True, return_T_world_sensors=True
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
                save_path = output_path / (str(cam_timestamp) + ".png")
            else:
                save_path = output_path / (padded_index_string(frame_index) + ".png")

        plot_points_on_image(
            np.concatenate((image_point_coords[:, :2], dist_rs), axis=1),
            img_frame,
            f"Projection with {pose} poses (torch implementation @ {device})" if not encode_images else "",
            point_size=point_size,
            show=not encode_images,
            save_path=str(save_path),
        )


if __name__ == "__main__":
    ncore_project_pc_to_img(show_default=True)
