# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import logging

from pathlib import Path
from typing import Optional

import click
import tqdm
import numpy as np
from point_cloud_utils import TriangleMesh

from ncore.impl.data.data3 import ShardDataLoader, PointCloudSensor, CameraSensor
from ncore.impl.data import types
from ncore.impl.common.transformations import transform_point_cloud
from ncore.impl.sensors.camera import CameraModel
from ncore.impl.data.util import padded_index_string


@click.command()
@click.option(
    "--shard-file-pattern", type=str, help="Data shard pattern to load (supports range expansion)", required=True
)
@click.option("--output-dir", type=str, help="Path to the output folder", required=True)
@click.option("--sensor-id", type=str, help="Sensor to export ply files for", default="lidar_gt_top_p128_v4p5")
@click.option(
    "--camera-id",
    type=str,
    help="Sensor on which points will be projected to color",
    default="camera_front_wide_120fov",
)
@click.option(
    "--start-frame",
    type=click.IntRange(min=0, max_open=True),
    help="Initial point-cloud frame to be used",
    default=None,
)
@click.option(
    "--stop-frame",
    type=click.IntRange(min=0, max_open=True),
    help="Past-the-end point-cloud frame to be exported",
    default=None,
)
@click.option(
    "--step-frame",
    type=click.IntRange(min=1, max_open=True),
    help="Step used to downsample the number of point-cloud frames",
    default=None,
)
@click.option(
    "--device", type=click.Choice(["cuda", "cpu"]), help="Device used for the computation via torch", default="cuda"
)
@click.option(
    "--camera-pose",
    type=click.Choice(["rolling-shutter", "mean", "start", "end"]),
    help="Per-pixel poses to use (rolling-shutter optimization, mean frame pose, start frame pose, end frame pose) ",
    default="rolling-shutter",
)
@click.option(
    "--output-filepattern",
    type=click.Choice(["frame-index", "timestamps-us"]),
    help="PLY output filename pattern, either store by <frame-index>.ply or by <timestamp-us>.ply [end-of-frame timestamp]",
    default="frame-index",
)
def ncore_export_colored_pc(
    shard_file_pattern: str,
    output_dir: str,
    sensor_id: str,
    camera_id: str,
    start_frame: Optional[int],
    stop_frame: Optional[int],
    step_frame: Optional[int],
    device: str,
    camera_pose: str,
    output_filepattern: str,
):
    """Projects the point cloud to the camera image, comparing projection w. and w/o rolling shutter compensation"""

    # Initialize the logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    shards = ShardDataLoader.evaluate_shard_file_pattern(shard_file_pattern)
    loader = ShardDataLoader(shards)
    assert isinstance(
        pc_sensor := loader.get_sensor(sensor_id), PointCloudSensor
    ), "only point-cloud sensors are supported as source sensors"
    assert isinstance(
        cam_sensor := loader.get_sensor(camera_id), CameraSensor
    ), "only camera sensors are supported as color sensors"

    # Initialize the camera model on requested device
    cam_model = CameraModel.from_parameters(cam_sensor.get_camera_model_parameters(), device=device)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get the camera frame indices from the index range
    pc_frame_indices = pc_sensor.get_frame_index_range(start_frame, stop_frame, step_frame)
    logger.info(f"Starting the pc projection. {len(pc_frame_indices)} frames will be processed.")
    for pc_frame_index in tqdm.tqdm(pc_frame_indices):

        # Get the pc timestamp and find the closes camera frame
        pc_timestamp_us = pc_sensor.get_frame_timestamp_us(pc_frame_index)
        cam_frame_index = cam_sensor.get_closest_frame_index(pc_timestamp_us)

        # Load the camera image and the point cloud
        img_frame = cam_sensor.get_frame_image_array(cam_frame_index)
        pc = pc_sensor.get_frame_data(pc_frame_index, "xyz_e")

        # Transform the point cloud to the world coordinate frame
        pc = transform_point_cloud(pc, pc_sensor.get_frame_T_sensor_world(pc_frame_index))

        T_world_sensor_start = cam_sensor.get_frame_T_world_sensor(cam_frame_index, types.FrameTimepoint.START)
        T_world_sensor_end = cam_sensor.get_frame_T_world_sensor(cam_frame_index, types.FrameTimepoint.END)

        logger.info(f"Starting the projection with torch implementation on device={device}")

        match camera_pose:
            case "rolling-shutter":
                world_point_projections = cam_model.world_points_to_image_points_shutter_pose(
                    pc, T_world_sensor_start, T_world_sensor_end, return_valid_indices=True, return_T_world_sensors=True
                )

            case "mean":
                world_point_projections = cam_model.world_points_to_image_points_mean_pose(
                    pc, T_world_sensor_start, T_world_sensor_end, return_valid_indices=True, return_T_world_sensors=True
                )

            case "start":
                world_point_projections = cam_model.world_points_to_image_points_static_pose(
                    pc, T_world_sensor_start, return_valid_indices=True, return_T_world_sensors=True
                )

            case "end":
                world_point_projections = cam_model.world_points_to_image_points_static_pose(
                    pc, T_world_sensor_end, return_valid_indices=True, return_T_world_sensors=True
                )

        assert world_point_projections.T_world_sensors is not None and world_point_projections.valid_indices is not None

        image_point_coords = world_point_projections.image_points.cpu().numpy()
        valid_idx = world_point_projections.valid_indices.cpu().numpy()

        point_colors = img_frame[
            np.floor(image_point_coords[:, 1]).astype(int), np.floor(image_point_coords[:, 0]).astype(int)
        ]

        tm = TriangleMesh()
        tm.vertex_data.positions = pc[valid_idx]
        tm.vertex_data.colors = point_colors

        # Save the ply file
        match output_filepattern:
            case "frame-index":
                tm.save(str(output_path / (padded_index_string(pc_frame_index) + ".ply")))
            case "timestamps-us":
                tm.save(str(output_path / (str(pc_timestamp_us) + ".ply")))


if __name__ == "__main__":
    ncore_export_colored_pc()
