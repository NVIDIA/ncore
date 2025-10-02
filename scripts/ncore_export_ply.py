# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import logging

from pathlib import Path
from typing import Optional

import click
import numpy as np
import tqdm

from point_cloud_utils import TriangleMesh

from ncore.impl.common.transformations import transform_point_cloud
from ncore.impl.data.data3 import LidarSensor, PointCloudSensor, ShardDataLoader
from ncore.impl.data.util import padded_index_string
from scripts.util import get_dynamic_flag


@click.command()
@click.option(
    "--shard-file-pattern", type=str, help="Data shard pattern to load (supports range expansion)", required=True
)
@click.option("--output-dir", type=str, help="Path to the output folder", required=True)
@click.option("--sensor-id", type=str, help="Sensor to export ply files for", default="lidar_gt_top_p128_v4p5")
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
@click.option(
    "--frame",
    type=click.Choice(["sensor", "rig", "world"]),
    help="Frame to represent the point-cloud in",
    default="world",
)
@click.option(
    "--timestamp-frame-names/--no-timestamp-frame-names",
    is_flag=True,
    default=False,
    help="Store ply's with timestamp filenames or frame-index filenames",
)
def ncore_export_ply(
    shard_file_pattern: str,
    output_dir: str,
    sensor_id: str,
    start_frame: Optional[int],
    stop_frame: Optional[int],
    step_frame: Optional[int],
    frame: str,
    timestamp_frame_names: bool,
):
    """Exports the point cloud data to the ply format with named attributes"""

    # Initialize the logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    shards = ShardDataLoader.evaluate_shard_file_pattern(shard_file_pattern)
    loader = ShardDataLoader(shards)
    sensor = loader.get_sensor(sensor_id)
    assert isinstance(sensor, PointCloudSensor), "only point-cloud sensors supported"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    indices = sensor.get_frame_index_range(start_frame, stop_frame, step_frame)
    logger.info(f"Starting '.ply' export. {len(indices)} frames will be exported.")

    for frame_index in tqdm.tqdm(indices):
        # Setup target transformation
        if frame == "sensor":
            T_sensor_target = np.identity(4)
        elif frame == "rig":
            T_sensor_target = sensor.get_T_sensor_rig()
        elif frame == "world":
            T_sensor_target = sensor.get_frame_T_sensor_world(frame_index)

        pc = TriangleMesh()
        pc.vertex_data.positions = transform_point_cloud(sensor.get_frame_data(frame_index, "xyz_e"), T_sensor_target)
        pc.vertex_data.custom_attributes["xyz_s"] = transform_point_cloud(
            sensor.get_frame_data(frame_index, "xyz_s"), T_sensor_target
        )
        if isinstance(sensor, LidarSensor):
            # intensity N x 1
            pc.vertex_data.custom_attributes["intensity"] = sensor.get_frame_data(frame_index, "intensity")

            # conditional dynamic_flag N x 1
            if (dynamic_flag := get_dynamic_flag(sensor, frame_index)) is not None:
                pc.vertex_data.custom_attributes["dynamic_flag"] = dynamic_flag

            # Compute offset in "inverse" fashion to prevent wrapping around zero for uint64
            negative_offset_timestamp = (
                sensor.get_frame_timestamp_us(frame_index) - sensor.get_frame_data(frame_index, "timestamp_us")
            ).astype(np.int32)
            pc.vertex_data.custom_attributes["negative_offset_timestamp_us"] = negative_offset_timestamp

        # Save the ply file
        fname = (
            padded_index_string(frame_index)
            if not timestamp_frame_names
            else str(sensor.get_frame_timestamp_us(frame_index))
        )
        pc.save(str(output_path / (fname + ".ply")))


if __name__ == "__main__":
    ncore_export_ply()
