# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import numpy as np
import tqdm

from point_cloud_utils import TriangleMesh

from ncore.impl.common.common import unpack_optional
from ncore.impl.common.transformations import transform_point_cloud
from ncore.impl.data.data3 import ShardDataLoader
from ncore.impl.data.data4.compat import SequenceLoaderProtocol, SequenceLoaderV3, SequenceLoaderV4
from ncore.impl.data.data4.components import SequenceComponentGroupsReader
from ncore.impl.data.util import padded_index_string


@dataclass(kw_only=True, slots=True, frozen=True)
class CLIBaseParams:
    """Parameters passed to non-command-based CLI part.

    Attributes:
        output_dir: Path to the output folder
        lidar_id: ID of the lidar sensor to export PLY files for
        lidar_return_index: Return index of the lidar ray bundle sensor
        start_frame: Optional starting frame index for export range
        stop_frame: Optional ending frame index (exclusive) for export range
        step_frame: Optional step size for downsampling frames
        frame: Reference frame for point cloud representation ('sensor', 'rig', or 'world')
        timestamp_frame_names: Whether to use timestamps for PLY filenames
        motion_compensation: Whether to use motion-compensated point clouds
    """

    output_dir: str
    lidar_id: str
    lidar_return_index: int
    start_frame: Optional[int]
    stop_frame: Optional[int]
    step_frame: Optional[int]
    frame: str
    timestamp_frame_names: bool
    motion_compensation: bool


@click.group()
@click.option("--output-dir", type=str, help="Path to the output folder", required=True)
@click.option("--lidar-id", type=str, help="Lidar sensor to export ply files for", default="lidar_gt_top_p128")
@click.option(
    "--lidar-return-index",
    type=int,
    help="Return index of the lidar ray bundle sensor",
    default=0,
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
@click.option(
    "--motion-compensation/--no-motion-compensation",
    default=True,
    help="Whether to use motion-compensated point clouds",
)
@click.pass_context
def cli(ctx, **kwargs) -> None:
    """Exports the point cloud data to the ply format with named attributes"""
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
    """Export PLY files from NCore V3 (shard-based) sequence data.

    Args:
        shard_file_pattern: Glob pattern for shard files (supports range expansion like shard_[0-10].zarr)
    """
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
    """Export PLY files from NCore V4 (component-based) sequence data.

    Args:
        component_groups: Paths to V4 component groups (can specify multiple)
        poses_component_group: Name of the poses component group to use
        intrinsics_component_group: Name of the intrinsics component group to use
        masks_component_group: Name of the masks component group to use
        cuboids_component_group: Name of the cuboids component group to use
    """
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
    """Exports point cloud frames as PLY files with named attributes.

    Saves each frame as a PLY file containing:
    - Point positions (xyz_e) transformed to the target frame
    - Start-of-frame positions (xyz_s) transformed to the target frame
    - Intensity values (for lidar sensors)
    - Dynamic flag (if available)
    - Negative offset timestamps (for lidar sensors)

    Args:
        params: CLI parameters specifying output location, sensor, and options
        loader: Sequence loader (V3 or V4) providing unified data access
    """

    # Initialize the logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    sensor = loader.get_lidar_sensor(params.lidar_id)

    output_path = Path(params.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    indices = sensor.get_frame_index_range(params.start_frame, params.stop_frame, params.step_frame)
    logger.info(
        f"Starting '.ply' export for '{params.lidar_id}' into '{output_path}'. {len(indices)} frames will be exported."
    )

    for frame_index in tqdm.tqdm(indices):
        # Setup target transformation
        if params.frame == "sensor":
            T_sensor_target = np.identity(4)
        elif params.frame == "rig":
            T_sensor_target = unpack_optional(sensor.T_sensor_rig)
        elif params.frame == "world":
            T_sensor_target = sensor.get_frames_T_sensor_target("world", frame_index)

        pc = TriangleMesh()
        pc_return = sensor.get_frame_point_cloud(
            frame_index,
            motion_compensation=params.motion_compensation,
            with_start_points=True,
            return_index=params.lidar_return_index,
        )
        pc.vertex_data.positions = transform_point_cloud(pc_return.xyz_m_end, T_sensor_target)
        pc.vertex_data.custom_attributes["xyz_s"] = transform_point_cloud(pc_return.xyz_m_start, T_sensor_target)

        # intensity N x 1
        pc.vertex_data.custom_attributes["intensity"] = sensor.get_frame_ray_bundle_return_intensity(
            frame_index, return_index=params.lidar_return_index
        )

        # conditional dynamic_flag N x 1
        if sensor.has_frame_generic_data(frame_index, "dynamic_flag"):
            pc.vertex_data.custom_attributes["dynamic_flag"] = sensor.get_frame_generic_data(
                frame_index, "dynamic_flag"
            )

        # Compute offset in "inverse" fashion to prevent wrapping around zero for uint64
        negative_offset_timestamp = (
            sensor.get_frame_timestamp_us(frame_index) - sensor.get_frame_ray_bundle_timestamp_us(frame_index)
        ).astype(np.int32)
        pc.vertex_data.custom_attributes["negative_offset_timestamp_us"] = negative_offset_timestamp

        # Save the ply file
        fname = (
            padded_index_string(frame_index)
            if not params.timestamp_frame_names
            else str(sensor.get_frame_timestamp_us(frame_index))
        )
        pc.save(str(output_path / (fname + ".ply")))

    logger.info(f"Exported {len(indices)} PLY files to {output_path}")


if __name__ == "__main__":
    cli(show_default=True)
