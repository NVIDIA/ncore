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

import logging

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import click
import numpy as np
import tqdm

from point_cloud_utils import TriangleMesh

from ncore.impl.common.transformations import transform_point_cloud
from ncore.impl.common.util import unpack_optional
from ncore.impl.data.compat import PointCloudsSourceProtocol
from ncore.impl.data.util import padded_index_string
from ncore.impl.data.v4.compat import SequenceLoaderProtocol, SequenceLoaderV4
from ncore.impl.data.v4.components import SequenceComponentGroupsReader


try:
    from .cli import OptionalStrParamType
except ImportError:
    from tools.cli import OptionalStrParamType


@dataclass(kw_only=True, slots=True, frozen=True)
class CLIBaseParams:
    """Parameters passed to non-command-based CLI part.

    Attributes:
        output_dir: Path to the output folder
        source_id: ID of the point cloud source to export PLY files for
        lidar_return_index: Return index of the lidar ray bundle sensor
        start_pc: Optional starting pc index for export range
        stop_pc: Optional ending pc index (exclusive) for export range
        step_pc: Optional step size for downsampling point clouds
        frame: Reference frame for point cloud representation ('sensor', 'rig', or 'world')
        timestamp_frame_names: Whether to use timestamps for PLY filenames
        motion_compensation: Whether to use motion-compensated point clouds
    """

    output_dir: str
    source_id: str
    lidar_return_index: int
    start_pc: Optional[int]
    stop_pc: Optional[int]
    step_pc: Optional[int]
    frame: str
    timestamp_frame_names: bool
    motion_compensation: bool


@click.group()
@click.option("--output-dir", type=str, help="Path to the output folder", required=True)
@click.option("--source-id", type=str, help="Point cloud source to export PLY files for", default="lidar_gt_top_p128")
@click.option(
    "--lidar-return-index",
    type=int,
    help="Return index of the lidar ray bundle sensor",
    default=0,
)
@click.option(
    "--start-pc", type=click.IntRange(min=0, max_open=True), help="Initial pc index to be exported", default=None
)
@click.option(
    "--stop-pc", type=click.IntRange(min=0, max_open=True), help="Past-the-end pc index to be exported", default=None
)
@click.option(
    "--step-pc",
    type=click.IntRange(min=1, max_open=True),
    help="Step used to downsample the number of point clouds",
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
def cli(ctx, source_id, start_pc, stop_pc, step_pc, **kwargs) -> None:
    """Exports the point cloud data to the ply format with named attributes"""
    ctx.obj = CLIBaseParams(
        source_id=source_id,
        start_pc=start_pc,
        stop_pc=stop_pc,
        step_pc=step_pc,
        **kwargs,
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
    - Point positions (xyz) transformed to the target frame
    - Intensity values (for lidar sources)
    - RGB colors (if available from the source)
    - Dynamic flag (if available)
    - Negative offset timestamps (for lidar sources)

    For lidar-backed sources, also includes start-of-frame positions (xyz_s).

    Args:
        params: CLI parameters specifying output location, sensor, and options
        loader: Sequence loader providing unified data access
    """

    # Initialize the logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    source_id = params.source_id
    source: PointCloudsSourceProtocol = loader.get_point_clouds_source(
        source_id, return_index=params.lidar_return_index
    )

    # Determine if this source is backed by a sensor (lidar or radar)
    is_sensor_source = source_id in loader.lidar_ids or source_id in loader.radar_ids
    is_lidar = source_id in loader.lidar_ids
    lidar_sensor = loader.get_lidar_sensor(source_id) if is_lidar else None

    if params.frame == "sensor" and not is_sensor_source:
        raise ValueError(
            f"Cannot export in sensor frame: source '{source_id}' is a native point cloud. Use --frame world instead."
        )
    if params.frame == "rig" and not is_sensor_source:
        raise ValueError(
            f"Cannot export in rig frame: source '{source_id}' is a native point cloud. Use --frame world instead."
        )

    output_path = Path(params.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    indices = source.get_pc_index_range(params.start_pc, params.stop_pc, params.step_pc)
    logger.info(
        f"Starting '.ply' export for '{source_id}' into '{output_path}'. {len(indices)} point clouds will be exported."
    )

    for pc_index in tqdm.tqdm(indices):
        point_cloud = source.get_pc(pc_index)

        # Setup target transformation and get xyz in target frame
        if params.frame == "world":
            world_pc = point_cloud.transform("world", point_cloud.reference_frame_timestamp_us, loader.pose_graph)
            xyz_target = world_pc.xyz
        elif params.frame == "sensor":
            # Already in sensor frame, no transform needed
            xyz_target = point_cloud.xyz
        elif params.frame == "rig":
            if lidar_sensor is not None:
                T_sensor_rig = unpack_optional(lidar_sensor.T_sensor_rig)
                xyz_target = transform_point_cloud(point_cloud.xyz, T_sensor_rig)
            else:
                # For non-lidar sources, transform via pose graph to rig frame
                rig_pc = point_cloud.transform("rig", point_cloud.reference_frame_timestamp_us, loader.pose_graph)
                xyz_target = rig_pc.xyz

        tm = TriangleMesh()
        tm.vertex_data.positions = xyz_target

        # For lidar sources, include start-of-frame positions and lidar-specific attributes
        if is_lidar and lidar_sensor is not None:
            pc_return = lidar_sensor.get_frame_point_cloud(
                pc_index,
                motion_compensation=params.motion_compensation,
                with_start_points=True,
                return_index=params.lidar_return_index,
            )

            if params.frame == "sensor":
                T_sensor_target = np.identity(4)
            elif params.frame == "rig":
                T_sensor_target = unpack_optional(lidar_sensor.T_sensor_rig)
            elif params.frame == "world":
                T_sensor_target = lidar_sensor.get_frames_T_sensor_target("world", pc_index)
            tm.vertex_data.custom_attributes["xyz_s"] = transform_point_cloud(pc_return.xyz_m_start, T_sensor_target)

            # intensity N x 1
            if point_cloud.has_attribute("intensity"):
                tm.vertex_data.custom_attributes["intensity"] = point_cloud.get_attribute("intensity")

            # conditional dynamic_flag N x 1
            if source.has_pc_generic_data(pc_index, "dynamic_flag"):
                tm.vertex_data.custom_attributes["dynamic_flag"] = source.get_pc_generic_data(pc_index, "dynamic_flag")

            # Compute offset in "inverse" fashion to prevent wrapping around zero for uint64
            if point_cloud.has_attribute("timestamp_us"):
                frame_timestamp = source.pc_timestamps_us[pc_index]
                per_point_timestamps = point_cloud.get_attribute("timestamp_us")
                negative_offset_timestamp = (frame_timestamp - per_point_timestamps).astype(np.int32)
                tm.vertex_data.custom_attributes["negative_offset_timestamp_us"] = negative_offset_timestamp
        else:
            # Generic point cloud source: export available attributes
            if point_cloud.has_attribute("intensity"):
                tm.vertex_data.custom_attributes["intensity"] = point_cloud.get_attribute("intensity")

            if point_cloud.has_attribute("rgb"):
                tm.vertex_data.colors = point_cloud.get_attribute("rgb")

            if source.has_pc_generic_data(pc_index, "dynamic_flag"):
                tm.vertex_data.custom_attributes["dynamic_flag"] = source.get_pc_generic_data(pc_index, "dynamic_flag")

            if point_cloud.has_attribute("timestamp_us"):
                frame_timestamp = source.pc_timestamps_us[pc_index]
                per_point_timestamps = point_cloud.get_attribute("timestamp_us")
                negative_offset_timestamp = (frame_timestamp - per_point_timestamps).astype(np.int32)
                tm.vertex_data.custom_attributes["negative_offset_timestamp_us"] = negative_offset_timestamp

        # Save the ply file
        fname = (
            padded_index_string(pc_index)
            if not params.timestamp_frame_names
            else str(source.pc_timestamps_us[pc_index])
        )
        tm.save(str(output_path / (fname + ".ply")))

    logger.info(f"Exported {len(indices)} PLY files to {output_path}")


if __name__ == "__main__":
    cli(show_default=True)
