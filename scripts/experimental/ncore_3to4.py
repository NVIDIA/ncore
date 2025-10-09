# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Literal, Optional, Tuple

import click

from ncore.impl.common.common import time_bounds
from ncore.impl.data.data3 import ShardDataLoader
from ncore.impl.experimental.data.data4.conversion import NCore3To4
from scripts.util import TupleType


@dataclass(kw_only=True, slots=True, frozen=True)
class CLIBaseParams:
    """Parameters passed to non-command-based CLI part"""

    shard_file_pattern: str
    skip_suffixes: Tuple[str]
    open_consolidated: bool

    output_dir: str
    store_type: Literal["itar", "directory"]
    poses_component_group: Optional[str]
    intrinsics_component_group: Optional[str]
    camera_component_groups: Tuple[Tuple[str, str]]
    lidar_component_groups: Tuple[Tuple[str, str]]
    radar_component_groups: Tuple[Tuple[str, str]]
    cuboid_track_observations_component_group: Optional[str]

    no_cameras: bool
    camera_ids: Tuple[str]
    no_lidars: bool
    lidar_ids: Tuple[str]
    no_radars: bool
    radar_ids: Tuple[str]
    verbose: bool


@click.group()
@click.option(
    "--shard-file-pattern",
    type=str,
    help="NCore V3 data shard pattern to load (supports range expansion)",
    required=True,
)
@click.option(
    "--skip-suffix",
    "skip_suffixes",
    multiple=True,
    type=str,
    help="Shard suffixes to skip",
    default=None,
)
@click.option("--open-consolidated/--no-open-consolidated", default=True, help="Pre-load shard meta-data?")
@click.option("--output-dir", type=str, help="Path to the output folder", required=True)
@click.option(
    "--store-type",
    type=click.Choice(["itar", "directory"], case_sensitive=False),
    default="itar",
    show_default=True,
    help="Output store type",
)
@click.option("--poses-component-group", type=str, help="Target component group for 'poses'", default=None)
@click.option("--intrinsics-component-group", type=str, help="Target component group for 'intrinsics'", default=None)
@click.option(
    "--camera-component-group",
    "camera_component_groups",
    multiple=True,
    type=TupleType(2, ":"),
    help="Target component group for camera sensors (multiple value option, indexed by camera_id)",
    default=[],
)
@click.option(
    "--lidar-component-group",
    "lidar_component_groups",
    multiple=True,
    type=TupleType(2, ":"),
    help="Target component group for lidar sensors (multiple value option, indexed by lidar_id)",
    default=[],
)
@click.option(
    "--radar-component-group",
    "radar_component_groups",
    multiple=True,
    type=TupleType(2, ":"),
    help="Target component group for radar sensors (multiple value option, indexed by radar_id)",
    default=[],
)
@click.option(
    "--cuboid-track-observations-component-group",
    type=str,
    help="Target component group for 'cuboid_track_observations'",
    default=None,
)
@click.option("--no-cameras", is_flag=True, default=False, help="Disable exporting of any camera sensor")
@click.option(
    "--camera-id",
    "camera_ids",
    multiple=True,
    type=str,
    help="Cameras to be exported (multiple value option, all if not specified)",
    default=None,
)
@click.option("--no-lidars", is_flag=True, default=False, help="Disable exporting of any lidar sensor")
@click.option(
    "--lidar-id",
    "lidar_ids",
    multiple=True,
    type=str,
    help="Lidars to be exported (multiple value option, all if not specified)",
    default=None,
)
@click.option("--no-radars", is_flag=True, default=False, help="Disable exporting of any radar sensor")
@click.option(
    "--radar-id",
    "radar_ids",
    multiple=True,
    type=str,
    help="Radars to be exported (multiple value option, all if not specified)",
    default=None,
)
@click.option("--verbose", is_flag=True, default=False, help="Enables debug logging outputs")
@click.pass_context
def cli(ctx, **kwargs) -> None:
    """Extracts a time-based subrange of data from NCore shards and outputs the data as a new shard"""

    params = CLIBaseParams(**kwargs)

    # Initialize basic top-level logger configuration
    logging.basicConfig(
        level=logging.DEBUG if params.verbose else logging.INFO,
        format="<%(asctime)s|%(levelname)s|%(filename)s:%(lineno)d|%(name)s> %(message)s",
    )

    ctx.obj = params


def ncore_3to4(
    params: CLIBaseParams,
    loader: ShardDataLoader,
    start_timestamp_us: Optional[int] = None,
    end_timestamp_us: Optional[int] = None,
) -> None:
    """Execute common components of data conversion"""

    # Sensor selection
    camera_ids = list(params.camera_ids) if len(params.camera_ids) else None
    if params.no_cameras:
        camera_ids = []

    lidar_ids = list(params.lidar_ids) if len(params.lidar_ids) else None
    if params.no_lidars:
        lidar_ids = []

    radar_ids = list(params.radar_ids) if len(params.radar_ids) else None
    if params.no_radars:
        radar_ids = []

    ncore_4_paths = NCore3To4.convert(
        source_data_loader=loader,
        start_timestamp_us=start_timestamp_us,
        end_timestamp_us=end_timestamp_us,
        output_dir_path=Path(params.output_dir),
        store_type=params.store_type,
        camera_ids=camera_ids,
        lidar_ids=lidar_ids,
        radar_ids=radar_ids,
        poses_component_group=params.poses_component_group,
        intrinsics_component_group=params.intrinsics_component_group,
        camera_component_groups=dict(params.camera_component_groups),
        lidar_component_groups=dict(params.lidar_component_groups),
        radar_component_groups=dict(params.radar_component_groups),
        cuboid_track_observations_component_group=params.cuboid_track_observations_component_group,
    )

    logging.info(f"Wrote dataset to {ncore_4_paths}")


@cli.command()
@click.pass_context
def full(ctx) -> None:
    """Full dataset conversion"""

    params: CLIBaseParams = ctx.obj

    ncore_3to4(
        params,
        ShardDataLoader(
            ShardDataLoader.evaluate_shard_file_pattern(params.shard_file_pattern, params.skip_suffixes),
            params.open_consolidated,
        ),
    )


@cli.command()
@click.option(
    "--seek-sec",
    type=click.FloatRange(min=0.0, max_open=True),
    help="Time to skip for the dataset conversion (in seconds)",
)
@click.option(
    "--duration-sec",
    type=click.FloatRange(min=0.0, max_open=True),
    help="Restrict total duration of the dataset conversion (in seconds)",
)
@click.pass_context
def offset(ctx, seek_sec: float | None, duration_sec: float | None) -> None:
    """Offset-based subrange selection"""

    params: CLIBaseParams = ctx.obj

    # determine time-ranges from seek/duration relative to data
    loader = ShardDataLoader(
        ShardDataLoader.evaluate_shard_file_pattern(params.shard_file_pattern, params.skip_suffixes),
        params.open_consolidated,
    )

    start_timestamp_us, end_timestamp_us = time_bounds(
        loader.get_poses().T_rig_world_timestamps_us.tolist(), seek_sec, duration_sec
    )

    ncore_3to4(
        params,
        loader,
        start_timestamp_us,
        end_timestamp_us,
    )


if __name__ == "__main__":
    cli(show_default=True)
