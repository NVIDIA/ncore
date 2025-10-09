# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""NCore label visualization tool.

This command-line tool provides interactive 3D visualization of labeled data (cuboids/bounding boxes)
overlaid on sensor point clouds for both V3 and V4 NCore data formats. It supports:
    - Camera and lidar sensor visualization
    - 3D bounding box rendering with track IDs and class labels
    - Motion-compensated and non-motion-compensated point clouds
    - Frame-by-frame navigation
    - CSV export of label statistics
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import click
import numpy as np
import pandas as pd
import tqdm

from ncore.impl.common.transformations import transform_bbox
from ncore.impl.common.visualization import LabelVisualizer
from ncore.impl.data.data3 import ShardDataLoader
from ncore.impl.data.types import BBox3, FrameLabel3
from ncore.impl.experimental.data.data4.compat import SequenceLoaderProtocol, SequenceLoaderV3, SequenceLoaderV4
from ncore.impl.experimental.data.data4.components import SequenceComponentStoreReader
from ncore.impl.experimental.data.data4.types import CuboidTrackObservation


@dataclass(kw_only=True, slots=True, frozen=True)
class CLIBaseParams:
    """Parameters passed to non-command-based CLI part.

    Attributes:
        sensor_id: ID of the sensor to visualize (e.g., 'lidar_gt_top_p128', 'camera_front')
        start_frame: Optional starting frame index for visualization range
        stop_frame: Optional ending frame index (exclusive) for visualization range
        step_frame: Optional step size for downsampling frames
        open_consolidated: Whether to pre-load consolidated zarr metadata
        motion_compensation: Whether to use motion-compensated point clouds
    """

    sensor_id: str
    start_frame: Optional[int]
    stop_frame: Optional[int]
    step_frame: Optional[int]
    open_consolidated: bool
    motion_compensation: bool


@click.group()
@click.option("--sensor-id", type=str, help="Sensor to visualize labels for", default="lidar_gt_top_p128")
@click.option(
    "--start-frame", type=click.IntRange(min=0, max_open=True), help="Initial frame to be visualized", default=None
)
@click.option(
    "--stop-frame", type=click.IntRange(min=0, max_open=True), help="Past-the-end frame to be visualized", default=None
)
@click.option(
    "--step-frame",
    type=click.IntRange(min=1, max_open=True),
    help="Step used to downsample the number of frames",
    default=None,
)
@click.option("--open-consolidated/--no-open-consolidated", default=True, help="Pre-load shard meta-data?")
@click.option(
    "--motion-compensation/--no-motion-compensation",
    default=True,
    help="Whether to render motion compensated (default) or non-motion compensated point-clouds?",
)
@click.pass_context
def cli(ctx, **kwargs) -> None:
    """Main CLI entry point for label visualization."""
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
    """Visualize labels from NCore V3 (shard-based) sequence data.

    Args:
        shard_file_pattern: Glob pattern for shard files (supports range expansion)
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
@click.option("data_stores", "--data-store", multiple=True, type=str, help="Data store paths", required=True)
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
    data_stores: Tuple[str, ...],
    poses_component_group: str,
    intrinsics_component_group: str,
    masks_component_group: str,
    cuboids_component_group: str,
) -> None:
    """Visualize labels from NCore V4 (component-based) sequence data.

    Args:
        data_stores: Paths to V4 component store files (can specify multiple)
        poses_component_group: Name of the poses component group to use
        intrinsics_component_group: Name of the intrinsics component group to use
        masks_component_group: Name of the masks component group to use
        cuboids_component_group: Name of the cuboids component group to use
    """
    params: CLIBaseParams = ctx.obj

    loader = SequenceComponentStoreReader(
        [Path(store_path) for store_path in data_stores],
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
    """Visualizes labeled data overlaid on sensor point clouds.

    Loads cuboid track observations and renders them as 3D bounding boxes on top of
    the sensor's point cloud data. Provides interactive visualization with frame-by-frame
    navigation.

    Args:
        params: CLI parameters specifying sensor, frame range, and visualization options
        loader: Sequence loader (V3 or V4) providing unified data access
    """
    sensor = loader.get_lidar_sensor(params.sensor_id)
    pose_graph = loader.pose_graph

    def cuboid_track_observation_to_frame(
        frame_timestamp_end_us: int, cuboid_track_observation: CuboidTrackObservation
    ) -> FrameLabel3:
        """Converts an observation to a FrameLabel3 instance."""

        # Transform observation from reference frame at observation time to sensor frame at frame end time
        cuboid_track_observation = cuboid_track_observation.transform(
            target_frame_id=params.sensor_id,
            target_frame_timestamp_us=frame_timestamp_end_us,
            pose_graph=pose_graph,
        )

        return FrameLabel3(
            label_id=cuboid_track_observation.observation_id,
            track_id=cuboid_track_observation.track_id,
            label_class=cuboid_track_observation.class_id,
            bbox3=cuboid_track_observation.bbox3,
            global_speed=float("nan"),
            timestamp_us=cuboid_track_observation.timestamp_us,
            confidence=None,
            source=cuboid_track_observation.source,
            source_version=cuboid_track_observation.source_version,
        )

    # Load track obversations and load into dataframe for easy querying
    cuboid_df = pd.DataFrame.from_records([obs.to_dict() for obs in loader.get_cuboid_track_observations()])

    for frame_index in tqdm.tqdm(
        sensor.get_frame_index_range(params.start_frame, params.stop_frame, params.step_frame)
    ):
        # Initialize the visualizer
        viz = LabelVisualizer()

        # Import the point cloud and add it to the visualizer
        xyz_m_end = sensor.get_frame_point_cloud(
            frame_index, motion_compensation=params.motion_compensation, with_start_points=False
        ).xyz_m_end
        intensity = sensor.get_frame_data(frame_index, "intensity")
        timestamp_us = sensor.get_frame_data(frame_index, "timestamp_us")

        dynamic_flag = (
            sensor.get_frame_generic_data(frame_index, "dynamic_flag")
            if sensor.has_frame_generic_data(frame_index, "dynamic_flag")
            else None
        )

        semantic_class = (
            sensor.get_frame_generic_data(frame_index, "semantic_class")
            if sensor.has_frame_generic_data(frame_index, "semantic_class")
            else None
        )

        viz.add_pc(frame_index, xyz_m_end, intensity, timestamp_us, dynamic_flag, semantic_class)

        # Query cuboid observations for this frame and convert to FrameLabel3
        frame_timestamps_us = sensor.frames_timestamps_us[frame_index]
        frame_cuboid_obs = cuboid_df.loc[
            (cuboid_df["timestamp_us"] >= frame_timestamps_us[0])
            & (cuboid_df["timestamp_us"] <= frame_timestamps_us[1])
        ]

        viz.add_labels(
            [
                cuboid_track_observation_to_frame(
                    frame_timestamps_us[1], CuboidTrackObservation.from_dict(row.to_dict())
                )
                for _, row in frame_cuboid_obs.iterrows()
            ]
        )

        # Show the point clouds
        viz.show()


if __name__ == "__main__":
    cli(show_default=True)
