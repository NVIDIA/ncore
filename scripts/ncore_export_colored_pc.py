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

from ncore.impl.common.transformations import transform_point_cloud
from ncore.impl.data import types
from ncore.impl.data.data3 import ShardDataLoader
from ncore.impl.data.data4.compat import SequenceLoaderProtocol, SequenceLoaderV3, SequenceLoaderV4
from ncore.impl.data.data4.components import SequenceComponentGroupsReader
from ncore.impl.data.util import padded_index_string
from ncore.impl.sensors.camera import CameraModel


@dataclass(kw_only=True, slots=True, frozen=True)
class CLIBaseParams:
    """Parameters passed to non-command-based CLI part.

    Attributes:
        output_dir: Path to the output folder
        lidar_id: ID of the lidar sensor to export colored PLY files for
        lidar_return_index: Return index of the lidar ray bundle sensor
        camera_id: ID of the camera sensor to project points onto for coloring
        start_frame: Optional starting frame index for export range
        stop_frame: Optional ending frame index (exclusive) for export range
        step_frame: Optional step size for downsampling frames
        device: Device used for computation via torch ('cuda' or 'cpu')
        camera_pose: Per-pixel poses to use for projection
        point_cloud_space: Output space of the colored point cloud ('world' or 'sensor')
        output_filepattern: PLY output filename pattern ('frame-index' or 'timestamps-us')
    """

    output_dir: str
    lidar_id: str
    lidar_return_index: int
    camera_id: str
    start_frame: Optional[int]
    stop_frame: Optional[int]
    step_frame: Optional[int]
    device: str
    camera_pose: str
    point_cloud_space: str
    output_filepattern: str


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
    "--camera-id",
    type=str,
    help="Camera sensor on which points will be projected to color",
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
    help="Per-pixel poses to use (rolling-shutter optimization, mean frame pose, start frame pose, end frame pose)",
    default="rolling-shutter",
)
@click.option(
    "--point-cloud-space",
    type=click.Choice(["world", "sensor"]),
    help="Output space of the colored point-cloud, either world space or local sensor space",
    default="world",
)
@click.option(
    "--output-filepattern",
    type=click.Choice(["frame-index", "timestamps-us"]),
    help="PLY output filename pattern, either store by <frame-index>.ply or by <timestamp-us>.ply [end-of-frame timestamp]",
    default="frame-index",
)
@click.pass_context
def cli(ctx, **kwargs) -> None:
    """Projects the point cloud to the camera image and exports colored PLY files"""
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
    """Export colored PLY files from NCore V3 (shard-based) sequence data.

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
    "component_groups", "--component-group", multiple=True, type=str, help="Data component group paths", required=True
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
    """Export colored PLY files from NCore V4 (component-based) sequence data.

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
    """Exports colored point cloud frames as PLY files.

    Projects lidar point clouds onto camera images to obtain RGB colors for each point,
    accounting for rolling shutter effects if requested. Saves each frame as a PLY file
    containing both 3D positions and RGB colors.

    Args:
        params: CLI parameters specifying output location, sensors, and options
        loader: Sequence loader (V3 or V4) providing unified data access
    """

    # Initialize the logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    pc_sensor = loader.get_lidar_sensor(params.lidar_id)
    cam_sensor = loader.get_camera_sensor(params.camera_id)

    # Initialize the camera model on requested device
    cam_model = CameraModel.from_parameters(cam_sensor.model_parameters, device=params.device)

    output_path = Path(params.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get the point cloud frame indices from the index range
    pc_frame_indices = pc_sensor.get_frame_index_range(params.start_frame, params.stop_frame, params.step_frame)
    logger.info(
        f"Starting colored PLY export for '{params.lidar_id}' and '{params.camera_id}' into '{output_path}'. "
        f"{len(pc_frame_indices)} frames will be processed."
    )

    for pc_frame_index in tqdm.tqdm(pc_frame_indices):
        # Get the pc timestamp and find the closest camera frame
        pc_timestamp_us = pc_sensor.get_frame_timestamp_us(pc_frame_index)
        cam_frame_index = cam_sensor.get_closest_frame_index(pc_timestamp_us)

        # Load the camera image and the point cloud
        img_frame = cam_sensor.get_frame_image_array(cam_frame_index)
        xyz_sensor = pc_sensor.get_frame_point_cloud(
            pc_frame_index, motion_compensation=True, with_start_points=False, return_index=params.lidar_return_index
        ).xyz_m_end

        # Transform the point cloud to the world coordinate frame
        xyz_world = transform_point_cloud(xyz_sensor, pc_sensor.get_frame_T_sensor_world(pc_frame_index))

        T_world_sensor_start, T_world_sensor_end = cam_sensor.get_frames_T_source_target(
            "world", cam_sensor.sensor_id, np.array(cam_frame_index)
        )

        logger.debug(f"Starting the projection with torch implementation on device={params.device}")

        match params.camera_pose:
            case "rolling-shutter":
                world_point_projections = cam_model.world_points_to_image_points_shutter_pose(
                    xyz_world,
                    T_world_sensor_start,
                    T_world_sensor_end,
                    return_valid_indices=True,
                    return_T_world_sensors=True,
                )

            case "mean":
                world_point_projections = cam_model.world_points_to_image_points_mean_pose(
                    xyz_world,
                    T_world_sensor_start,
                    T_world_sensor_end,
                    return_valid_indices=True,
                    return_T_world_sensors=True,
                )

            case "start":
                world_point_projections = cam_model.world_points_to_image_points_static_pose(
                    xyz_world, T_world_sensor_start, return_valid_indices=True, return_T_world_sensors=True
                )

            case "end":
                world_point_projections = cam_model.world_points_to_image_points_static_pose(
                    xyz_world, T_world_sensor_end, return_valid_indices=True, return_T_world_sensors=True
                )

        assert world_point_projections.T_world_sensors is not None and world_point_projections.valid_indices is not None

        image_point_coords = world_point_projections.image_points.cpu().numpy()
        valid_idx = world_point_projections.valid_indices.cpu().numpy()

        point_colors = img_frame[
            np.floor(image_point_coords[:, 1]).astype(int), np.floor(image_point_coords[:, 0]).astype(int)
        ]

        tm = TriangleMesh()
        match params.point_cloud_space:
            case "world":
                tm.vertex_data.positions = xyz_world[valid_idx]
            case "sensor":
                tm.vertex_data.positions = xyz_sensor[valid_idx]
        tm.vertex_data.colors = point_colors

        # Save the ply file
        match params.output_filepattern:
            case "frame-index":
                tm.save(str(output_path / (padded_index_string(pc_frame_index) + ".ply")))
            case "timestamps-us":
                tm.save(str(output_path / (str(pc_timestamp_us) + ".ply")))

    logger.info(f"Exported {len(pc_frame_indices)} colored PLY files to {output_path}")


if __name__ == "__main__":
    cli(show_default=True)
