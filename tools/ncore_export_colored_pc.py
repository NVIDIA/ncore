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
from ncore.impl.data.compat import PointCloudsSourceProtocol
from ncore.impl.data.util import padded_index_string
from ncore.impl.data.v4.compat import SequenceLoaderProtocol, SequenceLoaderV4
from ncore.impl.data.v4.components import SequenceComponentGroupsReader
from ncore.impl.sensors.camera import CameraModel


try:
    from .cli import OptionalStrParamType
except ImportError:
    from tools.cli import OptionalStrParamType


@dataclass(kw_only=True, slots=True, frozen=True)
class CLIBaseParams:
    """Parameters passed to non-command-based CLI part.

    Attributes:
        output_dir: Path to the output folder
        source_id: ID of the point cloud source to export colored PLY files for
        lidar_return_index: Return index of the lidar ray bundle sensor
        camera_id: ID of the camera sensor to project points onto for coloring
        start_pc: Optional starting pc index for export range
        stop_pc: Optional ending pc index (exclusive) for export range
        step_pc: Optional step size for downsampling point clouds
        device: Device used for computation via torch ('cuda' or 'cpu')
        camera_pose: Per-pixel poses to use for projection
        point_cloud_space: Output space of the colored point cloud ('world' or 'sensor')
        output_filepattern: PLY output filename pattern ('frame-index' or 'timestamps-us')
        use_source_rgb: Whether to use source RGB colors directly instead of camera projection
    """

    output_dir: str
    source_id: str
    lidar_return_index: int
    camera_id: str
    start_pc: Optional[int]
    stop_pc: Optional[int]
    step_pc: Optional[int]
    device: str
    camera_pose: str
    point_cloud_space: str
    output_filepattern: str
    use_source_rgb: bool


@click.group()
@click.option("--output-dir", type=str, help="Path to the output folder", required=True)
@click.option(
    "--source-id", type=str, help="Point cloud source to export colored PLY files for", default="lidar_gt_top_p128"
)
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
    "--start-pc",
    type=click.IntRange(min=0, max_open=True),
    help="Initial point-cloud index to be used",
    default=None,
)
@click.option(
    "--stop-pc",
    type=click.IntRange(min=0, max_open=True),
    help="Past-the-end point-cloud index to be exported",
    default=None,
)
@click.option(
    "--step-pc",
    type=click.IntRange(min=1, max_open=True),
    help="Step used to downsample the number of point clouds",
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
@click.option(
    "--use-source-rgb/--no-use-source-rgb",
    is_flag=True,
    default=False,
    help="Use RGB colors from the source directly instead of camera projection (only for sources with rgb attribute)",
)
@click.pass_context
def cli(ctx, **kwargs) -> None:
    """Projects the point cloud to the camera image and exports colored PLY files"""
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

    Projects point clouds onto camera images to obtain RGB colors for each point,
    accounting for rolling shutter effects if requested. If the source already has
    RGB colors and --use-source-rgb is set, uses those directly instead of projecting.

    Saves each frame as a PLY file containing both 3D positions and RGB colors.

    Args:
        params: CLI parameters specifying output location, sensors, and options
        loader: Sequence loader providing unified data access
    """

    # Initialize the logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    source_id = params.source_id
    is_sensor_source = source_id in loader.lidar_ids or source_id in loader.radar_ids

    source: PointCloudsSourceProtocol = loader.get_point_clouds_source(
        source_id, return_index=params.lidar_return_index
    )
    cam_sensor = loader.get_camera_sensor(params.camera_id)

    # Validate sensor-space request
    if params.point_cloud_space == "sensor" and not is_sensor_source:
        raise ValueError(
            f"Source '{source_id}' is not a sensor source (lidar/radar). Use --point-cloud-space world instead."
        )

    # Check if we can use source RGB directly
    use_source_rgb = params.use_source_rgb
    if use_source_rgb:
        if source.pcs_count > 0:
            test_pc = source.get_pc(0)
            if not test_pc.has_attribute("rgb"):
                logger.warning("Source '%s' has no RGB; falling back to camera projection", source_id)
                use_source_rgb = False
        else:
            logger.warning("Source has no point clouds, nothing to export.")
            return

    # Initialize the camera model on requested device (not needed if using source rgb)
    cam_model = (
        None if use_source_rgb else CameraModel.from_parameters(cam_sensor.model_parameters, device=params.device)
    )

    output_path = Path(params.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get the point cloud indices from the index range
    pc_indices = source.get_pc_index_range(params.start_pc, params.stop_pc, params.step_pc)
    logger.info(
        f"Starting colored PLY export for '{source_id}' and '{params.camera_id}' into '{output_path}'. "
        f"{len(pc_indices)} point clouds will be processed."
        + (" Using source RGB colors directly." if use_source_rgb else "")
    )

    for pc_index in tqdm.tqdm(pc_indices):
        point_cloud = source.get_pc(pc_index)
        pc_timestamp_us = int(source.pc_timestamps_us[pc_index])

        # Get xyz in source frame and world frame
        xyz_source = point_cloud.xyz
        world_pc = point_cloud.transform("world", pc_timestamp_us, loader.pose_graph)
        xyz_world = world_pc.xyz

        if use_source_rgb:
            rgb = point_cloud.get_attribute("rgb")

            tm = TriangleMesh()
            match params.point_cloud_space:
                case "world":
                    tm.vertex_data.positions = xyz_world
                case "sensor":
                    tm.vertex_data.positions = xyz_source
            tm.vertex_data.colors = rgb
        else:
            assert cam_model is not None

            # Find the closest camera frame
            cam_frame_index = cam_sensor.get_closest_frame_index(pc_timestamp_us)

            # Load the camera image
            img_frame = cam_sensor.get_frame_image_array(cam_frame_index)

            T_world_sensor_start, T_world_sensor_end = cam_sensor.get_frames_T_source_sensor(
                "world", cam_frame_index, frame_timepoint=None
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

            assert (
                world_point_projections.T_world_sensors is not None
                and world_point_projections.valid_indices is not None
            )

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
                    tm.vertex_data.positions = xyz_source[valid_idx]
            tm.vertex_data.colors = point_colors

        # Save the ply file
        match params.output_filepattern:
            case "frame-index":
                tm.save(str(output_path / (padded_index_string(pc_index) + ".ply")))
            case "timestamps-us":
                tm.save(str(output_path / (str(pc_timestamp_us) + ".ply")))

    logger.info(f"Exported {len(pc_indices)} colored PLY files to {output_path}")


if __name__ == "__main__":
    cli(show_default=True)
