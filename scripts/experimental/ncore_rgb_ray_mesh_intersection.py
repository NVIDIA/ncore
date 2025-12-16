# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import point_cloud_utils as pcu
import torch
import tqdm

from scipy.ndimage.morphology import binary_dilation

from ncore.impl.data.data3 import ShardDataLoader
from ncore.impl.data.types import FrameTimepoint
from ncore.impl.data.util import padded_index_string
from ncore.impl.sensors.camera import CameraModel


@click.command()
@click.option(
    "--shard-file-pattern", type=str, help="Data shard pattern to load (supports range expansion)", required=True
)
@click.option("--mesh-path", type=str, help="Path to the reconstructed mesh", required=True)
@click.option("--camera-id", type=str, help="Camera sensor to be used for projection.", required=True)
@click.option("--start-frame", type=click.IntRange(min=0, max_open=True), help="Initial frame to be use", default=None)
@click.option(
    "--stop-frame", type=click.IntRange(min=0, max_open=True), help="Past-the-end frame to be used", default=None
)
@click.option(
    "--step-frame",
    type=click.IntRange(min=1, max_open=True),
    help="Step used to downsample the number of frames",
    default=None,
)
@click.option(
    "--static-camera-mask-dilations",
    type=click.IntRange(min=0, max_open=True),
    help="Number of image dilations to apply to static camera masks",
    default=20,
)
@click.option("--output-dir", type=str, help="Path to the output folder", required=True)
def ncore_rgb_ray_mesh_intersection(
    shard_file_pattern: str,
    mesh_path: str,
    camera_id: str,
    start_frame: Optional[int],
    stop_frame: Optional[int],
    step_frame: Optional[int],
    static_camera_mask_dilations: int,
    output_dir: str,
):
    # Initialize the logger
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    shards = ShardDataLoader.evaluate_shard_file_pattern(shard_file_pattern)

    logger.info(f"Shards: {shards}")

    loader = ShardDataLoader(shards)

    # Load camera sensor
    camera_sensor = loader.get_camera_sensor(camera_id)

    # Construct set of valid image pixels and precompute rays
    camera_model = CameraModel.from_parameters(
        camera_sensor.get_camera_model_parameters(), device="cpu", dtype=torch.float32
    )

    # sample pixel ranges
    w = int(camera_model.resolution[0].item())
    h = int(camera_model.resolution[1].item())

    # all pixels
    camera_pixels_x, camera_pixels_y = np.meshgrid(
        np.arange(w, dtype=np.int32), np.arange(h, dtype=np.int32)
    )  # [0, w-1] x [0, h-1]
    camera_all_pixels = np.stack([camera_pixels_x.flatten(), camera_pixels_y.flatten()], axis=1)

    # unmasked pixels
    if camera_mask_image := camera_sensor.get_camera_mask_image():
        # True for parts that we want to mask out
        camera_mask_array = np.asarray(camera_mask_image) != 0

        # Dilate mask boundary
        camera_mask_array = binary_dilation(camera_mask_array, iterations=static_camera_mask_dilations)

        # Subsample valid pixels relative to mask (True for parts that we want to keep)
        valid_pixel_mask = np.logical_not(camera_mask_array)

        camera_valid_pixels = np.stack([camera_pixels_x[valid_pixel_mask], camera_pixels_y[valid_pixel_mask]], axis=1)
    else:
        # No mask / consider all pixels as valid
        camera_valid_pixels = camera_all_pixels.copy()

    # precompute valid rays
    camera_valid_rays = camera_model.pixels_to_camera_rays(camera_valid_pixels).numpy()

    # Load mesh
    vertices, faces = pcu.load_mesh_vf(mesh_path)

    output_path = Path(output_dir) / "color_point_clouds" / camera_id
    output_path.mkdir(parents=True, exist_ok=True)

    # Get the camera frame indices from the index range
    camera_frame_indices = camera_sensor.get_frame_index_range(start_frame, stop_frame, step_frame)
    logger.info(
        f"Starting pc coloring. {len(camera_frame_indices)} frames will be processed and stored to {output_path}"
    )
    for camera_frame_index in tqdm.tqdm(camera_frame_indices):
        # Perform rolling-shutter-based world ray estimation
        start_timestamp_us = camera_sensor.get_frame_timestamp_us(
            camera_frame_index, frame_timepoint=FrameTimepoint.START
        )
        end_timestamp_us = camera_sensor.get_frame_timestamp_us(camera_frame_index, frame_timepoint=FrameTimepoint.END)
        T_sensor_world_start = camera_sensor.get_frame_T_sensor_world(
            camera_frame_index, frame_timepoint=FrameTimepoint.START
        )
        T_sensor_world_end = camera_sensor.get_frame_T_sensor_world(
            camera_frame_index, frame_timepoint=FrameTimepoint.END
        )
        world_rays_return = camera_model.pixels_to_world_rays_shutter_pose(
            camera_valid_pixels,
            T_sensor_world_start,
            T_sensor_world_end,
            start_timestamp_us=start_timestamp_us,
            end_timestamp_us=end_timestamp_us,
            camera_rays=camera_valid_rays,
            return_timestamps=True,
        )
        world_rays = world_rays_return.world_rays.numpy()

        # Perform rays-mesh intersection
        fid, bc, t = pcu.ray_mesh_intersection(
            vertices, faces, world_rays[:, :3].astype(np.float32), world_rays[:, 3:].astype(np.float32)
        )
        valid_rays = np.isfinite(t)

        # Load image
        img_frame_array = camera_sensor.get_frame_image_array(camera_frame_index)
        rgb_values = img_frame_array[camera_valid_pixels[:, 1], camera_valid_pixels[:, 0]].reshape([-1, 3])

        # Store colored point cloud
        pcu.save_mesh_vc(
            str(output_path / (padded_index_string(camera_frame_index) + ".ply")),
            pcu.interpolate_barycentric_coords(faces, fid[valid_rays], bc[valid_rays], vertices),
            rgb_values[valid_rays].astype(np.float32) / 255.0,
        )


if __name__ == "__main__":
    ncore_rgb_ray_mesh_intersection(show_default=True)
