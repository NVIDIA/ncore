# Copyright 2019 The Waymo Open Dataset Authors.
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
#
# ========================================================================
#
# Modifications Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This file contains functions derived from the Waymo Open Dataset
# (https://github.com/waymo-research/waymo-open-dataset, tag v1.5.1).
# The original functions have been modified to support additional outputs
# (timestamps, azimuths, ray origins) and use a fast protobuf parser.
#
# Verbatim (unmodified) utility functions are imported directly from the
# upstream waymo_open_dataset package rather than being duplicated here.
# See:
#   - waymo_open_dataset.utils.range_image_utils: compute_inclination,
#     _combined_static_and_dynamic_shape
#   - waymo_open_dataset.utils.transform_utils: get_rotation_matrix,
#     get_transform

"""Modified Waymo range image utilities for NCore data conversion.

Functions in this module are derived from the following upstream sources:
  - waymo_open_dataset/utils/range_image_utils.py
  - waymo_open_dataset/utils/frame_utils.py

Key modifications from upstream:
  - compute_range_image_polar: also returns azimuth tensor
  - compute_range_image_cartesian: adds timestamp computation, ray origin
    (range_image_center), and return_local_coordinates flag
  - extract_point_cloud_from_range_image: adds start_end_timestamps parameter,
    returns timestamps and azimuths
  - parse_range_image_and_segmentations: uses fast protobuf parser, processes
    a single laser at a time, drops camera projections
"""

from typing import Optional

import numpy as np

from tools.data_converter.waymo.deps import (
    dataset_pb2,
    tf,
    waymo_range_image_utils,
    waymo_transform_utils,
)
from tools.data_converter.waymo.proto_utils import ParsedMatrix, parse_matrix_proto


# Re-export verbatim upstream functions used by other modules in this package.
# These are NOT duplicated — they come directly from the Waymo Open Dataset.
compute_inclination = waymo_range_image_utils.compute_inclination
get_rotation_matrix = waymo_transform_utils.get_rotation_matrix
get_transform = waymo_transform_utils.get_transform
_combined_static_and_dynamic_shape = waymo_range_image_utils._combined_static_and_dynamic_shape


def parse_range_image_and_segmentations(frame, laser_name, ri_index: int = 0, range_image_top_pose=None):
    """Parse range images and segmentations given a frame.

    Derived from waymo_open_dataset.utils.frame_utils.parse_range_image_and_camera_projection
    with the following modifications:
      - Uses fast protobuf wire-format parser instead of ParseFromString
      - Processes a single laser at a time (caller specifies laser_name)
      - Drops camera projection parsing (not needed for NCore conversion)
      - Accepts pre-parsed range_image_top_pose to avoid redundant decompression

    Args:
       frame: open dataset frame proto
       laser_name: the name of the laser sensor to process
       ri_index: 0 for the first return, 1 for the second return.
       range_image_top_pose: optional pre-parsed top-lidar pose to avoid redundant decompression.

    Returns:
       range_image: parsed range image matrix.
       segmentations: Optional parsed segmentation matrix.
       range_image_top_pose: parsed top-lidar per-pixel pose matrix.
    """

    range_image: Optional[ParsedMatrix] = None
    segmentation: Optional[ParsedMatrix] = None
    need_top_pose = range_image_top_pose is None

    for laser in frame.lasers:
        laser_ri_return = laser.ri_return1 if ri_index == 0 else laser.ri_return2

        if need_top_pose and laser.name == dataset_pb2.LaserName.TOP:
            range_image_top_pose = parse_matrix_proto(laser_ri_return.range_image_pose_compressed, np.float32)

        if laser.name != laser_name:
            continue

        if len(laser_ri_return.range_image_compressed) > 0:  # pylint: disable=g-explicit-length-test
            range_image = parse_matrix_proto(laser_ri_return.range_image_compressed, np.float32)

        if len(laser_ri_return.segmentation_label_compressed) > 0:
            segmentation = parse_matrix_proto(laser_ri_return.segmentation_label_compressed, np.int32)

    return range_image, segmentation, range_image_top_pose


def extract_point_cloud_from_range_image(
    range_image,
    extrinsic,
    inclination,
    start_end_timestamps,
    pixel_pose=None,
    frame_pose=None,
    dtype=tf.float32,
    scope=None,
):
    """Extracts point cloud from range image.

    Derived from waymo_open_dataset.utils.range_image_utils.extract_point_cloud_from_range_image
    with the following modifications:
      - Accepts start_end_timestamps for per-point timestamp approximation
      - Returns (range_image_cartesian, timestamps, azimuths) instead of just cartesian
      - Uses modified compute_range_image_polar that also returns azimuths
      - Uses modified compute_range_image_cartesian that computes timestamps

    Args:
      range_image: [B, H, W] tensor. Lidar range images.
      extrinsic: [B, 4, 4] tensor. Lidar extrinsic.
      inclination: [B, H] tensor. Inclination for each row of the range image.
        0-th entry corresponds to the 0-th row of the range image.
      start_end_timestamps [2]: timestamp bounds to approximate point timestamps from.
      pixel_pose: [B, H, W, 4, 4] tensor. If not None, it sets pose for each range
        image pixel.
      frame_pose: [B, 4, 4] tensor. This must be set when pixel_pose is set. It
        decides the vehicle frame at which the cartesian points are computed.
      dtype: float type to use internally. This is needed as extrinsic and
        inclination sometimes have higher resolution than range_image.
      scope: the name scope.

    Returns:
      range_image_cartesian: [B, H, W, 3] with {x, y, z} as inner dims in world frame.
    """
    with tf.compat.v1.name_scope(
        scope,
        "ExtractPointCloudFromRangeImage",
        [range_image, extrinsic, inclination, pixel_pose, frame_pose],
    ):
        range_image_polar, azimuths = compute_range_image_polar(range_image, extrinsic, inclination, dtype=dtype)
        range_image_cartesian, timestamps = compute_range_image_cartesian(
            range_image_polar,
            extrinsic,
            start_end_timestamps,
            pixel_pose=pixel_pose,
            frame_pose=frame_pose,
            dtype=dtype,
            return_local_coordinates=False,
        )

        return range_image_cartesian, timestamps, azimuths


def compute_range_image_polar(range_image, extrinsic, inclination, dtype=tf.float32, scope=None):
    """Computes range image polar coordinates.

    Derived from waymo_open_dataset.utils.range_image_utils.compute_range_image_polar
    with the following modification:
      - Also returns the azimuth tensor as a second output

    Args:
      range_image: [B, H, W] tensor. Lidar range images.
      extrinsic: [B, 4, 4] tensor. Lidar extrinsic.
      inclination: [B, H] tensor. Inclination for each row of the range image.
        0-th entry corresponds to the 0-th row of the range image.
      dtype: float type to use internally. This is needed as extrinsic and
        inclination sometimes have higher resolution than range_image.
      scope: the name scope.

    Returns:
      range_image_polar: [B, H, W, 3] polar coordinates.
      azimuth: [B, W] azimuth angles.
    """
    # pylint: disable=unbalanced-tuple-unpacking
    _, height, width = _combined_static_and_dynamic_shape(range_image)
    range_image_dtype = range_image.dtype
    range_image = tf.cast(range_image, dtype=dtype)
    extrinsic = tf.cast(extrinsic, dtype=dtype)
    inclination = tf.cast(inclination, dtype=dtype)

    with tf.compat.v1.name_scope(scope, "ComputeRangeImagePolar", [range_image, extrinsic, inclination]):
        with tf.compat.v1.name_scope("Azimuth"):
            # [B].
            az_correction = tf.atan2(extrinsic[..., 1, 0], extrinsic[..., 0, 0])
            # [W].
            ratios = (tf.cast(tf.range(width, 0, -1), dtype=dtype) - 0.5) / tf.cast(width, dtype=dtype)
            # [B, W].
            azimuth = (ratios * 2.0 - 1.0) * np.pi - tf.expand_dims(az_correction, -1)

        # [B, H, W]
        azimuth_tile = tf.tile(azimuth[:, tf.newaxis, :], [1, height, 1])
        # [B, H, W]
        inclination_tile = tf.tile(inclination[:, :, tf.newaxis], [1, 1, width])
        range_image_polar = tf.stack([azimuth_tile, inclination_tile, range_image], axis=-1)
        return tf.cast(range_image_polar, dtype=range_image_dtype), azimuth


def compute_range_image_cartesian(
    range_image_polar,
    extrinsic,
    start_end_timestamps,
    pixel_pose=None,
    frame_pose=None,
    dtype=tf.float32,
    scope=None,
    return_local_coordinates=False,
):
    """Computes range image cartesian coordinates from polar ones.

    Derived from waymo_open_dataset.utils.range_image_utils.compute_range_image_cartesian
    with the following modifications:
      - Accepts start_end_timestamps and computes per-pixel timestamps
      - Computes ray origin (range_image_center) alongside hit points
      - Adds return_local_coordinates flag (upstream always returns vehicle frame;
        this defaults to world frame)
      - Returns ((range_image_points, range_image_center), timestamps) instead
        of just range_image_points

    Args:
      range_image_polar: [B, H, W, 3] float tensor. Lidar range image in polar
        coordinate in sensor frame.
      extrinsic: [B, 4, 4] float tensor. Lidar extrinsic.
      start_end_timestamps [2]: timestamp bounds to approximate point timestamps from.
      pixel_pose: [B, H, W, 4, 4] float tensor. If not None, it sets pose for each
        range image pixel.
      frame_pose: [B, 4, 4] float tensor. This must be set when pixel_pose is set.
        It decides the vehicle frame at which the cartesian points are computed.
      dtype: float type to use internally. This is needed as extrinsic and
        inclination sometimes have higher resolution than range_image.
      scope: the name scope.
      return_local_coordinates: if True, transform back to vehicle frame.

    Returns:
      range_image_cartesian: ((points [B,H,W,3], center [B,H,W,3]), timestamps [B,H,W]).
    """
    range_image_polar_dtype = range_image_polar.dtype
    range_image_polar = tf.cast(range_image_polar, dtype=dtype)
    extrinsic = tf.cast(extrinsic, dtype=dtype)
    if pixel_pose is not None:
        pixel_pose = tf.cast(pixel_pose, dtype=dtype)
    if frame_pose is not None:
        frame_pose = tf.cast(frame_pose, dtype=dtype)

    with tf.compat.v1.name_scope(
        scope,
        "ComputeRangeImageCartesian",
        [range_image_polar, extrinsic, pixel_pose, frame_pose],
    ):
        azimuth, inclination, range_image_range = tf.unstack(range_image_polar, axis=-1)

        cos_azimuth = tf.cos(azimuth)
        sin_azimuth = tf.sin(azimuth)
        cos_incl = tf.cos(inclination)
        sin_incl = tf.sin(inclination)

        # [B, H, W].
        x = cos_azimuth * cos_incl * range_image_range
        y = sin_azimuth * cos_incl * range_image_range
        z = sin_incl * range_image_range

        ## Approximate point timestamps (see https://github.com/waymo-research/waymo-open-dataset/issues/619#issuecomment-1496410092)
        #  inferred from column index [B, H, W]
        nRows, nCols = azimuth.shape[1:]
        column_relative_times = tf.reshape(tf.range(nCols) / (nCols - 1), (-1, 1, nCols))
        relative_times = tf.repeat(column_relative_times, nRows, axis=1)
        # *clockwise* spinning lidar lidar (largest azimuth are measured first)
        range_image_timestamps = start_end_timestamps[0] + tf.cast(
            relative_times * (start_end_timestamps[1] - start_end_timestamps[0]),
            dtype=start_end_timestamps[0].dtype,
        )

        # [B, H, W, 3]
        range_image_points = tf.stack([x, y, z], -1)
        range_image_center = tf.zeros_like(range_image_points)

        # [B, 3, 3]
        rotation = extrinsic[..., 0:3, 0:3]
        # translation [B, 1, 3]
        translation = tf.expand_dims(tf.expand_dims(extrinsic[..., 0:3, 3], 1), 1)

        # To vehicle frame.
        # [B, H, W, 3]
        range_image_points = tf.einsum("bkr,bijr->bijk", rotation, range_image_points) + translation

        range_image_center = tf.einsum("bkr,bijr->bijk", rotation, range_image_center) + translation
        if pixel_pose is not None:
            # To global frame.
            # [B, H, W, 3, 3]
            pixel_pose_rotation = pixel_pose[..., 0:3, 0:3]
            # [B, H, W, 3]
            pixel_pose_translation = pixel_pose[..., 0:3, 3]
            # [B, H, W, 3]
            range_image_points = (
                tf.einsum("bhwij,bhwj->bhwi", pixel_pose_rotation, range_image_points) + pixel_pose_translation
            )

            range_image_center = (
                tf.einsum("bhwij,bhwj->bhwi", pixel_pose_rotation, range_image_center) + pixel_pose_translation
            )

            if frame_pose is None:
                raise ValueError("frame_pose must be set when pixel_pose is set.")

            if return_local_coordinates:
                # To vehicle frame corresponding to the given frame_pose
                # [B, 4, 4]
                world_to_vehicle = tf.linalg.inv(frame_pose)
                world_to_vehicle_rotation = world_to_vehicle[:, 0:3, 0:3]
                world_to_vehicle_translation = world_to_vehicle[:, 0:3, 3]
                # [B, H, W, 3]
                range_image_points = (
                    tf.einsum("bij,bhwj->bhwi", world_to_vehicle_rotation, range_image_points)
                    + world_to_vehicle_translation[:, tf.newaxis, tf.newaxis, :]
                )
                range_image_center = (
                    tf.einsum("bij,bhwj->bhwi", world_to_vehicle_rotation, range_image_center)
                    + world_to_vehicle_translation[:, tf.newaxis, tf.newaxis, :]
                )

        range_image_points = tf.cast(range_image_points, dtype=range_image_polar_dtype)
        range_image_center = tf.cast(range_image_center, dtype=range_image_polar_dtype)

        return (range_image_points, range_image_center), range_image_timestamps
