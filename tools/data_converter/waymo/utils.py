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

"""NCore-specific utilities for Waymo data conversion.

This module contains original NVIDIA code for:
  - Pose extrapolation from velocity
  - Batched multi-return point cloud conversion with timestamps, inclinations,
    and azimuths
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from scipy.linalg import expm

from tools.data_converter.waymo.deps import dataset_pb2, tf
from tools.data_converter.waymo.proto_utils import ParsedMatrix
from tools.data_converter.waymo.waymo_range_image_utils import (
    compute_inclination,
    extract_point_cloud_from_range_image,
    get_rotation_matrix,
    get_transform,
)


def extrapolate_pose_based_on_velocity(
    T_SDC_global: np.ndarray, v_global: np.ndarray, w_global: np.ndarray, dt_sec: float
) -> np.ndarray:
    """Extrapolate a 4x4 pose matrix forward in time using linear and angular velocity.

    Args:
      T_SDC_global: [4, 4] current pose (rotation + translation).
      v_global: [3, 1] linear velocity in global frame.
      w_global: [3, 1] angular velocity in global frame.
      dt_sec: time delta in seconds.

    Returns:
      [4, 4] extrapolated pose.
    """
    T_extrapolated = np.eye(4, dtype=np.float32)
    T_extrapolated[:3, :3] = expm(get_skew_symmetric(w_global) * dt_sec) @ T_SDC_global[:3, :3]
    T_extrapolated[:3, 3:4] = T_SDC_global[:3, 3:4] + v_global * dt_sec

    return T_extrapolated


def get_skew_symmetric(vec: np.ndarray) -> np.ndarray:
    """Build the 3x3 skew-symmetric matrix for a 3-vector.

    Args:
      vec: [3, 1] or [3] vector.

    Returns:
      [3, 3] skew-symmetric matrix.
    """
    skew_sym = np.zeros((3, 3), dtype=np.float32)

    skew_sym[0, 1] = -vec[2]
    skew_sym[0, 2] = vec[1]
    skew_sym[1, 0] = vec[2]
    skew_sym[1, 2] = -vec[0]
    skew_sym[2, 0] = -vec[1]
    skew_sym[2, 1] = vec[0]

    return skew_sym


@dataclass
class _PointCloudReturn:
    """Result of converting a single range image return to a point cloud."""

    points: np.ndarray
    segmentation: np.ndarray | None
    timestamps: np.ndarray
    range_image_indices: np.ndarray
    inclinations: np.ndarray
    azimuths: np.ndarray


def convert_range_images_to_point_clouds(
    frame: Any,
    laser_name: str,
    range_images: list[ParsedMatrix],
    segmentation: ParsedMatrix | None,
    range_image_top_pose: ParsedMatrix,
    start_end_timestamps: np.ndarray,
) -> list[_PointCloudReturn]:
    """Convert one or more range image returns to point clouds in a single batched pass.

    Batches all returns along the B dimension so that the expensive TF operations
    (polar coordinate computation, extrinsic/pixel-pose transforms) run only once.
    The per-return mask and gather are applied afterwards since each return has
    different valid pixels.

    Args:
      frame: open dataset frame.
      laser_name: the name of the laser sensor to process.
      range_images: list of parsed range image matrices (one per return).
      segmentation: optional segmentation for the first return (None for others).
      range_image_top_pose: parsed top-lidar per-pixel pose matrix.
      start_end_timestamps: timestamp bounds to approximate point timestamps from.

    Returns:
      List of _PointCloudReturn, one per input range image.
    """
    n_returns = len(range_images)

    frame_pose = tf.convert_to_tensor(value=np.reshape(np.array(frame.pose.transform), [4, 4]))

    # Build top-pose [H, W, 4, 4] tensor
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(value=range_image_top_pose.data),
        range_image_top_pose.dims,
    )
    range_image_top_pose_tensor_rotation = get_rotation_matrix(
        range_image_top_pose_tensor[..., 0],
        range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2],
    )
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = get_transform(
        range_image_top_pose_tensor_rotation, range_image_top_pose_tensor_translation
    )

    results: list[_PointCloudReturn] = []

    for c in frame.context.laser_calibrations:
        if c.name != laser_name:
            continue

        if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
            beam_inclinations = compute_inclination(
                tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                height=range_images[0].dims[0],
            )
        else:
            beam_inclinations = tf.constant(c.beam_inclinations)

        beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
        extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

        # Stack all returns along B dimension: [n_returns, H, W, C]
        ri_tensors = [tf.reshape(tf.convert_to_tensor(value=ri.data), ri.dims) for ri in range_images]
        # Range channel only: [n_returns, H, W]
        range_batch = tf.stack([t[..., 0] for t in ri_tensors], axis=0)

        # Tile shared parameters to match batch size
        extrinsic_batch = tf.tile(tf.expand_dims(extrinsic, axis=0), [n_returns, 1, 1])
        inclination_batch = tf.tile(
            tf.expand_dims(tf.convert_to_tensor(value=beam_inclinations), axis=0),
            [n_returns, 1],
        )

        pixel_pose_batch = None
        frame_pose_batch = None
        if c.name == dataset_pb2.LaserName.TOP:
            pixel_pose_batch = tf.tile(
                tf.expand_dims(range_image_top_pose_tensor, axis=0),
                [n_returns, 1, 1, 1, 1],
            )
            frame_pose_batch = tf.tile(
                tf.expand_dims(frame_pose, axis=0),
                [n_returns, 1, 1],
            )

        # Single batched call for all returns
        range_image_cartesian, range_image_timestamps, azimuths = extract_point_cloud_from_range_image(
            range_batch,
            extrinsic_batch,
            inclination_batch,
            start_end_timestamps,
            pixel_pose=pixel_pose_batch,
            frame_pose=frame_pose_batch,
        )

        # Timestamps and azimuths are geometry-only (independent of range values),
        # so they're identical across returns -- use the first batch element.
        timestamps_full = range_image_timestamps[0]  # [H, W]
        azimuths_np = azimuths[0].numpy().astype(np.float32)
        inclinations_np = beam_inclinations.numpy().astype(np.float32)

        for ri_idx in range(n_returns):
            end_point_cartesian = range_image_cartesian[0][ri_idx]  # [H, W, 3]
            source_point_cartesian = range_image_cartesian[1][ri_idx]  # [H, W, 3]

            cartesian_full = tf.concat(
                [
                    source_point_cartesian,
                    end_point_cartesian,
                    ri_tensors[ri_idx][..., 1:2],  # intensity
                    ri_tensors[ri_idx][..., 2:3],  # elongation
                ],
                axis=-1,
            )  # [H, W, 8]

            range_image_mask = ri_tensors[ri_idx][..., 0] > 0
            range_image_indices_tensor = tf.compat.v1.where(range_image_mask)

            points = tf.gather_nd(cartesian_full, range_image_indices_tensor).numpy()

            seg_result = None
            if ri_idx == 0 and segmentation is not None:
                seq_tensor = tf.reshape(tf.convert_to_tensor(value=segmentation.data), segmentation.dims)
                seg_result = tf.gather_nd(seq_tensor, range_image_indices_tensor).numpy()

            ts = tf.gather_nd(timestamps_full, range_image_indices_tensor).numpy()
            ri_indices = range_image_indices_tensor.numpy().astype(np.uint32)

            results.append(
                _PointCloudReturn(
                    points=points,
                    segmentation=seg_result,
                    timestamps=ts,
                    range_image_indices=ri_indices,
                    inclinations=inclinations_np,
                    azimuths=azimuths_np,
                )
            )

    return results
