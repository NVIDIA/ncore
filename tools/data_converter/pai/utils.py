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

"""Utility functions for Physical AI data processing."""

import io
import json

from typing import Any, Dict, Tuple, cast

import numpy as np
import pandas as pd
import PIL.Image as PILImage

from scipy.spatial.transform import Rotation
from upath import UPath


def filter_ego_vehicle_points(
    points: np.ndarray,
    T_sensor_rig: np.ndarray,
    vehicle_bbox: Dict[str, float],
    padding: float = 0.5,
) -> np.ndarray:
    """Filter out points inside ego vehicle bounding box.

    Args:
        points: Nx3 point cloud in sensor frame
        T_sensor_rig: 4x4 SE3 transform (sensor to rig, despite the naming)
        vehicle_bbox: Dictionary with vehicle dimensions
        padding: Additional padding in meters

    Returns:
        Boolean mask of valid points (True = keep, False = filter)
    """

    # Transform points from sensor frame to rig frame
    # (T_sensor_rig is sensor-to-rig transform in NCore naming convention)
    points_hom = np.column_stack([points, np.ones(len(points))])
    points_rig = (T_sensor_rig @ points_hom.T).T[:, :3]

    # Define bounding box limits with padding
    half_length = vehicle_bbox["length"] / 2 + padding
    half_width = vehicle_bbox["width"] / 2 + padding
    half_height = vehicle_bbox["height"] / 2 + padding

    # Center offset
    center_x = vehicle_bbox["rear_axle_to_bbox_center"]

    # Check if points are outside bbox
    valid = np.logical_or.reduce(
        [
            points_rig[:, 0] < (center_x - half_length),
            points_rig[:, 0] > (center_x + half_length),
            points_rig[:, 1] < -half_width,
            points_rig[:, 1] > half_width,
            points_rig[:, 2] < -half_height,
            points_rig[:, 2] > half_height,
        ]
    )

    return valid


def find_clip_files(clip_dir: UPath, clip_id: str) -> Dict[str, UPath]:
    """Find all data files for a specific clip in pai-clip-dl output layout.

    For each parquet file, the ``.offline.`` variant is preferred if it exists
    (e.g. ``sensor_extrinsics.offline.parquet`` over ``sensor_extrinsics.parquet``).

    The pai-clip-dl tool produces per-clip directories with this structure::

        <clip_id>/
            calibration/
                camera_intrinsics[.offline].parquet
                sensor_extrinsics[.offline].parquet
                vehicle_dimensions.parquet
                lidar_intrinsics[.offline].parquet
            labels/
                <clip_id>.egomotion[.offline].parquet
                <clip_id>.obstacle[.offline].parquet
            camera/
                <clip_id>.<camera_name>.mp4
                <clip_id>.<camera_name>.timestamps.parquet
                <clip_id>.<camera_name>.blurred_boxes.parquet
            lidar/
                <clip_id>.lidar_top_360fov.parquet
            metadata/
                sensor_presence.parquet
                data_collection.parquet

    Args:
        clip_dir: Per-clip directory (e.g. ``<root>/<clip_id>/``)
        clip_id: Clip ID (UUID)

    Returns:
        Dictionary mapping data types to file paths. Keys ``"obstacle"`` and
        ``"lidar_intrinsics"`` are only present when the corresponding files exist.
    """

    def _prefer_offline(path: UPath) -> UPath:
        """Return the .offline. variant of a parquet path if it exists."""
        offline = path.parent / path.name.replace(".parquet", ".offline.parquet")
        return offline if offline.exists() else path

    # Calibration files are directly in calibration/
    calibration_dir = clip_dir / "calibration"
    files = {
        "egomotion": _prefer_offline(clip_dir / "labels" / f"{clip_id}.egomotion.parquet"),
        "camera_intrinsics": _prefer_offline(calibration_dir / "camera_intrinsics.parquet"),
        "sensor_extrinsics": _prefer_offline(calibration_dir / "sensor_extrinsics.parquet"),
        "vehicle_dimensions": _prefer_offline(calibration_dir / "vehicle_dimensions.parquet"),
    }

    # Obstacle labels
    obstacle_path = _prefer_offline(clip_dir / "labels" / f"{clip_id}.obstacle.parquet")
    if obstacle_path.exists():
        files["obstacle"] = obstacle_path

    # Lidar intrinsics (calibration)
    lidar_intrinsics_path = _prefer_offline(calibration_dir / "lidar_intrinsics.parquet")
    if lidar_intrinsics_path.exists():
        files["lidar_intrinsics"] = lidar_intrinsics_path

    # Camera files are flat in camera/
    camera_sensors = [
        "camera_cross_left_120fov",
        "camera_cross_right_120fov",
        "camera_front_tele_30fov",
        "camera_front_wide_120fov",
        "camera_rear_left_70fov",
        "camera_rear_right_70fov",
        "camera_rear_tele_30fov",
    ]

    camera_dir = clip_dir / "camera"
    for sensor in camera_sensors:
        video_path = camera_dir / f"{clip_id}.{sensor}.mp4"
        if video_path.exists():
            files[f"{sensor}_video"] = video_path
            files[f"{sensor}_timestamps"] = camera_dir / f"{clip_id}.{sensor}.timestamps.parquet"
            files[f"{sensor}_blurred_boxes"] = camera_dir / f"{clip_id}.{sensor}.blurred_boxes.parquet"

    # Lidar file is flat in lidar/
    lidar_path = clip_dir / "lidar" / f"{clip_id}.lidar_top_360fov.parquet"
    if lidar_path.exists():
        files["lidar_top_360fov"] = lidar_path

    return files


def quaternion_to_se3(qx: float, qy: float, qz: float, qw: float, x: float, y: float, z: float) -> np.ndarray:
    """Convert quaternion + position to SE3 transformation matrix.

    Args:
        qx, qy, qz, qw: Quaternion components
        x, y, z: Position components

    Returns:
        4x4 SE3 transformation matrix
    """

    # Create rotation matrix from quaternion
    rot = Rotation.from_quat([qx, qy, qz, qw])
    R = rot.as_matrix()

    # Construct SE3 matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]

    return T


def parse_egomotion_parquet(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Parse egomotion DataFrame.

    Args:
        df: Egomotion DataFrame with columns qx, qy, qz, qw, x, y, z, timestamp.

    Returns:
        Tuple of (T_rig_worlds, timestamps_us) where:
        - T_rig_worlds is [N, 4, 4] SE3 matrices
        - timestamps_us is [N] uint64 timestamps (non-negative only)
    """

    # Convert quaternion + position to SE3 matrices
    T_rig_worlds_list = []
    for _, row in df.iterrows():
        T = quaternion_to_se3(row["qx"], row["qy"], row["qz"], row["qw"], row["x"], row["y"], row["z"])
        T_rig_worlds_list.append(T)

    T_rig_worlds = np.array(T_rig_worlds_list, dtype=np.float64)

    # Handle potentially negative timestamps by shifting to make all positive
    # The offset is based on egomotion's minimum, which is the global time reference
    timestamps_raw = cast(np.ndarray, df["timestamp"].values.astype(np.int64))

    valid_timestamps = timestamps_raw >= 0

    return T_rig_worlds[valid_timestamps, :, :], timestamps_raw[valid_timestamps].astype(np.uint64)


def parse_camera_intrinsics(df: pd.DataFrame, clip_id: str, sensor_presence: pd.Series) -> Dict[str, Dict[str, Any]]:
    """Parse camera intrinsics DataFrame for a specific clip.

    The DataFrame has a model_parameters column containing a JSON-serialized
    camera model parameter pack.

    Args:
        df: Camera intrinsics DataFrame (may contain multiple clips).
        clip_id: Clip ID to extract intrinsics for.
        sensor_presence: Series of sensor presence flags.

    Returns:
        Dictionary mapping sensor names to intrinsic parameters.
        Each entry contains a "model_parameters" key with a parsed dict.
    """

    # Filter for this clip
    clip_df = df[df.index.get_level_values("clip_id") == clip_id]

    intrinsics = {}
    for sensor_name in clip_df.index.get_level_values("camera_name").unique():
        if not sensor_presence[sensor_name]:
            continue

        row = clip_df.xs(sensor_name, level="camera_name")

        intrinsics[sensor_name] = {
            "model_type": row["model_type"].iloc[0],
            "model_parameters": json.loads(row["model_parameters"].iloc[0]),
            "ego_mask_image": None,  # Placeholder for ego mask image
        }

        # Load ego mask image if available
        if len(ego_mask_image_png := row["ego_mask_image_png"].iloc[0]) > 0:
            intrinsics[sensor_name]["ego_mask_image"] = PILImage.open(io.BytesIO(ego_mask_image_png), formats=["PNG"])

    return intrinsics


def parse_sensor_extrinsics(df: pd.DataFrame, clip_id: str, sensor_presence: pd.Series) -> Dict[str, np.ndarray]:
    """Parse sensor extrinsics DataFrame for a specific clip.

    Physical AI provides sensor-to-rig transforms (T_rig_sensor), which is what
    NCore expects (despite the variable naming convention T_sensor_rig).

    Args:
        df: Sensor extrinsics DataFrame (may contain multiple clips).
        clip_id: Clip ID to extract extrinsics for.
        sensor_presence: Series of sensor presence flags.

    Returns:
        Dictionary mapping sensor names to 4x4 SE3 transformation matrices
    """

    # Filter for this clip
    clip_df = df[df.index.get_level_values("clip_id") == clip_id]

    extrinsics = {}
    for sensor_name in clip_df.index.get_level_values("sensor_name").unique():
        if not sensor_presence[sensor_name]:
            continue

        row = clip_df.xs(sensor_name, level="sensor_name")

        T = quaternion_to_se3(
            float(row["qx"].iloc[0]),
            float(row["qy"].iloc[0]),
            float(row["qz"].iloc[0]),
            float(row["qw"].iloc[0]),
            float(row["x"].iloc[0]),
            float(row["y"].iloc[0]),
            float(row["z"].iloc[0]),
        )

        extrinsics[sensor_name] = T.astype(np.float32)

    return extrinsics


def parse_vehicle_dimensions(df: pd.DataFrame, clip_id: str) -> Dict[str, float]:
    """Parse vehicle dimensions DataFrame for a specific clip.

    Args:
        df: Vehicle dimensions DataFrame (may contain multiple clips).
        clip_id: Clip ID to extract dimensions for.

    Returns:
        Dictionary with vehicle dimension parameters
    """

    # Filter for this clip
    row = df[df.index == clip_id].iloc[0]

    return {
        "length": float(row["length"]),
        "width": float(row["width"]),
        "height": float(row["height"]),
        "wheelbase": float(row["wheelbase"]),
        "track_width": float(row["track_width"]),
        "rear_axle_to_bbox_center": float(row["rear_axle_to_bbox_center"]),
    }
