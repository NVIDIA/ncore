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

"""KITTI raw data utilities.

OXTS-to-pose math is vendored from pykitti (MIT license, see NOTICE file).
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

from collections import namedtuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import numpy as np

from scipy.spatial.transform import Rotation

from ncore.impl.common.transformations import se3_inverse


# Type alias for path arguments (compatible with UPath, Path, and str)
PathLike = Union[str, "os.PathLike[str]"]


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

R_EARTH = 6378137.0  # WGS-84 Earth radius in meters

OXTS_FIELD_NAMES: list[str] = [
    "lat",
    "lon",
    "alt",
    "roll",
    "pitch",
    "yaw",
    "vn",
    "ve",
    "vf",
    "vl",
    "vu",
    "ax",
    "ay",
    "az",
    "af",
    "al",
    "au",
    "wx",
    "wy",
    "wz",
    "wf",
    "wl",
    "wu",
    "pos_accuracy",
    "vel_accuracy",
    "navstat",
    "numsats",
    "posmode",
    "velmode",
    "orimode",
]

OxtsPacket = namedtuple("OxtsPacket", OXTS_FIELD_NAMES)


# -----------------------------------------------------------------------------
# Calibration parsing
# -----------------------------------------------------------------------------


def parse_calib_cam_to_cam(filepath: PathLike) -> dict[str, np.ndarray]:
    """Parse calib_cam_to_cam.txt.

    Returns a dict with keys like ``P_rect_00`` (3x4), ``R_rect_00`` (3x3),
    ``S_rect_00`` (1x2), ``S_00`` (1x2), ``K_00`` (3x3), ``D_00`` (1x5),
    ``R_00`` (3x3), ``T_00`` (1x3) for cameras 00 through 03.
    """
    data: dict[str, np.ndarray] = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()

            # Skip non-numeric entries
            if key in ("calib_time", "corner_dist"):
                continue

            values = np.array([float(x) for x in value.split()], dtype=np.float64)

            # Reshape based on key prefix
            if key.startswith("S_rect_") or key.startswith("S_"):
                data[key] = values.reshape(1, 2)
            elif key.startswith("K_"):
                data[key] = values.reshape(3, 3)
            elif key.startswith("D_"):
                data[key] = values.reshape(1, 5)
            elif key.startswith("R_rect_"):
                data[key] = values.reshape(3, 3)
            elif key.startswith("R_"):
                data[key] = values.reshape(3, 3)
            elif key.startswith("T_"):
                data[key] = values.reshape(1, 3)
            elif key.startswith("P_rect_"):
                data[key] = values.reshape(3, 4)
            else:
                data[key] = values

    return data


def parse_calib_rigid(filepath: PathLike) -> np.ndarray:
    """Parse a rigid-body calibration file (calib_velo_to_cam.txt or calib_imu_to_velo.txt).

    Returns a 4x4 SE3 homogeneous transformation matrix (float64).
    """
    data: dict[str, np.ndarray] = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()

            if key in ("calib_time",):
                continue

            values = np.array([float(x) for x in value.split()], dtype=np.float64)
            data[key] = values

    R = data["R"].reshape(3, 3)
    T = data["T"].reshape(3, 1)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = R
    transform[:3, 3] = T.flatten()
    return transform


# -----------------------------------------------------------------------------
# OXTS parsing + pose computation (vendored from pykitti, MIT license)
# -----------------------------------------------------------------------------


def load_oxts_packets(oxts_dir: PathLike) -> tuple[list[OxtsPacket], np.ndarray]:
    """Load OXTS data from a directory of per-frame .txt files.

    Args:
        oxts_dir: Path to oxts/data/ directory containing numbered .txt files.

    Returns:
        A tuple of (list of OxtsPacket namedtuples, raw_array [N, 30] float64).
    """
    oxts_dir = Path(oxts_dir)
    files = sorted(oxts_dir.glob("*.txt"))

    packets: list[OxtsPacket] = []
    raw_rows: list[np.ndarray] = []

    for filepath in files:
        with open(filepath, "r") as f:
            values = [float(x) for x in f.readline().split()]
        packets.append(OxtsPacket(*values))
        raw_rows.append(np.array(values, dtype=np.float64))

    raw_array = np.stack(raw_rows, axis=0) if raw_rows else np.empty((0, 30), dtype=np.float64)
    return packets, raw_array


def poses_from_oxts(packets: list[OxtsPacket]) -> tuple[np.ndarray, np.ndarray]:
    """Compute ego poses from OXTS packets using Mercator projection.

    Vendored from pykitti (MIT license). See NOTICE file.

    Args:
        packets: List of OxtsPacket namedtuples.

    Returns:
        T_rig_world: [N, 4, 4] float64 array of rig-to-world poses, rebased to first frame.
        T_world_world_global: [4, 4] float64, the first pose before rebasing (global origin).
    """
    if not packets:
        return np.empty((0, 4, 4), dtype=np.float64), np.eye(4, dtype=np.float64)

    # Mercator scale factor from first packet latitude
    scale = np.cos(packets[0].lat * np.pi / 180.0)

    poses: list[np.ndarray] = []
    for packet in packets:
        # Mercator projection
        tx = scale * packet.lon * np.pi * R_EARTH / 180.0
        ty = scale * R_EARTH * np.log(np.tan((90.0 + packet.lat) * np.pi / 360.0))
        tz = packet.alt

        # Rotation from roll, pitch, yaw (ZYX intrinsic = XYZ extrinsic)
        R = Rotation.from_euler("ZYX", [packet.yaw, packet.pitch, packet.roll]).as_matrix()

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[0, 3] = tx
        T[1, 3] = ty
        T[2, 3] = tz
        poses.append(T)

    poses_array = np.stack(poses, axis=0)

    # The first pose defines the origin (world_global)
    T_world_world_global = poses_array[0].copy()

    # Rebase: make first pose the origin
    T_rig_world = se3_inverse(T_world_world_global)[None] @ poses_array

    return T_rig_world, T_world_world_global


# -----------------------------------------------------------------------------
# Velodyne
# -----------------------------------------------------------------------------


def load_velodyne_scan(filepath: PathLike) -> np.ndarray:
    """Load a Velodyne point cloud binary file.

    Returns:
        np.ndarray of shape [N, 4] float32 (x, y, z, reflectance).
    """
    points = np.fromfile(str(filepath), dtype=np.float32).reshape(-1, 4)
    return points


def compute_velodyne_timestamps_us(
    points: np.ndarray,
    start_us: int,
    end_us: int,
) -> np.ndarray:
    """Compute per-point timestamps from azimuth angle for a Velodyne HDL-64E scan.

    The HDL-64E fires starting at the back of the vehicle and rotates counter-clockwise
    (as seen from above). Using: fraction = (pi - atan2(y, x)) / (2 * pi)

    Args:
        points: [N, 4] float32 array (x, y, z, reflectance).
        start_us: Start-of-spin timestamp in microseconds.
        end_us: End-of-spin timestamp in microseconds.

    Returns:
        [N] uint64 array of per-point timestamps in microseconds.
    """
    x = points[:, 0]
    y = points[:, 1]

    # atan2(y, x) gives angle from +x axis; pi - atan2(y,x) gives angle from -x (rear)
    fraction = (np.pi - np.arctan2(y, x)) / (2.0 * np.pi)

    # Clamp to [0, 1] for safety
    fraction = np.clip(fraction, 0.0, 1.0)

    timestamps = start_us + fraction * (end_us - start_us)
    return timestamps.astype(np.uint64)


# -----------------------------------------------------------------------------
# Timestamps
# -----------------------------------------------------------------------------


def load_timestamps(filepath: PathLike) -> list[int]:
    """Load timestamps from a KITTI timestamps file.

    Each line is in the format 'YYYY-MM-DD HH:MM:SS.NNNNNNNNN'.

    Returns:
        List of timestamps in microseconds from epoch.
    """
    timestamps: list[int] = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Parse: "2011-09-26 14:14:10.924614488"
            # Python datetime only supports up to microsecond precision,
            # so we parse manually for nanosecond timestamps
            date_part, time_part = line.split(" ")
            time_main, frac = time_part.split(".")

            # Parse date and time components
            dt = datetime.strptime(f"{date_part} {time_main}", "%Y-%m-%d %H:%M:%S")
            dt = dt.replace(tzinfo=timezone.utc)

            # Convert to microseconds from epoch
            epoch_us = int(dt.timestamp() * 1_000_000)

            # Add fractional seconds (nanosecond precision -> microseconds)
            # Pad/truncate to 9 digits (nanoseconds)
            frac = frac.ljust(9, "0")[:9]
            frac_us = int(frac) // 1000  # nanoseconds to microseconds

            timestamps.append(epoch_us + frac_us)

    return timestamps


# -----------------------------------------------------------------------------
# Tracklets
# -----------------------------------------------------------------------------


@dataclass
class TrackletPose:
    """A single pose of a tracklet object at a specific frame."""

    tx: float
    ty: float
    tz: float
    rx: float
    ry: float
    rz: float
    state: int
    occlusion: int
    truncation: int


@dataclass
class Tracklet:
    """A tracklet object spanning multiple frames."""

    object_type: str
    h: float
    w: float
    l: float  # noqa: E741 -- KITTI length dimension (matches the <l> XML tag; paired with h/w)
    first_frame: int
    poses: list[TrackletPose] = field(default_factory=list)


def parse_tracklets(filepath: PathLike) -> list[Tracklet]:
    """Parse a KITTI tracklet_labels.xml file.

    Tracklet coordinates are in the Velodyne frame.

    Args:
        filepath: Path to tracklet_labels.xml.

    Returns:
        List of Tracklet objects with object_type, dimensions, first_frame, and per-frame poses.
    """
    tree = ET.parse(str(filepath))
    root = tree.getroot()

    tracklets: list[Tracklet] = []

    # Navigate to the tracklets element
    tracklets_elem = root.find("tracklets")
    if tracklets_elem is None:
        return tracklets

    for item in tracklets_elem.findall("item"):
        object_type_elem = item.find("objectType")
        h_elem = item.find("h")
        w_elem = item.find("w")
        l_elem = item.find("l")
        first_frame_elem = item.find("first_frame")

        if any(e is None for e in [object_type_elem, h_elem, w_elem, l_elem, first_frame_elem]):
            continue

        # After the None-check above, these are guaranteed to be non-None
        assert object_type_elem is not None
        assert h_elem is not None and h_elem.text is not None
        assert w_elem is not None and w_elem.text is not None
        assert l_elem is not None and l_elem.text is not None
        assert first_frame_elem is not None and first_frame_elem.text is not None
        assert object_type_elem.text is not None

        tracklet = Tracklet(
            object_type=object_type_elem.text.strip(),
            h=float(h_elem.text),
            w=float(w_elem.text),
            l=float(l_elem.text),
            first_frame=int(first_frame_elem.text),
        )

        # Parse per-frame poses
        poses_elem = item.find("poses")
        if poses_elem is not None:
            for pose_item in poses_elem.findall("item"):
                tx_elem = pose_item.find("tx")
                ty_elem = pose_item.find("ty")
                tz_elem = pose_item.find("tz")
                rx_elem = pose_item.find("rx")
                ry_elem = pose_item.find("ry")
                rz_elem = pose_item.find("rz")
                state_elem = pose_item.find("state")
                occlusion_elem = pose_item.find("occlusion")
                truncation_elem = pose_item.find("truncation")

                tracklet.poses.append(
                    TrackletPose(
                        tx=float(tx_elem.text) if tx_elem is not None and tx_elem.text else 0.0,
                        ty=float(ty_elem.text) if ty_elem is not None and ty_elem.text else 0.0,
                        tz=float(tz_elem.text) if tz_elem is not None and tz_elem.text else 0.0,
                        rx=float(rx_elem.text) if rx_elem is not None and rx_elem.text else 0.0,
                        ry=float(ry_elem.text) if ry_elem is not None and ry_elem.text else 0.0,
                        rz=float(rz_elem.text) if rz_elem is not None and rz_elem.text else 0.0,
                        state=int(state_elem.text) if state_elem is not None and state_elem.text else 0,
                        occlusion=int(occlusion_elem.text) if occlusion_elem is not None and occlusion_elem.text else 0,
                        truncation=int(truncation_elem.text)
                        if truncation_elem is not None and truncation_elem.text
                        else 0,
                    )
                )

        tracklets.append(tracklet)

    return tracklets
