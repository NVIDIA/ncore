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

"""Argoverse 2 specific utilities for the NCore V4 converter.

This module reads the Argoverse 2 Sensor Dataset directly from its on-disk
Apache Feather files using ``pyarrow`` only, deliberately avoiding the heavy
``av2`` devkit (which pulls in torch, kornia, numba, polars and PyAV). Quaternions
are converted with ``scipy.spatial.transform.Rotation`` (already an ncore
dependency), so no extra runtime dependency is introduced.

Reference (sourced from github.com/argoverse/av2-api and the AV2 User Guide):

- Lidar sweeps are *egomotion-compensated* to the sweep reference timestamp and
  stored in the **egovehicle** frame (not the individual sensor frame). The
  feather columns are ``x, y, z, intensity, laser_number, offset_ns``.
  Per-point absolute time is ``sweep_timestamp_ns + offset_ns``.
- The released imagery is **already undistorted**, so a pinhole model with zero
  distortion is exact and **global shutter is assumed** on that basis. The
  original lens radial-distortion coefficients ``(k1, k2, k3)`` are returned by
  :func:`read_intrinsics` for provenance but are never applied.
- All quaternions are scalar-first ``(qw, qx, qy, qz)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import numpy as np
import pyarrow.feather as feather

from scipy.spatial.transform import Rotation as R
from upath import UPath

from ncore.impl.common.transformations import se3_inverse
from ncore.impl.data.types import (
    IdealPinholeCameraModelParameters,
    RowOffsetStructuredSpinningLidarModelParameters,
    ShutterType,
)
from tools.data_converter.structured_lidar_model import enforce_spinning_monotonic


# --- Feather reading (no pandas) -----------------------------------------------
# We read Arrow tables directly and pull columns out as numpy arrays. This avoids
# pulling pandas into the dependency closure (pyarrow.read_feather would default to
# a pandas DataFrame).


def _read_columns(path: UPath) -> Dict[str, np.ndarray]:
    """Read a feather file into a ``column_name -> numpy array`` mapping."""
    table = feather.read_table(str(path))
    return {name: table.column(name).to_numpy(zero_copy_only=False) for name in table.column_names}


# --- Sensor ID mappings --------------------------------------------------------
# Argoverse 2 sensor names are already descriptive; we keep them verbatim as the
# NCore sensor IDs so that any downstream alignment with AV2 map / metadata stays
# unambiguous.

# All nine global-shutter cameras (7 ring + 2 stereo).
CAMERA_NAMES: List[str] = [
    "ring_front_center",
    "ring_front_left",
    "ring_front_right",
    "ring_side_left",
    "ring_side_right",
    "ring_rear_left",
    "ring_rear_right",
    "stereo_front_left",
    "stereo_front_right",
]

# The two stacked Velodyne VLP-32C units.
LIDAR_NAMES: List[str] = ["up_lidar", "down_lidar"]

# Number of beams per VLP-32C unit. laser_number spans [0, 63] across both units.
VLP32C_N_BEAMS: int = 32

# VLP-32C spins at 10 Hz (~100 ms per revolution). Both stacked units share this.
VLP32C_SCAN_DURATION_US: int = 100_000
VLP32C_SPINNING_FREQUENCY_HZ: float = 10.0
# Note: the apparent spin direction in a unit's own frame is detected per unit
# (the two stacked units fire in opposite phase), not assumed.

# AV2 ships no radar.

# --- Annotation taxonomy -------------------------------------------------------
# Argoverse 2 3D cuboid categories (the 30-class `AnnotationCategories` taxonomy)
# mapped to NCore class IDs. AV2 category strings are upper snake-case.
AV2_CATEGORY_MAP: Dict[str, str] = {
    "REGULAR_VEHICLE": "car",
    "LARGE_VEHICLE": "truck",
    "BOX_TRUCK": "truck",
    "TRUCK": "truck",
    "TRUCK_CAB": "truck",
    "VEHICULAR_TRAILER": "trailer",
    "SCHOOL_BUS": "bus",
    "ARTICULATED_BUS": "bus",
    "BUS": "bus",
    "MESSAGE_BOARD_TRAILER": "trailer",
    "RAILED_VEHICLE": "vehicle",
    "MOTORCYCLE": "motorcycle",
    "MOTORCYCLIST": "motorcyclist",
    "BICYCLE": "bicycle",
    "BICYCLIST": "bicyclist",
    "WHEELED_DEVICE": "wheeled_device",
    "WHEELED_RIDER": "wheeled_rider",
    "PEDESTRIAN": "pedestrian",
    "OFFICIAL_SIGNALER": "pedestrian",
    "STROLLER": "stroller",
    "WHEELCHAIR": "wheelchair",
    "DOG": "animal",
    "ANIMAL": "animal",
    "CONSTRUCTION_CONE": "traffic_cone",
    "CONSTRUCTION_BARREL": "barrier",
    "STOP_SIGN": "stop_sign",
    "BOLLARD": "bollard",
    "SIGN": "sign",
    "MOBILE_PEDESTRIAN_CROSSING_SIGN": "sign",
    "TRAFFIC_LIGHT_TRAILER": "trailer",
}


# --- Pose / quaternion helpers -------------------------------------------------


def se3_from_qwxyz_t(qw: float, qx: float, qy: float, qz: float, tx: float, ty: float, tz: float) -> np.ndarray:
    """Build a 4x4 SE(3) matrix from a scalar-first quaternion and translation.

    Argoverse 2 stores all rotations as ``(qw, qx, qy, qz)``.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat((qw, qx, qy, qz), scalar_first=True).as_matrix()
    T[:3, 3] = (tx, ty, tz)
    return T


# --- Dataset layout / feather readers ------------------------------------------


def list_log_ids(split_dir: UPath) -> List[str]:
    """Return the sorted log IDs (sub-directory names) under a split directory."""
    return sorted(p.name for p in split_dir.iterdir() if p.is_dir())


def read_city_se3_ego(log_dir: UPath) -> tuple[np.ndarray, np.ndarray]:
    """Read ``city_SE3_egovehicle.feather`` (at the log root).

    Returns:
        timestamps_ns: [N] uint64 sweep/pose timestamps (sorted ascending).
        T_ego_city: [N, 4, 4] float64 poses (egovehicle -> city/global frame).
    """
    cols = _read_columns(log_dir / "city_SE3_egovehicle.feather")
    order = np.argsort(cols["timestamp_ns"])
    timestamps_ns = cols["timestamp_ns"][order].astype(np.uint64)
    poses = np.stack(
        [
            se3_from_qwxyz_t(
                cols["qw"][i],
                cols["qx"][i],
                cols["qy"][i],
                cols["qz"][i],
                cols["tx_m"][i],
                cols["ty_m"][i],
                cols["tz_m"][i],
            )
            for i in order
        ]
    )
    return timestamps_ns, poses


def read_ego_se3_sensor(log_dir: UPath) -> Dict[str, np.ndarray]:
    """Read ``calibration/egovehicle_SE3_sensor.feather``.

    Returns a mapping ``sensor_name -> T_sensor_ego`` (4x4, sensor-frame point ->
    egovehicle frame).
    """
    cols = _read_columns(log_dir / "calibration" / "egovehicle_SE3_sensor.feather")
    result: Dict[str, np.ndarray] = {}
    for i, name in enumerate(cols["sensor_name"]):
        result[str(name)] = se3_from_qwxyz_t(
            cols["qw"][i],
            cols["qx"][i],
            cols["qy"][i],
            cols["qz"][i],
            cols["tx_m"][i],
            cols["ty_m"][i],
            cols["tz_m"][i],
        )
    return result


@dataclass(frozen=True)
class CameraIntrinsics:
    """An AV2 camera's ideal-pinhole model plus its original distortion provenance.

    AV2 ships already-undistorted imagery, so the converted ``model`` is a
    distortion-free ideal pinhole. The original lens radial-distortion coefficients
    ``(k1, k2, k3)`` from ``intrinsics.feather`` are kept here purely as provenance
    (they describe the *raw* lens and are never applied to the released images).
    """

    model: IdealPinholeCameraModelParameters
    original_distortion_k1k2k3: tuple[float, float, float]


def read_intrinsics(
    log_dir: UPath,
) -> Dict[str, CameraIntrinsics]:
    """Read ``calibration/intrinsics.feather`` into ideal-pinhole camera models.

    AV2 imagery is shipped already undistorted: the official av2 devkit projects
    with the intrinsic matrix K only (``PinholeCamera.project_ego_to_img``) and its
    ``Intrinsics`` dataclass does not even load the ``k1, k2, k3`` columns present
    in the file. Those coefficients describe the *original* lens (for re-distorting
    into the raw frame) and must not be applied to the released images, so an ideal
    (distortion-free) pinhole is the exact model. Because the released imagery is
    already undistorted, the cameras are modelled as global shutter.

    The raw ``k1, k2, k3`` coefficients are returned alongside the pinhole model so
    callers can preserve them as provenance metadata (they are *not* applied to the
    undistorted images).
    """
    cols = _read_columns(log_dir / "calibration" / "intrinsics.feather")
    result: Dict[str, CameraIntrinsics] = {}
    for i, name in enumerate(cols["sensor_name"]):
        model = IdealPinholeCameraModelParameters(
            resolution=np.array([int(cols["width_px"][i]), int(cols["height_px"][i])], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            external_distortion_parameters=None,
            principal_point=np.array([cols["cx_px"][i], cols["cy_px"][i]], dtype=np.float32),
            focal_length=np.array([cols["fx_px"][i], cols["fy_px"][i]], dtype=np.float32),
        )
        result[str(name)] = CameraIntrinsics(
            model=model,
            original_distortion_k1k2k3=(
                float(cols["k1"][i]),
                float(cols["k2"][i]),
                float(cols["k3"][i]),
            ),
        )
    return result


@dataclass(frozen=True)
class LidarSweep:
    """A single AV2 lidar sweep, in the egovehicle frame.

    xyz are egomotion-compensated to ``timestamp_ns``; per-point absolute time is
    ``timestamp_ns + offset_ns``.
    """

    xyz: np.ndarray  # [N, 3] float32, egovehicle frame
    intensity: np.ndarray  # [N] float32 in [0, 1]
    laser_number: np.ndarray  # [N] uint8 in [0, 63]
    offset_ns: np.ndarray  # [N] int64, offset from sweep start
    timestamp_ns: int  # sweep reference timestamp (filename)


def read_lidar_sweep(path: UPath) -> LidarSweep:
    """Read a single lidar sweep feather file (filename is the sweep timestamp)."""
    cols = _read_columns(path)
    timestamp_ns = int(UPath(path).stem)
    return LidarSweep(
        xyz=np.stack(
            [
                cols["x"].astype(np.float32),
                cols["y"].astype(np.float32),
                cols["z"].astype(np.float32),
            ],
            axis=1,
        ),
        intensity=(cols["intensity"].astype(np.float32) / 255.0),
        laser_number=cols["laser_number"].astype(np.uint8),
        offset_ns=cols["offset_ns"].astype(np.int64),
        timestamp_ns=timestamp_ns,
    )


def read_annotations(log_dir: UPath) -> Dict[str, np.ndarray]:
    """Read ``annotations.feather`` into a column -> numpy array mapping."""
    return _read_columns(log_dir / "annotations.feather")


def list_sensor_timestamps(log_dir: UPath, sensor_kind: str, sensor_name: Optional[str] = None) -> List[int]:
    """List the sorted nanosecond timestamps available for a sensor stream.
    Args:
        sensor_kind: ``"lidar"`` or ``"cameras"``.
        sensor_name: camera name (required for ``"cameras"``; ignored for lidar).
    """
    if sensor_kind == "lidar":
        sensor_dir = log_dir / "sensors" / "lidar"
        suffix = ".feather"
    elif sensor_kind == "cameras":
        assert sensor_name is not None, "sensor_name required for cameras"
        sensor_dir = log_dir / "sensors" / "cameras" / sensor_name
        suffix = ".jpg"
    else:
        raise ValueError(f"Unknown sensor_kind: {sensor_kind}")

    if not sensor_dir.exists():
        return []

    return sorted(int(p.stem) for p in sensor_dir.iterdir() if p.name.endswith(suffix))


def assign_lidar_units(
    laser_number: np.ndarray,
    xyz_ego: np.ndarray,
    T_up_ego: np.ndarray,
    T_down_ego: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Assign each point to ``up_lidar`` or ``down_lidar``.

    Argoverse 2 distributes a single aggregated sweep from two stacked Velodyne
    VLP-32C units whose 64 beams share one ``laser_number`` range ``[0, 63]``. The
    boundary that separates the two units is not documented in the AV2 devkit, so
    we recover it from the geometry of the calibrated extrinsics.

    The two units are split into the two laser-number halves (``< 32`` and
    ``>= 32``); empirically these are the two physical sensors (at any shared
    ``offset_ns`` they point ~180 deg apart in the ego frame). To decide *which*
    half is ``up_lidar`` vs ``down_lidar`` we use per-beam elevation flatness: a
    single laser ring traces a cone of (nearly) constant elevation only in its own
    sensor frame. Mapping a half into the wrong unit's extrinsic tilts that cone
    (the two units differ in pitch/roll), inflating the per-ring elevation spread.
    We pick the labelling that minimises the summed per-ring elevation spread,
    which separates the two assignments by a wide, stable margin (~2-10x).

    Returns a mapping ``unit_name -> boolean point mask``.
    """
    lo_mask = laser_number < VLP32C_N_BEAMS
    hi_mask = ~lo_mask

    # Cost of assigning lo->up_unit and hi->down_unit (assignment A) vs swapped (B).
    cost_a = _ring_elevation_spread(
        laser_number, xyz_ego, np.arange(VLP32C_N_BEAMS), T_up_ego
    ) + _ring_elevation_spread(laser_number, xyz_ego, np.arange(VLP32C_N_BEAMS, 2 * VLP32C_N_BEAMS), T_down_ego)
    cost_b = _ring_elevation_spread(
        laser_number, xyz_ego, np.arange(VLP32C_N_BEAMS), T_down_ego
    ) + _ring_elevation_spread(laser_number, xyz_ego, np.arange(VLP32C_N_BEAMS, 2 * VLP32C_N_BEAMS), T_up_ego)

    if cost_b < cost_a:
        return {"up_lidar": hi_mask, "down_lidar": lo_mask}
    return {"up_lidar": lo_mask, "down_lidar": hi_mask}


def _ring_elevation_spread(
    laser_number: np.ndarray,
    xyz_ego: np.ndarray,
    beams: np.ndarray,
    T_unit_ego: np.ndarray,
    min_valid_distance_m: float = 2.0,
    min_ring_points: int = 10,
) -> float:
    """Mean per-beam elevation standard deviation when ``beams`` are mapped to a unit.

    In the correct sensor frame each laser ring has near-constant elevation across
    azimuth, so a tight per-ring elevation distribution indicates the correct
    extrinsic. Returns the mean per-ring elevation std in degrees.

    Args:
        min_valid_distance_m: ignore near-range returns when estimating elevation.
        min_ring_points: minimum returns for a ring to contribute to the estimate.
    """
    T_ego_unit = se3_inverse(T_unit_ego)
    pts = (T_ego_unit[:3, :3] @ xyz_ego.T).T + T_ego_unit[:3, 3]
    dist = np.linalg.norm(pts, axis=1)

    spreads: List[float] = []
    for beam in beams:
        ring = (laser_number == beam) & (dist > min_valid_distance_m)
        if int(ring.sum()) < min_ring_points:
            continue
        elev = np.degrees(np.arcsin(np.clip(pts[ring, 2] / dist[ring], -1.0, 1.0)))
        spreads.append(float(np.std(elev)))

    return float(np.mean(spreads)) if spreads else float("inf")


# --- Structured VLP-32C lidar model --------------------------------------------
# AV2 provides no native firing-column index, only per-point ``laser_number``
# (0..31 within a unit) and ``offset_ns``. The two are a faithful proxy for the
# firing pattern: ``offset_ns`` quantizes into firing columns (one VLP-32C
# revolution at 10 Hz), and ``laser_number`` selects the beam (row). This lets us
# reconstruct the rows x columns structure required for a structured spinning
# lidar model and reuse the generic ``structured_lidar_model`` library.


@dataclass(frozen=True)
class Vlp32cGeometry:
    """Per-unit VLP-32C geometry recovered from a reference sweep.

    Derived empirically (and stably) from the data rather than a hard-coded spec
    table, so it self-corrects to the dataset's actual calibration.
    """

    elevations_rad: np.ndarray  # [32] float32, model row order (elevation high -> low)
    laser_to_row: np.ndarray  # [32] int, maps laser_number (0..31) -> model row index
    column_period_ns: float  # firing-column period (one beam refire interval)
    n_columns: int  # upsampled columns per revolution (native_columns * resolution_factor)
    spinning_direction: Literal["cw", "ccw"]  # apparent spin in this unit's own frame
    column_azimuths_rad: np.ndarray  # [n_columns] float32, per-(upsampled-)column azimuth
    row_azimuth_offsets_rad: np.ndarray  # [32] float32, per-row azimuth offset from the column
    resolution_factor: int  # column upsampling factor (1 = native firing resolution)


def derive_vlp32c_geometry(
    xyz_decompensated: np.ndarray,
    laser_number_in_unit: np.ndarray,
    offset_ns: np.ndarray,
    min_valid_distance_m: float = 2.0,
    far_range_m: float = 5.0,
    n_refine_iterations: int = 5,
    resolution_factor: int = 4,
) -> Vlp32cGeometry:
    """Recover the VLP-32C firing geometry from one *decompensated* sweep.

    The input must be the decompensated point cloud (each point in the sensor frame
    at its own measurement time). On the raw motion-compensated cloud the azimuths
    are smeared by ego motion (~0.5 deg), which would dominate the model error; on
    the decompensated cloud the firing geometry is clean (~0.05 deg).

    The model is ``azimuth(point) = column_azimuth[col] + row_azimuth_offset[row]``.
    The 32 beams of a firing column are *not* co-azimuthal -- VLP-32C fires them
    across a wide span -- so the per-row offset is essential (it is the dominant
    structure, several degrees) and is fit empirically rather than assumed.

    Args:
        xyz_decompensated: decompensated points in the unit's sensor frame, [N, 3].
        laser_number_in_unit: per-point beam index within the unit (0..31), [N].
        offset_ns: per-point nanosecond offset from the sweep start, [N].
        min_valid_distance_m: ignore near-range returns when estimating elevation.
        far_range_m: distance threshold for azimuth fitting (near returns are noisy).
        n_refine_iterations: alternating column/row-offset refinement passes.
    """
    dist = np.linalg.norm(xyz_decompensated, axis=1)
    valid = dist > min_valid_distance_m
    elev = np.arcsin(np.clip(xyz_decompensated[valid, 2] / dist[valid], -1.0, 1.0))
    lasers = laser_number_in_unit[valid]

    # Median elevation per laser, then sort lasers by elevation (high -> low).
    median_elev = np.full(VLP32C_N_BEAMS, np.nan, dtype=np.float64)
    for laser in range(VLP32C_N_BEAMS):
        sel = lasers == laser
        if sel.any():
            median_elev[laser] = float(np.median(elev[sel]))
    if np.isnan(median_elev).any():
        good = ~np.isnan(median_elev)
        median_elev[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), median_elev[good])

    laser_order_high_to_low = np.argsort(-median_elev)  # laser indices, highest elevation first
    elevations_rad = median_elev[laser_order_high_to_low].astype(np.float32)
    laser_to_row = np.empty(VLP32C_N_BEAMS, dtype=np.int64)
    laser_to_row[laser_order_high_to_low] = np.arange(VLP32C_N_BEAMS)
    row = laser_to_row[laser_number_in_unit]

    column_period_ns, n_columns = _estimate_column_timing(laser_number_in_unit, offset_ns)

    o = offset_ns.astype(np.int64)
    # Wrap modulo one revolution: the sweep slightly exceeds one revolution, and the
    # overlap folds back onto early columns (same physical azimuth).
    col = np.round((o - o.min()) / column_period_ns).astype(np.int64) % n_columns
    az = np.arctan2(xyz_decompensated[:, 1], xyz_decompensated[:, 0])
    far = dist > far_range_m

    # Detect spin direction from the sign of azimuth-vs-column (the two stacked units
    # fire in opposite phase, so they spin oppositely in their own frames).
    az_unwrapped = np.unwrap(az[far][np.argsort(col[far])])
    slope = float(np.polyfit(np.arange(len(az_unwrapped)), az_unwrapped, 1)[0]) if len(az_unwrapped) > 1 else -1.0
    spinning_direction: Literal["cw", "ccw"] = "cw" if slope < 0 else "ccw"

    # Jointly fit per-column azimuths and per-row azimuth offsets by alternating
    # circular medians. The 32 beams of a column span several degrees of azimuth,
    # captured by the per-row offset; averaging it into the column azimuth (offset=0)
    # would leave multi-degree per-row error.
    def circmean(a: np.ndarray) -> float:
        return float(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a))))

    # Steep downward beams (e.g. the lowest VLP-32C laser at ~-25 deg) only ever hit
    # nearby ground, so they have no far-range returns. Fit their azimuth offset from
    # whatever valid returns they do have (near-range) rather than leaving it at 0.
    valid = dist > min_valid_distance_m
    row_has_far = np.zeros(VLP32C_N_BEAMS, dtype=bool)
    for r in range(VLP32C_N_BEAMS):
        row_has_far[r] = bool(((row == r) & far).any())

    row_offsets = np.zeros(VLP32C_N_BEAMS, dtype=np.float64)
    col_az = np.zeros(n_columns, dtype=np.float64)
    for _ in range(max(n_refine_iterations, 1)):
        # column azimuth = circular median of (az - row_offset) over far points in the
        # column (far returns give the cleanest, least range-dependent azimuth).
        adj = np.angle(np.exp(1j * (az - row_offsets[row])))
        col_az[:] = np.nan
        for c in np.unique(col[far]):
            sel = (col == c) & far
            col_az[c] = np.arctan2(np.median(np.sin(adj[sel])), np.median(np.cos(adj[sel])))
        good = ~np.isnan(col_az)
        if not good.any():
            break
        col_az = np.interp(np.arange(n_columns), np.flatnonzero(good), np.unwrap(col_az[good]))
        # row offset = circular mean of (az - column azimuth). Use far returns where a
        # row has them; otherwise fall back to all valid (near-range) returns so every
        # row gets a real offset estimate.
        for r in range(VLP32C_N_BEAMS):
            sel = (row == r) & (far if row_has_far[r] else valid)
            if sel.any():
                row_offsets[r] = circmean(np.angle(np.exp(1j * (az[sel] - col_az[col[sel]]))))

    column_azimuths_rad = enforce_spinning_monotonic(col_az, n_columns, spinning_direction)
    row_azimuth_offsets_rad = np.angle(np.exp(1j * row_offsets)).astype(np.float32)

    # Upsample the column-azimuth grid. The native column step is ~0.2 deg, so the
    # per-frame integer column shift (see reconstruct_model_elements) would quantize
    # alignment to ~0.1 deg. Upsampling shrinks the step by resolution_factor,
    # removing that quantization (4x -> ~0.025 deg).
    factor = max(int(resolution_factor), 1)
    if factor > 1:
        column_azimuths_rad = _upsample_azimuths(column_azimuths_rad, factor, spinning_direction)

    return Vlp32cGeometry(
        elevations_rad=elevations_rad,
        laser_to_row=laser_to_row,
        column_period_ns=column_period_ns,
        n_columns=n_columns * factor,
        spinning_direction=spinning_direction,
        column_azimuths_rad=column_azimuths_rad,
        row_azimuth_offsets_rad=row_azimuth_offsets_rad,
        resolution_factor=factor,
    )


def _upsample_azimuths(
    column_azimuths_rad: np.ndarray, factor: int, spinning_direction: Literal["cw", "ccw"]
) -> np.ndarray:
    """Interpolate column azimuths to ``len * factor`` entries, preserving monotonicity."""
    n = len(column_azimuths_rad)
    src_idx = np.arange(n) * factor
    dst_idx = np.arange(n * factor)
    unwrapped = np.unwrap(column_azimuths_rad.astype(np.float64))
    upsampled = np.interp(dst_idx, src_idx, unwrapped)
    return enforce_spinning_monotonic(upsampled, n * factor, spinning_direction)


def build_vlp32c_model(geometry: Vlp32cGeometry) -> RowOffsetStructuredSpinningLidarModelParameters:
    """Build a structured VLP-32C model from recovered geometry.

    Uses the empirically measured per-column azimuths, per-row azimuth offsets and
    elevation table. The per-row offsets capture the (several-degree) intra-column
    firing spread and are essential for sub-degree reconstruction. The spin
    direction is the one detected for this unit (the two stacked units fire in
    opposite phase, so they spin oppositely in their own frames).
    """
    return RowOffsetStructuredSpinningLidarModelParameters(
        spinning_frequency_hz=VLP32C_SPINNING_FREQUENCY_HZ,
        spinning_direction=geometry.spinning_direction,
        n_rows=VLP32C_N_BEAMS,
        n_columns=geometry.n_columns,
        row_elevations_rad=geometry.elevations_rad,
        column_azimuths_rad=geometry.column_azimuths_rad,
        row_azimuth_offsets_rad=geometry.row_azimuth_offsets_rad,
    )


def _estimate_column_timing(laser_number_in_unit: np.ndarray, offset_ns: np.ndarray) -> tuple[float, int]:
    """Estimate the firing-column period and column count from beam-0 refire timing.

    The column count spans exactly one revolution. An AV2 sweep covers slightly
    more than one revolution (~1.02 rev over ~102 ms at 10 Hz), so columns are
    wrapped modulo this count (see :func:`reconstruct_model_elements`): the few
    degrees of overlap fold back onto the early columns, which represent the same
    physical azimuth. Sizing to one revolution keeps the column-azimuth ramp below
    2*pi, as the structured spinning-lidar model requires.
    """
    o = offset_ns.astype(np.int64)
    beam0_times = np.sort(o[laser_number_in_unit == 0])
    if len(beam0_times) >= 2:
        gaps = np.diff(beam0_times)
        # Use the median of the small, regular gaps (drop large no-return stretches).
        column_period_ns = float(np.median(gaps[gaps <= np.median(gaps) * 1.5]))
    else:
        # Fallback: one revolution divided by a nominal VLP-32C column count.
        column_period_ns = VLP32C_SCAN_DURATION_US * 1000.0 / 1800.0
    revolution_ns = VLP32C_SCAN_DURATION_US * 1000.0
    n_columns = max(int(round(revolution_ns / column_period_ns)), 2)
    return column_period_ns, n_columns


def reconstruct_model_elements(
    laser_number_in_unit: np.ndarray,
    offset_ns: np.ndarray,
    geometry: Vlp32cGeometry,
    xyz_decompensated: np.ndarray,
    min_valid_distance_m: float = 5.0,
) -> np.ndarray:
    """Build per-point ``model_element`` = (row, column) for one frame.

    Row comes from the laser->row map (elevation order). The column comes from
    quantizing ``offset_ns`` by the firing-column period (wrapped modulo one
    revolution), then applying a single per-frame column shift.

    The per-frame shift is essential: the static model fixes one mapping from
    ``offset_ns`` to azimuth, but the sensor's spin phase at a given ``offset_ns``
    drifts a degree or so between sweeps (and ``offset_ns`` is referenced to a
    per-sweep start). Without re-aligning, frames other than the one the model was
    derived from are systematically rotated by up to ~1.2 deg. We estimate the
    frame's rigid azimuth offset from the model (circular mean of the residual
    between measured and model-predicted azimuth over far returns) and convert it
    to an integer column shift, which restores sub-0.1 deg accuracy on every frame.

    Args:
        xyz_decompensated: per-point decompensated points in the sensor frame [N, 3],
            used to measure this frame's azimuth phase relative to the model.
        min_valid_distance_m: far-range threshold for the phase estimate.

    Returns a [N, 2] uint16 array (row, column).
    """
    row = geometry.laser_to_row[laser_number_in_unit]
    o = offset_ns.astype(np.int64)
    # Native firing column from the firing timing, scaled onto the (upsampled) grid.
    native_col = (o - o.min()) / geometry.column_period_ns
    col = (np.round(native_col).astype(np.int64) * geometry.resolution_factor) % geometry.n_columns

    # Re-align this frame to the model. The static model fixes one mapping from
    # offset_ns to azimuth, but between sweeps the spin phase drifts (a rigid
    # rotation) AND the spin rate varies slightly within a sweep (a drift that grows
    # with the column index). We fit the residual azimuth as an affine function of
    # the native column, residual ~= a + b * native_col, and fold it back into the
    # column index. The constant term handles the phase, the linear term the
    # intra-sweep rate drift; without the linear term some scenes retain ~0.25 deg.
    dist = np.linalg.norm(xyz_decompensated, axis=1)
    far = dist > min_valid_distance_m
    if far.any():
        az = np.arctan2(xyz_decompensated[far, 1], xyz_decompensated[far, 0])
        predicted = geometry.column_azimuths_rad[col[far]] + geometry.row_azimuth_offsets_rad[row[far]]
        residual = np.angle(np.exp(1j * (az - predicted)))
        column_step_rad = 2.0 * np.pi / geometry.n_columns
        sign = -1.0 if geometry.spinning_direction == "cw" else 1.0
        # Affine fit residual_columns ~= a + b * native_col (least squares).
        residual_cols = sign * residual / column_step_rad
        nc_far = native_col[far]
        coeffs = np.polyfit(nc_far, residual_cols, 1)
        col_shift = np.round(np.polyval(coeffs, native_col)).astype(np.int64)
        col = (col + col_shift) % geometry.n_columns

    return np.stack([row, col], axis=1).astype(np.uint16)
