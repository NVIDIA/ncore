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

"""Generic structured spinning lidar model extraction library.

This module provides composable, sensor-agnostic functions for deriving structured
(row, column) models from spinning lidar data. Each function operates on numpy
arrays and RowOffsetStructuredSpinningLidarModelParameters without knowledge of
any specific sensor.

Low-level composable steps:
    - extract_column_azimuths: per-column median azimuth from point clouds
    - compute_column_alignment: brute-force integer shift search
    - assign_model_columns: per-column fine refinement at model resolution
    - compute_frame_timestamps: linear timestamp interpolation from column position
    - enforce_spinning_monotonic: strictly-monotonic column azimuths in the spin direction
    - upsample_model: interpolate column azimuths to higher resolution
    - optimize_model: multi-frame median correction of azimuths and offsets
    - compute_model_consistency: angular error metrics
    - compute_intra_column_firing_offsets: generic intra-column timing model
    - derive_model_from_decompensated: empirical model from a decompensated frame

High-level convenience:
    - align_frame: iterative alignment + decompensation pipeline

Sensor presets (HDL-32E):
    Constants and a factory function for the Velodyne HDL-32E as a reference
    implementation. Other sensors follow the same pattern.

Resolution upsampling addresses mechanical azimuth drift: a spinning lidar does
not fire at exactly the same angles each revolution. The upsampled model (e.g.,
4x = 4340 columns for HDL-32E) allows per-frame alignment to snap to the actual
firing position rather than the nearest nominal column, reducing quantization
error from ~0.096 deg to ~0.03 deg.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from ncore.impl.common.transformations import MotionCompensator
from ncore.impl.data.types import RowOffsetStructuredSpinningLidarModelParameters


# --- Data structures -----------------------------------------------------------


@dataclass
class ColumnAlignment:
    """Result of aligning measured column azimuths to a static model."""

    spin_column_range: range  # columns used from the current frame
    static_column_range: range  # corresponding columns in the model
    mean_alignment_error_rad: float


@dataclass
class AlignedFrameData:
    """Output of per-frame alignment: decompensated points with model indices."""

    xyz_decompensated: np.ndarray  # [N, 3] points in sensor frame at measurement time
    intensity: np.ndarray  # [N] normalized intensity
    timestamps_us: np.ndarray  # [N] uint64 per-point timestamps
    model_element: np.ndarray  # [N, 2] uint16 (model_row, model_col)
    frame_start_us: int
    frame_end_us: int


# --- Internal helpers ----------------------------------------------------------


def _binned_correction(
    residual: np.ndarray, cols: np.ndarray, n_columns: int, min_obs_per_bin: int = 200
) -> np.ndarray:
    """Estimate a smooth per-column correction from residuals via adaptive binning.

    On an upsampled grid most sub-columns are observed by only a handful of
    points, so a per-sub-column median is dominated by point-assignment noise and
    reorders neighbouring columns. The underlying correction, however, varies
    smoothly with azimuth and is well-determined on a coarser grid: we group
    columns into contiguous bins each holding at least ``min_obs_per_bin``
    observations, take the median residual per bin, and linearly interpolate the
    per-bin estimate back to every column. This is a better estimate of the
    correction (it pools all the data), not a smoothing band-aid.

    Returns a dense per-column correction, shape [n_columns].
    """
    order = np.argsort(cols, kind="stable")
    sorted_cols = cols[order]
    sorted_res = residual[order]

    bin_centers: list[float] = []
    bin_values: list[float] = []
    col_idx = 0
    while col_idx < n_columns:
        lo_pt = np.searchsorted(sorted_cols, col_idx, side="left")
        hi_col = col_idx
        while hi_col < n_columns:
            hi_pt = np.searchsorted(sorted_cols, hi_col + 1, side="left")
            if hi_pt - lo_pt >= min_obs_per_bin:
                break
            hi_col += 1
        hi_pt = np.searchsorted(sorted_cols, hi_col + 1, side="left")
        if hi_pt > lo_pt:
            bin_centers.append(0.5 * (col_idx + hi_col))
            bin_values.append(float(np.median(sorted_res[lo_pt:hi_pt])))
        col_idx = hi_col + 1

    if not bin_centers:
        return np.zeros(n_columns, dtype=np.float64)
    return np.interp(np.arange(n_columns, dtype=np.float64), np.array(bin_centers), np.array(bin_values))


def _grouped_median(
    values: np.ndarray,
    groups: np.ndarray,
    n_groups: int,
    min_count: int = 3,
) -> np.ndarray:
    """Compute median of values grouped by integer index (sorted-group approach).

    Args:
        values: Values to aggregate, shape [N].
        groups: Integer group index per value, shape [N], values in [0, n_groups).
        n_groups: Total number of groups.
        min_count: Minimum number of values in a group to produce a valid median.

    Returns:
        Per-group medians, shape [n_groups]. Groups with fewer than min_count
        values receive 0.0.
    """
    result = np.zeros(n_groups, dtype=np.float64)
    sort_idx = np.argsort(groups)
    sorted_groups = groups[sort_idx]
    sorted_values = values[sort_idx]
    boundaries = np.searchsorted(sorted_groups, np.arange(n_groups + 1))
    for g in range(n_groups):
        start, end = boundaries[g], boundaries[g + 1]
        if end - start >= min_count:
            result[g] = np.median(sorted_values[start:end])
    return result


# --- Composable steps ----------------------------------------------------------


def extract_column_azimuths(
    xyz: np.ndarray,
    col_idx: np.ndarray,
    n_cols: int,
    min_range_m: float = 20.0,
    min_points_per_col: int = 3,
) -> np.ndarray:
    """Extract per-column median azimuths from far-range points.

    Filters points by minimum range to reduce motion-compensation artifacts,
    then computes the median azimuth within each column.

    Args:
        xyz: Point cloud, shape [N, 3].
        col_idx: Column index per point, shape [N], integer values in [0, n_cols).
        n_cols: Total number of columns in this frame.
        min_range_m: Minimum point range to include (filters close-range noise).
        min_points_per_col: Minimum valid points required per column.

    Returns:
        Per-column azimuths, shape [n_cols] float64. Columns with insufficient
        data contain NaN.
    """
    dist = np.linalg.norm(xyz, axis=1)
    azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])

    valid = dist > min_range_m
    valid_az = azimuth[valid]
    valid_cols = col_idx[valid]

    col_az = np.full(n_cols, np.nan, dtype=np.float64)
    if len(valid_az) > 0:
        medians = _grouped_median(valid_az, valid_cols, n_cols, min_count=min_points_per_col)
        # _grouped_median returns 0 for groups with insufficient data; mark those NaN
        sort_idx = np.argsort(valid_cols)
        sorted_cols = valid_cols[sort_idx]
        boundaries = np.searchsorted(sorted_cols, np.arange(n_cols + 1))
        for c in range(n_cols):
            if boundaries[c + 1] - boundaries[c] >= min_points_per_col:
                col_az[c] = medians[c]

    return col_az


def compute_column_alignment(
    spin_azimuths_rad: np.ndarray,
    model_column_azimuths_rad: np.ndarray,
    max_column_shift: int = 20,
) -> ColumnAlignment:
    """Align measured column azimuths to a static model via brute-force shift search.

    Tries integer column shifts in [-max_column_shift, +max_column_shift] and
    selects the shift that minimizes mean angular error. Uses cosine-distance
    (arccos(cos(delta))) for wrap-around safety.

    Args:
        spin_azimuths_rad: Per-column measured azimuths for the current frame,
            shape [n_spin_cols].
        model_column_azimuths_rad: Static model column azimuths,
            shape [n_model_cols].
        max_column_shift: Maximum shift in columns to search in each direction.

    Returns:
        ColumnAlignment with optimal spin/static column ranges and mean error.
    """
    n_spin = len(spin_azimuths_rad)
    n_model = len(model_column_azimuths_rad)

    best_spin_range = range(0, 0)
    best_static_range = range(0, 0)
    best_mean_error: Optional[float] = None

    for shift in range(-max_column_shift, max_column_shift + 1):
        spin_start = max(-shift, 0)
        static_start = max(shift, 0)
        spin_stop = min(n_spin, n_model - static_start + spin_start)
        static_stop = static_start + (spin_stop - spin_start)

        if spin_stop <= spin_start:
            continue

        abs_error_rad = np.arccos(
            np.clip(
                np.cos(spin_azimuths_rad[spin_start:spin_stop] - model_column_azimuths_rad[static_start:static_stop]),
                -1.0,
                1.0,
            )
        )
        mean_err = float(abs_error_rad.mean())

        if best_mean_error is None or mean_err < best_mean_error:
            best_mean_error = mean_err
            best_spin_range = range(spin_start, spin_stop)
            best_static_range = range(static_start, static_stop)

    assert best_mean_error is not None, "No valid shift found"

    return ColumnAlignment(
        spin_column_range=best_spin_range,
        static_column_range=best_static_range,
        mean_alignment_error_rad=best_mean_error,
    )


def assign_model_columns(
    spin_col_azimuths: np.ndarray,
    model_params: RowOffsetStructuredSpinningLidarModelParameters,
    alignment: ColumnAlignment,
    resolution_factor: int = 1,
) -> np.ndarray:
    """Assign per-physical-column model column indices.

    When resolution_factor=1, this is a simple 1:1 mapping from the alignment.
    When resolution_factor>1, each physical column is refined to the nearest
    model column within a +/-resolution_factor window around the coarse position.

    Args:
        spin_col_azimuths: Per-column azimuths from the current frame,
            shape [n_spin_cols] float64 (f64 needed for sub-column arccos precision).
        model_params: Model parameters (must have n_columns = native * resolution_factor).
        alignment: Column alignment result from compute_column_alignment.
        resolution_factor: Ratio of model columns to native physical columns.

    Returns:
        Per-physical-column model column indices, shape [n_spin_cols] int64.
        Columns outside the overlap range get the nearest boundary value.
    """
    n_spin_cols = len(spin_col_azimuths)
    n_model_cols = model_params.n_columns
    model_az = model_params.column_azimuths_rad.astype(np.float64)

    spin_range = alignment.spin_column_range
    static_start_native = alignment.static_column_range.start

    model_col_per_phys = np.zeros(n_spin_cols, dtype=np.int64)

    if resolution_factor <= 1:
        # Simple 1:1 mapping
        for c in range(n_spin_cols):
            if spin_range.start <= c < spin_range.stop:
                model_col_per_phys[c] = static_start_native + (c - spin_range.start)
            elif c < spin_range.start:
                model_col_per_phys[c] = static_start_native
            else:
                model_col_per_phys[c] = min(
                    static_start_native + (spin_range.stop - 1 - spin_range.start),
                    n_model_cols - 1,
                )
        np.clip(model_col_per_phys, 0, n_model_cols - 1, out=model_col_per_phys)
    else:
        # Per-column fine refinement within +/-resolution_factor window
        for c in range(n_spin_cols):
            if spin_range.start <= c < spin_range.stop:
                coarse = min(
                    (static_start_native + (c - spin_range.start)) * resolution_factor,
                    n_model_cols - 1,
                )
                lo = max(0, coarse - resolution_factor)
                hi = min(n_model_cols, coarse + resolution_factor + 1)
                if not np.isnan(spin_col_azimuths[c]):
                    dists = np.arccos(np.clip(np.cos(model_az[lo:hi] - spin_col_azimuths[c]), -1.0, 1.0))
                    model_col_per_phys[c] = lo + int(np.argmin(dists))
                else:
                    model_col_per_phys[c] = coarse
            elif c < spin_range.start:
                model_col_per_phys[c] = 0
            else:
                model_col_per_phys[c] = n_model_cols - 1

    return model_col_per_phys


def compute_frame_timestamps(
    model_col: np.ndarray,
    n_model_cols: int,
    frame_start_us: int,
    frame_end_us: int,
) -> np.ndarray:
    """Compute per-point timestamps from model column indices.

    Each column corresponds to a fixed fraction of the full rotation. Column k
    fires at: frame_start_us + (k / n_model_cols) * (frame_end_us - frame_start_us).

    Args:
        model_col: Model column index per point, shape [N], integer array.
        n_model_cols: Total number of model columns.
        frame_start_us: Frame start timestamp in microseconds.
        frame_end_us: Frame end timestamp in microseconds.

    Returns:
        Per-point timestamps, shape [N], dtype uint64.
    """
    # Fencepost: col/N (not col/(N-1)) because the next frame starts at frame_end_us.
    # Column 0 fires at frame_start, column N-1 fires at frame_start + (N-1)/N * duration.
    fraction = model_col.astype(np.float64) / n_model_cols
    duration_us = frame_end_us - frame_start_us
    timestamps = frame_start_us + fraction * duration_us
    return timestamps.astype(np.uint64)


def enforce_spinning_monotonic(
    azimuths_rad: np.ndarray, n_columns: int, spinning_direction: Literal["cw", "ccw"]
) -> np.ndarray:
    """Project a near-monotonic angle estimate onto the strictly-monotonic set.

    The ncore lidar model (RowOffsetStructuredSpinningLidarModelParameters)
    requires the relative angle between consecutive columns to be strictly
    positive in the spinning direction (see types.py __post_init__). After a
    global shift that places element 0 at the extremum, that means the angles
    must be strictly monotonic (decreasing for "cw", increasing for "ccw").

    The per-column reference azimuth is an estimate of a quantity that is
    monotonic in column index (the sensor sweeps continuously), but it is
    estimated from noisy data: a firing column's beams span a few degrees of
    azimuth (the sensor rotates during the column's firing sequence) and are
    partially occluded, so the per-column circular median fluctuates by a
    fraction of a column width and a handful of adjacent columns can come out
    slightly out of order. There is no cleaner upstream input to use -- this is a
    monotonic regression of an intrinsically noisy estimate, not a workaround for
    a bug. We perform the minimal monotonic adjustment: each out-of-order element
    is nudged just past its predecessor.

    Wholesale disorder or a span reaching a full revolution would instead
    indicate a malformed input (e.g. an upstream estimation bug) and is rejected,
    not silently reshaped.

    The nudge step is 1% of one nominal column width. That is large enough to
    survive the float32 cast (~8x the float32 ULP near +/-pi for a 1085-column
    model) yet tiny enough that, for the handful of real violations, the total
    span stays well below 2*pi.

    Args:
        azimuths_rad: Angles in radians (shape [N], float). Used for both column
            azimuths and row elevations -- any quantity the model requires to be
            strictly monotonic.
        n_columns: Nominal number of columns (sizes the nudge step).
        spinning_direction: "cw" (strictly decreasing) or "ccw" (strictly
            increasing).

    Returns:
        Strictly-monotonic (in the spin direction) angles as a float32 array
        spanning strictly less than 2*pi.

    Raises:
        ValueError: if spinning_direction is invalid, or if the repaired span
            reaches a full revolution (indicating malformed input).
    """
    if spinning_direction not in ("cw", "ccw"):
        raise ValueError(f"Invalid spinning direction: {spinning_direction}")

    # Solve everything as the CW (strictly-decreasing) problem. CCW is the exact
    # mirror image, so negate on the way in and on the way out: negation maps a
    # strictly-increasing (CCW) sequence to a strictly-decreasing (CW) one and
    # preserves spans.
    sign = 1.0 if spinning_direction == "cw" else -1.0

    # Unwrap into a single continuous ramp anchored at element 0. np.unwrap
    # removes the +/-pi branch cuts so the sequence is a continuous function of
    # column index, with element 0 left exactly as given. We deliberately do NOT
    # re-wrap to (-pi, pi] or globally shift relative to element 0: when element 0
    # sits on the +/-pi boundary that rewrap flips its branch and the shift then
    # rotates almost the whole array by 2*pi (a benign but alarming 360-degree
    # remap). The model only requires a monotonic ramp spanning < 2*pi, which the
    # unwrapped sequence already provides.
    az = np.unwrap(sign * azimuths_rad.astype(np.float64))

    n = len(az)
    if n < 2:
        return (sign * az).astype(np.float32)

    # Nudge out-of-order columns just past their predecessor. 1% of one column
    # width is sub-0.003 deg for a 1085-column model and >> the float32 ULP, so
    # the strict ordering survives the float32 cast.
    min_step = 2.0 * np.pi / n_columns / 100.0
    for i in range(1, n):
        if az[i] > az[i - 1] - min_step:
            az[i] = az[i - 1] - min_step

    # The model needs a sweep strictly within one revolution so columns do not
    # alias modulo 2*pi. A real frame can sweep marginally past 360 deg (the scan
    # overlaps slightly at the seam) or the min-step nudges above can add a sliver
    # of span; in that case compress the ramp proportionally about element 0 to
    # sit just under 2*pi -- an imperceptible, distortion-minimal adjustment
    # (e.g. ~0.02% for a 0.001-rad overflow). Only a gross overflow (well beyond
    # one revolution) indicates a malformed estimate and is rejected.
    span = az[0] - az[-1]
    max_span = 2.0 * np.pi * (1.0 - 1.0 / n_columns)  # leave one column-width of headroom
    if span > 2.0 * np.pi * 1.05:
        raise ValueError(
            f"Column azimuths span {span:.6f} rad exceeds one revolution by >5%; "
            "the input azimuth estimate is malformed (expected a single sub-revolution sweep)."
        )
    if span > max_span:
        az = az[0] + (az - az[0]) * (max_span / span)

    # Undo the CCW mirror.
    return (sign * az).astype(np.float32)


def upsample_model(
    model_params: RowOffsetStructuredSpinningLidarModelParameters,
    resolution_factor: int,
) -> RowOffsetStructuredSpinningLidarModelParameters:
    """Upsample a lidar model's column resolution by interpolation.

    Interpolates column_azimuths_rad to N * resolution_factor entries, providing
    finer-grained alignment precision without changing the model's physical
    meaning. Row elevations and offsets are preserved unchanged.

    Args:
        model_params: Model with N columns.
        resolution_factor: Upsampling factor (1 = no-op, 4 = 4x resolution).

    Returns:
        Model with N * resolution_factor columns and interpolated azimuths.
        If resolution_factor <= 1, returns model_params unchanged.
    """
    if resolution_factor <= 1:
        return model_params

    n_native = model_params.n_columns
    n_upsampled = n_native * resolution_factor

    native_az = model_params.column_azimuths_rad.astype(np.float64)
    native_unwrapped = np.unwrap(native_az)

    upsampled_unwrapped = np.interp(
        np.linspace(0, n_native - 1, n_upsampled),
        np.arange(n_native),
        native_unwrapped,
    )

    # Preserve strict monotonicity in the spin direction after interpolation.
    upsampled_az = enforce_spinning_monotonic(upsampled_unwrapped, n_upsampled, model_params.spinning_direction)

    return RowOffsetStructuredSpinningLidarModelParameters(
        spinning_frequency_hz=model_params.spinning_frequency_hz,
        spinning_direction=model_params.spinning_direction,
        n_rows=model_params.n_rows,
        n_columns=n_upsampled,
        row_elevations_rad=model_params.row_elevations_rad,
        column_azimuths_rad=upsampled_az,
        row_azimuth_offsets_rad=model_params.row_azimuth_offsets_rad,
    )


def optimize_model(
    model_params: RowOffsetStructuredSpinningLidarModelParameters,
    frame_azimuths: list[np.ndarray],
    frame_model_cols: list[np.ndarray],
    frame_model_rows: list[np.ndarray],
    frame_distances: list[np.ndarray],
    min_range_m: float = 10.0,
    n_iterations: int = 1,
) -> RowOffsetStructuredSpinningLidarModelParameters:
    """Optimize model parameters from multi-frame observations.

    Alternates per-column and per-row median corrections to minimize angular
    residuals across all frames. Only far-range points (above min_range_m) are
    used, since close-range points have larger motion-compensation artifacts.

    Each iteration:
        1. Compute residuals: actual_az - (column_az[col] + row_offset[row])
        2. Correct column_azimuths by per-column median residual
        3. Correct row_offsets by per-row median residual
    After all iterations, monotonicity is enforced on column azimuths.

    Args:
        model_params: Initial model to refine.
        frame_azimuths: Decompensated azimuth per point for each frame.
        frame_model_cols: Model column index per point for each frame.
        frame_model_rows: Model row index per point for each frame.
        frame_distances: Distance per point for each frame.
        min_range_m: Minimum distance for points to include in optimization.
        n_iterations: Number of alternating optimization iterations.

    Returns:
        Optimized model parameters with adjusted column azimuths and row offsets.
    """
    n_columns = model_params.n_columns
    n_rows = model_params.n_rows

    column_azimuths = model_params.column_azimuths_rad.astype(np.float64)
    row_offsets = model_params.row_azimuth_offsets_rad.astype(np.float64)

    # Concatenate all frames, filtering by distance
    all_azimuths: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    all_rows: list[np.ndarray] = []

    for az, cols, rows, dists in zip(frame_azimuths, frame_model_cols, frame_model_rows, frame_distances):
        far_mask = dists > min_range_m
        all_azimuths.append(az[far_mask].astype(np.float64))
        all_cols.append(cols[far_mask].astype(np.int64))
        all_rows.append(rows[far_mask].astype(np.int64))

    if not all_azimuths:
        return model_params

    cat_azimuths = np.concatenate(all_azimuths)
    cat_cols = np.concatenate(all_cols)
    cat_rows = np.concatenate(all_rows)

    if len(cat_azimuths) == 0:
        return model_params

    for _ in range(n_iterations):
        # Per-column correction.
        #
        # Split into a global component (applied to every column) and a local,
        # azimuth-varying deviation. The global offset (circular mean of all
        # residuals) absorbs a systematic model/data phase offset -- without it
        # only observed columns would shift, tearing the ramp apart. The local
        # deviation is estimated by adaptive binning rather than per-(sub-)column:
        # on a 4x-upsampled grid a sub-column is hit by only a few points, so its
        # raw median is assignment noise that reorders neighbours, whereas the
        # true correction varies smoothly with azimuth and is well-determined over
        # a bin of a few hundred points. Binning + interpolation pools all frames
        # for an accurate, smoothly-varying correction.
        predicted = column_azimuths[cat_cols] + row_offsets[cat_rows]
        residual = np.arctan2(np.sin(cat_azimuths - predicted), np.cos(cat_azimuths - predicted))

        global_correction = float(np.arctan2(np.sin(residual).mean(), np.cos(residual).mean()))
        local_residual = np.arctan2(np.sin(residual - global_correction), np.cos(residual - global_correction))
        local_correction = _binned_correction(local_residual, cat_cols, n_columns)

        column_azimuths += global_correction + local_correction

        # Per-row correction
        predicted = column_azimuths[cat_cols] + row_offsets[cat_rows]
        residual = np.arctan2(np.sin(cat_azimuths - predicted), np.cos(cat_azimuths - predicted))
        row_correction = _grouped_median(residual, cat_rows, n_rows, min_count=3)
        row_offsets += row_correction

    # The binned correction is smooth, so the corrected azimuths are monotonic
    # wherever the upsampled base ramp has room. On a 4x-upsampled grid a few
    # adjacent sub-columns can sit closer than the correction's local slope and
    # come out marginally out of order; this is sub-resolution ambiguity below the
    # model's accuracy, so we resolve it with the minimal monotonic projection
    # (measured: it touches <=66/4340 columns by <=0.4 deg, and never fires at
    # native resolution).
    column_azimuths_rad = enforce_spinning_monotonic(column_azimuths, n_columns, model_params.spinning_direction)

    return RowOffsetStructuredSpinningLidarModelParameters(
        spinning_frequency_hz=model_params.spinning_frequency_hz,
        spinning_direction=model_params.spinning_direction,
        n_rows=n_rows,
        n_columns=n_columns,
        row_elevations_rad=model_params.row_elevations_rad,
        column_azimuths_rad=column_azimuths_rad,
        row_azimuth_offsets_rad=row_offsets.astype(np.float32),
    )


def compute_model_consistency(
    directions: np.ndarray,
    model_element: np.ndarray,
    distances: np.ndarray,
    model_params: RowOffsetStructuredSpinningLidarModelParameters,
    far_range_m: float = 20.0,
) -> tuple[float, float, float]:
    """Compute model consistency metrics between stored directions and model predictions.

    Compares measured direction unit vectors against model-predicted directions
    (reconstructed from model_element indices and model parameters) to quantify
    alignment quality.

    Args:
        directions: Measured direction unit vectors, shape [N, 3] float32.
        model_element: Model element indices, shape [N, 2] uint16 as (row, col).
        distances: Per-point distances, shape [N] float32.
        model_params: Lidar model parameters.
        far_range_m: Distance threshold for the far-range metric subset.

    Returns:
        Tuple of (mean_err_all_deg, mean_err_far_deg, mean_az_shift_deg):
            - mean_err_all_deg: Mean angular error across all valid points (degrees).
            - mean_err_far_deg: Mean angular error for far-range points (degrees).
            - mean_az_shift_deg: Mean systematic azimuth shift for far-range (degrees).
    """
    model_row = model_element[:, 0].astype(np.int64)
    model_col = model_element[:, 1].astype(np.int64)

    # Reconstruct predicted directions from model parameters
    model_az = model_params.column_azimuths_rad[model_col].astype(np.float64) + model_params.row_azimuth_offsets_rad[
        model_row
    ].astype(np.float64)
    model_el = model_params.row_elevations_rad[model_row].astype(np.float64)

    cos_el = np.cos(model_el)
    model_dir = np.stack(
        [cos_el * np.cos(model_az), cos_el * np.sin(model_az), np.sin(model_el)],
        axis=1,
    ).astype(np.float32)

    # Filter to valid returns (distance > 0)
    valid_mask = distances > 0
    if not valid_mask.any():
        return (0.0, 0.0, 0.0)

    cos_angle = np.clip(np.sum(directions[valid_mask] * model_dir[valid_mask], axis=1), -1.0, 1.0)
    ang_err_deg = np.degrees(np.arccos(cos_angle))
    mean_err_all_deg = float(ang_err_deg.mean())

    # Far-range subset
    far_mask = distances[valid_mask] > far_range_m
    if far_mask.any():
        mean_err_far_deg = float(ang_err_deg[far_mask].mean())

        # Systematic azimuth shift (signed difference)
        actual_az = np.arctan2(
            directions[valid_mask][far_mask, 1],
            directions[valid_mask][far_mask, 0],
        )
        model_az_far = model_az[valid_mask][far_mask]
        az_shift = np.arctan2(np.sin(actual_az - model_az_far), np.cos(actual_az - model_az_far))
        mean_az_shift_deg = float(np.degrees(az_shift.mean()))
    else:
        mean_err_far_deg = mean_err_all_deg
        mean_az_shift_deg = 0.0

    return (mean_err_all_deg, mean_err_far_deg, mean_az_shift_deg)


# --- Convenience wrapper -------------------------------------------------------


def align_frame(
    xyz_mc: np.ndarray,
    ring_index: np.ndarray,
    intensity: np.ndarray,
    n_beams_per_column: int,
    model_params: RowOffsetStructuredSpinningLidarModelParameters,
    motion_compensator: MotionCompensator,
    sensor_id: str,
    frame_start_us: int,
    frame_end_us: int,
    *,
    timestamps_us: Optional[np.ndarray] = None,
    model_resolution_factor: int = 1,
    n_iterations: int = 2,
    min_valid_distance_m: float = 0.5,
) -> Optional[AlignedFrameData]:
    """Align a single frame to the model and produce decompensated output.

    Iteratively aligns physical firing columns to model columns, computes
    per-point timestamps, and decompensates the motion-compensated point cloud.
    Each iteration refines alignment using the decompensated azimuths from the
    previous iteration.

    When timestamps_us is provided, the first iteration uses those timestamps
    directly for decompensation (bypassing the column-based timestamp estimate).
    Subsequent iterations use column-based timestamps since the alignment may
    have shifted.

    Args:
        xyz_mc: Motion-compensated point cloud, shape [N, 3] float32.
        ring_index: Beam/ring ID per point, shape [N], values in [0, n_beams-1].
        intensity: Per-point intensity, shape [N].
        n_beams_per_column: Number of beams per firing column.
        model_params: Static lidar model parameters (possibly upsampled).
        motion_compensator: MotionCompensator instance with loaded poses.
        sensor_id: Sensor identifier for pose lookup.
        frame_start_us: Frame start timestamp in microseconds.
        frame_end_us: Frame end timestamp in microseconds.
        timestamps_us: Optional caller-supplied per-point timestamps, shape [N]
            uint64. Used for the first decompensation iteration when provided.
        model_resolution_factor: Ratio of model columns to native physical columns.
        n_iterations: Number of align-decompensate iterations (>=1).
        min_valid_distance_m: Minimum distance to consider a point valid.

    Returns:
        AlignedFrameData with filtered, aligned, and decompensated points, or
        None if alignment fails (insufficient valid columns).
    """
    n_points = len(xyz_mc)
    n_cols = n_points // n_beams_per_column
    n_model_cols = model_params.n_columns
    col_idx = np.arange(n_points, dtype=np.int64) // n_beams_per_column
    res_ratio = model_resolution_factor

    # Native-resolution model azimuths for coarse alignment
    if res_ratio > 1:
        model_az_native = model_params.column_azimuths_rad[::res_ratio]
    else:
        model_az_native = model_params.column_azimuths_rad

    xyz_decomp_full: Optional[np.ndarray] = None
    alignment: Optional[ColumnAlignment] = None
    spin_range: Optional[range] = None
    model_col_full: Optional[np.ndarray] = None

    for iteration in range(n_iterations):
        # Step 1: Extract column azimuths for alignment
        if iteration == 0:
            spin_col_azimuths = extract_column_azimuths(xyz_mc, col_idx, n_cols, min_range_m=20.0)
            valid_az_mask = ~np.isnan(spin_col_azimuths)
            if valid_az_mask.sum() < n_cols * 0.3:
                # Fallback to shorter range if insufficient far-range data
                spin_col_azimuths = extract_column_azimuths(xyz_mc, col_idx, n_cols, min_range_m=5.0)
                valid_az_mask = ~np.isnan(spin_col_azimuths)
        else:
            assert xyz_decomp_full is not None
            assert spin_range is not None
            spin_col_azimuths = extract_column_azimuths(xyz_decomp_full, col_idx, n_cols, min_range_m=0.5)
            # Invalidate columns outside previous overlap (unreliable timestamps)
            spin_col_azimuths[: spin_range.start] = np.nan
            spin_col_azimuths[spin_range.stop :] = np.nan
            valid_az_mask = ~np.isnan(spin_col_azimuths)

        # Fill NaN gaps by interpolation on unwrapped azimuths
        if valid_az_mask.any() and not valid_az_mask.all():
            valid_indices = np.where(valid_az_mask)[0]
            valid_values = spin_col_azimuths[valid_az_mask]
            valid_unwrapped = np.unwrap(valid_values)
            all_unwrapped = np.interp(np.arange(n_cols), valid_indices, valid_unwrapped)
            spin_col_azimuths = ((all_unwrapped + np.pi) % (2 * np.pi) - np.pi).astype(np.float64)
        elif not valid_az_mask.any():
            return None

        # Step 2: Compute alignment
        alignment = compute_column_alignment(
            spin_azimuths_rad=spin_col_azimuths,
            model_column_azimuths_rad=model_az_native,
            max_column_shift=20,
        )
        spin_range = alignment.spin_column_range
        static_start_native = alignment.static_column_range.start

        # Step 3: Map physical columns to model columns
        is_final_iteration = iteration == n_iterations - 1
        if res_ratio > 1 and is_final_iteration:
            # Fine-grained assignment on final iteration
            model_col_per_phys = assign_model_columns(
                spin_col_azimuths, model_params, alignment, resolution_factor=res_ratio
            )
            model_col_full = model_col_per_phys[col_idx]
        else:
            # Coarse mapping
            model_col_full = ((static_start_native + (col_idx - spin_range.start)) * res_ratio).astype(np.int64)
            model_col_full[col_idx < spin_range.start] = 0
            model_col_full[col_idx >= spin_range.stop] = n_model_cols - 1
            np.clip(model_col_full, 0, n_model_cols - 1, out=model_col_full)

        # Step 4: Compute timestamps
        if timestamps_us is not None and iteration == 0:
            # Use caller-supplied timestamps on first iteration
            point_timestamps = timestamps_us
        else:
            point_timestamps = compute_frame_timestamps(model_col_full, n_model_cols, frame_start_us, frame_end_us)

        # Step 5: Decompensate
        xyz_decomp_full = motion_compensator.motion_decompensate_points(
            sensor_id=sensor_id,
            xyz_reftime=xyz_mc,
            timestamp_us=point_timestamps,
            frame_start_timestamp_us=frame_start_us,
            frame_end_timestamp_us=frame_end_us,
        )

    # --- Final output: filter to overlap region + valid distance ---
    assert alignment is not None
    assert spin_range is not None
    assert model_col_full is not None

    in_overlap = (col_idx >= spin_range.start) & (col_idx < spin_range.stop)
    distance_mc = np.linalg.norm(xyz_mc, axis=1)
    valid_mask = in_overlap & (distance_mc > min_valid_distance_m)

    # Assemble model element indices
    ring_filtered = ring_index[valid_mask]
    model_row = (n_beams_per_column - 1 - ring_filtered).astype(np.uint16)
    model_col = model_col_full[valid_mask].astype(np.uint16)
    model_element = np.stack([model_row, model_col], axis=1)

    # Final timestamps and decompensation for filtered points
    final_timestamps = compute_frame_timestamps(model_col.astype(np.int64), n_model_cols, frame_start_us, frame_end_us)
    xyz_decompensated = motion_compensator.motion_decompensate_points(
        sensor_id=sensor_id,
        xyz_reftime=xyz_mc[valid_mask],
        timestamp_us=final_timestamps,
        frame_start_timestamp_us=frame_start_us,
        frame_end_timestamp_us=frame_end_us,
    )

    return AlignedFrameData(
        xyz_decompensated=xyz_decompensated,
        intensity=intensity[valid_mask],
        timestamps_us=final_timestamps,
        model_element=model_element,
        frame_start_us=frame_start_us,
        frame_end_us=frame_end_us,
    )


# --- Generic utility -----------------------------------------------------------


def compute_intra_column_firing_offsets(
    n_beams: int,
    beam_pair_interval_us: float,
    scan_duration_us: int,
    spinning_direction: Literal["cw", "ccw"] = "cw",
) -> np.ndarray:
    """Compute per-beam azimuth offsets from intra-column firing timing.

    Spinning lidars fire beams within each column sequentially. Two banks of
    n_beams//2 alternate, so beam k in a bank fires at:
        time_offset = beam_in_bank * beam_pair_interval_us * 2

    The resulting angular offset depends on the spinning rate and direction.
    Offsets are returned in model row order (reversed from ring order) and
    mean-subtracted so the model's column azimuth represents the column center.

    Args:
        n_beams: Total number of beams.
        beam_pair_interval_us: Time between consecutive firing pairs (one from
            each bank), e.g., 1.152 us for HDL-32E.
        scan_duration_us: Full rotation duration in microseconds.
        spinning_direction: "cw" (decreasing azimuth with time) or "ccw".

    Returns:
        Per-beam azimuth offsets, shape [n_beams] float32, in model row order,
        mean-subtracted.
    """
    angular_rate_rad_per_us = 2.0 * np.pi / scan_duration_us
    sign = -1.0 if spinning_direction == "cw" else 1.0

    beams_per_bank = n_beams // 2
    offsets_ring_order = np.zeros(n_beams, dtype=np.float64)
    for ring in range(n_beams):
        beam_in_bank = ring % beams_per_bank
        time_offset_us = beam_in_bank * beam_pair_interval_us * 2
        offsets_ring_order[ring] = sign * time_offset_us * angular_rate_rad_per_us

    # Convert to model row order (row 0 = highest elevation = ring n_beams-1)
    offsets_model_order = offsets_ring_order[::-1].copy()

    # Mean-subtract so column azimuth represents the column center
    offsets_model_order -= offsets_model_order.mean()

    return offsets_model_order.astype(np.float32)


def derive_model_from_decompensated(
    xyz_decompensated: np.ndarray,
    n_beams_per_column: int,
    n_target_cols: int,
    spinning_direction: Literal["cw", "ccw"],
    spinning_frequency_hz: float,
    beam_pair_interval_us: float = 0.0,
    min_valid_distance_m: float = 0.5,
    far_range_m: float = 20.0,
    min_obs_for_full_empirical: int = 0,
) -> Optional[RowOffsetStructuredSpinningLidarModelParameters]:
    """Derive a structured lidar model empirically from a decompensated point cloud.

    Extracts model parameters from a single decompensated frame:
    - column_azimuths: per-column circular median azimuth across all valid rows
    - row_azimuth_offsets: measured per-row offsets, blended with analytical
      firing offsets for rows with insufficient far-range observations
    - row_elevations: median elevation per row across all valid columns

    Points below min_valid_distance_m are treated as "no return" sentinel values
    and excluded from parameter estimation.

    Args:
        xyz_decompensated: Decompensated point cloud [N, 3] in sensor frame.
        n_beams_per_column: Number of beams per firing column.
        n_target_cols: Expected column count. Returns None if frame differs.
        spinning_direction: "cw" or "ccw".
        spinning_frequency_hz: Spinning frequency in Hz.
        beam_pair_interval_us: Interval between consecutive firing pairs (for
            analytical offset blending). Set to 0 to skip blending.
        min_valid_distance_m: Minimum distance for a valid return.
        far_range_m: Distance threshold for "good" azimuth observations used
            in blending weight computation.
        min_obs_for_full_empirical: Number of far-range observations per row
            needed for full empirical weight (0 = auto: n_cols // 2).

    Returns:
        Model parameters, or None if frame doesn't match target column count.
    """
    n_points = len(xyz_decompensated)
    n_cols = n_points // n_beams_per_column
    if n_cols != n_target_cols:
        return None

    # Reshape into [n_cols, n_beams_per_column, 3] grid
    xyz_grid = xyz_decompensated.reshape(n_cols, n_beams_per_column, 3)

    # Compute per-cell angles and distances
    dist_grid = np.linalg.norm(xyz_grid, axis=2)
    az_grid = np.arctan2(xyz_grid[:, :, 1], xyz_grid[:, :, 0])
    xy_range_grid = np.sqrt(xyz_grid[:, :, 0] ** 2 + xyz_grid[:, :, 1] ** 2)
    el_grid = np.arctan2(xyz_grid[:, :, 2], xy_range_grid)

    # Valid mask: exclude "no return" sentinel points
    valid_grid = dist_grid > min_valid_distance_m

    # Require enough columns with at least one valid return to estimate azimuths.
    col_has_return = valid_grid.any(axis=1)
    if col_has_return.sum() < n_cols * 0.9:
        return None

    # Per-column azimuth as the circular median over all valid rows in the column.
    # Every beam in a column fires at nearly the same azimuth (they differ only by
    # the small per-row firing offset), so aggregating across rows averages out
    # per-beam measurement noise and rejects gross outliers (spurious returns).
    # A single reference row, by contrast, is fragile: a few bad returns in that
    # row produce large azimuth jumps that reorder columns. The median across rows
    # yields a clean, already-monotonic per-column azimuth estimate.
    col_az = np.full(n_cols, np.nan, dtype=np.float64)
    for c in range(n_cols):
        rows_valid = valid_grid[c, :]
        if rows_valid.any():
            a = az_grid[c, rows_valid]
            col_az[c] = np.arctan2(np.median(np.sin(a)), np.median(np.cos(a)))

    # Interpolate any columns that had no valid return at all.
    col_has_az = ~np.isnan(col_az)
    if not col_has_az.all():
        valid_indices = np.where(col_has_az)[0]
        valid_az_unwrapped = np.unwrap(col_az[col_has_az])
        col_az = np.interp(np.arange(n_cols, dtype=np.float64), valid_indices, valid_az_unwrapped)

    # Enforce strict monotonicity in the spin direction. The raw per-column
    # estimate is not perfectly uniform, so after the float32 cast adjacent
    # columns can be equal (diff == 0) or slightly out of order, which trips the
    # strict-monotonicity assertion in the ncore model constructor.
    # enforce_spinning_monotonic repairs such near-degenerate pairs and returns
    # float32.
    column_azimuths_rad = enforce_spinning_monotonic(col_az, n_cols, spinning_direction)

    # Per-beam elevation: median measured elevation over all valid columns, in
    # firing order. Spinning lidars do not necessarily fire their beams in
    # monotonic elevation order (e.g. some sensors interleave adjacent lasers),
    # so we cannot assume firing index maps to elevation by a simple reversal.
    beam_elevations = np.zeros(n_beams_per_column, dtype=np.float32)
    for r in range(n_beams_per_column):
        valid_cols_r = valid_grid[:, r]
        if valid_cols_r.any():
            beam_elevations[r] = np.median(el_grid[valid_cols_r, r])

    # The RowOffsetStructuredSpinningLidarModelParameters format requires rows in
    # strictly descending elevation (row 0 highest), independent of sensor. The
    # firing order, however, is sensor-specific and not necessarily monotonic in
    # elevation, so recover the beam-to-row mapping from the data by sorting beams
    # on their measured elevation and apply that permutation consistently to every
    # per-beam quantity. This makes the elevation ramp monotonic by construction
    # rather than assuming a fixed firing order.
    row_order = np.argsort(-beam_elevations, kind="stable")

    # Per-beam azimuth offsets: each beam's offset from the column azimuths.
    center_col = n_cols // 2
    beam_azimuth_offsets = np.zeros(n_beams_per_column, dtype=np.float32)

    az_diff_grid = np.arctan2(
        np.sin(az_grid - column_azimuths_rad[:, np.newaxis]),
        np.cos(az_grid - column_azimuths_rad[:, np.newaxis]),
    )

    # Count far-range valid points per beam (for blending decision)
    far_valid_grid = valid_grid & (dist_grid > far_range_m)
    beam_far_range_count = far_valid_grid.sum(axis=0)

    # Find valid diffs in progressively wider windows around center
    for r in range(n_beams_per_column):
        for half_width in [5, 50, n_cols // 2]:
            window_start = max(0, center_col - half_width)
            window_stop = min(n_cols, center_col + half_width + 1)
            window_valid = valid_grid[window_start:window_stop, r]
            if window_valid.any():
                beam_azimuth_offsets[r] = np.median(az_diff_grid[window_start:window_stop, r][window_valid])
                break

    # Reorder per-beam quantities into model row order (descending elevation).
    row_elevations = beam_elevations[row_order]
    row_azimuth_offsets = beam_azimuth_offsets[row_order]
    row_far_range_count = beam_far_range_count[row_order]

    # Blend with analytical firing offsets for rows with few far-range observations
    if beam_pair_interval_us > 0:
        scan_duration_us = int(1e6 / spinning_frequency_hz)
        analytical_offsets = compute_intra_column_firing_offsets(
            n_beams_per_column, beam_pair_interval_us, scan_duration_us, spinning_direction
        )
        effective_min_obs = min_obs_for_full_empirical if min_obs_for_full_empirical > 0 else n_cols // 2
        weight = np.minimum(row_far_range_count.astype(np.float64) / effective_min_obs, 1.0)
        row_azimuth_offsets = (weight * row_azimuth_offsets + (1.0 - weight) * analytical_offsets).astype(np.float32)

    return RowOffsetStructuredSpinningLidarModelParameters(
        spinning_frequency_hz=spinning_frequency_hz,
        spinning_direction=spinning_direction,
        n_rows=n_beams_per_column,
        n_columns=n_cols,
        row_elevations_rad=row_elevations,
        column_azimuths_rad=column_azimuths_rad,
        row_azimuth_offsets_rad=row_azimuth_offsets,
    )


# --- HDL-32E presets -----------------------------------------------------------
# Velodyne HDL-32E spinning lidar, 32 beams, ~20 Hz, CW rotation.

HDL32E_N_BEAMS: int = 32
HDL32E_N_COLUMNS: int = 1085  # 50000 us / 46.08 us per firing cycle
HDL32E_SCAN_DURATION_US: int = 50_000
HDL32E_FIRING_PAIR_INTERVAL_US: float = 1.152  # time between consecutive firing pairs

# Nominal elevation angles (degrees), model row order: highest to lowest.
HDL32E_ELEVATIONS_RAD: np.ndarray = np.radians(
    np.array(
        [
            10.67,
            9.33,
            8.00,
            6.67,
            5.33,
            4.00,
            2.67,
            1.33,
            0.00,
            -1.33,
            -2.67,
            -4.00,
            -5.33,
            -6.67,
            -8.00,
            -9.33,
            -10.67,
            -12.00,
            -13.33,
            -14.67,
            -16.00,
            -17.33,
            -18.67,
            -20.00,
            -21.33,
            -22.67,
            -24.00,
            -25.33,
            -26.67,
            -28.00,
            -29.33,
            -30.67,
        ],
        dtype=np.float64,
    )
).astype(np.float32)


def derive_nominal_hdl32e(
    spinning_frequency_hz: float = 20.0,
    start_azimuth_rad: float = 0.0,
    n_beams: int = HDL32E_N_BEAMS,
    n_columns: int = HDL32E_N_COLUMNS,
    scan_duration_us: int = HDL32E_SCAN_DURATION_US,
    elevations_rad: Optional[np.ndarray] = None,
    beam_pair_interval_us: float = HDL32E_FIRING_PAIR_INTERVAL_US,
    spinning_direction: Literal["cw", "ccw"] = "cw",
) -> RowOffsetStructuredSpinningLidarModelParameters:
    """Create a nominal HDL-32E model from spec values (no data dependency).

    Produces a native-resolution model using:
        - Spec elevation angles (or custom elevations_rad)
        - Uniform column azimuths spanning one full rotation
        - Analytical row azimuth offsets from firing timing

    Use upsample_model() on the result for sub-column alignment precision.

    Args:
        spinning_frequency_hz: Spinning frequency in Hz.
        start_azimuth_rad: Starting azimuth of column 0 in radians.
        n_beams: Number of beams (default HDL32E_N_BEAMS).
        n_columns: Number of columns per revolution (default HDL32E_N_COLUMNS).
        scan_duration_us: Scan duration in microseconds (default HDL32E_SCAN_DURATION_US).
        elevations_rad: Per-row elevation angles [n_beams] float32 in model row order.
            If None, uses HDL32E_ELEVATIONS_RAD.
        beam_pair_interval_us: Firing pair interval for offset computation.
        spinning_direction: Rotation direction.

    Returns:
        RowOffsetStructuredSpinningLidarModelParameters at native resolution.
    """
    if elevations_rad is None:
        elevations_rad = HDL32E_ELEVATIONS_RAD

    # Uniform column azimuths
    sign = -1.0 if spinning_direction == "cw" else 1.0
    step = sign * 2.0 * np.pi / n_columns
    column_azimuths = (start_azimuth_rad + step * np.arange(n_columns, dtype=np.float64)).astype(np.float32)
    column_azimuths = ((column_azimuths + np.pi) % (2 * np.pi) - np.pi).astype(np.float32)
    column_azimuths[column_azimuths > column_azimuths[0]] -= np.float32(2 * np.pi)

    # Analytical row offsets from firing timing
    row_azimuth_offsets = compute_intra_column_firing_offsets(
        n_beams=n_beams,
        beam_pair_interval_us=beam_pair_interval_us,
        scan_duration_us=scan_duration_us,
        spinning_direction=spinning_direction,
    )

    return RowOffsetStructuredSpinningLidarModelParameters(
        spinning_frequency_hz=spinning_frequency_hz,
        spinning_direction=spinning_direction,
        n_rows=n_beams,
        n_columns=n_columns,
        row_elevations_rad=elevations_rad,
        column_azimuths_rad=column_azimuths,
        row_azimuth_offsets_rad=row_azimuth_offsets,
    )
