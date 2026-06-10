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

"""Evaluate lidar model quality by comparing model-predicted directions against stored native directions.

Produces numerical metrics (angular error, pixel-equivalent error) and optional visual output
(side-by-side projections and error-colored overlays).

Usage:
    bazel run //tools:ncore_evaluate_lidar_model -- \\
        --source-id lidar_top \\
        --camera-id camera_front \\
        --output-dir /tmp/eval_output \\
        v4 --component-group /path/to/scene.json
"""

from __future__ import annotations

import dataclasses
import logging

from pathlib import Path
from typing import List, Optional, cast

import click
import cv2
import numpy as np
import tqdm

from ncore.impl.common.transformations import MotionCompensator, se3_inverse, transform_point_cloud
from ncore.impl.common.util import unpack_optional
from ncore.impl.data import types
from ncore.impl.data.compat import LidarSensorProtocol
from ncore.impl.data.util import padded_index_string
from ncore.impl.data.v4.compat import SequenceLoaderProtocol, SequenceLoaderV4
from ncore.impl.data.v4.components import SequenceComponentGroupsReader
from ncore.impl.sensors.camera import CameraModel
from ncore.impl.sensors.lidar import StructuredLidarModel
from tools.colormaps import turbo as turbo_colormap


try:
    from .cli import OptionalStrParamType
except ImportError:
    from tools.cli import OptionalStrParamType


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -- Metric computation --------------------------------------------------------


@dataclasses.dataclass
class FrameMetrics:
    """Per-frame evaluation metrics."""

    frame_index: int
    n_points: int
    n_valid: int  # points with distance > 0
    n_far: int  # points with distance > far_range_m
    # Combined angular error
    mean_err_deg: float
    median_err_deg: float
    p95_err_deg: float
    max_err_deg: float
    mean_err_far_deg: float
    median_err_far_deg: float
    p95_err_far_deg: float
    # Azimuth error (signed, for far-range)
    mean_az_err_deg: float
    mean_az_err_far_deg: float
    # Elevation error (signed, for far-range)
    mean_el_err_deg: float
    mean_el_err_far_deg: float
    # Per-row breakdown
    per_row_mean_err_deg: np.ndarray  # [n_rows]
    per_row_mean_err_far_deg: np.ndarray  # [n_rows]


def compute_frame_metrics(
    directions: np.ndarray,
    model_directions: np.ndarray,
    distances: np.ndarray,
    model_element: np.ndarray,
    n_rows: int,
    frame_index: int,
    far_range_m: float = 20.0,
) -> FrameMetrics:
    """Compute angular error metrics for a single frame.

    Args:
        directions: Stored native direction unit vectors [N, 3].
        model_directions: Model-predicted direction unit vectors [N, 3].
        distances: Per-point distances [N].
        model_element: Model element indices [N, 2] as (row, col).
        n_rows: Total number of model rows (beams).
        frame_index: Frame index for reporting.
        far_range_m: Distance threshold for far-range metrics.

    Returns:
        FrameMetrics with all computed statistics.
    """
    valid_mask = distances > 0
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        empty_rows = np.zeros(n_rows, dtype=np.float64)
        return FrameMetrics(
            frame_index=frame_index,
            n_points=len(distances),
            n_valid=0,
            n_far=0,
            mean_err_deg=0.0,
            median_err_deg=0.0,
            p95_err_deg=0.0,
            max_err_deg=0.0,
            mean_err_far_deg=0.0,
            median_err_far_deg=0.0,
            p95_err_far_deg=0.0,
            mean_az_err_deg=0.0,
            mean_az_err_far_deg=0.0,
            mean_el_err_deg=0.0,
            mean_el_err_far_deg=0.0,
            per_row_mean_err_deg=empty_rows,
            per_row_mean_err_far_deg=empty_rows,
        )

    # Combined angular error for valid points
    cos_angle = np.clip(np.sum(directions[valid_mask] * model_directions[valid_mask], axis=1), -1.0, 1.0)
    err_rad = np.arccos(cos_angle)
    err_deg = np.degrees(err_rad)

    # Decompose into azimuth and elevation errors (signed)
    actual_az = np.arctan2(directions[valid_mask, 1], directions[valid_mask, 0])
    model_az = np.arctan2(model_directions[valid_mask, 1], model_directions[valid_mask, 0])
    az_err_rad = np.arctan2(np.sin(actual_az - model_az), np.cos(actual_az - model_az))

    actual_el = np.arcsin(np.clip(directions[valid_mask, 2], -1.0, 1.0))
    model_el = np.arcsin(np.clip(model_directions[valid_mask, 2], -1.0, 1.0))
    el_err_rad = actual_el - model_el

    mean_az_err = float(np.degrees(az_err_rad.mean()))
    mean_el_err = float(np.degrees(el_err_rad.mean()))

    # Far-range subset
    far_mask_local = distances[valid_mask] > far_range_m
    n_far = int(far_mask_local.sum())

    if n_far > 0:
        err_far_deg = err_deg[far_mask_local]
        mean_err_far = float(err_far_deg.mean())
        median_err_far = float(np.median(err_far_deg))
        p95_err_far = float(np.percentile(err_far_deg, 95))
        mean_az_err_far = float(np.degrees(az_err_rad[far_mask_local].mean()))
        mean_el_err_far = float(np.degrees(el_err_rad[far_mask_local].mean()))
    else:
        mean_err_far = float(err_deg.mean())
        median_err_far = float(np.median(err_deg))
        p95_err_far = float(np.percentile(err_deg, 95))
        mean_az_err_far = mean_az_err
        mean_el_err_far = mean_el_err

    # Per-row breakdown
    rows_valid = model_element[valid_mask, 0].astype(np.int64)
    per_row_mean_err = np.zeros(n_rows, dtype=np.float64)
    per_row_mean_err_far = np.zeros(n_rows, dtype=np.float64)

    for r in range(n_rows):
        row_mask = rows_valid == r
        if row_mask.any():
            per_row_mean_err[r] = float(err_deg[row_mask].mean())
            # Far-range for this row
            row_far = row_mask & far_mask_local
            if row_far.any():
                per_row_mean_err_far[r] = float(err_deg[row_far].mean())
            else:
                per_row_mean_err_far[r] = per_row_mean_err[r]

    return FrameMetrics(
        frame_index=frame_index,
        n_points=len(distances),
        n_valid=n_valid,
        n_far=n_far,
        mean_err_deg=float(err_deg.mean()),
        median_err_deg=float(np.median(err_deg)),
        p95_err_deg=float(np.percentile(err_deg, 95)),
        max_err_deg=float(err_deg.max()),
        mean_err_far_deg=mean_err_far,
        median_err_far_deg=median_err_far,
        p95_err_far_deg=p95_err_far,
        mean_az_err_deg=mean_az_err,
        mean_az_err_far_deg=mean_az_err_far,
        mean_el_err_deg=mean_el_err,
        mean_el_err_far_deg=mean_el_err_far,
        per_row_mean_err_deg=per_row_mean_err,
        per_row_mean_err_far_deg=per_row_mean_err_far,
    )


# -- Image rendering -----------------------------------------------------------


def _render_points_on_image(
    image: np.ndarray,
    pixel_coords: np.ndarray,
    colors: np.ndarray,
    point_size: float = 1.5,
) -> np.ndarray:
    """Render colored points on a camera image.

    Args:
        image: Camera image (H, W, 3) RGB uint8.
        pixel_coords: [N, 2] pixel coordinates.
        colors: [N, 3] RGB uint8 colors.
        point_size: Radius of rendered points.

    Returns:
        Copy of image with points rendered.
    """
    output = image.copy()
    radius = max(1, int(round(point_size)))
    for i in range(len(pixel_coords)):
        px = int(round(pixel_coords[i, 0]))
        py = int(round(pixel_coords[i, 1]))
        color = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))
        cv2.circle(output, (px, py), radius, color, thickness=-1, lineType=cv2.LINE_AA)
    return output


def _error_to_colors(error_deg: np.ndarray, max_error_deg: float = 1.0) -> np.ndarray:
    """Map angular error to a green-yellow-red colormap.

    Args:
        error_deg: Per-point error in degrees [N].
        max_error_deg: Error value that maps to maximum color (red).

    Returns:
        [N, 3] uint8 RGB colors. Green = 0 error, red = max_error_deg+.
    """
    # Use turbo colormap: blue/green = low error, red = high error
    normalized = np.clip(error_deg / max_error_deg, 0.0, 1.0)
    return turbo_colormap(normalized)


# -- Main evaluation logic -----------------------------------------------------


# -- Main evaluation logic -----------------------------------------------------


def _evaluate_sequence(
    ray_sensor: LidarSensorProtocol,
    lidar_model: StructuredLidarModel,
    source_id: str,
    frame_indices: List[int],
    far_range_m: float,
    cam_sensor=None,
    cam_model: Optional[CameraModel] = None,
    motion_compensator: Optional[MotionCompensator] = None,
    output_dir: Optional[Path] = None,
    point_size: float = 1.5,
    max_error_deg: float = 1.0,
    pose_mode: str = "rolling-shutter",
) -> List[FrameMetrics]:
    """Evaluate model quality across frames, optionally producing images.

    Returns list of per-frame metrics.
    """
    n_rows = cast(int, lidar_model.n_rows)
    all_metrics: List[FrameMetrics] = []

    for frame_index in tqdm.tqdm(frame_indices, desc="Evaluating frames"):
        # Read native sensor-frame points via compat API (non-motion-compensated)
        frame_pc = ray_sensor.get_frame_point_cloud(frame_index, motion_compensation=False, with_start_points=False)
        xyz_native = frame_pc.xyz_m_end  # [N, 3] in sensor frame at measurement time

        model_element = unpack_optional(
            ray_sensor.get_frame_ray_bundle_model_element(frame_index),
            msg=f"No model_element for frame {frame_index}",
        )
        distances = np.linalg.norm(xyz_native, axis=1).astype(np.float32)

        # Derive native directions from point positions
        valid_dist = distances > 0
        directions = np.zeros_like(xyz_native)
        directions[valid_dist] = xyz_native[valid_dist] / distances[valid_dist, np.newaxis]

        # Compute model-predicted directions via the torch lidar model
        model_directions = lidar_model.elements_to_sensor_rays(model_element).cpu().numpy()

        # Compute metrics
        metrics = compute_frame_metrics(
            directions=directions,
            model_directions=model_directions,
            distances=distances,
            model_element=model_element,
            n_rows=n_rows,
            frame_index=frame_index,
            far_range_m=far_range_m,
        )
        all_metrics.append(metrics)

        # Produce images if camera is available
        if cam_sensor is not None and cam_model is not None and output_dir is not None:
            _render_frame_comparison(
                ray_sensor=ray_sensor,
                source_id=source_id,
                frame_index=frame_index,
                directions=directions,
                model_directions=model_directions,
                distances=distances,
                cam_sensor=cam_sensor,
                cam_model=cam_model,
                motion_compensator=motion_compensator,
                output_dir=output_dir,
                point_size=point_size,
                max_error_deg=max_error_deg,
                pose_mode=pose_mode,
            )

    return all_metrics


def _render_frame_comparison(
    ray_sensor: LidarSensorProtocol,
    source_id: str,
    frame_index: int,
    directions: np.ndarray,
    model_directions: np.ndarray,
    distances: np.ndarray,
    cam_sensor,
    cam_model: CameraModel,
    motion_compensator: Optional[MotionCompensator],
    output_dir: Path,
    point_size: float,
    max_error_deg: float,
    pose_mode: str,
) -> None:
    """Render side-by-side and error overlay images for one frame."""
    # Find closest camera frame to this lidar frame
    lidar_start_us = ray_sensor.get_frame_timestamp_us(frame_index, types.FrameTimepoint.START)
    lidar_end_us = ray_sensor.get_frame_timestamp_us(frame_index, types.FrameTimepoint.END)
    lidar_center_us = lidar_start_us + (lidar_end_us - lidar_start_us) // 2

    cam_frame_index = cam_sensor.get_closest_frame_index(lidar_center_us, relative_frame_time=0.5)
    cam_image = cam_sensor.get_frame_image_array(cam_frame_index)

    # Get camera transforms
    T_world_cam_start = se3_inverse(
        cam_sensor.get_frames_T_sensor_target("world", cam_frame_index, types.FrameTimepoint.START)
    )
    T_world_cam_end = se3_inverse(
        cam_sensor.get_frames_T_sensor_target("world", cam_frame_index, types.FrameTimepoint.END)
    )

    # Compute world-frame points from NATIVE directions
    valid_mask = distances > 0
    native_sensor_pts = directions * distances[:, np.newaxis]

    # Compute world-frame points from MODEL directions
    model_sensor_pts = model_directions * distances[:, np.newaxis]

    # Motion compensate and transform to world
    timestamps = ray_sensor.get_frame_ray_bundle_timestamp_us(frame_index)
    frame_start_us = ray_sensor.get_frame_timestamp_us(frame_index, types.FrameTimepoint.START)
    frame_end_us = ray_sensor.get_frame_timestamp_us(frame_index, types.FrameTimepoint.END)
    T_lidar_world = ray_sensor.get_frames_T_sensor_target("world", frame_index, types.FrameTimepoint.END)

    if motion_compensator is not None:
        native_mc = motion_compensator.motion_compensate_points(
            sensor_id=source_id,
            xyz_pointtime=native_sensor_pts,
            timestamp_us=timestamps,
            frame_start_timestamp_us=frame_start_us,
            frame_end_timestamp_us=frame_end_us,
        ).xyz_e_reftime
        model_mc = motion_compensator.motion_compensate_points(
            sensor_id=source_id,
            xyz_pointtime=model_sensor_pts,
            timestamp_us=timestamps,
            frame_start_timestamp_us=frame_start_us,
            frame_end_timestamp_us=frame_end_us,
        ).xyz_e_reftime
    else:
        native_mc = native_sensor_pts
        model_mc = model_sensor_pts

    native_world = transform_point_cloud(native_mc, T_lidar_world)
    model_world = transform_point_cloud(model_mc, T_lidar_world)

    # Project both sets of points to camera
    native_proj, native_valid_idx = _project_points(
        native_world[valid_mask], cam_model, T_world_cam_start, T_world_cam_end, pose_mode
    )
    model_proj, model_valid_idx = _project_points(
        model_world[valid_mask], cam_model, T_world_cam_start, T_world_cam_end, pose_mode
    )

    # Compute per-point angular error for valid points (all valid-distance points)
    cos_angle = np.clip(np.sum(directions[valid_mask] * model_directions[valid_mask], axis=1), -1.0, 1.0)
    point_error_deg = np.degrees(np.arccos(cos_angle))

    frame_str = padded_index_string(frame_index)

    # Overlay image: native points (cyan) + model points (red) on same image.
    # Where model is accurate, both overlap and you see a blend.
    # Where model is wrong, you see the two colors separated.
    if native_proj is not None and model_proj is not None:
        n_native = len(native_proj)
        n_model = len(model_proj)
        cyan = np.full((n_native, 3), [0, 220, 220], dtype=np.uint8)
        red = np.full((n_model, 3), [220, 50, 50], dtype=np.uint8)

        # Draw model (red) first, then native (cyan) on top -- so native is visible
        overlay_img = _render_points_on_image(cam_image, model_proj, red, point_size)
        overlay_img = _render_points_on_image(overlay_img, native_proj, cyan, point_size)
        path_overlay = output_dir / f"{frame_str}_overlay.png"
        cv2.imwrite(str(path_overlay), cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))

    # Error heatmap: native points colored by angular error magnitude
    if native_proj is not None and native_valid_idx is not None:
        # Filter error to only the points that projected into the camera
        proj_error_deg = point_error_deg[native_valid_idx]
        error_colors = _error_to_colors(proj_error_deg, max_error_deg)
        error_img = _render_points_on_image(cam_image, native_proj, error_colors, point_size)
        path_err = output_dir / f"{frame_str}_error.png"
        cv2.imwrite(str(path_err), cv2.cvtColor(error_img, cv2.COLOR_RGB2BGR))


def _project_points(
    world_points: np.ndarray,
    cam_model: CameraModel,
    T_world_cam_start: np.ndarray,
    T_world_cam_end: np.ndarray,
    pose_mode: str,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Project world-frame points to pixel coordinates.

    Returns (pixel_coords [N_valid, 2], valid_indices [N_valid]) or (None, None).
    """
    if len(world_points) == 0:
        return None, None

    match pose_mode:
        case "rolling-shutter":
            result = cam_model.world_points_to_image_points_shutter_pose(
                world_points, T_world_cam_start, T_world_cam_end, return_valid_indices=True
            )
        case "mean":
            result = cam_model.world_points_to_image_points_mean_pose(
                world_points, T_world_cam_start, T_world_cam_end, return_valid_indices=True
            )
        case "start":
            result = cam_model.world_points_to_image_points_static_pose(
                world_points, T_world_cam_start, return_valid_indices=True
            )
        case "end":
            result = cam_model.world_points_to_image_points_static_pose(
                world_points, T_world_cam_end, return_valid_indices=True
            )
        case _:
            raise ValueError(f"Unsupported pose mode: {pose_mode}")

    coords = result.image_points.cpu().numpy()
    valid_idx = result.valid_indices
    if valid_idx is None or len(coords) == 0:
        return None, None
    return coords[:, :2], valid_idx.cpu().numpy()


# -- Reporting -----------------------------------------------------------------


def _print_summary(
    all_metrics: List[FrameMetrics],
    n_rows: int,
    focal_length_px: Optional[float],
    warn_threshold_deg: Optional[float] = None,
) -> bool:
    """Print a summary report of evaluation results.

    Returns True if all metrics are within the warning threshold (or no threshold set).
    """
    total_points = sum(m.n_valid for m in all_metrics)
    total_far = sum(m.n_far for m in all_metrics)

    if total_points == 0:
        logger.warning("No valid points found across all frames.")
        return False

    weighted_mean = sum(m.mean_err_deg * m.n_valid for m in all_metrics) / total_points
    weighted_mean_far = sum(m.mean_err_far_deg * m.n_far for m in all_metrics) / max(total_far, 1)
    weighted_az_err = sum(m.mean_az_err_far_deg * m.n_far for m in all_metrics) / max(total_far, 1)
    weighted_el_err = sum(m.mean_el_err_far_deg * m.n_far for m in all_metrics) / max(total_far, 1)

    all_p95 = [m.p95_err_deg for m in all_metrics]
    all_max = [m.max_err_deg for m in all_metrics]
    all_median = [m.median_err_deg for m in all_metrics]

    print("\n" + "=" * 70)
    print(f"LIDAR MODEL EVALUATION SUMMARY ({len(all_metrics)} frames, {total_points:,} valid points)")
    print("=" * 70)

    print(f"\n  Combined angular error (all valid points):")
    print(f"    mean   = {weighted_mean:.4f} deg")
    print(f"    median = {np.median(all_median):.4f} deg")
    print(f"    p95    = {np.median(all_p95):.4f} deg (median of per-frame p95)")
    print(f"    max    = {np.max(all_max):.4f} deg")

    print(f"\n  Combined angular error (far-range > 20m, {total_far:,} points):")
    print(f"    mean   = {weighted_mean_far:.4f} deg")

    print(f"\n  Azimuth error (signed mean, far-range): {weighted_az_err:+.4f} deg")
    print(f"  Elevation error (signed mean, far-range): {weighted_el_err:+.4f} deg")

    if focal_length_px is not None:
        px_mean = weighted_mean * np.pi / 180.0 * focal_length_px
        px_mean_far = weighted_mean_far * np.pi / 180.0 * focal_length_px
        px_p95 = np.median(all_p95) * np.pi / 180.0 * focal_length_px
        print(f"\n  Pixel-equivalent error (focal_length = {focal_length_px:.0f} px):")
        print(f"    mean (all)  = {px_mean:.2f} px")
        print(f"    mean (far)  = {px_mean_far:.2f} px")
        print(f"    p95         = {px_p95:.2f} px")

    # Per-frame table
    print(f"\n  Per-frame breakdown:")
    print(f"    {'Frame':>6} {'N pts':>8} {'Mean(deg)':>10} {'P95(deg)':>10} {'Az err':>10} {'El err':>10}")
    print(f"    {'-' * 6} {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
    for m in all_metrics:
        print(
            f"    {m.frame_index:6d} {m.n_valid:8d} {m.mean_err_deg:10.4f} {m.p95_err_deg:10.4f} {m.mean_az_err_deg:10.4f} {m.mean_el_err_deg:10.4f}"
        )

    # Per-row breakdown (aggregated across all frames)
    print(f"\n  Per-row (beam) breakdown:")
    print(f"    {'Row':>4} {'Mean(deg)':>10} {'Mean far(deg)':>14}")
    print(f"    {'-' * 4} {'-' * 10} {'-' * 14}")

    # Aggregate per-row across frames
    row_err_sum = np.zeros(n_rows, dtype=np.float64)
    row_err_far_sum = np.zeros(n_rows, dtype=np.float64)
    row_count = np.zeros(n_rows, dtype=np.float64)
    for m in all_metrics:
        row_err_sum += m.per_row_mean_err_deg * m.n_valid
        row_err_far_sum += m.per_row_mean_err_far_deg * m.n_valid
        row_count += m.n_valid

    for r in range(n_rows):
        if row_count[r] > 0:
            row_mean = row_err_sum[r] / row_count[r]
            row_mean_far = row_err_far_sum[r] / row_count[r]
            print(f"    {r:4d} {row_mean:10.4f} {row_mean_far:14.4f}")

    # Detect systematic patterns
    print(f"\n  Systematic pattern detection:")
    row_means = row_err_sum / np.maximum(row_count, 1)
    best_row = int(np.argmin(row_means))
    worst_row = int(np.argmax(row_means))
    print(f"    Best row:  {best_row} ({row_means[best_row]:.4f} deg)")
    print(f"    Worst row: {worst_row} ({row_means[worst_row]:.4f} deg)")
    print(f"    Row error range: {row_means.max() - row_means.min():.4f} deg")

    # Per-frame variance (indicates alignment instability)
    frame_means = [m.mean_err_deg for m in all_metrics]
    print(f"\n  Frame-to-frame consistency:")
    print(f"    Mean of frame means: {np.mean(frame_means):.4f} deg")
    print(f"    Std of frame means:  {np.std(frame_means):.4f} deg")
    print(f"    Best frame:  {all_metrics[int(np.argmin(frame_means))].frame_index} ({min(frame_means):.4f} deg)")
    print(f"    Worst frame: {all_metrics[int(np.argmax(frame_means))].frame_index} ({max(frame_means):.4f} deg)")

    print("\n" + "=" * 70)

    # Warning threshold check
    passed = True
    if warn_threshold_deg is not None:
        if weighted_mean_far > warn_threshold_deg:
            logger.warning(
                f"Mean far-range error ({weighted_mean_far:.4f} deg) exceeds threshold ({warn_threshold_deg:.4f} deg)"
            )
            passed = False
        if abs(weighted_az_err) > warn_threshold_deg:
            logger.warning(
                f"Systematic azimuth error ({weighted_az_err:+.4f} deg) exceeds "
                f"threshold ({warn_threshold_deg:.4f} deg)"
            )
            passed = False
        if abs(weighted_el_err) > warn_threshold_deg:
            logger.warning(
                f"Systematic elevation error ({weighted_el_err:+.4f} deg) exceeds "
                f"threshold ({warn_threshold_deg:.4f} deg)"
            )
            passed = False
        if passed:
            logger.info(f"All metrics within threshold ({warn_threshold_deg:.4f} deg)")

    return passed


# -- CLI -----------------------------------------------------------------------


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class CLIBaseParams:
    """Parameters for the lidar model evaluation tool."""

    source_id: str
    camera_id: Optional[str]
    output_dir: Optional[str]
    start_frame: Optional[int]
    stop_frame: Optional[int]
    step_frame: Optional[int]
    far_range_m: float
    point_size: float
    max_error_deg: float
    warn_threshold_deg: float
    device: str
    pose: str
    open_consolidated: bool


@click.group(invoke_without_command=False)
@click.option("--source-id", required=True, type=str, help="Lidar sensor ID to evaluate")
@click.option("--camera-id", default=None, type=str, help="Camera for pixel metrics and image output")
@click.option("--output-dir", default=None, type=str, help="Output directory for images (requires --camera-id)")
@click.option("--start-frame", default=None, type=click.IntRange(min=0, max_open=True), help="First lidar frame")
@click.option("--stop-frame", default=None, type=click.IntRange(min=0, max_open=True), help="Past-the-end frame")
@click.option("--step-frame", default=None, type=click.IntRange(min=1, max_open=True), help="Frame step")
@click.option("--far-range-m", default=20.0, type=float, help="Distance threshold for far-range metrics")
@click.option("--point-size", default=1.5, type=float, help="Point radius for rendered images")
@click.option("--max-error-deg", default=1.0, type=float, help="Error colormap ceiling (degrees)")
@click.option(
    "--warn-threshold-deg",
    default=0.05,
    type=float,
    help="Warn if mean far-range error or systematic az/el error exceeds this (degrees)",
)
@click.option("--device", default="cpu", type=click.Choice(["cpu", "cuda"]), help="Torch device")
@click.option(
    "--pose",
    default="rolling-shutter",
    type=click.Choice(["rolling-shutter", "mean", "start", "end"]),
    help="Pose mode for projection",
)
@click.option("--open-consolidated/--no-open-consolidated", default=True, help="Pre-load shard meta-data")
@click.pass_context
def cli(ctx, **kwargs) -> None:
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
@click.option("--poses-component-group", type=str, default="default", help="Component group for 'poses'")
@click.option("--intrinsics-component-group", type=str, default="default", help="Component group for 'intrinsics'")
@click.option(
    "--masks-component-group",
    type=OptionalStrParamType(),
    default="default",
    help="Component group for 'masks' (use 'none' to disable)",
)
@click.option(
    "--cuboids-component-group",
    type=OptionalStrParamType(),
    default="default",
    help="Component group for 'cuboids' (use 'none' to disable)",
)
@click.pass_context
def v4(
    ctx,
    component_groups: tuple,
    poses_component_group: str,
    intrinsics_component_group: str,
    masks_component_group: Optional[str],
    cuboids_component_group: Optional[str],
) -> None:
    """Evaluate lidar model from a V4 sequence."""
    params: CLIBaseParams = ctx.obj

    reader = SequenceComponentGroupsReader(
        [Path(p) for p in component_groups],
        open_consolidated=params.open_consolidated,
    )
    run(
        params,
        SequenceLoaderV4(
            reader,
            poses_component_group_name=poses_component_group,
            intrinsics_component_group_name=intrinsics_component_group,
            masks_component_group_name=masks_component_group,
            cuboids_component_group_name=cuboids_component_group,
        ),
    )


def run(params: CLIBaseParams, loader: SequenceLoaderProtocol) -> None:
    """Evaluate lidar model quality for all frames in the sequence."""
    source_id = params.source_id
    camera_id = params.camera_id
    output_dir = Path(params.output_dir) if params.output_dir else None
    device = params.device

    # Get the lidar sensor
    ray_sensor = loader.get_lidar_sensor(source_id)
    assert isinstance(ray_sensor, LidarSensorProtocol), f"Sensor {source_id} is not a lidar"

    # Load the structured lidar model
    lidar_model = StructuredLidarModel.maybe_from_parameters(ray_sensor.model_parameters, device=device)
    if lidar_model is None:
        raise click.ClickException(f"No structured lidar model available for sensor '{source_id}'")

    logger.info(f"Lidar model: {lidar_model.n_rows} rows x {lidar_model.n_columns} columns")
    n_rows = cast(int, lidar_model.n_rows)

    # Determine frame range
    frame_indices = list(ray_sensor.get_frame_index_range(params.start_frame, params.stop_frame, params.step_frame))
    logger.info(f"Evaluating {len(frame_indices)} frames")

    # Setup camera if requested
    cam_sensor = None
    cam_model = None
    focal_length_px = None

    if camera_id is not None:
        cam_sensor = loader.get_camera_sensor(camera_id)
        cam_model_params = cam_sensor.model_parameters
        cam_model = CameraModel.from_parameters(cam_model_params, device=device)
        fl = getattr(cam_model_params, "focal_length", None)
        focal_length_px = float(fl[0]) if fl is not None else None
        if focal_length_px is not None:
            logger.info(f"Camera '{camera_id}': focal_length = {focal_length_px:.0f} px")

    # Motion compensator (needed for image rendering)
    motion_compensator = MotionCompensator(ray_sensor.pose_graph)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing images to {output_dir}")

    all_metrics = _evaluate_sequence(
        ray_sensor=ray_sensor,
        lidar_model=lidar_model,
        source_id=source_id,
        frame_indices=frame_indices,
        far_range_m=params.far_range_m,
        cam_sensor=cam_sensor,
        cam_model=cam_model,
        motion_compensator=motion_compensator,
        output_dir=output_dir,
        point_size=params.point_size,
        max_error_deg=params.max_error_deg,
        pose_mode=params.pose,
    )

    _print_summary(
        all_metrics,
        n_rows=n_rows,
        focal_length_px=focal_length_px,
        warn_threshold_deg=params.warn_threshold_deg,
    )


if __name__ == "__main__":
    cli(show_default=True)
