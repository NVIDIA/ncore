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

from __future__ import annotations

import sys

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Literal, Optional, Set, Tuple, Union, overload

import numpy as np

from scipy import interpolate, spatial
from scipy.spatial.transform import Rotation as R

from ncore.impl.common.util import unpack_optional


if TYPE_CHECKING:
    import numpy.typing as npt  # noqa: F401

    # torch is used for type annotations only - this module deliberately keeps its
    # runtime dependencies to numpy + scipy (see //ncore/impl/common:pylib_transformations),
    # so 'torch' is not resolvable to the type checker here.
    import torch  # type: ignore[import-not-found]  # noqa: F401  # ty: ignore[unresolved-import]


def time_bounds(timestamps_us: List[int], seek_sec: Optional[float], duration_sec: Optional[float]) -> tuple[int, int]:
    """
    Determine start and end timestamps given optional seek and duration times

    Args:
        timestamps_us : list of all available timestamps (in microseconds)
        seek_sec: Optional: if non-None, the time (in seconds)  to skip starting from the first timestamp
        duration_sec: Optional: if non-None, the total time (in seconds) between the start and end time bounds

    Return:
        start_timestamp_us: first valid timestamp in restricted bounds (in microseconds)
        end_timestamp_us: last valid timestamp in restricted bounds (in microseconds)
    """

    start_timestamp_us = int(timestamps_us[0])
    end_timestamp_us = int(timestamps_us[-1])

    if seek_sec is not None:
        assert seek_sec >= 0.0, "Require positive seek time"
        start_timestamp_us += int(seek_sec * 1e6)

    if duration_sec is not None:
        assert duration_sec > 0.0, "Require positive duration time"
        end_timestamp_us = start_timestamp_us + int(duration_sec * 1e6)

    assert start_timestamp_us < end_timestamp_us, "Arguments lead to invalid time bounds"

    return start_timestamp_us, end_timestamp_us


@dataclass(**({"slots": True, "frozen": True} if sys.version_info >= (3, 10) else {"frozen": True}))
class HalfClosedInterval:
    """Represents a half closed interval [start, stop) of integers"""

    start: int
    stop: int

    @property
    def end(self) -> int:
        """Returns the inclusive end of the interval (i.e. ``stop - 1``)."""
        return self.stop - 1

    @staticmethod
    def from_start_end(start: int, end: int) -> HalfClosedInterval:
        """Creates a half-closed interval from start and end (inclusive)"""
        return HalfClosedInterval(start, end + 1)

    def __post_init__(self) -> None:
        """Makes sure interval is well-defined"""
        assert isinstance(self.start, int)
        assert isinstance(self.stop, int)
        assert self.start <= self.stop

    def __contains__(self, item: Union[int, np.integer, HalfClosedInterval]) -> bool:
        """Determines if an item / other interval is contained in the interval"""
        if isinstance(item, (int, np.integer)):
            return bool(self.start <= item and item < self.stop)
        elif isinstance(item, HalfClosedInterval):
            return (self.start <= item.start) and (item.stop <= self.stop)
        else:
            raise TypeError(f"Expected int, np.integer, or HalfClosedInterval, got {type(item).__name__}")

    def __len__(self) -> int:
        """Returns the number of elements in the interval"""
        return self.stop - self.start

    def intersection(self, other: HalfClosedInterval) -> Optional[HalfClosedInterval]:
        """Computes the intersection of two half-closed interval"""
        if other.start >= self.stop or other.stop <= self.start:
            return None

        return HalfClosedInterval(max(self.start, other.start), min(self.stop, other.stop))

    def overlaps(self, other: HalfClosedInterval) -> bool:
        """Checks if the interval has a non-zero overlap with an other closed interval"""
        return self.intersection(other) is not None

    def cover_range(self, sorted_samples: np.ndarray) -> range:
        """Given a set of *sorted* samples (not validated), return the corresponding range for samples
        that are within the interval"""
        if (
            not len(sorted_samples)
            or len(self) == 0
            or not self.intersection(
                # generate closed integer interval [floor(sample[0]), ceil(samples[-1])+1] guaranteed to containing all samples[i]
                HalfClosedInterval(int(np.floor(sorted_samples[0])), int(np.ceil(sorted_samples[-1])) + 1)
            )
        ):
            # empty range for empty samples, empty interval, or missing intersection
            return range(0)

        # non-empty range case
        cover_range_start = np.argmax(self.start <= sorted_samples).item()
        cover_range_stop = (
            np.argmin(sorted_samples < self.stop).item() if self.stop < sorted_samples[-1] else len(sorted_samples)
        )  # full range of frames

        return range(cover_range_start, cover_range_stop)


class PoseInterpolator:
    """
    Interpolates the poses to the desired time stamps. The translation component is interpolated linearly,
    while spherical linear interpolation (SLERP) is used for the rotations. https://en.wikipedia.org/wiki/Slerp
    """

    @property
    def poses(self) -> np.ndarray:
        """Returns the original poses used for interpolation"""
        return self._poses

    @property
    def timestamps(self) -> np.ndarray:
        """Returns the timestamps corresponding to the original poses used for interpolation"""
        return self._timestamps

    def __init__(self, poses, timestamps) -> None:
        """Initializes the PoseInterpolator

        Args:
            poses: Rigid transformations in SE3 representation [m, 4, 4]
            timestamps: Corresponding timestamps for each pose [m]
        """

        poses = np.asarray(poses)
        assert poses.ndim == 3 and poses.shape[1:] == (4, 4) and np.issubdtype(poses.dtype, np.floating), (
            "Invalid poses input"
        )

        timestamps = np.asarray(timestamps)
        assert timestamps.ndim == 1 and len(timestamps) > 1, "Invalid timestamps input"

        self._poses = poses
        self._timestamps = timestamps

        self.slerp = spatial.transform.Slerp(timestamps, R.from_matrix(poses[:, :3, :3]))
        self.f_t = interpolate.interp1d(timestamps, poses[:, 0:3, 3], axis=0)

        self.last_row = np.array([0, 0, 0, 1], dtype=np.float32).reshape(1, 1, -1)

    def in_range(self, ts) -> bool:
        """Returns true if all provided timestamps (scalar or array-like) are within the interpolation range"""
        return (
            np.logical_and(self._timestamps[0] <= (ts_array := np.asarray(ts)), ts_array <= self._timestamps[-1])
            .all()
            .item()
        )

    def interpolate_to_timestamps(self, ts_target, dtype: npt.DTypeLike = np.float32) -> np.ndarray:
        """Interpolates poses to target timestamps.
        Args:
            ts_target: Target timestamps for which poses will be computed in the same time-domain as the original timestamps [m]
            dtype: Data type for the output poses

        Returns:
            np.ndarray: Interpolated poses in SE3 representation [m, 4, 4]

        Raises:
            ValueError: If any timestamp is outside the interpolation range
        """
        t_interp = self.f_t(ts_target).reshape(-1, 3, 1).astype(dtype)
        R_interp = self.slerp(ts_target).as_matrix().reshape(-1, 3, 3).astype(dtype)

        return np.concatenate(
            (
                np.concatenate([R_interp, t_interp], axis=-1),
                np.tile(self.last_row.astype(dtype=dtype), (R_interp.shape[0], 1, 1)),
            ),
            axis=1,
        )

    @staticmethod
    def _extrapolate_poses(
        ts_target: np.ndarray,
        t_ref: np.int64,
        pose_ref: np.ndarray,
        pose_vel_start: np.ndarray,
        pose_vel_end: np.ndarray,
        dt_us: np.int64,
        dtype: type[np.floating],
    ) -> np.ndarray:
        """Extrapolates poses from a reference pose using constant velocity.

        Args:
            ts_target: Target timestamps to extrapolate to [n; int64]
            t_ref: Reference timestamp to extrapolate from [int64]
            pose_ref: Reference pose to extrapolate from [4, 4]
            pose_vel_start: First pose for velocity computation [4, 4]
            pose_vel_end: Second pose for velocity computation [4, 4]
            dt_us: Time delta between velocity poses [int64]
            dtype: Floating point type for output poses (e.g., np.float32, np.float64)

        Returns:
            np.ndarray: Extrapolated poses [n, 4, 4]
        """
        # Use output dtype for intermediate computations
        float_dtype = np.dtype(dtype)

        # Linear velocity from pose translations
        t_start, t_end = pose_vel_start[:3, 3], pose_vel_end[:3, 3]
        linear_vel = (t_end - t_start) / float_dtype.type(dt_us)

        # Angular velocity from rotations (using rotation vector representation)
        R_start = R.from_matrix(pose_vel_start[:3, :3])
        R_end = R.from_matrix(pose_vel_end[:3, :3])
        R_delta = R_end * R_start.inv()
        omega = R_delta.as_rotvec() / float_dtype.type(dt_us)

        # Compute time deltas from reference
        delta_t: np.ndarray = (ts_target - t_ref).astype(float_dtype)

        # Extrapolate translation
        t_ref_pos = pose_ref[:3, 3]
        t_extrap = t_ref_pos + np.outer(delta_t, linear_vel)

        # Extrapolate rotation: R_extrap = exp(omega * delta_t) @ R_ref
        R_ref = R.from_matrix(pose_ref[:3, :3])
        rotvecs = np.outer(delta_t, omega)
        R_extrap = R.from_rotvec(rotvecs) * R_ref

        # Build output poses
        n = len(ts_target)
        result = np.empty((n, 4, 4), dtype=float_dtype)
        result[:, :3, :3] = R_extrap.as_matrix()
        result[:, :3, 3] = t_extrap
        result[:, 3, :3] = 0
        result[:, 3, 3] = 1

        return result

    def extrapolate_to_timestamps(
        self,
        ts_target: np.ndarray,
        max_extrapolation_time_us: int = 1_000_000,
        dtype: "npt.DTypeLike" = np.float32,
    ) -> np.ndarray:
        """Extrapolates/interpolates poses to target timestamps, allowing extrapolation beyond the original range.

        This requires the original timestamps to be in absolute microseconds (us / int type).

        For timestamps within the original range, standard interpolation is used.
        For timestamps outside the range, linear extrapolation is used for translation
        and SLERP-based extrapolation (constant angular velocity) is used for rotation.

        Args:
            ts_target: Target timestamps for which poses will be computed [m; us]
            max_extrapolation_time_us: Maximum allowed extrapolation time in microseconds (default: 1 second).
                                       Raises ValueError if any timestamp exceeds this limit.
            dtype: Data type for the output poses and the extrapolation computations

        Returns:
            np.ndarray: Extrapolated/interpolated poses in SE3 representation [m, 4, 4]

        Raises:
            ValueError: If any timestamp requires extrapolation beyond max_extrapolation_time_us
        """

        # For extrapolation, we restrict to integer timestamps only to be able to apply us-based max-extrapolation checks
        if not np.issubdtype(self._timestamps.dtype, np.integer):
            raise TypeError("Timestamps must be of integer type for extrapolation as assuming microseconds")

        if not np.issubdtype(ts_target.dtype, np.integer):
            raise TypeError("Target timestamps must be of integer type for extrapolation as assuming microseconds")

        # Convert to signed int64 to handle arithmetic correctly (avoid unsigned overflow)
        ts_target = np.asarray(ts_target, dtype=np.int64).ravel()
        if len(ts_target) == 0:
            return np.empty((0, 4, 4), dtype=dtype)

        t_start = np.int64(self._timestamps[0])
        t_end = np.int64(self._timestamps[-1])

        # Identify timestamps requiring extrapolation
        mask_before = ts_target < t_start
        mask_after = ts_target > t_end
        mask_within = ~mask_before & ~mask_after

        # Check extrapolation limits (using signed arithmetic)
        if mask_before.any():
            max_before_delta = (t_start - ts_target[mask_before]).max()
            if max_before_delta > max_extrapolation_time_us:
                raise ValueError(
                    f"Extrapolation before trajectory start exceeds limit: "
                    f"{max_before_delta} us > {max_extrapolation_time_us} us"
                )

        if mask_after.any():
            max_after_delta = (ts_target[mask_after] - t_end).max()
            if max_after_delta > max_extrapolation_time_us:
                raise ValueError(
                    f"Extrapolation after trajectory end exceeds limit: "
                    f"{max_after_delta} us > {max_extrapolation_time_us} us"
                )

        # Convert dtype to floating type for internal use
        float_type: type[np.floating] = np.dtype(dtype).type

        # Allocate output
        result = np.empty((len(ts_target), 4, 4), dtype=float_type)
        result[:, 3, :] = self.last_row.astype(float_type)

        # Handle timestamps within range using standard interpolation
        if mask_within.any():
            result[mask_within] = self.interpolate_to_timestamps(ts_target[mask_within], dtype=float_type)

        # Extrapolate before trajectory start (using velocity from first two poses)
        if mask_before.any():
            dt_us = np.int64(self._timestamps[1]) - np.int64(self._timestamps[0])
            result[mask_before] = self._extrapolate_poses(
                ts_target[mask_before], t_start, self._poses[0], self._poses[0], self._poses[1], dt_us, float_type
            )

        # Extrapolate after trajectory end (using velocity from last two poses)
        if mask_after.any():
            dt_us = np.int64(self._timestamps[-1]) - np.int64(self._timestamps[-2])
            result[mask_after] = self._extrapolate_poses(
                ts_target[mask_after], t_end, self._poses[-1], self._poses[-2], self._poses[-1], dt_us, float_type
            )

        return result


def so3_trans_2_se3(so3, trans) -> np.ndarray:
    """Create a 4x4 rigid transformation matrix given so3 rotation and translation.

    Args:
        so3: rotation matrix [n,3,3]
        trans: x, y, z translation [n, 3]

    Returns:
        np.ndarray: the constructed transformation matrix [n,4,4]
    """

    if so3.ndim > 2:
        T = np.eye(4)
        T = np.tile(T, (so3.shape[0], 1, 1))
        T[:, 0:3, 0:3] = so3
        T[:, 0:3, 3] = trans.reshape(
            -1,
            3,
        )

    else:
        T = np.eye(4)
        T[0:3, 0:3] = so3
        T[0:3, 3] = trans.reshape(
            3,
        )

    return T


def se3_inverse(T: np.ndarray, unbatch: bool = True) -> np.ndarray:
    """Computes the inverse of multiple rigid transformations

    Args:
        Ts (np.ndarray): se3 transformation matrices to invert [N, 4, 4] or [4, 4]
        unbatch (bool): if the single matrix should be unbatched (first dimension removed) or not

    Returns:
        (np array): Inverse transformations [N, 4, 4] or [4, 4]
    """

    # batch dimensions unconditionally
    T = T.reshape((-1, 4, 4))
    ret = np.stack([np.eye(4, dtype=T.dtype)] * len(T), axis=0)
    ret[:, :3, :3] = (Rt := T[:, :3, :3].transpose(0, 2, 1))
    ret[:, :3, 3:] = np.negative(Rt @ T[:, :3, 3:])

    if unbatch:  # unbatch dimensions conditionally
        ret = ret.squeeze()

    return ret


@overload
def transform_point_cloud(pc: np.ndarray, T) -> np.ndarray: ...


@overload
def transform_point_cloud(pc: "torch.Tensor", T) -> "torch.Tensor": ...


def transform_point_cloud(pc, T):
    """Transform the point cloud with the provided transformation matrix,
        support torch.Tensor and np.ndarry.

    The returned type matches the type of ``pc``: a numpy array in, a numpy array
    out; a torch tensor in, a torch tensor out.

    Args:
        pc (np.array): point cloud coordinates (x,y,z) [num_pts, 3] or [bs, num_pts, 3]
        T (np.array): se3 transformation matrix  [4, 4] or [bs, 4, 4]

    Out:
        (np array): transformed point cloud coordinated [num_pts, 3] or [bs, num_pts, 3]
    """
    if len(pc.shape) == 3:
        if isinstance(pc, np.ndarray):
            trans_pts = T[:, :3, :3] @ pc.transpose(0, 2, 1) + T[:, :3, 3:4]
            return trans_pts.transpose(0, 2, 1)
        else:
            trans_pts = T[:, :3, :3] @ pc.permute(0, 2, 1) + T[:, :3, 3:4]
            return trans_pts.permute(0, 2, 1)

    else:
        trans_pts = T[:3, :3] @ pc.transpose() + T[:3, 3:4]
        return trans_pts.transpose()


def bbox_pose(bbox: np.ndarray) -> np.ndarray:
    """Converts an array-encoded bounding-box into a corresponding pose"""

    return np.block(
        [
            [R.from_euler("xyz", bbox[6:9], degrees=False).as_matrix(), np.array(bbox[:3]).reshape((3, 1))],
            [np.array([0, 0, 0, 1])],
        ]
    )


def pose_bbox(pose: np.ndarray, dimensions: np.ndarray) -> np.ndarray:
    """Converts a pose with extents to an array-encoded bounding-box"""

    bbox = np.empty(9, dtype=dimensions.dtype)
    bbox[:3] = pose[:3, 3]  # centroid
    bbox[3:6] = dimensions  # dimensions from input
    bbox[6:9] = R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=False)  # orientation

    return bbox


def transform_bbox(bbox_source: np.ndarray, T_source_target: np.ndarray) -> np.ndarray:
    """Applies a rigid-transformation to a bounding box
    Args:
       bbox (np.ndarray): bounding-box in source-frame parameterized by [x, y, z, length, width, height, eulerX, eulerY, eulerZ]
       T (np.array): se3 source->target transformation matrix to apply [4,4]
    Out:
       (np array): transformed bounding-box [num_pts, 3] or [bs, num_pts, 3]
    """

    # Convert bbox to corresponding pose
    T_bbox_source = bbox_pose(bbox_source)

    # Apply transformation
    T_bbox_target = T_source_target @ T_bbox_source

    # Convert back to bbox parametrization (dimensions stay unchanged)
    return pose_bbox(T_bbox_target, bbox_source[3:6])


def is_within_3d_bboxes(pc: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
    """Checks whether points of a point cloud are in an array of 3d boxes

    Args:
        pc: [N, 3] tensor. Inner dims are: [x, y, z].
        bboxes: [M, 9] tensor. Inner dims are: [center_x, center_y, center_z, length, width, height, roll, pitch, yaw].
                        roll/pitch/yaw are in radians.
    Returns:
        point_in_bboxes; [N,M] boolean array.
    """

    assert np.shape(pc)[1] == 3, "Wrong PC input size"
    assert np.ndim(bboxes) == 2, "bboxes need to be a 2D numpy array"
    assert np.shape(bboxes)[1] == 9, "bboxes need to be a 2D numpy array"

    centers = bboxes[..., 0:3]
    dims = bboxes[..., 3:6]
    rot_angles = bboxes[..., 6:9]

    # Get the rotation matrices from the heading angles
    rotations = R.from_euler("xyz", rot_angles, degrees=False).as_matrix()

    # [M, 4, 4]
    transforms = so3_trans_2_se3(rotations, centers)

    # [M, 4, 4]
    transforms = se3_inverse(transforms, unbatch=False)

    # [M, 3, 3]
    rotations = transforms[..., 0:3, 0:3]

    # [M, 3]
    translations = transforms[..., 0:3, 3]

    # [N, M, 3]
    points_in_boxes_frames = np.matmul(rotations, pc.transpose()).transpose(2, 0, 1) + translations

    # [N, M, 3]
    points_in_bboxes = np.logical_and(
        np.logical_and(points_in_boxes_frames <= dims * 0.5, points_in_boxes_frames >= -dims * 0.5),
        np.all(np.not_equal(dims, 0), axis=-1, keepdims=True),
    )

    # [N, M]
    points_in_bboxes = np.prod(points_in_bboxes, axis=-1).astype(bool)

    return points_in_bboxes


MotionCompensationMethod = Literal["exact", "interpolated"]
"""Method of :class:`MotionCompensator` to evaluate the per-point transformations:

- ``"exact"``: interpolates the poses at each distinct point timestamp
- ``"interpolated"``: interpolates the poses at the timestamps of an adaptive grid spanning the frame, and linearly
  interpolates the transformations of the points in between (faster). The grid contains the pose timestamps of the
  pose graph within the frame, and is refined such that the interpolated transformations move the points by at most
  the interpolation tolerance from the exact ones (in addition to the float32 rounding of both methods).
"""


@dataclass(frozen=True)
class MotionCompensatorParameters:
    """Parameters of the motion compensation of :class:`MotionCompensator` (passed through, e.g., by sequence
    loaders to their sensors' point clouds)"""

    #: How the per-point transformations are evaluated, ``"exact"`` (default) or the faster ``"interpolated"`` (see
    #: :data:`MotionCompensationMethod`)
    method: MotionCompensationMethod = "exact"

    #: Maximum deviation of the points of the ``"interpolated"`` method from the exact transformations, in the units
    #: of the pose graph's translations (e.g. meters). The float32 rounding of both methods is not included. Smaller
    #: tolerances refine the interpolation grid at practically no cost (the per-point work doesn't depend on the grid
    #: size).
    interpolation_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        if self.method not in ("exact", "interpolated"):
            raise ValueError(f"Unsupported motion compensation method {self.method!r}")
        if not self.interpolation_tolerance > 0:
            raise ValueError(f"Interpolation tolerance must be positive, got {self.interpolation_tolerance}")


def _unique_timestamps(timestamp_us: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``np.unique(timestamp_us, return_inverse=True)``, without sorting for timestamps spanning a small range
    (e.g., the points of a frame)"""
    timestamp_us = np.asarray(timestamp_us)
    t_min = timestamp_us.min()
    span = int(timestamp_us.max() - t_min) + 1
    if span > 4 * timestamp_us.size + 1024:
        return np.unique(timestamp_us, return_inverse=True)
    offsets = (timestamp_us - t_min).astype(np.int64)
    present = np.zeros(span, dtype=bool)
    present[offsets] = True
    rank = np.cumsum(present, dtype=np.int64) - 1
    return (np.flatnonzero(present) + t_min).astype(timestamp_us.dtype), rank[offsets]


def _transform_points_per_point(
    xyz: np.ndarray, T_unique: np.ndarray, unique_idxs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Transforms each point by its transformation ``T_unique[unique_idxs]``, [N,3], [U,4,4], [N]

    Returns:
        the transformed points [N,3] and the translations of the transformations [N,3]
    """
    T = T_unique[unique_idxs, :3, :]  # [N,3,4]
    translations = T[:, :, 3]
    return np.einsum("nij,nj->ni", T[:, :, :3], xyz) + translations, translations


def _grid_intervals(timestamp_us: np.ndarray, grid_timestamps_us: np.ndarray) -> np.ndarray:
    """Index of the interval [grid[i], grid[i+1]] of each timestamp (clamped to the first / last interval), for a
    sorted grid of integer timestamps"""
    timestamp_us = np.asarray(timestamp_us).astype(np.int64)
    grid_timestamps_us = grid_timestamps_us.astype(np.int64)
    n_intervals = len(grid_timestamps_us) - 1
    t_min = int(grid_timestamps_us[0])
    span = int(grid_timestamps_us[-1]) - t_min + 1
    if span > 4 * timestamp_us.size + 1024:
        intervals = np.searchsorted(grid_timestamps_us, timestamp_us, side="right") - 1
    else:
        # lookup table of the interval of each timestamp within the grid's (small) time range
        table = np.searchsorted(grid_timestamps_us, np.arange(t_min, t_min + span, dtype=np.int64), side="right") - 1
        intervals = table[np.clip(timestamp_us - t_min, 0, span - 1)]
    return np.clip(intervals, 0, n_intervals - 1).astype(np.int32)


def _transform_points_interpolated(
    xyz: np.ndarray,
    timestamp_us: np.ndarray,
    grid_timestamps_us: np.ndarray,
    T_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transforms each point by the linear interpolation of the transformations ``T_grid`` [M+1,4,4] at its timestamp,
    ``T_grid`` evaluated at the sorted grid timestamps [M+1] (M >= 1)

    Returns:
        the transformed points [N,3] and the interpolated translations [N,3] (float32)
    """
    grid_timestamps_us = grid_timestamps_us.astype(np.int64)

    # interval index and interpolation weight of each point (from the interval's start and inverse length)
    interval = _grid_intervals(timestamp_us, grid_timestamps_us)
    interval_start_us = grid_timestamps_us[:-1].astype(np.float64)
    inverse_length = 1.0 / np.maximum(np.diff(grid_timestamps_us), 1).astype(np.float64)
    weight = (np.asarray(timestamp_us).astype(np.float64) - np.take(interval_start_us, interval)).astype(np.float32)
    weight *= np.take(inverse_length.astype(np.float32), interval)

    # transformation entries and their per-interval differences, one row per entry [12,M+1] / [12,M]
    T_grid = T_grid[:, :3, :].astype(np.float32)
    entries = np.ascontiguousarray(T_grid.reshape(-1, 12).T)
    deltas = np.ascontiguousarray(np.diff(T_grid, axis=0).reshape(-1, 12).T)

    x, y, z = np.asarray(xyz, dtype=np.float32).T.copy()
    transformed = np.empty((3, len(interval)), dtype=np.float32)
    translations = np.empty((3, len(interval)), dtype=np.float32)
    entry = np.empty(len(interval), dtype=np.float32)

    def interpolate_entry(index: int, out: np.ndarray) -> np.ndarray:
        np.take(deltas[index], interval, out=out)
        out *= weight
        out += np.take(entries[index], interval)
        return out

    for row in range(3):
        interpolate_entry(4 * row + 3, translations[row])
        transformed[row] = translations[row]
        transformed[row] += interpolate_entry(4 * row + 0, entry) * x
        transformed[row] += interpolate_entry(4 * row + 1, entry) * y
        transformed[row] += interpolate_entry(4 * row + 2, entry) * z

    return transformed.T, translations.T


def _interpolation_grid(
    evaluate_transformations: Callable[[np.ndarray], np.ndarray],
    breakpoint_timestamps_us: np.ndarray,
    max_point_distance: float,
    tolerance: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Grid of timestamps such that linearly interpolating the transformations between consecutive grid timestamps
    moves points up to ``max_point_distance`` from the origin by at most ``tolerance`` from the exact transformations

    The grid contains the (sorted, unique) breakpoint timestamps (the frame boundaries and the pose timestamps within
    the frame, at which the interpolated motion changes). Between breakpoints, the deviation of the interpolation is
    estimated at the interval midpoint (where it is largest), and the interval is subdivided as the deviation decreases
    quadratically with the interval length (with a safety factor of 2).

    Returns:
        the grid timestamps [M+1] and the transformations at them [M+1,4,4]
    """
    breakpoints = breakpoint_timestamps_us.astype(np.int64)
    if len(breakpoints) < 2:
        breakpoints = np.array([breakpoints[0], breakpoints[0] + 1], dtype=np.int64)
    midpoints = (breakpoints[:-1] + breakpoints[1:]) // 2
    T = evaluate_transformations(np.concatenate([breakpoints, midpoints]).astype(np.uint64)).astype(np.float64)
    T_breakpoints, T_midpoints = T[: len(breakpoints)], T[len(breakpoints) :]

    # deviation of the exact transformation at the midpoint from the interpolation of the interval's end
    # transformations, for points up to max_point_distance from the origin
    T_linear = 0.5 * (T_breakpoints[:-1] + T_breakpoints[1:])
    deviation = np.linalg.norm(
        T_midpoints[:, :3, :3] - T_linear[:, :3, :3], ord=2, axis=(1, 2)
    ) * max_point_distance + np.linalg.norm(T_midpoints[:, :3, 3] - T_linear[:, :3, 3], axis=1)
    subdivisions = np.maximum(1, np.ceil(np.sqrt(2.0 * deviation / tolerance))).astype(np.int64)
    # (intervals can't be subdivided below single microseconds)
    subdivisions = np.minimum(subdivisions, np.maximum(1, breakpoints[1:] - breakpoints[:-1]))

    if (subdivisions == 1).all():
        return breakpoints, T_breakpoints

    grid = np.unique(
        np.concatenate(
            [
                np.round(np.linspace(start, end, count + 1)[:-1]).astype(np.int64)
                for start, end, count in zip(breakpoints[:-1], breakpoints[1:], subdivisions)
            ]
            + [breakpoints[-1:]]
        )
    )
    return grid, evaluate_transformations(grid.astype(np.uint64))


class MotionCompensator:
    """Performs motion compensation / decompensation of points measured relative to time-dependent frames"""

    @staticmethod
    def from_sensor_rig(
        sensor_id: str,
        T_sensor_rig: np.ndarray,
        T_rig_worlds: np.ndarray,
        T_rig_worlds_timestamps_us: np.ndarray,
        anchor_frame_id: str = "world",
        rig_frame_id: str = "rig",
        parameters: Optional[MotionCompensatorParameters] = None,
    ) -> MotionCompensator:
        """Constructs a motion-compensator from a sensor-rig transformations for a specific sensor only

        Args:
            sensor_id (str): id of the sensor node
            T_sensor_rig (np.ndarray): static sensor -> rig transform, [4, 4]
            T_rig_worlds (np.ndarray): time-dependent rig -> anchor transforms, [N, 4, 4]
            T_rig_worlds_timestamps_us (np.ndarray): timestamps for the rig -> anchor transforms, [N]
            anchor_frame_id (str): name of the common anchor frame node (defaults to "world").
                Pass the same value as ``anchor_frame_id`` to the (de)compensation methods.
            rig_frame_id (str): name of the rig frame node (defaults to "rig").
            parameters (Optional[MotionCompensatorParameters]): parameters of the motion compensation
        """

        return MotionCompensator(
            PoseGraphInterpolator(
                [
                    PoseGraphInterpolator.Edge(
                        source_node=rig_frame_id,
                        target_node=anchor_frame_id,
                        T_source_target=T_rig_worlds,
                        timestamps_us=T_rig_worlds_timestamps_us,
                    ),
                    PoseGraphInterpolator.Edge(
                        source_node=sensor_id,
                        target_node=rig_frame_id,
                        T_source_target=T_sensor_rig,
                        timestamps_us=None,
                    ),
                ]
            ),
            parameters,
        )

    def __init__(
        self,
        pose_graph: PoseGraphInterpolator,
        parameters: Optional[MotionCompensatorParameters] = None,
    ) -> None:
        """
        Initializes motion-compensator to use a pose-graph interpolator

        Args:
            pose_graph (PoseGraphInterpolator): Should contain transformations from sensors to 'world' frame
            parameters (Optional[MotionCompensatorParameters]): parameters of the motion compensation (defaults to
                the exact method)
        """

        self._pose_graph = pose_graph
        self._parameters = parameters if parameters is not None else MotionCompensatorParameters()

    @property
    def parameters(self) -> MotionCompensatorParameters:
        """The parameters of the motion compensation"""
        return self._parameters

    def _point_transformations(
        self,
        sensor_id: str,
        anchor_frame_id: str,
        xyz: np.ndarray,
        timestamp_us: np.ndarray,
        frame_start_timestamp_us: int,
        frame_end_timestamp_us: int,
        evaluate_transformations: Callable[[np.ndarray], np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transforms each point by the transformation at its timestamp, evaluated by ``evaluate_transformations``
        ([K] timestamps -> [K,4,4] transformations), according to the method

        Returns:
            the transformed points [N,3] and the translations of the per-point transformations [N,3] (float32)
        """
        if self._parameters.method == "interpolated":
            # the interpolated motion changes at the pose timestamps of the time-dependent edges between the frames
            breakpoints = [np.array([frame_start_timestamp_us, frame_end_timestamp_us], dtype=np.int64)]
            for pose_timestamps_us in self._pose_graph.get_path_timestamps_us(sensor_id, anchor_frame_id):
                pose_timestamps_us = pose_timestamps_us.astype(np.int64)
                breakpoints.append(
                    pose_timestamps_us[
                        (pose_timestamps_us > frame_start_timestamp_us) & (pose_timestamps_us < frame_end_timestamp_us)
                    ]
                )
            xyz = np.asarray(xyz, dtype=np.float32)
            max_point_distance = float(np.sqrt(np.max(np.einsum("ij,ij->i", xyz, xyz))))
            grid_timestamps_us, T_grid = _interpolation_grid(
                evaluate_transformations,
                np.unique(np.concatenate(breakpoints)),
                max_point_distance,
                self._parameters.interpolation_tolerance,
            )
            return _transform_points_interpolated(xyz, timestamp_us, grid_timestamps_us, T_grid)

        # Determine unique timestamps to only perform actually required pose interpolations (a lot of points share
        # the same timestamp)
        timestamp_unique, unique_timestamp_reverse_idxs = _unique_timestamps(timestamp_us)
        return _transform_points_per_point(
            xyz,
            evaluate_transformations(timestamp_unique).astype(np.float32),
            unique_timestamp_reverse_idxs,
        )

    @dataclass
    class MotionCompensationResult:
        # motion-compensated ray segment start points, relative to the sensor frame at the reference time, [N,3]
        xyz_s_reftime: np.ndarray
        # motion-compensated ray segment end points, relative to the sensor frame at the reference time, [N,3]
        xyz_e_reftime: np.ndarray
        # reference sensor frame the compensated points are expressed in (equals the sensor_id)
        reference_frame_id: str
        # timestamp of the reference sensor frame the points are expressed in
        reference_timestamp_us: int
        # common anchor frame used to compose the compensation transforms
        anchor_frame_id: str

    def motion_compensate_points(
        self,
        sensor_id: str,
        xyz_pointtime: np.ndarray,
        timestamp_us: np.ndarray,
        frame_start_timestamp_us: int,
        frame_end_timestamp_us: int,
        reference_timestamp_us: Optional[int] = None,
        anchor_frame_id: str = "world",
    ) -> MotionCompensationResult:
        """
        Perform motion compensation of points in time-dependent sensor frame at measurement time to a common reference sensor frame

        By default the reference is the end of the frame (``frame_end_timestamp_us``).
        Pass ``reference_timestamp_us`` to compensate to a different reference frame
        (e.g. the start of the frame). The chosen reference
        must be inverted by :meth:`motion_decompensate_points` using the same
        ``reference_timestamp_us`` and ``anchor_frame_id``.

        Args:
            sensor_id (str): sensor the points are relative to. This is also the reference
                frame the compensated points are expressed in (``reference_frame_id``).
            xyz_pointtime(np.ndarray): points in time-dependent sensor frame (~before motion compensation), [N,3]
            timestamp_us(np.ndarray): timestamps of points, [N]
            frame_start_timestamp_us(int): frame start timestamp
            frame_end_timestamp_us(int): frame end timestamp
            reference_timestamp_us(Optional[int]): timestamp of the reference sensor frame to
                compensate to; defaults to ``frame_end_timestamp_us``. Must lie within
                ``[frame_start_timestamp_us, frame_end_timestamp_us]``.
            anchor_frame_id(str): common anchor frame used to compose the transforms
                (defaults to "world"). Use a non-default anchor when the pose graph is
                anchored at a frame other than "world".
        Returns:
            MotionCompensationResult: result of the motion-compensation
        """

        # Sanity check timestamp consistency
        assert len(xyz_pointtime) == len(timestamp_us)

        if reference_timestamp_us is None:
            reference_timestamp_us = frame_end_timestamp_us

        if not len(xyz_pointtime):
            return self.MotionCompensationResult(
                np.empty_like(xyz_pointtime, shape=(0, 3)),
                np.empty_like(xyz_pointtime, shape=(0, 3)),
                reference_frame_id=sensor_id,
                reference_timestamp_us=reference_timestamp_us,
                anchor_frame_id=anchor_frame_id,
            )

        assert frame_start_timestamp_us <= timestamp_us.min() and timestamp_us.max() <= frame_end_timestamp_us, (
            "Point timestamps not in frame time bounds"
        )
        assert frame_start_timestamp_us <= reference_timestamp_us <= frame_end_timestamp_us, (
            "Reference timestamp not in frame time bounds"
        )

        # Interpolate egomotion at the reference timestamp for the reference sensor pose
        T_anchor_sensor_reftime = self._pose_graph.evaluate_poses(
            anchor_frame_id, sensor_id, np.array(reference_timestamp_us, dtype=np.uint64)
        )

        # Per-point transform mapping the point-time sensor frame to the reference-time
        # sensor frame:
        #   T_sensor_pointtime_sensor_reftime = T_anchor_sensor_reftime @ T_sensor_pointtime_anchor
        # Its translation is the sensor position (in the reference-time pose) of each start point.
        xyz_e_reftime, xyz_s_reftime = self._point_transformations(
            sensor_id,
            anchor_frame_id,
            xyz_pointtime,
            timestamp_us,
            frame_start_timestamp_us,
            frame_end_timestamp_us,
            lambda timestamps_us: (
                T_anchor_sensor_reftime @ self._pose_graph.evaluate_poses(sensor_id, anchor_frame_id, timestamps_us)
            ),
        )

        return self.MotionCompensationResult(
            xyz_s_reftime,
            xyz_e_reftime,
            reference_frame_id=sensor_id,
            reference_timestamp_us=reference_timestamp_us,
            anchor_frame_id=anchor_frame_id,
        )

    def motion_decompensate_points(
        self,
        sensor_id: str,
        xyz_reftime: np.ndarray,
        timestamp_us: np.ndarray,
        frame_start_timestamp_us: int,
        frame_end_timestamp_us: int,
        reference_timestamp_us: Optional[int] = None,
        anchor_frame_id: str = "world",
    ) -> np.ndarray:
        """
        Decompensate motion of motion-compensated points to bring points into time-dependent sensor-frame

        The input points are assumed to be expressed in the sensor frame at a single
        *reference* timestamp (the timestamp the data was motion-compensated to). By
        default this reference is the end of the frame (``frame_end_timestamp_us``),
        matching :meth:`motion_compensate_points`. Datasets that compensate to a
        different reference (e.g. the start of the frame) can pass
        ``reference_timestamp_us`` explicitly. ``anchor_frame_id`` must match the value
        passed to :meth:`motion_compensate_points`.

        Args:
            sensor_id (str): sensor the points are relative to
            xyz_reftime (np.array): motion-compensated points relative to the reference-time sensor frame to be decompensated, [N,3]
            timestamp_us(np.ndarray): timestamps of points, [N]
            frame_start_timestamp_us(int): frame start timestamp
            frame_end_timestamp_us(int): frame end timestamp
            reference_timestamp_us(Optional[int]): timestamp of the sensor frame the points are
                expressed in; defaults to ``frame_end_timestamp_us``. Must lie within
                ``[frame_start_timestamp_us, frame_end_timestamp_us]``.
            anchor_frame_id(str): common anchor frame used to compose the transforms
                (defaults to "world"). Must match the anchor used for compensation.
        Returns:
            xyz_pointtime(np.array): points in time-dependent sensor frame after motion-decompensation [n,3]
        """

        # Sanity check timestamp consistency
        assert len(xyz_reftime) == len(timestamp_us)

        if not len(xyz_reftime):
            return np.empty_like(xyz_reftime, shape=(0, 3))

        if reference_timestamp_us is None:
            reference_timestamp_us = frame_end_timestamp_us

        assert frame_start_timestamp_us <= timestamp_us.min() and timestamp_us.max() <= frame_end_timestamp_us, (
            "Point timestamps not in frame time bounds"
        )
        assert frame_start_timestamp_us <= reference_timestamp_us <= frame_end_timestamp_us, (
            "Reference timestamp not in frame time bounds"
        )

        # Decompensation maps points from the reference-time sensor frame to each
        # point's own sensor frame. This is the exact inverse of
        # :meth:`motion_compensate_points`: per point we build the transform from the
        # reference-time sensor frame to the point-time sensor frame:
        #   T_sensor_reftime_sensor_pointtime = T_anchor_sensor_pointtime @ T_sensor_reftime_anchor
        # A point at the reference timestamp maps through the identity, as expected.
        T_sensor_reftime_anchor = self._pose_graph.evaluate_poses(
            sensor_id, anchor_frame_id, np.array(reference_timestamp_us, dtype=np.uint64)
        )  # [4, 4]

        # evaluate_poses(anchor, sensor) yields T_anchor_sensor_pointtime (the inverse of T_sensor_pointtime_anchor).
        xyz_pointtime, _ = self._point_transformations(
            sensor_id,
            anchor_frame_id,
            xyz_reftime,
            timestamp_us,
            frame_start_timestamp_us,
            frame_end_timestamp_us,
            lambda timestamps_us: (
                self._pose_graph.evaluate_poses(anchor_frame_id, sensor_id, timestamps_us) @ T_sensor_reftime_anchor
            ),
        )

        return xyz_pointtime


class PoseGraphInterpolator:
    """
    Interpolates rigid poses between named nodes in a pose graph. Edges in the pose graph can be static or dynamic (time-dependent)
    """

    class Edge:
        """
        Represents an edge in the pose graph, either static or dynamic (time-dependent).

        The computational data-type is given by the input arrays.
        """

        def __init__(
            self,
            source_node: str,
            target_node: str,
            T_source_target: npt.NDArray[np.floating],
            timestamps_us: Optional[npt.NDArray[np.uint64]],
        ) -> None:
            self.source_node = source_node
            self.target_node = target_node
            self.T_source_target = T_source_target  # either [4,4] (static) or [n,4,4] (dynamic)
            self.timestamps_us = timestamps_us  # either None (static) or [n] (dynamic)

            self.interpolator: Optional[PoseInterpolator] = None
            if timestamps_us is not None:
                assert len(self.T_source_target.shape) == 3 and self.T_source_target.shape[0] == len(timestamps_us)
                assert self.T_source_target.shape[1:] == (4, 4)
                self.interpolator = PoseInterpolator(self.T_source_target, timestamps_us)
            else:
                assert self.T_source_target.shape == (4, 4)

        def get_T(
            self, source_node: str, target_node: str, timestamps_us: npt.NDArray[np.uint64]
        ) -> npt.NDArray[np.floating]:
            """Returns the transformation from source to target node at the given timestamps

            Args:
                source_node (str): name of the source node
                target_node (str): name of the target node
                timestamps_us (np.ndarray): timestamps at which to evaluate the pose - defines the batch shape

            Returns:
                np.ndarray: evaluated poses from source to target node at the given timestamps [batch-shape,4,4]
                            in the data-type of the edge
            """

            assert np.issubdtype(timestamps_us.dtype, np.integer), "Timestamps must be of integer type"

            batch_shape = timestamps_us.shape

            if self.interpolator is not None:
                # perform interpolation
                T_source_target = self.interpolator.interpolate_to_timestamps(
                    timestamps_us.flatten(), dtype=self.T_source_target.dtype
                ).reshape(batch_shape + (4, 4))
            else:
                # static edge case, not time-dependent, simply repeat for batch shape
                T_source_target = np.tile(self.T_source_target, (batch_shape + (1, 1)))

            if source_node == self.source_node and target_node == self.target_node:
                # return direct edge value
                return T_source_target
            elif source_node == self.target_node and target_node == self.source_node:
                # return inverted edge value
                return se3_inverse(T_source_target, unbatch=False).reshape(batch_shape + (4, 4))
            else:
                raise KeyError("Invalid source/target node for edge")

    def __init__(self, edges: List[Edge]) -> None:
        # Collect graph
        self._nodes: Set[str] = set()
        for edge in edges:
            self._nodes.add(edge.source_node)
            self._nodes.add(edge.target_node)

        # Precompute all-pairs paths assuming undirected graph, check for cycles
        self._edge_map = {(edge.source_node, edge.target_node): edge for edge in edges}
        self._paths: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}
        for node in self._nodes:
            self._paths[node] = {node: []}
            visited = {node}
            queue = [node]
            while queue:
                current = queue.pop(0)
                for neighbor in [e.target_node for e in edges if e.source_node == current] + [
                    e.source_node for e in edges if e.target_node == current
                ]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        self._paths[node][neighbor] = self._paths[node][current] + [
                            (current, neighbor) if (current, neighbor) in self._edge_map else (neighbor, current)
                        ]
            if len(visited) != len(self._nodes):
                raise ValueError("Pose graph is not fully connected")

        # Map of normalized (sorted) node names to edges for more efficient lookup at graph traversal time
        self._normalized_edge_map = {self.normalize_node_pair(k[0], k[1]): edge for k, edge in self._edge_map.items()}

        # Check for cycles (simple check: number of edges must be |nodes|-1 for a tree)
        if len(edges) >= len(self._nodes):
            raise ValueError("Pose graph contains cycles")

    @property
    def nodes(self) -> Set[str]:
        """Returns the set of nodes in the pose graph"""
        return self._nodes

    def get_path_timestamps_us(self, source_node: str, target_node: str) -> List[npt.NDArray[np.uint64]]:
        """Returns the pose timestamps of the time-dependent edges on the path between two nodes (the timestamps at
        which the interpolated relative pose of the nodes may change its motion)"""
        if (path := self._paths.get(source_node, {}).get(target_node)) is None:
            raise KeyError(f"No path from {source_node} to {target_node} in pose graph")
        timestamps_us: List[npt.NDArray[np.uint64]] = []
        for edge_nodes in path:
            edge = self._normalized_edge_map[self.normalize_node_pair(edge_nodes[0], edge_nodes[1])]
            if edge.timestamps_us is not None:
                timestamps_us.append(edge.timestamps_us)
        return timestamps_us

    @staticmethod
    def normalize_node_pair(node_a: str, node_b: str) -> Tuple[str, str]:
        """Returns a normalized (sorted) tuple of the given node names

        Args:
            node_a (str): name of the first node
            node_b (str): name of the second node

        Returns:
            Tuple[str, str]: normalized (sorted) tuple of the given node names
        """
        return (node_a, node_b) if node_a < node_b else (node_b, node_a)

    @property
    def normalized_edge_map(self) -> Dict[Tuple[str, str], Edge]:
        """Returns the normalized (sorted by node names) edge map of the pose graph"""
        return self._normalized_edge_map

    def get_edge(self, source_node: str, target_node: str, normalized: bool = False) -> Optional[Edge]:
        """Returns the edge between the given source and target node, or None if no such edge exists

        Args:
            source_node (str): name of the source
            target_node (str): name of the target
            normalized (bool): if True, looks up the edge in normalized (sorted) manner

        Returns:
            Optional[Edge]: edge between source and target node, or None if no such edge exists
        """
        if normalized:
            return self._normalized_edge_map.get(self.normalize_node_pair(source_node, target_node), None)
        else:
            return self._edge_map.get((source_node, target_node), None)

    def evaluate_poses(
        self, source_node: str, target_node: str, timestamps_us: npt.NDArray[np.uint64]
    ) -> npt.NDArray[np.floating]:
        """
        Evaluates relative pose from source to target node frames at the given timestamps. Performs
        graph traversal and pose evaluation / composition (interpolation for time-dependent edges) as needed.

        Args:
            source_node (str): name of the source node
            target_node (str): name of the target node
            timestamps_us (np.ndarray): timestamps at which to evaluate the pose - defines the batch shape

        Returns:
            np.ndarray: evaluated poses from source to target node at the given timestamps [batch-shape,4,4]
        """

        batch_shape = timestamps_us.shape

        if source_node not in self._nodes:
            raise KeyError(f"Source node {source_node} not in pose graph")
        if target_node not in self._nodes:
            raise KeyError(f"Target node {target_node} not in pose graph")
        if source_node == target_node:
            return np.tile(np.eye(4, dtype=np.float32), (batch_shape + (1, 1)))

        if (path := self._paths[source_node].get(target_node)) is None:
            raise KeyError(f"No path from {source_node} to {target_node} in pose graph")

        # traverse path, compose associated transformations
        T_source_target: Optional[np.ndarray] = None
        for edge_nodes in path:
            edge = self._normalized_edge_map[self.normalize_node_pair(edge_nodes[0], edge_nodes[1])]

            # determine evaluation direction and perform edge evaluation for requested timestamps + map to required dtype
            if edge.source_node == source_node:
                T_edge = edge.get_T(edge_nodes[0], edge_nodes[1], timestamps_us)
                source_node = edge_nodes[1]
            else:
                T_edge = edge.get_T(edge_nodes[1], edge_nodes[0], timestamps_us)
                source_node = edge_nodes[0]

            if T_source_target is None:
                T_source_target = T_edge
            else:
                T_source_target = T_edge @ T_source_target

        return unpack_optional(T_source_target)
