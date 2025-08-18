# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from dataclasses import field, dataclass
from typing import TYPE_CHECKING, Literal, NamedTuple, TypeVar, Any, Generic, cast

import dataclasses_json
import numpy as np

from ncore.impl.common.common import PoseInterpolator
from ncore.impl.common.transformations import se3_inverse, transform_point_cloud

if TYPE_CHECKING:
    import numpy.typing as npt  # type: ignore[import-not-found]

## Constants
INDEX_DIGITS = 6  # the number of integer digits to pad counters in output filenames


## Types
@dataclass
class FOV:
    """Represents a field-of-view with start and span in radians"""

    start_rad: float  #: Start angle of the field-of-view in radians
    span_rad: float  #: Span of the valid field-of-view region in radians in [0, 2π]
    direction: Literal[
        "cw", "ccw"
    ]  #: Direction of the valid field-of-view region, either clockwise or counter-clockwise


## Functions
def padded_index_string(index: int, index_digits=INDEX_DIGITS) -> str:
    """Pads an integer with leading zeros to a fixed number of digits"""
    return str(index).zfill(index_digits)


def closest_index_sorted(sorted_array: np.ndarray, value: int) -> int:
    """Returns the index of the closest value within a *sorted* array relative to a query value.

    Note: we are *not* checking that the input is sorted
    """
    if not len(sorted_array):
        raise ValueError("input array is empty")

    idx = int(np.searchsorted(sorted_array, value, side="left"))

    if idx > 0:
        if idx == len(sorted_array):
            return idx - 1
        if abs(value - sorted_array[idx - 1]) < abs(sorted_array[idx] - value):
            return idx - 1

    return idx


def numpy_array_field(datatype: "npt.DTypeLike", default=None):
    """Provides encoder / decoder functionality for numpy arrays into field types compatible with dataclass-JSON"""

    def decoder(*args, **kwargs):
        return np.array(*args, dtype=datatype, **kwargs)

    metadata = dataclasses_json.config(encoder=np.ndarray.tolist, decoder=decoder)

    if default is not None:
        return field(default_factory=lambda: default, metadata=metadata)
    else:
        return field(default=None, metadata=metadata)


def enum_field(enum_class, default=None):
    """Provides encoder / decoder functionality for enum types into field types compatible with dataclass-JSON"""

    def encoder(variant):
        """encode enum as name's string representation. This way values in JSON are "human-readable"""
        return variant.name

    def decoder(variant):
        """load enum variant from name's string to value map of the enumeration type"""
        return enum_class.__members__[variant]

    return field(default=default, metadata=dataclasses_json.config(encoder=encoder, decoder=decoder))


# A generic type supporting basic artithmetic operations like +, -, *, /, etc. - in particular implemented by float, torch.Tensor, np.ndarray, etc.
# Used here to not depend on torch.Tensor in the public data API
TensorLike = TypeVar("TensorLike", bound=Any)


@dataclass
class RelativeAngleResult(Generic[TensorLike]):
    relative_angle_rad: TensorLike
    wrap_around_flag: TensorLike


def relative_angle(
    ref_angle_rad: float, angle_rad: TensorLike, direction: Literal["cw", "ccw"]
) -> "RelativeAngleResult[TensorLike]":
    """
    Compute the relative angle from ref_angle_rad to angle_rad in the specified direction

    Args:
        ref_angle_rad: reference angle in radians [float]
        angle_rad: tensor of angles to compute relative angles for, in radians
        direction: If "cw", measure clockwise; if "ccw", measure counter-clockwise
    Returns:
        A RelativeAngleResult containing:
        - relative_angle: Tensor of relative angles [same dimension as 'angle_rad', always positive in range [0, 2π)]
        - wrap_around_flag: Tensor of flags whether the relative angle computation required a wrap-around at multiples of 2π
    """

    two_pi = 2 * np.pi

    # Check for wrap-around condition
    wrap_around_flag = abs(angle_rad - ref_angle_rad) >= two_pi

    # Project both angles to [0, 2π)
    ref_angle_rad = ref_angle_rad % two_pi
    angle_rad = angle_rad % two_pi

    if direction == "cw":
        # Clockwise: going from ref to angle in CW direction
        diff_angle = ref_angle_rad - angle_rad
    elif direction == "ccw":
        # Counter-clockwise: going from ref to angle in CCW direction
        diff_angle = angle_rad - ref_angle_rad
    else:
        raise ValueError(f"Invalid spinning direction: {direction}")

    return RelativeAngleResult(
        relative_angle_rad=cast(TensorLike, diff_angle % two_pi), wrap_around_flag=wrap_around_flag
    )


def undo_motion_compensation(
    xyz: np.ndarray, T_sensor_end_sensor_start: np.ndarray, timestamps_startend_us: list, timestamp_us: np.ndarray
) -> np.ndarray:
    """
    undo motion-compensation to bring ray's into time-dependent sensor-frame

    Args:
        xyz (np.array): points from the sensor space [n,3]
        T_sensor_end_sensor_start (np.array): relative pose from end-of-frame to start-of-frame in sensor space [4,4]
        timestamps_startend_us (list): contains [start timestamp, end timestamp]
        timestamp_us (np.array): recoding target per-point timestamps [n]
    Out:
        (np.array): points after undo motion-compensation[n,3]
    """

    pose_interpolator = PoseInterpolator(
        np.stack([T_sensor_end_sensor_start, np.eye(4, dtype=np.float32)]),
        timestamps_startend_us,
    )

    # Note: this interpolation will fail if the point's timestamps are outside of the frame's start/end times - issue dedicated error in that case
    assert (timestamps_startend_us[0] <= timestamp_us).all() and (timestamp_us <= timestamps_startend_us[1]).all(), (
        f"{undo_motion_compensation}: Lidar point timestamps out of frame timestamp bounds - this is an inconsistency in the dataset's internal data and needs to be fixed at dataset creation time"
    )
    T_sensor_end_sensor_pointtime = pose_interpolator.interpolate_to_timestamps(timestamp_us)

    xyz = transform_point_cloud(xyz[:, np.newaxis, :], T_sensor_end_sensor_pointtime).squeeze(1)

    return xyz


def motion_compensation(
    xyz: np.ndarray,
    T_sensor_rig: np.ndarray,
    T_rig_worlds: np.ndarray,
    frame_start_end_timestamps_us: list,
    timestamps_us: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        xyz(np.ndarray): points before motion compensation, [n,3]
        T_sensor_rig(np.ndarray): sensor extrinsics, [4,4]
        T_rig_worlds(np.ndarray): ego poses at frame start and end timestamps, [2,4,4]
        frame_start_end_timestamps_us(list): frame start and end timestamps, [2]
        timestamps_us(np.ndarray): timestamps of points, [n]
    Returns:
        xyz_s(np.ndarray): points start after motion compensation, [n,3]
        xyz_e(np.ndarray): points end after motion compensation, [n,3]
    """

    frame_start_end_timestamps_us = np.array(frame_start_end_timestamps_us)
    pose_interpolator = PoseInterpolator(T_rig_worlds, frame_start_end_timestamps_us)

    # Interpolate egomotion at frame end timestamp for sensor reference pose at end-of-spin time
    T_world_sensorRef = se3_inverse(T_rig_worlds[1] @ T_sensor_rig)

    # Determine unique timestamps to only perform actually required pose interpolations (a lot of points share the same timestamp)
    timestamp_unique, unique_timestamp_reverse_idxs = np.unique(timestamps_us, return_inverse=True)

    # Lidar frame poses for each point (will throw in case invalid timestamps are loaded) expressed in the reference sensor's frame
    # sensor_sensorRef = world_sensorRef * sensor_world = world_sensorRef * (rig_world * sensor_rig)
    T_sensor_sensorRef_unique = (
        T_world_sensorRef @ pose_interpolator.interpolate_to_timestamps(timestamp_unique) @ T_sensor_rig
    )

    # Pick sensor positions (in end-of-spin reference pose) for each start point (blow up to original potentially non-unique timestamp range)
    xyz_s = T_sensor_sensorRef_unique[unique_timestamp_reverse_idxs, :3, -1]  # N x 3

    xyz_e = (T_sensor_sensorRef_unique[unique_timestamp_reverse_idxs, :3, :3] @ xyz[:, :, None]).squeeze(
        -1
    ) + xyz_s  # N x 3
    return xyz_s, xyz_e
