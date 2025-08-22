# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from dataclasses import field, dataclass
from typing import TYPE_CHECKING, Literal, TypeVar, Any, Generic, cast

import dataclasses_json
import numpy as np

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
