# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Iterable, List, Literal, TypeVar, cast

import dataclasses_json
import numpy as np

from upath import UPath


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
        return np.array(*args, **kwargs).astype(datatype)

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


def evaluate_file_pattern(pattern: str, skip_suffixes: Iterable[str] = ()) -> List[str]:
    """Given a file-pattern returns a list of matching and existing files

    Supported patterns (mutually exclusive):
    - integer-ranges: '/some/path/file-[1-3]' will be expanded to [/some/path/file-1, /some/path/file-2, /some/path/file-3]

    """

    pattern_basepath = UPath(pattern).parent
    pattern_name = UPath(pattern).name

    evaluated_name_patterns = []

    # expand integer ranges like '[1-13]'
    if range_match := re.search(r"\[(\d+)-(\d+)\]", pattern_name):
        low = int(range_match.group(1))
        high = int(range_match.group(2))

        for i in range(low, high + 1):
            evaluated_name_patterns.append(pattern_name.replace(f"[{low}-{high}]", str(i) + "-"))
    else:
        evaluated_name_patterns.append(pattern_name)

    matches: set[UPath] = set()
    for evaluated_pattern in evaluated_name_patterns:
        for candidate in pattern_basepath.iterdir():
            if candidate.name.startswith(evaluated_pattern):
                skip = False
                for skip_suffix in skip_suffixes:
                    if str(candidate).endswith(skip_suffix):
                        skip = True
                        break
                if not skip:
                    matches.add(candidate)

    return [str(match) for match in list(matches)]


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
