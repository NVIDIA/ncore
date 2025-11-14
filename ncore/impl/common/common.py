# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import annotations

import hashlib
import json
import sys
import time

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional, Tuple, TypeVar, Union, cast

import numpy as np
import PIL.Image as PILImage

from scipy import interpolate, spatial
from scipy.spatial.transform import Rotation as R


def load_jsonl(jsonl_path: Union[str, Path]) -> List[dict]:
    """
    Loads a jsonl (json-lines) file (each line corresponds to a serialized json object) - see jsonlines.org

    Args:
        jsonl_path: json-lines file path
    Return:
        object_list: list of parsed objects
    """

    object_list = []
    with open(jsonl_path, "r") as fp:
        for line in fp:
            object_list.append(json.loads(line))

    return object_list


def md5(path: Path, chunk_size: int = 128 * 2**9) -> str:
    """Compute the MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class PoseInterpolator:
    """
    Interpolates the poses to the desired time stamps. The translation component is interpolated linearly,
    while spherical linear interpolation (SLERP) is used for the rotations. https://en.wikipedia.org/wiki/Slerp

    Args:
        poses (np.array): poses at given timestamps in a se3 representation [n,4,4]
        timestamps (np.array): timestamps of the known poses [n]
        ts_target (np.array): timestamps for which the poses will be interpolated [m]
    Out:
        (np.array): interpolated poses in se3 representation [m,4,4]
    """

    @property
    def poses(self) -> np.ndarray:
        """Returns the original poses used for interpolation"""
        return self._poses

    @property
    def timestamps(self) -> np.ndarray:
        """Returns the timestamps corresponding to the original poses used for interpolation"""
        return self._timestamps

    def __init__(self, poses, timestamps):
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

    def interpolate_to_timestamps(self, ts_target, dtype=np.float32) -> np.ndarray:
        t_interp = self.f_t(ts_target).reshape(-1, 3, 1).astype(dtype)
        R_interp = self.slerp(ts_target).as_matrix().reshape(-1, 3, 3).astype(dtype)

        return np.concatenate(
            (
                np.concatenate([R_interp, t_interp], axis=-1),
                np.tile(self.last_row.astype(dtype=dtype), (R_interp.shape[0], 1, 1)),
            ),
            axis=1,
        )


class MaskImage:
    """
    Image encoding *per-pixel* annotation mask types:

        - dynamic [255, 255, 255] - e.g., dynamic vehicles / objects
        - ego     [0, 0, 255] - pixels corresponding to projections of the ego vehicle

    Properties can be set using binary input images. A pixel can only have a single property assigned.
    Output images are represented using color pallets to reduce memory footprints.
    """

    class MaskType(Enum):
        """Enumerates supported mask types"""

        NONE = 0
        DYNAMIC = 1
        EGO = 2

    # Define colors of mask types
    mask_colors = {MaskType.NONE: [0, 0, 0], MaskType.DYNAMIC: [255, 255, 255], MaskType.EGO: [0, 0, 255]}
    # Initialize color pallet with all color entries (flattening all individual RBG colors into pallet)
    palette = [
        color_component
        for mask_color in [
            # note: ordering is crucial as integer color indices map to mask types
            mask_colors[MaskType.NONE],
            mask_colors[MaskType.DYNAMIC],
            mask_colors[MaskType.EGO],
        ]
        for color_component in mask_color
    ]

    def __init__(self, shape, initial_masks=None):
        """
        Initializes a MaskImage object to a given mask shape with optional initial masks
        Args:
            shape: array shape corresponding to image (height, width)
            initial_masks: if provided, an iterable of [(binary_mask, MaskType), ...] tuples to initialize the mask image with in order
        """
        # initialize empty mask array corresponding to NONE type of appropriate type
        self.mask_array = np.full(shape, MaskImage.MaskType.NONE.value, dtype=np.uint8)

        # apply initial masks if available
        if initial_masks:
            for initial_mask in initial_masks:
                self.set(*initial_mask)

    def set(self, binary_mask, mask_type):
        """
        Sets the mask type of all enabled pixels in the binary_mask to mask_type.
        Args:
            binary_mask: 2D binary array of same shape as underlying image.
            mask_type: the MaskType to set the pixels to
        """
        assert isinstance(binary_mask, np.ndarray), "expecting array as input"
        assert isinstance(mask_type, MaskImage.MaskType), "expecting MaskType as input"
        assert binary_mask.dtype is np.dtype("bool"), "expecting binary array as input"
        assert binary_mask.shape == self.mask_array.shape, (
            f"invalid array resolution, expecting shape {self.mask_array.shape}"
        )

        # set new values for masked pixels
        self.mask_array[binary_mask] = mask_type.value

    def get_image(self) -> PILImage.Image:
        """
        Returns the color-paletted mask image with all mask types set
        Returns:
            mask_image: mask image with all pixel colors set to the corresponding mask types
        """
        # convert mask array to image
        mask_image = PILImage.fromarray(self.mask_array, mode="P")

        # apply color palette
        mask_image.putpalette(self.palette)

        return mask_image


class SimpleTimer:
    """Simple Timer to track runtimes"""

    def __init__(self):
        """Starts timer immediately"""

        self.start()

    def start(self) -> None:
        """(Re-)start the timer"""

        self._start_time = time.perf_counter()

    def elapsed_sec(self, restart: bool = False) -> float:
        """Returns elapsed time (in seconds) since start, optionally restarting the timer"""

        elapsed = time.perf_counter() - self._start_time

        if restart:
            self.start()

        return elapsed


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


def uniform_subdivide_range(
    subdiv_id: int, subdiv_count: int, range_start: int, range_end: int
) -> Tuple[np.ndarray, int]:
    """
    Splits the index range range_start:range_end into (approximately) uniform intervals
    based on the requested number of subvisions.

    Args:
        subdiv_id (int): Subdivision id, with subdiv_id < subdiv_count
        subdiv_count (int): Number of subdivisions to compute
        range_start (int): Full range's start index
        range_end (int): Full range's past-the-end index

    Return:
        local_range array[int]: The subdivided interval's index range
        local_offset int: If local range is nonempty, the offset of the
                         subdivided interval's start from the original start, otherwise
                         -1
    """

    assert subdiv_count > 0 and subdiv_id < subdiv_count, (
        f"Invalid subdivision specification id={subdiv_id} count={subdiv_count}"
    )

    assert range_start >= 0 and range_end >= range_start, (
        f"Range specification {range_start}:{range_end} invalid / not providing *absolute* range"
    )

    # Create full index range and split according to subdiv-count
    split_range = np.array_split(np.arange(range_start, range_end), subdiv_count)

    # Grab local range
    local_range = cast(np.ndarray, split_range[subdiv_id])

    return local_range, local_range[0] - range_start if len(local_range) else -1


@dataclass(**({"slots": True, "frozen": True} if sys.version_info >= (3, 10) else {"frozen": True}))
class HalfClosedInterval:
    """Represents a half closed interval [start, stop) of integers"""

    start: int
    stop: int

    def __post_init__(self) -> None:
        """Makes sure interval is well-defined"""
        assert isinstance(self.start, int)
        assert isinstance(self.stop, int)
        assert self.start <= self.stop

    def __contains__(self, item: int) -> bool:
        """Determines if an item is contained in the interval"""
        return self.start <= item < self.stop

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


# Helper functions for working with optionals
T = TypeVar("T")
U = TypeVar("U")


def unpack_optional(maybe_value: Optional[T], default: Optional[T] = None, msg: Optional[str] = None) -> T:
    """Unpacks the value of an optional or returns a default if provided, otherwise raises a ValueError with custom message (if provided)."""
    if maybe_value is None:
        # Check if we can return a default value instead
        if default is not None:
            return default
        # Not possible to unpack an empty optional and no default is given -> raise ValueError
        raise ValueError(msg or "Can't unpack empty optional")

    # If the optional is not empty, return its value
    return maybe_value


def map_optional(maybe_value: Optional[T], func: Callable[[T], U]) -> Optional[U]:
    """Applies a function `func` to an optional value if it's set, otherwise returns None"""
    if maybe_value is None:
        return None

    return func(maybe_value)
