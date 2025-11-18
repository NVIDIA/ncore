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
import logging
import sys
import time

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Generator, Iterable, List, Optional, Tuple, TypeVar, Union, cast

import numpy as np
import PIL.Image as PILImage

from scipy import interpolate, spatial
from scipy.spatial.transform import Rotation as R
from upath import UPath


if TYPE_CHECKING:
    from _hashlib import HASH as Hash

    import numpy.typing as npt  # type: ignore[import-not-found]


def load_jsonl(jsonl_path: Union[str, Path, UPath]) -> List[dict]:
    """
    Loads a jsonl (json-lines) file (each line corresponds to a serialized json object) - see jsonlines.org

    Args:
        jsonl_path: json-lines file path
    Return:
        object_list: list of parsed objects
    """

    object_list = []
    with UPath(jsonl_path).open("r") as fp:
        for line in fp:
            object_list.append(json.loads(line))

    return object_list


class MD5Hasher:
    """Helper class for MD5 hashing operations on files or full directories."""

    @staticmethod
    def _update_from_file(filename: UPath, hash: "Hash", chunk_size: int) -> "Hash":
        """Update the provided hash object with the contents of the file.

        Reads the file in chunks and updates the hash object incrementally to handle large files efficiently.

        Args:
            filename: Path to the file to hash
            hash: Hash object to update (e.g., hashlib.md5())
            chunk_size: Size of chunks to read from the file

        Returns:
            The updated hash object
        """
        assert filename.is_file()
        with filename.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hash.update(chunk)
        return hash

    @staticmethod
    def _hash_file(filename: UPath, chunk_size: int) -> str:
        """Compute the MD5 hash of a file.

        Args:
            filename: Path to the file to hash
            chunk_size: Size of chunks to read from the file

        Returns:
            Hexadecimal string representation of the file's MD5 hash
        """
        return str(MD5Hasher._update_from_file(filename, hashlib.md5(), chunk_size).hexdigest())

    @staticmethod
    def _update_from_dir(directory: UPath, hash: "Hash", chunk_size: int) -> "Hash":
        """Update the provided hash object with the contents of the directory (recursively).

        Traverses the directory tree in sorted order (case-insensitive) and updates the hash with:
        - Each file/directory name (encoded as bytes)
        - Contents of each file
        - Recursively processes subdirectories

        Args:
            directory: Path to the directory to hash
            hash: Hash object to update (e.g., hashlib.md5())
            chunk_size: Size of chunks to read when processing files

        Returns:
            The updated hash object
        """
        assert directory.is_dir()
        for path in sorted(directory.iterdir(), key=lambda p: str(p).lower()):
            hash.update(path.name.encode())
            if path.is_file():
                hash = MD5Hasher._update_from_file(path, hash, chunk_size)
            elif path.is_dir():
                hash = MD5Hasher._update_from_dir(path, hash, chunk_size)
        return hash

    @staticmethod
    def _hash_dir(directory: UPath, chunk_size: int) -> str:
        """Compute the MD5 hash of a directory (recursively).

        Computes a deterministic hash of the entire directory structure including all files,
        subdirectories, and their names.

        Args:
            directory: Path to the directory to hash
            chunk_size: Size of chunks to read when processing files

        Returns:
            Hexadecimal string representation of the directory's MD5 hash
        """
        return str(MD5Hasher._update_from_dir(directory, hashlib.md5(), chunk_size).hexdigest())

    @staticmethod
    def hash(path: UPath, chunk_size: int = 128 * 2**9) -> str:
        """Compute the MD5 hash of a file or directory.

        Args:
            path: Path to the file or directory to hash
            chunk_size: Size of chunks to read when processing files (default: 128 * 512 bytes)

        Returns:
            Hexadecimal string representation of the MD5 hash

        Raises:
            ValueError: If path is neither a file nor a directory
        """
        if path.is_file():
            return MD5Hasher._hash_file(path, chunk_size)
        elif path.is_dir():
            return MD5Hasher._hash_dir(path, chunk_size)
        else:
            raise ValueError(f"Path '{path}' is neither a file nor a directory")


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

    def interpolate_to_timestamps(self, ts_target, dtype: npt.DTypeLike = np.float32) -> np.ndarray:
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


def log_progress(
    iterable: Iterable[T],
    logger: logging.Logger,
    total: Optional[int] = None,
    label: str = "",
    step_frequency: int = 1,
    level: int = logging.INFO,
    nest_level: int = 0,
) -> Generator[T, None, None]:
    """
    Generator wrapper that logs progress at specified frequency with nesting support.

    Args:
        iterable: The iterable to wrap
        logger: Logger instance to use for logging
        total: Total count (auto-computed if not provided)
        label: Label prefix for log messages
        step_frequency: Log every N steps (1 = every step, 10 = every 10th step)
        level: Logging level (default INFO)
        nest_level: Nesting level for indentation (0 = no indent, 1 = "  ", 2 = "    ", etc.)

    Yields:
        Items from the iterable
    """
    if total is None:
        iterable = list(iterable)
        total = len(iterable)

    indent = "  " * nest_level

    for current, item in enumerate(iterable, 1):
        yield item

        if current % step_frequency == 0 or current == total:
            percent = current / total
            bar = "█" * int(30 * percent) + "-" * (30 - int(30 * percent))
            msg = f"{indent}[{bar}] {current}/{total}"
            if label:
                msg = f"{indent}{label}: [{bar}] {current}/{total}"
            logger.log(level, msg)
