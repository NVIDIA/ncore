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
import logging
import sys

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Generator, Iterable, List, Optional, TypeVar, Union

import numpy as np

from upath import UPath


if TYPE_CHECKING:
    from _hashlib import HASH as Hash


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
