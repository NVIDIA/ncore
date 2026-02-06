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

from __future__ import annotations

import hashlib
import logging

from typing import TYPE_CHECKING, Callable, Generator, Iterable, Optional, TypeVar

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
