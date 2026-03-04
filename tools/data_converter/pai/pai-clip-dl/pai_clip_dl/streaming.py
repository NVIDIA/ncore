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

"""Streaming access to individual files inside remote zip archives on HuggingFace.

Uses ``remotezip`` to read only the zip central directory and then fetch
individual entries via HTTP range requests — without downloading the entire
archive.
"""

from __future__ import annotations

import io
import logging

from typing import BinaryIO

import remotezip

from pai_clip_dl.index import FeatureSpec
from pai_clip_dl.remote import HFRemote


log = logging.getLogger(__name__)


class StreamingZipAccess:
    """Transparent access to files inside a remote zip archive.

    The zip's central directory is fetched on open (a few KB), then
    individual entries are fetched on demand via HTTP range requests.

    Usage::

        remote = HFRemote(config)
        sza = StreamingZipAccess(remote, "camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_0000.zip")
        sza.open()

        # List all entries
        print(sza.namelist())

        # Read a specific file
        data = sza.read("25cd4769-....mp4")

        # Get a seekable BytesIO handle
        handle = sza.open_stream("25cd4769-....timestamps.parquet")
        df = pd.read_parquet(handle)

        sza.close()

    Or as a context manager::

        with StreamingZipAccess(remote, repo_path) as sza:
            data = sza.read("somefile.parquet")

    Higher-level helpers accept a :class:`FeatureSpec` and clip_id to
    resolve the correct repo path and filenames automatically.
    """

    def __init__(self, remote: HFRemote, repo_path: str) -> None:
        self._remote = remote
        self._repo_path = repo_path
        self._rz: remotezip.RemoteZip | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> StreamingZipAccess:
        """Resolve the CDN URL and open the remote zip (reads central directory)."""
        if self._rz is not None:
            return self
        cdn_url = self._remote.resolve_cdn_url(self._repo_path)
        log.debug("Opening remote zip: %s", self._repo_path)
        self._rz = remotezip.RemoteZip(cdn_url, support_suffix_range=False)
        return self

    def close(self) -> None:
        if self._rz is not None:
            self._rz.close()
            self._rz = None

    def __enter__(self) -> StreamingZipAccess:
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _ensure_open(self) -> remotezip.RemoteZip:
        if self._rz is None:
            self.open()
        assert self._rz is not None
        return self._rz

    # ------------------------------------------------------------------
    # Zip operations
    # ------------------------------------------------------------------

    def namelist(self) -> list[str]:
        """List all entry names in the zip archive."""
        return self._ensure_open().namelist()

    def read(self, filename: str) -> bytes:
        """Read and return the full contents of a zip entry."""
        rz = self._ensure_open()
        log.debug("Reading %s from %s", filename, self._repo_path)
        return rz.read(filename)

    def open_stream(self, filename: str) -> BinaryIO:
        """Return a seekable BytesIO handle over a zip entry's contents.

        This fetches the entry data via a range request and wraps it in a
        ``BytesIO`` so it can be passed to consumers that expect a file-like
        object (e.g. ``pd.read_parquet(handle)``).
        """
        data = self.read(filename)
        return io.BytesIO(data)

    def infolist(self) -> list:
        """Return ZipInfo objects for all entries."""
        return self._ensure_open().infolist()

    # ------------------------------------------------------------------
    # Clip-level helpers
    # ------------------------------------------------------------------

    def get_clip_entries(self, clip_id: str) -> list[str]:
        """Return all zip entry names that belong to the given clip_id."""
        prefix = f"{clip_id}."
        return [n for n in self.namelist() if n.startswith(prefix)]

    def read_clip_files(self, clip_id: str) -> dict[str, bytes]:
        """Read all files belonging to a clip, returning {filename: bytes}."""
        entries = self.get_clip_entries(clip_id)
        return {name: self.read(name) for name in entries}

    def stream_clip_files(self, clip_id: str) -> dict[str, BinaryIO]:
        """Return seekable BytesIO handles for all files belonging to a clip."""
        entries = self.get_clip_entries(clip_id)
        return {name: self.open_stream(name) for name in entries}

    # ------------------------------------------------------------------
    # Factory from FeatureSpec
    # ------------------------------------------------------------------

    @classmethod
    def from_feature(
        cls,
        remote: HFRemote,
        feature: FeatureSpec,
        chunk_id: int,
    ) -> StreamingZipAccess:
        """Create a StreamingZipAccess for a specific feature and chunk.

        Only valid for zip-based features.
        """
        if not feature.is_zip:
            raise ValueError(
                f"Feature {feature.name!r} is parquet-based, not zip-based. Use direct parquet download instead."
            )
        repo_path = feature.chunk_path(chunk_id)
        return cls(remote, repo_path)
