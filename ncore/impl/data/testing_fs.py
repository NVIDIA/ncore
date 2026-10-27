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

"""Test support: an asynchronous fsspec filesystem serving local files as remote files (test-only)"""

from __future__ import annotations

import asyncio
import os
import threading

from types import SimpleNamespace
from typing import IO, Any, Dict, List, Optional, Tuple, cast

import fsspec
import fsspec.asyn

from typing_extensions import override


class RemoteFileSystem(fsspec.asyn.AsyncFileSystem):
    """An asynchronous filesystem serving local files as remote files (with a request latency), recording the range
    requests of records and the maximum number of requests in flight"""

    protocol = "ncoretestremote"
    async_impl = True
    cachable = False  # (a filesystem per archive path, recording the requests of a test)

    #: The requests of all instances while not None (set it to a list to record them): (kind, path), kind "cat" (range
    #: requests), "info" (metadata), "open" (file objects)
    all_requests: Optional[List[Tuple[str, str]]] = None

    def _record_request(self, kind: str, path: str) -> None:
        if (requests := RemoteFileSystem.all_requests) is not None:
            requests.append((kind, os.path.basename(path.rstrip("/"))))

    def __init__(self, latency: float = 0.0, **kwargs: Any) -> None:
        super().__init__(loop=fsspec.asyn.get_loop(), **kwargs)
        self.latency = latency
        self.requests: List[Tuple[int, int]] = []
        #: All requests by kind: ("cat", start, end) range requests, ("info",) metadata requests, ("open",) file opens
        self.calls: List[Tuple[Any, ...]] = []
        self.file_reads = 0  # (reads of the buffered file objects)
        self.max_in_flight = 0
        self._in_flight = 0
        self._lock = threading.Lock()

    @staticmethod
    def _local(path: str) -> str:
        # (stripped paths of universal_pathlib < 0.3 are relative)
        return "/" + path.lstrip("/")

    @override
    async def _cat_file(
        self, path: str, start: Optional[int] = None, end: Optional[int] = None, **kwargs: Any
    ) -> bytes:
        self._record_request("cat", path)
        with self._lock:
            self.requests.append((start or 0, end or 0))
            self.calls.append(("cat", start, end))
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        await asyncio.sleep(self.latency)
        with self._lock:
            self._in_flight -= 1
        with open(self._local(path), "rb") as f:
            # (negative offsets count from the end of the file, as for HTTP suffix ranges `bytes=-N`)
            size = f.seek(0, os.SEEK_END)
            start_offset, end_offset, _ = slice(start, end).indices(size)
            f.seek(start_offset)
            return f.read(max(0, end_offset - start_offset))

    @override
    async def _info(self, path: str, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(("info",))
        self._record_request("info", path)
        await asyncio.sleep(self.latency)
        local = self._local(path)
        if not os.path.exists(local):
            raise FileNotFoundError(path)
        return {"name": path, "size": os.path.getsize(local), "type": "directory" if os.path.isdir(local) else "file"}

    @override
    def _open(
        self,
        path: str,
        mode: str = "rb",
        block_size: Optional[int] = None,
        autocommit: bool = True,
        cache_options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> IO[bytes]:
        # (the archive's buffered file object, counting its reads)
        self.calls.append(("open",))
        self._record_request("open", path)
        file = open(self._local(path), mode)
        read = file.read

        def counting_read(*args: Any) -> bytes:
            self.file_reads += 1
            return read(*args)

        return cast(IO[bytes], SimpleNamespace(read=counting_read, seek=file.seek, tell=file.tell, close=file.close))


fsspec.register_implementation(RemoteFileSystem.protocol, RemoteFileSystem)
