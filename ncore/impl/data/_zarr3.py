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

"""zarr-python 3 backend of :mod:`ncore.impl.data.nodes` and :mod:`ncore.impl.data.stores` (only imported if
zarr-python 3 is installed).

zarr-python 3 executes all I/O as coroutines. Its synchronous API submits each call to a global event loop running on
a dedicated I/O thread, and codecs additionally hand off (de)compression to worker threads. For many small accesses
(as typical for reading individual frames) these thread hand-offs dominate. The backend instead runs zarr-python's
coroutines on an event loop owned by the *calling* thread (see :func:`run`), which executes the same zarr-python code
paths without thread hand-offs, and lets multiple reader threads decode data concurrently.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import os
import threading
import warnings

from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Coroutine,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numcodecs
import numpy as np
import zarr
import zarr.api.asynchronous as zarr_async

from zarr.abc.store import ByteRequest, OffsetByteRequest, RangeByteRequest, Store, SuffixByteRequest
from zarr.core.buffer import Buffer, BufferPrototype, default_buffer_prototype
from zarr.core.group import AsyncGroup
from zarr.core.sync import sync
from zarr.storage import LocalStore, StorePath

from ncore.impl.data._backend import (
    V2_CONSOLIDATED_METADATA_KEY,
    Attributes,
    BloscCompression,
    Selection,
    StoreLike,
    ZarrStore,
    attributes_dict,
    is_v2_metadata_key,
)
from ncore.impl.data.encoding import decode_cbor_xz, encode_cbor_xz
from ncore.impl.data.itar import IndexedTarStore


if TYPE_CHECKING:
    import numpy.typing as npt  # type: ignore[import-not-found]


T = TypeVar("T")


# -----------------------------------------------------------------------------
# Execution of coroutines on event loops of the calling threads
# -----------------------------------------------------------------------------


class _InlineExecutor(concurrent.futures.ThreadPoolExecutor):
    """Executor running submitted work immediately in the submitting thread (asyncio requires default executors to be
    thread pools, which start no threads until work is submitted to them)"""

    def submit(self, fn, /, *args, **kwargs):  # type: ignore[override]
        future: concurrent.futures.Future = concurrent.futures.Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as e:  # forwarded to the awaiting coroutine
            future.set_exception(e)
        return future


class _Loop:
    """An event loop of a thread (closed when the thread ends)"""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.loop.set_default_executor(_InlineExecutor())  # (codecs offload (de)compression to the default executor)
        self.pid = os.getpid()

    def __del__(self) -> None:
        # Loops of parent processes aren't closed in forked processes: their selectors (e.g., epoll instances) are
        # shared with the parent, closing them would unregister the parent's file descriptors (e.g., the self-pipe of
        # the parent's loop, which then misses wake-ups by other threads)
        if self.pid == os.getpid() and not self.loop.is_closed() and not self.loop.is_running():
            self.loop.close()


class _ThreadLoop(threading.local):
    state: Optional[_Loop] = None


_thread_loop = _ThreadLoop()


def run(coroutine: Coroutine[Any, Any, T]) -> T:
    """Runs a zarr-python coroutine to completion on an event loop owned by the calling thread.

    The loop runs offloaded work (e.g., of codecs) inline. Processes forked by a thread with a loop create a new loop.
    Within running event loops (e.g., of asynchronous applications), coroutines run on zarr-python's I/O loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        if (state := _thread_loop.state) is None or state.pid != os.getpid():
            state = _thread_loop.state = _Loop()  # (see _Loop.__del__ for loops of parent processes)
        return state.loop.run_until_complete(coroutine)
    return sync(coroutine)


# -----------------------------------------------------------------------------
# zarr-python 3 store of indexed tar archives
# -----------------------------------------------------------------------------


def _byte_range(byte_range: Optional[ByteRequest], size: int) -> Tuple[int, int]:
    """Maps a zarr-python byte request to the (start, end) range of a record of the given size"""
    if byte_range is None:
        return 0, size
    if isinstance(byte_range, RangeByteRequest):
        return min(byte_range.start, size), min(byte_range.end, size)
    if isinstance(byte_range, OffsetByteRequest):
        return min(byte_range.offset, size), size
    if isinstance(byte_range, SuffixByteRequest):
        return max(0, size - byte_range.suffix), size
    raise TypeError(f"Unexpected byte range {byte_range}")


class _TarStore(Store):
    """zarr-python 3 store of the records of an :class:`IndexedTarStore` (synchronous I/O)"""

    supports_writes: bool = True
    supports_deletes: bool = False
    supports_partial_writes: Literal[False] = False
    supports_listing: bool = True

    def __init__(self, store: IndexedTarStore) -> None:
        super().__init__(read_only=store.mode == "r")
        self.store = store
        self._is_open = True

    def __eq__(self, value: object) -> bool:
        return isinstance(value, _TarStore) and self.store == value.store

    def __hash__(self) -> int:
        return hash(self.store)

    def __repr__(self) -> str:
        return f"_TarStore({self.store!r})"

    def with_read_only(self, read_only: bool = False) -> _TarStore:
        if read_only != self.read_only:
            raise NotImplementedError("IndexedTarStore: the access mode of archives can't be changed")
        return self

    def close(self) -> None:
        self.store.close()
        self._is_open = False

    def _get(
        self, key: str, prototype: Optional[BufferPrototype], byte_range: Optional[ByteRequest]
    ) -> Optional[Buffer]:
        try:
            value = self.store[key]
        except KeyError:
            return None
        start, end = _byte_range(byte_range, len(value))
        return (prototype or default_buffer_prototype()).buffer.from_bytes(value[start:end])

    async def get(
        self, key: str, prototype: BufferPrototype, byte_range: Optional[ByteRequest] = None
    ) -> Optional[Buffer]:
        return self._get(key, prototype, byte_range)

    async def get_partial_values(
        self, prototype: BufferPrototype, key_ranges: Iterable[Tuple[str, Optional[ByteRequest]]]
    ) -> List[Optional[Buffer]]:
        return [self._get(key, prototype, byte_range) for key, byte_range in key_ranges]

    async def exists(self, key: str) -> bool:
        return key in self.store

    async def set(self, key: str, value: Buffer) -> None:
        self._check_writable()
        self.store[key] = value.to_bytes()

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        if key not in self.store:
            await self.set(key, value)

    async def delete(self, key: str) -> None:
        """No-op for records that don't exist (zarr-python deletes them when creating nodes)"""
        self._check_writable()
        if key in self.store:
            raise NotImplementedError("IndexedTarStore: records can't be deleted")

    async def delete_dir(self, prefix: str) -> None:
        """No-op for prefixes without records (zarr-python deletes them when creating nodes)"""
        self._check_writable()
        prefix = prefix if prefix == "" or prefix.endswith("/") else prefix + "/"
        if any(key.startswith(prefix) for key in self.store.keys()):
            raise NotImplementedError("IndexedTarStore: records can't be deleted")

    async def list(self) -> AsyncIterator[str]:
        for key in self.store.keys():
            yield key

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        for key in self.store.keys():
            if key.startswith(prefix):
                yield key

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        prefix = prefix.rstrip("/")
        prefix = prefix + "/" if prefix else ""
        children = {key[len(prefix) :].split("/", 1)[0]: None for key in self.store.keys() if key.startswith(prefix)}
        for child in children:
            yield child


def _zarr_store(store: StoreLike) -> Store:
    """The zarr-python 3 store of a store (wraps indexed tar stores)"""
    return _TarStore(store) if isinstance(store, IndexedTarStore) else cast(Store, store)


def _keys(store: Store) -> List[str]:
    async def keys() -> List[str]:
        return [key async for key in store.list()]

    return run(keys())


def _get(store: Store, key: str) -> Optional[bytes]:
    value = run(store.get(key, prototype=default_buffer_prototype()))
    return None if value is None else value.to_bytes()


def _parent_grouped(metadata: dict) -> dict:
    """Orders consolidated node metadata (by node metadata key) by node depth, with the members of each group adjacent.

    zarr-python < 3.4 nests flat consolidated metadata assuming this order (zarr-python#4226: the members of a group
    are attached to the first run of keys of the group only), which isn't the order of hierarchies created in arbitrary
    order (e.g., by multiple component writers).
    """

    def order(key: str) -> Tuple[int, str, str]:
        parent, _, name = key.rsplit("/", 1)[0].rpartition("/") if "/" in key else ("", "", "")
        return key.count("/"), parent, name

    return {key: metadata[key] for key in sorted(metadata, key=order)}


@contextlib.contextmanager
def _ignored_warnings() -> Iterator[None]:
    with warnings.catch_warnings():
        # compressed consolidated metadata records are not zarr nodes
        warnings.filterwarnings("ignore", message="Object at .* is not recognized as a component")
        yield


# -----------------------------------------------------------------------------
# Backend
# -----------------------------------------------------------------------------


class Backend:
    """zarr-python 3 implementation of the version specific operations of nodes and stores"""

    supported_zarr_formats: Tuple[int, ...] = (2,)

    # -- nodes -----------------------------------------------------------------

    def zarr_format(self, node: Union[zarr.Group, zarr.Array]) -> int:
        return node.metadata.zarr_format

    def open_member(self, group: zarr.Group, path: str) -> Union[zarr.Group, zarr.Array]:
        node = run(group._async_group.getitem(path))
        return zarr.Group(node) if isinstance(node, AsyncGroup) else zarr.Array(node)

    def member_names(self, group: zarr.Group) -> List[str]:
        with _ignored_warnings():  # (members of groups with consolidated metadata are listed without I/O)
            return list(group.keys())

    def has_member(self, group: zarr.Group, name: str) -> bool:
        return name in self.member_names(group)

    def member_groups(self, group: zarr.Group) -> List[Tuple[str, zarr.Group]]:
        with _ignored_warnings():
            return list(group.groups())

    def create_group(self, group: zarr.Group, name: str, attributes: Optional[Attributes]) -> zarr.Group:
        return group.create_group(name, attributes=None if attributes is None else attributes_dict(attributes))

    def require_group(self, group: zarr.Group, name: str) -> zarr.Group:
        return group.require_group(name)

    def create_array(
        self,
        group: zarr.Group,
        name: str,
        data: npt.NDArray[np.generic],
        chunks: Union[Tuple[int, ...], Literal["auto"]],
        compression: Optional[BloscCompression],
        attributes: Optional[Attributes],
    ) -> zarr.Array:
        return group.create_array(
            name,
            data=data,
            chunks=chunks,
            compressors=None if compression is None else compression.to_numcodecs(),
            attributes=None if attributes is None else attributes_dict(attributes),
            # (store chunks equal to the fill value too, as zarr-python 2: identical records of both versions)
            config={"write_empty_chunks": True},
        )

    def read(self, array: zarr.Array, selection: Selection) -> object:
        return run(array.async_array.getitem(selection))

    def compression(self, array: zarr.Array) -> Optional[BloscCompression]:
        if not (compressors := array.compressors):
            return None
        if len(compressors) != 1 or not isinstance(compressor := compressors[0], numcodecs.Blosc):
            raise ValueError(f"Unsupported compressors {compressors}")
        return BloscCompression.from_numcodecs(compressor)

    # -- stores ----------------------------------------------------------------

    def open_directory_store(self, path: str) -> ZarrStore:
        return cast(ZarrStore, LocalStore(path))

    def create_root_group(self, store: StoreLike, zarr_format: Literal[2, 3], attributes: Attributes) -> zarr.Group:
        node = run(
            zarr_async.create_group(
                store=_zarr_store(store), zarr_format=zarr_format, attributes=attributes_dict(attributes)
            )
        )
        return zarr.Group(node)

    def open_store(self, store: StoreLike, consolidated: bool) -> zarr.Group:
        zarr_store = _zarr_store(store)
        if not consolidated:
            return zarr.Group(
                run(zarr_async.open_group(store=zarr_store, zarr_format=2, mode="r", use_consolidated=False))
            )

        if (compressed := _get(zarr_store, V2_CONSOLIDATED_METADATA_KEY)) is None:
            raise KeyError(V2_CONSOLIDATED_METADATA_KEY)
        document = decode_cbor_xz(compressed)
        if (consolidated_format := document.get("zarr_consolidated_format")) != 1:
            raise ValueError(f"Unsupported zarr consolidated metadata format: {consolidated_format}")
        metadata = document["metadata"]

        # the root group of the hierarchy, with the node metadata of all members (no node metadata reads)
        prototype = default_buffer_prototype()
        root = AsyncGroup._from_bytes_v2(
            StorePath(zarr_store, ""),
            prototype.buffer.from_bytes(json.dumps(metadata[".zgroup"]).encode()),
            prototype.buffer.from_bytes(json.dumps(metadata.get(".zattrs", {})).encode()),
            prototype.buffer.from_bytes(json.dumps({**document, "metadata": _parent_grouped(metadata)}).encode()),
        )
        return zarr.Group(root)

    def consolidate_store(self, store: StoreLike) -> None:
        zarr_store = _zarr_store(store)
        metadata = {}
        for key in _keys(zarr_store):
            if is_v2_metadata_key(key) and (value := _get(zarr_store, key)) is not None:
                metadata[key] = json.loads(value)
        value = encode_cbor_xz({"zarr_consolidated_format": 1, "metadata": metadata})
        run(zarr_store.set(V2_CONSOLIDATED_METADATA_KEY, default_buffer_prototype().buffer.from_bytes(value)))

    def group_store(self, group: zarr.Group) -> StoreLike:
        store = group.store
        return store.store if isinstance(store, _TarStore) else cast(ZarrStore, store)


backend = Backend()
