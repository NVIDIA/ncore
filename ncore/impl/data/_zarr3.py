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

Arrays of indexed tar archives are read by decompressing their chunks directly into the output array (see
:func:`_direct_read`).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import os
import sys
import threading
import warnings

from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    NamedTuple,
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

from zarr.abc.codec import Codec
from zarr.abc.store import ByteRequest, OffsetByteRequest, RangeByteRequest, Store, SuffixByteRequest
from zarr.codecs import BloscCodec
from zarr.core.buffer import Buffer, BufferPrototype, default_buffer_prototype
from zarr.core.group import AsyncGroup
from zarr.core.indexing import BasicIndexer
from zarr.core.sync import sync
from zarr.storage import LocalStore, StorePath

from ncore.impl.data._backend import (
    V2_CONSOLIDATED_METADATA_KEY,
    V3_CONSOLIDATED_METADATA_KEY,
    Attributes,
    BloscCompression,
    Selection,
    StoreLike,
    ZarrStore,
    attributes_dict,
    is_v2_metadata_key,
)
from ncore.impl.data.encoding import decode_cbor_xz, decode_cbor_zlib, encode_cbor_xz, encode_cbor_zlib
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


def _on_own_loop() -> bool:
    """True if running on the event loop of :func:`run` of the calling thread (blocking work runs inline), False on
    other event loops (blocking work is offloaded to threads, to not block them)"""
    state = _thread_loop.state
    return state is not None and state.pid == os.getpid() and asyncio.get_running_loop() is state.loop


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


def _parent_grouped(metadata: dict, node_path: Callable[[str], str]) -> dict:
    """Orders consolidated node metadata by node depth, with the members of each group adjacent.

    zarr-python < 3.4 nests flat consolidated metadata assuming this order (zarr-python#4226: the members of a group
    are attached to the first run of keys of the group only), which isn't the order of hierarchies created in arbitrary
    order (e.g., by multiple component writers). Keys are node metadata keys (zarr format v2) or node paths (zarr format
    v3), ``node_path`` maps them to node paths.
    """

    def order(key: str) -> Tuple[int, str, str]:
        parent, _, name = (path := node_path(key)).rpartition("/")
        return path.count("/"), parent, name

    return {key: metadata[key] for key in sorted(metadata, key=order)}


def _v2_node_path(key: str) -> str:
    return key.rpartition("/")[0]


def _v3_node_documents(store: Store) -> Dict[str, object]:
    """The node metadata documents of all members of a zarr format v3 hierarchy, by node path"""
    documents: Dict[str, object] = {}
    for key in _keys(store):
        if key.endswith("/zarr.json") and (value := _get(store, key)) is not None:
            documents[key[: -len("/zarr.json")]] = json.loads(value)
    return documents


@contextlib.contextmanager
def _ignored_warnings() -> Iterator[None]:
    with warnings.catch_warnings():
        # compressed consolidated metadata records are not zarr nodes
        warnings.filterwarnings("ignore", message="Object at .* is not recognized as a component")
        yield


# -----------------------------------------------------------------------------
# Direct chunk reads of indexed tar archives
# -----------------------------------------------------------------------------

#: Environment variable disabling direct reads (all reads via zarr-python's codec pipeline), e.g., to compare against
#: or to benefit from zarr-python improvements
DIRECT_READS_ENV = "NCORE_ZARR3_DIRECT_READS"

#: Minimum number of decompressed bytes per chunk to decode the chunks of a read concurrently (smaller chunks are
#: cheaper to decode than to hand off to a thread pool)
_PARALLEL_CHUNK_BYTES = 2 << 20

#: Maximum number of concurrent reads (e.g., of multiple reader threads) decoding their chunks concurrently, further
#: concurrent reads decode their chunks sequentially (the reads themselves already utilize the cores)
_PARALLEL_READS = 2


class _ParallelReads:
    """Bounds the number of reads decoding their chunks concurrently (per process)"""

    def __init__(self) -> None:
        self.semaphore = threading.BoundedSemaphore(_PARALLEL_READS)
        self.executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self.lock = threading.Lock()
        self.pid = os.getpid()

    def pool(self) -> concurrent.futures.ThreadPoolExecutor:
        with self.lock:
            if self.executor is None:
                self.executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(8, os.cpu_count() or 1), thread_name_prefix="ncore_zarr3_chunks"
                )
            return self.executor


_parallel_reads = _ParallelReads()


def _parallel() -> _ParallelReads:
    """The parallel reads state of this process (re-created in forked processes, whose locks / threads are unusable)"""
    global _parallel_reads
    if _parallel_reads.pid != os.getpid():
        _parallel_reads = _ParallelReads()
    return _parallel_reads


class _DirectRead(NamedTuple):
    """How to read the chunks of an array directly (see :func:`_direct_read`)"""

    store: IndexedTarStore
    path: str
    dtype: np.dtype
    chunks: Tuple[int, ...]
    chunk_key: Callable[[Tuple[int, ...]], str]
    compressor: Optional[numcodecs.Blosc]
    fill_value: object


def _direct_reads_enabled() -> bool:
    return os.environ.get(DIRECT_READS_ENV, "1").strip().lower() not in ("0", "false", "no", "off")


def _direct_read(array: zarr.Array) -> Optional[_DirectRead]:
    """How to read the chunks of an array directly (None if the array is read via zarr-python).

    Supported are arrays of indexed tar archives opened for reading with a regular chunk grid, C order, numeric /
    boolean data of native byte order or fixed-length bytes, no filters, and no compression or a single blosc compressor
    (all arrays written by ncore). zarr-python determines the chunks and their selections (its indexers), but the
    chunks are read and decompressed directly into the output array, instead of via zarr-python's codec pipeline, which
    decodes each chunk into a temporary buffer and copies it (see zarr-python#2904).

    Planned per read (sub-microsecond), so nothing outlives the array.
    """
    store_path = array.async_array.store_path
    if not isinstance(store := store_path.store, _TarStore) or store.store.mode != "r":
        return None

    metadata: Any = array.metadata
    dtype = np.dtype(metadata.dtype.to_native_dtype())
    if dtype.hasobject or not dtype.isnative or dtype.kind not in "biufS":
        return None
    if metadata.zarr_format == 2:
        if metadata.order != "C" or metadata.filters:
            return None
        compressor = metadata.compressor
        chunks = tuple(metadata.chunks)
    else:
        codecs = list(metadata.codecs)
        if not codecs or type(codecs[0]).__name__ != "BytesCodec":
            return None
        if dtype.itemsize > 1 and str(getattr(codecs[0].endian, "value", codecs[0].endian)) != sys.byteorder:
            return None
        if len(codecs) == 1:
            compressor = None
        elif len(codecs) == 2 and isinstance(codecs[1], BloscCodec):
            compressor = codecs[1]._blosc_codec
        else:
            return None
        if not hasattr(metadata.chunk_grid, "chunk_shape"):  # (regular chunk grids only)
            return None
        chunks = tuple(metadata.chunk_grid.chunk_shape)
    if compressor is not None and not isinstance(compressor, numcodecs.Blosc):
        return None
    return _DirectRead(
        store.store, store_path.path, dtype, chunks, metadata.encode_chunk_key, compressor, metadata.fill_value
    )


def _is_basic_selection(selection: Selection) -> bool:
    """True for selections of (tuples of) integers, slices with unit step, and Ellipsis"""
    for item in selection if isinstance(selection, tuple) else (selection,):
        if item is Ellipsis or (isinstance(item, (int, np.integer)) and not isinstance(item, (bool, np.bool_))):
            continue
        if isinstance(item, slice) and item.step in (None, 1):
            continue
        return False
    return True


def _chunk_grid(array: zarr.Array) -> Any:
    """zarr-python's chunk grid of an array (an array attribute since zarr-python 3.2, metadata before)"""
    if (grid := getattr(array.async_array, "_chunk_grid", None)) is not None:
        return grid
    return cast(Any, array.metadata).chunk_grid


class _DirectReadOp:
    """A direct read of a basic selection of an array: the chunks to read (see :meth:`keys`), and their decoding into
    the output array (see :meth:`decode`)"""

    def __init__(self, plan: _DirectRead, array: zarr.Array, selection: Selection) -> None:
        self.plan = plan
        self.shape = tuple(array.metadata.shape)
        indexer = BasicIndexer(selection, self.shape, _chunk_grid(array))
        self.out_shape = tuple(indexer.shape)
        self.projections: List[Any] = list(indexer) if all(indexer.shape) else []
        self.keys = [self._key(projection) for projection in self.projections]

    def _key(self, projection: Any) -> str:
        key = self.plan.chunk_key(projection.chunk_coords)
        return f"{self.plan.path}/{key}" if self.plan.path else key

    @property
    def is_single_raw_chunk(self) -> bool:
        """True for selections of complete uncompressed single-chunk arrays (e.g., encoded images), whose chunk bytes
        are the result"""
        plan = self.plan
        return (
            plan.compressor is None
            and len(self.projections) == 1
            and self.projections[0].is_complete_chunk
            and plan.chunks == self.shape == self.out_shape
        )

    @property
    def decodes_concurrently(self) -> bool:
        """True for reads of multiple large chunks (see :data:`_PARALLEL_CHUNK_BYTES`)"""
        chunk_bytes = self.plan.dtype.itemsize * int(np.prod(self.plan.chunks))
        return len(self.projections) > 1 and chunk_bytes >= _PARALLEL_CHUNK_BYTES

    def decode(self, raws: List[Optional[bytes]], fetch: Optional[Callable[[str], Optional[bytes]]] = None) -> Any:
        """Decodes the chunks (None: not stored, read as the fill value) into the output array. With ``fetch``, chunks
        are read by the decoding threads (raws are ignored)"""
        plan = self.plan
        if self.is_single_raw_chunk and (raw := raws[0] if fetch is None else fetch(self.keys[0])) is not None:
            # (bytearray: results are writeable, as results of reads via zarr-python)
            return np.frombuffer(bytearray(raw), dtype=plan.dtype).reshape(self.shape)

        out = np.empty(self.out_shape, dtype=plan.dtype)

        def decode_chunk(index: int) -> None:
            projection = self.projections[index]
            raw = raws[index] if fetch is None else fetch(self.keys[index])
            target = out[projection.out_selection] if projection.out_selection != () else out
            if raw is None:  # chunk not stored (fill value)
                target[...] = np.asarray(plan.fill_value, dtype=plan.dtype) if plan.fill_value is not None else 0
                return
            if (
                plan.compressor is not None
                and projection.is_complete_chunk
                and target.shape == plan.chunks
                and target.flags.c_contiguous
            ):
                plan.compressor.decode(raw, out=target)
                return
            decoded = plan.compressor.decode(raw) if plan.compressor is not None else raw
            selected = np.frombuffer(decoded, dtype=plan.dtype).reshape(plan.chunks)[projection.chunk_selection]
            target[...] = selected.reshape(target.shape)  # (integer selections drop axes)

        parallel = _parallel()
        if self.decodes_concurrently and parallel.semaphore.acquire(blocking=False):
            try:
                for future in [parallel.pool().submit(decode_chunk, i) for i in range(len(self.projections))]:
                    future.result()
            finally:
                parallel.semaphore.release()
        else:
            for index in range(len(self.projections)):
                decode_chunk(index)
        return out  # (0-d arrays for 0-d results, see nodes.Array.read)


def _get_record(store: IndexedTarStore, key: str) -> Optional[bytes]:
    try:
        return store[key]
    except KeyError:
        return None


async def _get_record_async(store: IndexedTarStore, key: str) -> Optional[bytes]:
    try:
        return await store.read_async(key)
    except KeyError:
        return None


def _read_direct(plan: _DirectRead, array: zarr.Array, selection: Selection) -> Any:
    """Reads a basic selection of an array, decompressing its chunks directly into the output array.

    Reads of multiple large chunks (see :data:`_PARALLEL_CHUNK_BYTES`) read and decode their chunks concurrently on a
    thread pool (positional reads and blosc decompression release the GIL), unless :data:`_PARALLEL_READS` reads do so
    already. All other reads run on the calling thread.
    """
    op = _DirectReadOp(plan, array, selection)
    return op.decode([], fetch=lambda key: _get_record(plan.store, key))


async def _read_direct_async(plan: _DirectRead, array: zarr.Array, selection: Selection) -> Any:
    """Asynchronous :func:`_read_direct`: chunks of remote archives are read via concurrent asynchronous range
    requests. Blocking work (local reads, decompression) runs on the calling thread on the event loop of :func:`run`,
    on worker threads on other event loops (e.g., of asynchronous applications), so they aren't blocked."""
    op = _DirectReadOp(plan, array, selection)
    if plan.store.is_remote:
        raws = list(await asyncio.gather(*(_get_record_async(plan.store, key) for key in op.keys)))
        if _on_own_loop():
            return op.decode(raws)
        return await asyncio.get_running_loop().run_in_executor(_read_executor(), op.decode, raws)
    if _on_own_loop():
        return op.decode([], fetch=lambda key: _get_record(plan.store, key))
    return await asyncio.get_running_loop().run_in_executor(
        _read_executor(), lambda: op.decode([], fetch=lambda key: _get_record(plan.store, key))
    )


def _read_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Executor of blocking parts of reads awaited on event loops other than the one of :func:`run` (per process)"""
    global _read_executor_state
    if _read_executor_state is None or _read_executor_state[0] != os.getpid():
        _read_executor_state = (
            os.getpid(),
            concurrent.futures.ThreadPoolExecutor(
                max_workers=min(32, (os.cpu_count() or 1) + 4), thread_name_prefix="ncore_zarr3_reads"
            ),
        )
    return _read_executor_state[1]


_read_executor_state: Optional[Tuple[int, concurrent.futures.ThreadPoolExecutor]] = None


# -----------------------------------------------------------------------------
# Backend
# -----------------------------------------------------------------------------


class Backend:
    """zarr-python 3 implementation of the version specific operations of nodes and stores"""

    supported_zarr_formats: Tuple[int, ...] = (2, 3)

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
        compressor: Union[numcodecs.Blosc, Codec, None] = None
        if compression is not None:
            if group.metadata.zarr_format == 2:
                compressor = compression.to_numcodecs()
            else:
                compressor = BloscCodec(**cast(Dict[str, Any], compression.to_config()))
        return group.create_array(
            name,
            data=data,
            chunks=chunks,
            compressors=compressor,
            attributes=None if attributes is None else attributes_dict(attributes),
            # (store chunks equal to the fill value too, as zarr-python 2: identical records of both versions)
            config={"write_empty_chunks": True},
        )

    def read(self, array: zarr.Array, selection: Selection) -> object:
        if _direct_reads_enabled() and _is_basic_selection(selection) and (plan := _direct_read(array)) is not None:
            return _read_direct(plan, array, selection)
        return run(array.async_array.getitem(selection))

    async def read_async(self, array: zarr.Array, selection: Selection) -> object:
        if _direct_reads_enabled() and _is_basic_selection(selection) and (plan := _direct_read(array)) is not None:
            return await _read_direct_async(plan, array, selection)
        if _on_own_loop():
            return await array.async_array.getitem(selection)
        # (zarr-python's codec pipeline decodes on the calling event loop: run it on a worker thread's own loop)
        return await asyncio.get_running_loop().run_in_executor(_read_executor(), self.read, array, selection)

    def compression(self, array: zarr.Array) -> Optional[BloscCompression]:
        if not (compressors := array.compressors):
            return None
        if len(compressors) == 1:
            if isinstance(compressor := compressors[0], numcodecs.Blosc):  # (zarr format v2)
                return BloscCompression.from_numcodecs(compressor)
            if isinstance(compressor, BloscCodec):  # (zarr format v3)
                return BloscCompression.from_config(cast(Dict[str, object], compressor.to_dict()["configuration"]))
        raise ValueError(f"Unsupported compressors {compressors}")

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

        if consolidated:
            # the compressed consolidated metadata of zarr format v2 / v3 hierarchies (root zarr.json documents of zarr
            # format v3 hierarchies with consolidated metadata are only stored compressed)
            if (compressed := _get(zarr_store, V3_CONSOLIDATED_METADATA_KEY)) is not None:
                return self._open_v3(zarr_store, decode_cbor_zlib(compressed), consolidated=True)
            if (compressed := _get(zarr_store, V2_CONSOLIDATED_METADATA_KEY)) is None:
                raise KeyError(V2_CONSOLIDATED_METADATA_KEY)
        else:
            if (compressed := _get(zarr_store, V3_CONSOLIDATED_METADATA_KEY)) is not None:
                return self._open_v3(zarr_store, decode_cbor_zlib(compressed), consolidated=False)
            if (root := _get(zarr_store, "zarr.json")) is not None:  # (zarr format v3 without consolidated metadata)
                return self._open_v3(zarr_store, json.loads(root), consolidated=False)
            return zarr.Group(
                run(zarr_async.open_group(store=zarr_store, zarr_format=2, mode="r", use_consolidated=False))
            )

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
            prototype.buffer.from_bytes(
                json.dumps({**document, "metadata": _parent_grouped(metadata, _v2_node_path)}).encode()
            ),
        )
        return zarr.Group(root)

    def _open_v3(self, store: Store, document: Dict[str, Any], consolidated: bool) -> zarr.Group:
        """Opens the root group of a zarr format v3 hierarchy from its root document"""
        if consolidated:
            nodes = document["consolidated_metadata"]
            nodes = {**nodes, "metadata": _parent_grouped(nodes["metadata"], str)}
            document = {**document, "consolidated_metadata": nodes}
        else:
            # (members are opened from their node metadata records)
            document = {key: value for key, value in document.items() if key != "consolidated_metadata"}
        with _ignored_warnings():
            return zarr.Group(AsyncGroup.from_dict(StorePath(store, ""), document))

    def consolidate_store(self, store: StoreLike) -> None:
        zarr_store = _zarr_store(store)
        if (root := _get(zarr_store, "zarr.json")) is not None:
            # zarr format v3: the root document with the consolidated metadata of the hierarchy (as written by
            # zarr-python's consolidate_metadata)
            document = {
                **json.loads(root),
                "consolidated_metadata": {
                    "kind": "inline",
                    "must_understand": False,
                    "metadata": _v3_node_documents(zarr_store),
                },
            }
            value = encode_cbor_zlib(document)
            run(zarr_store.set(V3_CONSOLIDATED_METADATA_KEY, default_buffer_prototype().buffer.from_bytes(value)))
            return

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
