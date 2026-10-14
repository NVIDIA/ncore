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
:meth:`_DirectRead.plan`).
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
    TypedDict,
    TypeVar,
    Union,
    cast,
)

import numcodecs
import numpy as np
import zarr
import zarr.api.asynchronous as zarr_async

from typing_extensions import override
from upath import UPath
from zarr.abc.codec import Codec
from zarr.abc.store import ByteRequest, OffsetByteRequest, RangeByteRequest, Store, SuffixByteRequest
from zarr.codecs import BloscCodec, BytesCodec, TransposeCodec, ZstdCodec
from zarr.core.buffer import Buffer, BufferPrototype, default_buffer_prototype
from zarr.core.chunk_grids import ChunkGrid
from zarr.core.common import JSON
from zarr.core.group import AsyncGroup
from zarr.core.indexing import BasicIndexer, ChunkProjection
from zarr.core.metadata import ArrayV2Metadata
from zarr.core.sync import sync
from zarr.storage import LocalStore, StorePath

from ncore.impl.data.backend import (
    V2_CONSOLIDATED_METADATA_KEY,
    V3_CONSOLIDATED_METADATA_KEY,
    Attributes,
    Backend,
    BloscCompression,
    BloscConfigV3,
    ReadResult,
    Selection,
    ZarrFormat,
    ZarrNode,
)
from ncore.impl.data.encoding import decode_cbor_xz, decode_cbor_zlib, encode_cbor_xz, encode_cbor_zlib
from ncore.impl.data.itar import IndexedTarArchive, is_node_metadata_key
from ncore.impl.data.json_types import JsonDocument, to_json_document


if TYPE_CHECKING:
    import numpy.typing as npt


__all__ = ["TransposeCodec", "ZstdCodec"]  # (zarr-python 3 codecs of the tests)

T = TypeVar("T")


class _V2ConsolidatedMetadata(TypedDict):
    """zarr-python 2's consolidated metadata document (``.zmetadata``)"""

    zarr_consolidated_format: int
    #: node metadata documents by node metadata key (e.g., ``a/.zarray``)
    metadata: Dict[str, JsonDocument]


class _InlineConsolidatedMetadata(TypedDict):
    """zarr-python 3's inline consolidated metadata of the root group document (``consolidated_metadata``)"""

    kind: Literal["inline"]
    must_understand: bool
    #: node metadata documents of all members by node path
    metadata: Dict[str, JsonDocument]


# -----------------------------------------------------------------------------
# Execution of coroutines on event loops of the calling threads
# -----------------------------------------------------------------------------


class _InlineExecutor(concurrent.futures.ThreadPoolExecutor):
    """Executor running submitted work immediately in the submitting thread (asyncio requires default executors to be
    thread pools, which start no threads until work is submitted to them)"""

    @override
    def submit(self, fn, /, *args, **kwargs):
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

    class _ThreadState(threading.local):
        loop: Optional[_Loop] = None

    #: the loop of each thread
    _thread = _ThreadState()

    @classmethod
    def is_current(cls) -> bool:
        """True if running on the event loop of :func:`run` of the calling thread (blocking work runs inline), False
        on other event loops (blocking work is offloaded to threads, to not block them)"""
        state = cls._thread.loop
        return state is not None and state.pid == os.getpid() and asyncio.get_running_loop() is state.loop

    #: executor of blocking parts of reads awaited on event loops other than the one of :func:`run` (per process)
    _executor: Optional[Tuple[int, concurrent.futures.ThreadPoolExecutor]] = None

    @classmethod
    def executor(cls) -> concurrent.futures.ThreadPoolExecutor:
        """The executor of blocking parts of reads awaited on event loops other than the one of :func:`run` (re-created
        in forked processes)"""
        if cls._executor is None or cls._executor[0] != os.getpid():
            cls._executor = (
                os.getpid(),
                concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(32, (os.cpu_count() or 1) + 4), thread_name_prefix="ncore_zarr3_reads"
                ),
            )
        return cls._executor[1]


def run(coroutine: Coroutine[Any, Any, T]) -> T:
    """Runs a zarr-python coroutine to completion on an event loop owned by the calling thread.

    The loop runs offloaded work (e.g., of codecs) inline. Processes forked by a thread with a loop create a new loop.
    Within running event loops (e.g., of asynchronous applications), coroutines run on zarr-python's I/O loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        if (state := _Loop._thread.loop) is None or state.pid != os.getpid():
            state = _Loop._thread.loop = _Loop()  # (see _Loop.__del__ for loops of parent processes)
        return state.loop.run_until_complete(coroutine)
    return sync(coroutine)


# -----------------------------------------------------------------------------
# zarr-python 3 store of indexed tar archives
# -----------------------------------------------------------------------------


class _IndexedTarStore(Store):
    """zarr-python 3 store of the records of an :class:`IndexedTarArchive` (synchronous I/O)"""

    supports_writes: bool = True
    supports_deletes: bool = False
    supports_partial_writes: Literal[False] = False
    supports_listing: bool = True

    def __init__(self, archive: IndexedTarArchive) -> None:
        super().__init__(read_only=archive.mode == "r")
        self.archive = archive
        self._is_open = True

    @override
    def __eq__(self, value: object) -> bool:
        return isinstance(value, _IndexedTarStore) and self.archive == value.archive

    def __hash__(self) -> int:
        return hash(self.archive)

    def __repr__(self) -> str:
        return f"_IndexedTarStore({self.archive!r})"

    @override
    def with_read_only(self, read_only: bool = False) -> _IndexedTarStore:
        if read_only != self.read_only:
            raise NotImplementedError("IndexedTarArchive: the access mode of archives can't be changed")
        return self

    @override
    def close(self) -> None:
        self.archive.close()
        self._is_open = False

    @staticmethod
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

    def _get(
        self, key: str, prototype: Optional[BufferPrototype], byte_range: Optional[ByteRequest]
    ) -> Optional[Buffer]:
        try:
            value = self.archive.read(key)
        except KeyError:
            return None
        start, end = self._byte_range(byte_range, len(value))
        return (prototype or default_buffer_prototype()).buffer.from_bytes(value[start:end])

    @override
    async def get(
        self, key: str, prototype: BufferPrototype, byte_range: Optional[ByteRequest] = None
    ) -> Optional[Buffer]:
        return self._get(key, prototype, byte_range)

    @override
    async def get_partial_values(
        self, prototype: BufferPrototype, key_ranges: Iterable[Tuple[str, Optional[ByteRequest]]]
    ) -> List[Optional[Buffer]]:
        return [self._get(key, prototype, byte_range) for key, byte_range in key_ranges]

    @override
    async def exists(self, key: str) -> bool:
        return key in self.archive

    @override
    async def set(self, key: str, value: Buffer) -> None:
        self._check_writable()
        self.archive[key] = value.to_bytes()

    @override
    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        if key not in self.archive:
            await self.set(key, value)

    @override
    async def delete(self, key: str) -> None:
        """No-op for records that don't exist (zarr-python deletes them when creating nodes)"""
        self._check_writable()
        if key in self.archive:
            raise NotImplementedError("IndexedTarArchive: records can't be deleted")

    @override
    async def delete_dir(self, prefix: str) -> None:
        """No-op for prefixes without records (zarr-python deletes them when creating nodes)"""
        self._check_writable()
        prefix = prefix if prefix == "" or prefix.endswith("/") else prefix + "/"
        if any(key.startswith(prefix) for key in self.archive.keys()):
            raise NotImplementedError("IndexedTarArchive: records can't be deleted")

    @override
    async def list(self) -> AsyncIterator[str]:
        for key in self.archive.keys():
            yield key

    @override
    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        for key in self.archive.keys():
            if key.startswith(prefix):
                yield key

    @override
    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        prefix = prefix.rstrip("/")
        prefix = prefix + "/" if prefix else ""
        children = {key[len(prefix) :].split("/", 1)[0]: None for key in self.archive.keys() if key.startswith(prefix)}
        for child in children:
            yield child


# -----------------------------------------------------------------------------
# Direct chunk reads
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
    """Bounds the number of reads decoding their chunks concurrently (per process).

    At most :data:`_PARALLEL_READS` reads decode concurrently, each on all threads of the shared pool (one per core, at
    most 8), so the pool is sized for the chunks of one read, not for the number of reads.
    """

    def __init__(self) -> None:
        self.semaphore = threading.BoundedSemaphore(_PARALLEL_READS)
        self.executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self.lock = threading.Lock()
        self.pid = os.getpid()

    def pool(self) -> concurrent.futures.ThreadPoolExecutor:
        """The thread pool decoding the chunks of reads (created on first use)"""
        with self.lock:
            if self.executor is None:
                self.executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(8, os.cpu_count() or 1), thread_name_prefix="ncore_zarr3_chunks"
                )
            return self.executor

    #: the parallel reads state of the process
    _state: Optional[_ParallelReads] = None

    @classmethod
    def get(cls) -> _ParallelReads:
        """The parallel reads state of this process (re-created in forked processes, whose locks / threads are
        unusable)"""
        if cls._state is None or cls._state.pid != os.getpid():
            cls._state = _ParallelReads()
        return cls._state


class _ArchiveRecords(NamedTuple):
    """The chunk records of an indexed tar archive opened for reading (positional reads of local archives, records of
    the index tail read from memory, asynchronous range requests of remote archives, see
    :class:`~ncore.impl.data.itar.IndexedTarArchive`)"""

    archive: IndexedTarArchive

    @property
    def is_remote(self) -> bool:
        """True if records are fetched concurrently by :meth:`get_async` (remote archives)"""
        return self.archive.is_remote

    def get(self, key: str) -> Optional[bytes]:
        """The record (None if it doesn't exist)"""
        try:
            return self.archive.read(key)
        except KeyError:
            return None

    async def get_async(self, key: str) -> Optional[bytes]:
        """The record, read asynchronously (None if it doesn't exist)"""
        try:
            return await self.archive.read_async(key)
        except KeyError:
            return None


class _StoreRecords(NamedTuple):
    """The chunk records of any other zarr-python store opened for reading (e.g., a local directory, or a remote store,
    whose records are fetched concurrently by :meth:`get_async`)"""

    store: Store

    #: (fetched via the store's asynchronous API: concurrent fetches overlap for all stores)
    is_remote: bool = True

    def get(self, key: str) -> Optional[bytes]:
        """The record (None if it doesn't exist)"""
        value = run(self.store.get(key, prototype=default_buffer_prototype()))
        return None if value is None else value.to_bytes()

    async def get_async(self, key: str) -> Optional[bytes]:
        """The record, read asynchronously (None if it doesn't exist)"""
        value = await self.store.get(key, prototype=default_buffer_prototype())
        return None if value is None else value.to_bytes()


class _DirectRead(NamedTuple):
    """How to read the chunks of an array directly (see :meth:`plan`)"""

    records: Union[_ArchiveRecords, _StoreRecords]
    path: str
    dtype: np.dtype
    chunks: Tuple[int, ...]
    chunk_key: Callable[[Tuple[int, ...]], str]
    compressor: Optional[numcodecs.Blosc]
    fill_value: object

    @staticmethod
    def enabled() -> bool:
        return os.environ.get(DIRECT_READS_ENV, "1").strip().lower() not in ("0", "false", "no", "off")

    @staticmethod
    def plan(array: zarr.Array) -> Optional[_DirectRead]:
        """How to read the chunks of an array directly (None if the array is read via zarr-python).

        zarr-python determines the chunks of a read and their selections (its indexers), but the chunk records are
        read from the store and decompressed directly into the output array, instead of via zarr-python's codec
        pipeline, which decodes each chunk into a temporary buffer and copies it (see zarr-python#2904).

        Read directly are all arrays written by ncore, i.e. arrays of stores opened for reading with

        - any number of chunks of a regular chunk grid, in C order,
        - numeric / boolean data of native byte order, or fixed-length bytes,
        - uncompressed chunks, or chunks compressed by blosc (the compressor ncore writes), without filters.

        Arrays using other codecs (e.g., zstd, gzip, filters, transposes) are read via zarr-python.

        Records of indexed tar archives are read directly from the archive (positional reads), records of other
        zarr-python stores (e.g., directories) via the store.

        Planned per read (sub-microsecond), so nothing outlives the array.
        """
        store_path = array.async_array.store_path
        records: Union[_ArchiveRecords, _StoreRecords]
        if isinstance(store := store_path.store, _IndexedTarStore):
            if store.archive.mode != "r":  # (node metadata of archives opened for writing is not final)
                return None
            records = _ArchiveRecords(store.archive)
        else:
            records = _StoreRecords(store)

        metadata = array.metadata
        dtype = np.dtype(metadata.dtype.to_native_dtype())
        if dtype.hasobject or not dtype.isnative or dtype.kind not in "biufS":
            return None
        compressor: object
        if isinstance(metadata, ArrayV2Metadata):
            if metadata.order != "C" or metadata.filters:
                return None
            compressor = metadata.compressor
            chunks = tuple(metadata.chunks)
        else:
            codecs = list(metadata.codecs)
            if not codecs or not isinstance(bytes_codec := codecs[0], BytesCodec):
                return None
            if dtype.itemsize > 1 and str(getattr(bytes_codec.endian, "value", bytes_codec.endian)) != sys.byteorder:
                return None
            if len(codecs) == 1:
                compressor = None
            elif len(codecs) == 2 and isinstance(codecs[1], BloscCodec):
                compressor = codecs[1]._blosc_codec
            else:
                return None
            if (chunk_shape := getattr(metadata.chunk_grid, "chunk_shape", None)) is None:  # (regular grids only)
                return None
            chunks = tuple(chunk_shape)
        if compressor is not None and not isinstance(compressor, numcodecs.Blosc):
            return None
        return _DirectRead(
            records, store_path.path, dtype, chunks, metadata.encode_chunk_key, compressor, metadata.fill_value
        )

    @staticmethod
    def is_basic_selection(selection: Selection) -> bool:
        """True for selections of (tuples of) integers, slices with unit step, and Ellipsis"""
        for item in selection if isinstance(selection, tuple) else (selection,):
            if item is Ellipsis or (isinstance(item, (int, np.integer)) and not isinstance(item, (bool, np.bool_))):
                continue
            if isinstance(item, slice) and item.step in (None, 1):
                continue
            return False
        return True

    @staticmethod
    def chunk_grid(array: zarr.Array) -> ChunkGrid:
        """zarr-python's chunk grid of an array, as its indexers take it.

        zarr-python moved the chunk grid from the array metadata (``metadata.chunk_grid``, zarr-python < 3.2) to the
        array (``async_array._chunk_grid``, the metadata keeps a chunk grid description since then), so both locations
        are needed for the supported zarr-python versions (>= 3.1.4).
        """
        if (grid := getattr(array.async_array, "_chunk_grid", None)) is not None:
            return grid
        return getattr(array.metadata, "chunk_grid")

    def read(self, array: zarr.Array, selection: Selection) -> npt.NDArray[Any]:
        """Reads a basic selection of an array (see :meth:`is_basic_selection`), decompressing its chunks directly into
        the output array (see :class:`_DirectReadOp`).

        Reads of multiple large chunks (see :data:`_PARALLEL_CHUNK_BYTES`) read and decode their chunks concurrently
        on a thread pool (positional reads and blosc decompression release the GIL), unless :data:`_PARALLEL_READS`
        reads do so already. All other reads run on the calling thread.
        """
        op = _DirectReadOp(self, array, selection)
        return op.decode([], fetch=self.records.get)

    async def read_async(self, array: zarr.Array, selection: Selection) -> npt.NDArray[Any]:
        """Asynchronous :meth:`read`: chunks of remote archives and of other zarr-python stores are fetched concurrently.
        Blocking work (local reads, decompression) runs on the calling thread on the event loop of :func:`run`, on
        worker threads on other event loops (e.g., of asynchronous applications), so they aren't blocked."""
        op = _DirectReadOp(self, array, selection)
        if self.records.is_remote:
            raws = list(await asyncio.gather(*(self.records.get_async(key) for key in op.keys)))
            if _Loop.is_current():
                return op.decode(raws)
            return await asyncio.get_running_loop().run_in_executor(_Loop.executor(), op.decode, raws)
        if _Loop.is_current():
            return op.decode([], fetch=self.records.get)
        return await asyncio.get_running_loop().run_in_executor(
            _Loop.executor(), lambda: op.decode([], fetch=self.records.get)
        )


class _DirectReadOp:
    """A direct read of a basic selection of an array: the chunks to read (see :attr:`keys`), and their decoding into
    the output array (see :meth:`decode`).

    zarr-python's indexer splits the selection into chunk projections, one per chunk the selection touches: the chunk
    (``chunk_coords``), the part of the chunk selected (``chunk_selection``), and where it goes in the output array
    (``out_selection``). Arrays stored as a single chunk have a single projection, however large.
    """

    def __init__(self, plan: _DirectRead, array: zarr.Array, selection: Selection) -> None:
        self.plan = plan
        self.shape = tuple(array.metadata.shape)
        indexer = BasicIndexer(selection, self.shape, plan.chunk_grid(array))
        self.out_shape = tuple(indexer.shape)
        self.projections: List[ChunkProjection] = list(indexer) if all(indexer.shape) else []  # (none if empty)
        self.keys = [self._key(projection) for projection in self.projections]

    def _key(self, projection: ChunkProjection) -> str:
        """The record key of the chunk of a projection"""
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

    def decode(
        self, raws: List[Optional[bytes]], fetch: Optional[Callable[[str], Optional[bytes]]] = None
    ) -> npt.NDArray[Any]:
        """Decodes the chunks (None: not stored, read as the fill value) into the output array. With ``fetch``, chunks
        are read by the decoding threads (raws are ignored).

        Returns a new array of the selection's shape, except for complete reads of arrays stored as a single
        uncompressed chunk (e.g., encoded images), which return a read-only view of the chunk record (no copies).
        Reads of multiple large chunks (see :data:`_PARALLEL_CHUNK_BYTES`) decode their chunks concurrently on a thread
        pool (positional reads and blosc decompression release the GIL), unless :data:`_PARALLEL_READS` reads do so
        already. All other reads run on the calling thread.
        """
        plan = self.plan
        if self.is_single_raw_chunk and (raw := raws[0] if fetch is None else fetch(self.keys[0])) is not None:
            # the complete array, stored as a single uncompressed chunk (e.g., an encoded image): the result is the
            # chunk record itself (a read-only view, no copies; chunks not stored read as the fill value, see below)
            if (view := Zarr3Backend.chunk_view(raw, plan.dtype, self.shape)) is None:
                raise ValueError(f"Chunk {self.keys[0]} doesn't match its array ({len(raw)} bytes)")
            return view

        out = np.empty(self.out_shape, dtype=plan.dtype)

        def decode_chunk(index: int) -> None:
            """Decodes one chunk of the selection into its part of the output array"""
            projection = self.projections[index]
            raw = raws[index] if fetch is None else fetch(self.keys[index])
            # the part of the output array the chunk's selection goes to (all of it for 0-dimensional results)
            target = out[projection.out_selection] if projection.out_selection != () else out
            if raw is None:
                # chunk not stored (e.g., chunks of the fill value written by other writers): the fill value
                target[...] = np.asarray(plan.fill_value, dtype=plan.dtype) if plan.fill_value is not None else 0
                return
            if (
                plan.compressor is not None
                and projection.is_complete_chunk
                and target.shape == plan.chunks
                and target.flags.c_contiguous
            ):
                # the complete chunk is selected, and its part of the output array is a contiguous block of the chunk's
                # shape (e.g., chunks along the first dimension, as for point clouds): decompress in place
                plan.compressor.decode(raw, out=target)
                return
            # all other cases (uncompressed chunks, partial chunk selections, or output parts that aren't contiguous,
            # e.g., chunks along later dimensions): decompress the chunk, and copy its selected part
            decoded = plan.compressor.decode(raw) if plan.compressor is not None else raw
            selected = np.frombuffer(decoded, dtype=plan.dtype).reshape(plan.chunks)[projection.chunk_selection]
            target[...] = selected.reshape(target.shape)  # (integer selections drop axes)

        parallel = _ParallelReads.get()
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


# -----------------------------------------------------------------------------
# Backend
# -----------------------------------------------------------------------------


class Zarr3Backend(Backend[Store]):
    """zarr-python 3 implementation of the version specific operations of nodes and stores"""

    supported_zarr_formats: Tuple[ZarrFormat, ...] = (2, 3)

    @staticmethod
    @contextlib.contextmanager
    def _ignored_warnings() -> Iterator[None]:
        with warnings.catch_warnings():
            # compressed consolidated metadata records are not zarr nodes
            warnings.filterwarnings("ignore", message="Object at .* is not recognized as a component")
            yield

    @staticmethod
    def _json_document(attributes: Attributes) -> Dict[str, JSON]:
        """Attributes as JSON document of zarr-python 3's JSON type (of plain dicts / lists, see
        :func:`~ncore.impl.data.json_types.to_json`)"""
        document: Dict[str, JSON] = {**to_json_document(attributes)}
        return document

    # -- nodes -----------------------------------------------------------------

    @override
    def zarr_format(self, node: ZarrNode) -> ZarrFormat:
        return node.metadata.zarr_format

    @override
    def open_member(self, group: zarr.Group, path: str) -> ZarrNode:
        node = run(group._async_group.getitem(path))
        return zarr.Group(node) if isinstance(node, AsyncGroup) else zarr.Array(node)

    @override
    def member_names(self, group: zarr.Group) -> List[str]:
        with self._ignored_warnings():  # (members of groups with consolidated metadata are listed without I/O)
            return list(group.keys())

    @override
    def has_member(self, group: zarr.Group, name: str) -> bool:
        return name in self.member_names(group)

    @override
    def member_groups(self, group: zarr.Group) -> List[Tuple[str, zarr.Group]]:
        with self._ignored_warnings():
            return list(group.groups())

    @override
    def create_group(self, group: zarr.Group, name: str, attributes: Optional[Attributes]) -> zarr.Group:
        return group.create_group(name, attributes=None if attributes is None else self._json_document(attributes))

    @override
    def update_attributes(self, node: ZarrNode, attributes: Attributes) -> None:
        node.attrs.update(self._json_document(attributes))

    @override
    def require_group(self, group: zarr.Group, name: str) -> zarr.Group:
        return group.require_group(name)

    @override
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
                compressor = BloscCodec(
                    cname=compression.cname,
                    clevel=compression.clevel,
                    shuffle=compression.shuffle,
                    blocksize=compression.blocksize,
                )
        return group.create_array(
            name,
            data=data,
            chunks=chunks,
            compressors=compressor,
            attributes=None if attributes is None else self._json_document(attributes),
            # (store chunks equal to the fill value too, as zarr-python 2: identical records of both versions)
            config={"write_empty_chunks": True},
        )

    @override
    def read(self, array: zarr.Array, selection: Selection) -> ReadResult:
        """Fast path: basic selections of arrays of indexed tar archives opened for reading are read directly (see
        :class:`_DirectRead`): complete reads of arrays stored as a single uncompressed chunk are views of the chunk
        record (no copies), chunks of all other arrays are decompressed into the output array. Fallback: zarr-python's
        read (e.g., of directories, non-basic selections, or with direct reads disabled)"""
        if _DirectRead.enabled() and _DirectRead.is_basic_selection(selection) and (plan := _DirectRead.plan(array)):
            return self.read_result(plan.read(array, selection))
        return self.read_result(run(array.async_array.getitem(selection)))

    @override
    async def read_async(self, array: zarr.Array, selection: Selection) -> ReadResult:
        """Asynchronous :meth:`read`, returning the same results.

        Direct reads (see :class:`_DirectRead`) fetch the chunks of remote archives and of other zarr-python stores
        concurrently. All other reads run zarr-python's codec pipeline: inline on the event loop of :func:`run`
        (the calling thread's), on a worker thread's event loop when awaited on other event loops (e.g., of
        applications), so they aren't blocked by decoding.
        """
        if _DirectRead.enabled() and _DirectRead.is_basic_selection(selection) and (plan := _DirectRead.plan(array)):
            return self.read_result(await plan.read_async(array, selection))
        if _Loop.is_current():
            return self.read_result(await array.async_array.getitem(selection))
        # (zarr-python's codec pipeline decodes on the calling event loop: run it on a worker thread's own loop)
        return await asyncio.get_running_loop().run_in_executor(_Loop.executor(), self.read, array, selection)

    @override
    def compression(self, array: zarr.Array) -> Optional[BloscCompression]:
        if not (compressors := array.compressors):
            return None
        if len(compressors) == 1:
            if isinstance(compressor := compressors[0], numcodecs.Blosc):  # (zarr format v2)
                return BloscCompression.from_numcodecs(compressor)
            if isinstance(compressor, BloscCodec):  # (zarr format v3)
                return BloscCompression.from_config_v3(cast(BloscConfigV3, compressor.to_dict()["configuration"]))
        raise ValueError(f"Unsupported compressors {compressors}")

    # -- stores ----------------------------------------------------------------

    @override
    def zarr_store(self, store: object) -> Store:
        if not isinstance(store, Store):
            raise TypeError(f"Not a zarr-python 3 store: {store!r}")
        return store

    @staticmethod
    def _keys(store: Store) -> List[str]:
        async def keys() -> List[str]:
            return [key async for key in store.list()]

        return run(keys())

    @staticmethod
    def _get(store: Store, key: str) -> Optional[bytes]:
        value = run(store.get(key, prototype=default_buffer_prototype()))
        return None if value is None else value.to_bytes()

    @staticmethod
    def _parent_grouped(metadata: Dict[str, JsonDocument], node_path: Callable[[str], str]) -> Dict[str, JsonDocument]:
        """Orders consolidated node metadata by node depth, with the members of each group adjacent.

        zarr-python < 3.4 nests flat consolidated metadata assuming this order (zarr-python#4226: the members of a group
        are attached to the first run of keys of the group only), which isn't the order of hierarchies created in
        arbitrary order (e.g., by multiple component writers). Keys are node metadata keys (zarr format v2) or node
        paths (zarr format v3), ``node_path`` maps them to node paths.
        """

        def order(key: str) -> Tuple[int, str, str]:
            parent, _, name = (path := node_path(key)).rpartition("/")
            return path.count("/"), parent, name

        return {key: metadata[key] for key in sorted(metadata, key=order)}

    @classmethod
    def _open_v2_consolidated(cls, store: Store, record: bytes) -> zarr.Group:
        """Opens the root group of a zarr format v2 hierarchy from its consolidated metadata record (see
        :data:`V2_CONSOLIDATED_METADATA_KEY`), without reads.

        zarr-python 3 represents consolidated metadata of both zarr formats as inline consolidated metadata of the root
        group document (members by node path), which is built from the ``.zmetadata`` document (members by node
        metadata key) as zarr-python's ``AsyncGroup._from_bytes_v2`` does.
        """
        consolidated = cast(_V2ConsolidatedMetadata, decode_cbor_xz(record))
        if (consolidated_format := consolidated["zarr_consolidated_format"]) != 1:
            raise ValueError(f"Unsupported zarr consolidated metadata format: {consolidated_format}")
        metadata = cls._parent_grouped(consolidated["metadata"], lambda key: key.rpartition("/")[0])
        members: Dict[str, JsonDocument] = {}
        for key, value in metadata.items():
            path, _, name = key.rpartition("/")
            if not path:  # (the documents of the root group)
                continue
            node = members.setdefault(path, {})
            if name == ".zattrs":
                node["attributes"] = value
            else:  # .zarray / .zgroup
                node.update(value)
        inline: _InlineConsolidatedMetadata = {"kind": "inline", "must_understand": False, "metadata": members}
        document = {**metadata[".zgroup"], "attributes": metadata.get(".zattrs", {}), "consolidated_metadata": inline}
        return zarr.Group(AsyncGroup.from_dict(StorePath(store, ""), document))

    @classmethod
    def _open_v3_consolidated(cls, store: Store, record: bytes) -> zarr.Group:
        """Opens the root group of a zarr format v3 hierarchy from its consolidated metadata record (see
        :data:`V3_CONSOLIDATED_METADATA_KEY`), without reads (members ordered by :meth:`_parent_grouped`)"""
        root = decode_cbor_zlib(record)
        inline = cast(_InlineConsolidatedMetadata, root["consolidated_metadata"])
        inline = {**inline, "metadata": cls._parent_grouped(inline["metadata"], str)}
        return zarr.Group(AsyncGroup.from_dict(StorePath(store, ""), {**root, "consolidated_metadata": inline}))

    @classmethod
    def _v3_member_documents(cls, store: Store) -> Dict[str, JsonDocument]:
        """The node metadata documents of all members of a zarr format v3 hierarchy, by node path"""
        documents: Dict[str, JsonDocument] = {}
        for key in cls._keys(store):
            if key.endswith("/zarr.json") and (value := cls._get(store, key)) is not None:
                documents[key[: -len("/zarr.json")]] = json.loads(value)
        return documents

    @override
    def indexed_tar_store(self, archive: IndexedTarArchive) -> Store:
        return _IndexedTarStore(archive)

    @override
    def directory_store(self, path: UPath) -> Store:
        return LocalStore(path.path)

    @override
    def create_root_group(self, store: Store, zarr_format: ZarrFormat, attributes: Attributes) -> zarr.Group:
        node = run(
            zarr_async.create_group(store=store, zarr_format=zarr_format, attributes=self._json_document(attributes))
        )
        return zarr.Group(node)

    @override
    def open_root_group(self, store: Store, consolidated: bool) -> zarr.Group:
        if not consolidated:
            # (from the node metadata records, of either zarr format)
            return zarr.Group(run(zarr_async.open_group(store=store, mode="r", use_consolidated=False)))
        if (record := self._get(store, V3_CONSOLIDATED_METADATA_KEY)) is not None:
            return self._open_v3_consolidated(store, record)
        if (record := self._get(store, V2_CONSOLIDATED_METADATA_KEY)) is not None:
            return self._open_v2_consolidated(store, record)
        raise KeyError(f"No consolidated metadata ({V2_CONSOLIDATED_METADATA_KEY} / {V3_CONSOLIDATED_METADATA_KEY})")

    @override
    def consolidate(self, store: Store) -> None:
        if (root := self._get(store, "zarr.json")) is not None:
            # zarr format v3 specifies no consolidated metadata, zarr-python 3 stores it in the root group document
            # (``consolidated_metadata``, which zarr format v3 permits as extension of node metadata documents): the
            # consolidated metadata record is the root zarr.json document with it, as written by zarr-python 3's
            # consolidate_metadata. (It has no format version: its documents are versioned by their zarr_format.)
            inline: _InlineConsolidatedMetadata = {
                "kind": "inline",
                "must_understand": False,
                "metadata": self._v3_member_documents(store),
            }
            document: JsonDocument = {**json.loads(root), "consolidated_metadata": cast(JsonDocument, inline)}
            key, value = V3_CONSOLIDATED_METADATA_KEY, encode_cbor_zlib(document)
        else:
            # zarr format v2: the .zmetadata document (node metadata documents by node metadata key)
            consolidated: _V2ConsolidatedMetadata = {
                "zarr_consolidated_format": 1,
                "metadata": {
                    key: json.loads(value)
                    for key in self._keys(store)
                    if is_node_metadata_key(key) and (value := self._get(store, key)) is not None
                },
            }
            key, value = V2_CONSOLIDATED_METADATA_KEY, encode_cbor_xz(cast(JsonDocument, consolidated))
        run(store.set(key, default_buffer_prototype().buffer.from_bytes(value)))


backend = Zarr3Backend()
