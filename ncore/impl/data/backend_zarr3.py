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
from zarr.codecs import BloscCodec, BytesCodec
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
    ReadResult,
    Selection,
    ZarrNode,
)
from ncore.impl.data.encoding import decode_cbor_xz, decode_cbor_zlib, encode_cbor_xz, encode_cbor_zlib
from ncore.impl.data.itar import IndexedTarArchive, is_node_metadata_key
from ncore.impl.data.json_types import JsonDocument


if TYPE_CHECKING:
    import numpy.typing as npt


T = TypeVar("T")


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
            value = self.archive[key]
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

    #: the parallel reads state of the process
    _state: Optional[_ParallelReads] = None

    @classmethod
    def get(cls) -> _ParallelReads:
        """The parallel reads state of this process (re-created in forked processes, whose locks / threads are
        unusable)"""
        if cls._state is None or cls._state.pid != os.getpid():
            cls._state = _ParallelReads()
        return cls._state


class _DirectRead(NamedTuple):
    """How to read the chunks of an array directly (see :meth:`plan`)"""

    archive: IndexedTarArchive
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

        Supported are arrays of indexed tar archives opened for reading with a regular chunk grid, C order, numeric /
        boolean data of native byte order or fixed-length bytes, no filters, and no compression or a single blosc
        compressor (all arrays written by ncore). zarr-python determines the chunks and their selections (its
        indexers), but the chunks are read and decompressed directly into the output array, instead of via
        zarr-python's codec pipeline, which decodes each chunk into a temporary buffer and copies it (see
        zarr-python#2904).

        Planned per read (sub-microsecond), so nothing outlives the array.
        """
        store_path = array.async_array.store_path
        if not isinstance(store := store_path.store, _IndexedTarStore) or store.archive.mode != "r":
            return None

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
            store.archive, store_path.path, dtype, chunks, metadata.encode_chunk_key, compressor, metadata.fill_value
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
        """zarr-python's chunk grid of an array (an array attribute since zarr-python 3.2, metadata before)"""
        if (grid := getattr(array.async_array, "_chunk_grid", None)) is not None:
            return grid
        return getattr(array.metadata, "chunk_grid")

    def read(self, array: zarr.Array, selection: Selection) -> npt.NDArray[Any]:
        """Reads a basic selection of an array, decompressing its chunks directly into the output array.

        Reads of multiple large chunks (see :data:`_PARALLEL_CHUNK_BYTES`) read and decode their chunks concurrently
        on a thread pool (positional reads and blosc decompression release the GIL), unless :data:`_PARALLEL_READS`
        reads do so already. All other reads run on the calling thread.
        """
        shape = tuple(array.metadata.shape)
        indexer = BasicIndexer(selection, shape, self.chunk_grid(array))
        projections = list(indexer) if all(indexer.shape) else []

        if (
            self.compressor is None
            and len(projections) == 1
            and projections[0].is_complete_chunk
            and self.chunks == shape == tuple(indexer.shape)
        ):
            # an uncompressed chunk with the complete array (e.g., an encoded image): use its bytes without copies
            key = self.chunk_key(projections[0].chunk_coords)
            try:
                raw = self.archive[f"{self.path}/{key}" if self.path else key]
            except KeyError:
                pass  # (chunk not stored: fill value, see below)
            else:
                # (a read-only view of the record, no copies)
                if (view := Zarr3Backend.chunk_view(raw, self.dtype, shape)) is None:
                    raise ValueError(f"Chunk {key} doesn't match its array ({len(raw)} bytes)")
                return view

        out = np.empty(indexer.shape, dtype=self.dtype)

        def read_chunk(projection: ChunkProjection) -> None:
            key = self.chunk_key(projection.chunk_coords)
            target = out[projection.out_selection] if projection.out_selection != () else out
            try:
                raw = self.archive[f"{self.path}/{key}" if self.path else key]
            except KeyError:  # chunk not stored (fill value)
                target[...] = np.asarray(self.fill_value, dtype=self.dtype) if self.fill_value is not None else 0
                return
            if (
                self.compressor is not None
                and projection.is_complete_chunk
                and target.shape == self.chunks
                and target.flags.c_contiguous
            ):
                self.compressor.decode(raw, out=target)
                return
            decoded = self.compressor.decode(raw) if self.compressor is not None else raw
            selected = np.frombuffer(decoded, dtype=self.dtype).reshape(self.chunks)[projection.chunk_selection]
            target[...] = selected.reshape(target.shape)  # (integer selections drop axes)

        chunk_bytes = self.dtype.itemsize * int(np.prod(self.chunks))
        parallel = _ParallelReads.get()
        if len(projections) > 1 and chunk_bytes >= _PARALLEL_CHUNK_BYTES and parallel.semaphore.acquire(blocking=False):
            try:
                for future in [parallel.pool().submit(read_chunk, projection) for projection in projections]:
                    future.result()
            finally:
                parallel.semaphore.release()
        else:
            for projection in projections:
                read_chunk(projection)
        return out  # (0-d arrays for 0-d results, see nodes.Array.read)


# -----------------------------------------------------------------------------
# Backend
# -----------------------------------------------------------------------------


class Zarr3Backend(Backend[Store]):
    """zarr-python 3 implementation of the version specific operations of nodes and stores"""

    supported_zarr_formats: Tuple[int, ...] = (2, 3)

    @staticmethod
    @contextlib.contextmanager
    def _ignored_warnings() -> Iterator[None]:
        with warnings.catch_warnings():
            # compressed consolidated metadata records are not zarr nodes
            warnings.filterwarnings("ignore", message="Object at .* is not recognized as a component")
            yield

    @staticmethod
    def _json_document(attributes: Attributes) -> Dict[str, JSON]:
        """Attributes as JSON document of zarr-python 3's JSON type (as :data:`JsonDocument`, typed with abstract
        collections)"""
        document: Dict[str, JSON] = {**attributes}
        return document

    # -- nodes -----------------------------------------------------------------

    @override
    def zarr_format(self, node: ZarrNode) -> int:
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
        node.attrs.update(attributes)

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
    def compression(self, array: zarr.Array) -> Optional[BloscCompression]:
        if not (compressors := array.compressors):
            return None
        if len(compressors) == 1:
            if isinstance(compressor := compressors[0], numcodecs.Blosc):  # (zarr format v2)
                return BloscCompression.from_numcodecs(compressor)
            if isinstance(compressor, BloscCodec):  # (zarr format v3)
                return BloscCompression.from_config(compressor.to_dict()["configuration"])
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
    def _parent_grouped(metadata: JsonDocument, node_path: Callable[[str], str]) -> JsonDocument:
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
    def _open_v2(cls, store: Store, record: bytes) -> zarr.Group:
        """Opens the root group of a zarr format v2 hierarchy from its consolidated metadata record (CBOR + LZMA encoded
        ``.zmetadata``: node metadata documents by node metadata key), without reads, building the root group document
        with the inline consolidated metadata of all members as zarr-python's ``AsyncGroup._from_bytes_v2`` does"""
        consolidated = decode_cbor_xz(record)
        if (consolidated_format := consolidated.get("zarr_consolidated_format")) != 1:
            raise ValueError(f"Unsupported zarr consolidated metadata format: {consolidated_format}")
        metadata = cls._parent_grouped(cast(JsonDocument, consolidated["metadata"]), lambda key: key.rpartition("/")[0])
        members: Dict[str, JsonDocument] = {}
        for key, value in metadata.items():
            path, _, name = key.rpartition("/")
            if not path:  # (the documents of the root group)
                continue
            node = members.setdefault(path, {})
            if name == ".zattrs":
                node["attributes"] = value
            else:  # .zarray / .zgroup
                node.update(cast(JsonDocument, value))
        document: JsonDocument = {
            **cast(JsonDocument, metadata[".zgroup"]),
            "attributes": metadata.get(".zattrs", {}),
            "consolidated_metadata": {"kind": "inline", "must_understand": False, "metadata": dict(members)},
        }
        return zarr.Group(AsyncGroup.from_dict(StorePath(store, ""), document))

    @classmethod
    def _open_v3(cls, store: Store, record: bytes) -> zarr.Group:
        """Opens the root group of a zarr format v3 hierarchy from its consolidated metadata record (CBOR + zlib encoded
        root ``zarr.json`` with the inline consolidated metadata of all members), without reads"""
        root = decode_cbor_zlib(record)
        consolidated = cast(JsonDocument, root["consolidated_metadata"])
        metadata = cls._parent_grouped(cast(JsonDocument, consolidated["metadata"]), str)
        document = {**root, "consolidated_metadata": {**consolidated, "metadata": metadata}}
        with cls._ignored_warnings():
            return zarr.Group(AsyncGroup.from_dict(StorePath(store, ""), document))

    @classmethod
    def _v3_member_documents(cls, store: Store) -> JsonDocument:
        """The node metadata documents of all members of a zarr format v3 hierarchy, by node path"""
        documents: JsonDocument = {}
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
    def create_root_group(self, store: Store, zarr_format: Literal[2, 3], attributes: Attributes) -> zarr.Group:
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
            return self._open_v3(store, record)
        if (record := self._get(store, V2_CONSOLIDATED_METADATA_KEY)) is not None:
            return self._open_v2(store, record)
        raise KeyError(f"No consolidated metadata ({V2_CONSOLIDATED_METADATA_KEY} / {V3_CONSOLIDATED_METADATA_KEY})")

    @override
    def consolidate(self, store: Store) -> None:
        if (root := self._get(store, "zarr.json")) is not None:
            # zarr format v3: the root zarr.json with the inline consolidated metadata of all members
            document: JsonDocument = {
                **json.loads(root),
                "consolidated_metadata": {
                    "kind": "inline",
                    "must_understand": False,
                    "metadata": self._v3_member_documents(store),
                },
            }
            key, value = V3_CONSOLIDATED_METADATA_KEY, encode_cbor_zlib(document)
        else:
            # zarr format v2: the .zmetadata document (node metadata documents by node metadata key)
            metadata: JsonDocument = {
                key: json.loads(value)
                for key in self._keys(store)
                if is_node_metadata_key(key) and (value := self._get(store, key)) is not None
            }
            key, value = (
                V2_CONSOLIDATED_METADATA_KEY,
                encode_cbor_xz({"zarr_consolidated_format": 1, "metadata": metadata}),
            )
        run(store.set(key, default_buffer_prototype().buffer.from_bytes(value)))


backend = Zarr3Backend()
