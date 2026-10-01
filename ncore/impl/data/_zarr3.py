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

"""zarr-python 3 backend of ncore nodes and stores (use via :mod:`ncore.impl.data.nodes` / :mod:`.stores`).

Only imported if zarr-python 3 is installed. All zarr-python 3 specific APIs are bound to the protocols of
:mod:`ncore.impl.data._zarr_api` (symmetric to the zarr-python 2 backend).

zarr-python 3 executes all I/O as coroutines. Its synchronous API submits each call to a global event loop running on
a dedicated I/O thread, and codecs additionally hand off (de)compression to worker threads. For many small accesses
(as typical for reading individual frames) these thread hand-offs dominate. The backend instead executes zarr-python's
coroutines on an event loop owned by the *calling* thread (see :func:`run`), which runs the same zarr-python code
paths without thread hand-offs, and lets multiple reader threads decode data concurrently.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import importlib
import json
import os
import threading
import warnings

from typing import (
    TYPE_CHECKING,
    AsyncIterator,
    Coroutine,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)

import numcodecs
import zarr

from ncore.impl.data import _zarr_api as api
from ncore.impl.data._itar import (
    INTERNAL_KEYS,
    V2_CONSOLIDATED_METADATA_KEY,
    ZARR_JSON_KEY,
    ZMETADATA_KEY,
    IndexedTarStore,
    JsonDocument,
    compress_document,
    decompress_document,
    is_v2_metadata_key,
    is_v3_metadata_key,
)


if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt  # type: ignore[import-not-found]


# zarr-python 3 specific APIs
_zarr = cast(api.Zarr3Module, zarr)
_store_module = cast(api.Zarr3StoreModule, importlib.import_module("zarr.abc.store"))
_buffer_module = cast(api.Zarr3BufferModule, importlib.import_module("zarr.core.buffer"))
_storage = cast(api.Zarr3Storage, importlib.import_module("zarr.storage"))
_async_api = cast(api.Zarr3AsyncApi, importlib.import_module("zarr.api.asynchronous"))
_codecs = cast(api.Zarr3Codecs, importlib.import_module("zarr.codecs"))
_sync = cast(api.Zarr3Sync, importlib.import_module("zarr.core.sync"))

# zarr-python 3 store base classes
_StoreBase = cast(Type[api.Zarr3StoreBase], importlib.import_module("zarr.abc.store").Store)
_WrapperStoreBase = cast(Type[api.Zarr3WrapperStoreBase], importlib.import_module("zarr.storage").WrapperStore)


def _group(group: zarr.Group) -> api.Zarr3Group:
    return cast(api.Zarr3Group, group)


def _array(array: zarr.Array) -> api.Zarr3Array:
    return cast(api.Zarr3Array, array)


def _zarr_store(store: api.Zarr3Store) -> api.ZarrStore:
    return cast(api.ZarrStore, store)


# -----------------------------------------------------------------------------
# Execution of coroutines in the calling thread
# -----------------------------------------------------------------------------


class _InlineExecutor(concurrent.futures.ThreadPoolExecutor):
    """Executor running submitted work immediately in the submitting thread"""

    def __init__(self) -> None:
        super().__init__(max_workers=1)

    def submit(self, fn, /, *args, **kwargs):  # type: ignore[override]
        future: concurrent.futures.Future = concurrent.futures.Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as e:  # forwarded to the awaiting coroutine
            future.set_exception(e)
        return future


class _ThreadLoop(threading.local):
    loop: Optional[asyncio.AbstractEventLoop] = None
    pid: int = -1


_thread_loop = _ThreadLoop()


def run(coroutine: Coroutine[object, object, api.T]) -> api.T:
    """Runs a zarr-python 3 coroutine to completion on an event loop owned by the calling thread.

    The loop's default executor runs offloaded work (e.g., codecs) inline. Loops are re-created in forked processes.
    Falls back to zarr-python's global I/O loop if called from within a running event loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        state = _thread_loop
        if state.loop is None or state.pid != os.getpid():
            state.loop = asyncio.new_event_loop()
            state.loop.set_default_executor(_InlineExecutor())
            state.pid = os.getpid()
        return state.loop.run_until_complete(coroutine)

    return _sync.sync(coroutine)


# -----------------------------------------------------------------------------
# Indexed tar store adapter
# -----------------------------------------------------------------------------


def _byte_range(byte_range: Optional[api.Zarr3ByteRequest], size: Optional[int]) -> Tuple[int, Optional[int]]:
    """Maps a zarr-python byte request to a (start, length) sub-range of a record with the given size"""
    if byte_range is None:
        return 0, None
    if isinstance(byte_range, _store_module.RangeByteRequest):
        return byte_range.start, byte_range.end - byte_range.start
    if isinstance(byte_range, _store_module.OffsetByteRequest):
        return byte_range.offset, None
    if isinstance(byte_range, _store_module.SuffixByteRequest):
        return max(0, (size or 0) - byte_range.suffix), None
    raise TypeError(f"Unexpected byte_range, got {byte_range}.")


def _buffer(prototype: Optional[api.Zarr3BufferPrototype], value: Optional[bytes]) -> Optional[api.Zarr3Buffer]:
    if value is None:
        return None
    return (prototype or _buffer_module.default_buffer_prototype()).buffer.from_bytes(value)


class _TarStoreAdapter(_StoreBase):
    """zarr-python 3 store of an :class:`IndexedTarStore` (forwards all operations to it)"""

    supports_writes: bool = True
    supports_deletes: bool = False
    supports_partial_writes: Literal[False] = False
    supports_listing: bool = True

    def __init__(self, store: IndexedTarStore) -> None:
        super().__init__(read_only=store.mode == "r")
        self.store = store
        self._is_open = True
        if store.mode == "w":
            # zarr-python 3 rewrites node metadata on every update
            store.defer_metadata()

    def __eq__(self, value: object) -> bool:
        return isinstance(value, _TarStoreAdapter) and self.store == value.store

    def __hash__(self) -> int:
        return hash(self.store)

    def __repr__(self) -> str:
        return f"_TarStoreAdapter({self.store!r})"

    def with_read_only(self, read_only: bool = False) -> _TarStoreAdapter:
        """Returns the store if the access mode matches (access modes of indexed tar stores can't be changed)"""
        if read_only != self.read_only:
            raise NotImplementedError("IndexedTarStore access mode can't be changed after opening")
        return self

    async def _open(self) -> None:
        self._is_open = True  # opened in the constructor

    def close(self) -> None:
        """Closes the indexed tar store"""
        self.store.close()
        self._is_open = False

    def _read(self, key: str, byte_range: Optional[api.Zarr3ByteRequest]) -> Optional[bytes]:
        start, length = _byte_range(byte_range, self.store.record_size(key))
        return self.store.get(key, start, length)

    async def get(
        self,
        key: str,
        prototype: api.Zarr3BufferPrototype,
        byte_range: Optional[api.Zarr3ByteRequest] = None,
    ) -> Optional[api.Zarr3Buffer]:
        """Reads (a byte range of) a record, see :meth:`IndexedTarStore.get_async` (asynchronous range requests for
        chunks of remote archives, synchronous reads otherwise)"""
        start, length = _byte_range(byte_range, self.store.record_size(key))
        return _buffer(prototype, await self.store.get_async(key, start, length))

    async def get_partial_values(
        self,
        prototype: api.Zarr3BufferPrototype,
        key_ranges: Iterable[Tuple[str, Optional[api.Zarr3ByteRequest]]],
    ) -> List[Optional[api.Zarr3Buffer]]:
        """Reads (byte ranges of) multiple records concurrently, see :meth:`get`"""
        return list(await asyncio.gather(*(self.get(key, prototype, byte_range) for key, byte_range in key_ranges)))

    async def exists(self, key: str) -> bool:
        """True if a record exists (synchronous lookup)"""
        return self.store.exists(key)

    async def set(self, key: str, value: api.Zarr3Buffer) -> None:
        """Writes a record, see :meth:`IndexedTarStore.set` (synchronous)"""
        self._check_writable()
        self.store.set(key, value.to_bytes())

    async def set_if_not_exists(self, key: str, value: api.Zarr3Buffer) -> None:
        """Writes a record if it doesn't exist yet (synchronous)"""
        if not self.store.exists(key):
            await self.set(key, value)

    async def delete(self, key: str) -> None:
        """No-op for non-existing records (issued by zarr-python when creating nodes), deleting existing records is
        not supported"""
        self.delete_sync(key)

    async def delete_dir(self, prefix: str) -> None:
        """No-op for non-existing prefixes (issued by zarr-python when creating nodes), deleting existing records is
        not supported"""
        self._check_writable()
        prefix = prefix if prefix == "" or prefix.endswith("/") else prefix + "/"
        if any(key.startswith(prefix) for key in self.store.keys()):
            raise NotImplementedError("Deleting records is not supported by IndexedTarStore")

    async def list(self) -> AsyncIterator[str]:
        """Lists the keys of all records (excluding internal records, synchronous)"""
        for key in self.store.keys():
            if key not in INTERNAL_KEYS:
                yield key

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        """Lists the keys of all records starting with a prefix (excluding internal records, synchronous)"""
        for key in self.store.keys():
            if key.startswith(prefix) and key not in INTERNAL_KEYS:
                yield key

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        """Lists the names of the direct children of a prefix (excluding internal records, synchronous)"""
        for child in self.store.list_dir(prefix):
            yield child

    # synchronous store protocol (zarr.abc.store.SupportsSyncStore, used by fast paths of zarr-python>=3.4)

    def get_sync(
        self,
        key: str,
        *,
        prototype: Optional[api.Zarr3BufferPrototype] = None,
        byte_range: Optional[api.Zarr3ByteRequest] = None,
    ) -> Optional[api.Zarr3Buffer]:
        """Synchronously reads (a byte range of) a record, see :meth:`IndexedTarStore.get`"""
        return _buffer(prototype, self._read(key, byte_range))

    def set_sync(self, key: str, value: api.Zarr3Buffer) -> None:
        """Synchronously writes a record, see :meth:`IndexedTarStore.set`"""
        self._check_writable()
        self.store.set(key, value.to_bytes())

    def delete_sync(self, key: str) -> None:
        """Synchronous version of :meth:`delete`"""
        self._check_writable()
        if self.store.exists(key):
            raise NotImplementedError("Deleting records is not supported by IndexedTarStore")


class _OverlayStore(_WrapperStoreBase):
    """Read-through store wrapper serving additional in-memory records"""

    def __init__(self, store: api.Zarr3Store, overlay: Dict[str, bytes]) -> None:
        super().__init__(store)
        self._overlay = overlay

    async def get(
        self,
        key: str,
        prototype: api.Zarr3BufferPrototype,
        byte_range: Optional[api.Zarr3ByteRequest] = None,
    ) -> Optional[api.Zarr3Buffer]:
        """Reads a record from the overlay, or from the wrapped store"""
        if (value := self._overlay.get(key)) is not None:
            start, length = _byte_range(byte_range, len(value))
            return _buffer(prototype, value[start:] if length is None else value[start : start + length])
        return await self._store.get(key, prototype, byte_range)

    async def exists(self, key: str) -> bool:
        """True if a record exists in the overlay, or in the wrapped store"""
        return key in self._overlay or await self._store.exists(key)

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        """Lists the names of the direct children of a prefix of the wrapped store (excluding internal records)"""
        async for key in self._store.list_dir(prefix):
            if key not in INTERNAL_KEYS:
                yield key


# -----------------------------------------------------------------------------
# Backend
# -----------------------------------------------------------------------------


def _store(store: api.StoreLike) -> api.Zarr3Store:
    """The zarr-python 3 store of a store (wraps indexed tar stores)"""
    if isinstance(store, IndexedTarStore):
        return cast(api.Zarr3Store, _TarStoreAdapter(store))
    return cast(api.Zarr3Store, store)


def _store_keys(store: api.Zarr3Store) -> List[str]:
    if isinstance(store, _TarStoreAdapter):
        return store.store.keys()

    async def collect() -> List[str]:
        return [key async for key in store.list()]

    return run(collect())


def _store_get(store: api.Zarr3Store, key: str) -> Optional[bytes]:
    if isinstance(store, _TarStoreAdapter):
        return store.store.get(key)
    value = run(store.get(key, prototype=_buffer_module.default_buffer_prototype()))
    return None if value is None else value.to_bytes()


def _open_group(
    store: api.Zarr3Store, zarr_format: Literal[2, 3], mode: Literal["r", "r+"], use_consolidated: Optional[bool]
) -> zarr.Group:
    with warnings.catch_warnings():
        # zarr format v3 consolidated metadata is not part of the zarr specification yet
        warnings.filterwarnings("ignore", message="Consolidated metadata is currently not part")
        node = run(
            _async_api.open_group(store=store, zarr_format=zarr_format, mode=mode, use_consolidated=use_consolidated)
        )
    return _zarr.Group(node)


class _Backend:
    """zarr-python 3 implementation of :class:`ncore.impl.data._zarr_api.Backend`"""

    supported_zarr_formats: Tuple[int, ...] = (2, 3)

    def run(self, coroutine: Coroutine[object, object, api.T]) -> api.T:
        return run(coroutine)

    # -- nodes -----------------------------------------------------------------

    def zarr_format(self, node: Union[zarr.Group, zarr.Array]) -> int:
        if isinstance(node, zarr.Group):
            return _group(node).metadata.zarr_format
        return _array(node).metadata.zarr_format

    def open_member(self, group: zarr.Group, path: str) -> Union[zarr.Group, zarr.Array]:
        node = run(_group(group)._async_group.getitem(path))
        return _zarr.Group(node) if isinstance(node, _zarr.AsyncGroup) else _zarr.Array(node)

    def member_names(self, group: zarr.Group) -> List[str]:
        if (consolidated := _group(group).metadata.consolidated_metadata) is not None:
            return list(consolidated.metadata.keys())
        with _ignore_foreign_objects():
            return list(_group(group).keys())

    def has_member(self, group: zarr.Group, name: str) -> bool:
        if (consolidated := _group(group).metadata.consolidated_metadata) is not None:
            return name in consolidated.metadata
        return name in _group(group)

    def member_groups(self, group: zarr.Group) -> List[Tuple[str, zarr.Group]]:
        with _ignore_foreign_objects():
            return list(_group(group).groups())

    def create_group(self, group: zarr.Group, name: str, attributes: Optional[api.Attributes]) -> zarr.Group:
        return _group(group).create_group(name, attributes=None if attributes is None else dict(attributes))

    def create_array(
        self,
        group: zarr.Group,
        name: str,
        data: npt.NDArray[np.generic],
        chunks: Union[Tuple[int, ...], Literal["auto"]],
        compression: Optional[api.BloscParameters],
        attributes: Optional[api.Attributes],
    ) -> zarr.Array:
        compressor: Union[numcodecs.Blosc, api.Zarr3Codec, None] = None
        if compression is not None:
            if self.zarr_format(group) == 3:
                cname, clevel, shuffle, blocksize = compression
                compressor = _codecs.BloscCodec(cname=cname, clevel=clevel, shuffle=shuffle, blocksize=blocksize)
            else:
                compressor = api.numcodecs_blosc(compression)
        return _group(group).create_array(
            name,
            data=data,
            chunks=chunks,
            compressors=compressor,
            attributes=None if attributes is None else dict(attributes),
        )

    def read(self, array: zarr.Array, selection: api.Selection) -> object:
        return run(_array(array).async_array.getitem(selection))

    def compression(self, array: zarr.Array) -> Optional[api.BloscParameters]:
        compressors = _array(array).compressors
        if not compressors:
            return None
        if len(compressors) != 1:
            raise ValueError(f"Unsupported compressor chain {compressors}")
        if isinstance(compressor := compressors[0], numcodecs.Blosc):  # numcodecs codec (zarr format v2)
            return api.blosc_parameters(compressor.get_config(), shuffle_ids=True)
        codec = cast(api.Zarr3Codec, compressor).to_dict()  # zarr-python 3 codec (zarr format v3)
        if codec.get("name") != "blosc" or not isinstance(config := codec.get("configuration"), dict):
            raise ValueError(f"Unsupported compressor {compressor}")
        return api.blosc_parameters(cast(Dict[str, object], config), shuffle_ids=False)

    # -- stores ----------------------------------------------------------------

    def open_directory_store(self, path: str, mode: Literal["r", "w"]) -> api.ZarrStore:
        store = _storage.LocalStore(path, read_only=mode == "r")
        if mode == "w":
            run(store.delete_dir(""))
        return _zarr_store(store)

    def create_root_group(
        self, store: api.StoreLike, zarr_format: Literal[2, 3], attributes: api.Attributes
    ) -> zarr.Group:
        node = run(_async_api.create_group(store=_store(store), zarr_format=zarr_format, attributes=dict(attributes)))
        return _zarr.Group(node)

    def open_store(self, store: api.StoreLike, open_consolidated: bool) -> zarr.Group:
        if isinstance(store, IndexedTarStore) and store.mode == "r":
            if (zarr_format := store.zarr_format) is not None and open_consolidated:
                # node metadata is served from the consolidated metadata, parsed lazily on access
                store.prefetch_metadata()
                return _open_group(_store(store), zarr_format, mode="r", use_consolidated=False)
            if open_consolidated:
                raise KeyError(V2_CONSOLIDATED_METADATA_KEY)  # same as the zarr-python 2 backend

        zarr_store = _store(store)
        if _store_get(zarr_store, ZARR_JSON_KEY) is not None:
            return _open_group(zarr_store, 3, mode="r", use_consolidated=None if open_consolidated else False)
        if open_consolidated:
            return self.open_compressed_consolidated(store, V2_CONSOLIDATED_METADATA_KEY, "r")
        return _open_group(zarr_store, 2, mode="r", use_consolidated=False)

    def open_compressed_consolidated(
        self, store: api.StoreLike, metadata_key: str, mode: Literal["r", "r+"]
    ) -> zarr.Group:
        if isinstance(store, IndexedTarStore) and store.mode == "r" and metadata_key == V2_CONSOLIDATED_METADATA_KEY:
            if store.zarr_format != 2:
                raise KeyError(metadata_key)
            return _open_group(_store(store), 2, mode=mode, use_consolidated=True)

        zarr_store = _store(store)
        if (compressed := _store_get(zarr_store, metadata_key)) is None:
            raise KeyError(metadata_key)

        # serve the decompressed document under zarr-python's consolidated metadata key
        document = json.dumps(decompress_document(compressed)).encode("utf-8")
        overlay = cast(api.Zarr3Store, _OverlayStore(zarr_store, {ZMETADATA_KEY: document}))
        return _open_group(overlay, 2, mode=mode, use_consolidated=True)

    def consolidate_store(self, store: api.StoreLike) -> None:
        zarr_store = _store(store)
        if _store_get(zarr_store, ZARR_JSON_KEY) is None:
            self.consolidate_compressed_metadata(store, V2_CONSOLIDATED_METADATA_KEY)
            return

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Consolidated metadata is currently not part")
            run(_async_api.consolidate_metadata(zarr_store, zarr_format=3))

    def consolidate_compressed_metadata(self, store: api.StoreLike, metadata_key: str) -> None:
        zarr_store = _store(store)
        keys = _store_keys(zarr_store)
        metadata: JsonDocument = {}
        for key in keys:
            if is_v2_metadata_key(key) and (value := _store_get(zarr_store, key)) is not None:
                metadata[key] = json.loads(value)
        if not metadata and any(is_v3_metadata_key(key) for key in keys):
            raise NotImplementedError("Only supporting V2 stores")

        value = compress_document({"zarr_consolidated_format": 1, "metadata": metadata})
        if isinstance(store, IndexedTarStore):
            store[metadata_key] = value  # raw record
        else:
            buffer = _buffer_module.default_buffer_prototype().buffer.from_bytes(value)
            run(zarr_store.set(metadata_key, buffer))

    def close_store(self, store: api.StoreLike) -> None:
        if isinstance(store, IndexedTarStore):
            store.close()
        else:
            _store(store).close()

    def group_store(self, group: zarr.Group) -> api.StoreLike:
        store: object = _group(group).store
        while isinstance(store, _OverlayStore):
            store = store._store
        if isinstance(store, _TarStoreAdapter):
            return store.store
        return cast(api.ZarrStore, store)


@contextlib.contextmanager
def _ignore_foreign_objects() -> Iterator[None]:
    """zarr-python 3 warns about non-zarr records in groups (e.g., compressed consolidated metadata records)"""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Object at .* is not recognized as a component")
        yield


backend = _Backend()
