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

"""zarr-python 2 backend of :mod:`ncore.impl.data.nodes` and :mod:`ncore.impl.data.stores`"""

from __future__ import annotations

import itertools
import json

from typing import TYPE_CHECKING, Iterator, List, Literal, Optional, Tuple, Union

import numcodecs
import numpy as np
import zarr
import zarr.errors
import zarr.indexing
import zarr.storage

from numcodecs.compat import ensure_bytes
from typing_extensions import override
from upath import UPath
from zarr._storage.store import BaseStore, Store

from ncore.impl.common.async_util import run_in_thread
from ncore.impl.data.backend import (
    V2_CONSOLIDATED_METADATA_KEY,
    Attributes,
    Backend,
    BloscCompression,
    EncodedChunks,
    FillValue,
    ReadResult,
    Selection,
    ZarrFormat,
    ZarrNode,
)
from ncore.impl.data.encoding import decode_cbor_xz, encode_cbor_xz
from ncore.impl.data.itar import IndexedTarArchive, is_node_metadata_key
from ncore.impl.data.json_types import to_json_document


if TYPE_CHECKING:
    import numpy.typing as npt


class _IndexedTarStore(Store):
    """zarr-python 2 store of the records of an :class:`IndexedTarArchive`"""

    _erasable = False

    def __init__(self, archive: IndexedTarArchive) -> None:
        self.archive = archive

    @override
    def __getitem__(self, key: str) -> bytes:
        return self.archive[key]

    @override
    def __setitem__(self, key: str, value: object) -> None:
        if self.archive.mode != "w":
            raise zarr.errors.ReadOnlyError()
        self.archive[key] = ensure_bytes(value)

    @override
    def __delitem__(self, key: str) -> None:
        raise NotImplementedError("IndexedTarArchive: records can't be deleted")

    @override
    def __contains__(self, key: object) -> bool:
        return key in self.archive

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self.archive.keys())

    @override
    def __len__(self) -> int:
        return len(self.archive)

    @override
    def __eq__(self, value: object) -> bool:
        return isinstance(value, _IndexedTarStore) and self.archive == value.archive

    def __hash__(self) -> int:
        return hash(self.archive)

    @override
    def close(self) -> None:
        self.archive.close()


class _CompressedConsolidatedStore(zarr.storage.ConsolidatedMetadataStore):
    """A store serving all node metadata from the compressed consolidated metadata of the wrapped store"""

    def __init__(self, store: BaseStore) -> None:
        self.store = store
        consolidated = decode_cbor_xz(store[V2_CONSOLIDATED_METADATA_KEY])
        if (consolidated_format := consolidated.get("zarr_consolidated_format")) != 1:
            raise zarr.errors.MetadataError(f"unsupported zarr consolidated metadata format: {consolidated_format}")
        self.meta_store = zarr.storage.KVStore(consolidated["metadata"])


class Zarr2Backend(Backend[BaseStore]):
    """zarr-python 2 implementation of the version specific operations of nodes and stores"""

    supported_zarr_formats: Tuple[ZarrFormat, ...] = (2,)

    # -- nodes -----------------------------------------------------------------

    @override
    def zarr_format(self, node: ZarrNode) -> ZarrFormat:
        return 2

    @override
    def open_member(self, group: zarr.Group, path: str) -> ZarrNode:
        return group[path]

    @override
    def member_names(self, group: zarr.Group) -> List[str]:
        return list(group.keys())

    @override
    def has_member(self, group: zarr.Group, name: str) -> bool:
        return name in group

    @override
    def member_groups(self, group: zarr.Group) -> List[Tuple[str, zarr.Group]]:
        return list(group.groups())

    @override
    def create_group(self, group: zarr.Group, name: str, attributes: Optional[Attributes]) -> zarr.Group:
        child = group.create_group(name)
        if attributes is not None:
            child.attrs.update(to_json_document(attributes))  # (writes .zattrs once)
        return child

    @override
    def update_attributes(self, node: ZarrNode, attributes: Attributes) -> None:
        node.attrs.update(to_json_document(attributes))

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
        array = group.create_dataset(
            name,
            data=data,
            chunks=True if chunks == "auto" else chunks,
            compressor=None if compression is None else compression.to_numcodecs(),
        )
        if attributes is not None:
            array.attrs.update(to_json_document(attributes))
        return array

    @override
    def create_empty_array(
        self,
        group: zarr.Group,
        name: str,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        chunks: Tuple[int, ...],
        compression: Optional[BloscCompression],
        fill_value: FillValue,
        attributes: Optional[Attributes],
    ) -> zarr.Array:
        array = group.create_dataset(
            name,
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            compressor=None if compression is None else compression.to_numcodecs(),
            fill_value=fill_value,
        )
        if attributes is not None:
            array.attrs.update(to_json_document(attributes))
        return array

    @override
    def encode_chunks(
        self,
        data: npt.NDArray[np.generic],
        zarr_format: ZarrFormat,
        chunks: Union[Tuple[int, ...], Literal["auto"]],
        compression: Optional[BloscCompression],
    ) -> EncodedChunks:
        if zarr_format != 2:
            raise ValueError(f"zarr-python 2 only supports zarr format v2, got v{zarr_format}")
        store = zarr.storage.KVStore(dict())
        array = self.create_array(zarr.group(store=store), "a", data, chunks, compression, None)
        records = {}
        for coords in itertools.product(*(range(n) for n in array.cdata_shape)):
            if (record := store.get(array._chunk_key(coords))) is not None:
                records[coords] = ensure_bytes(record)
        # (zarr-python 2 reports Python scalars for some dtypes, e.g. str for "<U")
        return EncodedChunks(tuple(array.chunks), np.array([array.fill_value], dtype=array.dtype)[0], records)

    @staticmethod
    @override
    def chunk_grid_shape(shape: Tuple[int, ...], chunks: Tuple[int, ...]) -> Tuple[int, ...]:
        # (zarr-python 2 stores 0-dimensional arrays as a single chunk at coordinates (0,))
        return tuple(-(-size // chunk) for size, chunk in zip(shape, chunks)) or (1,)

    @override
    def write_chunk_record(self, array: zarr.Array, coords: Tuple[int, ...], record: bytes) -> None:
        array.chunk_store[array._chunk_key(coords)] = record

    @override
    def write(self, array: zarr.Array, selection: Selection, data: npt.ArrayLike) -> None:
        # (zarr-python 2 stores chunks equal to the fill value by default, as create_array does)
        array[selection] = data

    @override
    def read(self, array: zarr.Array, selection: Selection, out: Optional[np.ndarray] = None) -> ReadResult:
        """Fast path: complete reads of arrays stored as a single uncompressed chunk are views of the chunk record (no
        copies, see :meth:`_single_chunk_record`). Fallback: zarr-python's read (e.g., of compressed or multi-chunk
        arrays, partial reads, and chunks not stored, which read as the fill value). Reads into ``out`` are
        zarr-python's reads into ``out``."""
        if out is not None:
            Backend.check_read_out(out, zarr.indexing.BasicIndexer(selection, array).shape, array.dtype)
            # (zarr-python 2 reads into numpy arrays, type checking runs against zarr-python 3's NDBuffer signature)
            array.get_basic_selection(selection, out=out)  # ty: ignore[invalid-argument-type]
            return out
        if (record := self._single_chunk_record(array, selection)) is not None and (
            view := self.chunk_view(record, array.dtype, array.shape)
        ) is not None:
            return view
        return self.read_result(array[selection])

    @staticmethod
    def _single_chunk_record(array: zarr.Array, selection: Selection) -> Optional[bytes]:
        """The chunk record of complete reads of arrays stored as a single uncompressed chunk of the array's shape
        (None otherwise, also for chunks not stored, which read as the fill value)"""
        if selection not in (Ellipsis, ()) or array.compressor is not None or array.filters or array.order != "C":
            return None
        if tuple(array.chunks) != tuple(array.shape) and not (array.shape == () and array.chunks == (1,)):
            return None
        if not array.dtype.isnative or array.dtype.kind not in "biufS":
            return None
        try:
            record = array.chunk_store[array._chunk_key((0,) * max(array.ndim, 1))]
        except KeyError:
            return None
        return record if isinstance(record, bytes) else None

    @override
    async def read_async(self, array: zarr.Array, selection: Selection, out: Optional[np.ndarray] = None) -> ReadResult:
        """Asynchronous :meth:`read`, returning the same results (zarr-python 2 is synchronous: reads run on worker
        threads, so the running event loop isn't blocked)"""
        return await run_in_thread(self.read, array, selection, out)

    @override
    def compression(self, array: zarr.Array) -> Optional[BloscCompression]:
        if (compressor := array.compressor) is None:
            return None
        if not isinstance(compressor, numcodecs.Blosc):
            raise ValueError(f"Unsupported compressor {compressor}")
        return BloscCompression.from_numcodecs(compressor)

    # -- stores ----------------------------------------------------------------

    @override
    def zarr_store(self, store: object) -> BaseStore:
        if not isinstance(store, BaseStore):
            raise TypeError(f"Not a zarr-python 2 store: {store!r}")
        return store

    @override
    def indexed_tar_store(self, archive: IndexedTarArchive) -> BaseStore:
        return _IndexedTarStore(archive)

    @override
    def directory_store(self, path: UPath) -> BaseStore:
        return zarr.storage.DirectoryStore(path.path)

    @override
    def create_root_group(self, store: BaseStore, zarr_format: ZarrFormat, attributes: Attributes) -> zarr.Group:
        if zarr_format != 2:
            raise ValueError(f"zarr-python 2 only supports zarr format v2, got v{zarr_format}")
        root = zarr.group(store=store)
        root.attrs.put(dict(to_json_document(attributes)))
        return root

    @override
    def open_root_group(self, store: BaseStore, consolidated: bool) -> zarr.Group:
        if consolidated:
            return zarr.open_group(store=_CompressedConsolidatedStore(store), chunk_store=store, mode="r")
        return zarr.open_group(store=store, mode="r")

    @override
    def consolidate(self, store: BaseStore) -> None:
        metadata = {key: json.loads(store[key]) for key in store if is_node_metadata_key(key)}
        store[V2_CONSOLIDATED_METADATA_KEY] = encode_cbor_xz({"zarr_consolidated_format": 1, "metadata": metadata})


backend = Zarr2Backend()
