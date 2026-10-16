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

import json

from typing import TYPE_CHECKING, Iterator, List, Literal, Optional, Tuple, Union

import numcodecs
import numpy as np
import zarr
import zarr.errors
import zarr.storage

from numcodecs.compat import ensure_bytes
from typing_extensions import override
from upath import UPath
from zarr._storage.store import BaseStore, Store

from ncore.impl.data.backend import (
    V2_CONSOLIDATED_METADATA_KEY,
    Attributes,
    Backend,
    BloscCompression,
    ReadResult,
    Selection,
    ZarrNode,
)
from ncore.impl.data.encoding import decode_cbor_xz, encode_cbor_xz
from ncore.impl.data.itar import IndexedTarArchive, is_node_metadata_key


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

    supported_zarr_formats: Tuple[int, ...] = (2,)

    # -- nodes -----------------------------------------------------------------

    @override
    def zarr_format(self, node: ZarrNode) -> int:
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
            child.attrs.update(attributes)  # (writes .zattrs once)
        return child

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
        array = group.create_dataset(
            name,
            data=data,
            chunks=True if chunks == "auto" else chunks,
            compressor=None if compression is None else compression.to_numcodecs(),
        )
        if attributes is not None:
            array.attrs.update(attributes)
        return array

    @override
    def read(self, array: zarr.Array, selection: Selection) -> ReadResult:
        """Fast path: complete reads of arrays stored as a single uncompressed chunk are views of the chunk record (no
        copies, see :meth:`_single_chunk_record`). Fallback: zarr-python's read (e.g., of compressed or multi-chunk
        arrays, partial reads, and chunks not stored, which read as the fill value)"""
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
    def create_root_group(self, store: BaseStore, zarr_format: Literal[2, 3], attributes: Attributes) -> zarr.Group:
        if zarr_format != 2:
            raise ValueError(f"zarr-python 2 only supports zarr format v2, got v{zarr_format}")
        root = zarr.group(store=store)
        root.attrs.put(dict(attributes))
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
