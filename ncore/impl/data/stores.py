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

"""zarr stores of NCore data: indexed tar archives (``.itar``) and their compressed consolidated metadata"""

from __future__ import annotations

from typing import Iterator, List, Mapping, Union

import zarr

from numcodecs import compat
from zarr._storage.store import Store
from zarr.util import json_loads

from ncore.impl.data._itar import IndexedTarStore, decode_cbor_xz, encode_cbor_xz


#: Default key of the compressed consolidated metadata of zarr hierarchies (CBOR + LZMA encoded ``.zmetadata``)
CONSOLIDATED_METADATA_KEY = ".zmetadata.cbor.xz"


class ZarrIndexedTarStore(Store):
    """zarr-python store of the records of an :class:`IndexedTarStore`"""

    _erasable = False

    def __init__(self, store: IndexedTarStore) -> None:
        self.store = store

    def __getitem__(self, key: str) -> bytes:
        return self.store[key]

    def __setitem__(self, key: str, value: object) -> None:
        if self.store.mode != "w":
            raise zarr.errors.ReadOnlyError()
        self.store[key] = compat.ensure_bytes(value)

    def __delitem__(self, key: str) -> None:
        raise NotImplementedError("IndexedTarStore: records can't be deleted")

    def __contains__(self, key: object) -> bool:
        return key in self.store

    def __iter__(self) -> Iterator[str]:
        return iter(self.store.keys())

    def __len__(self) -> int:
        return len(self.store)

    def __eq__(self, value: object) -> bool:
        return isinstance(value, ZarrIndexedTarStore) and self.store == value.store

    def __hash__(self) -> int:
        return hash(self.store)

    def close(self) -> None:
        self.store.close()


StoreLike = Union[IndexedTarStore, zarr.storage.BaseStore]


def zarr_store(store: StoreLike) -> zarr.storage.BaseStore:
    """Returns the zarr-python store of a store (wraps indexed tar stores)"""
    return ZarrIndexedTarStore(store) if isinstance(store, IndexedTarStore) else store


def consolidate_compressed_metadata(store: StoreLike, metadata_key: str = CONSOLIDATED_METADATA_KEY) -> None:
    """Consolidates the metadata of all groups and arrays of a store into a single compressed record (CBOR + LZMA
    encoded zarr consolidated metadata)

    See Also
    --------
    zarr.consolidate_metadata
    """
    store = zarr.storage.normalize_store_arg(zarr_store(store), mode="w")
    if store._store_version != 2:
        raise NotImplementedError("Only supporting V2 stores")

    metadata = {key: json_loads(store[key]) for key in store if key.endswith((".zarray", ".zgroup", ".zattrs"))}
    store[metadata_key] = encode_cbor_xz({"zarr_consolidated_format": 1, "metadata": metadata})


class ConsolidatedCompressedMetadataStore(zarr.storage.ConsolidatedMetadataStore):
    """A layer over a store, where the metadata of all nodes is read from the compressed consolidated metadata"""

    def __init__(self, store: StoreLike, metadata_key: str = CONSOLIDATED_METADATA_KEY) -> None:
        self.store = Store._ensure_store(zarr_store(store))

        consolidated = decode_cbor_xz(self.store[metadata_key])
        if (consolidated_format := consolidated.get("zarr_consolidated_format")) != 1:
            raise zarr.MetadataError(f"unsupported zarr consolidated metadata format: {consolidated_format}")

        self.meta_store: Mapping[str, object] = zarr.KVStore(consolidated["metadata"])


def open_compressed_consolidated(
    store: StoreLike, metadata_key: str = CONSOLIDATED_METADATA_KEY, mode: str = "r+", **kwargs
) -> zarr.Group:
    """Opens a group of a store via its compressed consolidated metadata

    See Also
    --------
    consolidate_compressed_metadata
    zarr.open_consolidated
    """
    if mode not in {"r", "r+"}:
        raise ValueError(f"invalid mode, expected either 'r' or 'r+'; found {mode!r}")
    meta_store = ConsolidatedCompressedMetadataStore(store, metadata_key=metadata_key)
    chunk_store = kwargs.pop("chunk_store", None) or meta_store.store
    return zarr.open(store=meta_store, chunk_store=chunk_store, mode=mode, **kwargs)


def get_indexed_tar_store(group: zarr.Group) -> Union[IndexedTarStore, None]:
    """Returns the indexed tar store of a group (None for groups of other stores)"""
    store: object = group.store
    if isinstance(store, ConsolidatedCompressedMetadataStore):
        store = store.store
    return store.store if isinstance(store, ZarrIndexedTarStore) else None


__all__: List[str] = [
    "CONSOLIDATED_METADATA_KEY",
    "ConsolidatedCompressedMetadataStore",
    "IndexedTarStore",
    "StoreLike",
    "ZarrIndexedTarStore",
    "consolidate_compressed_metadata",
    "get_indexed_tar_store",
    "open_compressed_consolidated",
    "zarr_store",
]
