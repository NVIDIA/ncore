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

"""zarr-python 2 backend of ncore nodes and stores (use via :mod:`ncore.impl.data.nodes` / :mod:`.stores`).

Only imported if zarr-python 2 is installed. All zarr-python 2 specific APIs are bound to the protocols of
:mod:`ncore.impl.data._zarr_api` (symmetric to the zarr-python 3 backend).
"""

from __future__ import annotations

import importlib
import json

from typing import TYPE_CHECKING, Coroutine, Dict, Iterator, List, Literal, Optional, Tuple, Union, cast

import numcodecs
import zarr

from numcodecs.compat import ensure_bytes

from ncore.impl.data import _zarr_api as api
from ncore.impl.data._itar import (
    INTERNAL_KEYS,
    V2_CONSOLIDATED_METADATA_KEY,
    IndexedTarStore,
    JsonDocument,
    compress_document,
    decompress_document,
    is_v2_metadata_key,
)


if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt  # type: ignore[import-not-found]


# zarr-python 2 specific APIs
_zarr = cast(api.Zarr2Module, zarr)
_storage = cast(api.Zarr2Storage, importlib.import_module("zarr.storage"))
_errors = cast(api.Zarr2Errors, importlib.import_module("zarr.errors"))

# zarr-python 2 store base classes
_StoreBase = cast(api.Zarr2StoreModule, importlib.import_module("zarr._storage.store")).Store
_ConsolidatedMetadataStoreBase = cast(api.Zarr2StorageBases, _storage).ConsolidatedMetadataStore


def _group(group: zarr.Group) -> api.Zarr2Group:
    return cast(api.Zarr2Group, group)


def _array(array: zarr.Array) -> api.Zarr2Array:
    return cast(api.Zarr2Array, array)


# -----------------------------------------------------------------------------
# Indexed tar store adapter
# -----------------------------------------------------------------------------


class _TarStoreAdapter(_StoreBase):
    """zarr-python 2 store of an :class:`IndexedTarStore` (forwards all operations to it)"""

    _erasable = False

    def __init__(self, store: IndexedTarStore) -> None:
        self.store = store

    def __getitem__(self, key: str) -> bytes:
        return self.store[key]

    def __setitem__(self, key: str, value: object) -> None:
        if self.store.mode != "w":
            raise _errors.ReadOnlyError()
        self.store[key] = ensure_bytes(value)

    def __delitem__(self, _: str) -> None:
        raise NotImplementedError("Deleting records is not supported by IndexedTarStore")

    def __contains__(self, key: object) -> bool:
        return key in self.store

    def __iter__(self) -> Iterator[str]:
        return (key for key in self.store.keys() if key not in INTERNAL_KEYS)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __eq__(self, value: object) -> bool:
        return isinstance(value, _TarStoreAdapter) and self.store == value.store

    def __hash__(self) -> int:
        return hash(self.store)

    def listdir(self, path: str = "") -> List[str]:
        """Names of the direct children of a path"""
        return sorted(self.store.list_dir(path))

    def close(self) -> None:
        """Closes the indexed tar store"""
        self.store.close()


class _ConsolidatedCompressedMetadataStore(_ConsolidatedMetadataStoreBase):
    """A layer over other storage, where the metadata has been consolidated into a single compressed key"""

    def __init__(self, store: api.Zarr2Store, metadata_key: str = V2_CONSOLIDATED_METADATA_KEY) -> None:
        self.store = store

        # retrieve and check format of consolidated metadata
        meta = decompress_document(self.store[metadata_key])
        if (consolidated_format := meta.get("zarr_consolidated_format")) != 1:
            raise _errors.MetadataError(f"unsupported zarr consolidated metadata format: {consolidated_format}")

        self.meta_store = _storage.KVStore(cast(Dict[str, object], meta["metadata"]))


# -----------------------------------------------------------------------------
# Backend
# -----------------------------------------------------------------------------


def _store(store: api.StoreLike) -> api.Zarr2Store:
    """The zarr-python 2 store of a store (wraps indexed tar stores)"""
    if isinstance(store, IndexedTarStore):
        return cast(api.Zarr2Store, _TarStoreAdapter(store))
    return cast(api.Zarr2Store, store)


class _Backend:
    """zarr-python 2 implementation of :class:`ncore.impl.data._zarr_api.Backend`"""

    supported_zarr_formats: Tuple[int, ...] = (2,)

    def run(self, coroutine: Coroutine[object, object, api.T]) -> api.T:
        coroutine.close()
        raise NotImplementedError("Coroutines are only supported with zarr-python 3")

    # -- nodes -----------------------------------------------------------------

    def zarr_format(self, node: Union[zarr.Group, zarr.Array]) -> int:
        return 2

    def open_member(self, group: zarr.Group, path: str) -> Union[zarr.Group, zarr.Array]:
        return _group(group)[path]

    def member_names(self, group: zarr.Group) -> List[str]:
        return list(_group(group).keys())

    def has_member(self, group: zarr.Group, name: str) -> bool:
        return name in _group(group)

    def member_groups(self, group: zarr.Group) -> List[Tuple[str, zarr.Group]]:
        return list(_group(group).groups())

    def create_group(self, group: zarr.Group, name: str, attributes: Optional[api.Attributes]) -> zarr.Group:
        child = _group(group).create_group(name)
        if attributes is not None:
            _group(child).attrs.put(dict(attributes))
        return child

    def create_array(
        self,
        group: zarr.Group,
        name: str,
        data: npt.NDArray[np.generic],
        chunks: Union[Tuple[int, ...], Literal["auto"]],
        compression: Optional[api.BloscParameters],
        attributes: Optional[api.Attributes],
    ) -> zarr.Array:
        array = _group(group).create_dataset(
            name,
            data=data,
            chunks=True if chunks == "auto" or not data.ndim else chunks,
            compressor=None if compression is None else api.numcodecs_blosc(compression),
        )
        if attributes is not None:
            cast(api.Zarr2Node, array).attrs.put(dict(attributes))
        return array

    def read(self, array: zarr.Array, selection: api.Selection) -> object:
        return _array(array)[selection]

    def compression(self, array: zarr.Array) -> Optional[api.BloscParameters]:
        if (compressor := _array(array).compressor) is None:
            return None
        if not isinstance(compressor, numcodecs.Blosc):
            raise ValueError(f"Unsupported compressor {compressor}")
        return api.blosc_parameters(compressor.get_config(), shuffle_ids=True)

    # -- stores ----------------------------------------------------------------

    def open_directory_store(self, path: str, mode: Literal["r", "w"]) -> api.ZarrStore:
        store = _storage.DirectoryStore(path)
        if mode == "w":
            store.rmdir()
        return cast(api.ZarrStore, store)

    def create_root_group(
        self, store: api.StoreLike, zarr_format: Literal[2, 3], attributes: api.Attributes
    ) -> zarr.Group:
        if zarr_format != 2:
            raise ValueError(f"zarr format v{zarr_format} requires zarr-python>=3 (installed: {zarr.__version__})")
        group = _zarr.group(store=_store(store))
        _group(group).attrs.put(dict(attributes))
        return group

    def open_store(self, store: api.StoreLike, open_consolidated: bool) -> zarr.Group:
        if open_consolidated:
            return self.open_compressed_consolidated(store, V2_CONSOLIDATED_METADATA_KEY, "r")
        return _zarr.open_group(store=_store(store), mode="r")

    def open_compressed_consolidated(
        self, store: api.StoreLike, metadata_key: str, mode: Literal["r", "r+"]
    ) -> zarr.Group:
        zarr_store = _storage.normalize_store_arg(_store(store), mode=mode)
        meta_store = cast(api.Zarr2Store, _ConsolidatedCompressedMetadataStore(zarr_store, metadata_key=metadata_key))
        return _zarr.open(store=meta_store, chunk_store=zarr_store, mode=mode)

    def consolidate_store(self, store: api.StoreLike) -> None:
        self.consolidate_compressed_metadata(store, V2_CONSOLIDATED_METADATA_KEY)

    def consolidate_compressed_metadata(self, store: api.StoreLike, metadata_key: str) -> None:
        zarr_store = _storage.normalize_store_arg(_store(store), mode="w")
        metadata: JsonDocument = {key: _json(zarr_store[key]) for key in zarr_store if is_v2_metadata_key(key)}
        zarr_store[metadata_key] = compress_document({"zarr_consolidated_format": 1, "metadata": metadata})

    def close_store(self, store: api.StoreLike) -> None:
        if isinstance(store, IndexedTarStore):
            store.close()
        else:
            _store(store).close()

    def group_store(self, group: zarr.Group) -> api.StoreLike:
        store = _group(group).chunk_store or _group(group).store
        if isinstance(store, _ConsolidatedCompressedMetadataStore):
            store = store.store
        if isinstance(store, _TarStoreAdapter):
            return store.store
        return cast(api.ZarrStore, store)


def _json(value: bytes) -> JsonDocument:
    return cast(JsonDocument, json.loads(value))


backend = _Backend()
