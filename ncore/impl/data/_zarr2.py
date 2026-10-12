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

import asyncio
import json

from typing import TYPE_CHECKING, Any, Iterator, List, Literal, Optional, Tuple, Union

import numcodecs
import numpy as np
import zarr

from numcodecs.compat import ensure_bytes

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

    _StoreBase = object
    _ConsolidatedMetadataStoreBase = object
else:
    from zarr._storage.store import Store as _StoreBase
    from zarr.storage import ConsolidatedMetadataStore as _ConsolidatedMetadataStoreBase


#: zarr-python 2 API (untyped, type checking may run against zarr-python 3)
_zarr: Any = zarr


def _z(node: object) -> Any:
    """A zarr-python 2 group / array / store (untyped, see :data:`_zarr`)"""
    return node


class _TarStore(_StoreBase):
    """zarr-python 2 store of the records of an :class:`IndexedTarStore`"""

    _erasable = False

    def __init__(self, store: IndexedTarStore) -> None:
        self.store = store

    def __getitem__(self, key: str) -> bytes:
        return self.store[key]

    def __setitem__(self, key: str, value: object) -> None:
        if self.store.mode != "w":
            raise _zarr.errors.ReadOnlyError()
        self.store[key] = ensure_bytes(value)

    def __delitem__(self, key: str) -> None:
        raise NotImplementedError("IndexedTarStore: records can't be deleted")

    def __contains__(self, key: object) -> bool:
        return key in self.store

    def __iter__(self) -> Iterator[str]:
        return iter(self.store.keys())

    def __len__(self) -> int:
        return len(self.store)

    def __eq__(self, value: object) -> bool:
        return isinstance(value, _TarStore) and self.store == value.store

    def __hash__(self) -> int:
        return hash(self.store)

    def close(self) -> None:
        self.store.close()


class _CompressedConsolidatedStore(_ConsolidatedMetadataStoreBase):
    """A store serving all node metadata from the compressed consolidated metadata of the wrapped store"""

    def __init__(self, store: Any) -> None:
        self.store = store
        consolidated = decode_cbor_xz(store[V2_CONSOLIDATED_METADATA_KEY])
        if (consolidated_format := consolidated.get("zarr_consolidated_format")) != 1:
            raise _zarr.errors.MetadataError(f"unsupported zarr consolidated metadata format: {consolidated_format}")
        self.meta_store = _zarr.storage.KVStore(consolidated["metadata"])


def _zarr_store(store: StoreLike) -> Any:
    """The zarr-python 2 store of a store (wraps indexed tar stores)"""
    return _TarStore(store) if isinstance(store, IndexedTarStore) else store


class Backend:
    """zarr-python 2 implementation of the version specific operations of nodes and stores"""

    supported_zarr_formats: Tuple[int, ...] = (2,)

    # -- nodes -----------------------------------------------------------------

    def zarr_format(self, node: Union[zarr.Group, zarr.Array]) -> int:
        return 2

    def open_member(self, group: zarr.Group, path: str) -> Union[zarr.Group, zarr.Array]:
        return _z(group)[path]

    def member_names(self, group: zarr.Group) -> List[str]:
        return list(_z(group).keys())

    def has_member(self, group: zarr.Group, name: str) -> bool:
        return name in _z(group)

    def member_groups(self, group: zarr.Group) -> List[Tuple[str, zarr.Group]]:
        return list(_z(group).groups())

    def create_group(self, group: zarr.Group, name: str, attributes: Optional[Attributes]) -> zarr.Group:
        child = _z(group).create_group(name)
        if attributes is not None:
            child.attrs.update(attributes_dict(attributes))  # (writes .zattrs once)
        return child

    def require_group(self, group: zarr.Group, name: str) -> zarr.Group:
        return _z(group).require_group(name)

    def create_array(
        self,
        group: zarr.Group,
        name: str,
        data: npt.NDArray[np.generic],
        chunks: Union[Tuple[int, ...], Literal["auto"]],
        compression: Optional[BloscCompression],
        attributes: Optional[Attributes],
    ) -> zarr.Array:
        array = _z(group).create_dataset(
            name,
            data=data,
            chunks=True if chunks == "auto" else chunks,
            compressor=None if compression is None else compression.to_numcodecs(),
        )
        if attributes is not None:
            array.attrs.update(attributes_dict(attributes))
        return array

    def read(self, array: zarr.Array, selection: Selection) -> object:
        return _z(array)[selection]

    async def read_async(self, array: zarr.Array, selection: Selection) -> object:
        # (zarr-python 2 is synchronous: reads run on worker threads, to not block the running event loop)
        return await asyncio.get_running_loop().run_in_executor(None, self.read, array, selection)

    def compression(self, array: zarr.Array) -> Optional[BloscCompression]:
        if (compressor := _z(array).compressor) is None:
            return None
        if not isinstance(compressor, numcodecs.Blosc):
            raise ValueError(f"Unsupported compressor {compressor}")
        return BloscCompression.from_numcodecs(compressor)

    # -- stores ----------------------------------------------------------------

    def open_directory_store(self, path: str) -> ZarrStore:
        return _zarr.storage.DirectoryStore(path)

    def create_root_group(self, store: StoreLike, zarr_format: Literal[2, 3], attributes: Attributes) -> zarr.Group:
        root = _zarr.group(store=_zarr_store(store))
        root.attrs.put(attributes_dict(attributes))
        return root

    def open_store(self, store: StoreLike, consolidated: bool) -> zarr.Group:
        zarr_store = _zarr_store(store)
        if consolidated:
            return _zarr.open_group(store=_CompressedConsolidatedStore(zarr_store), chunk_store=zarr_store, mode="r")
        return _zarr.open_group(store=zarr_store, mode="r")

    def consolidate_store(self, store: StoreLike) -> None:
        zarr_store = _zarr_store(store)
        metadata = {key: json.loads(zarr_store[key]) for key in zarr_store if is_v2_metadata_key(key)}
        zarr_store[V2_CONSOLIDATED_METADATA_KEY] = encode_cbor_xz({"zarr_consolidated_format": 1, "metadata": metadata})

    def group_store(self, group: zarr.Group) -> StoreLike:
        store = _z(group).store
        if isinstance(store, _CompressedConsolidatedStore):
            store = store.store
        return store.store if isinstance(store, _TarStore) else store


backend = Backend()
