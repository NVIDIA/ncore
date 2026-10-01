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

"""Structural types of the zarr-python version specific APIs used by ncore.

``zarr.Group`` / ``zarr.Array`` and their common API (``path``, ``attrs``, ``shape``, ``dtype``, ``require_group``)
exist in both zarr-python versions and are used directly. All APIs specific to one zarr-python version are accessed
via the protocols of this module only: the version specific backends (``_zarr2.py`` / ``_zarr3.py``) bind the APIs and
store base classes of their zarr-python version to these protocols, and implement the version-independent
:class:`Backend` used by :mod:`ncore.impl.data.nodes` and :mod:`ncore.impl.data.stores`. This keeps all modules
type-checkable with either zarr-python version installed.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Coroutine,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numcodecs


if TYPE_CHECKING:
    from types import EllipsisType

    import numpy as np
    import numpy.typing as npt  # type: ignore[import-not-found]
    import zarr

    from ncore.impl.data._itar import IndexedTarStore
else:
    EllipsisType = type(Ellipsis)  # types.EllipsisType requires python>=3.10


T = TypeVar("T")

#: Selections of array data (indices, slices, Ellipsis, or tuples of them)
Selection = Union[int, slice, EllipsisType, Tuple[Union[int, slice, EllipsisType], ...]]

#: Blosc parameters (compressor name, compression level, shuffle mode name, block size)
BloscParameters = Tuple[str, int, str, int]

#: Attributes passed to zarr-python (values need to be JSON-serializable)
Attributes = Mapping[str, object]


def zarr_attributes(attributes: Attributes) -> Dict[str, Any]:  # zarr-python's attribute value type
    """Attributes as passed to zarr-python"""
    return dict(attributes)


class ZarrStore(Protocol):
    """An (opaque) store of the installed zarr-python version (e.g., a directory store)"""


#: Stores accepted by :mod:`ncore.impl.data.stores`
StoreLike = Union["IndexedTarStore", ZarrStore]


# -----------------------------------------------------------------------------
# Backend (implemented by _zarr2.py / _zarr3.py)
# -----------------------------------------------------------------------------


class Backend(Protocol):
    """zarr-python version specific operations of ncore nodes and stores"""

    #: zarr formats supported by the zarr-python version
    supported_zarr_formats: Tuple[int, ...]

    def run(self, coroutine: Coroutine[object, object, T]) -> T:
        """Runs a zarr-python 3 coroutine to completion on an event loop owned by the calling thread"""
        ...

    # -- nodes -----------------------------------------------------------------

    def zarr_format(self, node: Union[zarr.Group, zarr.Array]) -> int:
        """zarr format of a group / array"""
        ...

    def open_member(self, group: zarr.Group, path: str) -> Union[zarr.Group, zarr.Array]:
        """Opens a (nested) member of a group in a single lookup (raises KeyError if not present)"""
        ...

    def member_names(self, group: zarr.Group) -> List[str]:
        """Names of the direct members of a group (from consolidated metadata without I/O, if available)"""
        ...

    def has_member(self, group: zarr.Group, name: str) -> bool:
        """True if a group has a direct member with the given name"""
        ...

    def member_groups(self, group: zarr.Group) -> List[Tuple[str, zarr.Group]]:
        """The direct member groups of a group"""
        ...

    def create_group(self, group: zarr.Group, name: str, attributes: Optional[Attributes]) -> zarr.Group:
        """Creates a member group"""
        ...

    def create_array(
        self,
        group: zarr.Group,
        name: str,
        data: npt.NDArray[np.generic],
        chunks: Union[Tuple[int, ...], Literal["auto"]],
        compression: Optional[BloscParameters],
        attributes: Optional[Attributes],
    ) -> zarr.Array:
        """Creates a member array initialized with the given data (``compression`` None: uncompressed)"""
        ...

    def read(self, array: zarr.Array, selection: Selection) -> object:
        """Reads (a selection of) the data of an array (0-dimensional selections may result in numpy scalars)"""
        ...

    def read_async(self, array: zarr.Array, selection: Selection) -> Coroutine[object, object, object]:
        """Reads (a selection of) the data of an array asynchronously, without blocking the running event loop"""
        ...

    def compression(self, array: zarr.Array) -> Optional[BloscParameters]:
        """Blosc parameters of the compressor of an array (None if uncompressed, raises ValueError for other
        compressors)"""
        ...

    # -- stores ----------------------------------------------------------------

    def open_directory_store(self, path: str, mode: Literal["r", "w"]) -> ZarrStore:
        """Opens a directory store for reading or (truncating) writing"""
        ...

    def create_root_group(self, store: StoreLike, zarr_format: Literal[2, 3], attributes: Attributes) -> zarr.Group:
        """Creates the root group of a new store"""
        ...

    def open_store(self, store: StoreLike, open_consolidated: bool) -> zarr.Group:
        """Opens the root group of a store read-only (raises KeyError if consolidated metadata is requested, but
        missing)"""
        ...

    def open_compressed_consolidated(self, store: StoreLike, metadata_key: str, mode: Literal["r", "r+"]) -> zarr.Group:
        """Opens a zarr format v2 group using compressed consolidated metadata (raises KeyError if missing)"""
        ...

    def consolidate_store(self, store: StoreLike) -> None:
        """Consolidates the metadata of a store (compressed for indexed tar stores)"""
        ...

    def consolidate_compressed_metadata(self, store: StoreLike, metadata_key: str) -> None:
        """Consolidates the zarr format v2 metadata of a store into a single compressed record"""
        ...

    def close_store(self, store: StoreLike) -> None:
        """Closes a store"""
        ...

    def group_store(self, group: zarr.Group) -> StoreLike:
        """The store of a group (unwrapping internal store adapters / wrappers)"""
        ...


# -----------------------------------------------------------------------------
# numcodecs blosc compressors (zarr format v2, used by both zarr-python versions)
# -----------------------------------------------------------------------------

_SHUFFLE_IDS: Dict[str, int] = {"noshuffle": 0, "shuffle": 1, "bitshuffle": 2}
_SHUFFLE_NAMES: Dict[int, str] = {i: name for name, i in _SHUFFLE_IDS.items()}


def numcodecs_blosc(parameters: BloscParameters) -> numcodecs.Blosc:
    """numcodecs ``Blosc`` compressor with the given parameters"""
    cname, clevel, shuffle, blocksize = parameters
    return numcodecs.Blosc(cname=cname, clevel=clevel, shuffle=_SHUFFLE_IDS[shuffle], blocksize=blocksize)


def blosc_parameters(config: Mapping[str, object], shuffle_ids: bool) -> BloscParameters:
    """Blosc parameters of a blosc codec configuration (shuffle modes as ids for numcodecs, names otherwise)"""
    cname, clevel, shuffle, blocksize = (config.get(key) for key in ("cname", "clevel", "shuffle", "blocksize"))
    if shuffle_ids and isinstance(shuffle, int):
        shuffle = _SHUFFLE_NAMES.get(shuffle)
    if not (isinstance(cname, str) and isinstance(clevel, int) and isinstance(shuffle, str)):
        raise ValueError(f"Unsupported blosc configuration {dict(config)}")
    if not isinstance(blocksize, int):
        raise ValueError(f"Unsupported blosc configuration {dict(config)}")
    return cname, clevel, shuffle, blocksize


# -----------------------------------------------------------------------------
# zarr-python 2 specific APIs (bound by _zarr2.py)
# -----------------------------------------------------------------------------


class Zarr2Store(Protocol):
    """A mapping based zarr-python 2 store (``zarr.storage.BaseStore``)"""

    def __getitem__(self, key: str) -> bytes: ...

    def __setitem__(self, key: str, value: bytes) -> None: ...

    def __contains__(self, key: object) -> bool: ...

    def __iter__(self) -> Iterator[str]: ...

    def close(self) -> None: ...


class Zarr2StoreBase(Protocol):
    """``zarr._storage.store.Store``: base class of zarr-python 2 stores"""

    _erasable: bool


class Zarr2ConsolidatedMetadataStoreBase(Protocol):
    """``zarr.storage.ConsolidatedMetadataStore``: base class of zarr-python 2 consolidated metadata stores"""

    store: Zarr2Store
    meta_store: Zarr2Store


class Zarr2StoreModule(Protocol):
    """``zarr._storage.store``"""

    Store: Type[Zarr2StoreBase]


class Zarr2StorageBases(Protocol):
    """``zarr.storage`` (store base classes)"""

    ConsolidatedMetadataStore: Type[Zarr2ConsolidatedMetadataStoreBase]


class Zarr2Attributes(Protocol):
    def put(self, attributes: Dict[str, object]) -> None: ...


class Zarr2Node(Protocol):
    @property
    def attrs(self) -> Zarr2Attributes: ...


class Zarr2Group(Protocol):
    """``zarr.hierarchy.Group``"""

    @property
    def store(self) -> Zarr2Store: ...

    @property
    def chunk_store(self) -> Optional[Zarr2Store]: ...

    @property
    def attrs(self) -> Zarr2Attributes: ...

    def keys(self) -> Iterable[str]: ...

    def __contains__(self, name: str) -> bool: ...

    def __getitem__(self, path: str) -> Union[zarr.Group, zarr.Array]: ...

    def groups(self) -> Iterable[Tuple[str, zarr.Group]]: ...

    def create_group(self, name: str) -> zarr.Group: ...

    def create_dataset(
        self,
        name: str,
        *,
        data: npt.NDArray[np.generic],
        chunks: Union[Tuple[int, ...], bool],
        compressor: Optional[numcodecs.Blosc],
    ) -> zarr.Array: ...


class Zarr2Array(Protocol):
    """``zarr.core.Array``"""

    @property
    def compressor(self) -> Optional[object]: ...

    def __getitem__(self, selection: Selection) -> object: ...


class Zarr2DirectoryStore(Zarr2Store, Protocol):
    def rmdir(self) -> None: ...


class Zarr2Storage(Protocol):
    """``zarr.storage``"""

    def DirectoryStore(self, path: str) -> Zarr2DirectoryStore: ...

    def KVStore(self, mutable_mapping: Dict[str, object]) -> Zarr2Store: ...

    def normalize_store_arg(self, store: object, *, mode: str) -> Zarr2Store: ...


class Zarr2Errors(Protocol):
    """``zarr.errors``"""

    ReadOnlyError: Type[Exception]
    MetadataError: Type[Exception]


class Zarr2Module(Protocol):
    """``zarr`` (zarr-python 2 specific functions)"""

    def group(self, *, store: Zarr2Store) -> zarr.Group: ...

    def open_group(self, *, store: Zarr2Store, mode: str) -> zarr.Group: ...

    def open(self, *, store: Zarr2Store, chunk_store: Zarr2Store, mode: str) -> zarr.Group: ...


# -----------------------------------------------------------------------------
# zarr-python 3 specific APIs (bound by _zarr3.py)
# -----------------------------------------------------------------------------


class Zarr3Buffer(Protocol):
    """``zarr.core.buffer.Buffer``"""

    def to_bytes(self) -> bytes: ...


class Zarr3BufferType(Protocol):
    def from_bytes(self, value: bytes) -> Zarr3Buffer: ...


class Zarr3BufferPrototype(Protocol):
    """``zarr.core.buffer.BufferPrototype``"""

    @property
    def buffer(self) -> Zarr3BufferType: ...


class Zarr3RangeByteRequest(Protocol):
    start: int
    end: int


class Zarr3OffsetByteRequest(Protocol):
    offset: int


class Zarr3SuffixByteRequest(Protocol):
    suffix: int


#: ``zarr.abc.store.ByteRequest``
Zarr3ByteRequest = Union[Zarr3RangeByteRequest, Zarr3OffsetByteRequest, Zarr3SuffixByteRequest]


class Zarr3Store(Protocol):
    """``zarr.abc.store.Store`` (operations used by ncore)"""

    async def get(
        self, key: str, prototype: Zarr3BufferPrototype, byte_range: Optional[Zarr3ByteRequest] = None
    ) -> Optional[Zarr3Buffer]: ...

    async def set(self, key: str, value: Zarr3Buffer) -> None: ...

    async def exists(self, key: str) -> bool: ...

    async def delete_dir(self, prefix: str) -> None: ...

    def list(self) -> AsyncIterator[str]: ...

    def list_dir(self, prefix: str) -> AsyncIterator[str]: ...

    def close(self) -> None: ...


class Zarr3StoreBase(Protocol):
    """``zarr.abc.store.Store``: base class of zarr-python 3 stores (inherited members used by ncore)"""

    def __init__(self, *, read_only: bool = False) -> None: ...

    @property
    def read_only(self) -> bool: ...

    def _check_writable(self) -> None: ...


class Zarr3WrapperStoreBase(Protocol):
    """``zarr.storage.WrapperStore``: base class of zarr-python 3 store wrappers (inherited members used by ncore)"""

    _store: Zarr3Store

    def __init__(self, store: Zarr3Store) -> None: ...


class Zarr3StoreModule(Protocol):
    """``zarr.abc.store``"""

    RangeByteRequest: Type[Zarr3RangeByteRequest]
    OffsetByteRequest: Type[Zarr3OffsetByteRequest]
    SuffixByteRequest: Type[Zarr3SuffixByteRequest]


class Zarr3BufferModule(Protocol):
    """``zarr.core.buffer``"""

    def default_buffer_prototype(self) -> Zarr3BufferPrototype: ...


class Zarr3Storage(Protocol):
    """``zarr.storage``"""

    def LocalStore(self, root: str, *, read_only: bool) -> Zarr3Store: ...


class Zarr3AsyncNode(Protocol):
    """``zarr.AsyncGroup`` / ``zarr.AsyncArray``"""


class Zarr3AsyncApi(Protocol):
    """``zarr.api.asynchronous``"""

    def open_group(
        self, *, store: object, zarr_format: int, mode: str, use_consolidated: Optional[bool]
    ) -> Coroutine[object, object, Zarr3AsyncNode]: ...

    def create_group(
        self, *, store: object, zarr_format: int, attributes: Dict[str, object]
    ) -> Coroutine[object, object, Zarr3AsyncNode]: ...

    def consolidate_metadata(self, store: object, *, zarr_format: int) -> Coroutine[object, object, object]: ...


class Zarr3Module(Protocol):
    """``zarr`` (zarr-python 3 specific classes)"""

    AsyncGroup: Type[Zarr3AsyncNode]

    def Group(self, node: Zarr3AsyncNode) -> zarr.Group: ...

    def Array(self, node: Zarr3AsyncNode) -> zarr.Array: ...


class Zarr3Sync(Protocol):
    """``zarr.core.sync``"""

    def sync(self, coroutine: Coroutine[object, object, T]) -> T: ...


class Zarr3Codec(Protocol):
    def to_dict(self) -> Dict[str, object]: ...


class Zarr3Codecs(Protocol):
    """``zarr.codecs``"""

    def BloscCodec(self, *, cname: str, clevel: int, shuffle: str, blocksize: int) -> Zarr3Codec: ...


class Zarr3AsyncGroup(Protocol):
    def getitem(self, path: str) -> Coroutine[object, object, Zarr3AsyncNode]: ...


class Zarr3StorePath(Protocol):
    """``zarr.storage.StorePath`` (zarr-python 3)"""

    @property
    def store(self) -> Zarr3Store: ...

    @property
    def path(self) -> str: ...


class Zarr3AsyncArray(Protocol):
    @property
    def store_path(self) -> Zarr3StorePath: ...

    def getitem(self, selection: Selection) -> Coroutine[object, object, object]: ...


class Zarr3ConsolidatedMetadata(Protocol):
    @property
    def metadata(self) -> Mapping[str, object]: ...


class Zarr3Metadata(Protocol):
    @property
    def zarr_format(self) -> int: ...


class Zarr3GroupMetadata(Zarr3Metadata, Protocol):
    @property
    def consolidated_metadata(self) -> Optional[Zarr3ConsolidatedMetadata]: ...


class Zarr3Group(Protocol):
    """``zarr.Group`` (zarr-python 3)"""

    @property
    def store(self) -> Zarr3Store: ...

    @property
    def metadata(self) -> Zarr3GroupMetadata: ...

    @property
    def _async_group(self) -> Zarr3AsyncGroup: ...

    def keys(self) -> Iterator[str]: ...

    def __contains__(self, name: str) -> bool: ...

    def groups(self) -> Iterator[Tuple[str, zarr.Group]]: ...

    def create_group(self, name: str, *, attributes: Optional[Dict[str, object]]) -> zarr.Group: ...

    def create_array(
        self,
        name: str,
        *,
        data: npt.NDArray[np.generic],
        chunks: Union[Tuple[int, ...], Literal["auto"]],
        compressors: Union[numcodecs.Blosc, Zarr3Codec, None],
        attributes: Optional[Dict[str, object]],
    ) -> zarr.Array: ...


class Zarr3Array(Protocol):
    """``zarr.Array`` (zarr-python 3)"""

    @property
    def metadata(self) -> Zarr3Metadata: ...

    @property
    def async_array(self) -> Zarr3AsyncArray: ...

    @property
    def compressors(self) -> Tuple[object, ...]: ...
