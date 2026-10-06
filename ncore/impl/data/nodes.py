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

"""Groups and arrays of ncore data stores, independent of the installed zarr-python version.

:class:`Group` and :class:`Array` wrap zarr-python 2 / 3 groups and arrays with a restricted, strongly-typed API using
non-deprecated APIs of the installed zarr-python version. New nodes use the zarr format of the group they are created
in. The wrapped zarr-python objects remain accessible via the ``zarr`` properties.

All zarr-python version specific operations are implemented by the backend of the installed zarr-python version
(``_zarr2.py`` / ``_zarr3.py``, see :mod:`ncore.impl.data._zarr_api`).
"""

from __future__ import annotations

import collections
import dataclasses
import threading
import typing

from typing import (
    TYPE_CHECKING,
    Any,
    Coroutine,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import typing_extensions
import zarr

from zarr import Array as _ZarrArray
from zarr import Group as _ZarrGroup

from ncore.impl.data import _zarr_api
from ncore.impl.data._json import JsonLike
from ncore.impl.data._zarr_api import Selection, StoreLike, ZarrStore


if TYPE_CHECKING:
    import numpy.typing as npt  # type: ignore[import-not-found]


#: True if zarr-python 3 (or newer) is installed
ZARR_PYTHON_3: bool = int(zarr.__version__.split(".")[0]) >= 3

if ZARR_PYTHON_3:
    from ncore.impl.data._zarr3 import backend as _zarr3_backend

    _backend: _zarr_api.Backend = _zarr3_backend
else:
    from ncore.impl.data._zarr2 import backend as _zarr2_backend

    _backend = _zarr2_backend

#: zarr formats supported by the installed zarr-python version
SUPPORTED_ZARR_FORMATS: Tuple[int, ...] = _backend.supported_zarr_formats

T = TypeVar("T")

#: zarr-python node objects (groups / arrays)
ZarrNode = Union[zarr.Group, zarr.Array]

__all__ = [
    "ZARR_PYTHON_3",
    "SUPPORTED_ZARR_FORMATS",
    "Selection",
    "StoreLike",
    "ZarrStore",
    "check_zarr_format",
    "run",
    "BloscCompression",
    "NodeCache",
    "Group",
    "Array",
]


def check_zarr_format(zarr_format: int) -> Literal[2, 3]:
    """Validates that a zarr format is supported by the installed zarr-python version"""
    if zarr_format not in SUPPORTED_ZARR_FORMATS:
        raise ValueError(
            f"zarr format v{zarr_format} is not supported by zarr-python {zarr.__version__} "
            f"(supported: {', '.join(f'v{f}' for f in SUPPORTED_ZARR_FORMATS)})"
        )
    return 2 if zarr_format == 2 else 3


def run(coroutine: Coroutine[object, object, T]) -> T:
    """Runs a zarr-python 3 coroutine to completion on an event loop owned by the calling thread (zarr-python 3 only).

    The loop's default executor runs offloaded work (e.g., codecs) inline. Loops are re-created in forked processes.
    Falls back to zarr-python's global I/O loop if called from within a running event loop.
    """
    return _backend.run(coroutine)


# -----------------------------------------------------------------------------
# Compression
# -----------------------------------------------------------------------------

#: Blosc compressors supported by both zarr-python 2 (numcodecs ``Blosc``) and zarr-python 3 (``BloscCodec``)
BloscCname = Literal["blosclz", "lz4", "lz4hc", "zlib", "zstd"]

#: Blosc shuffle modes (names of zarr-python 3's ``BloscCodec``, numcodecs ``Blosc`` uses 0 / 1 / 2)
BloscShuffle = Literal["noshuffle", "shuffle", "bitshuffle"]


@dataclasses.dataclass(frozen=True)
class BloscCompression:
    """Blosc compression of array chunks, with the parameters of numcodecs ``Blosc`` / zarr-python 3 ``BloscCodec``.

    Serialized as numcodecs ``Blosc`` compressor for zarr format v2 and as ``blosc`` codec for zarr format v3.
    """

    #: Compressor used by blosc
    cname: BloscCname = "lz4"
    #: Compression level (0: no compression, 9: maximum compression)
    clevel: int = 5
    #: Shuffle filter applied before compression
    shuffle: BloscShuffle = "bitshuffle"
    #: Block size in bytes (0: automatic)
    blocksize: int = 0

    def __post_init__(self) -> None:
        if not _is_blosc_cname(self.cname):
            raise ValueError(f"Unsupported blosc compressor {self.cname!r}, supported: {typing.get_args(BloscCname)}")
        if not 0 <= self.clevel <= 9:
            raise ValueError(f"Blosc compression level needs to be in [0, 9], got {self.clevel}")
        if not _is_blosc_shuffle(self.shuffle):
            raise ValueError(f"Unsupported blosc shuffle {self.shuffle!r}, supported: {typing.get_args(BloscShuffle)}")
        if self.blocksize < 0:
            raise ValueError(f"Blosc block size needs to be non-negative, got {self.blocksize}")

    @property
    def _parameters(self) -> _zarr_api.BloscParameters:
        return self.cname, self.clevel, self.shuffle, self.blocksize

    @staticmethod
    def _from_parameters(parameters: _zarr_api.BloscParameters) -> BloscCompression:
        cname, clevel, shuffle, blocksize = parameters
        if not (_is_blosc_cname(cname) and _is_blosc_shuffle(shuffle)):
            raise ValueError(f"Unsupported blosc configuration {parameters}")
        return BloscCompression(cname=cname, clevel=clevel, shuffle=shuffle, blocksize=blocksize)


def _is_blosc_cname(value: str) -> typing_extensions.TypeGuard[BloscCname]:
    return value in typing.get_args(BloscCname)


def _is_blosc_shuffle(value: str) -> typing_extensions.TypeGuard[BloscShuffle]:
    return value in typing.get_args(BloscShuffle)


# -----------------------------------------------------------------------------
# Node cache
# -----------------------------------------------------------------------------


class NodeCache:
    """Cache of opened nodes of an immutable hierarchy, keyed by node path.

    Opening a node constructs its zarr-python object (including its codec pipeline, and parsing its metadata for
    hierarchies opened without consolidated metadata), which dominates repeated reads of small arrays. Cached nodes
    hold their metadata only (typically a few kB per node), never array data.

    ``max_entries=None`` caches all opened nodes for the lifetime of the cache, ``max_entries=N`` retains the N most
    recently used nodes (least recently used nodes are evicted and re-opened on their next access).
    """

    def __init__(self, max_entries: Optional[int] = None) -> None:
        if max_entries is not None and max_entries < 1:
            raise ValueError(f"max_entries needs to be positive or None (unbounded), got {max_entries}")
        self.max_entries = max_entries
        self._nodes: collections.OrderedDict[str, ZarrNode] = collections.OrderedDict()
        self._mutex = threading.Lock()

    @staticmethod
    def create(max_entries: Optional[int]) -> Optional[NodeCache]:
        """Creates a node cache for a ``node_cache_size`` reader option: None (unbounded), 0 (disabled), or a positive
        number of entries (LRU-capped)"""
        if max_entries == 0:
            return None
        return NodeCache(max_entries)

    def __len__(self) -> int:
        return len(self._nodes)

    def keys(self) -> List[str]:
        """The (absolute) paths of the cached nodes, from least to most recently used"""
        with self._mutex:
            return list(self._nodes)

    def _get(self, group: zarr.Group, path: str) -> ZarrNode:
        key = _join(group.path, path)
        with self._mutex:
            if (node := self._nodes.get(key)) is not None:
                if self.max_entries is not None:
                    self._nodes.move_to_end(key)
                return node

        # outside of the lock, concurrent first accesses open the node redundantly
        node = _backend.open_member(group, path)

        with self._mutex:
            self._nodes[key] = node
            if self.max_entries is not None:
                self._nodes.move_to_end(key)
                while len(self._nodes) > self.max_entries:
                    self._nodes.popitem(last=False)
        return node


# -----------------------------------------------------------------------------
# Nodes
# -----------------------------------------------------------------------------


class _Node:
    """Common functionality of groups and arrays"""

    def __init__(self, node: ZarrNode, cache: Optional[NodeCache]) -> None:
        self._node = node
        self._cache = cache

    @property
    def path(self) -> str:
        """The path of the node within its store ("" for root groups)"""
        return self._node.path

    @property
    def name(self) -> str:
        """The name of the node within its parent group ("" for root groups)"""
        return self._node.path.rsplit("/", 1)[-1]

    @property
    def zarr_format(self) -> int:
        """The zarr format of the node"""
        return _backend.zarr_format(self._node)

    @property
    def attrs(self) -> Dict[str, JsonLike]:
        """A copy of the attributes of the node"""
        return cast(Dict[str, JsonLike], dict(self._node.attrs.asdict()))

    def attr_str(self, name: str) -> str:
        """Returns a string attribute (raises KeyError if missing, TypeError for other types)"""
        return _typed_attr(self._node.attrs[name], str, self.path, name)

    def attr_int(self, name: str) -> int:
        """Returns an integer attribute (raises KeyError if missing, TypeError for other types)"""
        return _typed_attr(self._node.attrs[name], int, self.path, name)

    def attr_dict(self, name: str) -> Dict[str, JsonLike]:
        """Returns a dictionary attribute (raises KeyError if missing, TypeError for other types)"""
        return cast(Dict[str, JsonLike], _typed_attr(self._node.attrs[name], dict, self.path, name))

    def attr_list(self, name: str) -> List[JsonLike]:
        """Returns a list attribute (raises KeyError if missing, TypeError for other types)"""
        return cast(List[JsonLike], _typed_attr(self._node.attrs[name], list, self.path, name))

    def update_attrs(self, attributes: _zarr_api.Attributes) -> None:
        """Updates (merges) attributes of the node (attribute values need to be JSON-serializable)"""
        self._node.attrs.update(_zarr_api.zarr_attributes(attributes))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.path!r}, zarr_format={self.zarr_format})"


class Array(_Node):
    """A zarr array of an ncore data store"""

    def __init__(self, array: _ZarrArray, cache: Optional[NodeCache] = None) -> None:
        super().__init__(array, cache)
        self._array = array

    @property
    def zarr(self) -> _ZarrArray:
        """The wrapped zarr-python array (for APIs of the installed zarr-python version not exposed by ncore)"""
        return self._array

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._array.shape)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._array.dtype)

    @property
    def compression(self) -> Optional[BloscCompression]:
        """The blosc compression of the array (None if uncompressed, raises ValueError for other compressors)"""
        if (parameters := _backend.compression(self._array)) is None:
            return None
        return BloscCompression._from_parameters(parameters)

    def read(self, selection: Selection = Ellipsis) -> npt.NDArray[Any]:
        """Reads (a selection of) the data of the array (0-dimensional selections result in 0-d arrays)"""
        return np.asarray(_backend.read(self._array, selection))

    async def read_async(self, selection: Selection = Ellipsis) -> npt.NDArray[Any]:
        """Reads (a selection of) the data of the array asynchronously, see :meth:`read`.

        Concurrent reads (e.g., via ``asyncio.gather``) overlap their I/O and decompression, the running event loop
        isn't blocked (blocking work runs on worker threads).
        """
        return np.asarray(await _backend.read_async(self._array, selection))

    def read_bytes(self) -> bytes:
        """Reads a binary blob stored by :meth:`Group.create_bytes_array` (supports all representations)"""
        return self._blob(_backend.read(self._array, ()))

    async def read_bytes_async(self) -> bytes:
        """Reads a binary blob asynchronously, see :meth:`read_bytes` and :meth:`read_async`"""
        return self._blob(await _backend.read_async(self._array, ()))

    def _blob(self, value: object) -> bytes:
        # fixed-length bytes (zarr format v2) are numpy scalars or 0-d arrays, uint8 arrays otherwise (zarr format v3)
        if isinstance(value, np.ndarray):
            if value.dtype.kind == "S":
                # fixed-length bytes scalars strip trailing NUL bytes (as in ncore <= 19.8)
                return value.tobytes().rstrip(b"\x00")
            return value.tobytes()
        if isinstance(value, bytes):
            return bytes(value)
        raise TypeError(f"Array {self.path!r} doesn't store a binary blob ({self.dtype})")


class Group(_Node):
    """A zarr group of an ncore data store.

    Nodes accessed via a group share the group's node cache (if any), so nested nodes are opened once only.
    """

    def __init__(self, group: _ZarrGroup, cache: Optional[NodeCache] = None) -> None:
        super().__init__(group, cache)
        self._group = group

    @property
    def zarr(self) -> _ZarrGroup:
        """The wrapped zarr-python group (for APIs of the installed zarr-python version not exposed by ncore)"""
        return self._group

    @property
    def cache(self) -> Optional[NodeCache]:
        """The node cache shared by the nodes accessed via this group"""
        return self._cache

    def with_cache(self, cache: Optional[NodeCache]) -> Group:
        """Returns this group using the given node cache for accessed nodes"""
        return Group(self._group, cache)

    # -- reading ---------------------------------------------------------------

    def _member(self, path: str) -> ZarrNode:
        if self._cache is not None:
            return self._cache._get(self._group, path)
        return _backend.open_member(self._group, path)

    def node(self, path: str) -> Union[Group, Array]:
        """Returns a (nested) member group or array, resolved in a single lookup (raises KeyError if not present)"""
        node = self._member(path)
        return Group(node, self._cache) if isinstance(node, _ZarrGroup) else Array(node, self._cache)

    def group(self, path: str) -> Group:
        """Returns a (nested) member group, resolved in a single lookup (raises KeyError if not present / no group)"""
        if not isinstance(node := self._member(path), _ZarrGroup):
            raise KeyError(f"{_join(self.path, path)} is not a group")
        return Group(node, self._cache)

    def array(self, path: str) -> Array:
        """Returns a (nested) member array, resolved in a single lookup (raises KeyError if not present / no array)"""
        if not isinstance(node := self._member(path), _ZarrArray):
            raise KeyError(f"{_join(self.path, path)} is not an array")
        return Array(node, self._cache)

    def members(self) -> List[str]:
        """Names of all direct members (from consolidated metadata without I/O, if available)"""
        return _backend.member_names(self._group)

    def __contains__(self, name: str) -> bool:
        """True if the group has a direct member with the given name"""
        return _backend.has_member(self._group, name)

    def groups(self) -> Iterator[Tuple[str, Group]]:
        """Iterates over the direct member groups (sorted by name)"""
        members = sorted(_backend.member_groups(self._group), key=lambda item: item[0])
        return iter([(name, Group(group, self._cache)) for name, group in members])

    # -- writing ---------------------------------------------------------------

    def create_group(self, name: str, *, attributes: Optional[_zarr_api.Attributes] = None) -> Group:
        """Creates a member group (raises if a node with the name already exists)"""
        return Group(_backend.create_group(self._group, name, attributes), self._cache)

    def require_group(self, name: str) -> Group:
        """Returns a member group, creating it if it doesn't exist yet"""
        return Group(self._group.require_group(name), self._cache)

    def create_array(
        self,
        name: str,
        data: npt.ArrayLike,
        *,
        dtype: Optional[npt.DTypeLike] = None,
        chunks: Union[Tuple[int, ...], Literal["auto"], None] = None,
        compression: Optional[BloscCompression] = BloscCompression(),
        attributes: Optional[_zarr_api.Attributes] = None,
    ) -> Array:
        """Creates a member array initialized with the given data.

        Parameters
        ----------
        name
            Name of the array within the group.
        data
            Array data (defines the shape of the array, and its dtype if ``dtype`` isn't given).
        dtype
            Optional dtype of the array, the data is converted to it (e.g., to store python floats as ``float32``).
        chunks
            Chunk shape, defaults to a single chunk (zero-sized dimensions are clamped to 1). "auto" uses the chunk
            shape heuristic of zarr-python (identical for zarr-python 2 and 3).
        compression
            Blosc compression of the array chunks (default: lz4, level 5, bit-shuffle), or None for uncompressed data.
        attributes
            Optional array attributes.
        """
        values = np.asarray(data, dtype=dtype)
        if chunks != "auto":
            chunks = tuple(max(1, int(c)) for c in (values.shape if chunks is None else chunks))
        array = _backend.create_array(
            self._group,
            name,
            values,
            chunks,
            None if compression is None else compression._parameters,
            attributes,
        )
        return Array(array, self._cache)

    def create_bytes_array(self, name: str, data: bytes, *, attributes: Optional[_zarr_api.Attributes] = None) -> Array:
        """Stores a binary blob (e.g., an encoded image) as uncompressed member array.

        zarr format v2: 0-dimensional fixed-length bytes array (``|S<N>``, the layout of ncore <= 19.8).
        zarr format v3: 1-dimensional ``uint8`` array (fixed-length bytes are not part of the zarr v3 specification).
        """
        value = np.asarray(data) if self.zarr_format == 2 else np.frombuffer(data, dtype=np.uint8)
        return self.create_array(name, value, compression=None, attributes=attributes)


def _typed_attr(value: object, expected: Type[T], path: str, name: str) -> T:
    if not isinstance(value, expected) or (isinstance(value, bool) and expected is int):
        raise TypeError(f"Attribute {name!r} of {path!r} is not of type {expected.__name__}: {value!r}")
    return value


def _join(parent: str, path: str) -> str:
    return f"{parent}/{path}" if parent else path
