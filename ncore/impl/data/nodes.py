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

"""Groups and arrays of NCore data stores.

:class:`Group` and :class:`Array` wrap zarr-python groups and arrays with a restricted, typed API, implemented by a
backend of the installed zarr-python version (see :mod:`ncore.impl.data._backend`). The wrapped zarr-python objects
are accessible via the ``zarr`` properties (for APIs of the installed zarr-python version not exposed by NCore).
"""

from __future__ import annotations

import collections
import threading

from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Literal, Optional, Tuple, Type, TypeVar, Union, cast

import numpy as np
import zarr

from zarr import Array as ZarrArray
from zarr import Group as ZarrGroup

from ncore.impl.data._backend import Attributes, Backend, BloscCompression, Selection, attributes_dict
from ncore.impl.data._json import JsonLike


if TYPE_CHECKING:
    import numpy.typing as npt  # type: ignore[import-not-found]


def _installed_backend() -> Backend:
    """The backend of the installed zarr-python version"""
    if int(zarr.__version__.split(".", 1)[0]) >= 3:
        from ncore.impl.data._zarr3 import backend as zarr3_backend

        return zarr3_backend

    from ncore.impl.data._zarr2 import backend as zarr2_backend

    return zarr2_backend


_backend: Backend = _installed_backend()

#: zarr formats supported by the installed zarr-python version
SUPPORTED_ZARR_FORMATS: Tuple[int, ...] = _backend.supported_zarr_formats

T = TypeVar("T")


class NodeCache:
    """Cache of the opened nodes of a hierarchy opened for reading, by node path.

    Opening a node constructs its zarr-python object (decoding its metadata), which dominates repeated reads of small
    arrays (e.g., of frames). Cached nodes reference their metadata only, never array data. ``max_entries=None`` caches
    all opened nodes, ``max_entries=N`` the N most recently used nodes (evicted nodes are re-opened on access).

    All methods are thread-safe.
    """

    def __init__(self, max_entries: Optional[int] = None) -> None:
        if max_entries is not None and max_entries < 1:
            raise ValueError(f"max_entries needs to be positive or None (unbounded), got {max_entries}")
        self._max_entries = max_entries
        self._nodes: collections.OrderedDict[str, Union[ZarrGroup, ZarrArray]] = collections.OrderedDict()
        self._lock = threading.Lock()

    @property
    def max_entries(self) -> Optional[int]:
        return self._max_entries

    def __len__(self) -> int:
        return len(self._nodes)

    def keys(self) -> List[str]:
        """The paths of the cached nodes, from least to most recently used"""
        with self._lock:
            return list(self._nodes)

    def _open(self, group: ZarrGroup, path: str) -> Union[ZarrGroup, ZarrArray]:
        key = _join(group.path, path)
        with self._lock:
            if (node := self._nodes.get(key)) is not None:
                self._nodes.move_to_end(key)
                return node

        # (concurrent first accesses open nodes redundantly, which is cheaper than serializing all accesses)
        node = _backend.open_member(group, path)

        with self._lock:
            self._nodes[key] = node
            if self._max_entries is not None:
                while len(self._nodes) > self._max_entries:
                    self._nodes.popitem(last=False)
        return node


class _Node:
    """Common functionality of groups and arrays"""

    def __init__(self, node: Union[ZarrGroup, ZarrArray]) -> None:
        self._node = node

    @property
    def path(self) -> str:
        """The path of the node within its store ("" for root groups)"""
        return self._node.path

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
        return _typed_attr(self, name, str)

    def attr_int(self, name: str) -> int:
        """Returns an integer attribute (raises KeyError if missing, TypeError for other types)"""
        return _typed_attr(self, name, int)

    def attr_dict(self, name: str) -> Dict[str, JsonLike]:
        """Returns a dictionary attribute (raises KeyError if missing, TypeError for other types)"""
        return _typed_attr(self, name, dict)

    def attr_list(self, name: str) -> List[JsonLike]:
        """Returns a list attribute (raises KeyError if missing, TypeError for other types)"""
        return _typed_attr(self, name, list)

    def update_attrs(self, attributes: Attributes) -> None:
        """Updates (merges) attributes of the node"""
        self._node.attrs.update(attributes_dict(attributes))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.path!r}, zarr_format={self.zarr_format})"


class Array(_Node):
    """An array of an NCore data store"""

    def __init__(self, array: ZarrArray) -> None:
        super().__init__(array)
        self._array = array

    @property
    def zarr(self) -> ZarrArray:
        """The wrapped zarr-python array"""
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
        return _backend.compression(self._array)

    def read(self, selection: Selection = Ellipsis) -> npt.NDArray[Any]:
        """Reads (a selection of) the data of the array (0-dimensional selections result in 0-d arrays)"""
        return np.asarray(_backend.read(self._array, selection))

    def read_bytes(self) -> bytes:
        """Reads a binary blob stored by :meth:`Group.create_bytes_array`"""
        value = _backend.read(self._array, ())
        if isinstance(value, bytes):  # fixed-length bytes scalar (trailing NUL bytes are stripped)
            return bytes(value)
        if isinstance(value, np.ndarray) and value.dtype == np.uint8:
            return value.tobytes()
        raise TypeError(f"Array {self.path!r} doesn't store a binary blob ({self.dtype})")


class Group(_Node):
    """A group of an NCore data store.

    Members of groups using a :class:`NodeCache` are opened via the cache (also members of member groups).
    """

    def __init__(self, group: ZarrGroup, cache: Optional[NodeCache] = None) -> None:
        super().__init__(group)
        self._group = group
        self._cache = cache

    @property
    def zarr(self) -> ZarrGroup:
        """The wrapped zarr-python group"""
        return self._group

    @property
    def cache(self) -> Optional[NodeCache]:
        """The node cache used to open members (None if not cached)"""
        return self._cache

    def with_cache(self, cache: Optional[NodeCache]) -> Group:
        """Returns the group, using the given node cache to open its members"""
        return Group(self._group, cache)

    # -- reading ---------------------------------------------------------------

    def _open(self, path: str) -> Union[ZarrGroup, ZarrArray]:
        if self._cache is not None:
            return self._cache._open(self._group, path)
        return _backend.open_member(self._group, path)

    def node(self, path: str) -> Union[Group, Array]:
        """Returns a (nested) member group or array (raises KeyError if it doesn't exist)"""
        node = self._open(path)
        return Group(node, self._cache) if isinstance(node, ZarrGroup) else Array(node)

    def group(self, path: str) -> Group:
        """Returns a (nested) member group (raises KeyError if it doesn't exist or isn't a group)"""
        if not isinstance(node := self._open(path), ZarrGroup):
            raise KeyError(f"{_join(self.path, path)} is not a group")
        return Group(node, self._cache)

    def array(self, path: str) -> Array:
        """Returns a (nested) member array (raises KeyError if it doesn't exist or isn't an array)"""
        if not isinstance(node := self._open(path), ZarrArray):
            raise KeyError(f"{_join(self.path, path)} is not an array")
        return Array(node)

    def members(self) -> List[str]:
        """Names of all direct members"""
        return _backend.member_names(self._group)

    def __contains__(self, name: str) -> bool:
        """True if the group has a direct member with the given name"""
        return _backend.has_member(self._group, name)

    def groups(self) -> Iterator[Tuple[str, Group]]:
        """Iterates over the direct member groups (ordered by name)"""
        for name, group in sorted(_backend.member_groups(self._group), key=lambda member: member[0]):
            yield name, Group(group, self._cache)

    # -- writing ---------------------------------------------------------------

    def create_group(self, name: str, *, attributes: Optional[Attributes] = None) -> Group:
        """Creates a member group (raises if a member with the name exists already)"""
        return Group(_backend.create_group(self._group, name, attributes))

    def require_group(self, name: str) -> Group:
        """Returns a member group, creating it if it doesn't exist yet"""
        return Group(_backend.require_group(self._group, name))

    def create_array(
        self,
        name: str,
        data: npt.ArrayLike,
        *,
        dtype: Optional[npt.DTypeLike] = None,
        chunks: Union[Tuple[int, ...], Literal["auto"], None] = None,
        compression: Optional[BloscCompression] = BloscCompression(),
        attributes: Optional[Attributes] = None,
    ) -> Array:
        """Creates a member array initialized with data.

        Parameters
        ----------
        name
            Name of the array within the group.
        data
            Data of the array (defines the shape of the array, and its dtype if ``dtype`` isn't given).
        dtype
            Optional dtype of the array (the data is converted to it).
        chunks
            Chunk shape, a single chunk by default (zero-sized dimensions are clamped to 1). "auto" uses the chunk
            shape heuristic of zarr-python.
        compression
            Blosc compression of the chunks (default: lz4, level 5, bit-shuffle), or None for uncompressed chunks.
        attributes
            Optional attributes of the array.
        """
        values = np.asarray(data, dtype=dtype)
        if chunks != "auto":
            chunks = tuple(max(1, int(size)) for size in (values.shape if chunks is None else chunks))
        return Array(_backend.create_array(self._group, name, values, chunks, compression, attributes))

    def create_bytes_array(self, name: str, data: bytes, *, attributes: Optional[Attributes] = None) -> Array:
        """Stores a binary blob (e.g., an encoded image) as uncompressed member array (0-dimensional fixed-length
        bytes array for zarr format v2)"""
        return self.create_array(name, np.asarray(data), compression=None, attributes=attributes)


def _typed_attr(node: _Node, name: str, expected: Type[T]) -> T:
    value = node._node.attrs[name]
    if not isinstance(value, expected) or (isinstance(value, bool) and expected is int):
        raise TypeError(f"Attribute {name!r} of {node.path!r} is not of type {expected.__name__}: {value!r}")
    return cast(T, value)


def _join(parent: str, path: str) -> str:
    return f"{parent}/{path}" if parent else path


__all__ = [
    "Array",
    "BloscCompression",
    "Group",
    "NodeCache",
    "SUPPORTED_ZARR_FORMATS",
]
