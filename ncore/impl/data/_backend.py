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

"""Interface of the zarr-python version specific backends of :mod:`ncore.impl.data.nodes` / :mod:`.stores`, and
types shared by the backends"""

from __future__ import annotations

import dataclasses
import sys
import typing

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Mapping, Optional, Tuple, Union, cast

import numcodecs
import zarr

from ncore.impl.data.itar import IndexedTarStore


if sys.version_info >= (3, 10):
    from types import EllipsisType
else:  # pragma: no cover
    EllipsisType = type(Ellipsis)

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt  # type: ignore[import-not-found]

    from typing_extensions import Protocol
else:
    Protocol = object


#: Key of the compressed consolidated metadata of zarr format v2 hierarchies (CBOR + LZMA encoded ``.zmetadata``)
V2_CONSOLIDATED_METADATA_KEY = ".zmetadata.cbor.xz"

#: Key of the compressed consolidated metadata of zarr format v3 hierarchies (CBOR + zlib encoded root ``zarr.json``
#: including the consolidated metadata of the hierarchy, which decodes several times faster than CBOR + LZMA). The root
#: ``zarr.json`` of such hierarchies is only stored in this compressed form.
V3_CONSOLIDATED_METADATA_KEY = "zarr.json.cbor.zlib"

#: Basic selections of array data (integers, slices and Ellipsis)
Selection = Union[int, slice, EllipsisType, Tuple[Union[int, slice, EllipsisType], ...]]

#: Attributes of nodes (JSON-serializable values, also nested mappings / sequences: values are JSON-encoded by zarr-python,
#: see :data:`ncore.impl.data.types.JsonLike` for the attributes read back)
Attributes = Mapping[str, object]


def attributes_dict(attributes: Attributes) -> Dict[str, Any]:
    """A (shallow) copy of attributes as dictionary, as accepted by zarr-python"""
    return dict(attributes)


class ZarrStore:
    """A zarr-python store of the installed zarr-python version (e.g., of a directory)"""


#: Stores of zarr hierarchies: indexed tar archives, or zarr-python stores
StoreLike = Union[IndexedTarStore, ZarrStore]


def is_v2_metadata_key(key: str) -> bool:
    """True for keys of zarr format v2 node metadata"""
    return key.endswith((".zarray", ".zgroup", ".zattrs"))


#: Blosc compressors supported by zarr-python 2 (numcodecs ``Blosc``) and zarr-python 3 (``BloscCodec``)
BloscCname = Literal["blosclz", "lz4", "lz4hc", "zlib", "zstd"]

#: Blosc shuffle modes (numcodecs ``Blosc`` encodes them as 0 / 1 / 2)
BloscShuffle = Literal["noshuffle", "shuffle", "bitshuffle"]

_BLOSC_SHUFFLES: List[BloscShuffle] = ["noshuffle", "shuffle", "bitshuffle"]


@dataclasses.dataclass(frozen=True)
class BloscCompression:
    """Blosc compression of array chunks"""

    #: Compressor used by blosc
    cname: BloscCname = "lz4"
    #: Compression level (0: no compression, 9: maximum compression)
    clevel: int = 5
    #: Shuffle filter applied before compression
    shuffle: BloscShuffle = "bitshuffle"
    #: Block size in bytes (0: automatic)
    blocksize: int = 0

    def __post_init__(self) -> None:
        if self.cname not in typing.get_args(BloscCname):
            raise ValueError(f"Unsupported blosc compressor {self.cname!r}, supported: {typing.get_args(BloscCname)}")
        if not 0 <= self.clevel <= 9:
            raise ValueError(f"Blosc compression level needs to be in [0, 9], got {self.clevel}")
        if self.shuffle not in _BLOSC_SHUFFLES:
            raise ValueError(f"Unsupported blosc shuffle {self.shuffle!r}, supported: {_BLOSC_SHUFFLES}")
        if self.blocksize < 0:
            raise ValueError(f"Blosc block size needs to be non-negative, got {self.blocksize}")

    def to_config(self) -> Dict[str, object]:
        """The configuration of the zarr format v3 ``blosc`` codec"""
        return {"cname": self.cname, "clevel": self.clevel, "shuffle": self.shuffle, "blocksize": self.blocksize}

    @staticmethod
    def from_config(config: Mapping[str, object]) -> BloscCompression:
        """From the configuration of a zarr format v3 ``blosc`` codec"""
        return BloscCompression(
            cname=cast(BloscCname, config["cname"]),
            clevel=cast(int, config["clevel"]),
            shuffle=cast(BloscShuffle, config["shuffle"]),
            blocksize=cast(int, config.get("blocksize", 0)),
        )

    def to_numcodecs(self) -> numcodecs.Blosc:
        return numcodecs.Blosc(
            cname=self.cname,
            clevel=self.clevel,
            shuffle=_BLOSC_SHUFFLES.index(self.shuffle),
            blocksize=self.blocksize,
        )

    @staticmethod
    def from_numcodecs(compressor: numcodecs.Blosc) -> BloscCompression:
        config = compressor.get_config()
        return BloscCompression(
            cname=config["cname"],
            clevel=config["clevel"],
            shuffle=_BLOSC_SHUFFLES[config["shuffle"]],
            blocksize=config["blocksize"],
        )


ZarrNode = Union[zarr.Group, zarr.Array]


class Backend(Protocol):
    """zarr-python version specific operations of nodes and stores"""

    #: zarr formats supported by the zarr-python version
    supported_zarr_formats: Tuple[int, ...]

    # -- nodes -----------------------------------------------------------------

    def zarr_format(self, node: ZarrNode) -> int:
        """The zarr format of a node"""
        ...

    def open_member(self, group: zarr.Group, path: str) -> ZarrNode:
        """Opens a (nested) member of a group (raises KeyError if it doesn't exist)"""
        ...

    def member_names(self, group: zarr.Group) -> List[str]:
        """Names of the direct members of a group"""
        ...

    def has_member(self, group: zarr.Group, name: str) -> bool:
        """True if a group has a direct member with the given name"""
        ...

    def member_groups(self, group: zarr.Group) -> List[Tuple[str, zarr.Group]]:
        """Direct member groups of a group"""
        ...

    def create_group(self, group: zarr.Group, name: str, attributes: Optional[Attributes]) -> zarr.Group:
        """Creates a member group (raises if a member with the name exists already)"""
        ...

    def require_group(self, group: zarr.Group, name: str) -> zarr.Group:
        """Returns a member group, creating it if it doesn't exist yet"""
        ...

    def create_array(
        self,
        group: zarr.Group,
        name: str,
        data: npt.NDArray[np.generic],
        chunks: Union[Tuple[int, ...], Literal["auto"]],
        compression: Optional[BloscCompression],
        attributes: Optional[Attributes],
    ) -> zarr.Array:
        """Creates a member array initialized with data"""
        ...

    def read(self, array: zarr.Array, selection: Selection) -> object:
        """Reads (a selection of) the data of an array (0-dimensional selections may result in numpy scalars)"""
        ...

    async def read_async(self, array: zarr.Array, selection: Selection) -> object:
        """Reads (a selection of) the data of an array asynchronously, without blocking the running event loop"""
        ...

    def compression(self, array: zarr.Array) -> Optional[BloscCompression]:
        """The blosc compression of an array (None if uncompressed, raises ValueError for other compressors)"""
        ...

    # -- stores ----------------------------------------------------------------

    def open_directory_store(self, path: str) -> ZarrStore:
        """Opens a directory store (a directory per group / array, a file per record)"""
        ...

    def create_root_group(self, store: StoreLike, zarr_format: Literal[2, 3], attributes: Attributes) -> zarr.Group:
        """Creates the root group of a hierarchy in a store"""
        ...

    def open_store(self, store: StoreLike, consolidated: bool) -> zarr.Group:
        """Opens the root group of a hierarchy for reading, via its compressed consolidated metadata if consolidated
        (raises KeyError if there is no compressed consolidated metadata)"""
        ...

    def consolidate_store(self, store: StoreLike) -> None:
        """Stores the compressed consolidated metadata of the hierarchy of a store"""
        ...

    def group_store(self, group: zarr.Group) -> StoreLike:
        """The store of a group (the indexed tar store for groups of indexed tar archives)"""
        ...


__all__ = [
    "Attributes",
    "attributes_dict",
    "Backend",
    "BloscCompression",
    "Selection",
    "StoreLike",
    "V2_CONSOLIDATED_METADATA_KEY",
    "V3_CONSOLIDATED_METADATA_KEY",
    "ZarrStore",
    "is_v2_metadata_key",
]
