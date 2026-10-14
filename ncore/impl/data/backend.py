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
import math
import sys
import typing

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, List, Literal, Optional, Tuple, TypeVar, Union

import numcodecs
import numpy as np
import zarr

from typing_extensions import TypedDict
from upath import UPath

from ncore.impl.data.itar import IndexedTarArchive
from ncore.impl.data.json_types import JsonDocumentLike


if sys.version_info >= (3, 10):
    from types import EllipsisType
else:  # pragma: no cover
    EllipsisType = type(Ellipsis)

if TYPE_CHECKING:
    import numpy.typing as npt

    from zarr.abc.store import Store as ZarrStore  # (type checking runs against zarr-python 3)
else:
    ZarrStore = object


# Consolidation: after writing all nodes of a hierarchy, ncore stores the node metadata of all its members in a single
# compressed record (its consolidated metadata, written by Backend.consolidate), from which the hierarchy is opened
# without reading individual node metadata records. The records have ncore specific keys: zarr-python itself can open
# such hierarchies only as unconsolidated hierarchies (from their node metadata records, which are complete as well).

#: Key of the consolidated metadata of ncore zarr format v2 hierarchies (the ``.zmetadata`` document of zarr-python 2's
#: ``consolidate_metadata``, CBOR + LZMA encoded)
V2_CONSOLIDATED_METADATA_KEY = ".zmetadata.cbor.xz"

#: Key of the consolidated metadata of ncore zarr format v3 hierarchies (the root ``zarr.json`` document with the inline
#: consolidated metadata of all members, as written by zarr-python 3's ``consolidate_metadata``, CBOR + zlib encoded;
#: zlib decodes several times faster than LZMA)
V3_CONSOLIDATED_METADATA_KEY = "zarr.json.cbor.zlib"

#: zarr formats (on-disk format versions of zarr hierarchies)
ZarrFormat = Literal[2, 3]

#: Basic selections of array data (integers, slices and Ellipsis)
Selection = Union[int, slice, EllipsisType, Tuple[Union[int, slice, EllipsisType], ...]]

#: Results of reads of (selections of) arrays: a read-only numpy array of the array's dtype with the shape of the
#: selection, or a numpy scalar of the array's dtype if the selection is a single element (zarr-python versions differ in
#: whether they return 0-dimensional arrays or scalars, e.g., ``numpy.bytes_`` for fixed-length bytes). Arrays stored as
#: a single uncompressed chunk are read as views of their chunk record (no copies, see :meth:`Backend.chunk_view`).
#: :class:`ncore.impl.data.nodes.Array` normalizes results (to arrays, or bytes)
ReadResult = Union["npt.NDArray[Any]", np.generic]

#: Attributes of nodes to write (written as plain JSON, read back as :data:`~ncore.impl.data.json_types.JsonDocument`, see
#: :func:`~ncore.impl.data.json_types.to_json`)
Attributes = JsonDocumentLike

#: The zarr-python store type of a backend (zarr-python 2: ``zarr.storage.BaseStore``, zarr-python 3:
#: ``zarr.abc.store.Store``, which have no common base class). :data:`ZarrStore` is the store type of the installed
#: zarr-python version (statically the one type checking runs against)
StoreT = TypeVar("StoreT")


#: Blosc compressors supported by zarr-python 2 (numcodecs ``Blosc``) and zarr-python 3 (``BloscCodec``)
BloscCname = Literal["blosclz", "lz4", "lz4hc", "zlib", "zstd"]

#: Blosc shuffle modes (numcodecs ``Blosc`` encodes them as 0 / 1 / 2)
BloscShuffle = Literal["noshuffle", "shuffle", "bitshuffle"]

_BLOSC_SHUFFLES: List[BloscShuffle] = ["noshuffle", "shuffle", "bitshuffle"]


class BloscConfigV3(TypedDict):
    """Configuration of the ``blosc`` codec of zarr format v3 (as in zarr-python 3's ``BloscCodec.to_dict()``)"""

    cname: BloscCname
    clevel: int
    shuffle: BloscShuffle
    blocksize: int


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

    @staticmethod
    def from_config_v3(config: BloscConfigV3) -> BloscCompression:
        """From the configuration of a zarr format v3 ``blosc`` codec"""
        return BloscCompression(
            cname=config["cname"], clevel=config["clevel"], shuffle=config["shuffle"], blocksize=config["blocksize"]
        )


ZarrNode = Union[zarr.Group, zarr.Array]


class Backend(ABC, Generic[StoreT]):
    """zarr-python version specific operations of nodes and stores (implemented by the backend of each zarr-python
    version, for the store type of its zarr-python version)"""

    #: zarr formats supported by the zarr-python version
    supported_zarr_formats: Tuple[ZarrFormat, ...]

    # -- nodes -----------------------------------------------------------------

    @abstractmethod
    def zarr_format(self, node: ZarrNode) -> ZarrFormat:
        """The zarr format of a node"""
        ...

    @abstractmethod
    def open_member(self, group: zarr.Group, path: str) -> ZarrNode:
        """Opens a (nested) member of a group (raises KeyError if it doesn't exist)"""
        ...

    @abstractmethod
    def member_names(self, group: zarr.Group) -> List[str]:
        """Names of the direct members of a group"""
        ...

    @abstractmethod
    def has_member(self, group: zarr.Group, name: str) -> bool:
        """True if a group has a direct member with the given name"""
        ...

    @abstractmethod
    def member_groups(self, group: zarr.Group) -> List[Tuple[str, zarr.Group]]:
        """Direct member groups of a group"""
        ...

    @abstractmethod
    def create_group(self, group: zarr.Group, name: str, attributes: Optional[Attributes]) -> zarr.Group:
        """Creates a member group (raises if a member with the name exists already)"""
        ...

    @abstractmethod
    def update_attributes(self, node: ZarrNode, attributes: Attributes) -> None:
        """Updates (merges) the attributes of a node"""
        ...

    @abstractmethod
    def require_group(self, group: zarr.Group, name: str) -> zarr.Group:
        """Returns a member group, creating it if it doesn't exist yet"""
        ...

    @abstractmethod
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

    @staticmethod
    def read_result(value: npt.ArrayLike) -> ReadResult:
        """The result of a read of zarr-python as :data:`ReadResult`.

        numpy scalars are returned as is (they are immutable). Arrays are returned without copies and marked read-only.
        Other values (python scalars, array-likes, which zarr-python's result types include, but its numpy buffers
        don't return) are converted to read-only numpy arrays.
        """
        if isinstance(value, np.generic):
            return value
        array: npt.NDArray[Any] = np.asarray(value)
        array.setflags(write=False)
        return array

    @staticmethod
    def chunk_view(record: bytes, dtype: np.dtype, shape: Tuple[int, ...]) -> Optional[npt.NDArray[Any]]:
        """The data of an array stored as a single uncompressed chunk (a read-only view of the chunk record, no copies),
        None if the record doesn't match the array"""
        if len(record) != dtype.itemsize * math.prod(shape):
            return None
        return np.frombuffer(record, dtype=dtype).reshape(shape)

    @staticmethod
    def blob(value: ReadResult) -> bytes:
        """A binary blob from the result of a complete read of a blob array (see
        :meth:`ncore.impl.data.nodes.Group.create_bytes_array`).

        Blob arrays are 0-dimensional fixed-length bytes arrays (zarr format v2), or 1-dimensional ``uint8`` arrays
        (zarr format v3). Results that are views of a chunk record (see :meth:`chunk_view`) return this record (no
        copy), all other results are copied once. Raises TypeError for results of other arrays.
        """
        if isinstance(value, bytes):  # fixed-length bytes scalar (numpy.bytes_, trailing NUL bytes are stripped)
            return value
        if not isinstance(value, np.ndarray) or not (
            (value.dtype == np.uint8 and value.ndim == 1) or (value.dtype.kind == "S" and value.ndim == 0)
        ):
            raise TypeError(f"Not a binary blob: {getattr(value, 'dtype', type(value).__name__)}")
        record = value.base
        while isinstance(record, np.ndarray):  # (views of views, e.g. reshaped)
            record = record.base
        if isinstance(record, bytes) and len(record) == value.nbytes:
            blob = record  # (the chunk record the array is a view of: no copy)
        else:
            blob = value.tobytes()  # (the copy, e.g. of blobs of multiple chunks)
        # (fixed-length bytes strip trailing NUL bytes, as numpy: no copy without trailing NUL bytes)
        return blob.rstrip(b"\x00") if value.dtype.kind == "S" else blob

    @abstractmethod
    def read(self, array: zarr.Array, selection: Selection) -> ReadResult:
        """Reads (a selection of) the data of an array (see :data:`ReadResult`).

        Fast path: complete reads of arrays stored as a single uncompressed chunk return a read-only view of the chunk
        record (see :meth:`chunk_view`, no copies). Fallback: all other reads (partial reads, compressed / multi-chunk
        arrays, chunks not stored, which read as the fill value) are decoded by zarr-python (see :meth:`read_result`).
        """
        ...

    @abstractmethod
    async def read_async(self, array: zarr.Array, selection: Selection) -> ReadResult:
        """Reads (a selection of) the data of an array asynchronously (see :data:`ReadResult`), without blocking the
        running event loop"""
        ...

    @abstractmethod
    def compression(self, array: zarr.Array) -> Optional[BloscCompression]:
        """The blosc compression of an array (None if uncompressed, raises ValueError for other compressors)"""
        ...

    # -- stores ----------------------------------------------------------------

    @abstractmethod
    def zarr_store(self, store: object) -> StoreT:
        """A zarr-python store of the backend's zarr-python version (raises TypeError for other objects, e.g. stores of
        other zarr-python versions)"""
        ...

    @abstractmethod
    def indexed_tar_store(self, archive: IndexedTarArchive) -> StoreT:
        """A zarr-python store of the records of an indexed tar archive"""
        ...

    @abstractmethod
    def directory_store(self, path: UPath) -> StoreT:
        """A zarr-python store of a local directory (a directory per group / array, a file per record)"""
        ...

    @abstractmethod
    def create_root_group(self, store: StoreT, zarr_format: ZarrFormat, attributes: Attributes) -> zarr.Group:
        """Creates the root group of a hierarchy in a store"""
        ...

    @abstractmethod
    def open_root_group(self, store: StoreT, consolidated: bool) -> zarr.Group:
        """Opens the root group of a hierarchy for reading: from its consolidated metadata record if consolidated
        (raises KeyError if the hierarchy has none), otherwise from its node metadata records"""
        ...

    @abstractmethod
    def consolidate(self, store: StoreT) -> None:
        """Stores the consolidated metadata record of the hierarchy of a store (after writing all nodes)"""
        ...
