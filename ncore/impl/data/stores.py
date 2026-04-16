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

from __future__ import annotations

import io
import logging
import lzma
import os
import struct
import tarfile

from dataclasses import dataclass, field
from enum import IntEnum, auto, unique
from pathlib import Path
from threading import RLock
from typing import IO, Any, Dict, Iterator, Literal, NamedTuple, Optional, Union, cast

import cbor2
import zarr

from numcodecs import compat
from upath import UPath
from zarr._storage.store import Store
from zarr.util import json_loads


_logger = logging.getLogger(__name__)


class IndexedTarStore(Store):
    """A zarr store over *indexed* tar files

    Parameters
    ----------
    itar_path : string
        Location of the tar file (needs to end with '.itar').
    mode : string, optional, default 'r'
        One of 'r' to read an existing file, or 'w' to truncate and write a new
        file.
    index_tail_read_size : int or None, optional, default 1 << 20 (1 MiB)
        Maximum bytes read from the tail of the file in a single I/O call to load the tar index.
        Covers both the small index header and (ideally) the compressed index payload.
        If not specified, perform separate reads for header and payload,
        which is more robust for large indices, but requires two I/O calls.
        Will be increased to at least one tar block size if specified smaller,
        since the header is located in the last block.
        Default 1 MiB size fits typical use cases, but may be increased if the compressed
        index size is expected to be larger or decreased if remote access chunk sizes are smaller.

    Read mode retains the byte range loaded for the index (the tail read) as an in-memory
    cache. Any store key whose payload lies entirely in that range is served from the cache
    without an additional read, so overlapping zarr chunks near the end of the archive do
    not repeat the same high-latency tail I/O.

    After modifying a IndexedTarStore, the ``close()`` method must be called, otherwise
    essential data will not be written to the underlying files. The IndexedTarStore
    class also supports the context manager protocol, which ensures the ``close()``
    method is called on leaving the context, e.g.::

        >>> with IndexedTarStore('data/array.itar', mode='w') as store:
        ...     z = zarr.zeros((10, 10), chunks=(5, 5), store=store)
        ...     z[...] = 42
        ...     # no need to call store.close()

    """

    _erasable = False

    @dataclass
    class TarRecord:
        """A file record within a tar file"""

        offset_data: int
        size: int

    @dataclass
    class TarRecordIndex:
        """All file records within a tar file"""

        records: Dict[str, IndexedTarStore.TarRecord] = field(default_factory=dict)

    class TailBuffer(NamedTuple):
        """Cached byte range from the tail of the archive file.

        Retains the raw bytes of the single tail I/O read performed when the
        index is loaded so that payloads falling entirely within that range can
        be served from memory without an additional file read.
        """

        start: int  #: File byte offset where the tail read began
        data: bytes  #: Raw bytes read from *start* to the end of the file

    def __init__(
        self,
        itar_path: Union[str, Path, UPath],
        mode: Literal["r", "w"] = "r",
        index_tail_read_size: Optional[int] = 1 << 20,  # 1 MiB by default
    ):
        if mode not in ["r", "w"]:
            raise ValueError("TarRecordIndex: only r/w modes supported")

        # store properties
        self.mode = mode

        # Current understanding is that tarfile module in stdlib is not thread-safe,
        # and so locking is required for both read and write. However, this has not
        # been investigated in detail, perhaps no lock is needed if mode='r'.
        self.mutex = RLock()

        # convert str / Path to absolute UPath uncondtionally
        itar_upath = UPath(itar_path)
        if itar_upath.protocol == "":
            # use UPath-internal `file://` protocol for local files
            itar_upath = UPath("file://" + str(itar_upath))

        self.itar_upath = itar_upath.absolute()

        self._tail_buffer: Optional[IndexedTarStore.TailBuffer] = None

        # open file object (require file to be both writeable and readable when writing) and
        # tar file (only required for write mode, read mode directly uses file object)
        self.tar_file_object: IO[Any]
        self.tar_file: Optional[tarfile.TarFile] = None
        if self.mode == "r":
            self.tar_file_object = self.itar_upath.open("rb")
        else:
            # universal_path for Python 3.8 (<=0.2.6) doesn't expose a
            # write/read mode in it's static type-hints, although "wb+" is still accepted
            # if the FS supports it, so ignore type-checker here
            self.tar_file_object = self.itar_upath.open("wb+")  # type: ignore[call-overload]

            self.tar_file = tarfile.TarFile(fileobj=self.tar_file_object, mode="w")

        # init / load index table
        if self.mode == "r":
            self.index, self._tail_buffer = self._load_tar_index(self.tar_file_object, index_tail_read_size)
        else:
            self.index = self.TarRecordIndex()

    def __delitem__(self, _: str):
        raise NotImplementedError("Deleting items is not supported")

    def __iter__(self) -> Iterator[str]:
        with self.mutex:
            return iter(self.index.records.keys())

    def __len__(self) -> int:
        with self.mutex:
            return len(self.index.records)

    def __contains__(self, item: object) -> bool:
        with self.mutex:
            return item in self.index.records

    def __getitem__(self, item: str) -> bytes:
        with self.mutex:
            # Query index for file record
            record = self.index.records[item]  # raises KeyError if not in archive

            # Check if payload is contained in tail buffer
            if (tail := self._tail_buffer) is not None:
                tail_start = tail.start
                tail_end = tail_start + len(tail.data)
                offset_data = record.offset_data
                if offset_data >= tail_start and offset_data + record.size <= tail_end:
                    return tail.data[offset_data - tail_start : offset_data - tail_start + record.size]

            # Remember current tar file position
            current_position = self.tar_file_object.tell()

            # Read the value
            self.tar_file_object.seek(record.offset_data)
            value = self.tar_file_object.read(record.size)

            # Return tar file to previous location
            self.tar_file_object.seek(current_position)

            return value

    def __setitem__(self, item: str, value):
        if self.mode != "w":
            raise zarr.errors.ReadOnlyError

        with self.mutex:
            if item in self.index.records:
                raise ValueError(f"{item} already exists, update is not supported")

            value_bytes: bytes = compat.ensure_bytes(value)
            value_size: int = len(value_bytes)

            # Remember current tar file position, which is the start of the header
            header_start_position = self.tar_file_object.tell()

            # Store value in tar-file (will pre-pend a potentially *multi*-block header depending on item path-lengths)
            tarinfo = tarfile.TarInfo(item)
            tarinfo.size = value_size

            cast(tarfile.TarFile, self.tar_file).addfile(tarinfo, fileobj=io.BytesIO(value_bytes))

            # End position after writing both header and payload
            end_position = self.tar_file_object.tell()

            # Determine the effective value's payload size as a multiple of blocksize
            payload_size = value_size
            if remainder := payload_size % tarfile.BLOCKSIZE:
                payload_size += tarfile.BLOCKSIZE - remainder

            # Determine the effective header-size (can be multiple blocks for long path names)
            header_size = end_position - header_start_position - payload_size

            # Construct record from reconstructed size-information
            record = self.TarRecord(
                # Effective start of the data in the tar file (current tar file position + header-size)
                header_start_position + header_size,
                # Length of the data
                value_size,
            )

            self.index.records[item] = record

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """Needs to be called after finishing updating the store"""
        with self.mutex:
            if self.mode == "w":
                # Closing the tar file appends two finishing blocks to the end of the file
                # if in write mode, but doesn't close the internal file object yet
                cast(tarfile.TarFile, self.tar_file).close()

                # Add index if writing
                self._save_tar_index(self.tar_file_object, self.index)

            self.tar_file_object.close()

    def reload_resources(self):
        """Reloads the tar file object *only* - useful to re-initialize the store in multi-process 'fork()' settings"""
        with self.mutex:
            # get current tar file path and seek positions, and close file object
            current_position = self.tar_file_object.tell()
            self.tar_file_object.close()

            # reload file object (require file to be both writeable and readable when writing)
            if self.mode == "r":
                self.tar_file_object = self.itar_upath.open("rb")
            else:
                # universal_path for Python 3.8 (<=0.2.6) doesn't expose a
                # write/read mode in it's static type-hints, although "wb+" is still accepted
                # if the FS supports it, so ignore type-checker here
                self.tar_file_object = self.itar_upath.open("wb+")  # type: ignore[call-overload]

            if self.mode == "w":
                # assign the new file object to the tar file (only required for write mode)
                cast(tarfile.TarFile, self.tar_file).fileobj = self.tar_file_object

            # seek to previous position
            self.tar_file_object.seek(current_position)

    # Methods / constants for storing index header and payload
    INDEX_HEADER_MAGIC = b"itar"

    # Index header binary format (20-bytes)
    #
    # <little-endian
    # IndexMagic  - 4s - 4xchar             - 4bytes
    # IndexType   - I  - unsigned int       - 4bytes
    # IndexOffset - Q  - unsigned long long - 8bytes
    # IndexSize   - I  - unsigned int       - 4bytes
    INDEX_HEADER_FORMAT = "<4sIQI"

    class IndexHeader(NamedTuple):
        """A decoded index header"""

        magic: bytes
        type: int
        offset: int
        size: int

    @unique
    class IndexType(IntEnum):
        """Enumerates different possible index storage types"""

        CBOR_LZMA_XZ_V1 = auto()

    @classmethod
    def _load_tar_index(
        cls, tar_file_object: IO[Any], index_tail_read_size: Optional[int]
    ) -> tuple[TarRecordIndex, TailBuffer]:
        """Loads a tar record index from the end of a tar file object.

        Returns the parsed index and a :class:`TailBuffer` representing the
        single tail read used to locate the index.  Callers may reuse the
        buffer to satisfy reads for payloads that fall entirely within that range.
        """
        # Determine tail read size
        if index_tail_read_size is None or index_tail_read_size < tarfile.BLOCKSIZE:
            index_tail_read_size = tarfile.BLOCKSIZE  # explicit separate reads

        original_file_position = tar_file_object.tell()

        # Determine file size
        tar_file_object.seek(0, os.SEEK_END)
        file_size = tar_file_object.tell()

        # Read up to index_tail_read_size from the tail in one read call
        tail_buffer_size = min(file_size, index_tail_read_size)
        tar_file_object.seek(file_size - tail_buffer_size)
        tail_buffer = tar_file_object.read(tail_buffer_size)

        # Extract the header from the last 512-byte block (it's guaranteed that the header fits into a single block)
        header_binary_size = struct.calcsize(cls.INDEX_HEADER_FORMAT)
        header_offset_in_tail_buffer = tail_buffer_size - tarfile.BLOCKSIZE
        header = cls.IndexHeader._make(
            struct.unpack(
                cls.INDEX_HEADER_FORMAT,
                tail_buffer[header_offset_in_tail_buffer : header_offset_in_tail_buffer + header_binary_size],
            )
        )

        if header.magic != cls.INDEX_HEADER_MAGIC:
            raise ValueError("IndexedTarStore: invalid index header, can't load indexed tar file")

        # Try to extract the compressed index payload from the tail buffer
        index_start_in_file = file_size - tail_buffer_size
        if header.offset >= index_start_in_file:
            # Payload is fully within the buffer we already read
            index_start_in_tail_buffer = header.offset - index_start_in_file
            index_binary = tail_buffer[index_start_in_tail_buffer : index_start_in_tail_buffer + header.size]
        else:
            # Rare: compressed index > index_tail_read_size — fall back to a second read
            tar_file_object.seek(header.offset)
            index_binary = tar_file_object.read(header.size)

        tar_file_object.seek(original_file_position)

        if header.type == cls.IndexType.CBOR_LZMA_XZ_V1.value:
            _logger.debug(f"IndexedTarStore: lzma-compressed (xz archive format) index load size={len(index_binary)}")

            # load table (SOA)
            table = cbor2.loads(lzma.LZMADecompressor().decompress(index_binary))
            items = table["items"]
            offset_datas = table["offset_datas"]
            sizes = table["sizes"]
        else:
            raise TypeError(f"IndexedTarStore: unsupported header type {header.type}")

        # Construct record index from loaded table
        index = cls.TarRecordIndex({item: cls.TarRecord(offset_datas[i], sizes[i]) for i, item in enumerate(items)})
        return index, cls.TailBuffer(start=index_start_in_file, data=tail_buffer)

    @classmethod
    def _save_tar_index(cls, tar_file_object: IO[Any], index: TarRecordIndex):
        """Saves a tar record index at the end of a tar file object (needs to be finalized / have two empty blocks appended already)"""

        def fill_block():
            # Fill up block with zeros
            _, remainder = divmod(tar_file_object.tell(), tarfile.BLOCKSIZE)
            if remainder > 0:
                tar_file_object.write(tarfile.NUL * (tarfile.BLOCKSIZE - remainder))

            assert tar_file_object.tell() % tarfile.BLOCKSIZE == 0, "Tar file not at block boundary"

        # Remember where we are storing the index
        index_offset = tar_file_object.tell()

        assert index_offset % tarfile.BLOCKSIZE == 0, "Tar file not at block boundary"

        # Reformat index table as SOA (sorted by offset)
        table = [(item, record.offset_data, record.size) for (item, record) in index.records.items()]
        items, offset_datas, sizes = list(zip(*sorted(table, key=lambda data: data[1]))) if len(table) else ([], [], [])

        # Append compressed table to tar file
        with io.BytesIO() as index_buffer:
            # Compress table to in-memory buffer
            with lzma.open(index_buffer, "wb", format=lzma.FORMAT_XZ) as lzma_file:
                cbor2.dump({"items": items, "offset_datas": offset_datas, "sizes": sizes}, lzma_file)

            index_binary = index_buffer.getvalue()
            index_size = len(index_binary)

            _logger.debug(f"IndexedTarStore lzma-compressed index store size={index_size}")

            # Append buffer to tar file
            tar_file_object.write(index_binary)

            fill_block()

        # Create index header block
        assert struct.calcsize(cls.INDEX_HEADER_FORMAT) <= tarfile.BLOCKSIZE, (
            "Index header larger than single block size"
        )
        header_binary = struct.pack(
            cls.INDEX_HEADER_FORMAT,
            cls.INDEX_HEADER_MAGIC,
            cls.IndexType.CBOR_LZMA_XZ_V1.value,
            index_offset,
            index_size,
        )
        _logger.debug(f"IndexedTarStore: header store size={len(header_binary)}")

        # Append index header to tar file
        tar_file_object.write(header_binary)
        fill_block()


def consolidate_compressed_metadata(store: zarr.storage.BaseStore, metadata_key=".zmetadata.cbor.xz"):
    """Consolidate all metadata for groups and arrays within the given store
    into a single compressed cbor resource and put it under the given key.

    See Also
    --------
    zarr.consolidate_metadata
    """
    store = zarr.storage.normalize_store_arg(store, mode="w")

    version = store._store_version

    if version == 2:

        def is_zarr_key(key):
            return key.endswith(".zarray") or key.endswith(".zgroup") or key.endswith(".zattrs")

    else:
        raise NotImplementedError("Only supporting V2 stores")

    # Collect all meta-data
    out = {
        "zarr_consolidated_format": 1,
        "metadata": {key: json_loads(store[key]) for key in store if is_zarr_key(key)},
    }

    with io.BytesIO() as metadata_buffer:
        # Compress meta-data to in-memory buffer
        with lzma.open(metadata_buffer, "wb") as lzma_file:
            cbor2.dump(out, lzma_file)

        store[metadata_key] = metadata_buffer.getvalue()


class ConsolidatedCompressedMetadataStore(zarr.storage.ConsolidatedMetadataStore):
    """A layer over other storage, where the metadata has been consolidated into a single compressed key."""

    # Overwrite constructor to perform decompression of metadata
    def __init__(self, store: zarr.storage.StoreLike, metadata_key=".zmetadata.cbor.xz"):
        self.store = Store._ensure_store(store)

        # retrieve consolidated metadata
        meta = cbor2.loads(lzma.LZMADecompressor().decompress(self.store[metadata_key]))

        # check format of consolidated metadata
        consolidated_format = meta.get("zarr_consolidated_format", None)
        if consolidated_format != 1:
            raise zarr.MetadataError("unsupported zarr consolidated metadata format: %s" % consolidated_format)

        # decode metadata
        self.meta_store: zarr.storage.Store = zarr.KVStore(meta["metadata"])


def open_compressed_consolidated(
    store: zarr.storage.StoreLike, metadata_key=".zmetadata.cbor.xz", mode="r+", **kwargs
) -> zarr.hierarchy.Group:
    """Open group using metadata previously consolidated and compressed into a single key.

    See Also
    --------
    consolidate_compressed_metadata
    zarr.open_consolidated
    """

    # normalize parameters
    zarr_version = kwargs.get("zarr_version")
    store = zarr.storage.normalize_store_arg(
        store, storage_options=kwargs.get("storage_options"), mode=mode, zarr_version=zarr_version
    )
    if mode not in {"r", "r+"}:
        raise ValueError("invalid mode, expected either 'r' or 'r+'; found {!r}".format(mode))

    path = kwargs.pop("path", None)
    if store._store_version == 2:
        ConsolidatedStoreClass = ConsolidatedCompressedMetadataStore
    else:
        raise NotImplementedError("Only supporting V2 stores")

    # setup metadata store
    meta_store = ConsolidatedStoreClass(store, metadata_key=metadata_key)

    # pass through
    chunk_store = kwargs.pop("chunk_store", None) or store
    return zarr.convenience.open(store=meta_store, chunk_store=chunk_store, mode=mode, path=path, **kwargs)
