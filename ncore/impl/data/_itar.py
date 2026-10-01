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

"""zarr-independent implementation of *indexed* tar archive stores (``.itar``).

An indexed tar archive is a regular tar archive with a compressed record index appended after the tar end-of-archive
blocks, enabling O(1) lookups of record payloads. :class:`IndexedTarStore` implements zarr hierarchies stored in such
archives independently of zarr-python, the zarr-python 2 / 3 specific store adapters (used internally by
:mod:`ncore.impl.data.stores`) only forward to it.

On-disk layout
--------------

* **zarr format v2** hierarchies store ``.zgroup`` / ``.zattrs`` / ``.zarray`` node metadata and chunk records, plus
  the compressed consolidated metadata of the hierarchy (``.zmetadata.cbor.xz``, CBOR + LZMA encoded ``.zmetadata``).
  Archives are identical for zarr-python 2 and 3.
* **zarr format v3** hierarchies store ``zarr.json`` node metadata and chunk records. The root ``zarr.json``, which
  contains the consolidated metadata of the hierarchy, is only stored in compressed form (``zarr.json.cbor.zlib``,
  CBOR + zlib encoded, which decodes several times faster than CBOR + LZMA).

Records are *write-once*: archives are append-only, and writing a record twice raises a ``ValueError``. zarr-python 3
rewrites node metadata on every attribute update, so with zarr-python 3 node metadata writes are deferred until the
store is closed and each metadata record is written exactly once, in its final state. Chunk records are written
immediately. zarr-python 2 writes each node metadata record once (in creation order, the layout of ncore <= 19.8).

Reading
-------

The compressed consolidated metadata is decoded on first use and served from memory: as consolidated metadata
document (``.zmetadata`` for zarr format v2, the root ``zarr.json`` including its consolidated metadata for zarr
format v3), and as individual node metadata documents (for hierarchies opened without consolidated metadata).
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import lzma
import os
import struct
import tarfile
import threading
import weakref
import zlib

from dataclasses import dataclass, field
from enum import IntEnum, auto, unique
from pathlib import Path
from threading import RLock
from typing import IO, ClassVar, Dict, Iterator, List, Literal, NamedTuple, Optional, Tuple, Union, cast

import cbor2

from fsspec.asyn import AsyncFileSystem
from upath import UPath

from ncore.impl.data._json import JsonLike


_logger = logging.getLogger(__name__)


#: Key of the compressed consolidated metadata of zarr format v2 hierarchies (CBOR + LZMA encoded ``.zmetadata``)
V2_CONSOLIDATED_METADATA_KEY = ".zmetadata.cbor.xz"

#: Key of the compressed consolidated metadata of zarr format v3 hierarchies (CBOR + zlib encoded root ``zarr.json``,
#: which contains the consolidated metadata of the hierarchy)
V3_CONSOLIDATED_METADATA_KEY = "zarr.json.cbor.zlib"

#: zarr-python's key of uncompressed zarr format v2 consolidated metadata
ZMETADATA_KEY = ".zmetadata"

#: zarr-python's key of zarr format v3 node metadata
ZARR_JSON_KEY = "zarr.json"

#: Archive records which are not part of the zarr hierarchy (hidden from listings)
INTERNAL_KEYS = frozenset((V2_CONSOLIDATED_METADATA_KEY, V3_CONSOLIDATED_METADATA_KEY))

#: A JSON-like document (dictionary at the top level)
JsonDocument = Dict[str, JsonLike]


def compress_document(document: JsonDocument) -> bytes:
    """Encodes a JSON-like document as CBOR + LZMA"""
    with io.BytesIO() as buffer:
        with lzma.open(buffer, "wb") as lzma_file:
            cbor2.dump(document, lzma_file)
        return buffer.getvalue()


def decompress_document(data: bytes) -> JsonDocument:
    """Decodes a CBOR + LZMA encoded JSON-like document (raises ValueError for non-dictionary documents)"""
    return _document(cbor2.loads(lzma.LZMADecompressor().decompress(data)))


def compress_document_zlib(document: JsonDocument) -> bytes:
    """Encodes a JSON-like document as CBOR + zlib (decodes several times faster than CBOR + LZMA)"""
    return zlib.compress(cbor2.dumps(document), 9)


def decompress_document_zlib(data: bytes) -> JsonDocument:
    """Decodes a CBOR + zlib encoded JSON-like document (raises ValueError for non-dictionary documents)"""
    return _document(cbor2.loads(zlib.decompress(data)))


def _document(document: object) -> JsonDocument:
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON-like dictionary document, got {type(document).__name__}")
    return cast(JsonDocument, document)


def is_v2_metadata_key(key: str) -> bool:
    """Returns true if the key refers to zarr format v2 node metadata"""
    return key.endswith(".zarray") or key.endswith(".zgroup") or key.endswith(".zattrs")


def is_v3_metadata_key(key: str) -> bool:
    """Returns true if the key refers to zarr format v3 node metadata"""
    return key == ZARR_JSON_KEY or key.endswith("/" + ZARR_JSON_KEY)


def is_metadata_key(key: str) -> bool:
    """Returns true if the key refers to zarr format v2 / v3 node metadata"""
    return is_v2_metadata_key(key) or is_v3_metadata_key(key)


def parent_grouped(node_documents: JsonDocument, separator: str = "/") -> JsonDocument:
    """Orders flat consolidated node metadata by node depth, with members of the same parent group adjacent.

    zarr-python < 3.4 nests flat consolidated metadata assuming this order (zarr-python#4226, the members of a group
    are attached to the first run of keys of the group only), which isn't guaranteed for hierarchies created in
    arbitrary order (e.g., by multiple component writers). Keys are node paths (zarr format v3, ``separator="/"``) or
    node metadata keys (zarr format v2, ``separator="/."``).
    """

    def order(key: str) -> Tuple[int, str, str]:
        path = key.rsplit(separator, 1)[0] if separator != "/" and separator in key else key
        parent, _, name = path.rpartition("/")
        return path.count("/"), parent, name

    return {key: node_documents[key] for key in sorted(node_documents, key=order)}


def _close_owned_fd(fd: int, owner_pid: int) -> None:
    """Closes a file descriptor if it was opened by the current process (descriptors inherited by forked processes
    are left to the parent)"""
    if os.getpid() == owner_pid:
        os.close(fd)


class _TarArchive:
    """The record storage of an indexed tar archive opened for reading or (truncating) writing.

    Records are *write-once*: the archive is append-only and records can't be updated or deleted.

    Read mode retains the byte range loaded for the index (the tail read) as an in-memory cache. Payloads lying
    entirely in that range are served from the cache, so records near the end of the archive don't repeat the same
    high-latency tail I/O. Local archives are read via lock-free positional reads (``os.pread``), which allows
    concurrent reads from multiple threads.

    All methods are thread-safe.
    """

    @dataclass
    class TarRecord:
        """A file record within a tar file"""

        offset_data: int
        size: int

    @dataclass
    class TarRecordIndex:
        """All file records within a tar file"""

        records: Dict[str, _TarArchive.TarRecord] = field(default_factory=dict)

    class TailBuffer(NamedTuple):
        """Cached byte range from the tail of the archive file"""

        start: int  #: File byte offset where the tail read began
        data: bytes  #: Raw bytes read from *start* to the end of the file

    #: Raw file descriptor for lock-free positional reads of local archives in read mode (None if not supported)
    _pread_fd: Optional[int]
    #: Process which opened the positional read file descriptor
    _pread_pid: int
    #: Closes the positional read file descriptor (also if the archive is garbage-collected without being closed)
    _pread_finalizer: Optional[weakref.finalize]

    def __init__(
        self,
        itar_path: Union[str, Path, UPath],
        mode: Literal["r", "w"] = "r",
        index_tail_read_size: Optional[int] = 1 << 20,  # 1 MiB by default
    ) -> None:
        if mode not in ["r", "w"]:
            raise ValueError("IndexedTarStore: only r/w modes supported")

        self.mode = mode

        # The tarfile module is not thread-safe, locking is required for writing and file-object based reading
        self.mutex = RLock()

        # Convert str / Path to absolute UPath unconditionally (use UPath-internal `file://` protocol for local files)
        itar_upath = UPath(itar_path)
        if itar_upath.protocol == "":
            itar_upath = UPath("file://" + str(itar_upath))
        self.itar_upath = itar_upath.absolute()

        self._tail_buffer: Optional[_TarArchive.TailBuffer] = None

        # Open file object (writing requires the file to be both writeable and readable) and tar file (writing only)
        self.tar_file_object: IO[bytes]
        self.tar_file: Optional[tarfile.TarFile] = None
        if self.mode == "r":
            self.tar_file_object = self.itar_upath.open("rb")
            self.index, self._tail_buffer = self._load_tar_index(self.tar_file_object, index_tail_read_size)
        else:
            # universal_path for Python 3.8 (<=0.2.6) doesn't expose a write/read mode in its static type-hints,
            # although "wb+" is still accepted if the FS supports it
            self.tar_file_object = self.itar_upath.open("wb+")  # type: ignore[call-overload]
            self.tar_file = tarfile.TarFile(fileobj=self.tar_file_object, mode="w")
            self.index = self.TarRecordIndex()

        self._init_positional_reads()

    # -------------------------------------------------------------------------
    # Record access
    # -------------------------------------------------------------------------

    def keys(self) -> List[str]:
        """Returns the keys of all records (in insertion / archive order)"""
        with self.mutex:
            return list(self.index.records.keys())

    def __contains__(self, key: object) -> bool:
        return key in self.index.records

    def __len__(self) -> int:
        return len(self.index.records)

    def record_size(self, key: str) -> int:
        """Returns the payload size of a record, raises KeyError if not in the archive"""
        return self.index.records[key].size

    def read(self, key: str, start: int = 0, length: Optional[int] = None) -> bytes:
        """Reads (a sub-range of) the payload of a record, raises KeyError if not in the archive"""
        offset, length = self._payload_range(key, start, length)

        if (value := self._read_from_tail(offset, length)) is not None:
            return value

        if self._pread_fd is not None and self._pread_pid == os.getpid():
            return self._pread(offset, length)

        with self.mutex:
            # Read the value and return tar file to previous location (which is the append position when writing)
            current_position = self.tar_file_object.tell()
            self.tar_file_object.seek(offset)
            value = self.tar_file_object.read(length)
            self.tar_file_object.seek(current_position)
            return value

    async def read_async(self, key: str, start: int = 0, length: Optional[int] = None) -> bytes:
        """Asynchronously reads (a sub-range of) the payload of a record, raises KeyError if not in the archive.

        Archives on asynchronous fsspec filesystems (e.g., s3fs, gcsfs, http) issue concurrent range requests,
        other archives are read synchronously.
        """
        if (fs := self._async_fs()) is None:
            return self.read(key, start, length)

        offset, length = self._payload_range(key, start, length)
        if (value := self._read_from_tail(offset, length)) is not None:
            return value
        if length == 0:
            return b""
        request = fs._cat_file(self.itar_upath.path, start=offset, end=offset + length)
        if (fs_loop := getattr(fs, "loop", None)) is None or fs_loop is asyncio.get_running_loop():
            return await request
        # filesystems are bound to fsspec's I/O event loop (sessions can't be used from other event loops)
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(request, fs_loop))

    def write(self, key: str, value: bytes) -> None:
        """Appends a new record to the archive (requires write mode, keys are write-once)"""
        if self.mode != "w":
            raise ValueError("IndexedTarStore: archive not opened for writing")

        with self.mutex:
            if key in self.index.records:
                raise ValueError(f"{key} already exists, update is not supported")

            value_size = len(value)

            # Current tar file position is the start of the header
            header_start_position = self.tar_file_object.tell()

            # Store value in tar-file (will pre-pend a potentially *multi*-block header depending on key lengths)
            tarinfo = tarfile.TarInfo(key)
            tarinfo.size = value_size
            cast(tarfile.TarFile, self.tar_file).addfile(tarinfo, fileobj=io.BytesIO(value))

            # Determine the effective payload / header sizes (payload rounded up to the block size)
            end_position = self.tar_file_object.tell()
            payload_size = value_size
            if remainder := payload_size % tarfile.BLOCKSIZE:
                payload_size += tarfile.BLOCKSIZE - remainder
            header_size = end_position - header_start_position - payload_size

            self.index.records[key] = self.TarRecord(header_start_position + header_size, value_size)

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    @property
    def closed(self) -> bool:
        return self.tar_file_object.closed

    def __enter__(self) -> _TarArchive:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        """Finalizes the archive (if writing) and closes it - needs to be called after writing"""
        with self.mutex:
            if self.tar_file_object.closed:
                return

            if self.mode == "w":
                # Closing the tar file appends two finishing blocks, but doesn't close the internal file object yet
                cast(tarfile.TarFile, self.tar_file).close()
                self._save_tar_index(self.tar_file_object, self.index)

            self.tar_file_object.close()
            self._close_positional_reads()

    def reload_resources(self) -> None:
        """Reloads the file objects *only* - useful to re-initialize the archive in multi-process 'fork()' settings"""
        with self.mutex:
            current_position = self.tar_file_object.tell()
            self.tar_file_object.close()

            if self.mode == "r":
                self.tar_file_object = self.itar_upath.open("rb")
            else:
                # re-open *without* truncation, see "wb+" comment above for type-hints
                self.tar_file_object = self.itar_upath.open("rb+")  # type: ignore[call-overload]
                cast(tarfile.TarFile, self.tar_file).fileobj = self.tar_file_object

            self.tar_file_object.seek(current_position)

            self._close_positional_reads()
            self._init_positional_reads()

    # -------------------------------------------------------------------------
    # Internal read helpers
    # -------------------------------------------------------------------------

    def _init_positional_reads(self) -> None:
        """Opens a raw file descriptor for lock-free positional reads of local archives in read mode.

        The descriptor is also closed if the archive is garbage-collected without being closed.
        """
        self._pread_fd = None
        self._pread_pid = -1
        self._pread_finalizer = None
        if self.mode == "r" and self.itar_upath.protocol in ("file", "local") and hasattr(os, "pread"):
            self._pread_fd = os.open(self.itar_upath.path, os.O_RDONLY)
            self._pread_pid = os.getpid()
            self._pread_finalizer = weakref.finalize(self, _close_owned_fd, self._pread_fd, self._pread_pid)

    def _close_positional_reads(self) -> None:
        if self._pread_finalizer is not None:
            self._pread_finalizer()
        self._pread_finalizer = None
        self._pread_fd = None

    def _pread(self, offset: int, length: int) -> bytes:
        assert self._pread_fd is not None
        value = os.pread(self._pread_fd, length, offset)
        while len(value) < length:  # short reads are possible in principle
            if not (chunk := os.pread(self._pread_fd, length - len(value), offset + len(value))):
                break
            value += chunk
        return value

    def _payload_range(self, key: str, start: int, length: Optional[int]) -> Tuple[int, int]:
        """Returns the absolute (offset, length) of (a sub-range of) a record's payload, raises KeyError if missing"""
        record = self.index.records[key]
        start = min(max(0, start), record.size)
        length = record.size - start if length is None else max(0, min(length, record.size - start))
        return record.offset_data + start, length

    def _read_from_tail(self, offset: int, length: int) -> Optional[bytes]:
        if (tail := self._tail_buffer) is not None:
            if offset >= tail.start and offset + length <= tail.start + len(tail.data):
                return tail.data[offset - tail.start : offset - tail.start + length]
        return None

    def _async_fs(self) -> Optional[AsyncFileSystem]:
        """Returns the asynchronous fsspec filesystem of a remote archive in read mode (if supported)"""
        if self.mode != "r" or self._pread_fd is not None:
            return None
        fs = self.itar_upath.fs
        return fs if isinstance(fs, AsyncFileSystem) and fs.async_impl else None

    # -------------------------------------------------------------------------
    # Index serialization
    # -------------------------------------------------------------------------

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
        cls, tar_file_object: IO[bytes], index_tail_read_size: Optional[int]
    ) -> Tuple[TarRecordIndex, TailBuffer]:
        """Loads a tar record index from the end of a tar file object.

        Returns the parsed index and a :class:`TailBuffer` representing the single tail read used to locate the index.
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
            # Rare: compressed index > index_tail_read_size - fall back to a second read
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
    def _save_tar_index(cls, tar_file_object: IO[bytes], index: TarRecordIndex) -> None:
        """Saves a tar record index at the end of a tar file object (needs to be finalized / have two empty blocks appended already)"""

        def fill_block() -> None:
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


# -----------------------------------------------------------------------------
# zarr hierarchy store
# -----------------------------------------------------------------------------


class IndexedTarStore:
    """A zarr store over an *indexed* tar archive (independent of the installed zarr-python version).

    Parameters
    ----------
    itar_path : string
        Location of the tar file (needs to end with '.itar').
    mode : string, optional, default 'r'
        One of 'r' to read an existing file, or 'w' to truncate and write a new file.
    index_tail_read_size : int or None, optional, default 1 << 20 (1 MiB)
        Maximum bytes read from the tail of the file in a single I/O call to load the tar index. Covers both the small
        index header and (ideally) the compressed index payload. If None, separate reads are performed for header and
        payload, which is more robust for large indices, but requires two I/O calls.

    Stores are passed to the functions of :mod:`ncore.impl.data.stores`, which use them with the installed
    zarr-python version. Besides that, stores provide a mapping interface for raw record access (``store[key]``,
    ``store[key] = value``, ``key in store``, ``iter(store)``, ``len(store)``, ``store.keys()``), and support the
    context manager protocol.

    After writing to a store, the ``close()`` method must be called, otherwise essential data will not be written to
    the archive file.

    All methods are thread-safe.
    """

    #: Records of zarr-python's keys of consolidated metadata
    _CONSOLIDATED_RECORDS: ClassVar[Dict[str, str]] = {
        ZMETADATA_KEY: V2_CONSOLIDATED_METADATA_KEY,
        ZARR_JSON_KEY: V3_CONSOLIDATED_METADATA_KEY,
    }

    def __init__(
        self,
        itar_path: Union[str, Path, UPath],
        mode: Literal["r", "w"] = "r",
        index_tail_read_size: Optional[int] = 1 << 20,  # 1 MiB by default
    ) -> None:
        self._archive = _TarArchive(itar_path, mode, index_tail_read_size)

        # write mode: node metadata records deferred until closing the store (enabled by zarr-python 3)
        self._defer_metadata = False
        self._deferred: Dict[str, bytes] = {}
        self._deferred_lock = threading.Lock()

        # read mode: node metadata documents of the compressed consolidated metadata (for hierarchies opened without
        # consolidated metadata, lazily initialized), and encoded metadata documents
        self._node_documents: Optional[Dict[str, JsonDocument]] = None
        self._documents_lock = threading.Lock()
        self._encoded_documents: Dict[str, bytes] = {}

    @property
    def mode(self) -> Literal["r", "w"]:
        """Access mode of the store ('r' for reading, 'w' for writing)"""
        return self._archive.mode

    @property
    def itar_upath(self) -> UPath:
        """Absolute location of the archive file"""
        return self._archive.itar_upath

    @property
    def tar_file_object(self) -> IO[bytes]:
        """File object of the archive file"""
        return self._archive.tar_file_object

    def __eq__(self, value: object) -> bool:
        return isinstance(value, IndexedTarStore) and (self.itar_upath, self.mode) == (value.itar_upath, value.mode)

    def __hash__(self) -> int:
        return hash((self.itar_upath, self.mode))

    def __repr__(self) -> str:
        return f"IndexedTarStore('{self.itar_upath}', mode='{self.mode}')"

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    @property
    def closed(self) -> bool:
        """True if the store is closed"""
        return self._archive.closed

    def __enter__(self) -> IndexedTarStore:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        """Closes the store. Needs to be called after writing (writes deferred node metadata and the archive index)"""
        if self.mode == "w" and not self._archive.closed:
            self._flush_deferred()
        self._archive.close()

    def reload_resources(self) -> None:
        """Reloads the file objects *only* - useful to re-initialize the store in multi-process 'fork()' settings"""
        self._archive.reload_resources()

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    @property
    def is_remote(self) -> bool:
        """True for archives read via asynchronous range requests (archives on asynchronous fsspec filesystems)"""
        return self._archive._async_fs() is not None

    @property
    def zarr_format(self) -> Optional[Literal[2, 3]]:
        """zarr format of the hierarchy of an archive with compressed consolidated metadata (None otherwise)"""
        if V3_CONSOLIDATED_METADATA_KEY in self._archive:
            return 3
        if V2_CONSOLIDATED_METADATA_KEY in self._archive:
            return 2
        return None

    def consolidated_metadata(self) -> Optional[JsonDocument]:
        """Decodes the compressed consolidated metadata (read mode, None if not available).

        zarr format v2: the ``.zmetadata`` document, zarr format v3: the root ``zarr.json`` document including the
        consolidated metadata of the hierarchy. Consolidated node metadata is ordered by depth, with the members of
        each group adjacent (see :func:`parent_grouped`). Each call returns a newly decoded document (owned by the
        caller), only the compressed record is retained.
        """
        if self.mode != "r":
            return None
        if V2_CONSOLIDATED_METADATA_KEY in self._archive:
            consolidated = decompress_document(self._archive.read(V2_CONSOLIDATED_METADATA_KEY))
            if (consolidated_format := consolidated.get("zarr_consolidated_format")) != 1:
                raise ValueError(f"unsupported zarr consolidated metadata format: {consolidated_format}")
            return {**consolidated, "metadata": parent_grouped(cast(JsonDocument, consolidated["metadata"]), "/.")}
        if V3_CONSOLIDATED_METADATA_KEY in self._archive:
            return _grouped_root(decompress_document_zlib(self._archive.read(V3_CONSOLIDATED_METADATA_KEY)))
        return None

    def _metadata_documents(self) -> Dict[str, JsonDocument]:
        """Returns all node metadata documents of the consolidated metadata (empty if not available), only used for
        hierarchies opened without consolidated metadata"""
        if (documents := self._node_documents) is not None:
            return documents

        consolidated = self.consolidated_metadata() or {}
        documents: Dict[str, JsonDocument] = {}
        if V2_CONSOLIDATED_METADATA_KEY in self._archive:
            documents[ZMETADATA_KEY] = consolidated
            for key, document in cast(JsonDocument, consolidated.get("metadata") or {}).items():
                documents[key] = cast(JsonDocument, document)
        elif consolidated:
            documents[ZARR_JSON_KEY] = consolidated
            nodes = cast(
                JsonDocument, cast(JsonDocument, consolidated.get("consolidated_metadata") or {}).get("metadata") or {}
            )
            for path, document in nodes.items():
                document = {k: v for k, v in cast(JsonDocument, document).items() if k != "consolidated_metadata"}
                documents[f"{path}/{ZARR_JSON_KEY}"] = document

        with self._documents_lock:
            if self._node_documents is None:
                self._node_documents = documents
            return self._node_documents

    def _metadata_document(self, key: str) -> Optional[bytes]:
        """Returns a node metadata document (or v2 consolidated metadata) from the compressed metadata (read mode)"""
        if self.mode != "r" or not (is_metadata_key(key) or key == ZMETADATA_KEY):
            return None

        # physically present (zarr format v2) node metadata is read without decoding the consolidated metadata
        if self._node_documents is None and key in self._archive:
            return None

        if (encoded := self._encoded_documents.get(key)) is not None:
            return encoded
        if (document := self._metadata_documents().get(key)) is None:
            return None
        # sizes and contents of documents are requested separately (encode once)
        encoded = self._encoded_documents[key] = json.dumps(document).encode("utf-8")
        return encoded

    def _flush_deferred(self) -> None:
        """Writes all deferred node metadata records (top-down), compressing hierarchy-level metadata"""
        with self._deferred_lock:
            deferred, self._deferred = self._deferred, {}

        for key in sorted(deferred, key=lambda k: (k.count("/"), k)):
            if key == ZMETADATA_KEY:
                self._archive.write(V2_CONSOLIDATED_METADATA_KEY, compress_document(json.loads(deferred[key])))
            elif key == ZARR_JSON_KEY:
                self._archive.write(V3_CONSOLIDATED_METADATA_KEY, compress_document_zlib(json.loads(deferred[key])))
            else:
                self._archive.write(key, deferred[key])

    # -------------------------------------------------------------------------
    # Record access
    # -------------------------------------------------------------------------

    def get(self, key: str, start: int = 0, length: Optional[int] = None) -> Optional[bytes]:
        """Reads (a sub-range of) a record (None if it doesn't exist).

        Reads are synchronous: local archives are read via positional reads, remote archives via their file object,
        and node metadata is served from memory (decoded compressed consolidated metadata, or deferred records).
        """
        with self._deferred_lock:
            value = self._deferred.get(key)
        if value is None:
            value = self._metadata_document(key)
        if value is not None:
            return _slice(value, start, length)

        if key in self._archive:
            return self._archive.read(key, start, length)
        return None

    async def get_async(self, key: str, start: int = 0, length: Optional[int] = None) -> Optional[bytes]:
        """Asynchronously reads (a sub-range of) a record (None if it doesn't exist).

        Chunk records of archives on asynchronous fsspec filesystems (e.g., s3fs, gcsfs, http) in read mode are read
        via asynchronous range requests (so concurrent reads overlap), all other reads are synchronous (see
        :meth:`get`).
        """
        if self.mode == "r" and key in self._archive and not is_metadata_key(key):
            return await self._archive.read_async(key, start, length)
        return self.get(key, start, length)

    def record_size(self, key: str) -> Optional[int]:
        """Returns the size of a record (None if it doesn't exist)"""
        with self._deferred_lock:
            value = self._deferred.get(key)
        if value is None:
            value = self._metadata_document(key)
        if value is not None:
            return len(value)
        return self._archive.record_size(key) if key in self._archive else None

    def defer_metadata(self) -> None:
        """Defers writing node metadata records until the store is closed (write mode).

        zarr-python 3 rewrites node metadata on every attribute update, which write-once archives don't support. With
        deferred metadata, only the last version of each node metadata record is written (on closing the store).
        zarr-python 2 writes each node metadata record once, in creation order (the layout of ncore <= 19.8).
        """
        self._defer_metadata = True

    def set(self, key: str, value: bytes) -> None:
        """Writes a record (requires write mode).

        Records are write-once: records are appended to the archive immediately (writing an existing record raises a
        ValueError), except node metadata records if metadata is deferred (see :meth:`defer_metadata`).
        """
        if self.mode != "w":
            raise ValueError("IndexedTarStore: store was opened in read-only mode and doesn't support writing")
        if self._defer_metadata and (is_metadata_key(key) or key in self._CONSOLIDATED_RECORDS):
            with self._deferred_lock:
                self._deferred[key] = value
        else:
            self._archive.write(key, value)

    def exists(self, key: str) -> bool:
        """Returns true if a record exists"""
        with self._deferred_lock:
            if key in self._deferred:
                return True
        return self._metadata_document(key) is not None or key in self._archive

    def keys(self) -> List[str]:
        """Returns the keys of all records (including node metadata served from the compressed consolidated metadata,
        or deferred node metadata)"""
        keys = self._archive.keys()
        if self.mode == "r":
            present = set(keys)
            keys.extend(k for k in self._metadata_documents() if k not in present)
        else:
            with self._deferred_lock:
                keys.extend(k for k in self._deferred if k not in self._archive)
        return keys

    def list_dir(self, prefix: str) -> List[str]:
        """Returns the names of the direct children of a prefix (excluding internal records)"""
        prefix = prefix.rstrip("/")
        prefix = prefix + "/" if prefix else ""
        children: Dict[str, None] = {}
        for key in self.keys():
            if key.startswith(prefix) and key not in INTERNAL_KEYS:
                children[key[len(prefix) :].split("/", 1)[0]] = None
        return list(children)

    # -------------------------------------------------------------------------
    # Mapping interface (raw records)
    # -------------------------------------------------------------------------

    def __getitem__(self, key: str) -> bytes:
        if (value := self.get(key)) is None:
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: Union[bytes, bytearray, memoryview]) -> None:
        self.set(key, bytes(value))

    def __delitem__(self, _: str) -> None:
        raise NotImplementedError("IndexedTarStore: deleting records is not supported")

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self.exists(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())


def _grouped_root(root: JsonDocument) -> JsonDocument:
    """Returns a root zarr.json document with its consolidated node metadata ordered by parent group"""
    root = dict(root)
    if isinstance(consolidated := root.get("consolidated_metadata"), dict) and isinstance(
        nodes := consolidated.get("metadata"), dict
    ):
        root["consolidated_metadata"] = {**consolidated, "metadata": parent_grouped(cast(JsonDocument, nodes))}
    return root


def _slice(value: bytes, start: int, length: Optional[int]) -> bytes:
    start = min(max(0, start), len(value))
    return value[start:] if length is None else value[start : start + max(0, length)]
