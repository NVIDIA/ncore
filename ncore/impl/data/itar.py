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

"""Indexed tar archives (``.itar``), independent of zarr-python.

An indexed tar archive is a regular tar archive with a compressed record index appended after the tar end-of-archive
blocks, enabling O(1) lookups of record payloads. :class:`IndexedTarStore` provides raw access to the records of such
archives; zarr-python version specific store adapters (see :mod:`ncore.impl.data.stores`) store zarr hierarchies in
them.

Records are *write-once*: archives are append-only, and writing a record twice raises a ``ValueError``. The only
exception are zarr node metadata records (see :func:`is_node_metadata_key`), which zarr-python 3 rewrites on every
update of a node: they are kept in memory while writing and are appended (top-down) when closing the archive.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import struct
import tarfile
import threading
import weakref

from enum import IntEnum, unique
from pathlib import Path
from typing import IO, Any, Callable, ClassVar, Dict, Iterator, List, Literal, NamedTuple, Optional, Tuple, Union, cast

from upath import UPath

from ncore.impl.data.encoding import decode_cbor_xz, encode_cbor_xz


_logger = logging.getLogger(__name__)


#: Names of the records of zarr node metadata (zarr format v2 and v3)
NODE_METADATA_NAMES = (".zarray", ".zgroup", ".zattrs", "zarr.json")


def is_node_metadata_key(key: str) -> bool:
    """True for keys of zarr node metadata records (zarr format v2 and v3)"""
    return key.rsplit("/", 1)[-1] in NODE_METADATA_NAMES


class IndexedTarStore:
    """The records of an indexed tar archive, opened for reading or (truncating) writing.

    Parameters
    ----------
    itar_path : string, Path or UPath
        Location of the archive (local or remote).
    mode : string, optional, default 'r'
        'r' to read an existing archive, or 'w' to create (truncate) and write an archive.
    index_tail_read_size : int or None, optional, default 1 << 20 (1 MiB)
        Maximum bytes read from the end of the archive in a single I/O call to load the record index. Covers both the
        small index header and (ideally) the compressed index. Values smaller than the tar block size (512 bytes), or
        None, result in separate reads of the index header and index, which is more robust for large indices.

    Records are accessed via a mapping interface (``store[key]``, ``store[key] = value``, ``key in store``,
    ``keys()``, ``len(store)``). Records of the index tail read are served from memory, so records stored near the end
    of the archive don't repeat its high-latency I/O. Local archives are read via positional reads (``os.pread``),
    which don't serialize reads of multiple threads.

    Archives opened for writing need to be closed (``close()``, or by using the store as context manager), which
    appends the node metadata records and the record index. Until then, node metadata records can be rewritten (see
    the module documentation), and are served from memory.

    All methods are thread-safe.
    """

    class Record(NamedTuple):
        """The payload of a record within the archive file"""

        offset: int
        size: int

    class TailBuffer(NamedTuple):
        """Bytes read from the end of the archive file when loading the index"""

        start: int  #: File offset of the first byte
        data: bytes

    @unique
    class IndexType(IntEnum):
        """Serialization formats of the record index"""

        CBOR_LZMA_XZ_V1 = 1

    #: Index header (little-endian): magic (4 bytes), index type (uint32), index offset (uint64), index size (uint32)
    INDEX_HEADER_FORMAT: ClassVar[str] = "<4sIQI"
    INDEX_HEADER_MAGIC: ClassVar[bytes] = b"itar"

    #: Index type of written archives
    INDEX_TYPE: ClassVar[IndexType] = IndexType.CBOR_LZMA_XZ_V1

    #: Decoders / encoders of the supported index types (the index is a structure of arrays: record keys, payload
    #: offsets and payload sizes, ordered by payload offsets)
    INDEX_CODECS: ClassVar[Dict[IndexType, Tuple[Callable[[bytes], Any], Callable[[Any], bytes]]]] = {
        IndexType.CBOR_LZMA_XZ_V1: (decode_cbor_xz, encode_cbor_xz),
    }

    def __init__(
        self,
        itar_path: Union[str, Path, UPath],
        mode: Literal["r", "w"] = "r",
        index_tail_read_size: Optional[int] = 1 << 20,  # 1 MiB by default
    ) -> None:
        if mode not in ("r", "w"):
            raise ValueError(f"IndexedTarStore: unsupported mode {mode!r}, supported are 'r' and 'w'")
        self._mode: Literal["r", "w"] = mode

        # convert str / Path to an absolute UPath (with the UPath-internal `file://` protocol for local files)
        itar_upath = UPath(itar_path)
        if itar_upath.protocol == "":
            itar_upath = UPath("file://" + str(itar_upath))
        self._itar_upath = itar_upath.absolute()

        # the tarfile module isn't thread-safe: writing and file-object based reading are serialized
        self._lock = threading.RLock()

        self._records: Dict[str, IndexedTarStore.Record] = {}
        self._node_metadata: Dict[str, bytes] = {}  # (write mode, written when closing)
        self._tail: Optional[IndexedTarStore.TailBuffer] = None
        self._tar_file: Optional[tarfile.TarFile] = None
        self._file: IO[bytes] = self._open_file(truncate=True)
        if mode == "r":
            self._records, self._tail = self._load_index(self._file, index_tail_read_size)
        else:
            self._tar_file = tarfile.TarFile(fileobj=self._file, mode="w")

        self._pread_fd: Optional[int] = None
        self._pread_finalizer: Optional[weakref.finalize] = None
        self._open_positional_reads()

    @property
    def mode(self) -> Literal["r", "w"]:
        """Access mode of the archive ('r' for reading, 'w' for writing)"""
        return self._mode

    @property
    def itar_upath(self) -> UPath:
        """Absolute location of the archive"""
        return self._itar_upath

    def __eq__(self, value: object) -> bool:
        return isinstance(value, IndexedTarStore) and (self._itar_upath, self._mode) == (value._itar_upath, value._mode)

    def __hash__(self) -> int:
        return hash((self._itar_upath, self._mode))

    def __repr__(self) -> str:
        return f"IndexedTarStore('{self._itar_upath}', mode='{self._mode}')"

    # -------------------------------------------------------------------------
    # Records
    # -------------------------------------------------------------------------

    def keys(self) -> List[str]:
        """Returns the keys of all records (in archive order, followed by node metadata records that are written when
        closing the archive)"""
        with self._lock:
            return list(self._records) + list(self._node_metadata)

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._records) + len(self._node_metadata)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._records or key in self._node_metadata

    def __getitem__(self, key: str) -> bytes:
        """Reads the payload of a record (raises KeyError if it doesn't exist)"""
        if self._node_metadata:
            with self._lock:
                if (value := self._node_metadata.get(key)) is not None:
                    return value
        record = self._records[key]

        # (records end before the index, which ends the tail read: records starting within the tail read are part of it)
        if (tail := self._tail) is not None and record.offset >= tail.start:
            start = record.offset - tail.start
            return tail.data[start : start + record.size]

        if self._pread_fd is not None:
            return self._pread(self._pread_fd, record.offset, record.size)

        with self._lock:
            # restore the position afterwards (the position to append records to when writing)
            position = self._file.tell()
            self._file.seek(record.offset)
            value = self._file.read(record.size)
            self._file.seek(position)
            return value

    @property
    def is_remote(self) -> bool:
        """True for archives opened for reading on asynchronous fsspec filesystems (e.g., s3fs, gcsfs, http), whose
        records are read via asynchronous range requests by :meth:`read_async`"""
        return self._async_fs() is not None

    async def read_async(self, key: str) -> bytes:
        """Reads the payload of a record asynchronously (raises KeyError if it doesn't exist).

        Records of remote archives (see :attr:`is_remote`) are read via asynchronous range requests (concurrent reads
        overlap), all other records synchronously (see :meth:`__getitem__`).
        """
        if (fs := self._async_fs()) is None or key in self._node_metadata:
            return self[key]
        record = self._records[key]
        if ((tail := self._tail) is not None and record.offset >= tail.start) or record.size == 0:
            return self[key]

        request = fs._cat_file(self._itar_upath.path, start=record.offset, end=record.offset + record.size)
        if (fs_loop := getattr(fs, "loop", None)) is None or fs_loop is asyncio.get_running_loop():
            return await request
        # (filesystems are bound to fsspec's I/O event loop, their sessions can't be used on other event loops)
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(request, fs_loop))

    def _async_fs(self) -> Any:
        """The asynchronous fsspec filesystem of remote archives opened for reading (None otherwise)"""
        if self._mode != "r" or self._pread_fd is not None:
            return None
        fs = self._itar_upath.fs
        return fs if getattr(fs, "async_impl", False) and hasattr(fs, "_cat_file") else None

    def __setitem__(self, key: str, value: Union[bytes, bytearray, memoryview]) -> None:
        """Appends a record (requires write mode, raises ValueError if the record exists already). Node metadata
        records are appended when closing the archive (and can be rewritten until then)."""
        if self._mode != "w":
            raise ValueError("IndexedTarStore: archive is opened for reading and doesn't support writing")
        payload = bytes(value)

        with self._lock:
            if self._file.closed:
                raise ValueError("IndexedTarStore: archive is closed")
            if is_node_metadata_key(key):
                self._node_metadata[key] = payload
                return
            if key in self._records:
                raise ValueError(f"IndexedTarStore: record {key} already exists, records are write-once")
            self._append(key, payload)

    def _append(self, key: str, payload: bytes) -> None:
        with self._lock:
            # the header of a record can span multiple tar blocks (long keys), the payload is padded to full blocks
            info = tarfile.TarInfo(key)
            info.size = len(payload)
            cast(tarfile.TarFile, self._tar_file).addfile(info, fileobj=io.BytesIO(payload))
            padded_size = -(-len(payload) // tarfile.BLOCKSIZE) * tarfile.BLOCKSIZE
            self._records[key] = self.Record(self._file.tell() - padded_size, len(payload))

    def __delitem__(self, key: str) -> None:
        raise NotImplementedError("IndexedTarStore: records can't be deleted")

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def __enter__(self) -> IndexedTarStore:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        """Closes the archive (when writing: appends the record index, which is required to read the archive)"""
        with self._lock:
            if self._file.closed:
                return
            if self._mode == "w":
                # (deterministic order independent of the order of node updates: parents before children)
                node_metadata, self._node_metadata = self._node_metadata, {}
                for key in sorted(node_metadata, key=lambda key: (key.count("/"), key)):
                    self._append(key, node_metadata[key])
                # closing the tar file appends the end-of-archive blocks, but keeps the file object open
                cast(tarfile.TarFile, self._tar_file).close()
                self._save_index(self._file, self._records)
            self._file.close()
            self._close_positional_reads()

    def reload_resources(self) -> None:
        """Re-opens the file objects (required in processes forked after opening the archive)"""
        with self._lock:
            position = self._file.tell()
            self._file.close()
            self._file = self._open_file(truncate=False)
            self._file.seek(position)
            if self._tar_file is not None:
                self._tar_file.fileobj = self._file
            self._close_positional_reads()
            self._open_positional_reads()

    def _open_file(self, truncate: bool) -> IO[bytes]:
        if self._mode == "r":
            return self._itar_upath.open("rb")
        # writing requires a readable file (universal_pathlib <= 0.2.6, the last version supporting python 3.8,
        # supports "wb+" / "rb+" without declaring them in its type annotations)
        return self._itar_upath.open("wb+" if truncate else "rb+")  # type: ignore[call-overload]

    def _open_positional_reads(self) -> None:
        """Opens a file descriptor for positional reads of local archives opened for reading, which is closed with the
        archive (or when the store is garbage-collected)"""
        if self._mode == "r" and self._itar_upath.protocol in ("file", "local") and hasattr(os, "pread"):
            self._pread_fd = os.open(self._itar_upath.path, os.O_RDONLY)
            self._pread_finalizer = weakref.finalize(self, os.close, self._pread_fd)

    def _close_positional_reads(self) -> None:
        if self._pread_finalizer is not None:
            self._pread_finalizer()
        self._pread_finalizer = None
        self._pread_fd = None

    # -------------------------------------------------------------------------
    # Positional reads / index serialization
    # -------------------------------------------------------------------------

    @staticmethod
    def _pread(fd: int, offset: int, size: int) -> bytes:
        value = os.pread(fd, size, offset)
        while len(value) < size:  # positional reads may return fewer bytes than requested
            if not (chunk := os.pread(fd, size - len(value), offset + len(value))):
                raise EOFError(f"IndexedTarStore: archive ends within record at offset {offset}")
            value += chunk
        return value

    @classmethod
    def _load_index(
        cls, file: IO[bytes], tail_read_size: Optional[int]
    ) -> Tuple[Dict[str, IndexedTarStore.Record], IndexedTarStore.TailBuffer]:
        """Loads the record index from the end of an archive file (with a single read if the index lies within the tail
        read), and returns it together with the tail read"""
        file.seek(0, os.SEEK_END)
        file_size = file.tell()

        tail_size = min(file_size, max(tarfile.BLOCKSIZE, tail_read_size or 0))
        file.seek(file_size - tail_size)
        tail = cls.TailBuffer(file_size - tail_size, file.read(tail_size))

        # the index header is stored in the last tar block
        header_offset = len(tail.data) - tarfile.BLOCKSIZE
        magic, index_type, index_offset, index_size = struct.unpack_from(
            cls.INDEX_HEADER_FORMAT, tail.data, header_offset
        )
        if magic != cls.INDEX_HEADER_MAGIC:
            raise ValueError("IndexedTarStore: invalid index header, can't load indexed tar archive")
        if index_type not in cls.INDEX_CODECS:
            raise TypeError(f"IndexedTarStore: unsupported index type {index_type}")
        decode_index = cls.INDEX_CODECS[cls.IndexType(index_type)][0]

        if index_offset >= tail.start:
            index = tail.data[index_offset - tail.start : index_offset - tail.start + index_size]
        else:
            file.seek(index_offset)
            index = file.read(index_size)

        table = decode_index(index)
        _logger.debug(f"IndexedTarStore: loaded index of {len(table['items'])} records ({index_size} bytes)")
        records = {
            key: cls.Record(offset, size)
            for key, offset, size in zip(table["items"], table["offset_datas"], table["sizes"])
        }
        return records, tail

    @classmethod
    def _save_index(cls, file: IO[bytes], records: Dict[str, IndexedTarStore.Record]) -> None:
        """Appends the record index and its header to a finalized archive file (after the end-of-archive blocks)"""

        def pad_to_block() -> None:
            if remainder := file.tell() % tarfile.BLOCKSIZE:
                file.write(tarfile.NUL * (tarfile.BLOCKSIZE - remainder))

        index_offset = file.tell()
        assert index_offset % tarfile.BLOCKSIZE == 0, "IndexedTarStore: archive end not at a tar block boundary"

        table = sorted(((key, record.offset, record.size) for key, record in records.items()), key=lambda item: item[1])
        keys, offsets, sizes = (list(column) for column in zip(*table)) if table else ([], [], [])
        index = cls.INDEX_CODECS[cls.INDEX_TYPE][1]({"items": keys, "offset_datas": offsets, "sizes": sizes})
        file.write(index)
        pad_to_block()

        file.write(
            struct.pack(cls.INDEX_HEADER_FORMAT, cls.INDEX_HEADER_MAGIC, cls.INDEX_TYPE, index_offset, len(index))
        )
        pad_to_block()
        _logger.debug(f"IndexedTarStore: saved index of {len(records)} records ({len(index)} bytes)")
