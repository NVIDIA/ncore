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
blocks, enabling O(1) lookups of record payloads. :class:`IndexedTarArchive` provides raw access to the records of such
archives; zarr-python version specific store adapters (see :mod:`ncore.impl.data.stores`) store zarr hierarchies in
them.

Records are *write-once*: archives are append-only, and writing a record twice raises a ``ValueError``. The only
exception are zarr node metadata records (see :func:`is_node_metadata_key`), which zarr-python 3 rewrites on every
update of a node: they are kept in memory while writing and are appended (top-down) when closing the archive.

Layout: the tar records, the tar end-of-archive blocks, the record index (padded to a tar block) and the index header
(one tar block, the last block of the archive), see :class:`IndexFormat`.

Implementation: :class:`IndexFormat` (de)serializes the index (no I/O), the record index of an opened archive
(:class:`RecordIndex`) serves records held in memory, and the bytes of all other records come from a record source
(:class:`_Source`, by the archive's location and mode): positional reads of local archives, range requests or a buffered
file object of remote archives, the file being written. :class:`IndexedTarArchive` composes them.
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

import fsspec.asyn

from typing_extensions import Self
from upath import UPath

from ncore.impl.common.async_util import run_in_thread
from ncore.impl.data.encoding import decode_cbor_xz, encode_cbor_xz


_logger = logging.getLogger(__name__)


#: Names of the records of zarr node metadata (zarr format v2 and v3)
NODE_METADATA_NAMES = (".zarray", ".zgroup", ".zattrs", "zarr.json")


def is_node_metadata_key(key: str) -> bool:
    """True for keys of zarr node metadata records (zarr format v2 and v3)"""
    return key.rsplit("/", 1)[-1] in NODE_METADATA_NAMES


# -------------------------------------------------------------------------------------------------------------------
# Index format
# -------------------------------------------------------------------------------------------------------------------


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


class IndexFormat:
    """(De)serialization of the record index and its header (no I/O)"""

    #: Index header (little-endian): magic (4 bytes), index type (uint32), index offset (uint64), index size (uint32)
    HEADER_FORMAT: ClassVar[str] = "<4sIQI"
    HEADER_MAGIC: ClassVar[bytes] = b"itar"

    #: Index type of written archives
    INDEX_TYPE: ClassVar[IndexType] = IndexType.CBOR_LZMA_XZ_V1

    #: Decoders / encoders of the supported index types (the index is a structure of arrays: record keys, payload
    #: offsets and payload sizes, ordered by payload offsets)
    CODECS: ClassVar[Dict[IndexType, Tuple[Callable[[bytes], Any], Callable[[Any], bytes]]]] = {
        IndexType.CBOR_LZMA_XZ_V1: (decode_cbor_xz, encode_cbor_xz),
    }

    class Location(NamedTuple):
        """The location of the record index (from the index header in the last tar block)"""

        index_type: IndexType
        offset: int
        size: int

    @classmethod
    def locate(cls, tail: TailBuffer) -> IndexFormat.Location:
        """The location of the record index, from a tail read ending at the end of the archive"""
        if len(tail.data) < tarfile.BLOCKSIZE:
            raise ValueError("IndexedTarArchive: invalid index header, can't load indexed tar archive")
        magic, index_type, offset, size = struct.unpack_from(
            cls.HEADER_FORMAT, tail.data, len(tail.data) - tarfile.BLOCKSIZE
        )
        if magic != cls.HEADER_MAGIC:
            raise ValueError("IndexedTarArchive: invalid index header, can't load indexed tar archive")
        if index_type not in cls.CODECS:
            raise TypeError(f"IndexedTarArchive: unsupported index type {index_type}")
        return cls.Location(IndexType(index_type), offset, size)

    @staticmethod
    def index_in_tail(location: IndexFormat.Location, tail: TailBuffer) -> Optional[bytes]:
        """The serialized index, if it lies within the tail read (None otherwise)"""
        if location.offset < tail.start:
            return None
        start = location.offset - tail.start
        return tail.data[start : start + location.size]

    @classmethod
    def decode(cls, location: IndexFormat.Location, index: bytes) -> Dict[str, Record]:
        """The records of a serialized index (of the type and size of its location)"""
        table = cls.CODECS[location.index_type][0](index)
        _logger.debug(f"IndexedTarArchive: loaded index of {len(table['items'])} records ({location.size} bytes)")
        return {
            key: Record(offset, size)
            for key, offset, size in zip(table["items"], table["offset_datas"], table["sizes"])
        }

    @classmethod
    def encode(cls, records: Dict[str, Record], index_offset: int) -> Tuple[bytes, bytes]:
        """The serialized index and its header (not padded) for an index stored at `index_offset`"""
        table = sorted(((key, record.offset, record.size) for key, record in records.items()), key=lambda item: item[1])
        keys, offsets, sizes = (list(column) for column in zip(*table)) if table else ([], [], [])
        index = cls.CODECS[cls.INDEX_TYPE][1]({"items": keys, "offset_datas": offsets, "sizes": sizes})
        return index, struct.pack(cls.HEADER_FORMAT, cls.HEADER_MAGIC, cls.INDEX_TYPE, index_offset, len(index))

    @classmethod
    def tail(cls, data: bytes) -> TailBuffer:
        """The tail buffer of a read of the archive's end of unknown offset (e.g., a suffix range request): the archive
        ends one block after the index, padded to blocks (the index header in the last block locates the index)"""
        location = cls.locate(TailBuffer(0, data))
        archive_size = (
            -(-(location.offset + location.size) // tarfile.BLOCKSIZE) * tarfile.BLOCKSIZE + tarfile.BLOCKSIZE
        )
        return TailBuffer(archive_size - len(data), data)

    @staticmethod
    def tail_size(tail_read_size: Optional[int]) -> int:
        """The size of tail reads (at least the index header block)"""
        return max(tarfile.BLOCKSIZE, tail_read_size or 0)


class RecordIndex:
    """The records of an archive opened for reading, with the tail read of loading the index (records within it are
    served from memory)"""

    def __init__(self, records: Dict[str, Record], tail: TailBuffer) -> None:
        """The records by key, and the tail read the index was loaded from (ending at the end of the archive)"""
        self.records = records
        self.tail = tail
        # records prefetched when opening (see :meth:`IndexedTarArchive.open_async`, e.g. the consolidated metadata
        # record of a zarr hierarchy), held in memory until read once (single-use: removed by the first read)
        self._prefetched: Dict[str, bytes] = {}
        self._lock = threading.Lock()

    def prefetch(self, key: str, value: bytes) -> None:
        """Holds the payload of a record in memory until it's read once (see :meth:`take_prefetched`)"""
        with self._lock:
            self._prefetched[key] = value

    def take_prefetched(self, key: str) -> Optional[bytes]:
        """The payload of a prefetched record, which is removed (None if it wasn't prefetched or read already)"""
        if not self._prefetched:
            return None
        with self._lock:
            return self._prefetched.pop(key, None)

    def in_memory(self, record: Record) -> Optional[bytes]:
        """The payload of a record held in memory (records of the tail read, empty records), None otherwise"""
        if record.size == 0:
            return b""
        # (records end before the index, which ends the tail read: records starting within the tail read are part of it)
        if record.offset >= self.tail.start:
            start = record.offset - self.tail.start
            return self.tail.data[start : start + record.size]
        return None


# -------------------------------------------------------------------------------------------------------------------
# Record sources
# -------------------------------------------------------------------------------------------------------------------


class _Source:
    """Reads byte ranges of an archive and owns its file resources: reads via the archive's (buffered) file object,
    serialized by the lock (archives being written, which the writer appends to with the lock held, and archives of
    filesystems without positional or range reads), see the subclasses for local and remote archives"""

    def __init__(self, open_file: Callable[[bool], IO[bytes]], truncate: bool = False, lazy: bool = False) -> None:
        """Opens the archive's file object via `open_file` (on first use if `lazy`; truncating the file if requested:
        archives being written, re-opened without truncation by :meth:`reload`)"""
        self._open_file = open_file
        self._file: Optional[IO[bytes]] = None if lazy else open_file(truncate)
        self.lock = threading.RLock()  # (shared with the writer: file object positions)

    @property
    def file(self) -> IO[bytes]:
        """The file object (opened on first use if the source is lazy), use with the lock held"""
        if self._file is None:
            self._file = self._open_file(False)
        return self._file

    def read(self, offset: int, size: int) -> bytes:
        """Reads a byte range (thread-safe)"""
        with self.lock:
            # restore the position afterwards (the position to append records to when writing)
            file = self.file
            position = file.tell()
            file.seek(offset)
            value = file.read(size)
            file.seek(position)
            return value

    async def read_async(self, offset: int, size: int) -> bytes:
        """Reads a byte range without blocking the running event loop (on a worker thread)"""
        return await run_in_thread(self.read, offset, size)

    async def load_index_async(self, tail_read_size: Optional[int]) -> RecordIndex:
        """Loads the record index without blocking the running event loop (on a worker thread)"""
        return await run_in_thread(self.load_index, tail_read_size)

    def load_index(self, tail_read_size: Optional[int]) -> RecordIndex:
        """Loads the record index from the end of the archive (with a single read if the index lies within the tail
        read)"""
        with self.lock:
            file_size = self.file.seek(0, os.SEEK_END)
            tail_size = min(file_size, IndexFormat.tail_size(tail_read_size))
            self.file.seek(file_size - tail_size)
            tail = TailBuffer(file_size - tail_size, self.file.read(tail_size))
            location = IndexFormat.locate(tail)
            if (index := IndexFormat.index_in_tail(location, tail)) is None:
                self.file.seek(location.offset)
                index = self.file.read(location.size)
            return RecordIndex(IndexFormat.decode(location, index), tail)

    def reload(self) -> None:
        """Re-opens the file resources (e.g., in a forked process): the file object, if it's open (lazy sources open it
        on first use)"""
        with self.lock:
            if (file := self._file) is not None:
                position = file.tell()
                file.close()
                self._file = self._open_file(False)
                self._file.seek(position)

    def close(self) -> None:
        """Releases the file resources"""
        with self.lock:
            if self._file is not None:
                self._file.close()


class _LocalSource(_Source):
    """Reads a local archive via positional reads (``os.pread``, not serialized) of a file descriptor (closed with the
    source, or when the source is garbage-collected), and loads its index via the file object"""

    def __init__(self, path: str, open_file: Callable[[bool], IO[bytes]]) -> None:
        """Opens the file object (via `open_file`) and the file descriptor of the archive at the local `path`"""
        super().__init__(open_file, truncate=False)
        self._path = path
        self._open_fd()

    def _open_fd(self) -> None:
        """Opens the file descriptor for positional reads (closed by the finalizer: on close, reload, or when the source
        is garbage-collected)"""
        self.fd = os.open(self._path, os.O_RDONLY)
        self._fd_finalizer = weakref.finalize(self, os.close, self.fd)

    def read(self, offset: int, size: int) -> bytes:
        """Reads a byte range via positional reads (thread-safe, not serialized)"""
        value = os.pread(self.fd, size, offset)
        while len(value) < size:  # positional reads may return fewer bytes than requested
            if not (chunk := os.pread(self.fd, size - len(value), offset + len(value))):
                raise EOFError(f"IndexedTarArchive: archive ends within record at offset {offset}")
            value += chunk
        return value

    def reload(self) -> None:
        """Re-opens the file object and the file descriptor"""
        super().reload()
        self._fd_finalizer()
        self._open_fd()

    def close(self) -> None:
        """Closes the file object and the file descriptor"""
        super().close()
        self._fd_finalizer()


class _RemoteSource(_Source):
    """Reads a remote archive (on an asynchronous fsspec filesystem): asynchronously via range requests, synchronously
    via range requests (``remote_range_reads``) or the buffered file object (opened on first use)"""

    def __init__(
        self, fs: fsspec.asyn.AsyncFileSystem, path: str, open_file: Callable[[bool], IO[bytes]], range_reads: bool
    ) -> None:
        """A source of the archive at `path` of the filesystem `fs` (synchronous range reads if `range_reads`), opening
        the buffered file object via `open_file` on first use (only synchronous reads without range reads use it)"""
        super().__init__(open_file, lazy=True)
        self.fs = fs
        self._path = path
        self.range_reads = range_reads

    def load_index(self, tail_read_size: Optional[int]) -> RecordIndex:
        """Loads the record index synchronously, via the requests of :meth:`load_index_async` (no file object)"""
        return fsspec.asyn.sync(self.fs.loop, self.load_index_async, tail_read_size)

    async def load_index_async(self, tail_read_size: Optional[int]) -> RecordIndex:
        """Loads the record index via a suffix range request of the end of the archive (``bytes=-N``), and a range
        request of the index if it doesn't lie within the tail read. The archive's size isn't needed (no metadata
        request): the archive ends one block after the padded index (see :meth:`IndexFormat.tail`)."""
        tail = IndexFormat.tail(await self._cat(start=-IndexFormat.tail_size(tail_read_size)))
        location = IndexFormat.locate(tail)
        if (index := IndexFormat.index_in_tail(location, tail)) is None:
            index = await self._cat(start=location.offset, end=location.offset + location.size)
        # (decoding is CPU work: on a worker thread, not on the filesystem's I/O event loop)
        return RecordIndex(await run_in_thread(IndexFormat.decode, location, index), tail)

    async def _cat(self, start: int, end: Optional[int] = None) -> bytes:
        """A range request on the filesystem's I/O event loop (awaitable from any event loop)"""
        # (`_cat_file` is the coroutine asynchronous fsspec filesystems implement, `cat_file` its synchronous wrapper)
        request = self.fs._cat_file(self._path, start=start, end=end)
        if (fs_loop := self.fs.loop) is None or fs_loop is asyncio.get_running_loop():
            return await request
        # (filesystems are bound to fsspec's I/O event loop, their sessions can't be used on other event loops)
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(request, fs_loop))

    def read(self, offset: int, size: int) -> bytes:
        """Reads a byte range synchronously: via a range request (``remote_range_reads``, concurrent for multiple
        threads), or via the buffered file object (serialized)"""
        if self.range_reads:
            # (an exact range request on the filesystem's I/O loop, concurrent with those of other threads)
            return self.fs.cat_file(self._path, start=offset, end=offset + size)
        return super().read(offset, size)

    async def read_async(self, offset: int, size: int) -> bytes:
        """Reads a byte range via an asynchronous range request (concurrent reads overlap)"""
        return await self._cat(start=offset, end=offset + size)


# -------------------------------------------------------------------------------------------------------------------
# Writing
# -------------------------------------------------------------------------------------------------------------------


class _Writer:
    """Appends records to an archive opened for writing, via the file object of its source (not thread-safe: used with
    the source's lock held, which serializes reads of the file object with appends)"""

    def __init__(self, source: _Source) -> None:
        """A writer appending to the file object of `source` (truncated, opened for writing)"""
        self._source = source
        self._tar_file = tarfile.TarFile(fileobj=source.file, mode="w")
        self.records: Dict[str, Record] = {}
        #: Node metadata records, held in memory until closing (rewritable, see the module documentation)
        self.node_metadata: Dict[str, bytes] = {}

    def append(self, key: str, payload: bytes) -> None:
        """Appends a record (a tar member) to the archive"""
        # the header of a record can span multiple tar blocks (long keys), the payload is padded to full blocks
        info = tarfile.TarInfo(key)
        info.size = len(payload)
        self._tar_file.addfile(info, fileobj=io.BytesIO(payload))
        padded_size = -(-len(payload) // tarfile.BLOCKSIZE) * tarfile.BLOCKSIZE
        self.records[key] = Record(self._source.file.tell() - padded_size, len(payload))

    def reload(self) -> None:
        """Appends to the source's re-opened file object (after :meth:`_Source.reload`)"""
        self._tar_file.fileobj = self._source.file

    def finish(self) -> None:
        """Appends the node metadata records, the tar end-of-archive blocks, the record index and its header"""
        # (deterministic order independent of the order of node updates: parents before children)
        node_metadata, self.node_metadata = self.node_metadata, {}
        for key in sorted(node_metadata, key=lambda key: (key.count("/"), key)):
            self.append(key, node_metadata[key])
        # closing the tar file appends the end-of-archive blocks, but keeps the file object open
        self._tar_file.close()

        file = self._source.file

        def pad_to_block() -> None:
            if remainder := file.tell() % tarfile.BLOCKSIZE:
                file.write(tarfile.NUL * (tarfile.BLOCKSIZE - remainder))

        index_offset = file.tell()
        assert index_offset % tarfile.BLOCKSIZE == 0, "IndexedTarArchive: archive end not at a tar block boundary"
        index, header = IndexFormat.encode(self.records, index_offset)
        file.write(index)
        pad_to_block()
        file.write(header)
        pad_to_block()
        _logger.debug(f"IndexedTarArchive: saved index of {len(self.records)} records ({len(index)} bytes)")


# -------------------------------------------------------------------------------------------------------------------
# Archive
# -------------------------------------------------------------------------------------------------------------------


class IndexedTarArchive:
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
    remote_range_reads : bool, optional, default False
        Read records of remote archives (see :attr:`is_remote`) synchronously via exact byte-range requests, like
        :meth:`read_async`, instead of via the archive's buffered file object (blocks of the filesystem's block size,
        cached by its cache type, one request at a time). Synchronous reads of multiple threads then fetch their records
        concurrently, which suits random access to high-latency storage (e.g., thread pools reading frames of S3
        archives); the buffered file object suits sequential reading of nearby records.

    Records are accessed via a mapping interface (``archive[key]``, ``archive[key] = value``, ``key in archive``,
    ``keys()``, ``len(archive)``). Records of the index tail read are served from memory, so records stored near the end
    of the archive don't repeat its high-latency I/O. Local archives are read via positional reads (``os.pread``),
    which don't serialize reads of multiple threads.

    Archives opened for writing need to be closed (``close()``, or by using the archive as context manager), which
    appends the node metadata records and the record index. Until then, node metadata records can be rewritten (see
    the module documentation), and are served from memory. Closed archives release their file resources: records not
    held in memory can't be read anymore.

    All methods are thread-safe.
    """

    def __init__(
        self,
        itar_path: Union[str, Path, UPath],
        mode: Literal["r", "w"] = "r",
        index_tail_read_size: Optional[int] = 1 << 20,  # 1 MiB by default
        remote_range_reads: bool = False,
    ) -> None:
        """Opens the archive synchronously (blocking: e.g., the index read of remote archives, see the class
        documentation for the parameters), see :meth:`open_async` for opening archives for reading asynchronously"""
        self._open(itar_path, mode, remote_range_reads)
        if mode == "r":
            self._index = self._source.load_index(index_tail_read_size)

    @classmethod
    async def open_async(
        cls,
        itar_path: Union[str, Path, UPath],
        index_tail_read_size: Optional[int] = 1 << 20,
        remote_range_reads: bool = False,
        prefetch_keys: Tuple[str, ...] = (),
    ) -> IndexedTarArchive:
        """Opens an archive for reading asynchronously, without blocking the running event loop (see the class
        documentation for the parameters).

        The index of remote archives (see :attr:`is_remote`) is loaded via asynchronous range requests (one if it lies
        within the tail read), so opening many archives concurrently (e.g., via ``asyncio.gather``) overlaps their
        requests. Other archives (e.g., local) are opened synchronously on a worker thread.

        Records of ``prefetch_keys`` (if they exist) are read asynchronously while opening and held in memory until
        they are read (once): e.g., the consolidated metadata record of a zarr hierarchy, which is read synchronously
        when opening the hierarchy, then not needed anymore.
        """
        archive = cls.__new__(cls)
        archive._open(itar_path, "r", remote_range_reads)
        index = archive._index = await archive._source.load_index_async(index_tail_read_size)
        keys = [key for key in prefetch_keys if key in index.records]
        for key, value in zip(keys, await asyncio.gather(*(archive.read_async(key) for key in keys))):
            index.prefetch(key, value)
        return archive

    def _open(self, itar_path: Union[str, Path, UPath], mode: Literal["r", "w"], remote_range_reads: bool) -> None:
        """Opens the archive's record source by the archive's location and mode (archives opened for reading load
        their index after, see :meth:`__init__` and :meth:`open_async`)"""
        if mode not in ("r", "w"):
            raise ValueError(f"IndexedTarArchive: unsupported mode {mode!r}, supported are 'r' and 'w'")
        self._mode: Literal["r", "w"] = mode
        self._remote_range_reads = remote_range_reads
        self._itar_upath = _absolute_upath(itar_path)
        self._closed = False

        self._writer: Optional[_Writer] = None
        self._index: Optional[RecordIndex] = None
        self._source: _Source
        if mode == "w":
            self._source = source = _Source(self._open_file, truncate=True)
            self._writer = _Writer(source)
        elif self._itar_upath.protocol in ("file", "local") and hasattr(os, "pread"):
            self._source = _LocalSource(self._itar_upath.path, self._open_file)
        elif isinstance(fs := self._itar_upath.fs, fsspec.asyn.AsyncFileSystem):
            self._source = _RemoteSource(fs, self._itar_upath.path, self._open_file, remote_range_reads)
        else:
            self._source = _Source(self._open_file)

    def _open_file(self, truncate: bool) -> IO[bytes]:
        """Opens the archive's file object (read-only for reading, readable and writable for writing)"""
        if self._mode == "r":
            return self._itar_upath.open("rb")
        # writing requires a readable file (universal_pathlib <= 0.2.6, the last version supporting python 3.8,
        # supports "wb+" / "rb+" without declaring them in its type annotations)
        return self._itar_upath.open("wb+" if truncate else "rb+")

    @property
    def mode(self) -> Literal["r", "w"]:
        """Access mode of the archive ('r' for reading, 'w' for writing)"""
        return self._mode

    @property
    def itar_upath(self) -> UPath:
        """Absolute location of the archive"""
        return self._itar_upath

    @property
    def remote_range_reads(self) -> bool:
        """True if synchronous reads of remote archives use exact range requests (see the class documentation)"""
        return self._remote_range_reads

    @property
    def is_remote(self) -> bool:
        """True for archives opened for reading on asynchronous fsspec filesystems (e.g., s3fs, gcsfs, http), whose
        records are read via asynchronous range requests by :meth:`read_async`"""
        return isinstance(self._source, _RemoteSource)

    def __eq__(self, value: object) -> bool:
        return isinstance(value, IndexedTarArchive) and (self._itar_upath, self._mode) == (
            value._itar_upath,
            value._mode,
        )

    def __hash__(self) -> int:
        return hash((self._itar_upath, self._mode))

    def __repr__(self) -> str:
        return f"IndexedTarArchive('{self._itar_upath}', mode='{self._mode}')"

    # -------------------------------------------------------------------------
    # Records
    # -------------------------------------------------------------------------

    @property
    def _records(self) -> Dict[str, Record]:
        """The records of the archive (written so far, or of the index)"""
        return self._writer.records if self._writer is not None else cast(RecordIndex, self._index).records

    def _node_metadata(self) -> Dict[str, bytes]:
        """The node metadata records held in memory (archives opened for writing)"""
        return self._writer.node_metadata if self._writer is not None else {}

    def keys(self) -> List[str]:
        """Returns the keys of all records (in archive order, followed by node metadata records that are written when
        closing the archive)"""
        with self._lock:
            return list(self._records) + list(self._node_metadata())

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._records) + len(self._node_metadata())

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._records or key in self._node_metadata()

    def __getitem__(self, key: str) -> bytes:
        """The payload of a record (raises KeyError if it doesn't exist), see :meth:`read`"""
        return self.read(key)

    def read(self, key: str) -> bytes:
        """Reads the payload of a record synchronously (raises KeyError if it doesn't exist, ValueError if it has to be
        read from a closed archive).

        Records held in memory (node metadata records of archives opened for writing, records of the index tail read,
        empty records) are returned directly, records of local archives are read via positional reads, records of
        remote archives via the archive's buffered file object or via exact range requests (see
        ``remote_range_reads``), all others via the archive's file object.
        """
        value = self._lookup(key)
        return value if isinstance(value, bytes) else self._source.read(*value)

    async def read_async(self, key: str) -> bytes:
        """Reads the payload of a record asynchronously, without blocking the running event loop (raises KeyError if it
        doesn't exist, ValueError if it has to be read from a closed archive).

        Records held in memory are returned directly (see :meth:`read`). Records of remote archives (see
        :attr:`is_remote`) are read via asynchronous range requests (concurrent reads overlap), records of all other
        archives (e.g., local) on a worker thread.
        """
        value = self._lookup(key)
        return value if isinstance(value, bytes) else await self._source.read_async(*value)

    def _lookup(self, key: str) -> Union[bytes, Record]:
        """The payload of a record held in memory, or the record to read from the source (raises KeyError if it doesn't
        exist, ValueError if it has to be read from a closed archive, whose file resources are released: e.g., the file
        descriptor of a closed local archive may have been reused for another file)"""
        if self._writer is not None:
            with self._lock:
                if (value := self._writer.node_metadata.get(key)) is not None:
                    return value
                record = self._writer.records[key]
            if record.size == 0:
                return b""
        else:
            index = cast(RecordIndex, self._index)
            if (value := index.take_prefetched(key)) is not None:
                return value
            record = index.records[key]
            if (value := index.in_memory(record)) is not None:
                return value
        if self._closed:
            raise ValueError("IndexedTarArchive: archive is closed")
        return record

    def __setitem__(self, key: str, value: Union[bytes, bytearray, memoryview]) -> None:
        """Appends a record (requires write mode, raises ValueError if the record exists already). Node metadata
        records are appended when closing the archive (and can be rewritten until then)."""
        if self._writer is None:
            raise ValueError("IndexedTarArchive: archive is opened for reading and doesn't support writing")
        payload = bytes(value)

        with self._lock:
            if self._closed:
                raise ValueError("IndexedTarArchive: archive is closed")
            if is_node_metadata_key(key):
                self._writer.node_metadata[key] = payload
            elif key in self._writer.records:
                raise ValueError(f"IndexedTarArchive: record {key} already exists, records are write-once")
            else:
                self._writer.append(key, payload)

    def __delitem__(self, key: str) -> None:
        raise NotImplementedError("IndexedTarArchive: records can't be deleted")

    @property
    def _lock(self) -> threading.RLock:
        """The lock of the archive's file object (writing, file object based reads)"""
        return self._source.lock

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        """Closes the archive: when writing, appends the node metadata records and the record index (which is required
        to read the archive), and releases the archive's file resources (records held in memory remain readable)"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._writer is not None:
                self._writer.finish()
            self._source.close()

    def reload_resources(self) -> None:
        """Re-opens the file resources of the archive's record source (file object, file descriptor of local archives),
        and lets the writer of archives opened for writing append to the re-opened file object (required in processes
        forked after opening the archive)"""
        with self._lock:
            self._source.reload()
            if self._writer is not None:
                self._writer.reload()


def _absolute_upath(itar_path: Union[str, Path, UPath]) -> UPath:
    """An absolute UPath (with the UPath-internal `file://` protocol for local files)"""
    itar_upath = UPath(itar_path)
    if itar_upath.protocol == "":
        itar_upath = UPath("file://" + str(itar_upath))
    return itar_upath.absolute()
