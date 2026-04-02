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

"""Zarr store implementations and utilities for ncore data storage.

Backwards Compatibility with v2 .zarr.itar Files
=================================================

Files written by ncore versions prior to the zarr3 migration use zarr format v2
with custom compressed consolidated metadata stored under the ``.zmetadata.cbor.xz``
key (CBOR-encoded, LZMA-compressed).

The :class:`IndexedTarStore` transparently handles reading these legacy files:

- **Non-consolidated reads** (``zarr.open``): zarr3 auto-detects format v2 and reads
  ``.zarray`` / ``.zgroup`` / ``.zattrs`` keys directly from the tar index.
- **Consolidated reads** (``zarr.open_consolidated``): The store intercepts requests
  for ``zarr.json`` and checks for compressed consolidated metadata:

  1. ``zarr.cbor.xz`` (new v3 format) -- decompress CBOR+LZMA, return as JSON
  2. ``.zmetadata.cbor.xz`` (old v2 format) -- decompress, return v2 metadata dict

- :func:`open_store` provides a fallback chain: if consolidated read fails, falls back
  to non-consolidated ``zarr.open()``.

New files are written in zarr format v3 with native zarr3 consolidation,
compressed via the same CBOR+LZMA scheme into ``zarr.cbor.xz``.
"""

from __future__ import annotations

import io
import json
import logging
import lzma
import os
import struct
import tarfile
import threading

from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from enum import IntEnum, auto, unique
from pathlib import Path
from typing import IO, Any, Dict, Literal, NamedTuple, Union

import cbor2
import zarr

from zarr.abc.store import ByteRequest, OffsetByteRequest, RangeByteRequest, Store, SuffixByteRequest
from zarr.core.buffer import Buffer, BufferPrototype

from upath import UPath


_logger = logging.getLogger(__name__)


class IndexedTarStore(Store):
    """A zarr store over *indexed* tar files.

    Parameters
    ----------
    itar_path : string
        Location of the tar file (needs to end with '.itar').
    mode : string, optional
        One of 'r' to read an existing file, or 'w' to truncate and write a new
        file.

    After modifying a IndexedTarStore, the ``close()`` method must be called, otherwise
    essential data will not be written to the underlying files. The IndexedTarStore
    class also supports the context manager protocol, which ensures the ``close()``
    method is called on leaving the context, e.g.::

        >>> with IndexedTarStore('data/array.itar', mode='w') as store:
        ...     z = zarr.open_group(store=store, mode='w', zarr_format=3)
        ...     z.create_array('data', data=np.zeros((10, 10)))
        ...     # no need to call store.close()

    Thread safety
    -------------
    A ``threading.RLock`` protects the shared tar file handle. This is needed because:

    1. zarr3's async event loop runs store coroutines sequentially (no lock needed for
       that alone), but ``reload_resources()`` may be called from user threads
       concurrently with zarr reads, creating a race on the shared file handle.
    2. If ``asyncio.to_thread`` is adopted in the future for non-blocking I/O,
       concurrent thread access becomes real. The RLock handles this already.
    """

    supports_writes: bool = True
    supports_deletes: bool = False
    supports_partial_writes: bool = False
    supports_listing: bool = True

    itar_path: Path | UPath

    _tar_file: tarfile.TarFile
    _index: TarRecordIndex
    _lock: threading.RLock
    _deferred_zarr_json: Dict[str, bytes]

    @dataclass
    class TarRecord:
        """A file record within a tar file"""

        offset_data: int
        size: int

    @dataclass
    class TarRecordIndex:
        """All file records within a tar file"""

        records: Dict[str, IndexedTarStore.TarRecord] = field(default_factory=dict)

    def __init__(self, itar_path: Union[str, Path, UPath], mode: Literal["r", "w"] = "r"):
        if mode not in ["r", "w"]:
            raise ValueError("IndexedTarStore: only r/w modes supported")

        super().__init__(read_only=mode == "r")

        # store properties
        itar_upath = UPath(itar_path)
        if itar_upath.protocol == "":
            # use UPath-internal `file://` protocol for local files
            itar_upath = UPath("file://" + str(itar_upath))

        self.itar_path = itar_upath.absolute()
        self._mode = mode

    def _sync_open(self) -> None:
        """Eagerly open the tar file and load/initialize the index."""
        if self._is_open:
            raise ValueError("store is already open")

        self._lock = threading.RLock()
        self._deferred_zarr_json = {}

        # Open file object and tar file (require file to be both writeable and readable when writing)
        self._tar_file_object: IO[Any]
        if self._mode == "r":
            self._tar_file_object = self.itar_path.open("rb")
        else:
            self._tar_file_object = self.itar_path.open("wb+")  # type: ignore[call-overload]

        self._tar_file = tarfile.TarFile(fileobj=self._tar_file_object, mode=self._mode)

        # init / load index table
        if self._mode == "r":
            self._index = self._load_tar_index(self._tar_file_object)
        else:
            self._index = self.TarRecordIndex()

        self._is_open = True

    async def _open(self) -> None:
        self._sync_open()

    def _ensure_open_sync(self) -> None:
        """Lazily open the store on first I/O if not already open (sync version).

        This is used by sync internal methods (_get, _set) that may be called
        before the async _open() has been awaited (e.g., from close() or reload_resources()).
        """
        if not self._is_open:
            self._sync_open()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.itar_path == other.itar_path

    # -------------------------------------------------------------------------
    # Read operations
    # -------------------------------------------------------------------------

    @staticmethod
    def _apply_byte_range_to_bytes(data: bytes, byte_range: ByteRequest) -> bytes:
        """Apply a :class:`ByteRequest` to an in-memory ``bytes`` object."""
        if isinstance(byte_range, RangeByteRequest):
            return data[byte_range.start : byte_range.end]
        elif isinstance(byte_range, OffsetByteRequest):
            return data[byte_range.offset :]
        elif isinstance(byte_range, SuffixByteRequest):
            return data[-byte_range.suffix :] if byte_range.suffix > 0 else b""
        else:
            raise TypeError(f"Unexpected byte_range type, got {type(byte_range)}.")

    def _get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        """Synchronous get implementation. Must be called under ``self._lock``.

        Checks the in-memory ``_deferred_zarr_json`` overlay first for buffered
        ``zarr.json`` writes that have not yet been flushed to the tar archive
        (see :meth:`_set` for details).

        Then handles transparent decompression of consolidated metadata:

        - When ``key == "zarr.json"`` and a ``zarr.cbor.xz`` record exists, the
          compressed consolidated metadata is decompressed (LZMA -> CBOR -> JSON)
          and returned. This is the path for **new v3 itar files**.
        - When ``key == "zarr.json"`` and a ``.zmetadata.cbor.xz`` record exists,
          the **legacy v2 compressed consolidated metadata** is decompressed and
          returned as-is (the v2 metadata dict). This enables backwards-compatible
          reading of old v2 itar files via ``zarr.open_consolidated()``.
        """
        self._ensure_open_sync()

        # Check the deferred zarr.json overlay first (write-back buffer for
        # child node metadata that zarr3 may rewrite multiple times).
        if key in self._deferred_zarr_json:
            value = self._deferred_zarr_json[key]
            if byte_range is not None:
                value = self._apply_byte_range_to_bytes(value, byte_range)
            return prototype.buffer.from_bytes(value)

        try:
            # Handle consolidated metadata intercept for zarr.json
            consolidated_metadata = False
            legacy_v2_consolidated = False

            if key == "zarr.json":
                if (record := self._index.records.get("zarr.cbor.xz")) is not None:
                    # New v3 compressed consolidated metadata
                    consolidated_metadata = True
                    assert byte_range is None, "Byte range not supported for consolidated metadata"
                elif (record := self._index.records.get(".zmetadata.cbor.xz")) is not None:
                    # Legacy v2 compressed consolidated metadata -- backwards compat
                    legacy_v2_consolidated = True
                    assert byte_range is None, "Byte range not supported for consolidated metadata"
                else:
                    # Regular key lookup
                    record = self._index.records[key]
            else:
                # Regular key lookup
                record = self._index.records[key]
        except KeyError:
            return None

        fileobj = self._tar_file_object

        # Remember current tar file position
        current_position = fileobj.tell()

        # Read the value depending on the byte_range
        if byte_range is None:
            fileobj.seek(record.offset_data)
            value = fileobj.read(record.size)
        elif isinstance(byte_range, RangeByteRequest):
            fileobj.seek(record.offset_data + byte_range.start)
            value = fileobj.read(byte_range.end - byte_range.start)
        elif isinstance(byte_range, OffsetByteRequest):
            fileobj.seek(record.offset_data + byte_range.offset)
            value = fileobj.read(record.size - byte_range.offset)
        elif isinstance(byte_range, SuffixByteRequest):
            fileobj.seek(max(0, record.offset_data + record.size - byte_range.suffix))
            value = fileobj.read(byte_range.suffix)
        else:
            raise TypeError(f"Unexpected byte_range, got {byte_range}.")

        if consolidated_metadata:
            # Decompress new v3 compressed consolidated metadata (LZMA -> CBOR -> JSON)
            meta = cbor2.loads(lzma.LZMADecompressor().decompress(value))

            consolidated_format = meta.get("zarr_consolidated_format", None)
            if consolidated_format != 1:
                raise zarr.errors.MetadataError(
                    "unsupported zarr consolidated metadata format: %s" % consolidated_format
                )

            # Return the inner metadata dict as JSON bytes (expected zarr.json format)
            value = json.dumps(meta["metadata"]).encode("utf-8")

        elif legacy_v2_consolidated:
            # Decompress legacy v2 compressed consolidated metadata
            # and return the raw metadata dict -- zarr.open_consolidated() will
            # fail to parse this as v3 metadata, triggering the fallback in open_store()
            meta = cbor2.loads(lzma.LZMADecompressor().decompress(value))

            consolidated_format = meta.get("zarr_consolidated_format", None)
            if consolidated_format != 1:
                raise zarr.errors.MetadataError(
                    "unsupported zarr consolidated metadata format: %s" % consolidated_format
                )

            # Return raw v2 metadata as JSON -- this won't parse as v3 consolidated
            # metadata, but the fallback in open_store() handles this correctly
            value = json.dumps(meta["metadata"]).encode("utf-8")

        ret = prototype.buffer.from_bytes(value)

        # Return tar file to previous location
        fileobj.seek(current_position)

        return ret

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        assert isinstance(key, str)

        with self._lock:
            return self._get(key, prototype=prototype, byte_range=byte_range)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        out = []

        with self._lock:
            for key, byte_range in key_ranges:
                out.append(self._get(key, prototype=prototype, byte_range=byte_range))

        return out

    async def exists(self, key: str) -> bool:
        self._ensure_open_sync()
        with self._lock:
            return key in self._deferred_zarr_json or key in self._index.records

    # -------------------------------------------------------------------------
    # Write operations
    # -------------------------------------------------------------------------

    def _set(self, key: str, value: Buffer) -> None:
        """Synchronous set implementation. Must be called under ``self._lock``.

        Deferred-write strategy for ``zarr.json`` keys
        -----------------------------------------------
        **All** ``zarr.json`` writes (both first-write and subsequent overwrites)
        are buffered in memory (``_deferred_zarr_json``) and only the final
        version is materialized to the tar when the store is closed (see
        :meth:`close` and :meth:`_flush_deferred`).

        This guarantees **zero dead space** in the tar archive regardless of how
        many times zarr3 rewrites a node's metadata during a session.  zarr3
        writes a ``zarr.json`` once on ``create_group()`` / ``create_array()``
        and then again every time ``group.attrs.update()`` is called -- deferring
        the first write as well means neither version reaches the tar until
        close, when only the final version is written.

        The **root** ``zarr.json`` is also deferred.  During
        :meth:`_flush_deferred`, it is intercepted and compressed as
        ``zarr.cbor.xz`` (CBOR+LZMA) -- the consolidated-metadata format used
        by ncore itar files.

        All other (non ``zarr.json``) keys remain strictly write-once.
        """
        self._ensure_open_sync()

        # --- zarr.json keys: always defer (first-write OR overwrite) ----------
        if key == "zarr.json" or key.endswith("/zarr.json"):
            _logger.debug(f"IndexedTarStore: deferring write of {key}")
            self._deferred_zarr_json[key] = value.to_bytes()
            return

        # --- Non zarr.json keys: write-once -----------------------------------
        key_exists = key in self._index.records or key in self._deferred_zarr_json
        if key_exists:
            raise ValueError(
                f"{key} already exists and is not a zarr.json metadata key; "
                f"overwriting non-metadata keys is not supported in itar format"
            )

        value_bytes: bytes = value.to_bytes()
        self._write_to_tar(key, value_bytes)

    async def set(self, key: str, value: Buffer) -> None:
        self._check_writable()
        self._ensure_open_sync()

        assert isinstance(key, str)

        if not isinstance(value, Buffer):
            raise TypeError(
                f"IndexedTarStore.set(): `value` must be a Buffer instance. Got an instance of {type(value)} instead."
            )

        with self._lock:
            self._set(key, value)

    async def set_partial_values(self, key_start_values: Iterable[tuple[str, int, bytes]]) -> None:
        raise NotImplementedError

    async def delete(self, key: str) -> None:
        # Only raise if the key actually exists -- allows zarr APIs to avoid overhead
        self._check_writable()
        if await self.exists(key):
            raise NotImplementedError("Deleting items is not supported by IndexedTarStore")

    # -------------------------------------------------------------------------
    # Listing operations
    # -------------------------------------------------------------------------

    async def list(self) -> AsyncIterator[str]:
        self._ensure_open_sync()
        with self._lock:
            # Yield keys from the tar index
            for key in self._index.records.keys():
                yield key
            # Yield deferred zarr.json keys not yet in the tar index
            for key in self._deferred_zarr_json:
                if key not in self._index.records:
                    yield key

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        async for key in self.list():
            if key.startswith(prefix):
                yield key

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        prefix = prefix.rstrip("/")

        self._ensure_open_sync()
        # Merge keys from tar index and deferred buffer
        all_keys = set(self._index.records.keys()) | set(self._deferred_zarr_json.keys())
        seen: set[str] = set()

        if prefix == "":
            for key in all_keys:
                top = key.split("/")[0]
                if top not in seen:
                    seen.add(top)
                    yield top
        else:
            for key in all_keys:
                if key.startswith(prefix + "/") and key.strip("/") != prefix:
                    k = key.removeprefix(prefix + "/").split("/")[0]
                    if k not in seen:
                        seen.add(k)
                        yield k

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def _flush_deferred(self) -> None:
        """Write all deferred ``zarr.json`` entries to the tar archive.

        Must be called under ``self._lock`` and BEFORE ``self._tar_file.close()``
        (which finalizes the tar with two empty 512-byte blocks).

        Because ALL ``zarr.json`` writes are deferred (both first-write and
        overwrites), each key is written to the tar exactly **once** -- the
        final version.  This guarantees zero dead space in the archive.

        The root ``zarr.json`` receives special treatment: instead of writing
        it as-is, its content is intercepted and compressed as ``zarr.cbor.xz``
        (CBOR + LZMA) -- the consolidated-metadata format used by ncore itar
        files.  The plain ``zarr.json`` key is NOT written to the tar for the
        root; only ``zarr.cbor.xz`` is.
        """
        if not self._deferred_zarr_json:
            return

        # Handle root zarr.json consolidation intercept: compress as zarr.cbor.xz
        root_zarr_json = self._deferred_zarr_json.pop("zarr.json", None)
        if root_zarr_json is not None:
            consolidated = {
                "zarr_consolidated_format": 1,
                "metadata": json.loads(root_zarr_json),
            }

            with io.BytesIO() as buffer:
                with lzma.open(buffer, "wb") as lzma_file:
                    cbor2.dump(consolidated, lzma_file)
                compressed_bytes = buffer.getvalue()

            # Write zarr.cbor.xz to the tar (not zarr.json)
            self._write_to_tar("zarr.cbor.xz", compressed_bytes)

        # Write remaining (non-root) zarr.json entries
        for key, value_bytes in self._deferred_zarr_json.items():
            self._write_to_tar(key, value_bytes)

        _logger.debug(
            "IndexedTarStore: flushed %d deferred zarr.json entries%s",
            len(self._deferred_zarr_json) + (1 if root_zarr_json is not None else 0),
            " (including root zarr.json → zarr.cbor.xz)" if root_zarr_json is not None else "",
        )
        self._deferred_zarr_json.clear()

    def _write_to_tar(self, key: str, value_bytes: bytes) -> None:
        """Append a single key/value pair to the tar archive and update the index.

        Must be called under ``self._lock``.
        """
        value_size = len(value_bytes)

        # Current position is the start of the new header
        header_start = self._tar_file_object.tell()

        tarinfo = tarfile.TarInfo(key)
        tarinfo.size = value_size
        self._tar_file.addfile(tarinfo, fileobj=io.BytesIO(value_bytes))

        end_position = self._tar_file_object.tell()

        # Compute effective payload size (rounded up to block boundary)
        payload_size = value_size
        if remainder := payload_size % tarfile.BLOCKSIZE:
            payload_size += tarfile.BLOCKSIZE - remainder

        header_size = end_position - header_start - payload_size

        # Update the index to point to the newly written data
        self._index.records[key] = self.TarRecord(
            offset_data=header_start + header_size,
            size=value_size,
        )

    def close(self) -> None:
        """Needs to be called after finishing updating the store.

        Flushes deferred ``zarr.json`` writes, saves the tar index (if
        writing), closes the tar file and underlying file object, and marks the
        store as closed via the zarr3 lifecycle.
        """
        if self._is_open:
            with self._lock:
                if self._mode == "w":
                    # Flush deferred zarr.json overwrites BEFORE closing the
                    # tar (which appends two finishing 512-byte blocks).
                    self._flush_deferred()

                # Closing the tar file appends two finishing blocks to the end of the file
                # if in write mode, but doesn't close the internal file object yet
                self._tar_file.close()

                if self._mode == "w":
                    # Add index if writing
                    self._save_tar_index(self._tar_file_object, self._index)

                self._tar_file_object.close()

        super().close()

    def reload_resources(self) -> None:
        """Reloads the tar file object *only* - useful to re-initialize the store in multi-process 'fork()' settings"""
        with self._lock:
            # get current seek positions and close file object
            current_position = self._tar_file_object.tell()
            self._tar_file_object.close()

            # reload file object (require file to be both writeable and readable when writing)
            if self._mode == "r":
                self._tar_file_object = self.itar_path.open("rb")
            else:
                self._tar_file_object = self.itar_path.open("wb+")  # type: ignore[call-overload]

            self._tar_file.fileobj = self._tar_file_object

            # seek to previous position
            self._tar_file_object.seek(current_position)

    # -------------------------------------------------------------------------
    # Index serialization
    # -------------------------------------------------------------------------

    INDEX_HEADER_MAGIC = b"itar"

    # Index header binary format
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
    def _load_tar_index(cls, tar_file_object: IO[Any]) -> TarRecordIndex:
        """Loads a tar record index from the end of a tar file object"""

        # Load header
        original_file_position = tar_file_object.tell()
        tar_file_object.seek(-tarfile.BLOCKSIZE, os.SEEK_END)
        header_binary = tar_file_object.read(struct.calcsize(cls.INDEX_HEADER_FORMAT))

        # Decode header
        header = cls.IndexHeader._make(struct.unpack(cls.INDEX_HEADER_FORMAT, header_binary))

        # Check magic bytes
        if header.magic != cls.INDEX_HEADER_MAGIC:
            raise ValueError("IndexedTarStore: invalid index header, can't load indexed tar file")

        # Load index based on type
        tar_file_object.seek(header.offset)
        header_binary = tar_file_object.read(header.size)
        tar_file_object.seek(original_file_position)

        if header.type == cls.IndexType.CBOR_LZMA_XZ_V1.value:
            _logger.debug(f"IndexedTarStore: lzma-compressed (xz archive format) index load size={len(header_binary)}")

            # load table (SOA)
            table = cbor2.loads(lzma.LZMADecompressor().decompress(header_binary))
            items = table["items"]
            offset_datas = table["offset_datas"]
            sizes = table["sizes"]
        else:
            raise TypeError(f"IndexedTarStore: unsupported header type {header.type}")

        # Construct record index from loaded table
        return cls.TarRecordIndex({item: cls.TarRecord(offset_datas[i], sizes[i]) for i, item in enumerate(items)})

    @classmethod
    def _save_tar_index(cls, tar_file_object: IO[Any], index: TarRecordIndex) -> None:
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
        header_size = len(header_binary)
        _logger.debug(f"IndexedTarStore: header store size={header_size}")

        # Append index header to tar file
        tar_file_object.write(header_binary)
        fill_block()


def open_store(
    store: Store, open_consolidated: bool, mode: str = "r", **kwargs: Any
) -> zarr.Group:
    """Open a zarr group from a store, with optional consolidated metadata.

    Implements a backwards-compatible interface that handles both new v3 itar files
    and legacy v2 itar files:

    1. If ``open_consolidated=True``, tries ``zarr.open_consolidated()`` first.
       For v3 files, this succeeds because the store intercepts ``zarr.json`` and
       decompresses ``zarr.cbor.xz``. For v2 files, this may fail because the
       decompressed metadata is in v2 format.
    2. On failure, falls back to ``zarr.open()`` which auto-detects format v2
       and reads the individual metadata keys directly.
    3. If ``open_consolidated=False``, uses ``zarr.open()`` directly.

    Parameters
    ----------
    store : zarr.abc.store.Store
        The store to open.
    open_consolidated : bool
        If True, attempt to use consolidated metadata for faster loading.
    mode : str
        Access mode ('r' or 'r+').
    **kwargs
        Additional keyword arguments passed to ``zarr.open()`` / ``zarr.open_consolidated()``.

    Returns
    -------
    zarr.Group
        The opened zarr group.
    """
    try:
        if open_consolidated:
            return zarr.open_consolidated(store=store, mode=mode, **kwargs)
        else:
            return zarr.open(store=store, mode=mode, **kwargs)
    except Exception as e:
        # Fall back to regular opening if consolidated read fails.
        # This handles legacy v2 itar files whose compressed consolidated metadata
        # cannot be parsed as v3 consolidated metadata.
        _logger.warning(f"Failed to open consolidated store: {e}. Falling back to regular store opening.")
        return zarr.open(store=store, mode=mode, **kwargs)


def lz4_codecs() -> list:
    """Default zarr3 compressors for LZ4 Blosc compression.

    Produces compact, quickly-decodable chunks suitable for both local and remote
    storage. Used as the default compressor pipeline for ncore data arrays.

    Returns
    -------
    list
        A list containing a single ``BloscCodec`` configured for LZ4 compression
        with bitshuffle, suitable for the ``compressors`` argument of
        ``group.create_array()``.
    """
    from zarr.codecs import BloscCodec

    return [BloscCodec(cname="lz4", clevel=5, shuffle="bitshuffle")]
