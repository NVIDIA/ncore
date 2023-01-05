# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import lzma
import tarfile
import os
import io
import struct
import logging

from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Iterator, Literal, NamedTuple, Union
from threading import RLock
from enum import IntEnum, auto, unique

import cbor2
import zarr
import numcodecs


class IndexedTarStore(zarr._storage.store.Store):
    """ A zarr store over *indexed* tar files
    
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
        ...     z = zarr.zeros((10, 10), chunks=(5, 5), store=store)
        ...     z[...] = 42
        ...     # no need to call store.close()
    
    """

    _erasable = False

    @dataclass
    class TarRecord():
        """ A file record within a tar file """

        offset_data: int
        size: int

    @dataclass
    class TarRecordIndex():
        """ All file records within a tar file """

        records: dict[str, IndexedTarStore.TarRecord] = field(default_factory=dict)

    def __init__(self, itar_path: Union[str, Path], mode: Literal['r', 'w'] = 'r'):

        if mode not in ['r', 'w']:
            raise ValueError('TarRecordIndex: only r/w modes supported')

        # store properties
        itar_path = Path(itar_path).absolute()
        assert itar_path.suffix == '.itar', f"{itar_path} is not a '.itar' file path"

        self.mode = mode

        # Current understanding is that tarfile module in stdlib is not thread-safe,
        # and so locking is required for both read and write. However, this has not
        # been investigated in detail, perhaps no lock is needed if mode='r'.
        self.mutex = RLock()

        # open tar file and file object
        if self.mode == 'w':
            # require file to be both writeable and readable
            self.tar_file = tarfile.TarFile(fileobj=open(itar_path, 'wb+'), mode=self.mode)
        else:
            self.tar_file = tarfile.TarFile(fileobj=open(itar_path, 'rb'), mode=self.mode, ignore_zeros=False)
        self.tar_file_object: BinaryIO = self.tar_file.fileobj  # type: ignore

        # init / load index table
        if mode == 'r':
            self.index = self._load_tar_index(self.tar_file_object)
        else:
            self.index = self.TarRecordIndex()

    def __delitem__(self, _: str):
        raise NotImplementedError('Deleting items is not supported')

    def __iter__(self) -> Iterator[str]:
        with self.mutex:
            return iter(self.index.records.keys())

    def __len__(self) -> int:
        with self.mutex:
            return len(self.index.records)

    def __getitem__(self, item: str) -> bytes:
        # Query index for file record
        record = self.index.records[item]  # raises KeyError if not in archive

        # Remember current tar file position
        current_position = self.tar_file_object.tell()

        # Read the value
        self.tar_file_object.seek(record.offset_data)
        value = self.tar_file_object.read(record.size)

        # Return tar file to previous location
        self.tar_file_object.seek(current_position)

        return value

    def __setitem__(self, item: str, value):
        if self.mode != 'w':
            raise zarr.errors.ReadOnlyError

        with self.mutex:
            if item in self.index.records:
                raise ValueError(f'{item} already exists, update is not supported')

            value_bytes: bytes = numcodecs.compat.ensure_bytes(value)

            record = self.TarRecord(
                # Start of data in tar file (current tar file position + header-size)
                self.tar_file_object.tell() + tarfile.BLOCKSIZE,
                # Length of the data
                len(value_bytes))

            tarinfo = tarfile.TarInfo(item)
            tarinfo.size = record.size

            self.tar_file.addfile(tarinfo, fileobj=io.BytesIO(value_bytes))

            self.index.records[item] = record

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        """ Needs to be called after finishing updating the store """
        with self.mutex:
            self.tar_file.close(
            )  # appends two finishing blocks to the end of the file if in write mode, but doesn't close it yet

            if self.mode == 'w':
                # Add index if writing
                self._save_tar_index(self.tar_file_object, self.index)

            self.tar_file_object.close()

    # Methods / constants for storing index header and payload
    INDEX_HEADER_MAGIC = b"itar"

    # Index header binary format
    #
    # <little-endian
    # IndexMagic  - 4s - 4xchar             - 4bytes
    # IndexType   - I  - unsigned int       - 4bytes
    # IndexOffset - Q  - unsigned long long - 8bytes
    # IndexSize   - I  - unsigned int       - 4bytes
    INDEX_HEADER_FORMAT = '<4sIQI'

    class IndexHeader(NamedTuple):
        ''' A decoded index header '''
        magic: bytes
        type: int
        offset: int
        size: int

    @unique
    class IndexType(IntEnum):
        ''' Enumerates different possible index storage types '''
        CBOR_LZMA_V1 = auto()

    @classmethod
    def _load_tar_index(cls, tar_file_object: BinaryIO) -> TarRecordIndex:
        ''' Loads a tar record index from the end of a tar file object '''

        # Load header
        original_file_position = tar_file_object.tell()
        tar_file_object.seek(-tarfile.BLOCKSIZE, os.SEEK_END)
        header_binary = tar_file_object.read(struct.calcsize(cls.INDEX_HEADER_FORMAT))

        # Decode header
        header = cls.IndexHeader._make(struct.unpack(cls.INDEX_HEADER_FORMAT, header_binary))

        # Check magic bytes
        if header.magic != cls.INDEX_HEADER_MAGIC:
            raise ValueError('IndexedTarStore: invalid index header, can\'t load indexed tar file')

        # Load index based on type
        tar_file_object.seek(header.offset)
        header_binary = tar_file_object.read(header.size)
        tar_file_object.seek(original_file_position)

        match header.type:
            case cls.IndexType.CBOR_LZMA_V1.value:
                logging.debug(f'IndexedTarStore: lzma-compressed index load size={len(header_binary)}')

                # load table (SOA)
                with lzma.open(io.BytesIO(header_binary), 'rb') as lzma_file:
                    table = cbor2.load(lzma_file)
                    items = table['items']
                    offset_datas = table['offset_datas']
                    sizes = table['sizes']
            case _:
                raise TypeError(f"IndexedTarStore: unsupported header type {header.type}")

        # Construct record index from loaded table
        return cls.TarRecordIndex({item: cls.TarRecord(offset_datas[i], sizes[i]) for i, item in enumerate(items)})

    @classmethod
    def _save_tar_index(cls, tar_file_object: BinaryIO, index: TarRecordIndex):
        ''' Saves a tar record index at the end of a tar file object (needs to be finalized / have two empty blocks appended already) '''
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
        items, offset_datas, sizes = list(zip(*sorted(table, key=lambda data: data[1])))

        # Append compressed table to tar file
        with io.BytesIO() as index_buffer:
            # Compress table to in-memory buffer
            with lzma.open(index_buffer, 'wb') as lzma_file:
                cbor2.dump({'items': items, 'offset_datas': offset_datas, 'sizes': sizes}, lzma_file)  # type: ignore

            index_binary = index_buffer.getvalue()
            index_size = len(index_binary)

            logging.debug(f'IndexedTarStore lzma-compressed index store size={index_size}')

            # Append buffer to tar file
            tar_file_object.write(index_binary)

            fill_block()

        # Create index header block
        assert struct.calcsize(
            cls.INDEX_HEADER_FORMAT) <= tarfile.BLOCKSIZE, "Index header larger than single block size"
        header_binary = struct.pack(cls.INDEX_HEADER_FORMAT, cls.INDEX_HEADER_MAGIC, cls.IndexType.CBOR_LZMA_V1.value,
                                    index_offset, index_size)
        header_size = len(header_binary)
        logging.debug(f'IndexedTarStore: header store size={header_size}')

        # Append index header to tar file
        tar_file_object.write(header_binary)
        fill_block()


def consolidate_compressed_metadata(store: zarr.BaseStore, metadata_key=".zmetadata.cbor.xz"):
    """ Consolidate all metadata for groups and arrays within the given store
    into a single compressed cbor resource and put it under the given key.

    See Also
    --------
    zarr.consolidate_metadata
    """
    store = zarr.storage.normalize_store_arg(store, mode="w")

    version = store._store_version

    if version == 2:

        def is_zarr_key(key):
            return (key.endswith('.zarray') or key.endswith('.zgroup') or key.endswith('.zattrs'))

    else:
        raise NotImplementedError("Only supporting V2 stores")

    # Collect all meta-data
    out = {
        'zarr_consolidated_format': 1,
        'metadata': {key: zarr.util.json_loads(store[key])
                     for key in store if is_zarr_key(key)}
    }

    with io.BytesIO() as metadata_buffer:
        # Compress meta-data to in-memory buffer
        with lzma.open(metadata_buffer, 'wb') as lzma_file:
            cbor2.dump(out, lzma_file)  # type: ignore

        store[metadata_key] = metadata_buffer.getvalue()


class ConsolidatedCompressedMetadataStore(zarr.storage.ConsolidatedMetadataStore):
    """ A layer over other storage, where the metadata has been consolidated into a single compressed key.
    """

    # Overwrite constructor to perform decompression of metadata
    def __init__(self, store: zarr.StoreLike, metadata_key=".zmetadata.cbor.xz"):
        self.store = zarr._storage.store.Store._ensure_store(store)

        # retrieve consolidated metadata
        with lzma.open(io.BytesIO(self.store[metadata_key]), 'rb') as lzma_file:
            meta = cbor2.load(lzma_file)

        # check format of consolidated metadata
        consolidated_format = meta.get('zarr_consolidated_format', None)
        if consolidated_format != 1:
            raise zarr.MetadataError('unsupported zarr consolidated metadata format: %s' % consolidated_format)

        # decode metadata
        self.meta_store: zarr.Store = zarr.KVStore(meta["metadata"])


def open_compressed_consolidated(store: zarr.StoreLike, metadata_key=".zmetadata.cbor.xz", mode="r+", **kwargs):
    """ Open group using metadata previously consolidated and compressed into a single key.

    See Also
    --------
    consolidate_compressed_metadata
    zarr.open__consolidated
    """

    # normalize parameters
    zarr_version = kwargs.get('zarr_version')
    store = zarr.storage.normalize_store_arg(store,
                                             storage_options=kwargs.get("storage_options"),
                                             mode=mode,
                                             zarr_version=zarr_version)
    if mode not in {'r', 'r+'}:
        raise ValueError("invalid mode, expected either 'r' or 'r+'; found {!r}".format(mode))

    path = kwargs.pop('path', None)
    if store._store_version == 2:
        ConsolidatedStoreClass = ConsolidatedCompressedMetadataStore
    else:
        raise NotImplementedError("Only supporting V2 stores")

    # setup metadata store
    meta_store = ConsolidatedStoreClass(store, metadata_key=metadata_key)

    # pass through
    chunk_store = kwargs.pop('chunk_store', None) or store
    return zarr.convenience.open(store=meta_store, chunk_store=chunk_store, mode=mode, path=path, **kwargs)
