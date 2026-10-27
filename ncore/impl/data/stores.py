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

"""Stores of NCore data: zarr hierarchies in indexed tar archives (:class:`IndexedTarZarrStore`), or in other
zarr-python stores such as directories (:class:`ZarrStoreAdapter`), with compressed consolidated metadata"""

from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Literal, Optional, Union

from typing_extensions import Self, override
from upath import UPath

from ncore.impl.data.backend import (
    V2_CONSOLIDATED_METADATA_KEY,
    V3_CONSOLIDATED_METADATA_KEY,
    Attributes,
    ZarrFormat,
    ZarrStore,
)
from ncore.impl.data.itar import IndexedTarArchive
from ncore.impl.data.nodes import Group, _backend, check_zarr_format


def is_consolidated_metadata_key(key: str) -> bool:
    """True for keys of consolidated metadata records (ncore's compressed consolidated metadata of both zarr formats,
    and zarr-python 2's ``.zmetadata``), which :meth:`Store.consolidate` regenerates (e.g., to skip when copying the
    records of stores)"""
    return key in (V2_CONSOLIDATED_METADATA_KEY, V3_CONSOLIDATED_METADATA_KEY, ".zmetadata")


class Store(ABC):
    """A store of a zarr hierarchy.

    Stores wrap a zarr-python store of the installed zarr-python version (:attr:`zarr`), and are context managers
    (closing them on exit). Other zarr-python stores (e.g., in-memory stores) can be used via :class:`ZarrStoreAdapter`.
    """

    def __init__(self, zarr_store: ZarrStore) -> None:
        # (also validated at runtime: e.g. stores of other zarr-python versions)
        self._zarr_store: ZarrStore = _backend.zarr_store(zarr_store)

    @property
    def zarr(self) -> ZarrStore:
        """The wrapped zarr-python store (of the installed zarr-python version)"""
        return self._zarr_store

    def create_root_group(self, attributes: Attributes, *, zarr_format: ZarrFormat = 2) -> Group:
        """Creates the root group of the hierarchy of the store, in zarr format v2 (default, readable with all
        zarr-python versions) or v3 (requires zarr-python 3 for writing and reading)"""
        return Group(
            _backend.create_root_group(self._zarr_store, check_zarr_format(zarr_format), attributes), store=self
        )

    def open_root_group(self, consolidated: bool = True) -> Group:
        """Opens the root group of the hierarchy of the store for reading: from its consolidated metadata record if
        consolidated (raises KeyError if the store has none, see :meth:`consolidate`), otherwise from its node metadata
        records"""
        return Group(_backend.open_root_group(self._zarr_store, consolidated), store=self)

    def consolidate(self) -> None:
        """Stores the consolidated metadata of the hierarchy of the store (after writing all nodes): a single compressed
        record with the node metadata of all members (zarr format v2: ``.zmetadata.cbor.xz``, v3:
        ``zarr.json.cbor.zlib``), from which :meth:`open_root_group` opens the hierarchy without reading node metadata
        records. Node metadata records are complete as well (e.g., for zarr-python, which opens such hierarchies as
        unconsolidated)."""
        _backend.consolidate(self._zarr_store)

    def reload_resources(self) -> None:
        """Re-initializes the resources of the store (e.g., file objects in forked processes), no-op by default"""

    def close(self) -> None:
        """Closes the store (finishes writing, no-op by default)"""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class IndexedTarZarrStore(Store):
    """A zarr hierarchy in an indexed tar archive (local or remote, see :class:`IndexedTarArchive` for the
    parameters).

    The constructor opens the archive synchronously (blocking), see :meth:`open_async` for opening stores for reading
    asynchronously."""

    def __init__(
        self,
        itar_path: Union[str, Path, UPath],
        mode: Literal["r", "w"] = "r",
        index_tail_read_size: Optional[int] = 1 << 20,
        remote_range_reads: bool = False,
    ) -> None:
        """Opens the store synchronously (blocking, see :class:`IndexedTarArchive`), see :meth:`open_async` for opening
        stores for reading asynchronously"""
        self._init(
            IndexedTarArchive(
                itar_path, mode=mode, index_tail_read_size=index_tail_read_size, remote_range_reads=remote_range_reads
            )
        )

    def _init(self, archive: IndexedTarArchive) -> None:
        self._archive = archive
        super().__init__(_backend.indexed_tar_store(self._archive))

    @classmethod
    async def open_async(
        cls,
        itar_path: Union[str, Path, UPath],
        index_tail_read_size: Optional[int] = 1 << 20,
        remote_range_reads: bool = False,
    ) -> IndexedTarZarrStore:
        """Opens a store for reading asynchronously, see :meth:`IndexedTarArchive.open_async` (the consolidated
        metadata record, read when opening the hierarchy, is fetched asynchronously as well)"""
        archive = await IndexedTarArchive.open_async(
            itar_path,
            index_tail_read_size=index_tail_read_size,
            remote_range_reads=remote_range_reads,
            # (read when opening the hierarchy of the store: fetched with the index, so it's read without blocking)
            prefetch_keys=(V3_CONSOLIDATED_METADATA_KEY, V2_CONSOLIDATED_METADATA_KEY),
        )
        store = cls.__new__(cls)
        store._init(archive)
        return store

    @property
    def archive(self) -> IndexedTarArchive:
        """The records of the archive"""
        return self._archive

    @override
    def reload_resources(self) -> None:
        self._archive.reload_resources()

    @override
    def close(self) -> None:
        self._archive.close()

    def __repr__(self) -> str:
        return f"IndexedTarZarrStore({self._archive.itar_upath!r}, mode={self._archive.mode!r})"


class ZarrStoreAdapter(Store):
    """A zarr hierarchy in any other zarr-python store of the installed zarr-python version (e.g., an in-memory store,
    or a local directory, see :meth:`directory`)"""

    @classmethod
    def directory(cls, path: Union[str, Path, UPath]) -> ZarrStoreAdapter:
        """A zarr hierarchy in a local directory (a directory per group / array, a file per record, e.g., for
        debugging). Existing files of the directory are kept when writing."""
        upath = UPath(path)
        if upath.protocol not in ("", "file", "local"):
            raise ValueError(f"Directory stores need to be local directories, got {upath}")
        return cls(_backend.directory_store(upath))

    def __repr__(self) -> str:
        return f"ZarrStoreAdapter({self._zarr_store!r})"
