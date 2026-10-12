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

"""Stores of NCore data: zarr hierarchies in indexed tar archives (:class:`IndexedTarStore`) or directories, with
compressed consolidated metadata"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from upath import UPath

from ncore.impl.data._backend import (
    V2_CONSOLIDATED_METADATA_KEY,
    V3_CONSOLIDATED_METADATA_KEY,
    Attributes,
    StoreLike,
    ZarrStore,
)
from ncore.impl.data.itar import IndexedTarStore
from ncore.impl.data.nodes import Group, _backend, check_zarr_format


def open_directory_store(path: Union[str, Path, UPath]) -> ZarrStore:
    """Opens a directory store (a directory per group / array, a file per record) for reading or writing"""
    return _backend.open_directory_store(str(path))


def create_root_group(store: StoreLike, attributes: Attributes, *, zarr_format: int = 2) -> Group:
    """Creates the root group of a hierarchy in a store, in zarr format v2 (default, readable with all zarr-python
    versions) or v3 (requires zarr-python 3 for writing and reading)"""
    return Group(_backend.create_root_group(store, check_zarr_format(zarr_format), attributes))


def open_store(store: StoreLike, consolidated: bool = True) -> Group:
    """Opens the root group of the hierarchy of a store for reading, via its compressed consolidated metadata if
    consolidated (raises KeyError if the store has no compressed consolidated metadata)"""
    return Group(_backend.open_store(store, consolidated))


def consolidate_store(store: StoreLike) -> None:
    """Stores the compressed consolidated metadata of the hierarchy of a store (after writing all nodes)"""
    _backend.consolidate_store(store)


def close_store(store: StoreLike) -> None:
    """Closes a store (required for indexed tar stores opened for writing)"""
    if isinstance(store, IndexedTarStore):
        store.close()


def group_store(group: Group) -> StoreLike:
    """The store of a group (the :class:`IndexedTarStore` of groups of indexed tar archives)"""
    return _backend.group_store(group.zarr)


__all__ = [
    "IndexedTarStore",
    "StoreLike",
    "V2_CONSOLIDATED_METADATA_KEY",
    "V3_CONSOLIDATED_METADATA_KEY",
    "ZarrStore",
    "close_store",
    "consolidate_store",
    "create_root_group",
    "group_store",
    "open_directory_store",
    "open_store",
]
