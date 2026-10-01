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

"""zarr stores used by ncore, supporting both zarr-python 2 and zarr-python 3.

The implementation of the installed zarr-python version is selected at import time (see :mod:`.nodes`):

* zarr-python 2: reads / writes zarr format v2 hierarchies.
* zarr-python 3: reads / writes zarr format v2 and v3 hierarchies.

zarr format v2 data is identical on disk independent of the zarr-python version that wrote it (including the
compressed consolidated metadata of ``.itar`` archives), so it's readable by ncore V4 readers of all versions
(ncore <= 19.8 support zarr-python 2 / zarr format v2 only).

Stores are :class:`IndexedTarStore` instances (``.itar`` archives, independent of zarr-python), or stores of the
installed zarr-python version (e.g., directory stores of :func:`open_directory_store`). Groups are
:class:`ncore.impl.data.nodes.Group` instances.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Union

from upath import UPath

from ncore.impl.data._itar import V2_CONSOLIDATED_METADATA_KEY, V3_CONSOLIDATED_METADATA_KEY, IndexedTarStore
from ncore.impl.data._zarr_api import Attributes, StoreLike, ZarrStore
from ncore.impl.data.nodes import ZARR_PYTHON_3, Group, _backend, check_zarr_format


def open_directory_store(path: Union[str, Path, UPath], mode: Literal["r", "w"]) -> ZarrStore:
    """Opens a directory store for reading or (truncating) writing"""
    return _backend.open_directory_store(str(path), mode)


def create_root_group(store: StoreLike, zarr_format: int, attributes: Attributes) -> Group:
    """Creates the root group of a new store"""
    return Group(_backend.create_root_group(store, check_zarr_format(zarr_format), attributes))


def open_store(store: StoreLike, open_consolidated: bool) -> Group:
    """Opens the root group of a store read-only (raises KeyError if consolidated metadata is requested but missing)"""
    return Group(_backend.open_store(store, open_consolidated))


def open_compressed_consolidated(
    store: StoreLike, metadata_key: str = V2_CONSOLIDATED_METADATA_KEY, mode: Literal["r", "r+"] = "r+"
) -> Group:
    """Opens a zarr format v2 group using metadata previously consolidated and compressed into a single key (raises
    KeyError if the compressed consolidated metadata is missing)"""
    return Group(_backend.open_compressed_consolidated(store, metadata_key, mode))


def consolidate_store(store: StoreLike) -> None:
    """Consolidates the metadata of a store (compressed for ``.itar`` archives)"""
    _backend.consolidate_store(store)


def consolidate_compressed_metadata(store: StoreLike, metadata_key: str = V2_CONSOLIDATED_METADATA_KEY) -> None:
    """Consolidates all zarr format v2 metadata of a store into a single compressed record under the given key"""
    _backend.consolidate_compressed_metadata(store, metadata_key)


def close_store(store: StoreLike) -> None:
    """Closes a store"""
    _backend.close_store(store)


def get_group_store(group: Group) -> StoreLike:
    """Returns the store of a group (:class:`IndexedTarStore` for groups of ``.itar`` archives)"""
    return _backend.group_store(group.zarr)


__all__ = [
    "IndexedTarStore",
    "StoreLike",
    "ZarrStore",
    "consolidate_compressed_metadata",
    "open_compressed_consolidated",
    "open_directory_store",
    "create_root_group",
    "consolidate_store",
    "open_store",
    "close_store",
    "get_group_store",
    "V2_CONSOLIDATED_METADATA_KEY",
    "V3_CONSOLIDATED_METADATA_KEY",
    "ZARR_PYTHON_3",
]
