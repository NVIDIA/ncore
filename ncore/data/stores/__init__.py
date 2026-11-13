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

"""Package exposing NCore's storage APIs: indexed tar archives, the zarr hierarchies stored in them (or in other
zarr-python stores), and typed access to their groups and arrays"""

from ncore.impl.data.backend import BloscCompression
from ncore.impl.data.itar import IndexedTarArchive, Record, RecordExistsError
from ncore.impl.data.nodes import Array, EncodedArray, Group, ZarrFormat
from ncore.impl.data.stores import IndexedTarZarrStore, Store, ZarrStoreAdapter


__all__ = [
    # indexed tar archives
    "IndexedTarArchive",
    "Record",
    "RecordExistsError",
    # zarr hierarchy stores
    "Store",
    "IndexedTarZarrStore",
    "ZarrStoreAdapter",
    # groups / arrays of zarr hierarchies
    "Group",
    "Array",
    "ZarrFormat",
    # compression of arrays
    "BloscCompression",
    # arrays encoded off the writing thread
    "EncodedArray",
]
