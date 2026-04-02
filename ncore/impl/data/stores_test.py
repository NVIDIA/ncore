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

import tempfile
import unittest

import numpy as np
import parameterized
import zarr

from .stores import IndexedTarStore, open_store, lz4_codecs


def create_array(group: zarr.Group, name: str, data: np.ndarray, attributes: dict | None = None) -> zarr.Array:
    """Helper function to create an array in a group with given name and data."""
    return group.create_array(name=name, data=data, attributes=attributes or {}, compressors=lz4_codecs())


def copy_group(source: zarr.Group, target: zarr.Group) -> None:
    """Helper function to recursively copy a zarr group to another group."""
    new_group = target.create_group(source.basename, attributes=source.attrs.asdict())

    for key in source.keys():
        child = source[key]
        if isinstance(child, zarr.Array):
            create_array(new_group, key, child[()], child.attrs.asdict())
        elif isinstance(child, zarr.Group):
            copy_group(child, new_group)
        else:
            raise TypeError(f"Unsupported type for key {key}: {type(child)}")


def copy_store(source_store: zarr.abc.store.Store, target_store: zarr.abc.store.Store) -> None:
    """Helper function to copy a zarr store to another store.

    zarr3 removed ``zarr.copy_store()``, so this is a group-level replacement.
    """
    source = zarr.open(store=source_store, mode="r")
    target = zarr.create_group(store=target_store, zarr_format=3, attributes=source.attrs.asdict(), overwrite=True)

    for key in source.keys():
        child = source[key]
        if isinstance(child, zarr.Array):
            create_array(target, key, child[()], child.attrs.asdict())
        elif isinstance(child, zarr.Group):
            copy_group(child, target)
        else:
            raise TypeError(f"Unsupported type for key {key}: {type(child)}")


class TestIndexedTarStore(unittest.TestCase):
    """Test to verify functionality of IndexedTarStore"""

    def setUp(self):
        # Fill a reference group with an in-memory store
        self.g_ref: zarr.Group = zarr.open_group(
            store=zarr.storage.MemoryStore(), mode="w", zarr_format=3, attributes={"some": "thing"}
        )
        create_array(self.g_ref, "foo", np.random.rand(3, 3, 3))
        sub = self.g_ref.require_group("subgroup")
        create_array(sub, "foo", np.random.rand(5, 5, 5), {"some": "other thing"})

    def check_with_reference(self, group: zarr.Group) -> None:
        """Verifies all values of a group against the reference"""
        self.assertIsNone(np.testing.assert_array_equal(self.g_ref["foo"][()], group["foo"][()]))
        self.assertIsNone(
            np.testing.assert_array_equal(self.g_ref["subgroup"]["foo"][()], group["subgroup"]["foo"][()])
        )
        self.assertDictEqual(self.g_ref.attrs.asdict(), group.attrs.asdict())
        self.assertDictEqual(self.g_ref["foo"].attrs.asdict(), group["foo"].attrs.asdict())
        self.assertDictEqual(self.g_ref["subgroup"].attrs.asdict(), group["subgroup"].attrs.asdict())
        self.assertDictEqual(self.g_ref["subgroup"]["foo"].attrs.asdict(), group["subgroup"]["foo"].attrs.asdict())

    def test_reserialization(self):
        """Make sure storing / loading of regular zarr data to .itar files works correctly"""

        # re-serialize to .itar archive
        with tempfile.NamedTemporaryFile(suffix=".itar") as f:
            with IndexedTarStore(f.name, mode="w") as s_itar_out:  # closes file on exit
                copy_store(self.g_ref.store, s_itar_out)

            # reload store from file
            store = IndexedTarStore(f.name)
            g_reload = zarr.open(store=store, mode="r")

            # check all data was correctly serialized / deserialized
            self.check_with_reference(g_reload)

            # check reloading resources is functional
            store.reload_resources()
            self.check_with_reference(g_reload)

    @parameterized.parameterized.expand(
        [
            False,
            True,
        ]
    )
    def test_consolidated(self, open_consolidated: bool):
        """Make sure consolidated meta data is stored/loaded correctly"""

        # serialize to .itar archive (will also serialize consolidated meta-data)
        with tempfile.NamedTemporaryFile(suffix=".itar") as f:
            with IndexedTarStore(f.name, mode="w") as s_itar_out:  # closes file on exit
                copy_store(self.g_ref.store, s_itar_out)

                # consolidate meta-data (triggers the store's zarr.json -> zarr.cbor.xz intercept)
                zarr.consolidate_metadata(s_itar_out)

            # reload store from file with consolidated meta-data
            store = IndexedTarStore(f.name)
            g_reload = open_store(store=store, open_consolidated=open_consolidated, mode="r")

            # check all data was correctly serialized / deserialized
            self.check_with_reference(g_reload)

            # check reloading resources is functional
            store.reload_resources()
            self.check_with_reference(g_reload)


if __name__ == "__main__":
    unittest.main()
