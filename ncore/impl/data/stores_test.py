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

import itertools
import tempfile
import unittest

from pathlib import Path
from typing import IO, Any, Optional

import numpy as np
import parameterized
import zarr

from .stores import IndexedTarStore, consolidate_compressed_metadata, open_compressed_consolidated


COMPRESSED_CONSOLIDATED_VALUES = [False, True]
INDEX_TAIL_READ_SIZES = [
    None,
    1 << 10,
    1 << 20,
    1,
    512,
    513,
]  # None (explicit separate reads), 0.5 MiB, 1 MiB (default), 1 byte / 512 / 513 bytes (edge cases resulting in separate reads)


class TestIndexedTarStore(unittest.TestCase):
    """Test to verify functionality of IndexedTarStore"""

    def setUp(self):
        # Fill a reference group with an in-memory store
        self.g_ref = zarr.open(store=zarr.MemoryStore())
        self.g_ref.create_dataset("foo", data=np.random.rand(3, 3, 3))
        self.g_ref.attrs.update({"some": "thing"})
        self.g_ref.require_group("subgroup").create_dataset("foo", data=np.random.rand(5, 5, 5))

    def check_with_reference(self, group):
        """Verifies all values of a group against the reference"""
        self.assertIsNone(np.testing.assert_array_equal(self.g_ref["foo"][()], group["foo"][()]))
        self.assertIsNone(
            np.testing.assert_array_equal(self.g_ref["subgroup"]["foo"][()], group["subgroup"]["foo"][()])
        )
        self.assertDictEqual(self.g_ref.attrs.asdict(), group.attrs.asdict())

    @parameterized.parameterized.expand(
        itertools.product(
            INDEX_TAIL_READ_SIZES,
        )
    )
    def test_reserialization(self, index_tail_read_size: Optional[int]):
        """Make sure storing / loading of regular zarr data to .itar files works correctly"""

        # re-serialize to .itar archive
        with tempfile.NamedTemporaryFile(suffix=".itar") as f:
            with IndexedTarStore(f.name, mode="w") as s_itar_out:  # closes file on exit
                zarr.copy_store(self.g_ref.store, s_itar_out)

            # reload store from file
            store = IndexedTarStore(f.name, index_tail_read_size=index_tail_read_size)
            g_reload = zarr.open(store=store, mode="r")

            # check all data was correctly serialized / deserialized
            self.check_with_reference(g_reload)

            # check reloading resources is functional
            store.reload_resources()
            self.check_with_reference(g_reload)

    @parameterized.parameterized.expand(
        itertools.product(
            INDEX_TAIL_READ_SIZES,
        )
    )
    def test_compressed_consolidated(self, index_tail_read_size: Optional[int]):
        """Make sure compressed consolidated meta data is stored/loaded correctly"""

        # serialize to .itar archive (will also serialize compressed-consolidated meta-data)
        with tempfile.NamedTemporaryFile(suffix=".itar") as f:
            with IndexedTarStore(f.name, mode="w") as s_itar_out:  # closes file on exit
                zarr.copy_store(self.g_ref.store, s_itar_out)

                # consolidate compress meta-data
                consolidate_compressed_metadata(s_itar_out)

            # reload store from file with compressed consolidated meta-data
            store = IndexedTarStore(f.name, index_tail_read_size=index_tail_read_size)
            g_reload = open_compressed_consolidated(store=store, mode="r")

            # check all data was correctly serialized / deserialized
            self.check_with_reference(g_reload)

            # check reloading resources is functional
            store.reload_resources()
            self.check_with_reference(g_reload)

    @parameterized.parameterized.expand(
        itertools.product(
            COMPRESSED_CONSOLIDATED_VALUES,
            INDEX_TAIL_READ_SIZES,
        )
    )
    def test_empty(self, compressed_consolidate: bool, index_tail_read_size: Optional[int]):
        """Verify edge case of serialization of empty store is possible without errors"""
        with tempfile.NamedTemporaryFile(suffix=".itar") as f:
            with IndexedTarStore(f.name, mode="w") as s_itar_out:  # closes file on exit
                # Don't write any zarr data (still serializes empty tar / seek tables)

                if compressed_consolidate:
                    consolidate_compressed_metadata(s_itar_out)

            with IndexedTarStore(f.name, index_tail_read_size=index_tail_read_size) as s_itar_in:
                # Loading store should work without errors

                # But loading a non-existing group should then fail
                with self.assertRaises(zarr.errors.PathNotFoundError):
                    if compressed_consolidate:
                        open_compressed_consolidated(s_itar_in, mode="r")
                    else:
                        zarr.open(store=s_itar_in, mode="r")

    def test_index_tail_cache_avoids_extra_file_reads(self):
        """Payloads fully covered by the index tail read must not trigger further ``read()`` calls."""

        class _CountingReader(IO[Any]):
            __slots__ = ("_f", "read_calls")

            def __init__(self, f: IO[Any]):
                self._f = f
                self.read_calls = 0

            def read(self, n=-1):
                self.read_calls += 1
                return self._f.read(n)

            def seek(self, *args, **kwargs):
                return self._f.seek(*args, **kwargs)

            def tell(self):
                return self._f.tell()

            def close(self):
                return self._f.close()

            def __getattr__(self, name):
                return getattr(self._f, name)

        with tempfile.NamedTemporaryFile(suffix=".itar") as f:
            g_ref = zarr.open(store=zarr.MemoryStore())
            g_ref.create_dataset("x", data=np.array([1.0, 2.0]))
            with IndexedTarStore(f.name, mode="w") as s_out:
                zarr.copy_store(g_ref.store, s_out)

            file_size = Path(f.name).stat().st_size
            self.assertGreater(file_size, 0)
            store = IndexedTarStore(f.name, index_tail_read_size=file_size)

            # Wrap the file object to intercept read counts
            inner = store.tar_file_object
            counter = _CountingReader(inner)
            store.tar_file_object = counter
            self.assertEqual(counter.read_calls, 0)

            # Read every key via the tail buffer and verify no extra file reads occurred
            values = {key: store[key] for key in store}
            self.assertEqual(counter.read_calls, 0)

            # Verify actual data was served from the store
            self.assertGreater(len(values), 0)
            g_reload = zarr.open(store=store, mode="r")
            self.assertIsNone(np.testing.assert_array_equal(g_ref["x"][()], g_reload["x"][()]))
