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

import asyncio
import concurrent.futures
import gc
import importlib
import itertools
import multiprocessing
import os
import sys
import tarfile
import tempfile
import unittest
import warnings

from pathlib import Path
from typing import Any, Optional
from unittest import mock

import numpy as np
import parameterized
import zarr
import zarr.errors

from ncore.impl.data import nodes
from ncore.impl.data._itar import _TarArchive, decompress_document
from ncore.impl.data.stores import (
    V2_CONSOLIDATED_METADATA_KEY,
    V3_CONSOLIDATED_METADATA_KEY,
    ZARR_PYTHON_3,
    IndexedTarStore,
    consolidate_compressed_metadata,
    consolidate_store,
    create_root_group,
    get_group_store,
    open_compressed_consolidated,
    open_directory_store,
    open_store,
)


COMPRESSED_CONSOLIDATED_VALUES = [False, True]
INDEX_TAIL_READ_SIZES = [
    None,
    1 << 10,
    1 << 20,
    1,
    512,
    513,
]  # None (explicit separate reads), 0.5 MiB, 1 MiB (default), 1 byte / 512 / 513 bytes (edge cases resulting in separate reads)
ZARR_FORMATS = list(nodes.SUPPORTED_ZARR_FORMATS)


_ROOT_ATTRIBUTES = {"some": "thing"}


def _fill_reference(root: nodes.Group) -> None:
    """Fills a reference hierarchy below a root group created with `_ROOT_ATTRIBUTES` (format-independent)"""
    rng = np.random.default_rng(0)
    root.create_array("foo", rng.random((3, 3, 3)))
    sub = root.create_group("subgroup", attributes={"other": [1, 2, 3]})
    sub.create_array("foo", rng.random((5, 5, 5)), chunks=(2, 5, 5), attributes={"unit": "m"})
    sub.create_array("empty", np.zeros((0, 3), dtype=np.float32))
    sub.create_bytes_array("blob", b"\x89PNG\x00some binary data", attributes={"format": "png"})


def _check_reference(test: unittest.TestCase, group: nodes.Group) -> None:
    """Verifies the contents of a reference hierarchy"""
    rng = np.random.default_rng(0)
    np.testing.assert_array_equal(rng.random((3, 3, 3)), group.array("foo").read())
    np.testing.assert_array_equal(rng.random((5, 5, 5)), group.array("subgroup/foo").read())
    test.assertEqual(group.array("subgroup/empty").read().shape, (0, 3))
    test.assertEqual(group.array("subgroup/blob").read_bytes(), b"\x89PNG\x00some binary data")
    test.assertDictEqual(group.attrs, {"some": "thing"})
    test.assertDictEqual(group.group("subgroup").attrs, {"other": [1, 2, 3]})
    test.assertDictEqual(group.array("subgroup/foo").attrs, {"unit": "m"})
    test.assertDictEqual(group.array("subgroup/blob").attrs, {"format": "png"})
    test.assertEqual(sorted(group.members()), ["foo", "subgroup"])
    test.assertEqual(sorted(group.group("subgroup").members()), ["blob", "empty", "foo"])
    test.assertIn("blob", group.group("subgroup"))
    test.assertNotIn("missing", group.group("subgroup"))
    test.assertEqual([name for name, _ in group.groups()], ["subgroup"])
    with test.assertRaises(KeyError):
        group.group("foo")
    with test.assertRaises(KeyError):
        group.array("subgroup")


def _write_itar(path: str, zarr_format: int, consolidate: bool = True) -> None:
    with IndexedTarStore(path, mode="w") as store:
        root = create_root_group(store, zarr_format, _ROOT_ATTRIBUTES)
        _fill_reference(root)
        if consolidate:
            consolidate_store(store)


class TestIndexedTarStore(unittest.TestCase):
    """Test to verify functionality of IndexedTarStore"""

    def setUp(self):
        # don't tolerate deprecated zarr API usage
        warnings.simplefilter("error", DeprecationWarning)
        if (zarr_deprecation_warning := getattr(zarr.errors, "ZarrDeprecationWarning", None)) is not None:
            warnings.simplefilter("error", zarr_deprecation_warning)

    def tearDown(self):
        warnings.resetwarnings()

    @parameterized.parameterized.expand(
        itertools.product(ZARR_FORMATS, COMPRESSED_CONSOLIDATED_VALUES, INDEX_TAIL_READ_SIZES)
    )
    def test_reserialization(self, zarr_format: int, open_consolidated: bool, index_tail_read_size: Optional[int]):
        """Make sure storing / loading of zarr data to .itar files works correctly"""

        with tempfile.NamedTemporaryFile(suffix=".itar") as f:
            _write_itar(f.name, zarr_format)

            # reload store from file
            store = IndexedTarStore(f.name, index_tail_read_size=index_tail_read_size)
            g_reload = open_store(store, open_consolidated)

            # check all data was correctly serialized / deserialized
            _check_reference(self, g_reload)

            # check reloading resources is functional
            store.reload_resources()
            _check_reference(self, g_reload)

    @parameterized.parameterized.expand(itertools.product(ZARR_FORMATS, COMPRESSED_CONSOLIDATED_VALUES))
    def test_directory_store(self, zarr_format: int, open_consolidated: bool):
        """Make sure storing / loading of zarr data to directory stores works correctly"""

        with tempfile.TemporaryDirectory(suffix=".zarr") as d:
            store = open_directory_store(d, mode="w")
            root = create_root_group(store, zarr_format, _ROOT_ATTRIBUTES)
            _fill_reference(root)
            consolidate_store(store)

            _check_reference(self, open_store(open_directory_store(d, mode="r"), open_consolidated))

    @parameterized.parameterized.expand(itertools.product(INDEX_TAIL_READ_SIZES))
    def test_compressed_consolidated_v2(self, index_tail_read_size: Optional[int]):
        """Make sure the compressed consolidated meta data API of ncore <= 19.8 is functional (zarr format v2)"""

        with tempfile.NamedTemporaryFile(suffix=".itar") as f:
            with IndexedTarStore(f.name, mode="w") as s_itar_out:
                _fill_reference(create_root_group(s_itar_out, 2, _ROOT_ATTRIBUTES))
                consolidate_compressed_metadata(s_itar_out)

            store = IndexedTarStore(f.name, index_tail_read_size=index_tail_read_size)
            g_reload = open_compressed_consolidated(store=store, mode="r")
            _check_reference(self, g_reload)

            store.reload_resources()
            _check_reference(self, g_reload)

    @parameterized.parameterized.expand(itertools.product(INDEX_TAIL_READ_SIZES))
    @unittest.skipUnless(3 in ZARR_FORMATS, "zarr format v3 requires zarr-python 3")
    def test_compressed_consolidated_v3(self, index_tail_read_size: Optional[int]):
        """zarr format v3 archives store the consolidated metadata in the compressed root metadata only, and node
        metadata is served from it without reading node metadata records"""

        with tempfile.NamedTemporaryFile(suffix=".itar") as f:
            _write_itar(f.name, 3)

            store = IndexedTarStore(f.name, index_tail_read_size=index_tail_read_size)
            self.assertEqual(store.zarr_format, 3)
            self.assertIsNotNone(decompress_document(store[V3_CONSOLIDATED_METADATA_KEY]).get("consolidated_metadata"))

            read_keys = []
            read = store._archive.read
            with mock.patch.object(
                store._archive, "read", side_effect=lambda key, *args: read_keys.append(key) or read(key, *args)
            ):
                g_reload = open_store(store, True)
                _check_reference(self, g_reload)
            self.assertFalse([key for key in read_keys if key.endswith("zarr.json")], read_keys)

            store.reload_resources()
            _check_reference(self, g_reload)

    @parameterized.parameterized.expand(itertools.product(ZARR_FORMATS))
    def test_records_are_write_once(self, zarr_format: int):
        """Chunk records can't be rewritten, node metadata updates are deferred and written once (final version)"""

        with tempfile.NamedTemporaryFile(suffix=".itar") as f:
            with IndexedTarStore(f.name, mode="w") as store:
                root = create_root_group(store, zarr_format, {"version": 1})
                group = root.create_group("g", attributes={"a": 1})
                group.create_array("x", np.arange(3))
                chunk_key = next(key for key in store.keys() if key.startswith("g/x/") and "zarr" not in key)
                with self.assertRaises(ValueError):
                    store[chunk_key] = b"rewritten"

                if ZARR_PYTHON_3:
                    # zarr-python 3 rewrites node metadata on every update (deferred until closing the store)
                    for version in range(2, 5):
                        root.update_attrs({"version": version})
                        group.update_attrs({"a": version})
                else:
                    # zarr-python 2 writes node metadata records immediately (layout of ncore <= 19.8)
                    with self.assertRaises(ValueError):
                        root.update_attrs({"version": 2})
                consolidate_store(store)

            with tarfile.open(f.name) as tar:
                names = [m.name for m in tar.getmembers()]
            self.assertEqual(len(names), len(set(names)))

            reloaded = open_store(IndexedTarStore(f.name), True)
            self.assertEqual(
                (reloaded.attrs["version"], reloaded.group("g").attrs["a"]), (4, 4) if ZARR_PYTHON_3 else (1, 1)
            )

    @parameterized.parameterized.expand(itertools.product(ZARR_FORMATS))
    def test_on_disk_layout(self, zarr_format: int):
        """Verify the on-disk layout of compressed consolidated metadata and write-once records"""

        with tempfile.NamedTemporaryFile(suffix=".itar") as f:
            _write_itar(f.name, zarr_format)

            with tarfile.open(f.name) as tar:
                names = [m.name for m in tar.getmembers()]

            # every record is written exactly once (no dead space from metadata rewrites)
            self.assertEqual(len(names), len(set(names)))

            if zarr_format == 2:
                self.assertIn(V2_CONSOLIDATED_METADATA_KEY, names)
                self.assertIn(".zgroup", names)
                self.assertIn("subgroup/foo/.zarray", names)
                self.assertNotIn(".zmetadata", names)
            else:
                self.assertIn(V3_CONSOLIDATED_METADATA_KEY, names)
                self.assertNotIn("zarr.json", names)  # root metadata only stored compressed
                self.assertIn("subgroup/foo/zarr.json", names)

    @parameterized.parameterized.expand(itertools.product(COMPRESSED_CONSOLIDATED_VALUES, INDEX_TAIL_READ_SIZES))
    def test_empty(self, compressed_consolidate: bool, index_tail_read_size: Optional[int]):
        """Verify edge case of serialization of empty store is possible without errors"""
        with tempfile.NamedTemporaryFile(suffix=".itar") as f:
            with IndexedTarStore(f.name, mode="w") as s_itar_out:  # closes file on exit
                # Don't write any zarr data (still serializes empty tar / seek tables)

                if compressed_consolidate:
                    consolidate_compressed_metadata(s_itar_out)

            with IndexedTarStore(f.name, index_tail_read_size=index_tail_read_size) as s_itar_in:
                # Loading store should work without errors, but loading a non-existing group should then fail
                with self.assertRaises(Exception):
                    open_store(s_itar_in, compressed_consolidate)

    def test_missing_consolidated_metadata_raises_key_error(self):
        """Opening without consolidated metadata raises a KeyError (relied on by downstream fallbacks)"""
        with tempfile.NamedTemporaryFile(suffix=".itar") as f:
            _write_itar(f.name, 2, consolidate=False)

            with self.assertRaises(KeyError):
                open_compressed_consolidated(store=IndexedTarStore(f.name), mode="r")
            with self.assertRaises(KeyError):
                open_store(IndexedTarStore(f.name), True)

            # non-consolidated access still works
            _check_reference(self, open_store(IndexedTarStore(f.name), False))

    def test_raw_mapping_interface(self):
        """Raw byte-level mapping access (used for, e.g., merging archives)"""
        with tempfile.NamedTemporaryFile(suffix=".itar") as f_in, tempfile.NamedTemporaryFile(suffix=".itar") as f_out:
            _write_itar(f_in.name, 2, consolidate=False)

            # copy all raw records into a new archive and consolidate
            store_in = IndexedTarStore(f_in.name)
            with IndexedTarStore(f_out.name, mode="w") as store_out:
                for key in store_in.keys():
                    if key.startswith(".zmetadata"):
                        continue
                    if key not in store_out:
                        store_out[key] = store_in[key]
                self.assertEqual(len(store_out), len([k for k in store_in if not k.startswith(".zmetadata")]))
                consolidate_compressed_metadata(store_out)

            _check_reference(self, open_compressed_consolidated(store=IndexedTarStore(f_out.name), mode="r"))

            with self.assertRaises(KeyError):
                store_in["missing"]

    def test_reload_resources_preserves_write_mode_archive(self):
        """Verify writer resource reload does not truncate already-written archive data."""
        with tempfile.NamedTemporaryFile(suffix=".itar") as f:
            with IndexedTarStore(f.name, mode="w") as store:
                store["first"] = b"first payload"
                store.tar_file_object.flush()

                size_before_reload = Path(f.name).stat().st_size
                self.assertGreater(size_before_reload, 0)

                store.reload_resources()

                self.assertEqual(Path(f.name).stat().st_size, size_before_reload)
                self.assertEqual(store["first"], b"first payload")

                store["second"] = b"second payload"
                self.assertEqual(store["first"], b"first payload")
                self.assertEqual(store["second"], b"second payload")

            with IndexedTarStore(f.name, mode="r") as store:
                self.assertEqual(store["first"], b"first payload")
                self.assertEqual(store["second"], b"second payload")

    def test_index_tail_cache_avoids_extra_file_reads(self):
        """Payloads fully covered by the index tail read must not trigger further file reads."""

        with tempfile.NamedTemporaryFile(suffix=".itar") as f:
            _write_itar(f.name, 2)

            file_size = Path(f.name).stat().st_size
            self.assertGreater(file_size, 0)
            archive = _TarArchive(f.name, index_tail_read_size=file_size)

            # Read every record and verify no file reads occurred (neither positional nor via the file object)
            with mock.patch("os.pread") as pread, mock.patch.object(archive, "tar_file_object") as file_object:
                values = {key: archive.read(key) for key in archive.keys()}
                pread.assert_not_called()
                file_object.read.assert_not_called()

            self.assertGreater(len(values), 0)
            self.assertEqual(values, {key: _TarArchive(f.name, index_tail_read_size=None).read(key) for key in values})

    def test_positional_reads(self):
        """Local archives are read via (thread-safe) positional reads, also from multiple threads"""
        with tempfile.NamedTemporaryFile(suffix=".itar") as f:
            payloads = {f"key{i}": os.urandom(1000 + i) for i in range(64)}
            with _TarArchive(f.name, mode="w") as archive:
                for key, value in payloads.items():
                    archive.write(key, value)

            archive = _TarArchive(f.name, index_tail_read_size=None)
            with concurrent.futures.ThreadPoolExecutor(8) as executor:
                results = dict(zip(payloads, executor.map(archive.read, payloads)))
            self.assertEqual(results, payloads)
            self.assertEqual(archive.read("key3", 10, 20), payloads["key3"][10:30])

    @parameterized.parameterized.expand([(zarr_format,) for zarr_format in ZARR_FORMATS])
    def test_dropped_stores_release_file_descriptors(self, zarr_format: int):
        """Read stores that are dropped without being closed don't leak file descriptors"""
        with tempfile.NamedTemporaryFile(suffix=".itar") as f:
            _write_itar(f.name, zarr_format)

            def open_fds() -> int:
                return len(os.listdir(f"/proc/{os.getpid()}/fd"))

            def open_and_drop() -> None:
                _check_reference(self, open_store(IndexedTarStore(f.name), True))

            open_and_drop()
            gc.collect()
            fds_before = open_fds()
            for _ in range(50):
                open_and_drop()
            gc.collect()
            self.assertEqual(open_fds(), fds_before)

    @unittest.skipIf(sys.platform == "win32", "fork not available")
    def test_fork_after_open(self):
        """Readers opened in a parent process remain functional in forked children after reloading resources"""
        with tempfile.NamedTemporaryFile(suffix=".itar") as f:
            _write_itar(f.name, ZARR_FORMATS[-1])

            store = IndexedTarStore(f.name)
            group = open_store(store, True)
            _check_reference(self, group)  # use (thread-local) zarr event loops in the parent

            ctx = multiprocessing.get_context("fork")
            queue = ctx.Queue()

            def child() -> None:
                try:
                    store.reload_resources()
                    queue.put(group.array("subgroup/blob").read_bytes())
                except BaseException as e:  # pragma: no cover
                    queue.put(repr(e))

            processes = [ctx.Process(target=child) for _ in range(2)]
            for p in processes:
                p.start()
            results = [queue.get(timeout=60) for _ in processes]
            for p in processes:
                p.join(60)

            self.assertEqual(results, [b"\x89PNG\x00some binary data"] * len(processes))
            self.assertTrue(all(p.exitcode == 0 for p in processes))
            self.assertEqual(os.getpid(), os.getpid())


class TestNodes(unittest.TestCase):
    """Group / array wrappers"""

    @parameterized.parameterized.expand([(zarr_format,) for zarr_format in ZARR_FORMATS])
    def test_create_array_dtype(self, zarr_format: int):
        """The array dtype defaults to the data's dtype and can be overridden"""
        with tempfile.TemporaryDirectory() as tmp:
            root = create_root_group(open_directory_store(tmp, mode="w"), zarr_format, {})
            self.assertEqual(root.create_array("default", [1.5, 2.5]).dtype, np.float64)
            converted = root.create_array("f32", [1.5, 2.5], dtype=np.float32)
            self.assertEqual(converted.dtype, np.float32)
            self.assertEqual(converted.read().dtype, np.float32)
            np.testing.assert_array_equal(converted.read(), np.array([1.5, 2.5], dtype=np.float32))

    @parameterized.parameterized.expand([(zarr_format,) for zarr_format in ZARR_FORMATS])
    def test_zarr_escape_hatch(self, zarr_format: int):
        """The wrapped zarr-python objects are accessible"""
        with tempfile.TemporaryDirectory() as tmp:
            root = create_root_group(open_directory_store(tmp, mode="w"), zarr_format, {"a": 1})
            array = root.create_array("x", np.arange(3))
            self.assertIsInstance(root.zarr, zarr.Group)
            self.assertIsInstance(array.zarr, zarr.Array)
            self.assertEqual(dict(root.zarr.attrs), {"a": 1})
            self.assertEqual((array.path, array.name, array.zarr_format), ("x", "x", zarr_format))
            np.testing.assert_array_equal(array.zarr[...], np.arange(3))

    @parameterized.parameterized.expand([(zarr_format,) for zarr_format in ZARR_FORMATS])
    def test_typed_attributes(self, zarr_format: int):
        """Typed attribute accessors validate the attribute types"""
        with tempfile.TemporaryDirectory() as tmp:
            attrs = {"s": "x", "i": 3, "b": True, "d": {"k": 1}, "l": [1, "a"]}
            group = create_root_group(open_directory_store(tmp, mode="w"), zarr_format, attrs)
            self.assertEqual(group.attrs, attrs)
            self.assertEqual(
                (group.attr_str("s"), group.attr_int("i"), group.attr_dict("d"), group.attr_list("l")),
                ("x", 3, {"k": 1}, [1, "a"]),
            )
            for accessor, name in [
                (group.attr_str, "i"),
                (group.attr_int, "s"),
                (group.attr_int, "b"),  # bools are not integers
                (group.attr_dict, "l"),
                (group.attr_list, "d"),
            ]:
                with self.assertRaises(TypeError):
                    accessor(name)
            with self.assertRaises(KeyError):
                group.attr_str("missing")

    @parameterized.parameterized.expand([(zarr_format,) for zarr_format in ZARR_FORMATS])
    def test_read_0d(self, zarr_format: int):
        """0-dimensional arrays are read as 0-d ndarrays"""
        with tempfile.TemporaryDirectory() as tmp:
            root = create_root_group(open_directory_store(tmp, mode="w"), zarr_format, {})
            value = root.create_array("v", np.int64(7)).read()
            self.assertIsInstance(value, np.ndarray)
            self.assertEqual((value.shape, value.item()), ((), 7))


class TestBloscCompression(unittest.TestCase):
    """Blosc compression of arrays (identical parameters for zarr-python 2 / 3 and zarr formats v2 / v3)"""

    @parameterized.parameterized.expand(
        [
            (zarr_format, compression)
            for zarr_format in ZARR_FORMATS
            for compression in [
                nodes.BloscCompression(),
                nodes.BloscCompression(cname="zstd", clevel=9, shuffle="shuffle"),
                nodes.BloscCompression(cname="zlib", clevel=1, shuffle="noshuffle", blocksize=4096),
                nodes.BloscCompression(cname="lz4hc"),
                nodes.BloscCompression(cname="blosclz"),
                None,
            ]
        ]
    )
    def test_round_trip(self, zarr_format: int, compression: Optional[nodes.BloscCompression]):
        with tempfile.TemporaryDirectory() as tmp:
            store = open_directory_store(tmp, mode="w")
            root = create_root_group(store, zarr_format, {})
            data = np.arange(4096, dtype=np.float32).reshape(64, 64)
            root.create_array("a", data, compression=compression)
            root.create_bytes_array("b", b"payload")
            consolidate_store(store)

            group = open_store(open_directory_store(tmp, mode="r"), True)
            array = group.array("a")
            np.testing.assert_array_equal(array.read(), data)
            self.assertEqual(array.compression, compression)
            self.assertIsNone(group.array("b").compression)

    def test_default(self):
        """The default compression is blosc lz4 / level 5 / bit-shuffle"""
        self.assertEqual(
            nodes.BloscCompression(),
            nodes.BloscCompression(cname="lz4", clevel=5, shuffle="bitshuffle", blocksize=0),
        )

    @parameterized.parameterized.expand(
        [({"cname": "snappy"},), ({"clevel": 10},), ({"clevel": -1},), ({"shuffle": "byte"},), ({"blocksize": -1},)]
    )
    def test_invalid_parameters(self, kwargs: Any):
        with self.assertRaises(ValueError):
            nodes.BloscCompression(**kwargs)


class TestNodeCache(unittest.TestCase):
    """Node caches of groups (disabled / unbounded / capped LRU)"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def _group(self, cache: Optional[nodes.NodeCache]) -> nodes.Group:
        root = create_root_group(open_directory_store(self._tmp.name, mode="w"), ZARR_FORMATS[-1], {})
        sub = root.create_group("g")
        for i in range(8):
            sub.create_array(f"a{i}", np.full(3, i))
        return root.with_cache(cache)

    def test_create(self):
        self.assertIsNone(nodes.NodeCache.create(0))
        for size in (None, 3):
            cache = nodes.NodeCache.create(size)
            assert cache is not None
            self.assertEqual(cache.max_entries, size)
        with self.assertRaises(ValueError):
            nodes.NodeCache.create(-1)

    def test_disabled(self):
        """Without a cache, nodes are opened on every access"""
        group = self._group(None)
        np.testing.assert_array_equal(group.array("g/a1").read(), [1] * 3)
        self.assertIsNot(group.array("g/a1").zarr, group.array("g/a1").zarr)

    def test_unbounded(self):
        cache = nodes.NodeCache(max_entries=None)
        group = self._group(cache)
        for i in range(8):
            np.testing.assert_array_equal(group.array(f"g/a{i}").read(), [i] * 3)
        self.assertEqual(len(cache), 8)
        self.assertIs(group.array("g/a3").zarr, group.array("g/a3").zarr)

    def test_nested_groups_share_cache(self):
        """Nodes accessed via member groups are cached under their absolute paths"""
        cache = nodes.NodeCache(max_entries=None)
        group = self._group(cache)
        sub = group.group("g")
        self.assertIs(sub.cache, cache)
        self.assertIs(sub.array("a3").zarr, group.array("g/a3").zarr)
        self.assertEqual(cache.keys(), ["g", "g/a3"])

    def test_capped_evicts_least_recently_used(self):
        cache = nodes.NodeCache(max_entries=3)
        group = self._group(cache)
        a0 = group.array("g/a0").zarr
        group.array("g/a1")
        group.array("g/a2")
        self.assertIs(group.array("g/a0").zarr, a0)  # refreshes a0, a1 is now least recently used
        group.array("g/a3")
        self.assertEqual(cache.keys(), ["g/a2", "g/a0", "g/a3"])
        for i in range(8):  # evicted nodes are re-opened transparently
            np.testing.assert_array_equal(group.array(f"g/a{i}").read(), [i] * 3)
        self.assertEqual(len(cache), 3)

    def test_missing_nodes_are_not_cached(self):
        cache = nodes.NodeCache(max_entries=None)
        group = self._group(cache)
        with self.assertRaises(KeyError):
            group.array("g/missing")
        self.assertEqual(len(cache), 0)


@unittest.skipUnless(nodes.ZARR_PYTHON_3, "requires zarr-python 3")
class TestIndexedTarStoreZarr3(unittest.TestCase):
    """zarr-python 3 store adapter of indexed tar stores (asynchronous / synchronous store protocols)"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "store.itar")
        with IndexedTarStore(self.path, mode="w") as store:
            store["a"] = b"0123456789"
            store["b"] = b"abcdef"
            store["dir/c"] = b"c"

    def tearDown(self):
        self._tmp.cleanup()

    def _adapter(self, store: IndexedTarStore) -> Any:
        from ncore.impl.data._zarr3 import _TarStoreAdapter

        return _TarStoreAdapter(store)

    def test_async_reads_in_event_loop(self):
        """All asynchronous read operations work in a running event loop (concurrently)"""
        zarr_abc_store: Any = importlib.import_module("zarr.abc.store")
        prototype = importlib.import_module("zarr.core.buffer").default_buffer_prototype()
        adapter = self._adapter(IndexedTarStore(self.path))

        async def read() -> Any:
            gets = asyncio.gather(
                *(
                    adapter.get(key, prototype, byte_range)
                    for key, byte_range in [
                        ("a", None),
                        ("a", zarr_abc_store.RangeByteRequest(2, 5)),
                        ("a", zarr_abc_store.OffsetByteRequest(7)),
                        ("b", zarr_abc_store.SuffixByteRequest(2)),
                        ("missing", None),
                    ]
                )
            )
            partial = adapter.get_partial_values(
                prototype, [("a", zarr_abc_store.RangeByteRequest(0, 2)), ("missing", None)]
            )
            exists = asyncio.gather(adapter.exists("a"), adapter.exists("missing"))
            listed = [key async for key in adapter.list()]
            prefixed = [key async for key in adapter.list_prefix("dir/")]
            children = [key async for key in adapter.list_dir("")]
            return (await gets, await partial, await exists, listed, prefixed, children)

        gets, partial, exists, listed, prefixed, children = asyncio.run(read())
        to_bytes = lambda values: [None if v is None else v.to_bytes() for v in values]  # noqa: E731
        self.assertEqual(to_bytes(gets), [b"0123456789", b"234", b"789", b"ef", None])
        self.assertEqual(to_bytes(partial), [b"01", None])
        self.assertEqual(list(exists), [True, False])
        self.assertEqual(sorted(listed), ["a", "b", "dir/c"])
        self.assertEqual(prefixed, ["dir/c"])
        self.assertEqual(sorted(children), ["a", "b", "dir"])

    def test_async_writes_in_event_loop(self):
        """Asynchronous writes are write-once, deletes are no-ops for non-existing records only"""
        prototype = importlib.import_module("zarr.core.buffer").default_buffer_prototype()
        path = os.path.join(self._tmp.name, "written.itar")
        store = IndexedTarStore(path, mode="w")
        adapter = self._adapter(store)

        async def write() -> None:
            await adapter.set("x", prototype.buffer.from_bytes(b"1"))
            await adapter.set_if_not_exists("x", prototype.buffer.from_bytes(b"2"))  # no-op
            await adapter.delete("missing")  # no-op
            await adapter.delete_dir("missing")  # no-op
            with self.assertRaises(ValueError):
                await adapter.set("x", prototype.buffer.from_bytes(b"3"))  # write-once
            with self.assertRaises(NotImplementedError):
                await adapter.delete("x")

        asyncio.run(write())
        store.close()
        self.assertEqual(IndexedTarStore(path)["x"], b"1")

    def test_sync_store_protocol(self):
        prototype = importlib.import_module("zarr.core.buffer").default_buffer_prototype()
        path = os.path.join(self._tmp.name, "sync.itar")
        with IndexedTarStore(path, mode="w") as store:
            self._adapter(store).set_sync("a", prototype.buffer.from_bytes(b"xyz"))
        reader = self._adapter(IndexedTarStore(path))
        self.assertEqual(reader.get_sync("a").to_bytes(), b"xyz")
        self.assertIsNone(reader.get_sync("missing"))

    @parameterized.parameterized.expand([(zarr_format,) for zarr_format in ZARR_FORMATS])
    def test_fused_codec_pipeline_uses_sync_fast_path(self, zarr_format: int):
        """zarr-python's fused codec pipeline (zarr-python>=3.4) reads chunks via the synchronous store protocol"""
        zarr_mod: Any = importlib.import_module("zarr")
        codec_pipeline: Any = importlib.import_module("zarr.core.codec_pipeline")
        if not hasattr(codec_pipeline, "FusedCodecPipeline"):
            self.skipTest(f"zarr-python {zarr_mod.__version__} has no fused codec pipeline (zarr-python>=3.4)")

        from ncore.impl.data._zarr3 import _TarStoreAdapter

        path = os.path.join(self._tmp.name, f"fused{zarr_format}.itar")
        _write_itar(path, zarr_format)
        with zarr_mod.config.set({"codec_pipeline.path": "zarr.core.codec_pipeline.FusedCodecPipeline"}):
            with mock.patch.object(
                _TarStoreAdapter, "get_sync", autospec=True, side_effect=_TarStoreAdapter.get_sync
            ) as get_sync:
                _check_reference(self, open_store(IndexedTarStore(path), True))
        self.assertGreater(get_sync.call_count, 0)

    def test_run_inside_event_loop(self):
        """Reads also work from within a running event loop (falls back to zarr's own loop)"""
        _write_itar(path := os.path.join(self._tmp.name, "loop.itar"), 3)
        group = open_store(IndexedTarStore(path), True)

        async def read() -> bytes:
            return group.array("subgroup/blob").read_bytes()

        self.assertEqual(asyncio.run(read()), b"\x89PNG\x00some binary data")


class TestGroupStore(unittest.TestCase):
    @parameterized.parameterized.expand([(zarr_format,) for zarr_format in ZARR_FORMATS])
    def test_group_store_is_indexed_tar_store(self, zarr_format: int):
        """Groups of indexed tar stores return the (zarr-python independent) indexed tar store"""
        with tempfile.TemporaryDirectory() as tmp:
            _write_itar(path := os.path.join(tmp, "a.itar"), zarr_format)
            store = IndexedTarStore(path)
            for open_consolidated in (True, False) if zarr_format == 2 else (True,):
                self.assertIs(get_group_store(open_store(store, open_consolidated)), store)
            self.assertIs(
                get_group_store(open_compressed_consolidated(store, mode="r")) if zarr_format == 2 else store, store
            )


if __name__ == "__main__":
    unittest.main()
