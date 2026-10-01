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
import threading
import unittest
import warnings

from pathlib import Path
from typing import Any, Dict, Optional, cast
from unittest import mock

import numpy as np
import parameterized
import zarr
import zarr.errors

from ncore.impl.data import nodes
from ncore.impl.data._itar import _TarArchive, decompress_document, decompress_document_zlib, is_metadata_key
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

            # the root zarr.json is stored CBOR + zlib encoded, with consolidated node metadata grouped by parent
            root = decompress_document_zlib(store[V3_CONSOLIDATED_METADATA_KEY])
            consolidated = cast(Dict[str, Any], root["consolidated_metadata"])["metadata"]
            self.assertEqual(list(consolidated), ["foo", "subgroup", "subgroup/blob", "subgroup/empty", "subgroup/foo"])
            self.assertEqual(store.consolidated_metadata(), root)

            read_keys = []
            read = store._archive.read
            with mock.patch.object(
                store._archive, "read", side_effect=lambda key, *args: read_keys.append(key) or read(key, *args)
            ):
                g_reload = open_store(store, True)
                _check_reference(self, g_reload)
            self.assertFalse([key for key in read_keys if key.endswith("zarr.json")], read_keys)

            # the root group is created from the decoded consolidated metadata (no JSON serialization round trip)
            with mock.patch.object(store, "get", side_effect=AssertionError("unexpected record read")):
                g_reopened = open_store(store, True)
            self.assertEqual(sorted(g_reopened.group("subgroup").members()), ["blob", "empty", "foo"])
            self.assertEqual(store.consolidated_metadata(), root)  # not modified by zarr-python

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


class TestConsolidatedLookups(unittest.TestCase):
    """Member lookups of consolidated stores are resolved from the consolidated metadata (without a node cache)"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def _open(self, store_type: str, zarr_format: int) -> nodes.Group:
        if store_type == "itar":
            path = os.path.join(self._tmp.name, "store.itar")
            _write_itar(path, zarr_format)
            return open_store(IndexedTarStore(path, mode="r"), open_consolidated=True)

        path = os.path.join(self._tmp.name, "store.zarr")
        root = create_root_group(open_directory_store(path, mode="w"), zarr_format, _ROOT_ATTRIBUTES)
        _fill_reference(root)
        consolidate_store(get_group_store(root))
        return open_store(open_directory_store(path, mode="r"), open_consolidated=True)

    @parameterized.parameterized.expand(itertools.product(["itar", "directory"], ZARR_FORMATS))
    def test_lookups_read_no_node_metadata(self, store_type: str, zarr_format: int):
        root = self._open(store_type, zarr_format)
        self.assertIsNone(root.cache)

        # Record all metadata documents requested from the store from here on
        requested = []
        store = get_group_store(root)
        if isinstance(store, IndexedTarStore):
            get = store.get

            def recording_get(key: str, *args: Any, **kwargs: Any) -> Optional[bytes]:
                requested.append(key)
                return get(key, *args, **kwargs)

            patch = mock.patch.object(store, "get", side_effect=recording_get)
        else:
            # record reads of the directory store class of the installed zarr-python version
            import zarr.storage

            if ZARR_PYTHON_3:
                local_store = getattr(zarr.storage, "LocalStore")
                local_get = local_store.get

                async def recording_local_get(self: Any, key: str, *args: Any, **kwargs: Any) -> Any:
                    requested.append(key)
                    return await local_get(self, key, *args, **kwargs)

                patch = mock.patch.object(local_store, "get", recording_local_get)
            else:
                directory_store = getattr(zarr.storage, "DirectoryStore")
                dir_getitem = directory_store.__getitem__

                def recording_dir_getitem(self: Any, key: str) -> bytes:
                    requested.append(key)
                    return dir_getitem(self, key)

                patch = mock.patch.object(directory_store, "__getitem__", recording_dir_getitem)

        with patch:
            for _ in range(2):
                self.assertEqual(root.group("subgroup").attrs, {"other": [1, 2, 3]})
                self.assertEqual(root.array("subgroup/foo").shape, (5, 5, 5))
                self.assertEqual(root.group("subgroup").array("foo").attrs, {"unit": "m"})
                self.assertEqual(sorted(root.group("subgroup").members()), ["blob", "empty", "foo"])
                self.assertIn("blob", root.group("subgroup"))

        self.assertEqual([key for key in requested if is_metadata_key(key)], [])

        # data reads are unaffected
        _check_reference(self, root)

    @parameterized.parameterized.expand(itertools.product(["itar", "directory"], ZARR_FORMATS))
    def test_interleaved_hierarchy(self, store_type: str, zarr_format: int):
        """Hierarchies created in interleaved order (e.g., by multiple component writers) are opened consolidated"""
        path = os.path.join(self._tmp.name, "store." + ("itar" if store_type == "itar" else "zarr"))
        store = IndexedTarStore(path, mode="w") if store_type == "itar" else open_directory_store(path, mode="w")
        root = create_root_group(store, zarr_format, _ROOT_ATTRIBUTES)
        components = root.create_group("components")
        instances = [components.create_group(name) for name in ("b", "a", "c")]
        for depth in range(3):  # grow the instances' nested groups alternately
            for i, instance in enumerate(instances):
                nested = instance.require_group("nested") if depth else instance.create_group("nested")
                for level in range(depth):
                    nested = nested.require_group(f"level{level}")
                nested.create_array(f"data{depth}", np.full(2, 10 * i + depth))
        consolidate_store(store)
        if isinstance(store, IndexedTarStore):
            store.close()

        reopened = open_store(
            IndexedTarStore(path, mode="r") if store_type == "itar" else open_directory_store(path, mode="r"),
            open_consolidated=True,
        )
        self.assertEqual(sorted(reopened.group("components").members()), ["a", "b", "c"])
        for i, name in enumerate(("b", "a", "c")):
            nested = reopened.group(f"components/{name}/nested")
            self.assertEqual(sorted(nested.members()), ["data0", "level0"])
            np.testing.assert_array_equal(nested.array("level0/level1/data2").read(), [10 * i + 2] * 2)

    @parameterized.parameterized.expand([(zarr_format,) for zarr_format in ZARR_FORMATS])
    def test_consolidated_metadata_in_any_order(self, zarr_format: int):
        """Consolidated metadata in arbitrary order (e.g., written by other tools) is opened with all members"""
        from ncore.impl.data import _itar

        path = os.path.join(self._tmp.name, "store.itar")
        with IndexedTarStore(path, mode="w") as store:
            root = create_root_group(store, zarr_format, _ROOT_ATTRIBUTES)
            for name in ("b", "a"):  # groups with multiple members (zarr-python#4226 drops all but the first run)
                x = root.create_group(name).create_group("x")
                x.create_group("y").create_array("v", np.arange(2))
                x.create_array("z", np.arange(3))
            consolidate_store(store)

        key = V2_CONSOLIDATED_METADATA_KEY if zarr_format == 2 else V3_CONSOLIDATED_METADATA_KEY
        decode, encode = (
            (decompress_document, _itar.compress_document)
            if zarr_format == 2
            else (decompress_document_zlib, _itar.compress_document_zlib)
        )
        with IndexedTarStore(path, mode="r") as store:
            records = {k: store[k] for k in store._archive.keys()}
        document = decode(records[key])
        container = document if zarr_format == 2 else cast(Dict[str, Any], document["consolidated_metadata"])
        separator = "/." if zarr_format == 2 else "/"

        # rewrite the archive with consolidated metadata where members of the same parent group aren't adjacent
        # (members of a/x and b/x interleaved: a/x/y, b/x/y, a/x/z, b/x/z)
        nodes_doc = cast(Dict[str, Any], container["metadata"])

        def interleaved(k: str) -> Any:
            path = k.rsplit(separator, 1)[0] if separator == "/." and separator in k else k
            return (path.count("/"), path.rsplit("/", 1)[-1], path, k)

        shuffled: Dict[str, Any] = {k: nodes_doc[k] for k in sorted(nodes_doc, key=interleaved)}
        container["metadata"] = shuffled
        self.assertNotEqual(list(shuffled), list(_itar.parent_grouped(shuffled, separator)))
        shuffled_path = os.path.join(self._tmp.name, "shuffled.itar")
        with IndexedTarStore(shuffled_path, mode="w") as store:
            for k, value in records.items():
                store[k] = encode(document) if k == key else value

        reopened = open_store(IndexedTarStore(shuffled_path, mode="r"), open_consolidated=True)
        self.assertEqual(sorted(reopened.members()), ["a", "b"])
        for name in ("a", "b"):
            self.assertEqual(sorted(reopened.group(f"{name}/x").members()), ["y", "z"])
            self.assertEqual(reopened.group(f"{name}/x/y").members(), ["v"])
            np.testing.assert_array_equal(reopened.array(f"{name}/x/y/v").read(), np.arange(2))
            np.testing.assert_array_equal(reopened.array(f"{name}/x/z").read(), np.arange(3))


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
class TestDirectReads(unittest.TestCase):
    """Basic selections of numeric arrays of indexed tar archives are read directly (chunks decompressed into the
    output array), identical to zarr-python's reads"""

    # (name, data, chunks, compression)
    CASES: Any = [
        (
            "single_chunk_2d",
            np.random.default_rng(0).random((1000, 3)).astype(np.float32),
            None,
            nodes.BloscCompression(),
        ),
        ("single_chunk_uncompressed", np.arange(1000, dtype=np.uint64), None, None),
        (
            "row_chunks",
            np.random.default_rng(1).random((1003, 3)).astype(np.float32),
            (250, 3),
            nodes.BloscCompression(),
        ),
        (
            "tiled_chunks",
            np.random.default_rng(2).random((101, 67, 5)).astype(np.float16),
            (32, 16, 2),
            nodes.BloscCompression(),
        ),
        ("tiled_uncompressed", np.arange(101 * 67, dtype=np.int32).reshape(101, 67), (32, 16), None),
        (
            "large_chunks",
            np.random.default_rng(5).random((1 << 20, 2)).astype(np.float32),
            (1 << 18, 2),
            nodes.BloscCompression(),
        ),
        ("zero_d", np.array(7.5, dtype=np.float64), None, nodes.BloscCompression()),
        ("empty", np.zeros((0, 3), dtype=np.float32), None, nodes.BloscCompression()),
        ("bool", np.random.default_rng(3).random(333) > 0.5, (100,), nodes.BloscCompression()),
        (
            "lz4hc_byteshuffle",
            np.random.default_rng(4).integers(0, 1000, (500, 7)).astype(np.int16),
            (128, 7),
            nodes.BloscCompression(cname="lz4hc", shuffle="shuffle"),
        ),
    ]

    #: arrays of zarr format v2 only (fixed-length bytes aren't part of the zarr format v3 specification)
    V2_CASES: Any = [
        ("fixed_bytes", np.array([b"ab", b"cde", b""], dtype="S3"), None, None),
        ("fixed_bytes_blosc", np.array([b"abcd", b"\x00ef", b"g"] * 50, dtype="S4"), (64,), nodes.BloscCompression()),
        ("fixed_bytes_0d", np.array(b"\x89PNG\x00x\x00\x00", dtype="S8"), None, None),
    ]

    #: basic selections (applied to all arrays of matching dimensionality)
    SELECTIONS: Any = [
        Ellipsis,
        (Ellipsis,),
        (),
        (slice(None),),
        (slice(3, 97),),
        (slice(-50, None),),
        (5,),
        (-1,),
        (slice(1, 40), Ellipsis),
        (Ellipsis, 1),
        (2, slice(3, 9)),
        (slice(10, 90), slice(5, 60)),
        (slice(10, 90), 3, slice(1, 4)),
        (slice(200, 100),),  # empty
    ]

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def _open(self, zarr_format: int) -> nodes.Group:
        path = os.path.join(self._tmp.name, f"direct{zarr_format}.itar")
        with IndexedTarStore(path, mode="w") as store:
            root = create_root_group(store, zarr_format, {})
            for name, data, chunks, compression in self.CASES + (self.V2_CASES if zarr_format == 2 else []):
                root.create_array(name, data, chunks=chunks, compression=compression)
            root.create_bytes_array("blob", b"\x89PNG\x00some binary data")
            consolidate_store(store)
        return open_store(IndexedTarStore(path, mode="r"), open_consolidated=True)

    @staticmethod
    def _applicable(selection: Any, data: np.ndarray) -> bool:
        """Selections need to index at most all dimensions, and integers need to be in bounds"""
        items = selection if isinstance(selection, tuple) else (selection,)
        indexed = [item for item in items if item is not Ellipsis]
        if len(indexed) > data.ndim:
            return False
        try:
            data[selection]
        except IndexError:
            return False
        return True

    def _check(self, value: Any, expected: Any, context: str) -> None:
        self.assertIs(type(value), type(expected), context)
        if isinstance(expected, np.ndarray):
            self.assertEqual((value.dtype, value.shape), (expected.dtype, expected.shape), context)
        np.testing.assert_array_equal(value, expected, context)

    @parameterized.parameterized.expand([(zarr_format,) for zarr_format in ZARR_FORMATS])
    def test_direct_reads_match_zarr(self, zarr_format: int):
        from ncore.impl.data import _zarr3

        root = self._open(zarr_format)
        with mock.patch.object(_zarr3, "_read_direct", side_effect=_zarr3._read_direct) as direct:
            for name, data, _, _ in self.CASES:
                array = root.array(name)
                for selection in [s for s in self.SELECTIONS if self._applicable(s, data)]:
                    context = f"{name}[{selection!r}]"
                    calls = direct.call_count
                    value = array.read(selection)
                    self.assertEqual(direct.call_count, calls + 1, f"{context} not read directly")
                    # identical to the reference data and to zarr-python's read (as arrays, see Array.read)
                    self._check(value, np.asarray(data[selection]), context)
                    self._check(value, np.asarray(array.zarr[selection]), context)

            # non-basic selections (stepped slices) are read via zarr-python
            calls = direct.call_count
            np.testing.assert_array_equal(root.array("row_chunks").read((slice(None, None, 3),)), self.CASES[2][1][::3])
            self.assertEqual(direct.call_count, calls)

            # binary blobs (fixed-length bytes for zarr format v2, uint8 arrays for zarr format v3) are read directly
            calls = direct.call_count
            self.assertEqual(root.array("blob").read_bytes(), b"\x89PNG\x00some binary data")
            self.assertEqual(direct.call_count, calls + 1)
            for name, data, _, _ in self.V2_CASES if zarr_format == 2 else []:
                array = root.array(name)
                for selection in [s for s in self.SELECTIONS if self._applicable(s, data)]:
                    context = f"{name}[{selection!r}]"
                    calls = direct.call_count
                    value = array.read(selection)
                    self.assertEqual(direct.call_count, calls + 1, f"{context} not read directly")
                    # (np.asarray of fixed-length bytes scalars infers their length, keep the array's dtype)
                    self._check(value, np.asarray(data[selection], dtype=data.dtype), context)
                    self._check(value, np.asarray(array.zarr[selection], dtype=data.dtype), context)
                if data.ndim == 0:  # binary blob
                    self.assertEqual(array.read_bytes(), data.item())

    def test_large_chunks_are_decoded_concurrently(self):
        from ncore.impl.data import _zarr3

        root = self._open(ZARR_FORMATS[-1])
        data = dict((name, data) for name, data, _, _ in self.CASES)
        threads = set()
        decode = _zarr3._decode_chunk

        def recording_decode(*args: Any) -> None:
            threads.add(threading.current_thread().name)
            decode(*args)

        main = threading.current_thread().name
        with mock.patch.object(_zarr3, "_decode_chunk", side_effect=recording_decode):
            np.testing.assert_array_equal(root.array("large_chunks").read(), data["large_chunks"])
            self.assertTrue(threads and all(name.startswith("ncore_zarr3_chunks") for name in threads), threads)

            threads.clear()  # small chunks are decoded on the calling thread
            np.testing.assert_array_equal(root.array("row_chunks").read(), data["row_chunks"])
            self.assertEqual(threads, {main})

            threads.clear()  # as are large chunks if enough reads decode their chunks concurrently already
            with mock.patch.object(_zarr3, "_parallel_reads", _zarr3._PARALLEL_READS):
                np.testing.assert_array_equal(root.array("large_chunks").read(), data["large_chunks"])
            self.assertEqual(threads, {main})
        self.assertEqual(_zarr3._parallel_reads, 0)

    def test_async_reads(self):
        """Direct reads are coroutines, which can be awaited concurrently on any event loop"""
        from ncore.impl.data import _zarr3

        root = self._open(ZARR_FORMATS[-1])

        async def read_all() -> Any:
            return await asyncio.gather(
                *(_zarr3._Backend().read_async(root.array(name).zarr, Ellipsis) for name, _, _, _ in self.CASES)
            )

        for value, (name, data, _, _) in zip(asyncio.run(read_all()), self.CASES):
            np.testing.assert_array_equal(value, data, name)

    @parameterized.parameterized.expand([("direct",), ("zarr",)])
    def test_async_reads_run_blocking_work_on_threads(self, mode: str):
        """On event loops of applications, blocking parts of reads (decompression, local reads) run on worker threads"""
        from ncore.impl.data import _zarr3

        root = self._open(ZARR_FORMATS[-1])
        data = dict((name, data) for name, data, _, _ in self.CASES)
        threads = set()
        decode, read = _zarr3._decode_chunk, _zarr3._Backend.read

        def recording_decode(*args: Any) -> None:
            threads.add(threading.current_thread().name)
            decode(*args)

        def recording_read(self: Any, *args: Any) -> Any:
            threads.add(threading.current_thread().name)
            return read(self, *args)

        async def read_all() -> Any:
            return await asyncio.gather(*(root.array(name).read_async() for name in ("row_chunks", "single_chunk_2d")))

        with mock.patch.dict(os.environ, {_zarr3.DIRECT_READS_ENV: "1" if mode == "direct" else "0"}):
            with mock.patch.object(_zarr3, "_decode_chunk", side_effect=recording_decode):
                with mock.patch.object(_zarr3._Backend, "read", autospec=True, side_effect=recording_read):
                    values = asyncio.run(read_all())
        np.testing.assert_array_equal(values[0], data["row_chunks"])
        np.testing.assert_array_equal(values[1], data["single_chunk_2d"])
        self.assertTrue(threads, "no blocking work recorded")
        self.assertNotIn(threading.current_thread().name, threads)  # loop thread (the test's main thread) not blocked

    def test_disabled(self):
        """Direct reads can be disabled (all reads via zarr-python's codec pipeline)"""
        from ncore.impl.data import _zarr3

        root = self._open(ZARR_FORMATS[-1])
        with mock.patch.dict(os.environ, {_zarr3.DIRECT_READS_ENV: "0"}):
            with mock.patch.object(_zarr3, "_read_direct", side_effect=_zarr3._read_direct) as direct:
                for name, data, _, _ in self.CASES:
                    np.testing.assert_array_equal(root.array(name).read(), data, name)
        self.assertEqual(direct.call_count, 0)


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
        from ncore.impl.data._zarr3 import DIRECT_READS_ENV

        # direct reads bypass zarr-python's codec pipeline, disable them to read via the fused pipeline
        with zarr_mod.config.set({"codec_pipeline.path": "zarr.core.codec_pipeline.FusedCodecPipeline"}):
            with mock.patch.dict(os.environ, {DIRECT_READS_ENV: "0"}):
                with mock.patch.object(
                    _TarStoreAdapter, "get_sync", autospec=True, side_effect=_TarStoreAdapter.get_sync
                ) as get_sync:
                    _check_reference(self, open_store(IndexedTarStore(path), True))
        self.assertGreater(get_sync.call_count, 0)

    def test_remote_reads_on_other_event_loops(self):
        """Asynchronous reads of remote archives run on the event loop of their filesystem, also when awaited on other
        event loops (e.g., fsspec's http filesystem binds its sessions to fsspec's I/O loop)"""
        import fsspec.asyn

        from ncore.impl.data import _itar

        _write_itar(path := os.path.join(self._tmp.name, "remote.itar"), ZARR_FORMATS[-1])
        expected = _check_reference

        class LoopBoundFileSystem(fsspec.asyn.AsyncFileSystem):
            """A filesystem whose coroutines only work on its own event loop (as fsspec's http filesystem)"""

            async_impl = True

            async def _cat_file(
                self, path: str, start: Optional[int] = None, end: Optional[int] = None, **kwargs: Any
            ) -> bytes:
                if asyncio.get_running_loop() is not self.loop:
                    raise RuntimeError("Timeout context manager should be used inside a task")
                with open(path, "rb") as f:
                    f.seek(start or 0)
                    return f.read(None if end is None else end - (start or 0))

        fs = LoopBoundFileSystem(loop=fsspec.asyn.get_loop())
        store = IndexedTarStore(path, index_tail_read_size=None)  # (chunks aren't served from the index tail read)
        cat_file = fs._cat_file
        requests = []

        async def recording_cat_file(*args: Any, **kwargs: Any) -> bytes:
            requests.append(args[0])
            return await cat_file(*args, **kwargs)

        with (
            mock.patch.object(_itar._TarArchive, "_async_fs", lambda self: fs),
            mock.patch.object(fs, "_cat_file", recording_cat_file),
        ):
            group = open_store(store, True)
            self.assertTrue(store.is_remote)

            async def read_all() -> Any:
                return await asyncio.gather(
                    group.array("foo").read_async(), group.array("subgroup/foo").read_async((slice(1, 3),))
                )

            foo, sub = asyncio.run(read_all())  # awaited on another loop
            expected(self, group)  # synchronous reads (event loop of the calling thread)
        self.assertGreater(len(requests), 1)  # chunks fetched via range requests
        rng = np.random.default_rng(0)
        np.testing.assert_array_equal(foo, rng.random((3, 3, 3)))
        np.testing.assert_array_equal(sub, rng.random((5, 5, 5))[1:3])

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
