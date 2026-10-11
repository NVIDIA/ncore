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

import gc
import hashlib
import os
import struct
import tarfile
import tempfile
import unittest

from pathlib import Path
from typing import Any, Optional, cast
from unittest import mock

import numpy as np
import parameterized

from ncore.impl.data import nodes
from ncore.impl.data.encoding import decode_cbor_xz
from ncore.impl.data.stores import (
    V2_CONSOLIDATED_METADATA_KEY,
    IndexedTarStore,
    StoreLike,
    close_store,
    consolidate_store,
    create_root_group,
    group_store,
    open_directory_store,
    open_store,
)


#: None / values < 512 (separate reads of index header and index), 1 KiB, 1 MiB (default), edge cases around 512
INDEX_TAIL_READ_SIZES = [None, 1, 512, 513, 1 << 10, 1 << 20]

_ROOT_ATTRIBUTES = {"some": "thing"}


def _fill_reference(root: nodes.Group) -> None:
    rng = np.random.default_rng(0)
    root.create_array("foo", rng.random((3, 3, 3)), chunks=(2, 3, 3))
    sub = root.create_group("subgroup", attributes={"other": [1, 2, 3]})
    sub.create_array("foo", rng.random((5, 5, 5)), attributes={"unit": "m"})
    sub.create_array("empty", np.zeros((0, 3), dtype=np.float32))
    sub.create_bytes_array("blob", b"\x89PNG\x00some binary data", attributes={"format": "png"})


def _check_reference(test: unittest.TestCase, root: nodes.Group) -> None:
    rng = np.random.default_rng(0)
    np.testing.assert_array_equal(root.array("foo").read(), rng.random((3, 3, 3)))
    np.testing.assert_array_equal(root.array("subgroup/foo").read(), rng.random((5, 5, 5)))
    test.assertEqual(root.array("subgroup/empty").shape, (0, 3))
    test.assertEqual(root.array("subgroup/blob").read_bytes(), b"\x89PNG\x00some binary data")
    test.assertEqual(root.attrs, _ROOT_ATTRIBUTES)
    test.assertEqual(root.group("subgroup").attrs, {"other": [1, 2, 3]})
    test.assertEqual(root.array("subgroup/foo").attrs, {"unit": "m"})
    test.assertEqual(sorted(root.members()), ["foo", "subgroup"])
    test.assertEqual(sorted(root.group("subgroup").members()), ["blob", "empty", "foo"])
    test.assertEqual([name for name, _ in root.groups()], ["subgroup"])
    test.assertIn("blob", root.group("subgroup"))
    test.assertNotIn("missing", root.group("subgroup"))
    with test.assertRaises(KeyError):
        root.group("foo")
    with test.assertRaises(KeyError):
        root.array("subgroup")
    with test.assertRaises(KeyError):
        root.array("missing")


def _write(store: StoreLike, consolidate: bool = True) -> None:
    _fill_reference(create_root_group(store, _ROOT_ATTRIBUTES))
    if consolidate:
        consolidate_store(store)
    close_store(store)


class TestIndexedTarStore(unittest.TestCase):
    """Records of indexed tar archives"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "store.itar")

    def tearDown(self):
        self._tmp.cleanup()

    def test_records(self):
        records = {"a": b"0123456789", "dir/b": b"x" * 513, "empty": b"", "l" * 150 + "/long-key": bytes(range(256))}
        with IndexedTarStore(self.path, mode="w") as store:
            for key, value in records.items():
                store[key] = value
            self.assertEqual(store["dir/b"], records["dir/b"])  # readable while writing
            with self.assertRaises(ValueError):  # write-once
                store["a"] = b"again"

        store = IndexedTarStore(self.path)
        self.assertEqual(store.keys(), list(records))
        self.assertEqual(len(store), len(records))
        self.assertEqual({key: store[key] for key in store}, records)
        self.assertIn("a", store)
        self.assertNotIn("missing", store)
        with self.assertRaises(KeyError):
            store["missing"]
        with self.assertRaises(ValueError):  # read-only
            store["c"] = b"c"
        with self.assertRaises(NotImplementedError):
            del store["a"]

        # archives are regular tar archives
        with tarfile.open(self.path) as tar:
            self.assertEqual({m.name: _read_member(tar, m) for m in tar.getmembers()}, records)

    @parameterized.parameterized.expand([(size,) for size in INDEX_TAIL_READ_SIZES])
    def test_index_tail_read_sizes(self, index_tail_read_size: Optional[int]):
        records = {f"r{i}": bytes([i]) * (i * 97) for i in range(40)}
        with IndexedTarStore(self.path, mode="w") as store:
            for key, value in records.items():
                store[key] = value

        store = IndexedTarStore(self.path, index_tail_read_size=index_tail_read_size)
        self.assertEqual({key: store[key] for key in store}, records)

    def test_records_within_the_index_tail_read_require_no_reads(self):
        with IndexedTarStore(self.path, mode="w") as store:
            store["first"] = b"1" * 1000
            store["last"] = b"2" * 1000

        store = IndexedTarStore(self.path, index_tail_read_size=Path(self.path).stat().st_size)
        with mock.patch("os.pread", side_effect=AssertionError("unexpected read")):
            self.assertEqual(store["first"], b"1" * 1000)
            self.assertEqual(store["last"], b"2" * 1000)

    def test_records_partially_within_the_index_tail_read(self):
        """Records starting before the index tail read are read from the archive file"""
        with IndexedTarStore(self.path, mode="w") as store:
            store["large"] = bytes(range(256)) * 64  # 16 KiB
            store["small"] = b"s"

        file_size = Path(self.path).stat().st_size
        store = IndexedTarStore(self.path, index_tail_read_size=file_size - 1024)  # ends within "large"
        self.assertEqual(store["large"], bytes(range(256)) * 64)
        self.assertEqual(store["small"], b"s")

    def test_positional_reads(self):
        """Local archives are read via positional reads"""
        with IndexedTarStore(self.path, mode="w") as store:
            store["a"] = b"a" * 4096

        store = IndexedTarStore(self.path, index_tail_read_size=None)
        with mock.patch("os.pread", side_effect=os.pread) as pread:
            self.assertEqual(store["a"], b"a" * 4096)
        self.assertGreater(pread.call_count, 0)

    def test_dropped_stores_release_file_descriptors(self):
        with IndexedTarStore(self.path, mode="w") as store:
            store["a"] = b"a"

        def open_fds() -> int:
            return len(os.listdir(f"/proc/{os.getpid()}/fd"))

        IndexedTarStore(self.path)["a"]
        gc.collect()
        before = open_fds()
        for _ in range(20):
            IndexedTarStore(self.path)["a"]
        gc.collect()
        self.assertEqual(open_fds(), before)

    def test_reload_resources(self):
        """Reloading resources neither truncates archives opened for writing nor affects reads"""
        with IndexedTarStore(self.path, mode="w") as store:
            store["first"] = b"first payload"
            store.reload_resources()
            self.assertEqual(store["first"], b"first payload")
            store["second"] = b"second payload"

        store = IndexedTarStore(self.path, index_tail_read_size=None)
        store.reload_resources()
        self.assertEqual(store["first"], b"first payload")
        self.assertEqual(store["second"], b"second payload")

    def test_invalid_archives(self):
        invalid_mode: Any = "a"
        with self.assertRaises(ValueError):
            IndexedTarStore(self.path, mode=invalid_mode)
        Path(self.path).write_bytes(b"\0" * 2048)
        with self.assertRaises(ValueError):
            IndexedTarStore(self.path)

    def test_unsupported_index_types(self):
        with IndexedTarStore(self.path, mode="w") as store:
            store["a"] = b"a"
        data = bytearray(Path(self.path).read_bytes())
        header = len(data) - tarfile.BLOCKSIZE  # (the index header is stored in the last tar block)
        magic, _, offset, size = struct.unpack_from(IndexedTarStore.INDEX_HEADER_FORMAT, data, header)
        struct.pack_into(IndexedTarStore.INDEX_HEADER_FORMAT, data, header, magic, 99, offset, size)
        Path(self.path).write_bytes(bytes(data))
        with self.assertRaisesRegex(TypeError, "unsupported index type 99"):
            IndexedTarStore(self.path)

    def test_empty(self):
        with IndexedTarStore(self.path, mode="w"):
            pass
        self.assertEqual(IndexedTarStore(self.path).keys(), [])


class TestStores(unittest.TestCase):
    """zarr hierarchies in indexed tar archives and directories"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.itar_path = os.path.join(self._tmp.name, "store.itar")
        self.directory_path = os.path.join(self._tmp.name, "store.zarr")

    def tearDown(self):
        self._tmp.cleanup()

    def _store(self, store_type: str, mode: str) -> StoreLike:
        if store_type == "itar":
            return IndexedTarStore(self.itar_path, mode="w" if mode == "w" else "r")
        return open_directory_store(self.directory_path)

    @parameterized.parameterized.expand(
        [(store_type, consolidated) for store_type in ("itar", "directory") for consolidated in (True, False)]
    )
    def test_reserialization(self, store_type: str, consolidated: bool):
        _write(self._store(store_type, "w"))

        store = self._store(store_type, "r")
        root = open_store(store, consolidated)
        _check_reference(self, root)

        if isinstance(store, IndexedTarStore):
            self.assertIs(group_store(root), store)
            self.assertIs(group_store(root.group("subgroup")), store)
            store.reload_resources()
            _check_reference(self, root)

    @parameterized.parameterized.expand([(size,) for size in INDEX_TAIL_READ_SIZES])
    def test_consolidated_open_reads_no_node_metadata(self, index_tail_read_size: Optional[int]):
        _write(self._store("itar", "w"))

        store = IndexedTarStore(self.itar_path, index_tail_read_size=index_tail_read_size)
        with mock.patch.object(
            IndexedTarStore, "__getitem__", autospec=True, side_effect=IndexedTarStore.__getitem__
        ) as getitem:
            _check_reference(self, open_store(store, consolidated=True))
        read_keys = [call.args[1] for call in getitem.call_args_list]
        self.assertIn(V2_CONSOLIDATED_METADATA_KEY, read_keys)
        self.assertEqual([key for key in read_keys if key.endswith((".zarray", ".zgroup", ".zattrs"))], [])

    @parameterized.parameterized.expand([("itar",), ("directory",)])
    def test_missing_consolidated_metadata(self, store_type: str):
        _write(self._store(store_type, "w"), consolidate=False)
        with self.assertRaises(KeyError):
            open_store(self._store(store_type, "r"), consolidated=True)
        _check_reference(self, open_store(self._store(store_type, "r"), consolidated=False))

    def test_stores_are_identical(self):
        """Indexed tar archives and directories store the same records"""
        _write(self._store("itar", "w"))
        _write(self._store("directory", "w"))

        archive = IndexedTarStore(self.itar_path)
        files = {
            str(path.relative_to(self.directory_path)): path.read_bytes()
            for path in Path(self.directory_path).rglob("*")
            if path.is_file()
        }
        self.assertEqual(sorted(archive.keys()), sorted(files))
        for key in archive.keys():
            if key == V2_CONSOLIDATED_METADATA_KEY:  # (ordered by the record order of the stores)
                self.assertEqual(decode_cbor_xz(archive[key]), decode_cbor_xz(files[key]))
            else:
                self.assertEqual(archive[key], files[key], key)

    def test_directory_stores_are_not_truncated(self):
        Path(self.directory_path).mkdir()
        (Path(self.directory_path) / "other").write_bytes(b"other")
        _write(self._store("directory", "w"))
        self.assertEqual((Path(self.directory_path) / "other").read_bytes(), b"other")

    def test_archives_of_ncore_19_8(self):
        """Archives are byte-identical to archives written by ncore 19.8 (sha256 of a fixed hierarchy written by
        ncore 19.8)"""
        store = IndexedTarStore(self.itar_path, mode="w")
        root = create_root_group(store, {"some": "thing", "nested": {"a": [1, 2.5, None]}})
        root.create_array(
            "foo",
            np.arange(27, dtype=np.float64).reshape(3, 3, 3),
            chunks=(2, 3, 3),
            compression=nodes.BloscCompression(shuffle="shuffle"),
        )
        sub = root.create_group("subgroup", attributes={"other": [1, 2, 3]})
        sub.create_array("bar", np.arange(10, dtype=np.uint16), compression=None)
        sub.create_bytes_array("blob", b"\x89PNG\x00data")
        consolidate_store(store)
        close_store(store)

        digest = hashlib.sha256(Path(self.itar_path).read_bytes()).hexdigest()
        self.assertIn(
            digest,
            {
                # (python's tarfile module encodes the device numbers of tar headers differently before python 3.9)
                "98852fe2314dc3c9680cd0997d6f914785f5f7d2e552c102ed71c9b400340226",  # python 3.8
                "016af4d3f0d4964c025832a4864b45ffe13cd68e74436da331086d255592a7de",  # python >= 3.9
            },
        )

    def test_empty(self):
        """Stores without a root group can be consolidated, but not opened"""
        store = IndexedTarStore(self.itar_path, mode="w")
        consolidate_store(store)
        close_store(store)
        with self.assertRaises(Exception):  # (zarr-python's error of missing groups)
            open_store(IndexedTarStore(self.itar_path), consolidated=True)

    def test_read_only(self):
        _write(self._store("itar", "w"))
        root = open_store(IndexedTarStore(self.itar_path))
        with self.assertRaises(Exception):
            root.create_group("new")


class TestNodes(unittest.TestCase):
    """Typed node API"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = create_root_group(
            open_directory_store(self._tmp.name), {"s": "x", "i": 3, "b": True, "l": [1], "d": {"k": 1}}
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_typed_attributes(self):
        self.assertEqual(self.root.attr_str("s"), "x")
        self.assertEqual(self.root.attr_int("i"), 3)
        self.assertEqual(self.root.attr_list("l"), [1])
        self.assertEqual(self.root.attr_dict("d"), {"k": 1})
        with self.assertRaises(TypeError):
            self.root.attr_int("b")  # (booleans aren't integers)
        with self.assertRaises(TypeError):
            self.root.attr_str("i")
        with self.assertRaises(KeyError):
            self.root.attr_str("missing")

        self.root.update_attrs({"i": 4, "new": None})
        self.assertEqual(self.root.attrs, {"s": "x", "i": 4, "b": True, "l": [1], "d": {"k": 1}, "new": None})

    def test_arrays(self):
        array = self.root.create_array("a", [1.5, 2.5], dtype=np.float32, attributes={"unit": "m"})
        self.assertEqual((array.shape, array.dtype, array.attrs), ((2,), np.dtype(np.float32), {"unit": "m"}))
        self.assertEqual(array.compression, nodes.BloscCompression())
        np.testing.assert_array_equal(array.read((slice(1, 2),)), [2.5])
        self.assertEqual(array.read((0,)).shape, ())  # 0-d arrays for scalar selections

        self.assertIsNone(self.root.create_array("u", np.arange(3), compression=None).compression)
        self.assertEqual(self.root.create_array("c", np.zeros((10, 0)), chunks=(4, 0)).zarr.chunks, (4, 1))
        with self.assertRaises(TypeError):
            array.read_bytes()

    def test_groups(self):
        self.root.create_group("b")
        self.root.create_group("a").create_group("nested")
        self.assertEqual([name for name, _ in self.root.groups()], ["a", "b"])
        self.assertEqual(self.root.require_group("a").group("nested").path, "a/nested")
        self.assertEqual(self.root.group("a/nested").path, "a/nested")
        with self.assertRaises(Exception):
            self.root.create_group("a")

    def test_nodes(self):
        """Members of unknown kind (groups or arrays)"""
        self.root.create_group("g").create_array("a", np.arange(3))
        assert isinstance(group := self.root.node("g"), nodes.Group)
        self.assertEqual(group.path, "g")
        assert isinstance(array := self.root.node("g/a"), nodes.Array)
        np.testing.assert_array_equal(array.read(), np.arange(3))
        with self.assertRaises(KeyError):
            self.root.node("g/missing")


class TestNodeCache(unittest.TestCase):
    """Node caches of groups (disabled / unbounded / least recently used)"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        path = os.path.join(self._tmp.name, "store.itar")
        store = IndexedTarStore(path, mode="w")
        sub = create_root_group(store, {}).create_group("g")
        for i in range(8):
            sub.create_array(f"a{i}", np.full(3, i))
        consolidate_store(store)
        close_store(store)
        self.root = open_store(IndexedTarStore(path))

    def tearDown(self):
        self._tmp.cleanup()

    def test_uncached(self):
        self.assertIsNone(self.root.cache)
        self.assertIsNot(self.root.array("g/a1").zarr, self.root.array("g/a1").zarr)

    def test_unbounded(self):
        root = self.root.with_cache(cache := nodes.NodeCache())
        for i in range(8):
            np.testing.assert_array_equal(root.array(f"g/a{i}").read(), [i] * 3)
        self.assertEqual(len(cache), 8)
        self.assertIs(root.array("g/a3").zarr, root.array("g/a3").zarr)

    def test_member_groups_share_the_cache(self):
        root = self.root.with_cache(cache := nodes.NodeCache())
        group = root.group("g")
        self.assertIs(group.cache, cache)
        node = root.node("g")
        assert isinstance(node, nodes.Group)
        self.assertIs(node.cache, cache)
        self.assertIs(cast(nodes.Array, root.node("g/a2")).zarr, root.array("g/a2").zarr)
        self.assertIs(group.array("a3").zarr, root.array("g/a3").zarr)
        self.assertIs(next(root.groups())[1].cache, cache)
        self.assertEqual(cache.keys(), ["g", "g/a2", "g/a3"])

    def test_least_recently_used(self):
        root = self.root.with_cache(cache := nodes.NodeCache(max_entries=3))
        a0 = root.array("g/a0").zarr
        root.array("g/a1")
        root.array("g/a2")
        self.assertIs(root.array("g/a0").zarr, a0)  # (a1 is the least recently used node now)
        root.array("g/a3")
        self.assertEqual(cache.keys(), ["g/a2", "g/a0", "g/a3"])
        for i in range(8):  # evicted nodes are re-opened
            np.testing.assert_array_equal(root.array(f"g/a{i}").read(), [i] * 3)
        self.assertEqual(len(cache), 3)

    def test_missing_nodes_are_not_cached(self):
        root = self.root.with_cache(cache := nodes.NodeCache())
        with self.assertRaises(KeyError):
            root.array("g/missing")
        self.assertEqual(len(cache), 0)

    def test_invalid_sizes(self):
        with self.assertRaises(ValueError):
            nodes.NodeCache(max_entries=0)


class TestBloscCompression(unittest.TestCase):
    @parameterized.parameterized.expand(
        [({"cname": "snappy"},), ({"clevel": 10},), ({"clevel": -1},), ({"shuffle": "byte"},), ({"blocksize": -1},)]
    )
    def test_invalid_parameters(self, kwargs: Any):
        with self.assertRaises(ValueError):
            nodes.BloscCompression(**kwargs)

    @parameterized.parameterized.expand([(shuffle,) for shuffle in ("noshuffle", "shuffle", "bitshuffle")])
    def test_numcodecs(self, shuffle: Any):
        compression = nodes.BloscCompression(cname="zstd", clevel=3, shuffle=shuffle, blocksize=1024)
        self.assertEqual(nodes.BloscCompression.from_numcodecs(compression.to_numcodecs()), compression)


def _read_member(tar: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    file = tar.extractfile(member)
    assert file is not None
    return file.read()
