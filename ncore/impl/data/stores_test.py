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
import importlib
import os
import struct
import tarfile
import tempfile
import types
import unittest

from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple, cast
from unittest import mock

import numcodecs
import numpy as np
import parameterized
import zarr
import zarr.errors
import zarr.storage

from ncore.impl.data import nodes
from ncore.impl.data.backend import V2_CONSOLIDATED_METADATA_KEY, Backend
from ncore.impl.data.encoding import decode_cbor_xz, encode_cbor_xz
from ncore.impl.data.itar import IndexedTarArchive
from ncore.impl.data.json_types import JsonDocument, JsonLike, to_json, to_json_document
from ncore.impl.data.stores import IndexedTarZarrStore, Store, ZarrStoreAdapter, is_consolidated_metadata_key


ZARR_PYTHON_3 = int(zarr.__version__.split(".", 1)[0]) >= 3

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


def _write(store: Store, consolidate: bool = True) -> None:
    with store:
        _fill_reference(store.create_root_group(_ROOT_ATTRIBUTES))
        if consolidate:
            store.consolidate()


class TestIndexedTarArchive(unittest.TestCase):
    """Records of indexed tar archives"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "store.itar")

    def tearDown(self):
        self._tmp.cleanup()

    def test_read(self):
        """read() is the synchronous read of the mapping interface (archive[key]), incl. empty records and records of the
        index tail read"""
        records = {"a": b"0123456789", "empty": b"", "large": bytes(range(256)) * 64}
        with IndexedTarArchive(self.path, mode="w") as archive:
            for key, value in records.items():
                archive[key] = value
        for tail in (None, 1 << 20):  # (records read from the file, records of the tail read)
            archive = IndexedTarArchive(self.path, index_tail_read_size=tail)
            for key, value in records.items():
                self.assertEqual(archive.read(key), value, key)
                self.assertEqual(archive[key], value, key)
            with self.assertRaises(KeyError):
                archive.read("missing")

    def test_records(self):
        records = {"a": b"0123456789", "dir/b": b"x" * 513, "empty": b"", "l" * 150 + "/long-key": bytes(range(256))}
        with IndexedTarArchive(self.path, mode="w") as archive:
            for key, value in records.items():
                archive[key] = value
            self.assertEqual(archive["dir/b"], records["dir/b"])  # readable while writing
            with self.assertRaises(ValueError):  # write-once
                archive["a"] = b"again"

        archive = IndexedTarArchive(self.path)
        self.assertEqual(archive.keys(), list(records))
        self.assertEqual(len(archive), len(records))
        self.assertEqual({key: archive[key] for key in archive}, records)
        self.assertIn("a", archive)
        self.assertNotIn("missing", archive)
        with self.assertRaises(KeyError):
            archive["missing"]
        with self.assertRaises(ValueError):  # read-only
            archive["c"] = b"c"
        with self.assertRaises(NotImplementedError):
            del archive["a"]

        # archives are regular tar archives
        with tarfile.open(self.path) as tar:
            self.assertEqual({m.name: _read_member(tar, m) for m in tar.getmembers()}, records)

    @parameterized.parameterized.expand([(size,) for size in INDEX_TAIL_READ_SIZES])
    def test_index_tail_read_sizes(self, index_tail_read_size: Optional[int]):
        records = {f"r{i}": bytes([i]) * (i * 97) for i in range(40)}
        with IndexedTarArchive(self.path, mode="w") as archive:
            for key, value in records.items():
                archive[key] = value

        archive = IndexedTarArchive(self.path, index_tail_read_size=index_tail_read_size)
        self.assertEqual({key: archive[key] for key in archive}, records)

    def test_records_within_the_index_tail_read_require_no_reads(self):
        with IndexedTarArchive(self.path, mode="w") as archive:
            archive["first"] = b"1" * 1000
            archive["last"] = b"2" * 1000

        archive = IndexedTarArchive(self.path, index_tail_read_size=Path(self.path).stat().st_size)
        with mock.patch("os.pread", side_effect=AssertionError("unexpected read")):
            self.assertEqual(archive["first"], b"1" * 1000)
            self.assertEqual(archive["last"], b"2" * 1000)

    def test_records_partially_within_the_index_tail_read(self):
        """Records starting before the index tail read are read from the archive file"""
        with IndexedTarArchive(self.path, mode="w") as archive:
            archive["large"] = bytes(range(256)) * 64  # 16 KiB
            archive["small"] = b"s"

        file_size = Path(self.path).stat().st_size
        archive = IndexedTarArchive(self.path, index_tail_read_size=file_size - 1024)  # ends within "large"
        self.assertEqual(archive["large"], bytes(range(256)) * 64)
        self.assertEqual(archive["small"], b"s")

    def test_positional_reads(self):
        """Local archives are read via positional reads"""
        with IndexedTarArchive(self.path, mode="w") as archive:
            archive["a"] = b"a" * 4096

        archive = IndexedTarArchive(self.path, index_tail_read_size=None)
        with mock.patch("os.pread", side_effect=os.pread) as pread:
            self.assertEqual(archive["a"], b"a" * 4096)
        self.assertGreater(pread.call_count, 0)

    def test_dropped_stores_release_file_descriptors(self):
        with IndexedTarArchive(self.path, mode="w") as archive:
            archive["a"] = b"a"

        def open_fds() -> int:
            return len(os.listdir(f"/proc/{os.getpid()}/fd"))

        IndexedTarArchive(self.path)["a"]
        gc.collect()
        before = open_fds()
        for _ in range(20):
            IndexedTarArchive(self.path)["a"]
        gc.collect()
        self.assertEqual(open_fds(), before)

    def test_reload_resources(self):
        """Reloading resources neither truncates archives opened for writing nor affects reads"""
        with IndexedTarArchive(self.path, mode="w") as archive:
            archive["first"] = b"first payload"
            archive.reload_resources()
            self.assertEqual(archive["first"], b"first payload")
            archive["second"] = b"second payload"

        archive = IndexedTarArchive(self.path, index_tail_read_size=None)
        archive.reload_resources()
        self.assertEqual(archive["first"], b"first payload")
        self.assertEqual(archive["second"], b"second payload")

    def test_invalid_archives(self):
        invalid_mode: Any = "a"
        with self.assertRaises(ValueError):
            IndexedTarArchive(self.path, mode=invalid_mode)
        Path(self.path).write_bytes(b"\0" * 2048)
        with self.assertRaises(ValueError):
            IndexedTarArchive(self.path)

    def test_unsupported_index_types(self):
        with IndexedTarArchive(self.path, mode="w") as archive:
            archive["a"] = b"a"
        data = bytearray(Path(self.path).read_bytes())
        header = len(data) - tarfile.BLOCKSIZE  # (the index header is stored in the last tar block)
        magic, _, offset, size = struct.unpack_from(IndexedTarArchive.INDEX_HEADER_FORMAT, data, header)
        struct.pack_into(IndexedTarArchive.INDEX_HEADER_FORMAT, data, header, magic, 99, offset, size)
        Path(self.path).write_bytes(bytes(data))
        with self.assertRaisesRegex(TypeError, "unsupported index type 99"):
            IndexedTarArchive(self.path)

    def test_empty(self):
        with IndexedTarArchive(self.path, mode="w"):
            pass
        self.assertEqual(IndexedTarArchive(self.path).keys(), [])


class TestStores(unittest.TestCase):
    """zarr hierarchies in indexed tar archives and directories"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.itar_path = os.path.join(self._tmp.name, "store.itar")
        self.directory_path = os.path.join(self._tmp.name, "store.zarr")

    def tearDown(self):
        self._tmp.cleanup()

    def _store(self, store_type: str, mode: str) -> Store:
        if store_type == "itar":
            return IndexedTarZarrStore(self.itar_path, mode="w" if mode == "w" else "r")
        return ZarrStoreAdapter.directory(self.directory_path)

    @parameterized.parameterized.expand(
        [(store_type, consolidated) for store_type in ("itar", "directory") for consolidated in (True, False)]
    )
    def test_reserialization(self, store_type: str, consolidated: bool):
        _write(self._store(store_type, "w"))

        with self._store(store_type, "r") as store:
            root = store.open_root_group(consolidated)
            _check_reference(self, root)

            self.assertIs(root.store, store)
            self.assertIs(root.group("subgroup").store, store)
            store.reload_resources()
            _check_reference(self, root)

    @parameterized.parameterized.expand([(size,) for size in INDEX_TAIL_READ_SIZES])
    def test_consolidated_open_reads_no_node_metadata(self, index_tail_read_size: Optional[int]):
        _write(self._store("itar", "w"))

        store = IndexedTarZarrStore(self.itar_path, index_tail_read_size=index_tail_read_size)
        with mock.patch.object(IndexedTarArchive, "read", autospec=True, side_effect=IndexedTarArchive.read) as getitem:
            _check_reference(self, store.open_root_group(consolidated=True))
        read_keys = [call.args[1] for call in getitem.call_args_list]
        self.assertIn(V2_CONSOLIDATED_METADATA_KEY, read_keys)
        self.assertEqual([key for key in read_keys if key.endswith((".zarray", ".zgroup", ".zattrs"))], [])

    @parameterized.parameterized.expand([("itar",), ("directory",)])
    def test_missing_consolidated_metadata(self, store_type: str):
        _write(self._store(store_type, "w"), consolidate=False)
        with self.assertRaises(KeyError):
            self._store(store_type, "r").open_root_group(consolidated=True)
        _check_reference(self, self._store(store_type, "r").open_root_group(consolidated=False))

    def test_stores_are_identical(self):
        """Indexed tar archives and directories store the same records"""
        _write(self._store("itar", "w"))
        _write(self._store("directory", "w"))

        archive = IndexedTarArchive(self.itar_path)
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
        with IndexedTarZarrStore(self.itar_path, mode="w") as store:
            root = store.create_root_group({"some": "thing", "nested": {"a": [1, 2.5, None]}})
            root.create_array(
                "foo",
                np.arange(27, dtype=np.float64).reshape(3, 3, 3),
                chunks=(2, 3, 3),
                compression=nodes.BloscCompression(shuffle="shuffle"),
            )
            sub = root.create_group("subgroup", attributes={"other": [1, 2, 3]})
            sub.create_array("bar", np.arange(10, dtype=np.uint16), compression=None)
            sub.create_bytes_array("blob", b"\x89PNG\x00data")
            store.consolidate()

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
        with IndexedTarZarrStore(self.itar_path, mode="w") as store:
            store.consolidate()
        with self.assertRaises(Exception):  # (zarr-python's error of missing groups)
            IndexedTarZarrStore(self.itar_path).open_root_group(consolidated=True)

    def test_read_only(self):
        _write(self._store("itar", "w"))
        root = IndexedTarZarrStore(self.itar_path).open_root_group()
        with self.assertRaises(Exception):
            root.create_group("new")

    def test_context_managers_return_the_store(self):
        with IndexedTarZarrStore(self.itar_path, mode="w") as store:
            store.create_root_group(_ROOT_ATTRIBUTES)
            store.consolidate()
            self.assertEqual(store.archive.mode, "w")  # (typed as IndexedTarZarrStore)
        with IndexedTarArchive(self.itar_path) as archive:
            self.assertIn(V2_CONSOLIDATED_METADATA_KEY, archive.keys())

    def test_consolidated_metadata_keys(self):
        with IndexedTarZarrStore(self.itar_path, mode="w") as store:
            _write(store)
            keys = store.archive.keys()
        self.assertEqual([key for key in keys if is_consolidated_metadata_key(key)], [V2_CONSOLIDATED_METADATA_KEY])
        self.assertTrue(is_consolidated_metadata_key(".zmetadata"))  # (zarr-python 2's consolidated metadata)
        for key in (".zgroup", ".zattrs", "zarr.json", "a/.zmetadata", "a/.zmetadata.cbor.xz"):
            self.assertFalse(is_consolidated_metadata_key(key), key)

    def test_remote_directory_stores_are_rejected(self):
        with self.assertRaises(ValueError):
            ZarrStoreAdapter.directory("memory://bucket/store.zarr")

    def test_other_zarr_stores(self):
        """Hierarchies in other zarr-python stores (e.g., in-memory stores)"""
        store = ZarrStoreAdapter(zarr.storage.MemoryStore())
        _write(store)
        _check_reference(self, store.open_root_group())


class TestNodes(unittest.TestCase):
    """Typed node API"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = ZarrStoreAdapter.directory(self._tmp.name).create_root_group(
            {"s": "x", "i": 3, "b": True, "l": [1], "d": {"k": 1}}
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

    def test_attributes_of_abstract_collections(self):
        """Attribute values of any mappings / sequences are written (and read back) as plain dicts / lists"""

        class CustomMapping(Mapping[str, JsonLike]):
            def __init__(self, **items: JsonLike) -> None:
                self._items = items

            def __getitem__(self, key: str) -> JsonLike:
                return self._items[key]

            def __iter__(self) -> Iterator[str]:
                return iter(self._items)

            def __len__(self) -> int:
                return len(self._items)

        ints: List[int] = [1, 2]
        attributes: Dict[str, JsonLike] = {
            "tuple": (1, ("nested", 2.5)),
            "proxy": types.MappingProxyType({"k": (True, None), "m": CustomMapping(x=(1,))}),
            "custom": CustomMapping(a=types.MappingProxyType({"b": [()]})),
            "ints": ints,
        }
        expected: JsonDocument = {
            "tuple": [1, ["nested", 2.5]],
            "proxy": {"k": [True, None], "m": {"x": [1]}},
            "custom": {"a": {"b": [[]]}},
            "ints": [1, 2],
        }

        def plain(value: Any) -> bool:
            """True for values of plain JSON types only (as read from JSON, e.g., no tuples or other mappings)"""
            if isinstance(value, (dict, list)):
                return type(value) in (dict, list) and all(
                    map(plain, value.values() if isinstance(value, dict) else value)
                )
            return value is None or type(value) in (str, int, float, bool)

        def check(attrs: JsonDocument, reference: JsonDocument) -> None:
            self.assertEqual(attrs, reference)  # (tuples compare unequal to lists)
            self.assertTrue(plain(attrs), attrs)

        check(to_json_document(attributes), expected)
        self.assertEqual(to_json(b"ab"), b"ab")  # (bytes aren't JSON values, writing them fails)

        group = self.root.create_group("g", attributes=attributes)
        array = self.root.create_array("a", np.arange(2), attributes=attributes)
        check(group.attrs, expected)
        check(array.attrs, expected)
        group.update_attrs({"tuple": (3,), "added": types.MappingProxyType({"t": ("x",)})})
        check(group.attrs, {**expected, "tuple": [3], "added": {"t": ["x"]}})

        assert (store := self.root.store) is not None
        store.consolidate()
        reopened = store.open_root_group()
        check(reopened.group("g").attrs, {**expected, "tuple": [3], "added": {"t": ["x"]}})
        check(reopened.array("a").attrs, expected)

        root = ZarrStoreAdapter.directory(os.path.join(self._tmp.name, "other")).create_root_group(attributes)
        check(root.attrs, expected)

    def test_arrays(self):
        array = self.root.create_array("a", [1.5, 2.5], dtype=np.float32, attributes={"unit": "m"})
        self.assertEqual((array.shape, array.dtype, array.attrs), ((2,), np.dtype(np.float32), {"unit": "m"}))
        self.assertEqual(array.compression, nodes.BloscCompression())
        np.testing.assert_array_equal(array.read((slice(1, 2),)), [2.5])
        self.assertEqual(array.read((0,)).shape, ())  # 0-d arrays for scalar selections

        self.assertIsNone(self.root.create_array("u", np.arange(3), compression=None).compression)
        # (zero-sized chunk dimensions are clamped to 1, also of single chunks)
        self.assertEqual(self.root.create_array("c", np.zeros((10, 0)), chunks=(4, 0)).zarr.chunks, (4, 1))
        self.assertEqual(self.root.create_array("s", np.zeros((0, 3))).zarr.chunks, (1, 3))
        self.assertEqual(self.root.create_array("s0", np.zeros((0, 3))).read().shape, (0, 3))
        self.assertEqual(self.root.create_array("auto", np.zeros(4096), chunks="auto").zarr.chunks, (4096,))
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

    def test_zarr_python_style_read_access(self):
        """Members via indexing, get, iteration, len, keys and items (as zarr-python groups)"""
        group = self.root.create_group("g")
        group.create_group("b").create_group("nested")
        group.create_array("a", np.arange(3))
        empty = group.create_group("empty")

        assert isinstance(nested := self.root["g/b/nested"], nodes.Group)
        self.assertEqual(nested.path, "g/b/nested")
        assert isinstance(array := group["a"], nodes.Array)
        np.testing.assert_array_equal(array.read(), np.arange(3))
        with self.assertRaises(KeyError):
            group["missing"]
        with self.assertRaises(KeyError):
            self.root["g/b/missing"]

        assert isinstance(member := group.get("b"), nodes.Group)
        self.assertEqual(member.path, "g/b")
        self.assertIsInstance(self.root.get("g/a"), nodes.Array)
        self.assertIsNone(group.get("missing"))
        self.assertIsNone(self.root.get("g/b/missing"))

        self.assertEqual(sorted(group), ["a", "b", "empty"])
        self.assertEqual(sorted(group.keys()), ["a", "b", "empty"])
        self.assertEqual(len(group), 3)
        self.assertEqual(len(self.root), 1)
        self.assertEqual((len(empty), list(empty), empty.keys(), list(empty.items())), (0, [], [], []))
        self.assertFalse(empty)  # (empty groups are falsy, as zarr-python groups)
        self.assertTrue(group)

        items = dict(group.items())
        self.assertEqual(sorted(items), ["a", "b", "empty"])
        self.assertIsInstance(items["a"], nodes.Array)
        self.assertIsInstance(items["b"], nodes.Group)
        self.assertEqual((items["a"].path, items["b"].path), ("g/a", "g/b"))

        # (groups are not mappings: they compare by identity)
        self.assertNotEqual(self.root.group("g/empty"), empty)

    def test_names(self):
        """Node names as zarr-python names them (paths with a leading slash)"""
        self.assertEqual((self.root.path, self.root.name), ("", "/"))
        nested = self.root.create_group("a").create_group("b")
        self.assertEqual((nested.path, nested.name), ("a/b", "/a/b"))
        self.assertEqual(self.root.group("a/b").name, "/a/b")
        array = nested.create_array("c", np.arange(2))
        self.assertEqual(array.name, "/a/b/c")
        self.assertEqual(self.root["a/b/c"].name, "/a/b/c")


class TestNodeCache(unittest.TestCase):
    """Node caches of groups (disabled / unbounded / least recently used)"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        path = os.path.join(self._tmp.name, "store.itar")
        with IndexedTarZarrStore(path, mode="w") as store:
            sub = store.create_root_group({}).create_group("g")
            for i in range(8):
                sub.create_array(f"a{i}", np.full(3, i))
            store.consolidate()
        self.root = IndexedTarZarrStore(path).open_root_group()

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

    def test_zarr_python_style_access_uses_the_cache(self):
        root = self.root.with_cache(cache := nodes.NodeCache())
        assert isinstance(group := root["g"], nodes.Group)
        self.assertIs(group.cache, cache)
        self.assertIs(cast(nodes.Array, root["g/a1"]).zarr, cast(nodes.Array, group.get("a1")).zarr)
        self.assertIsNone(group.get("missing"))
        members = dict(group.items())
        self.assertIs(cast(nodes.Array, members["a2"]).zarr, root.array("g/a2").zarr)
        self.assertEqual(sorted(cache.keys()), ["g"] + [f"g/a{i}" for i in range(8)])

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


class TestZeroCopyReads(unittest.TestCase):
    """Read results are read-only, arrays stored as a single uncompressed chunk are read as views of their records"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "store.itar")

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, zarr_format: int) -> None:
        with IndexedTarZarrStore(self.path, mode="w") as store:
            root = store.create_root_group({})
            root.create_bytes_array("blob", b"\x89PNG\x00data")
            root.create_array("single", np.arange(6, dtype=np.float32).reshape(2, 3), compression=None)
            root.create_array("compressed", np.arange(6, dtype=np.float32).reshape(2, 3))
            root.create_array("chunked", np.arange(6, dtype=np.float32), chunks=(4,), compression=None)
            store.consolidate()

    def _reads(self, root: nodes.Group) -> Dict[str, Tuple[Any, Dict[str, bytes]]]:
        """Read results, and the records read for them"""
        out: Dict[str, Tuple[Any, Dict[str, bytes]]] = {}
        getitem = IndexedTarArchive.read
        for name, read in [
            ("blob_bytes", lambda: root.array("blob").read_bytes()),
            ("blob", lambda: root.array("blob").read()),
            ("single", lambda: root.array("single").read()),
            ("single_row", lambda: root.array("single").read((1,))),
            ("compressed", lambda: root.array("compressed").read()),
            ("chunked", lambda: root.array("chunked").read()),
        ]:
            reads: Dict[str, bytes] = {}

            def recording_getitem(archive: IndexedTarArchive, key: str, reads: Dict[str, bytes] = reads) -> bytes:
                reads[key] = value = getitem(archive, key)
                return value

            with mock.patch.object(IndexedTarArchive, "read", autospec=True, side_effect=recording_getitem):
                out[name] = read(), reads
        return out

    @parameterized.parameterized.expand([(zarr_format,) for zarr_format in nodes.SUPPORTED_ZARR_FORMATS])
    def test_reads(self, zarr_format: int):
        self._write(zarr_format)
        reads = self._reads(IndexedTarZarrStore(self.path).open_root_group())

        def record(name: str) -> bytes:
            (value,) = [v for k, v in reads[name][1].items() if not k.endswith((".zarray", ".zattrs", "zarr.json"))]
            return value

        # read_bytes: the chunk record of the blob
        blob, _ = reads["blob_bytes"]
        self.assertEqual(blob, b"\x89PNG\x00data")
        self.assertIs(blob, record("blob_bytes"))

        # complete reads of single uncompressed chunks: read-only views of their records
        for name in ("blob", "single"):
            array, _ = reads[name]
            base = array
            while isinstance(base, np.ndarray):
                base = base.base
            self.assertIs(base, record(name), name)
        np.testing.assert_array_equal(reads["single"][0], np.arange(6, dtype=np.float32).reshape(2, 3))

        # all results are read-only arrays, also of partial reads, compressed and multi-chunk arrays
        for name, (value, _) in reads.items():
            if name != "blob_bytes":
                self.assertIsInstance(value, np.ndarray, name)
                self.assertFalse(value.flags.writeable, name)
        np.testing.assert_array_equal(reads["single_row"][0], [3, 4, 5])
        np.testing.assert_array_equal(reads["chunked"][0], np.arange(6, dtype=np.float32))

    @parameterized.parameterized.expand([(zarr_format,) for zarr_format in nodes.SUPPORTED_ZARR_FORMATS])
    def test_missing_single_chunks_read_as_fill_value(self, zarr_format: int):
        """(chunks equal to the fill value aren't stored by default by other zarr writers)"""
        self._write(zarr_format)
        source = IndexedTarArchive(self.path)
        copy_path = os.path.join(self._tmp.name, "copy.itar")
        with IndexedTarArchive(copy_path, mode="w") as copy:
            for key in source.keys():
                if not key.startswith("single/") or key.endswith((".zarray", ".zattrs", "zarr.json")):
                    copy[key] = source[key]
        array = IndexedTarZarrStore(copy_path).open_root_group().array("single").read()
        np.testing.assert_array_equal(array, np.zeros((2, 3), dtype=np.float32))
        self.assertFalse(array.flags.writeable)


@parameterized.parameterized_class(("zarr_format",), [(zarr_format,) for zarr_format in nodes.SUPPORTED_ZARR_FORMATS])
class TestEdgeCases(unittest.TestCase):
    """Edge cases of reads (blobs, fast-path fallbacks) and error paths of the backends"""

    zarr_format: int

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "store.itar")

    def tearDown(self):
        self._tmp.cleanup()

    def _root(self, fill: Any) -> nodes.Group:
        zarr_format = self.zarr_format  # noqa: F841 (used by stores of zarr format v3)
        with IndexedTarZarrStore(self.path, mode="w") as store:
            fill(store.create_root_group({}))
            store.consolidate()
        return IndexedTarZarrStore(self.path).open_root_group()

    def test_blobs_of_multiple_chunks(self):
        """Blobs of multiple chunks are copied once (not views of a single chunk record)"""
        data = bytes(range(256)) * 3

        def fill(root: nodes.Group) -> None:
            root.create_array("multi", np.frombuffer(data, dtype=np.uint8), chunks=(100,), compression=None)
            root.create_array("compressed", np.frombuffer(data, dtype=np.uint8))  # (single compressed chunk)

        root = self._root(fill)
        for name in ("multi", "compressed"):
            blob = root.array(name).read_bytes()
            self.assertEqual(blob, data, name)
            self.assertNotIn(blob, IndexedTarArchive(self.path).keys())  # (sanity: a blob, not a key)
            array = root.array(name).read()
            self.assertFalse(array.flags.writeable, name)
            base = array
            while isinstance(base, np.ndarray):
                base = base.base
            self.assertNotIsInstance(base, bytes, name)  # (decoded, not a view of a record)

    def test_non_blob_arrays(self):
        """read_bytes raises TypeError for arrays that aren't blob arrays"""

        def fill(root: nodes.Group) -> None:
            root.create_array("floats", np.arange(4, dtype=np.float32))
            root.create_array("scalar", np.asarray(3, dtype=np.int64))
            root.create_array("matrix", np.zeros((2, 2), dtype=np.uint8))

        root = self._root(fill)
        for name in ("floats", "scalar", "matrix"):
            with self.assertRaisesRegex(TypeError, "doesn't store a binary blob"):
                root.array(name).read_bytes()

    def test_mismatching_single_chunk_records(self):
        """Single chunk records not matching their array (e.g., truncated by other writers) aren't read as views (of
        wrong shape): reads raise"""

        def fill(root: nodes.Group) -> None:
            root.create_array("a", np.arange(6, dtype=np.float32), compression=None)

        self._root(fill)
        source = IndexedTarArchive(self.path)
        path = os.path.join(self._tmp.name, "truncated.itar")
        with IndexedTarArchive(path, mode="w") as copy:
            for key in source.keys():
                chunk = key.startswith("a/") and not key.endswith((".zarray", ".zattrs", "zarr.json"))
                copy[key] = source[key][:-4] if chunk else source[key]
        with mock.patch.object(Backend, "chunk_view", wraps=Backend.chunk_view) as chunk_view:
            with self.assertRaises(ValueError):
                IndexedTarZarrStore(path).open_root_group().array("a").read()
        # (the truncated record is rejected, not viewed)
        self.assertEqual([len(call.args[0]) for call in chunk_view.call_args_list][:1], [20])
        self.assertIsNone(Backend.chunk_view(bytes(20), np.dtype(np.float32), (6,)))
        self.assertIsNone(Backend.chunk_view(bytes(28), np.dtype(np.float32), (6,)))
        view = Backend.chunk_view(bytes(24), np.dtype(np.float32), (6,))
        assert view is not None
        self.assertEqual((view.shape, view.flags.writeable), ((6,), False))

    def test_unsupported_compressors(self):
        """Compression of arrays not compressed with blosc can't be represented"""
        with tempfile.TemporaryDirectory() as tmp:
            zarr_format = self.zarr_format
            root = ZarrStoreAdapter.directory(tmp).create_root_group({})
            group: Any = root.zarr  # (zarr-python 2 / 3 APIs: untyped)
            if ZARR_PYTHON_3:
                codecs: Any = importlib.import_module("zarr.codecs")  # (zarr-python 3 only)
                compressors: Any = numcodecs.Zstd() if zarr_format == 2 else codecs.ZstdCodec()
                group.create_array("zstd", data=np.arange(4), compressors=compressors)
            else:
                group.create_dataset("zstd", data=np.arange(4), compressor=numcodecs.Zstd())
            with self.assertRaisesRegex(ValueError, "Unsupported compressor"):
                _ = root.array("zstd").compression
            self.assertEqual(group["zstd"][...].tolist(), [0, 1, 2, 3])  # (readable nonetheless)

    def test_unsupported_consolidated_metadata_formats(self):
        def fill(root: nodes.Group) -> None:
            root.create_array("a", np.arange(3))

        self._root(fill)
        if self.zarr_format == 3:
            self.skipTest("(zarr format v3: the consolidated metadata is the root document)")
        source = IndexedTarArchive(self.path)
        path = os.path.join(self._tmp.name, "future.itar")
        with IndexedTarArchive(path, mode="w") as copy:
            for key in source.keys():
                if key == V2_CONSOLIDATED_METADATA_KEY:
                    document = decode_cbor_xz(source[key])
                    copy[key] = encode_cbor_xz({**document, "zarr_consolidated_format": 2})
                else:
                    copy[key] = source[key]
        with self.assertRaisesRegex(Exception, "consolidated metadata format: 2"):
            IndexedTarZarrStore(path).open_root_group(consolidated=True)
        IndexedTarZarrStore(path).open_root_group(consolidated=False)  # (node metadata records are still valid)

    def test_archive_adapter_errors(self):
        """The zarr-python store of archives: records of archives opened for reading can't be written, records can't
        be deleted (records of archives are write-once)"""

        def fill(root: nodes.Group) -> None:
            root.create_array("a", np.arange(3))

        self._root(fill)
        backend = nodes._backend
        # (the zarr-python store of the installed zarr-python version: untyped)
        reading: Any = backend.indexed_tar_store(IndexedTarArchive(self.path))
        writing: Any = backend.indexed_tar_store(IndexedTarArchive(os.path.join(self._tmp.name, "w.itar"), mode="w"))
        if ZARR_PYTHON_3:
            zarr3: Any = importlib.import_module("ncore.impl.data.backend_zarr3")  # (zarr-python 3 only)
            run = zarr3.run
            value = zarr3.default_buffer_prototype().buffer.from_bytes(b"x")
            with self.assertRaises(ValueError):  # (zarr-python's read-only store error)
                run(reading.set("new", value))
            run(writing.set("b", value))
            with self.assertRaises(NotImplementedError):
                run(writing.delete("b"))
            with self.assertRaises(NotImplementedError):
                run(writing.delete_dir(""))
            run(writing.delete("missing"))  # (no-op for records that don't exist)
            run(writing.delete_dir("missing"))
        else:
            with self.assertRaises(getattr(zarr.errors, "ReadOnlyError")):  # (zarr-python 2 only)
                reading["new"] = b"x"
            writing["b"] = b"x"
            with self.assertRaises(NotImplementedError):
                del writing["b"]
        writing.close()

    @unittest.skipIf(ZARR_PYTHON_3, "zarr-python 2 backend")
    def test_zarr_python_2_backend(self):
        from ncore.impl.data import backend_zarr2

        with self.assertRaisesRegex(ValueError, "only supports zarr format v2"):
            backend_zarr2.backend.create_root_group(zarr.storage.MemoryStore(), 3, {})
        with self.assertRaisesRegex(TypeError, "Not a zarr-python 2 store"):
            ZarrStoreAdapter(cast(Any, object()))  # (untyped callers: stores are also validated at runtime)


class TestEncoding(unittest.TestCase):
    def test_round_trips(self):
        document = {"a": [1, 2.5, None, True], "b": {"c": "d"}, "e": b"bytes"}
        self.assertEqual(decode_cbor_xz(encode_cbor_xz(document)), document)
        with self.assertRaises(Exception):
            decode_cbor_xz(b"not encoded")


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
