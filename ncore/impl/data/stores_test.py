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
import hashlib
import json
import os
import struct
import tarfile
import tempfile
import threading
import unittest
import zlib

from pathlib import Path
from typing import Any, Optional, Tuple, cast
from unittest import mock

import cbor2
import numpy as np
import parameterized
import zarr

from ncore.impl.data import nodes
from ncore.impl.data.encoding import decode_cbor_xz
from ncore.impl.data.itar import is_node_metadata_key
from ncore.impl.data.stores import (
    V2_CONSOLIDATED_METADATA_KEY,
    V3_CONSOLIDATED_METADATA_KEY,
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

ZARR_PYTHON_3 = int(zarr.__version__.split(".", 1)[0]) >= 3

#: All zarr formats (tests of zarr formats unsupported by the installed zarr-python version are skipped)
ZARR_FORMATS = (2, 3)


def _require(test: unittest.TestCase, zarr_format: int) -> None:
    if zarr_format not in nodes.SUPPORTED_ZARR_FORMATS:
        test.skipTest(f"zarr format v{zarr_format} requires zarr-python 3")


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


def _decode_v3_root(value: bytes) -> Any:
    return cbor2.loads(zlib.decompress(value))


def _write(store: StoreLike, consolidate: bool = True, zarr_format: int = 2) -> None:
    _fill_reference(create_root_group(store, _ROOT_ATTRIBUTES, zarr_format=zarr_format))
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

    def test_node_metadata_records(self):
        """Node metadata records can be rewritten while writing, they are appended when closing (parents first)"""
        with IndexedTarStore(self.path, mode="w") as store:
            store["g/a/zarr.json"] = b"a v1"
            store["g/a/c/0"] = b"chunk"
            store["zarr.json"] = b"root v1"
            store["g/a/zarr.json"] = b"a v2"  # rewritten
            store[".zattrs"] = b"attrs"
            self.assertEqual(store["g/a/zarr.json"], b"a v2")  # readable while writing
            self.assertEqual(len(store), 4)
            self.assertIn("zarr.json", store)
            with self.assertRaises(ValueError):  # other records are write-once
                store["g/a/c/0"] = b"again"
        with self.assertRaises(ValueError):  # closed
            store["zarr.json"] = b"root v2"

        store = IndexedTarStore(self.path)
        self.assertEqual(store.keys(), ["g/a/c/0", ".zattrs", "zarr.json", "g/a/zarr.json"])
        self.assertEqual(store["g/a/zarr.json"], b"a v2")
        self.assertEqual(store["zarr.json"], b"root v1")
        with tarfile.open(self.path) as tar:  # (written once each)
            self.assertEqual([m.name for m in tar.getmembers()], store.keys())

    def test_node_metadata_keys(self):
        for key in (".zarray", ".zgroup", ".zattrs", "zarr.json", "a/b/.zarray", "a/zarr.json"):
            self.assertTrue(is_node_metadata_key(key), key)
        for key in ("zarr.json.cbor.zlib", ".zmetadata.cbor.xz", "a/0.0", "a/c/0/0", "a/x.zarray", "a/zarr.json/0"):
            self.assertFalse(is_node_metadata_key(key), key)


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
        [
            (store_type, consolidated, zarr_format)
            for store_type in ("itar", "directory")
            for consolidated in (True, False)
            for zarr_format in ZARR_FORMATS
        ]
    )
    def test_reserialization(self, store_type: str, consolidated: bool, zarr_format: int):
        _require(self, zarr_format)
        _write(self._store(store_type, "w"), zarr_format=zarr_format)

        store = self._store(store_type, "r")
        root = open_store(store, consolidated)
        _check_reference(self, root)
        self.assertEqual(root.zarr_format, zarr_format)
        self.assertEqual(root.array("subgroup/foo").zarr_format, zarr_format)

        if isinstance(store, IndexedTarStore):
            self.assertIs(group_store(root), store)
            self.assertIs(group_store(root.group("subgroup")), store)
            store.reload_resources()
            _check_reference(self, root)

    @parameterized.parameterized.expand(
        [(size, zarr_format) for size in INDEX_TAIL_READ_SIZES for zarr_format in ZARR_FORMATS]
    )
    def test_consolidated_open_reads_no_node_metadata(self, index_tail_read_size: Optional[int], zarr_format: int):
        _require(self, zarr_format)
        _write(self._store("itar", "w"), zarr_format=zarr_format)

        store = IndexedTarStore(self.itar_path, index_tail_read_size=index_tail_read_size)
        with mock.patch.object(
            IndexedTarStore, "__getitem__", autospec=True, side_effect=IndexedTarStore.__getitem__
        ) as getitem:
            _check_reference(self, open_store(store, consolidated=True))
        read_keys = [call.args[1] for call in getitem.call_args_list]
        self.assertIn(V2_CONSOLIDATED_METADATA_KEY if zarr_format == 2 else V3_CONSOLIDATED_METADATA_KEY, read_keys)
        self.assertEqual([key for key in read_keys if is_node_metadata_key(key)], [])

    @parameterized.parameterized.expand(
        [(store_type, zarr_format) for store_type in ("itar", "directory") for zarr_format in ZARR_FORMATS]
    )
    def test_missing_consolidated_metadata(self, store_type: str, zarr_format: int):
        _require(self, zarr_format)
        _write(self._store(store_type, "w"), consolidate=False, zarr_format=zarr_format)
        with self.assertRaises(KeyError):
            open_store(self._store(store_type, "r"), consolidated=True)
        _check_reference(self, open_store(self._store(store_type, "r"), consolidated=False))

    @parameterized.parameterized.expand([(zarr_format,) for zarr_format in ZARR_FORMATS])
    def test_unconsolidated_open_reads_node_metadata(self, zarr_format: int):
        """Hierarchies opened without consolidated metadata are resolved from their node metadata records (e.g., nodes
        added after consolidating, as component writers extending existing directory stores)"""
        _require(self, zarr_format)
        root = create_root_group(store := self._store("directory", "w"), _ROOT_ATTRIBUTES, zarr_format=zarr_format)
        root.create_array("before", np.zeros(2))
        consolidate_store(store)
        root.create_array("after", np.ones(2))

        self.assertEqual(
            sorted(open_store(self._store("directory", "r"), consolidated=False).members()), ["after", "before"]
        )
        self.assertEqual(sorted(open_store(self._store("directory", "r"), consolidated=True).members()), ["before"])

    @parameterized.parameterized.expand([(zarr_format,) for zarr_format in ZARR_FORMATS])
    def test_stores_are_identical(self, zarr_format: int):
        """Indexed tar archives and directories store the same records"""
        _require(self, zarr_format)
        _write(self._store("itar", "w"), zarr_format=zarr_format)
        _write(self._store("directory", "w"), zarr_format=zarr_format)

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
            elif key == V3_CONSOLIDATED_METADATA_KEY:
                self.assertEqual(_decode_v3_root(archive[key]), _decode_v3_root(files[key]))
            else:
                self.assertEqual(archive[key], files[key], key)

    def test_directory_stores_are_not_truncated(self):
        Path(self.directory_path).mkdir()
        (Path(self.directory_path) / "other").write_bytes(b"other")
        _write(self._store("directory", "w"))
        self.assertEqual((Path(self.directory_path) / "other").read_bytes(), b"other")

    #: sha256 of the records of the hierarchy of :meth:`_write_golden` written by ncore 19.8 (zarr-python 2)
    GOLDEN_RECORDS = {
        ".zattrs": "0e326aaa71db8ecc9051e106a4a70695515ad95a95ff348445699fd79f907623",
        ".zgroup": "2383746e67b4bcc2762b3f100f06c3fa2d5f149ab5a8e5da5d33521464a01959",
        ".zmetadata.cbor.xz": "245a85421d308c67ea2e452746f0628fe0b3bb3a337af3fec3b46e75b044a4cc",
        "foo/.zarray": "28db182434d2702c2ccc384e0c1d386359f308bd275821655377b09a40c610fa",
        "foo/0.0.0": "e9361a616760d41ac367d54b2349f18184e8a226563ca34047e691cf393a9610",
        "foo/1.0.0": "8611647425ac52bdfd586f7342bdb768d363ae139aadafa7130c7a4368015614",
        "subgroup/.zattrs": "b638e55387a56bef81862ccd44889e0068a67b04cf00a4c4e90783cf2940e160",
        "subgroup/.zgroup": "2383746e67b4bcc2762b3f100f06c3fa2d5f149ab5a8e5da5d33521464a01959",
        "subgroup/bar/.zarray": "6c83cd89670edfe4f9f12cf48f7de5f55ed9cefbfa38b168f1ab14dfb082c449",
        "subgroup/bar/0": "3c7acfa845b57df9e3a46779d4f17c7eb9d697d63dd8b2c30c176c6fec90051b",
        "subgroup/blob/.zarray": "dcb9ebabd3bb2d803fd59453b16ba414f6d55d4451ba597170b4a90fae4e749b",
        "subgroup/blob/0": "5399a4c8461dd3fd590854ff73ec4e88db9462165e9160375113dfec0c741e0c",
    }

    def _write_golden(self) -> IndexedTarStore:
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
        return IndexedTarStore(self.itar_path)

    @unittest.skipIf(ZARR_PYTHON_3, "zarr-python 2 metadata serialization")
    def test_archives_of_ncore_19_8(self):
        """Records are byte-identical to records written by ncore 19.8, archives are deterministic (node metadata
        records are appended when closing archives)"""
        store = self._write_golden()
        self.assertEqual({key: hashlib.sha256(store[key]).hexdigest() for key in store}, self.GOLDEN_RECORDS)
        # data records in creation order, followed by node metadata records (parents first)
        self.assertEqual(
            store.keys(),
            [
                "foo/0.0.0",
                "foo/1.0.0",
                "subgroup/bar/0",
                "subgroup/blob/0",
                ".zmetadata.cbor.xz",
                ".zattrs",
                ".zgroup",
                "foo/.zarray",
                "subgroup/.zattrs",
                "subgroup/.zgroup",
                "subgroup/bar/.zarray",
                "subgroup/blob/.zarray",
            ],
        )
        digest = hashlib.sha256(Path(self.itar_path).read_bytes()).hexdigest()
        self.assertIn(
            digest,
            {
                # (python's tarfile module encodes the device numbers of tar headers differently before python 3.9)
                "f2f98df32a018a2f178f3561e21bb14eb3bba9da51e00956bb09ed53177000ea",  # python 3.8
                "10c02f6fdbaad3c048af23c47eb50fcb539d660d1487df6c42c6e14c14b98a13",  # python >= 3.9
            },
        )

    def test_archives_equivalent_to_ncore_19_8(self):
        """Chunk records are byte-identical to chunk records written by ncore 19.8, node metadata documents are
        equivalent (zarr-python 3 serializes them differently, and stores empty attributes of arrays)"""
        store = self._write_golden()
        # (zarr-python 3 writes the chunks of an array concurrently, in any order)
        chunks = sorted(key for key in store if not is_node_metadata_key(key) and key != ".zmetadata.cbor.xz")
        self.assertEqual(chunks, ["foo/0.0.0", "foo/1.0.0", "subgroup/bar/0", "subgroup/blob/0"])
        self.assertEqual(
            {key: hashlib.sha256(store[key]).hexdigest() for key in chunks},
            {key: self.GOLDEN_RECORDS[key] for key in chunks},
        )

        metadata = decode_cbor_xz(store[".zmetadata.cbor.xz"])["metadata"]
        self.assertEqual(metadata, {key: json.loads(store[key]) for key in store if is_node_metadata_key(key)})
        self.assertEqual(metadata[".zattrs"], {"some": "thing", "nested": {"a": [1, 2.5, None]}})
        self.assertEqual(metadata["subgroup/.zattrs"], {"other": [1, 2, 3]})
        self.assertEqual(metadata.get("foo/.zattrs", {}), {})
        self.assertEqual(
            {key: value for key, value in metadata["foo/.zarray"].items() if key != "dimension_separator"},
            {
                "shape": [3, 3, 3],
                "chunks": [2, 3, 3],
                "dtype": "<f8",
                "fill_value": 0.0,
                "order": "C",
                "filters": None,
                "compressor": {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1, "blocksize": 0},
                "zarr_format": 2,
            },
        )
        self.assertEqual(metadata["subgroup/blob/.zarray"]["dtype"], "|S9")
        self.assertEqual(metadata["subgroup/blob/.zarray"]["shape"], [])

    @parameterized.parameterized.expand([("itar",), ("directory",)])
    def test_chunks_of_fill_values_are_stored(self, store_type: str):
        """Chunks equal to the fill value are stored (zarr-python 3 omits them by default, zarr-python 2 doesn't)"""
        root = create_root_group(store := self._store(store_type, "w"), {})
        root.create_array("zeros", np.zeros((4, 3), dtype=np.float32), chunks=(2, 3))
        root.create_bytes_array("blob", b"\x00\x00")
        consolidate_store(store)
        close_store(store)

        if store_type == "itar":
            keys = set(IndexedTarStore(self.itar_path).keys())
        else:
            keys = {
                str(p.relative_to(self.directory_path)) for p in Path(self.directory_path).rglob("*") if p.is_file()
            }
        self.assertLessEqual({"zeros/0.0", "zeros/1.0", "blob/0"}, keys)

    @parameterized.parameterized.expand([("itar",), ("directory",)])
    def test_zarr_format_v3_layout(self, store_type: str):
        """The root zarr.json of consolidated zarr format v3 hierarchies is stored compressed only (CBOR + zlib, including
        the consolidated metadata), all other node metadata records as written by zarr-python"""
        _require(self, 3)
        _write(self._store(store_type, "w"), zarr_format=3)
        if store_type == "itar":
            records = {key: IndexedTarStore(self.itar_path)[key] for key in IndexedTarStore(self.itar_path)}
        else:
            root_path = Path(self.directory_path)
            records = {str(p.relative_to(root_path)): p.read_bytes() for p in root_path.rglob("*") if p.is_file()}

        self.assertIn(V3_CONSOLIDATED_METADATA_KEY, records)
        self.assertNotIn(V2_CONSOLIDATED_METADATA_KEY, records)
        self.assertFalse(any(key.endswith((".zarray", ".zgroup", ".zattrs")) for key in records))
        node_documents = {
            key[: -len("/zarr.json")]: json.loads(value) for key, value in records.items() if key.endswith("/zarr.json")
        }
        self.assertEqual(sorted(node_documents), ["foo", "subgroup", "subgroup/blob", "subgroup/empty", "subgroup/foo"])

        root = _decode_v3_root(records[V3_CONSOLIDATED_METADATA_KEY])
        self.assertEqual((root["zarr_format"], root["node_type"], root["attributes"]), (3, "group", _ROOT_ATTRIBUTES))
        consolidated = root["consolidated_metadata"]
        self.assertEqual((consolidated["kind"], consolidated["must_understand"]), ("inline", False))
        self.assertEqual(consolidated["metadata"], node_documents)

        # binary blobs are uint8 arrays, compression is the blosc codec of zarr format v3
        self.assertEqual(
            (node_documents["subgroup/blob"]["data_type"], node_documents["subgroup/blob"]["shape"]), ("uint8", [21])
        )
        self.assertEqual([codec["name"] for codec in node_documents["subgroup/foo"]["codecs"]], ["bytes", "blosc"])
        self.assertEqual(
            {k: v for k, v in node_documents["subgroup/foo"]["codecs"][1]["configuration"].items() if k != "typesize"},
            {"cname": "lz4", "clevel": 5, "shuffle": "bitshuffle", "blocksize": 0},
        )

    def test_zarr_format_v3_requires_zarr_python_3(self):
        if ZARR_PYTHON_3:
            self.skipTest("zarr-python 2 only")
        with self.assertRaisesRegex(ValueError, "zarr format v3 is not supported"):
            create_root_group(self._store("itar", "w"), {}, zarr_format=3)

    def test_unsupported_zarr_formats(self):
        with self.assertRaisesRegex(ValueError, "zarr format v4 is not supported"):
            create_root_group(self._store("directory", "w"), {}, zarr_format=4)

    def test_empty(self):
        """Stores without a root group can be consolidated, but not opened"""
        store = IndexedTarStore(self.itar_path, mode="w")
        consolidate_store(store)
        close_store(store)
        with self.assertRaises(Exception):  # (zarr-python's error of missing groups)
            open_store(IndexedTarStore(self.itar_path), consolidated=True)

    @parameterized.parameterized.expand([(zarr_format,) for zarr_format in ZARR_FORMATS])
    def test_read_only(self, zarr_format: int):
        _require(self, zarr_format)
        _write(self._store("itar", "w"), zarr_format=zarr_format)
        root = open_store(IndexedTarStore(self.itar_path))
        with self.assertRaises(Exception):
            root.create_group("new")


class TestHierarchies(unittest.TestCase):
    """Hierarchies created in arbitrary order (e.g., by multiple component writers)"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "store.itar")

    def tearDown(self):
        self._tmp.cleanup()

    @parameterized.parameterized.expand(
        [(consolidated, zarr_format) for consolidated in (True, False) for zarr_format in ZARR_FORMATS]
    )
    def test_interleaved_hierarchy(self, consolidated: bool, zarr_format: int):
        _require(self, zarr_format)
        store = IndexedTarStore(self.path, mode="w")
        root = create_root_group(store, {}, zarr_format=zarr_format)
        a, b = root.create_group("a"), root.create_group("b")
        for i in range(3):  # members of a and b created alternately
            a.create_array(f"x{i}", np.full(2, i))
            b.create_group(f"g{i}").create_array("y", np.full(2, 10 + i))
        root.create_array("z", np.zeros(1))
        consolidate_store(store)
        close_store(store)

        root = open_store(IndexedTarStore(self.path), consolidated=consolidated)
        self.assertEqual(sorted(root.members()), ["a", "b", "z"])
        self.assertEqual(sorted(root.group("a").members()), ["x0", "x1", "x2"])
        self.assertEqual(sorted(root.group("b").members()), ["g0", "g1", "g2"])
        for i in range(3):
            np.testing.assert_array_equal(root.array(f"a/x{i}").read(), [i, i])
            np.testing.assert_array_equal(root.group("b").group(f"g{i}").array("y").read(), [10 + i] * 2)


@unittest.skipUnless(ZARR_PYTHON_3, "zarr-python 3 backend")
class TestZarr3Backend(unittest.TestCase):
    """Execution of zarr-python 3 coroutines, and the zarr-python 3 store of indexed tar archives"""

    def test_coroutines_run_on_the_calling_thread(self):
        from ncore.impl.data import _zarr3

        async def thread() -> int:
            return threading.get_ident()

        def run_in_thread() -> int:
            return _zarr3.run(thread())

        self.assertEqual(_zarr3.run(thread()), threading.get_ident())
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            for future in [pool.submit(run_in_thread) for _ in range(4)]:
                self.assertNotEqual(future.result(), threading.get_ident())

        # offloaded work (e.g., of codecs) runs inline
        async def offloaded() -> int:
            return await asyncio.get_running_loop().run_in_executor(None, threading.get_ident)

        self.assertEqual(_zarr3.run(offloaded()), threading.get_ident())

    def test_coroutines_within_event_loops(self):
        from ncore.impl.data import _zarr3

        async def value() -> int:
            return 42

        async def main() -> int:
            return _zarr3.run(value())  # (runs on zarr-python's I/O loop)

        self.assertEqual(asyncio.run(main()), 42)

    @unittest.skipUnless(hasattr(os, "fork"), "requires fork")
    def test_forked_processes(self):
        """Forked processes don't reuse the event loop of the forking thread (its selector is shared with the parent)"""
        from ncore.impl.data import _zarr3

        async def loop_and_pid() -> Tuple[int, int]:
            return id(asyncio.get_running_loop()), os.getpid()

        parent_loop, _ = _zarr3.run(loop_and_pid())  # (creates the loop of this thread)
        read_fd, write_fd = os.pipe()
        if (pid := os.fork()) == 0:  # child (continues on this thread)
            try:
                child_loop, child_pid = _zarr3.run(loop_and_pid())
                os.write(write_fd, b"ok" if child_loop != parent_loop and child_pid == os.getpid() else b"reused")
            finally:
                os._exit(0)
        os.close(write_fd)
        result = os.read(read_fd, 16)
        os.close(read_fd)
        os.waitpid(pid, 0)
        self.assertEqual(result, b"ok")
        self.assertEqual(_zarr3.run(loop_and_pid())[0], parent_loop)  # (the parent keeps its loop)

        # the parent's loop is still woken by other threads (children don't close the selector shared with the parent)
        async def woken_by_thread() -> int:
            loop, future = asyncio.get_running_loop(), asyncio.get_running_loop().create_future()
            timer = threading.Timer(0.05, lambda: loop.call_soon_threadsafe(future.set_result, 42))
            timer.start()
            return await asyncio.wait_for(future, timeout=5)

        self.assertEqual(_zarr3.run(woken_by_thread()), 42)

    def test_byte_ranges(self):
        from ncore.impl.data import _zarr3
        from ncore.impl.data._zarr3 import (
            OffsetByteRequest,
            RangeByteRequest,
            SuffixByteRequest,
            default_buffer_prototype,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "store.itar")
            with IndexedTarStore(path, mode="w") as store:
                store["r"] = b"0123456789"
            tar_store = _zarr3._TarStore(IndexedTarStore(path))

            def get(byte_range: Any) -> Optional[bytes]:
                value = _zarr3.run(tar_store.get("r", default_buffer_prototype(), byte_range))
                return None if value is None else value.to_bytes()

            self.assertEqual(get(None), b"0123456789")
            self.assertEqual(get(RangeByteRequest(2, 5)), b"234")
            self.assertEqual(get(RangeByteRequest(8, 20)), b"89")
            self.assertEqual(get(OffsetByteRequest(7)), b"789")
            self.assertEqual(get(SuffixByteRequest(3)), b"789")
            self.assertEqual(get(SuffixByteRequest(30)), b"0123456789")
            self.assertIsNone(_zarr3.run(tar_store.get("missing", default_buffer_prototype())))


@parameterized.parameterized_class(("zarr_format",), [(zarr_format,) for zarr_format in ZARR_FORMATS])
class TestNodes(unittest.TestCase):
    """Typed node API"""

    zarr_format: int

    def setUp(self):
        _require(self, self.zarr_format)
        self._tmp = tempfile.TemporaryDirectory()
        self.root = create_root_group(
            open_directory_store(self._tmp.name),
            {"s": "x", "i": 3, "b": True, "l": [1], "d": {"k": 1}},
            zarr_format=self.zarr_format,
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

    def test_bytes_arrays(self):
        for data in (b"\x89PNG\x00binary", b"", b"\x00"):
            array = self.root.create_bytes_array(f"blob{len(data)}", data, attributes={"format": "png"})
            self.assertEqual(array.zarr_format, self.zarr_format)
            self.assertEqual(array.read_bytes(), data.rstrip(b"\x00") if self.zarr_format == 2 else data)
            self.assertEqual(array.attrs, {"format": "png"})
            self.assertIsNone(array.compression)
        # (zarr format v2: fixed-length bytes, as ncore <= 19.8, which strip trailing NUL bytes)
        self.assertEqual(self.root.array("blob11").dtype, np.dtype("S11") if self.zarr_format == 2 else np.uint8)

    @parameterized.parameterized.expand(
        [(None,), (nodes.BloscCompression(),), (nodes.BloscCompression("zstd", 3, "shuffle", 256),)]
    )
    def test_compression(self, compression: Optional[nodes.BloscCompression]):
        array = self.root.create_array("a", np.arange(1000, dtype=np.int32), compression=compression)
        self.assertEqual(array.compression, compression)
        np.testing.assert_array_equal(array.read(), np.arange(1000))

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
