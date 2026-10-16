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
import importlib
import json
import os
import struct
import tarfile
import tempfile
import threading
import unittest
import zlib

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast
from unittest import mock

import cbor2
import numcodecs
import numpy as np
import parameterized
import zarr
import zarr.errors
import zarr.storage

from ncore.impl.data import nodes
from ncore.impl.data.backend import V2_CONSOLIDATED_METADATA_KEY, V3_CONSOLIDATED_METADATA_KEY, Backend
from ncore.impl.data.encoding import decode_cbor_xz, encode_cbor_xz
from ncore.impl.data.itar import IndexedTarArchive, is_node_metadata_key
from ncore.impl.data.stores import IndexedTarZarrStore, Store, ZarrStoreAdapter


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


def _write(store: Store, consolidate: bool = True, zarr_format: int = 2) -> None:
    with store:
        _fill_reference(store.create_root_group(_ROOT_ATTRIBUTES, zarr_format=zarr_format))
        if consolidate:
            store.consolidate()


class TestIndexedTarArchive(unittest.TestCase):
    """Records of indexed tar archives"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "store.itar")

    def tearDown(self):
        self._tmp.cleanup()

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

    def test_node_metadata_records(self):
        """Node metadata records can be rewritten while writing, they are appended when closing (parents first)"""
        with IndexedTarArchive(self.path, mode="w") as archive:
            archive["g/a/zarr.json"] = b"a v1"
            archive["g/a/c/0"] = b"chunk"
            archive["zarr.json"] = b"root v1"
            archive["g/a/zarr.json"] = b"a v2"  # rewritten
            archive[".zattrs"] = b"attrs"
            self.assertEqual(archive["g/a/zarr.json"], b"a v2")  # readable while writing
            self.assertEqual(len(archive), 4)
            self.assertIn("zarr.json", archive)
            with self.assertRaises(ValueError):  # other records are write-once
                archive["g/a/c/0"] = b"again"
        with self.assertRaises(ValueError):  # closed
            archive["zarr.json"] = b"root v2"

        archive = IndexedTarArchive(self.path)
        self.assertEqual(archive.keys(), ["g/a/c/0", ".zattrs", "zarr.json", "g/a/zarr.json"])
        self.assertEqual(archive["g/a/zarr.json"], b"a v2")
        self.assertEqual(archive["zarr.json"], b"root v1")
        with tarfile.open(self.path) as tar:  # (written once each)
            self.assertEqual([m.name for m in tar.getmembers()], archive.keys())

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

    def _store(self, store_type: str, mode: str) -> Store:
        if store_type == "itar":
            return IndexedTarZarrStore(self.itar_path, mode="w" if mode == "w" else "r")
        return ZarrStoreAdapter.directory(self.directory_path)

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

        with self._store(store_type, "r") as store:
            root = store.open_root_group(consolidated)
            _check_reference(self, root)
            self.assertEqual(root.zarr_format, zarr_format)
            self.assertEqual(root.array("subgroup/foo").zarr_format, zarr_format)

            self.assertIs(root.store, store)
            self.assertIs(root.group("subgroup").store, store)
            store.reload_resources()
            _check_reference(self, root)

    @parameterized.parameterized.expand(
        [(size, zarr_format) for size in INDEX_TAIL_READ_SIZES for zarr_format in ZARR_FORMATS]
    )
    def test_consolidated_open_reads_no_node_metadata(self, index_tail_read_size: Optional[int], zarr_format: int):
        _require(self, zarr_format)
        _write(self._store("itar", "w"), zarr_format=zarr_format)

        store = IndexedTarZarrStore(self.itar_path, index_tail_read_size=index_tail_read_size)
        with mock.patch.object(
            IndexedTarArchive, "__getitem__", autospec=True, side_effect=IndexedTarArchive.__getitem__
        ) as getitem:
            _check_reference(self, store.open_root_group(consolidated=True))
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
            self._store(store_type, "r").open_root_group(consolidated=True)
        _check_reference(self, self._store(store_type, "r").open_root_group(consolidated=False))

    @parameterized.parameterized.expand([(zarr_format,) for zarr_format in ZARR_FORMATS])
    def test_unconsolidated_open_reads_node_metadata(self, zarr_format: int):
        """Hierarchies opened without consolidated metadata are resolved from their node metadata records (e.g., nodes
        added after consolidating, as component writers extending existing directory stores)"""
        _require(self, zarr_format)
        with self._store("directory", "w") as store:
            root = store.create_root_group(_ROOT_ATTRIBUTES, zarr_format=zarr_format)
            root.create_array("before", np.zeros(2))
            store.consolidate()
        root.create_array("after", np.ones(2))

        self.assertEqual(
            sorted(self._store("directory", "r").open_root_group(consolidated=False).members()), ["after", "before"]
        )
        self.assertEqual(sorted(self._store("directory", "r").open_root_group(consolidated=True).members()), ["before"])

    @parameterized.parameterized.expand([(zarr_format,) for zarr_format in ZARR_FORMATS])
    def test_stores_are_identical(self, zarr_format: int):
        """Indexed tar archives and directories store the same records"""
        _require(self, zarr_format)
        _write(self._store("itar", "w"), zarr_format=zarr_format)
        _write(self._store("directory", "w"), zarr_format=zarr_format)

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

    def _write_golden(self) -> IndexedTarArchive:
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
        return IndexedTarArchive(self.itar_path)

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
        with self._store(store_type, "w") as store:
            root = store.create_root_group({})
            root.create_array("zeros", np.zeros((4, 3), dtype=np.float32), chunks=(2, 3))
            root.create_bytes_array("blob", b"\x00\x00")
            store.consolidate()

        if store_type == "itar":
            keys = set(IndexedTarArchive(self.itar_path).keys())
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
            records = {key: IndexedTarArchive(self.itar_path)[key] for key in IndexedTarArchive(self.itar_path)}
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
            self._store("itar", "w").create_root_group({}, zarr_format=3)

    def test_unsupported_zarr_formats(self):
        with self.assertRaisesRegex(ValueError, "zarr format v4 is not supported"):
            self._store("directory", "w").create_root_group({}, zarr_format=4)

    def test_empty(self):
        """Stores without a root group can be consolidated, but not opened"""
        with IndexedTarZarrStore(self.itar_path, mode="w") as store:
            store.consolidate()
        with self.assertRaises(Exception):  # (zarr-python's error of missing groups)
            IndexedTarZarrStore(self.itar_path).open_root_group(consolidated=True)

    @parameterized.parameterized.expand([(zarr_format,) for zarr_format in ZARR_FORMATS])
    def test_read_only(self, zarr_format: int):
        _require(self, zarr_format)
        _write(self._store("itar", "w"), zarr_format=zarr_format)
        root = IndexedTarZarrStore(self.itar_path).open_root_group()
        with self.assertRaises(Exception):
            root.create_group("new")

    def test_remote_directory_stores_are_rejected(self):
        with self.assertRaises(ValueError):
            ZarrStoreAdapter.directory("memory://bucket/store.zarr")

    def test_other_zarr_stores(self):
        """Hierarchies in other zarr-python stores (e.g., in-memory stores)"""
        store = ZarrStoreAdapter(zarr.storage.MemoryStore())
        _write(store)
        _check_reference(self, store.open_root_group())


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
        with IndexedTarZarrStore(self.path, mode="w") as store:
            root = store.create_root_group({}, zarr_format=zarr_format)
            a, b = root.create_group("a"), root.create_group("b")
            for i in range(3):  # members of a and b created alternately
                a.create_array(f"x{i}", np.full(2, i))
                b.create_group(f"g{i}").create_array("y", np.full(2, 10 + i))
            root.create_array("z", np.zeros(1))
            store.consolidate()

        root = IndexedTarZarrStore(self.path).open_root_group(consolidated=consolidated)
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
        from ncore.impl.data import backend_zarr3

        async def thread() -> int:
            return threading.get_ident()

        def run_in_thread() -> int:
            return backend_zarr3.run(thread())

        self.assertEqual(backend_zarr3.run(thread()), threading.get_ident())
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            for future in [pool.submit(run_in_thread) for _ in range(4)]:
                self.assertNotEqual(future.result(), threading.get_ident())

        # offloaded work (e.g., of codecs) runs inline
        async def offloaded() -> int:
            return await asyncio.get_running_loop().run_in_executor(None, threading.get_ident)

        self.assertEqual(backend_zarr3.run(offloaded()), threading.get_ident())

    def test_coroutines_within_event_loops(self):
        from ncore.impl.data import backend_zarr3

        async def value() -> int:
            return 42

        async def main() -> int:
            return backend_zarr3.run(value())  # (runs on zarr-python's I/O loop)

        self.assertEqual(asyncio.run(main()), 42)

    @unittest.skipUnless(hasattr(os, "fork"), "requires fork")
    def test_forked_processes(self):
        """Forked processes don't reuse the event loop of the forking thread (its selector is shared with the parent)"""
        from ncore.impl.data import backend_zarr3

        async def loop_and_pid() -> Tuple[int, int]:
            return id(asyncio.get_running_loop()), os.getpid()

        parent_loop, _ = backend_zarr3.run(loop_and_pid())  # (creates the loop of this thread)
        read_fd, write_fd = os.pipe()
        if (pid := os.fork()) == 0:  # child (continues on this thread)
            try:
                child_loop, child_pid = backend_zarr3.run(loop_and_pid())
                os.write(write_fd, b"ok" if child_loop != parent_loop and child_pid == os.getpid() else b"reused")
            finally:
                os._exit(0)
        os.close(write_fd)
        result = os.read(read_fd, 16)
        os.close(read_fd)
        os.waitpid(pid, 0)
        self.assertEqual(result, b"ok")
        self.assertEqual(backend_zarr3.run(loop_and_pid())[0], parent_loop)  # (the parent keeps its loop)

        # the parent's loop is still woken by other threads (children don't close the selector shared with the parent)
        async def woken_by_thread() -> int:
            loop, future = asyncio.get_running_loop(), asyncio.get_running_loop().create_future()
            timer = threading.Timer(0.05, lambda: loop.call_soon_threadsafe(future.set_result, 42))
            timer.start()
            return await asyncio.wait_for(future, timeout=5)

        self.assertEqual(backend_zarr3.run(woken_by_thread()), 42)

    def test_byte_ranges(self):
        from ncore.impl.data import backend_zarr3
        from ncore.impl.data.backend_zarr3 import (
            OffsetByteRequest,
            RangeByteRequest,
            SuffixByteRequest,
            default_buffer_prototype,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "store.itar")
            with IndexedTarArchive(path, mode="w") as archive:
                archive["r"] = b"0123456789"
            tar_store = backend_zarr3._IndexedTarStore(IndexedTarArchive(path))

            def get(byte_range: Any) -> Optional[bytes]:
                value = backend_zarr3.run(tar_store.get("r", default_buffer_prototype(), byte_range))
                return None if value is None else value.to_bytes()

            self.assertEqual(get(None), b"0123456789")
            self.assertEqual(get(RangeByteRequest(2, 5)), b"234")
            self.assertEqual(get(RangeByteRequest(8, 20)), b"89")
            self.assertEqual(get(OffsetByteRequest(7)), b"789")
            self.assertEqual(get(SuffixByteRequest(3)), b"789")
            self.assertEqual(get(SuffixByteRequest(30)), b"0123456789")
            self.assertIsNone(backend_zarr3.run(tar_store.get("missing", default_buffer_prototype())))

    def test_archive_store_adapter(self):
        """The zarr-python 3 store of archives (as used by zarr-python)"""
        from ncore.impl.data import backend_zarr3
        from ncore.impl.data.backend_zarr3 import RangeByteRequest, default_buffer_prototype, run

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "store.itar")
            prototype = default_buffer_prototype()
            writing = backend_zarr3._IndexedTarStore(IndexedTarArchive(path, mode="w"))
            run(writing.set("a/0", prototype.buffer.from_bytes(b"0123")))
            run(writing.set_if_not_exists("a/0", prototype.buffer.from_bytes(b"other")))  # (exists: unchanged)
            run(writing.set_if_not_exists("a/b/1", prototype.buffer.from_bytes(b"45")))
            with self.assertRaises(NotImplementedError):
                writing.with_read_only(True)  # (the mode of archives can't be changed)
            self.assertIs(writing.with_read_only(False), writing)
            writing.close()
            self.assertEqual(IndexedTarArchive(path)["a/0"], b"0123")  # (not overwritten by set_if_not_exists)

            store = backend_zarr3._IndexedTarStore(IndexedTarArchive(path))
            self.assertTrue(store.read_only)
            self.assertEqual(store, backend_zarr3._IndexedTarStore(IndexedTarArchive(path)))
            self.assertEqual(len({store, backend_zarr3._IndexedTarStore(IndexedTarArchive(path))}), 1)

            async def collect(iterator: Any) -> List[str]:
                return sorted([key async for key in iterator])

            self.assertEqual(run(collect(store.list())), ["a/0", "a/b/1"])
            self.assertEqual(run(collect(store.list_prefix("a/b"))), ["a/b/1"])
            self.assertEqual(run(collect(store.list_dir("a"))), ["0", "b"])
            self.assertEqual(run(collect(store.list_dir(""))), ["a"])
            self.assertTrue(run(store.exists("a/0")))
            self.assertFalse(run(store.exists("a")))
            values = run(
                store.get_partial_values(
                    prototype, [("a/0", RangeByteRequest(1, 3)), ("missing", None), ("a/b/1", None)]
                )
            )
            self.assertEqual([None if v is None else v.to_bytes() for v in values], [b"12", None, b"45"])
            with self.assertRaises(ValueError):  # (zarr-python's read-only store error)
                run(store.set("new", prototype.buffer.from_bytes(b"x")))
            with self.assertRaises(ValueError):
                run(store.delete("a/0"))


@parameterized.parameterized_class(("zarr_format",), [(zarr_format,) for zarr_format in ZARR_FORMATS])
class TestNodes(unittest.TestCase):
    """Typed node API"""

    zarr_format: int

    def setUp(self):
        _require(self, self.zarr_format)
        self._tmp = tempfile.TemporaryDirectory()
        self.root = ZarrStoreAdapter.directory(self._tmp.name).create_root_group(
            {"s": "x", "i": 3, "b": True, "l": [1], "d": {"k": 1}}, zarr_format=self.zarr_format
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
        # (zero-sized chunk dimensions are clamped to 1, also of single chunks)
        self.assertEqual(self.root.create_array("s", np.zeros((0, 3))).zarr.chunks, (1, 3))
        self.assertEqual(self.root.create_array("s0", np.zeros((0, 3))).read().shape, (0, 3))
        self.assertEqual(self.root.create_array("auto", np.zeros(4096), chunks="auto").zarr.chunks, (4096,))
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
            root = store.create_root_group({}, zarr_format=zarr_format)
            root.create_bytes_array("blob", b"\x89PNG\x00data")
            root.create_array("single", np.arange(6, dtype=np.float32).reshape(2, 3), compression=None)
            root.create_array("compressed", np.arange(6, dtype=np.float32).reshape(2, 3))
            root.create_array("chunked", np.arange(6, dtype=np.float32), chunks=(4,), compression=None)
            store.consolidate()

    def _reads(self, root: nodes.Group) -> Dict[str, Tuple[Any, Dict[str, bytes]]]:
        """Read results, and the records read for them"""
        out: Dict[str, Tuple[Any, Dict[str, bytes]]] = {}
        getitem = IndexedTarArchive.__getitem__
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

            with mock.patch.object(IndexedTarArchive, "__getitem__", autospec=True, side_effect=recording_getitem):
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
            fill(store.create_root_group({}, zarr_format=zarr_format))
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
            root = ZarrStoreAdapter.directory(tmp).create_root_group({}, zarr_format=zarr_format)
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
    @parameterized.parameterized.expand([("xz",), ("zlib",)])
    def test_round_trips(self, codec: str):
        from ncore.impl.data import encoding

        document = {"a": [1, 2.5, None, True], "b": {"c": "d"}, "e": b"bytes"}
        encode, decode = getattr(encoding, f"encode_cbor_{codec}"), getattr(encoding, f"decode_cbor_{codec}")
        self.assertEqual(decode(encode(document)), document)
        with self.assertRaises(Exception):
            decode(b"not encoded")


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
