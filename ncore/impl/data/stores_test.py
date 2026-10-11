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
from typing import Any, Optional
from unittest import mock

import numpy as np
import parameterized
import zarr

from ncore.impl.data.encoding import decode_cbor_xz
from ncore.impl.data.stores import (
    CONSOLIDATED_METADATA_KEY,
    ConsolidatedCompressedMetadataStore,
    IndexedTarStore,
    ZarrIndexedTarStore,
    consolidate_compressed_metadata,
    open_compressed_consolidated,
)


#: None / values < 512 (separate reads of index header and index), 1 KiB, 1 MiB (default), edge cases around 512
INDEX_TAIL_READ_SIZES = [None, 1, 512, 513, 1 << 10, 1 << 20]


def _fill_reference(root: zarr.Group) -> None:
    rng = np.random.default_rng(0)
    root.attrs.put({"some": "thing"})
    root.create_dataset("foo", data=rng.random((3, 3, 3)), chunks=(2, 3, 3))
    sub = root.create_group("subgroup")
    sub.attrs.put({"other": [1, 2, 3]})
    sub.create_dataset("foo", data=rng.random((5, 5, 5)))
    sub.create_dataset("blob", data=np.asarray(b"\x89PNG\x00some binary data"), compressor=None)


def _check_reference(test: unittest.TestCase, root: zarr.Group) -> None:
    rng = np.random.default_rng(0)
    np.testing.assert_array_equal(root["foo"][...], rng.random((3, 3, 3)))
    np.testing.assert_array_equal(root["subgroup/foo"][...], rng.random((5, 5, 5)))
    test.assertEqual(bytes(root["subgroup/blob"][()]), b"\x89PNG\x00some binary data")
    test.assertEqual(root.attrs.asdict(), {"some": "thing"})
    test.assertEqual(root["subgroup"].attrs.asdict(), {"other": [1, 2, 3]})


def _write_itar(path: str, consolidate: bool = True) -> None:
    with IndexedTarStore(path, mode="w") as store:
        zarr_store = ZarrIndexedTarStore(store)
        _fill_reference(zarr.group(store=zarr_store))
        if consolidate:
            consolidate_compressed_metadata(zarr_store)


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
            self.assertEqual({m.name: unpack(tar.extractfile(m)) for m in tar.getmembers()}, records)

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


class TestZarrIndexedTarStore(unittest.TestCase):
    """zarr hierarchies in indexed tar archives"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "store.itar")

    def tearDown(self):
        self._tmp.cleanup()

    @parameterized.parameterized.expand([(size,) for size in INDEX_TAIL_READ_SIZES])
    def test_reserialization(self, index_tail_read_size: Optional[int]):
        _write_itar(self.path, consolidate=False)

        store = IndexedTarStore(self.path, index_tail_read_size=index_tail_read_size)
        root = zarr.open_group(store=ZarrIndexedTarStore(store), mode="r")
        _check_reference(self, root)

        store.reload_resources()
        _check_reference(self, root)

    @parameterized.parameterized.expand([(size,) for size in INDEX_TAIL_READ_SIZES])
    def test_compressed_consolidated(self, index_tail_read_size: Optional[int]):
        _write_itar(self.path)

        store = IndexedTarStore(self.path, index_tail_read_size=index_tail_read_size)
        with mock.patch.object(
            IndexedTarStore, "__getitem__", autospec=True, side_effect=IndexedTarStore.__getitem__
        ) as getitem:
            root = open_compressed_consolidated(store, mode="r")
            _check_reference(self, root)
        # node metadata is read from the compressed consolidated metadata only
        read_keys = [call.args[1] for call in getitem.call_args_list]
        self.assertIn(CONSOLIDATED_METADATA_KEY, read_keys)
        self.assertEqual([key for key in read_keys if key.endswith((".zarray", ".zgroup", ".zattrs"))], [])
        # (zarr group -> compressed consolidated metadata layer -> zarr-python adapter -> indexed tar store)
        consolidated_store = root.store
        assert isinstance(consolidated_store, ConsolidatedCompressedMetadataStore)
        assert isinstance(zarr_store := consolidated_store.store, ZarrIndexedTarStore)
        self.assertIs(zarr_store.store, store)

        store.reload_resources()
        _check_reference(self, root)

    def test_records_match_zarr_directory_stores(self):
        """Archives store the same records as zarr's directory stores (byte-identical)"""
        _write_itar(self.path)

        directory = os.path.join(self._tmp.name, "store.zarr")
        directory_store = zarr.storage.DirectoryStore(directory)
        _fill_reference(zarr.group(store=directory_store))
        consolidate_compressed_metadata(directory_store)

        store = IndexedTarStore(self.path)
        self.assertEqual(sorted(store.keys()), sorted(directory_store.keys()))
        for key in store.keys():
            if key == CONSOLIDATED_METADATA_KEY:  # (documents ordered by the key order of the stores)
                self.assertEqual(decode_cbor_xz(store[key]), decode_cbor_xz(directory_store[key]))
            else:
                self.assertEqual(store[key], directory_store[key], key)

    def test_archives_of_ncore_19_8(self):
        """Archives are byte-identical to archives written by ncore 19.8 (sha256 of a fixed hierarchy written by
        ncore 19.8)"""
        with IndexedTarStore(self.path, mode="w") as store:
            zarr_store = ZarrIndexedTarStore(store)
            root = zarr.group(store=zarr_store)
            root.attrs.put({"some": "thing", "nested": {"a": [1, 2.5, None]}})
            root.create_dataset("foo", data=np.arange(27, dtype=np.float64).reshape(3, 3, 3), chunks=(2, 3, 3))
            sub = root.create_group("subgroup")
            sub.attrs.put({"other": [1, 2, 3]})
            sub.create_dataset("bar", data=np.arange(10, dtype=np.uint16), compressor=None)
            sub.create_dataset("blob", data=np.asarray(b"\x89PNG\x00data"), compressor=None)
            consolidate_compressed_metadata(zarr_store)

        digest = hashlib.sha256(Path(self.path).read_bytes()).hexdigest()
        self.assertIn(
            digest,
            {
                # (python's tarfile module encodes the device numbers of tar headers differently before python 3.9)
                "98852fe2314dc3c9680cd0997d6f914785f5f7d2e552c102ed71c9b400340226",  # python 3.8
                "016af4d3f0d4964c025832a4864b45ffe13cd68e74436da331086d255592a7de",  # python >= 3.9
            },
        )

    def test_empty(self):
        with IndexedTarStore(self.path, mode="w") as store:
            consolidate_compressed_metadata(ZarrIndexedTarStore(store))

        with self.assertRaises(zarr.errors.PathNotFoundError):
            open_compressed_consolidated(IndexedTarStore(self.path), mode="r")
        with self.assertRaises(zarr.errors.PathNotFoundError):
            zarr.open(store=ZarrIndexedTarStore(IndexedTarStore(self.path)), mode="r")

    def test_read_only(self):
        _write_itar(self.path)
        root = zarr.open_group(store=ZarrIndexedTarStore(IndexedTarStore(self.path)), mode="r")
        with self.assertRaises(zarr.errors.ReadOnlyError):
            root.create_group("new")


def unpack(file: Optional[Any]) -> bytes:
    assert file is not None
    return file.read()
