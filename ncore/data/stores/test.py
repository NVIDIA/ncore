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
import tempfile
import unittest

from pathlib import Path
from typing import Any, cast


class TestPackage(unittest.TestCase):
    def test_public_names(self):
        """The package exposes the storage APIs of the implementation (the same objects)"""
        import ncore.data.stores as public

        from ncore.impl.data import backend, itar, nodes, stores

        expected = {
            "IndexedTarArchive": itar.IndexedTarArchive,
            "Record": itar.Record,
            "RecordExistsError": itar.RecordExistsError,
            "IndexTailRetention": itar.IndexTailRetention,
            "Store": stores.Store,
            "IndexedTarZarrStore": stores.IndexedTarZarrStore,
            "ZarrStoreAdapter": stores.ZarrStoreAdapter,
            "ZarrStore": backend.ZarrStore,
            "Group": nodes.Group,
            "Array": nodes.Array,
            "NodeCache": nodes.NodeCache,
            "ZarrFormat": backend.ZarrFormat,
            "Selection": backend.Selection,
            "FillValue": backend.FillValue,
            "Attributes": backend.Attributes,
            "BloscCompression": backend.BloscCompression,
            "BloscCname": backend.BloscCname,
            "BloscShuffle": backend.BloscShuffle,
            "EncodedArray": nodes.EncodedArray,
        }
        self.assertEqual(sorted(public.__all__), sorted(expected))
        for name, value in expected.items():
            self.assertIs(getattr(public, name), value, name)

    def test_zarr_store(self):
        """ZarrStore is the store base class of the installed zarr-python version (of the stores of adapters)"""
        from ncore.data.stores import ZarrStore, ZarrStoreAdapter

        with tempfile.TemporaryDirectory() as tmp:
            adapter = ZarrStoreAdapter.directory(Path(tmp) / "store.zarr")
            self.assertIsInstance(adapter.zarr, ZarrStore)
            self.assertIsNot(ZarrStore, object)

    def test_keyword_only_parameters(self):
        """The optional parameters of archives and stores are keyword-only"""
        from ncore.data.stores import IndexedTarArchive, IndexedTarZarrStore

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "store.itar"
            with IndexedTarZarrStore(path, mode="w") as store:
                store.create_root_group({})
                store.consolidate()

            # (untyped: the calls don't type check)
            archive: Any = IndexedTarArchive
            store: Any = IndexedTarZarrStore
            calls = {
                "IndexedTarArchive": lambda: archive(path, "r"),
                "IndexedTarArchive.open_async": lambda: archive.open_async(path, 1 << 20),
                "IndexedTarZarrStore": lambda: store(path, "r"),
                "IndexedTarZarrStore.open_async": lambda: store.open_async(path, 1 << 20),
            }
            for name, call in calls.items():
                with self.subTest(name), self.assertRaisesRegex(TypeError, "positional argument"):
                    call()

            # (keyword arguments are accepted)
            IndexedTarArchive(path, mode="r", index_tail_read_size=None).close()
            IndexedTarZarrStore(path, mode="r", index_tail_read_size=None).close()
            for opened in (
                asyncio.run(IndexedTarArchive.open_async(path, index_tail_read_size=None)),
                asyncio.run(IndexedTarZarrStore.open_async(path, index_tail_read_size=None)),
            ):
                opened.close()

    def test_archive_records(self):
        """Records of an archive are exposed read-only, with the offset and size of their payload"""
        from ncore.data.stores import IndexedTarArchive, Record

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "store.itar"
            with IndexedTarArchive(path, mode="w") as archive:
                archive["a"] = b"payload-a"
                archive["b"] = b"bb"
                self.assertEqual(sorted(archive.records), ["a", "b"])
                with self.assertRaises(TypeError):
                    cast(Any, archive.records)["c"] = Record(0, 1)  # (read-only)

            archive = IndexedTarArchive(path)
            self.assertEqual(sorted(archive.records), ["a", "b"])
            with path.open("rb") as file:
                for key, payload in (("a", b"payload-a"), ("b", b"bb")):
                    record = archive.records[key]
                    self.assertIsInstance(record, Record)
                    self.assertEqual(record.size, len(payload))
                    file.seek(record.offset)
                    self.assertEqual(file.read(record.size), payload)
            with self.assertRaises(TypeError):
                cast(Any, archive.records)["a"] = Record(0, 1)  # (read-only)
            archive.close()
