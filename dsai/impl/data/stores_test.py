# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

import unittest
import tempfile

import parameterized
import numpy as np
import zarr

from .stores import IndexedTarStore, consolidate_compressed_metadata, open_compressed_consolidated


class TestIndexedTarStore(unittest.TestCase):
    ''' Test to verify functionality of IndexedTarStore '''
    def setUp(self):
        # Fill a reference group with an in-memory store
        self.g_ref = zarr.open(store=zarr.MemoryStore())
        self.g_ref.create_dataset('foo', data=np.random.rand(3, 3, 3))
        self.g_ref.attrs.update({'some': 'thing'})
        self.g_ref.require_group('subgroup').create_dataset('foo', data=np.random.rand(5, 5, 5))

    def check_with_reference(self, group):
        ''' Verifies all values of a group against the reference '''
        self.assertIsNone(np.testing.assert_array_equal(self.g_ref['foo'][()], group['foo'][()]))
        self.assertIsNone(np.testing.assert_array_equal(self.g_ref['subgroup']['foo'][()],
                                                        group['subgroup']['foo'][()]))
        self.assertDictEqual(self.g_ref.attrs.asdict(), group.attrs.asdict())

    def test_reserialization(self):
        ''' Make sure storing / loading of regular zarr data to .itar files works correctly '''

        # re-serialize to .itar archive
        with tempfile.NamedTemporaryFile(suffix='.itar') as f:
            with IndexedTarStore(f.name, mode='w') as s_itar_out:  # closes file on exit
                zarr.copy_store(self.g_ref.store, s_itar_out)

            # reload store from file
            store = IndexedTarStore(f.name)
            g_reload = zarr.open(store=store, mode='r')

            # check all data was correctly serialized / deserialized
            self.check_with_reference(g_reload)

            # check reloading resources is functional
            store.reload_resources()
            self.check_with_reference(g_reload)

    def test_compressed_consolidated(self):
        ''' Make sure compressed consolidated meta data is stored/loaded correctly '''

        # serialize to .itar archive (will also serialize compressed-consolidated meta-data)
        with tempfile.NamedTemporaryFile(suffix='.itar') as f:
            with IndexedTarStore(f.name, mode='w') as s_itar_out:  # closes file on exit
                zarr.copy_store(self.g_ref.store, s_itar_out)

                # consolidate compress meta-data
                consolidate_compressed_metadata(s_itar_out)

            # reload store from file with compressed consolidated meta-data
            store = IndexedTarStore(f.name)
            g_reload = open_compressed_consolidated(store=store, mode='r')

            # check all data was correctly serialized / deserialized
            self.check_with_reference(g_reload)

            # check reloading resources is functional
            store.reload_resources()
            self.check_with_reference(g_reload)

    @parameterized.parameterized.expand([(
        "not-compressed_consolidate",
        False,
    ), (
        "compressed_consolidate",
        True,
    )])
    def test_empty(self, _, compressed_consolidate: bool):
        ''' Verify edge case of serialization of empty store is possible without errors '''
        with tempfile.NamedTemporaryFile(suffix='.itar') as f:
            with IndexedTarStore(f.name, mode='w') as s_itar_out:  # closes file on exit
                # Don't write any zarr data (still serializes empty tar / seek tables)

                if compressed_consolidate:
                    consolidate_compressed_metadata(s_itar_out)

            with IndexedTarStore(f.name) as s_itar_in:
                # Loading store should work without errors

                # But loading a non-existing group should then fail
                with self.assertRaises(zarr.errors.PathNotFoundError):

                    if compressed_consolidate:
                        open_compressed_consolidated(s_itar_in, mode='r')
                    else:
                        zarr.open(store=s_itar_in, mode='r')
