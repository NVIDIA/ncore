# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

import unittest
import tempfile

import numpy as np
import zarr

from .stores import IndexedTarStore


class TestIndexedTarStore(unittest.TestCase):
    ''' Test to verify functionality of IndexedTarStore '''
    def test_reserialization(self):
        ''' Make sure storing / loading of regular zarr data to .itar files works correctly '''

        # Fill a reference store
        s_ref = zarr.MemoryStore()
        g_ref = zarr.open(store=s_ref)
        g_ref.create_dataset('foo', data=np.random.rand(3, 3, 3))
        g_ref.attrs.update({'some': 'thing'})
        g_ref.require_group('subgroup').create_dataset('foo', data=np.random.rand(5, 5, 5))

        # re-serialize to .itar archive
        with tempfile.NamedTemporaryFile(suffix='.itar') as f:
            with IndexedTarStore(f.name, mode='w') as s_itar_out:  # closes file on exit
                zarr.copy_store(s_ref, s_itar_out)

            # reload store from file
            g_reload = zarr.open(store=IndexedTarStore(f.name, mode='r'), mode='r')

            # check all data was correctly serialized / deserialized
            self.assertIsNone(np.testing.assert_array_equal(g_ref['foo'][()], g_reload['foo'][()]))
            self.assertIsNone(
                np.testing.assert_array_equal(g_ref['subgroup']['foo'][()], g_reload['subgroup']['foo'][()]))
            self.assertDictEqual(g_ref.attrs.asdict(), g_reload.attrs.asdict())

    def test_empty(self):
        ''' Verify edge case of serialization of empty store is possible without errors '''
        with tempfile.NamedTemporaryFile(suffix='.itar') as f:
            with IndexedTarStore(f.name, mode='w') as _:  # closes file on exit
                # Don't write any zarr data (still serializes empty tar / seek tables)
                pass

            with IndexedTarStore(f.name, mode='r') as s_itar_in:
                # Loading store should work without errors

                # But loading a non-existing group should then fail
                with self.assertRaises(zarr.errors.PathNotFoundError):
                    zarr.open(store=s_itar_in, mode='r')
