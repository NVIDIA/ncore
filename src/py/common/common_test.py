# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import tempfile
import unittest

import numpy as np

from src.py.common.common import load_pkl, save_pkl, load_pc_dat, save_pc_dat


def test_save_load_pkl():
    """ Test to verify functionality of load_pkl / save_pkl """
    def check(obj):
        with tempfile.NamedTemporaryFile() as tmp:
            save_pkl(obj, tmp.name)
            assert obj == load_pkl(
                tmp.name
            ), "serialized object not equal to de-serialized version"

    check({})
    check({'some': 'entry', 'some-int': 5})
    check({'something': [1, 2, [3, 4]], 'some-int': {'foo': 'bar'}})


class TestSaveLoadPCDat(unittest.TestCase):
    def setUp(self):
        # create some test point-cloud
        self.pc_ref = np.random.rand(30, 9).astype(np.float32)

    def test_input_output(self):
        """ Test to verify functionality of all positive use cases of save_pc_dat / load_pc_dat"""
        def check(pc, suffix):
            with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
                save_pc_dat(tmp.name, pc)
                self.assertTrue((pc == load_pc_dat(tmp.name)).all())

        # Check uncompressed / compressed versions
        check(self.pc_ref, '.dat')
        check(self.pc_ref, '.dat.xz')

    def test_invalid_arguments(self):
        """ Test to verify correct behavior on invalid input """

        with self.assertRaises(ValueError):
            save_pc_dat('/some/wrong/path.txt', self.pc_ref)

        with self.assertRaises(ValueError):
            save_pc_dat('/some/valid/path.dat', self.pc_ref.astype(np.float64))

        with self.assertRaises(ValueError):
            load_pc_dat('/some/wrong/path.txt')
