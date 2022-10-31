# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import time
import tempfile
import unittest
import numpy as np

from src.dsai_internal.common.common import load_pkl, save_pkl, load_pc_dat, save_pc_dat, is_within_3d_bbox, uniform_subdivide_range
from src.dsai_internal.av_utils import isWithin3DBBox


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
        self.pc = np.random.rand(30, 9).astype(np.float32)

    def test_input_output(self):
        """ Test to verify functionality of all positive use cases of save_pc_dat / load_pc_dat"""
        def check(pc, suffix):
            with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
                save_pc_dat(tmp.name, pc)
                self.assertTrue((pc == load_pc_dat(tmp.name)).all())

        # Check uncompressed / compressed versions
        check(self.pc, '.dat')
        check(self.pc, '.dat.xz')

    def test_input_output_fallback(self):
        """ Test to verify functionality of load_pc_dat file lookup fallback (.dat <-> .dat.xz) """
        with tempfile.NamedTemporaryFile(suffix='.dat') as tmp:
            save_pc_dat(tmp.name, self.pc)  # store .dat

            fallback_path = tmp.name.replace('.dat', '.dat.xz')

            self.assertTrue((self.pc == load_pc_dat(fallback_path)
                             ).all())  # make sure we can load as .dat.xz also

            with self.assertRaises(FileNotFoundError):
                load_pc_dat(fallback_path, allow_lookup_fallback=False)

        with tempfile.NamedTemporaryFile(suffix='.dat.xz') as tmp:
            save_pc_dat(tmp.name, self.pc)  # store .dat.xz

            fallback_path = tmp.name.replace('.dat.xz', '.dat')

            self.assertTrue((self.pc == load_pc_dat(fallback_path)
                             ).all())  # make sure we can load as .dat also

            with self.assertRaises(FileNotFoundError):
                load_pc_dat(fallback_path, allow_lookup_fallback=False)

    def test_invalid_arguments(self):
        """ Test to verify correct behavior on invalid input """

        with self.assertRaises(ValueError):
            save_pc_dat('/some/wrong/path.txt', self.pc)

        with self.assertRaises(ValueError):
            save_pc_dat('/some/valid/path.dat', self.pc.astype(np.float64))

        with self.assertRaises(ValueError):
            load_pc_dat('/some/wrong/path.txt')


class TestIsWithin3DBBox(unittest.TestCase):
    def setUp(self):
        
        # Set the random seed 
        np.random.seed(41)

        # create some test point-cloud
        self.pc = np.random.rand(100000,3).astype(np.float32) * 3.0 # increase it to [0,3] range
        
        # create some bounding boxes
        center = np.random.rand(100,3).astype(np.float32) * 3.0
        dim = np.random.rand(100,3).astype(np.float32)
        rotation = np.random.rand(100,3).astype(np.float32) * 2 * np.pi

        self.bboxes = np.concatenate([center, dim, rotation], axis=-1).astype(np.float32)

    def test_oputput_vales(self):
        """ Test to verify functionality of the c++ implentation (the output should be the same to python) """
        any_true = False
        for i in range(self.bboxes.shape[0]):
            self.assertTrue((is_within_3d_bbox(self.pc, self.bboxes[i,:]) == isWithin3DBBox(self.pc, self.bboxes[i:i+1,:])
                    ).all())
            any_true = any_true or isWithin3DBBox(self.pc, self.bboxes[i:i+1,:]).any()

        self.assertTrue(any_true)

    def test_multi_bbox_processing(self):
        """ Test to verify that processing all the boxes at once is the same as doing it one by one """
        single_box = []
        for i in range(self.bboxes.shape[0]):
            single_box.append(isWithin3DBBox(self.pc, self.bboxes[i:i+1,:]).reshape(-1,1))
        single_box = np.concatenate(single_box, axis=1)

        self.assertTrue((single_box == isWithin3DBBox(self.pc, self.bboxes)).all())


    def test_efficiency(self):
        """ Test to verify that the cpp code is faster than python """

        start_time_python = time.time()
        for i in range(self.bboxes.shape[0]):
            is_within_3d_bbox(self.pc, self.bboxes[i,:])
        end_time_python = time.time()
        python_duration = end_time_python - start_time_python

        start_time_cpp = time.time()
        isWithin3DBBox(self.pc, self.bboxes)
        end_time_cpp = time.time()
        cpp_duration = end_time_cpp - start_time_cpp
        
        print(f"\nPython implementation took {python_duration} s for {self.bboxes.shape[0]} bboxes and {self.pc.shape[0]} points.")
        print(f"CPP implementation took {cpp_duration} s for {self.bboxes.shape[0]} bboxes and {self.pc.shape[0]} points.")
        self.assertLess(cpp_duration, python_duration)


    def test_invalid_arguments(self):
        """ Test to verify correct behavior on invalid input """

        with self.assertRaises(ValueError):
            isWithin3DBBox(self.pc.astype(np.float64), self.bboxes)

        with self.assertRaises(ValueError):
            isWithin3DBBox(self.pc, self.bboxes.astype(np.float64))

        with self.assertRaises(AssertionError):
            isWithin3DBBox(self.pc, self.bboxes[:,:6])


def test_uniform_subdivide_range():
    def check(actual, expected):
        range_equal = (actual[0] == expected[0]).all()
        offset_equal = actual[1] == expected[1]
        assert range_equal and offset_equal

    # range [0,9]
    check(uniform_subdivide_range(subdiv_id=0, subdiv_count=1, range_start=0, range_end=10), (np.arange(0, 10), 0))
    check(uniform_subdivide_range(subdiv_id=0, subdiv_count=2, range_start=0, range_end=10), (np.arange(0, 5), 0))
    check(uniform_subdivide_range(subdiv_id=1, subdiv_count=2, range_start=0, range_end=10), (np.arange(5, 10), 5))

    # range [5,14]
    check(uniform_subdivide_range(subdiv_id=0, subdiv_count=1, range_start=5, range_end=15), (np.arange(5, 15), 0))
    check(uniform_subdivide_range(subdiv_id=0, subdiv_count=2, range_start=5, range_end=15), (np.arange(5, 10), 0))
    check(uniform_subdivide_range(subdiv_id=1, subdiv_count=2, range_start=5, range_end=15), (np.arange(10, 15), 5))

    # empty range
    check(uniform_subdivide_range(subdiv_id=0, subdiv_count=1, range_start=0, range_end=0),
          (np.empty_like(np.arange(0, 0)), -1))
