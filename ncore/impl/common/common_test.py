# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.

import time
import tempfile
import unittest

import numpy as np

from scipy.spatial.transform import Rotation as R

from ncore.impl.common.common import (
    load_pkl,
    save_pkl,
    load_pc_dat,
    save_pc_dat,
    uniform_subdivide_range,
    HalfClosedInterval,
)
from ncore.impl.common.transformations import is_within_3d_bbox, is_within_3d_bboxes


def test_save_load_pkl():
    """Test to verify functionality of load_pkl / save_pkl"""

    def check(obj):
        with tempfile.NamedTemporaryFile() as tmp:
            save_pkl(obj, tmp.name)
            assert obj == load_pkl(tmp.name), "serialized object not equal to de-serialized version"

    check({})
    check({"some": "entry", "some-int": 5})
    check({"something": [1, 2, [3, 4]], "some-int": {"foo": "bar"}})


class TestSaveLoadPCDat(unittest.TestCase):
    def setUp(self):
        # create some test point-cloud
        self.pc = np.random.rand(30, 9).astype(np.float32)

    def test_input_output(self):
        """Test to verify functionality of all positive use cases of save_pc_dat / load_pc_dat"""

        def check(pc, suffix):
            with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
                save_pc_dat(tmp.name, pc)
                self.assertTrue((pc == load_pc_dat(tmp.name)).all())

        # Check uncompressed / compressed versions
        check(self.pc, ".dat")
        check(self.pc, ".dat.xz")

    def test_input_output_fallback(self):
        """Test to verify functionality of load_pc_dat file lookup fallback (.dat <-> .dat.xz)"""
        with tempfile.NamedTemporaryFile(suffix=".dat") as tmp:
            save_pc_dat(tmp.name, self.pc)  # store .dat

            fallback_path = tmp.name.replace(".dat", ".dat.xz")

            self.assertTrue((self.pc == load_pc_dat(fallback_path)).all())  # make sure we can load as .dat.xz also

            with self.assertRaises(FileNotFoundError):
                load_pc_dat(fallback_path, allow_lookup_fallback=False)

        with tempfile.NamedTemporaryFile(suffix=".dat.xz") as tmp:
            save_pc_dat(tmp.name, self.pc)  # store .dat.xz

            fallback_path = tmp.name.replace(".dat.xz", ".dat")

            self.assertTrue((self.pc == load_pc_dat(fallback_path)).all())  # make sure we can load as .dat also

            with self.assertRaises(FileNotFoundError):
                load_pc_dat(fallback_path, allow_lookup_fallback=False)

    def test_invalid_arguments(self):
        """Test to verify correct behavior on invalid input"""

        with self.assertRaises(ValueError):
            save_pc_dat("/some/wrong/path.txt", self.pc)

        with self.assertRaises(ValueError):
            save_pc_dat("/some/valid/path.dat", self.pc.astype(np.float64))

        with self.assertRaises(ValueError):
            load_pc_dat("/some/wrong/path.txt")


class TestIsWithin3DBBox(unittest.TestCase):
    def setUp(self):

        # Set the random seed
        np.random.seed(41)

        # create some test point-cloud
        self.pc = np.random.rand(100000, 3).astype(np.float32) * 3.0  # increase it to [0,3] range

        # create some bounding boxes
        center = np.random.rand(100, 3).astype(np.float32) * 3.0
        dim = np.random.rand(100, 3).astype(np.float32)
        rotation = np.random.rand(100, 3).astype(np.float32) * 2 * np.pi

        self.bboxes = np.concatenate([center, dim, rotation], axis=-1).astype(np.float32)

        # Create an outlier point that's outside the bboxes (guaranteed) by minimum of xyz of the
        # point being larger than maximum possible dimensions for bboxes
        self.outlier_point = np.random.uniform(1000, 2000, size=(1, 3)).astype(np.float32)

    def test_multi_bbox_processing(self):
        """Test to verify that processing all the boxes at once is the same as doing it one by one"""
        single_box = []
        for i in range(self.bboxes.shape[0]):
            single_box.append(is_within_3d_bboxes(self.pc, self.bboxes[i : i + 1, :]).reshape(-1, 1))
        single_box = np.concatenate(single_box, axis=1)

        self.assertTrue((single_box == is_within_3d_bboxes(self.pc, self.bboxes)).all())

    def test_multi_single_bbox_equivalent(self):
        """
        Test to verify that [is_within_3d_bboxes] results in the same output as individually
        calling [is_within_3d_bbox] on each bbox
        """
        any_true = False
        for i in range(self.bboxes.shape[0]):
            self.assertTrue(
                (
                    is_within_3d_bbox(self.pc, self.bboxes[i, :])
                    == is_within_3d_bboxes(self.pc, self.bboxes[i : i + 1, :]).flatten()
                ).all()
            )
            any_true = any_true or is_within_3d_bboxes(self.pc, self.bboxes[i : i + 1, :]).any()

        self.assertTrue(any_true)

        # test all bboxes in one go
        point_in_box = np.empty((self.pc.shape[0], self.bboxes.shape[0]), dtype=np.bool_)
        for i, bbox in enumerate(self.bboxes):
            point_in_box[:, i] = is_within_3d_bbox(self.pc, bbox)

        self.assertTrue((is_within_3d_bboxes(self.pc, self.bboxes) == point_in_box).all())

    def test_outlier_point_outside_bboxes(self):
        """Test to verify that the outlier point is not within any of the defined boxes"""
        self.assertFalse(is_within_3d_bboxes(self.outlier_point, self.bboxes).all())

    def test_multi_bbox_efficieny(self):
        """
        Test to verify that the runtime of [is_within_3d_bboxes] is faster than looping over each
        bbox and calling [is_within_3d_bbox]
        """
        loop_start_time = time.time()
        point_in_box = np.empty((self.pc.shape[0], self.bboxes.shape[0]), dtype=np.bool_)
        for i, bbox in enumerate(self.bboxes):
            point_in_box[:, i] = is_within_3d_bbox(self.pc, bbox)
        loop_runtime = time.time() - loop_start_time

        multi_start_time = time.time()
        _points_in_bboxes = is_within_3d_bboxes(self.pc, self.bboxes)
        multi_runtime = time.time() - multi_start_time

        self.assertTrue(multi_runtime < loop_runtime)


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
    check(
        uniform_subdivide_range(subdiv_id=0, subdiv_count=1, range_start=0, range_end=0),
        (np.empty_like(np.arange(0, 0)), -1),
    )


class TestHalfClosedInterval(unittest.TestCase):
    def test_init_len(self):
        """Test to verify HalfClosedInterval.__init__() / __len__()"""

        # Valid interval
        self.assertEqual(len(HalfClosedInterval(0, 1)), 1)

        # Empty interval
        self.assertEqual(len(HalfClosedInterval(1, 1)), 0)

        # Invalid interval -> exception
        with self.assertRaises(Exception):
            HalfClosedInterval(0, -1)

    def test_contains(self):
        """Test to verify HalfClosedInterval.__contains__()"""
        interval_0_3 = HalfClosedInterval(0, 3)

        self.assertTrue(0 in interval_0_3)
        self.assertTrue(1 in interval_0_3)
        self.assertTrue(2 in interval_0_3)
        self.assertFalse(-1 in interval_0_3)
        self.assertFalse(3 in interval_0_3)
        self.assertFalse(4 in interval_0_3)

    def test_intersection(self):
        """Test to verify HalfClosedInterval.intersection()"""
        interval_0_3 = HalfClosedInterval(0, 3)

        # self-intersection
        self.assertEqual(interval_0_3.intersection(interval_0_3), interval_0_3)

        # non-empty intersection
        self.assertEqual(interval_0_3.intersection(HalfClosedInterval(2, 10)), HalfClosedInterval(2, 3))

        # empty intersections
        self.assertEqual(interval_0_3.intersection(HalfClosedInterval(-5, -2)), None)
        self.assertEqual(interval_0_3.intersection(HalfClosedInterval(3, 4)), None)

    def test_overlaps(self):
        """Test to verify HalfClosedInterval.overlaps()"""
        interval_0_3 = HalfClosedInterval(0, 3)

        # self-intersection
        self.assertTrue(interval_0_3.overlaps(interval_0_3))

        # non-empty intersection
        self.assertTrue(interval_0_3.overlaps(HalfClosedInterval(2, 10)))

        # empty intersections
        self.assertFalse(interval_0_3.overlaps(HalfClosedInterval(-5, -2)))
        self.assertFalse(interval_0_3.overlaps(HalfClosedInterval(3, 4)))

    def test_cover_range(self):
        """Test to verify HalfClosedInterval.cover_range()"""
        interval_0_3 = HalfClosedInterval(0, 3)

        # self-intersection
        self.assertEqual(cover_range := interval_0_3.cover_range(test_range := np.arange(0, 3)), range(0, 3))
        self.assertTrue([test_range[i] in interval_0_3 for i in cover_range])

        # full interval cover
        self.assertEqual(cover_range := interval_0_3.cover_range(test_range := np.arange(-5, 10)), range(5, 8))
        self.assertTrue([test_range[i] in interval_0_3 for i in cover_range])

        # subranges (partial covers)
        self.assertEqual(cover_range := interval_0_3.cover_range(test_range := np.arange(1, 10)), range(0, 2))
        self.assertTrue([test_range[i] in interval_0_3 for i in cover_range])

        self.assertEqual(cover_range := interval_0_3.cover_range(test_range := np.arange(-5, 2)), range(5, 7))
        self.assertTrue([test_range[i] in interval_0_3 for i in cover_range])

        # no cover (left)
        self.assertEqual(cover_range := interval_0_3.cover_range(test_range := np.arange(-5, 0)), range(0, 0))
        self.assertFalse([test_range[i] in interval_0_3 for i in cover_range])

        # no cover (right)
        self.assertEqual(cover_range := interval_0_3.cover_range(test_range := np.arange(3, 10)), range(0, 0))
        self.assertFalse([test_range[i] in interval_0_3 for i in cover_range])

        # no samples
        self.assertEqual(cover_range := interval_0_3.cover_range(test_range := np.arange(0, 0)), range(0, 0))
        self.assertFalse([test_range[i] in interval_0_3 for i in cover_range])

        # empty interval case
        interval_5_5 = HalfClosedInterval(5, 5)
        self.assertEqual(cover_range := interval_5_5.cover_range(test_range := np.arange(0, 10)), range(0, 0))
        self.assertFalse([test_range[i] in interval_5_5 for i in cover_range])
