# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import unittest

import numpy as np

from ncore.impl.common.common import (
    HalfClosedInterval,
    uniform_subdivide_range,
)
from ncore.impl.common.transformations import is_within_3d_bboxes


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

        # Create inliner points that are guaranteed to be inside the bounding boxes to test that
        # points inside the bounding boxes are classified correctly
        self.inliner_points = center[np.random.choice(100, 10, replace=True)]

        # Making the dimensions larger than the upper limit of the inliner points to ensure that the
        # points are guaranteed to be inside the bounding box
        self.inliner_bboxes = np.concatenate(
            [
                np.zeros((10, 3), dtype=np.float32),  # centers
                np.random.rand(10, 3).astype(np.float32) * 1000.0,  # dims
                np.zeros((10, 3), dtype=np.float32),  # rotations
            ],
            axis=-1,
        ).astype(np.float32)

    def test_multi_bbox_processing(self):
        """Test to verify that processing all the boxes at once is the same as doing it one by one"""
        single_box = []
        for i in range(self.bboxes.shape[0]):
            single_box.append(is_within_3d_bboxes(self.pc, self.bboxes[i : i + 1, :]).reshape(-1, 1))
        single_box = np.concatenate(single_box, axis=1)

        self.assertTrue((single_box == is_within_3d_bboxes(self.pc, self.bboxes)).all())

    def test_outlier_point_outside_bboxes(self):
        """Test to verify that the outlier point is not within any of the defined boxes"""
        self.assertFalse(is_within_3d_bboxes(self.outlier_point, self.bboxes).all())

    def test_inliner_points_inside_bboxes(self):
        """Test to verify that points inside the bounding boxes are correctly classified as inside"""
        self.assertTrue(is_within_3d_bboxes(self.inliner_points, self.inliner_bboxes).all())


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

        # Check start/end
        self.assertEqual(len(HalfClosedInterval.from_start_end(0, 1)), 2)
        self.assertEqual(len(HalfClosedInterval.from_start_end(1, 1)), 1)
        self.assertEqual(len(HalfClosedInterval.from_start_end(1, 0)), 0)

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

        self.assertTrue(HalfClosedInterval(0, 3) in interval_0_3)
        self.assertTrue(HalfClosedInterval(1, 3) in interval_0_3)
        self.assertTrue(HalfClosedInterval(1, 2) in interval_0_3)
        self.assertFalse(HalfClosedInterval(-1, 3) in interval_0_3)
        self.assertFalse(HalfClosedInterval(0, 4) in interval_0_3)
        self.assertFalse(HalfClosedInterval(3, 4) in interval_0_3)

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
