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

from python.runfiles import Runfiles  # pyright: ignore[reportMissingImports] # ty:ignore[unresolved-import]

from ncore.impl.common.transformations import (
    HalfClosedInterval,
    MotionCompensator,
    PoseGraphInterpolator,
    is_within_3d_bboxes,
)
from ncore.impl.data import types
from ncore.impl.data.v4.compat import SequenceLoaderV4
from ncore.impl.data.v4.components import SequenceComponentGroupsReader


_RUNFILES = Runfiles.Create()


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


class TestMotionCompensator(unittest.TestCase):
    def setUp(self):
        # Make printed errors more representable numerically
        np.set_printoptions(floatmode="unique", linewidth=200, suppress=True)

        # Load a lidar sensor as a source for non-motion-compensated point cloud data
        self.loader = SequenceLoaderV4(
            SequenceComponentGroupsReader(
                [
                    _RUNFILES.Rlocation(
                        "test-data-v4/c9b05cf4-afb9-11ec-b3c2-00044bf65fcb@1648597318700123-1648599151600035.json"
                    )
                ],
            )
        )

    def test_idempotence(self):
        """Test to verify compensation / decompensation are symmetric"""

        motion_compensator = MotionCompensator(self.loader.pose_graph)
        lidar_sensor = self.loader.get_lidar_sensor("lidar_gt_top_p128_v4p5")

        # Check on a few frames only
        for frame_idx in range(0, 2):
            # Load non motion-compensated reference point cloud
            xyz_m_ref = lidar_sensor.get_frame_point_cloud(
                frame_idx, motion_compensation=False, with_start_points=False, return_index=0
            ).xyz_m_end
            timestamp_us = lidar_sensor.get_frame_ray_bundle_timestamp_us(frame_idx)

            frame_start_timestamps_us = lidar_sensor.get_frame_timestamp_us(frame_idx, types.FrameTimepoint.START)
            frame_end_timestamps_us = lidar_sensor.get_frame_timestamp_us(frame_idx, types.FrameTimepoint.END)

            # Run compensation, this gives both motion-compensated start/end points
            motion_compensation_result = motion_compensator.motion_compensate_points(
                lidar_sensor.sensor_id,
                xyz_m_ref,
                timestamp_us,
                frame_start_timestamps_us,
                frame_end_timestamps_us,
            )

            # Re-run decompensation on compensated points
            xyz_m = motion_compensator.motion_decompensate_points(
                lidar_sensor.sensor_id,
                motion_compensation_result.xyz_e_sensorend,
                timestamp_us,
                frame_start_timestamps_us,
                frame_end_timestamps_us,
            )

            xyz_s_m = motion_compensator.motion_decompensate_points(
                lidar_sensor.sensor_id,
                motion_compensation_result.xyz_s_sensorend,
                timestamp_us,
                frame_start_timestamps_us,
                frame_end_timestamps_us,
            )

            # Check for consistency
            self.assertIsNone(
                np.testing.assert_array_almost_equal(
                    np.zeros_like(delta_s := np.linalg.norm(xyz_s_m, axis=1)),
                    delta_s,
                    # lower-precision check only because of numerical errors building up + not doing
                    # *repeated* linear interpolatins in MotionCompensator for poses as in the source test data
                    decimal=2,
                ),
                f"inconsistent start points, frame_idx {frame_idx}",
            )

            self.assertIsNone(
                np.testing.assert_array_almost_equal(
                    np.zeros_like(delta_e := np.linalg.norm(xyz_m - xyz_m_ref, axis=1)),
                    delta_e,
                    # lower-precision check only because of numerical errors building up + not doing
                    # *repeated* linear interpolatins in MotionCompensator for poses as in the source test data
                    decimal=2,
                ),
                f"inconsistent end points, frame_idx {frame_idx}",
            )


def get_SE3(t: np.ndarray) -> np.ndarray:
    """SE3 matrix with variable translation part"""
    return np.block(
        [
            [
                np.eye(3),
                t.reshape((3, 1)),
            ],
            [np.array([0, 0, 0, 1])],
        ]
    )


class TestPoseGraphInterpolator(unittest.TestCase):
    def setUp(self):
        # Make printed errors more representable numerically
        np.set_printoptions(floatmode="unique", linewidth=200, suppress=True)

        # create simple connected pose graph (tree) with nodes/edges (only varying translation part)
        #          +---+
        #          |V7 |
        #          +---+
        #            ^
        #            |
        #        +-------+
        #        |  V1   |
        #        +-------+
        #          |  ||
        #      -----  |--------
        #      |      ---     |
        #      v        |     |
        #   +-----+     |     |
        #   | V2  |     |     |
        #   +-----+     |     |
        #     | ^       |     |
        #   --- ---     |     |
        #   |     |     |     |
        #   v     |     v     v
        # +---+ +---+ +---+ +---+
        # |V5 | |V6 | |V4 | |V3 |
        # +---+ +---+ +---+ +---+

        timestamps_us = np.array([0, 10], dtype=np.uint64)

        self.edges = [
            PoseGraphInterpolator.Edge(
                "V1", "V2", np.stack([np.eye(4), get_SE3(np.array([1, 0, 0]))]).astype(np.float32), timestamps_us
            ),
            PoseGraphInterpolator.Edge("V1", "V7", get_SE3(np.array([0, 1, 0])).astype(np.float64), None),
            PoseGraphInterpolator.Edge("V1", "V3", get_SE3(np.array([0, 1, 0])).astype(np.float32), None),
            PoseGraphInterpolator.Edge(
                "V1", "V4", np.stack([np.eye(4), get_SE3(np.array([2, 0, 0]))]).astype(np.float32), timestamps_us
            ),
            PoseGraphInterpolator.Edge("V2", "V5", get_SE3(np.array([0, 1, 0])).astype(np.float32), None),
            PoseGraphInterpolator.Edge(
                "V6", "V2", np.stack([np.eye(4), get_SE3(np.array([0, 0, 1]))]).astype(np.float32), timestamps_us
            ),
        ]

    def test_init_graph(self):
        """Test to verify pose graph initialization / path computation is correct"""

        with self.assertRaises(AssertionError):
            # invalid edge
            PoseGraphInterpolator.Edge("V3", "V4", np.stack([np.eye(4), get_SE3(np.array([0, 0, 1]))]), None)

        with self.assertRaises(AssertionError):
            # invalid edge
            PoseGraphInterpolator.Edge("V3", "V4", get_SE3(np.array([0, 1, 0])), np.array([0, 10], dtype=np.uint64))

        with self.assertRaises(ValueError):
            # cycle in graph
            edges_invalid = self.edges + [PoseGraphInterpolator.Edge("V3", "V4", np.eye(4), None)]
            PoseGraphInterpolator(edges_invalid)

        with self.assertRaises(ValueError):
            # disconnected graph
            edges_invalid = self.edges + [PoseGraphInterpolator.Edge("V8", "V9", np.eye(4), None)]
            PoseGraphInterpolator(edges_invalid)

        with self.assertRaises(ValueError):
            # duplicated edge
            edges_invalid = self.edges + [PoseGraphInterpolator.Edge("V1", "V4", np.eye(4), None)]
            PoseGraphInterpolator(edges_invalid)

        with self.assertRaises(ValueError):
            # self edge
            edges_invalid = self.edges + [PoseGraphInterpolator.Edge("V1", "V1", np.eye(4), None)]
            PoseGraphInterpolator(edges_invalid)

        PoseGraphInterpolator(self.edges)  # should not raise any exception on valid graph

    def test_interpolation(self):
        """Test to verify pose interpolation along different paths through a graph"""

        graph = PoseGraphInterpolator(self.edges)

        with self.assertRaises(KeyError):
            # non-existing start node
            graph.evaluate_poses("foo", "V1", np.array([0], dtype=np.uint64))

        with self.assertRaises(KeyError):
            # non-existing end node
            graph.evaluate_poses("V1", "foo", np.array([0], dtype=np.uint64))

        # self-pose is identity
        self.assertTrue(
            np.array_equal(
                res := graph.evaluate_poses("V1", "V1", np.array([0], dtype=np.uint64)),
                np.eye(4)[np.newaxis],
            )
        )
        self.assertTrue(res.dtype == np.float32)

        # verify different timestamp dtypes and empty batch shape on self / static edges
        self.assertTrue(
            np.array_equal(
                graph.evaluate_poses("V1", "V1", np.empty((), dtype=np.uint64)),
                np.eye(4),
            )
        )
        self.assertTrue(
            np.array_equal(
                res := graph.evaluate_poses("V7", "V1", np.empty((), dtype=np.uint64)),
                get_SE3(np.array([0, -1, 0])),
            )
        )
        self.assertTrue(
            # V7 <- V1 is static edge with float64 pose
            res.dtype == np.float64
        )

        # V7 < - V1 -> V3  is static
        self.assertTrue(
            # one hop
            np.array_equal(
                graph.evaluate_poses("V1", "V3", np.array([0, 5, 10], dtype=np.uint64)),
                np.repeat(get_SE3(np.array([0, 1, 0]))[np.newaxis], 3, axis=0),
            )
        )

        self.assertTrue(
            # two hops, one inverse
            np.array_equal(
                res := graph.evaluate_poses("V7", "V3", np.array([0, 5, 10], dtype=np.uint64)),
                np.repeat(get_SE3(np.array([0, 0, 0]))[np.newaxis], 3, axis=0),
            )
        )
        self.assertTrue(
            # V7 <- V1 is static edge with float64 pose
            res.dtype == np.float64
        )

        # V1 -> V4 is dynamic
        self.assertTrue(
            np.array_equal(
                res := graph.evaluate_poses("V1", "V4", np.array([0, 5, 10], dtype=np.uint64)),
                np.stack([get_SE3(np.array([0, 0, 0])), get_SE3(np.array([1, 0, 0])), get_SE3(np.array([2, 0, 0]))]),
            )
        )
        self.assertTrue(res.dtype == np.float32)

        # V1 -> V2 <- V6 is dynamic
        self.assertTrue(
            np.array_equal(
                res := graph.evaluate_poses("V1", "V6", np.array([0, 5, 10], dtype=np.uint64)),
                np.stack(
                    [get_SE3(np.array([0, 0, 0])), get_SE3(np.array([0.5, 0, -0.5])), get_SE3(np.array([1, 0, -1]))]
                ),
            )
        )
        self.assertTrue(res.dtype == np.float32)

        # V7 <- V1 -> V2 <- V6 is mixed static / dynamic
        self.assertTrue(
            np.array_equal(
                res := graph.evaluate_poses("V7", "V6", np.array([0, 5, 10], dtype=np.uint64)),
                np.stack(
                    [get_SE3(np.array([0, -1, 0])), get_SE3(np.array([0.5, -1, -0.5])), get_SE3(np.array([1, -1, -1]))]
                ),
            )
        )
        self.assertTrue(res.dtype == np.float64)

        # verify batch-shape handling
        self.assertTrue(
            np.array_equal(
                res := graph.evaluate_poses("V7", "V6", np.array([0, 5, 10], dtype=np.uint64).reshape((1, 3, 1, 1))),
                np.stack(
                    [get_SE3(np.array([0, -1, 0])), get_SE3(np.array([0.5, -1, -0.5])), get_SE3(np.array([1, -1, -1]))]
                ).reshape((1, 3, 1, 1, 4, 4)),
            )
        )
        self.assertTrue(res.dtype == np.float64)
