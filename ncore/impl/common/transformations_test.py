# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np

from python.runfiles import Runfiles  # pyright: ignore[reportMissingImports] # ty:ignore[unresolved-import]
from scipy.spatial.transform import Rotation as R

from ncore.impl.common.transformations import (
    HalfClosedInterval,
    MotionCompensator,
    PoseGraphInterpolator,
    PoseInterpolator,
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


def get_SE3_with_rotation(t: np.ndarray, rot: R) -> np.ndarray:
    """SE3 matrix with variable translation and rotation parts"""
    return np.block(
        [
            [
                rot.as_matrix(),
                t.reshape((3, 1)),
            ],
            [np.array([0, 0, 0, 1])],
        ]
    )


class TestPoseInterpolatorExtrapolation(unittest.TestCase):
    """Tests for PoseInterpolator.extrapolate_to_timestamps()"""

    def setUp(self):
        np.set_printoptions(floatmode="unique", linewidth=200, suppress=True)

    def test_extrapolate_linear_motion_forward(self):
        """Test forward extrapolation with pure linear motion (no rotation)"""
        # Create a trajectory with constant velocity: v = [1, 0, 0] m/us (1m per microsecond)
        # Poses at timestamps 0, 100, 200 us with positions [0,0,0], [100,0,0], [200,0,0]
        timestamps = np.array([0, 100, 200], dtype=np.uint64)
        velocity = np.array([1.0, 0.0, 0.0])  # 1 m/us in x direction

        poses = np.stack([get_SE3(velocity * t) for t in timestamps]).astype(np.float32)

        interpolator = PoseInterpolator(poses, timestamps)

        # Extrapolate forward to 300, 400 us
        ts_forward = np.array([300, 400], dtype=np.uint64)
        result = interpolator.extrapolate_to_timestamps(ts_forward)

        # Expected positions: [300,0,0] and [400,0,0]
        expected_positions = np.array([[300, 0, 0], [400, 0, 0]], dtype=np.float32)

        np.testing.assert_array_almost_equal(result[:, :3, 3], expected_positions, decimal=5)
        # Rotation should remain identity (compare each matrix to identity)
        for i in range(len(result)):
            np.testing.assert_array_almost_equal(result[i, :3, :3], np.eye(3), decimal=5)

    def test_extrapolate_linear_motion_backward(self):
        """Test backward extrapolation with pure linear motion (no rotation)"""
        # Trajectory with constant velocity starting at t=100
        timestamps = np.array([100, 200, 300], dtype=np.uint64)
        velocity = np.array([1.0, 0.0, 0.0])

        poses = np.stack([get_SE3(velocity * t) for t in timestamps]).astype(np.float32)

        interpolator = PoseInterpolator(poses, timestamps)

        # Extrapolate backward to t=0 and t=50
        ts_backward = np.array([0, 50], dtype=np.uint64)
        result = interpolator.extrapolate_to_timestamps(ts_backward)

        # Expected positions: [0,0,0] and [50,0,0]
        expected_positions = np.array([[0, 0, 0], [50, 0, 0]], dtype=np.float32)

        np.testing.assert_array_almost_equal(result[:, :3, 3], expected_positions, decimal=5)
        for i in range(len(result)):
            np.testing.assert_array_almost_equal(result[i, :3, :3], np.eye(3), decimal=5)

    def test_extrapolate_linear_motion_both_directions(self):
        """Test extrapolation in both directions simultaneously"""
        timestamps = np.array([100, 200], dtype=np.uint64)
        velocity = np.array([1.0, 2.0, 0.0])  # velocity in x and y

        poses = np.stack([get_SE3(velocity * t) for t in timestamps]).astype(np.float32)

        interpolator = PoseInterpolator(poses, timestamps)

        # Mix of backward, within range, and forward timestamps
        ts_mixed = np.array([0, 50, 100, 150, 200, 250, 300], dtype=np.uint64)
        result = interpolator.extrapolate_to_timestamps(ts_mixed)

        expected_positions = np.array(
            [
                [0, 0, 0],  # t=0 (backward extrapolation)
                [50, 100, 0],  # t=50 (backward extrapolation)
                [100, 200, 0],  # t=100 (boundary - interpolation)
                [150, 300, 0],  # t=150 (interpolation)
                [200, 400, 0],  # t=200 (boundary - interpolation)
                [250, 500, 0],  # t=250 (forward extrapolation)
                [300, 600, 0],  # t=300 (forward extrapolation)
            ],
            dtype=np.float32,
        )

        np.testing.assert_array_almost_equal(result[:, :3, 3], expected_positions, decimal=4)

    def test_extrapolate_circular_rotation_forward(self):
        """Test forward extrapolation with pure rotation (circular motion around Z-axis)"""
        # Create a trajectory rotating around the Z-axis at constant angular velocity
        # Angular velocity: 90 degrees per 100 us = 0.9 deg/us = pi/200 rad/us
        timestamps = np.array([0, 100, 200], dtype=np.uint64)
        angular_velocity_deg_per_us = 0.9  # degrees per microsecond

        poses = np.stack(
            [
                get_SE3_with_rotation(
                    np.array([0, 0, 0]), R.from_euler("z", angular_velocity_deg_per_us * t, degrees=True)
                )
                for t in timestamps
            ]
        ).astype(np.float32)

        interpolator = PoseInterpolator(poses, timestamps)

        # Extrapolate forward to 300, 400 us
        ts_forward = np.array([300, 400], dtype=np.uint64)
        result = interpolator.extrapolate_to_timestamps(ts_forward)

        # Expected rotations: 270 deg and 360 deg around Z
        expected_rotations = [
            R.from_euler("z", 270, degrees=True),
            R.from_euler("z", 360, degrees=True),
        ]

        for i, expected_rot in enumerate(expected_rotations):
            result_rot = R.from_matrix(result[i, :3, :3])
            # Compare using rotation vectors (more stable for comparison)
            angle_diff = (result_rot * expected_rot.inv()).magnitude()
            self.assertLess(angle_diff, 1e-4, f"Rotation mismatch at index {i}")

    def test_extrapolate_circular_rotation_backward(self):
        """Test backward extrapolation with pure rotation (circular motion around Z-axis)"""
        # Trajectory starting at t=200 with rotations at 180, 270, 360 degrees
        timestamps = np.array([200, 300, 400], dtype=np.uint64)
        angular_velocity_deg_per_us = 0.9

        poses = np.stack(
            [
                get_SE3_with_rotation(
                    np.array([0, 0, 0]), R.from_euler("z", angular_velocity_deg_per_us * t, degrees=True)
                )
                for t in timestamps
            ]
        ).astype(np.float32)

        interpolator = PoseInterpolator(poses, timestamps)

        # Extrapolate backward to 0 and 100 us
        ts_backward = np.array([0, 100], dtype=np.uint64)
        result = interpolator.extrapolate_to_timestamps(ts_backward)

        # Expected rotations: 0 deg and 90 deg around Z
        expected_rotations = [
            R.from_euler("z", 0, degrees=True),
            R.from_euler("z", 90, degrees=True),
        ]

        for i, expected_rot in enumerate(expected_rotations):
            result_rot = R.from_matrix(result[i, :3, :3])
            angle_diff = (result_rot * expected_rot.inv()).magnitude()
            self.assertLess(angle_diff, 1e-4, f"Rotation mismatch at index {i}")

    def test_extrapolate_combined_motion(self):
        """Test extrapolation with combined linear and rotational motion"""
        timestamps = np.array([0, 100, 200], dtype=np.uint64)
        linear_velocity = np.array([1.0, 0.0, 0.0])
        angular_velocity_deg_per_us = 0.45  # 45 deg per 100 us

        poses = np.stack(
            [
                get_SE3_with_rotation(
                    linear_velocity * t, R.from_euler("z", angular_velocity_deg_per_us * t, degrees=True)
                )
                for t in timestamps
            ]
        ).astype(np.float32)

        interpolator = PoseInterpolator(poses, timestamps)

        # Extrapolate forward to 300 us
        ts_forward = np.array([300], dtype=np.uint64)
        result = interpolator.extrapolate_to_timestamps(ts_forward)

        # Expected: position [300,0,0], rotation 135 deg around Z
        expected_position = np.array([300, 0, 0], dtype=np.float32)
        expected_rotation = R.from_euler("z", 135, degrees=True)

        np.testing.assert_array_almost_equal(result[0, :3, 3], expected_position, decimal=4)
        result_rot = R.from_matrix(result[0, :3, :3])
        angle_diff = (result_rot * expected_rotation.inv()).magnitude()
        self.assertLess(angle_diff, 1e-4)

    def test_extrapolate_within_range_delegates_to_interpolate(self):
        """Test that timestamps within range use standard interpolation"""
        timestamps = np.array([0, 100, 200], dtype=np.uint64)
        poses = np.stack([get_SE3(np.array([t, 0, 0])) for t in timestamps]).astype(np.float32)

        interpolator = PoseInterpolator(poses, timestamps)

        # All timestamps within range
        ts_within = np.array([0, 50, 100, 150, 200], dtype=np.uint64)
        result_extrapolate = interpolator.extrapolate_to_timestamps(ts_within)
        result_interpolate = interpolator.interpolate_to_timestamps(ts_within)

        np.testing.assert_array_almost_equal(result_extrapolate, result_interpolate, decimal=5)

    def test_extrapolate_exceeds_max_time_raises_error(self):
        """Test that extrapolation beyond max_extrapolation_time_us raises ValueError"""
        timestamps = np.array([0, 100, 200], dtype=np.uint64)
        poses = np.stack([get_SE3(np.array([t, 0, 0])) for t in timestamps]).astype(np.float32)

        interpolator = PoseInterpolator(poses, timestamps)

        # Try to extrapolate 2 seconds forward with default 1 second limit
        ts_too_far = np.array([2_000_200], dtype=np.uint64)  # 2 seconds after end

        with self.assertRaises(ValueError) as ctx:
            interpolator.extrapolate_to_timestamps(ts_too_far)
        self.assertIn("exceeds limit", str(ctx.exception))

        # Try to extrapolate 2 seconds backward
        ts_too_far_back = np.array([0], dtype=np.uint64)
        timestamps_late = np.array([2_000_000, 2_000_100, 2_000_200], dtype=np.uint64)
        poses_late = np.stack([get_SE3(np.array([t, 0, 0])) for t in timestamps_late]).astype(np.float32)
        interpolator_late = PoseInterpolator(poses_late, timestamps_late)

        with self.assertRaises(ValueError) as ctx:
            interpolator_late.extrapolate_to_timestamps(ts_too_far_back)
        self.assertIn("exceeds limit", str(ctx.exception))

    def test_extrapolate_custom_max_time(self):
        """Test that custom max_extrapolation_time_us is respected"""
        timestamps = np.array([0, 100, 200], dtype=np.uint64)
        poses = np.stack([get_SE3(np.array([t, 0, 0])) for t in timestamps]).astype(np.float32)

        interpolator = PoseInterpolator(poses, timestamps)

        # Should succeed with large enough limit
        ts_far = np.array([500], dtype=np.uint64)
        result = interpolator.extrapolate_to_timestamps(ts_far, max_extrapolation_time_us=1000)
        self.assertEqual(result.shape, (1, 4, 4))

        # Should fail with small limit
        with self.assertRaises(ValueError):
            interpolator.extrapolate_to_timestamps(ts_far, max_extrapolation_time_us=100)

    def test_extrapolate_empty_timestamps(self):
        """Test extrapolation with empty timestamp array"""
        timestamps = np.array([0, 100, 200], dtype=np.uint64)
        poses = np.stack([get_SE3(np.array([t, 0, 0])) for t in timestamps]).astype(np.float32)

        interpolator = PoseInterpolator(poses, timestamps)

        result = interpolator.extrapolate_to_timestamps(np.array([], dtype=np.uint64))
        self.assertEqual(result.shape, (0, 4, 4))

    def test_extrapolate_dtype_preservation(self):
        """Test that dtype parameter is respected and results are correct for both f32 and f64"""
        timestamps = np.array([0, 100, 200], dtype=np.uint64)
        velocity = np.array([1.0, 0.0, 0.0])

        poses = np.stack([get_SE3(velocity * t) for t in timestamps]).astype(np.float32)

        interpolator = PoseInterpolator(poses, timestamps)

        # Test forward extrapolation
        ts_forward = np.array([300, 400], dtype=np.uint64)

        result_f32 = interpolator.extrapolate_to_timestamps(ts_forward, dtype=np.float32)
        result_f64 = interpolator.extrapolate_to_timestamps(ts_forward, dtype=np.float64)

        # Verify dtypes
        self.assertEqual(result_f32.dtype, np.float32)
        self.assertEqual(result_f64.dtype, np.float64)

        # Verify correctness for both dtypes
        expected_positions = np.array([[300, 0, 0], [400, 0, 0]])

        np.testing.assert_array_almost_equal(result_f32[:, :3, 3], expected_positions, decimal=4)
        np.testing.assert_array_almost_equal(result_f64[:, :3, 3], expected_positions, decimal=10)

        # Verify rotation stays identity for both
        for i in range(len(result_f32)):
            np.testing.assert_array_almost_equal(result_f32[i, :3, :3], np.eye(3), decimal=5)
            np.testing.assert_array_almost_equal(result_f64[i, :3, :3], np.eye(3), decimal=10)

    def test_extrapolate_linear_motion_f32_f64(self):
        """Test linear motion extrapolation with both float32 and float64 dtypes"""
        for test_dtype in [np.float32, np.float64]:
            with self.subTest(dtype=test_dtype):
                timestamps = np.array([100, 200, 300], dtype=np.uint64)
                velocity = np.array([1.0, 2.0, -0.5])

                poses = np.stack([get_SE3(velocity * t) for t in timestamps]).astype(test_dtype)
                interpolator = PoseInterpolator(poses, timestamps)

                # Test backward extrapolation
                ts_backward = np.array([0, 50], dtype=np.uint64)
                result_back = interpolator.extrapolate_to_timestamps(ts_backward, dtype=test_dtype)

                self.assertEqual(result_back.dtype, test_dtype)
                expected_back = np.array([[0, 0, 0], [50, 100, -25]], dtype=test_dtype)
                decimal = 4 if test_dtype == np.float32 else 10
                np.testing.assert_array_almost_equal(result_back[:, :3, 3], expected_back, decimal=decimal)

                # Test forward extrapolation
                ts_forward = np.array([400, 500], dtype=np.uint64)
                result_fwd = interpolator.extrapolate_to_timestamps(ts_forward, dtype=test_dtype)

                self.assertEqual(result_fwd.dtype, test_dtype)
                expected_fwd = np.array([[400, 800, -200], [500, 1000, -250]], dtype=test_dtype)
                np.testing.assert_array_almost_equal(result_fwd[:, :3, 3], expected_fwd, decimal=decimal)

    def test_extrapolate_rotation_f32_f64(self):
        """Test rotation extrapolation with both float32 and float64 dtypes"""
        for test_dtype in [np.float32, np.float64]:
            with self.subTest(dtype=test_dtype):
                timestamps = np.array([0, 100, 200], dtype=np.uint64)
                angular_velocity_deg_per_us = 0.9  # 90 deg per 100 us

                poses = np.stack(
                    [
                        get_SE3_with_rotation(
                            np.array([0, 0, 0]), R.from_euler("z", angular_velocity_deg_per_us * t, degrees=True)
                        )
                        for t in timestamps
                    ]
                ).astype(test_dtype)

                interpolator = PoseInterpolator(poses, timestamps)

                # Test forward extrapolation to 300 us (expected: 270 deg)
                ts_forward = np.array([300], dtype=np.uint64)
                result = interpolator.extrapolate_to_timestamps(ts_forward, dtype=test_dtype)

                self.assertEqual(result.dtype, test_dtype)

                expected_rot = R.from_euler("z", 270, degrees=True)
                result_rot = R.from_matrix(result[0, :3, :3])
                angle_diff = (result_rot * expected_rot.inv()).magnitude()

                tolerance = 1e-4 if test_dtype == np.float32 else 1e-10
                self.assertLess(angle_diff, tolerance, f"Rotation mismatch for dtype {test_dtype}")
