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

import unittest

from pathlib import Path
from typing import cast

import numpy as np

from python.runfiles import Runfiles
from scipy.spatial.transform import Rotation as R

from ncore.impl.common.transformations import (
    HalfClosedInterval,
    MotionCompensator,
    PoseGraphInterpolator,
    PoseInterpolator,
    is_within_3d_bboxes,
    time_bounds,
    transform_point_cloud,
)
from ncore.impl.common.util import unpack_optional
from ncore.impl.data import types
from ncore.impl.data.v4.compat import SequenceLoaderV4
from ncore.impl.data.v4.components import SequenceComponentGroupsReader


_RUNFILES: Runfiles = unpack_optional(Runfiles.Create())


def _angle_between(a: R, b: R) -> float:
    """Returns the rotation angle between two rotations (radians)"""
    # Composes the rotation matrices instead of using `Rotation.__mul__`, whose return type annotation differs between
    # scipy versions (scipy 1.10: `Rotation`, scipy 1.18: `Rotation | NotImplementedType`)
    return float(R.from_matrix(a.as_matrix() @ b.inv().as_matrix()).magnitude())


class TestIsWithin3DBBox(unittest.TestCase):
    def setUp(self):
        # Set the random seed
        rng = np.random.default_rng(41)

        # create some test point-cloud
        self.pc = rng.random((100000, 3)).astype(np.float32) * 3.0  # increase it to [0,3] range

        # create some bounding boxes
        center = rng.random((100, 3)).astype(np.float32) * 3.0
        dim = rng.random((100, 3)).astype(np.float32)
        rotation = rng.random((100, 3)).astype(np.float32) * 2 * np.pi

        self.bboxes = np.concatenate([center, dim, rotation], axis=-1).astype(np.float32)

        # Create an outlier point that's outside the bboxes (guaranteed) by minimum of xyz of the
        # point being larger than maximum possible dimensions for bboxes
        self.outlier_point = rng.uniform(1000, 2000, size=(1, 3)).astype(np.float32)

        # Create inliner points that are guaranteed to be inside the bounding boxes to test that
        # points inside the bounding boxes are classified correctly
        self.inliner_points = center[rng.choice(100, 10, replace=True)]

        # Making the dimensions larger than the upper limit of the inliner points to ensure that the
        # points are guaranteed to be inside the bounding box
        self.inliner_bboxes = np.concatenate(
            [
                np.zeros((10, 3), dtype=np.float32),  # centers
                np.full((10, 3), 10.0, dtype=np.float32),  # dims large enough to contain all inlier points
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

    def test_invalid_shapes_raise(self):
        with self.assertRaisesRegex(ValueError, r"pc must have shape \[N, 3\]"):
            is_within_3d_bboxes(self.pc[:, :2], self.bboxes)
        with self.assertRaisesRegex(ValueError, "bboxes must be a 2D array, got ndim=1"):
            is_within_3d_bboxes(self.pc, self.bboxes[0])
        with self.assertRaisesRegex(ValueError, r"bboxes must have shape \[M, 9\]"):
            is_within_3d_bboxes(self.pc, self.bboxes[:, :8])


class TestTimeBounds(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(time_bounds([0, 10_000_000], None, None), (0, 10_000_000))
        self.assertEqual(time_bounds([0, 10_000_000], 1.0, 2.0), (1_000_000, 3_000_000))

    def test_invalid_arguments_raise(self):
        with self.assertRaisesRegex(ValueError, "seek_sec must be non-negative, got -1.0"):
            time_bounds([0, 10_000_000], -1.0, None)
        with self.assertRaisesRegex(ValueError, "duration_sec must be positive, got 0.0"):
            time_bounds([0, 10_000_000], None, 0.0)
        with self.assertRaisesRegex(ValueError, "duration_sec must be positive, got nan"):
            time_bounds([0, 10_000_000], None, float("nan"))
        with self.assertRaisesRegex(ValueError, "invalid time bounds"):
            time_bounds([0, 10_000_000], 20.0, None)


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
        with self.assertRaisesRegex(ValueError, "start must not exceed stop, got start=0, stop=-1"):
            HalfClosedInterval(0, -1)
        with self.assertRaisesRegex(ValueError, "start must be an int, got float"):
            HalfClosedInterval(cast(int, 0.0), 1)
        with self.assertRaisesRegex(ValueError, "stop must be an int, got float"):
            HalfClosedInterval(0, cast(int, 1.0))

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
        _test_data_path = unpack_optional(
            _RUNFILES.Rlocation(
                "test-data-v4/c9b05cf4-afb9-11ec-b3c2-00044bf65fcb@1648597318700123-1648599151600035.json"
            )
        )
        self.loader = SequenceLoaderV4(
            SequenceComponentGroupsReader(
                [Path(_test_data_path)],
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
                motion_compensation_result.xyz_e_reftime,
                timestamp_us,
                frame_start_timestamps_us,
                frame_end_timestamps_us,
            ).xyz_pointtime

            xyz_s_m = motion_compensator.motion_decompensate_points(
                lidar_sensor.sensor_id,
                motion_compensation_result.xyz_s_reftime,
                timestamp_us,
                frame_start_timestamps_us,
                frame_end_timestamps_us,
            ).xyz_pointtime

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

    def test_idempotence_custom_reference_timestamp(self):
        """compensate/decompensate round-trip with non-default reference timestamps.

        Compensating to a custom reference and decompensating back with the same
        reference must recover the original points. Several references spanning the
        [frame_start, frame_end] range are exercised (including the boundaries and
        a few interior timestamps).
        """
        motion_compensator = MotionCompensator(self.loader.pose_graph)
        lidar_sensor = self.loader.get_lidar_sensor("lidar_gt_top_p128_v4p5")

        for frame_idx in range(0, 2):
            xyz_m_ref = lidar_sensor.get_frame_point_cloud(
                frame_idx, motion_compensation=False, with_start_points=False, return_index=0
            ).xyz_m_end
            timestamp_us = lidar_sensor.get_frame_ray_bundle_timestamp_us(frame_idx)
            frame_start_us = lidar_sensor.get_frame_timestamp_us(frame_idx, types.FrameTimepoint.START)
            frame_end_us = lidar_sensor.get_frame_timestamp_us(frame_idx, types.FrameTimepoint.END)

            # Exercise references across the [start, end] range: both boundaries plus
            # several interior fractions.
            reference_candidates = {
                int(round(frame_start_us + fraction * (frame_end_us - frame_start_us)))
                for fraction in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
            }

            for reference_us in sorted(reference_candidates):
                compensated = motion_compensator.motion_compensate_points(
                    lidar_sensor.sensor_id,
                    xyz_m_ref,
                    timestamp_us,
                    frame_start_us,
                    frame_end_us,
                    reference_timestamp_us=reference_us,
                )
                self.assertEqual(compensated.reference_timestamp_us, reference_us)

                xyz_roundtrip = motion_compensator.motion_decompensate_points(
                    lidar_sensor.sensor_id,
                    compensated.xyz_e_reftime,
                    timestamp_us,
                    frame_start_us,
                    frame_end_us,
                    reference_timestamp_us=reference_us,
                ).xyz_pointtime

                self.assertIsNone(
                    np.testing.assert_array_almost_equal(
                        np.zeros_like(delta := np.linalg.norm(xyz_roundtrip - xyz_m_ref, axis=1)),
                        delta,
                        decimal=2,
                    ),
                    f"custom-reference round-trip inconsistent, frame_idx {frame_idx}, reference_us {reference_us}",
                )

    def test_default_reference_matches_explicit_end_of_frame(self):
        """Omitting reference_timestamp_us equals passing frame_end explicitly."""
        motion_compensator = MotionCompensator(self.loader.pose_graph)
        lidar_sensor = self.loader.get_lidar_sensor("lidar_gt_top_p128_v4p5")

        xyz_m_ref = lidar_sensor.get_frame_point_cloud(
            0, motion_compensation=False, with_start_points=False, return_index=0
        ).xyz_m_end
        timestamp_us = lidar_sensor.get_frame_ray_bundle_timestamp_us(0)
        frame_start_us = lidar_sensor.get_frame_timestamp_us(0, types.FrameTimepoint.START)
        frame_end_us = lidar_sensor.get_frame_timestamp_us(0, types.FrameTimepoint.END)

        default = motion_compensator.motion_decompensate_points(
            lidar_sensor.sensor_id, xyz_m_ref, timestamp_us, frame_start_us, frame_end_us
        ).xyz_pointtime
        explicit = motion_compensator.motion_decompensate_points(
            lidar_sensor.sensor_id,
            xyz_m_ref,
            timestamp_us,
            frame_start_us,
            frame_end_us,
            reference_timestamp_us=frame_end_us,
        ).xyz_pointtime
        np.testing.assert_array_equal(default, explicit)

    def test_result_exposes_frame_ids(self):
        """The result records the reference (sensor) frame and the anchor frame."""
        motion_compensator = MotionCompensator(self.loader.pose_graph)
        lidar_sensor = self.loader.get_lidar_sensor("lidar_gt_top_p128_v4p5")

        xyz_m_ref = lidar_sensor.get_frame_point_cloud(
            0, motion_compensation=False, with_start_points=False, return_index=0
        ).xyz_m_end
        timestamp_us = lidar_sensor.get_frame_ray_bundle_timestamp_us(0)
        frame_start_us = lidar_sensor.get_frame_timestamp_us(0, types.FrameTimepoint.START)
        frame_end_us = lidar_sensor.get_frame_timestamp_us(0, types.FrameTimepoint.END)

        result = motion_compensator.motion_compensate_points(
            lidar_sensor.sensor_id, xyz_m_ref, timestamp_us, frame_start_us, frame_end_us
        )
        self.assertEqual(result.reference_frame_id, lidar_sensor.sensor_id)
        self.assertEqual(result.anchor_frame_id, "world")
        self.assertEqual(result.reference_timestamp_us, frame_end_us)

        # The empty-input path carries the same frame metadata.
        empty = motion_compensator.motion_compensate_points(
            lidar_sensor.sensor_id,
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.uint64),
            frame_start_us,
            frame_end_us,
        )
        self.assertEqual(empty.reference_frame_id, lidar_sensor.sensor_id)
        self.assertEqual(empty.anchor_frame_id, "world")


EXACT = MotionCompensator.Parameters.Method.EXACT
INTERPOLATED = MotionCompensator.Parameters.Method.INTERPOLATED


class TestMotionCompensatorMethods(unittest.TestCase):
    """The ``INTERPOLATED`` method against the ``EXACT`` one"""

    def setUp(self):
        _test_data_path = unpack_optional(
            _RUNFILES.Rlocation(
                "test-data-v4/c9b05cf4-afb9-11ec-b3c2-00044bf65fcb@1648597318700123-1648599151600035.json"
            )
        )
        self.loader = SequenceLoaderV4(SequenceComponentGroupsReader([Path(_test_data_path)]))

    def _frame(self, frame_idx: int):
        lidar_sensor = self.loader.get_lidar_sensor("lidar_gt_top_p128_v4p5")
        xyz = lidar_sensor.get_frame_point_cloud(
            frame_idx, motion_compensation=False, with_start_points=False, return_index=0
        ).xyz_m_end
        return (
            lidar_sensor.sensor_id,
            xyz,
            lidar_sensor.get_frame_ray_bundle_timestamp_us(frame_idx),
            lidar_sensor.get_frame_timestamp_us(frame_idx, types.FrameTimepoint.START),
            lidar_sensor.get_frame_timestamp_us(frame_idx, types.FrameTimepoint.END),
        )

    def test_invalid_parameters_raise(self):
        for tolerance in (0.0, -1e-5):
            with self.assertRaises(ValueError):
                MotionCompensator.Parameters(INTERPOLATED, interpolation_tolerance=tolerance)

    def test_default_parameters_are_exact(self):
        compensator = MotionCompensator(self.loader.pose_graph)
        self.assertEqual(compensator.parameters, MotionCompensator.Parameters(EXACT))

    def test_interpolated_matches_exact_on_recorded_frames(self):
        """On recorded lidar frames the interpolated method agrees with the exact one far below a millimeter, for
        compensation (start / end points) and decompensation, and for interior reference timestamps"""
        exact = MotionCompensator(self.loader.pose_graph, parameters=MotionCompensator.Parameters(EXACT))
        interpolated = MotionCompensator(self.loader.pose_graph, parameters=MotionCompensator.Parameters(INTERPOLATED))
        self.assertIs(interpolated.parameters.method, INTERPOLATED)

        for frame_idx in range(0, 2):
            sensor_id, xyz, timestamp_us, start_us, end_us = self._frame(frame_idx)
            # the frame's points span a non-trivial time range, and the ego vehicle moves within it
            self.assertGreater(int(timestamp_us.max() - timestamp_us.min()), 10_000)

            for reference_us in (end_us, (start_us + end_us) // 2):
                expected = exact.motion_compensate_points(
                    sensor_id, xyz, timestamp_us, start_us, end_us, reference_timestamp_us=reference_us
                )
                actual = interpolated.motion_compensate_points(
                    sensor_id, xyz, timestamp_us, start_us, end_us, reference_timestamp_us=reference_us
                )
                # sanity: motion compensation moves the points (by more than the tolerance below)
                self.assertGreater(np.abs(expected.xyz_e_reftime - xyz).max(), 1e-2)
                np.testing.assert_allclose(actual.xyz_e_reftime, expected.xyz_e_reftime, rtol=0, atol=1e-4)
                np.testing.assert_allclose(actual.xyz_s_reftime, expected.xyz_s_reftime, rtol=0, atol=1e-4)
                self.assertEqual(actual.xyz_e_reftime.dtype, np.float32)
                self.assertEqual(actual.xyz_s_reftime.dtype, np.float32)

                np.testing.assert_allclose(
                    interpolated.motion_decompensate_points(
                        sensor_id,
                        expected.xyz_e_reftime,
                        timestamp_us,
                        start_us,
                        end_us,
                        reference_timestamp_us=reference_us,
                    ).xyz_pointtime,
                    exact.motion_decompensate_points(
                        sensor_id,
                        expected.xyz_e_reftime,
                        timestamp_us,
                        start_us,
                        end_us,
                        reference_timestamp_us=reference_us,
                    ).xyz_pointtime,
                    rtol=0,
                    atol=1e-4,
                )

    def test_interpolated_is_exact_for_linear_motion(self):
        """For a constant-velocity translation the per-point transformations are linear in time, so the
        interpolation is exact (also for timestamps at the frame boundaries and points in between grid timestamps)"""
        timestamps_us = np.array([0, 1_000_000], dtype=np.uint64)
        T_rig_world = np.stack([get_SE3(np.array([0.0, 0.0, 0.0])), get_SE3(np.array([30.0, -4.0, 1.0]))])
        pose_graph = PoseGraphInterpolator(
            [
                PoseGraphInterpolator.Edge("rig", "world", T_rig_world, timestamps_us),
                PoseGraphInterpolator.Edge("lidar", "rig", get_SE3(np.array([1.0, 0.0, 2.0])), None),
            ]
        )
        rng = np.random.default_rng(0)
        xyz = rng.normal(scale=50.0, size=(5000, 3)).astype(np.float32)
        start_us, end_us = 200_000, 300_000
        timestamp_us = rng.integers(start_us, end_us + 1, size=5000).astype(np.uint64)
        timestamp_us[:2] = (start_us, end_us)

        exact = MotionCompensator(pose_graph, parameters=MotionCompensator.Parameters(EXACT)).motion_compensate_points(
            "lidar", xyz, timestamp_us, start_us, end_us
        )
        interpolated = MotionCompensator(
            pose_graph, parameters=MotionCompensator.Parameters(INTERPOLATED)
        ).motion_compensate_points("lidar", xyz, timestamp_us, start_us, end_us)
        np.testing.assert_allclose(interpolated.xyz_e_reftime, exact.xyz_e_reftime, rtol=0, atol=1e-4)
        np.testing.assert_allclose(interpolated.xyz_s_reftime, exact.xyz_s_reftime, rtol=0, atol=1e-5)

    def _rotating_pose_graph(self) -> PoseGraphInterpolator:
        """A turning vehicle: 30 deg/s yaw rate, 10 m/s"""
        timestamps_us = np.array([0, 1_000_000], dtype=np.uint64)
        T_end = np.eye(4)
        T_end[:3, :3] = R.from_euler("z", np.deg2rad(30.0)).as_matrix()
        T_end[:3, 3] = (10.0, 3.0, 0.0)
        return PoseGraphInterpolator(
            [
                PoseGraphInterpolator.Edge("rig", "world", np.stack([np.eye(4), T_end]), timestamps_us),
                PoseGraphInterpolator.Edge("lidar", "rig", get_SE3(np.array([0.0, 0.0, 0.0])), None),
            ]
        )

    def test_interpolated_deviation_is_bounded_by_tolerance(self):
        """For a rotation the interpolation deviates from the exact transformation by the chord error of the rotation
        within the grid intervals, which the adaptive grid bounds by the tolerance (evaluated in float64 to separate
        the bound from float32 rounding)"""
        pose_graph = self._rotating_pose_graph()
        rng = np.random.default_rng(1)
        xyz = rng.normal(scale=60.0, size=(5000, 3))
        start_us, end_us = 400_000, 500_000
        timestamp_us = rng.integers(start_us, end_us + 1, size=5000).astype(np.uint64)

        T_ref = pose_graph.evaluate_poses("world", "lidar", np.array(end_us, dtype=np.uint64)).astype(np.float64)

        def evaluate(timestamps_us: np.ndarray) -> np.ndarray:
            return T_ref @ pose_graph.evaluate_poses("lidar", "world", timestamps_us).astype(np.float64)

        T_exact = evaluate(timestamp_us)
        exact = np.einsum("nij,nj->ni", T_exact[:, :3, :3], xyz) + T_exact[:, :3, 3]
        # the rotation within the frame moves far points by meters
        self.assertGreater(np.abs(exact - xyz).max(), 1.0)

        max_distance = float(np.max(np.linalg.norm(xyz, axis=1)))
        n_intervals = []
        for tolerance in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
            grid = MotionCompensator._InterpolationGrid.refined(  # noqa: SLF001
                evaluate, np.array([start_us, end_us], dtype=np.uint64), max_distance, tolerance
            )
            n_intervals.append(len(grid.timestamps_us) - 1)
            # interpolate in float64 (unlike transform_points) to separate the bound from float32 rounding
            interval = np.minimum(
                np.searchsorted(grid.timestamps_us, timestamp_us, side="right") - 1, n_intervals[-1] - 1
            )
            start, end = grid.timestamps_us[interval], grid.timestamps_us[interval + 1]
            weight = ((timestamp_us - start) / (end - start))[:, None, None]
            T_start, T_end = grid.transformations[interval], grid.transformations[interval + 1]
            T = T_start + weight * (T_end - T_start)
            interpolated = np.einsum("nij,nj->ni", T[:, :3, :3], xyz) + T[:, :3, 3]
            self.assertLessEqual(np.abs(interpolated - exact).max(), tolerance, f"tolerance {tolerance}")
        # finer tolerances refine the grid
        self.assertEqual(n_intervals, sorted(n_intervals))
        self.assertGreater(n_intervals[-1], n_intervals[0])

    def test_interpolated_matches_exact_within_tolerance_and_float32_rounding(self):
        """The float32 points of both methods differ by at most the tolerance plus the float32 rounding of the
        transformations (relative ~1e-7 of the point distances)"""
        pose_graph = self._rotating_pose_graph()
        rng = np.random.default_rng(2)
        xyz = rng.normal(scale=60.0, size=(5000, 3)).astype(np.float32)
        start_us, end_us = 400_000, 500_000
        timestamp_us = rng.integers(start_us, end_us + 1, size=5000).astype(np.uint64)
        # float32 rounding: each transformed coordinate is a sum of a few float32 products of magnitude up to the
        # point distance, each rounded by up to eps/2 relative (eps = 2^-23 ~= 1.2e-7); a few eps of the largest
        # distance bound the accumulated rounding of both methods
        rounding = 4 * float(np.finfo(np.float32).eps) * float(np.max(np.linalg.norm(xyz, axis=1)))

        exact = MotionCompensator(pose_graph, parameters=MotionCompensator.Parameters(EXACT)).motion_compensate_points(
            "lidar", xyz, timestamp_us, start_us, end_us
        )
        for tolerance in (1e-3, 1e-5):
            interpolated = MotionCompensator(
                pose_graph, parameters=MotionCompensator.Parameters(INTERPOLATED, tolerance)
            ).motion_compensate_points("lidar", xyz, timestamp_us, start_us, end_us)
            np.testing.assert_allclose(
                interpolated.xyz_e_reftime, exact.xyz_e_reftime, rtol=0, atol=tolerance + rounding
            )

    def test_interpolation_grid_contains_pose_timestamps(self):
        """The pose timestamps within the frame (where the interpolated motion changes) are grid timestamps, also
        for motions that don't need refinement"""
        timestamps_us = np.array([0, 150_000, 300_000], dtype=np.uint64)
        T_rig_world = np.stack(
            [get_SE3(np.array([0.0, 0.0, 0.0])), get_SE3(np.array([3.0, 0.0, 0.0])), get_SE3(np.array([3.0, 3.0, 0.0]))]
        )
        pose_graph = PoseGraphInterpolator(
            [
                PoseGraphInterpolator.Edge("rig", "world", T_rig_world, timestamps_us),
                PoseGraphInterpolator.Edge("lidar", "rig", get_SE3(np.array([0.0, 0.0, 0.0])), None),
            ]
        )
        self.assertEqual(
            [t.tolist() for t in pose_graph.get_path_timestamps_us("lidar", "world")], [timestamps_us.tolist()]
        )
        # the translation changes direction at 150 ms: piecewise linear, exact with the breakpoint, not without it
        rng = np.random.default_rng(3)
        xyz = rng.normal(scale=20.0, size=(2000, 3)).astype(np.float32)
        start_us, end_us = 100_000, 200_000
        timestamp_us = rng.integers(start_us, end_us + 1, size=2000).astype(np.uint64)
        exact = MotionCompensator(pose_graph, parameters=MotionCompensator.Parameters(EXACT)).motion_compensate_points(
            "lidar", xyz, timestamp_us, start_us, end_us
        )
        interpolated = MotionCompensator(
            pose_graph, parameters=MotionCompensator.Parameters(INTERPOLATED)
        ).motion_compensate_points("lidar", xyz, timestamp_us, start_us, end_us)
        np.testing.assert_allclose(interpolated.xyz_e_reftime, exact.xyz_e_reftime, rtol=0, atol=1e-4)

    def test_empty_and_single_timestamp(self):
        pose_graph = self.loader.pose_graph
        sensor_id, xyz, timestamp_us, start_us, end_us = self._frame(0)
        for method in (EXACT, INTERPOLATED):
            compensator = MotionCompensator(pose_graph, parameters=MotionCompensator.Parameters(method))
            empty = compensator.motion_compensate_points(
                sensor_id, np.empty((0, 3), np.float32), np.empty((0,), np.uint64), start_us, end_us
            )
            self.assertEqual(empty.xyz_e_reftime.shape, (0, 3))
            # all points at the reference timestamp map through the identity
            single = compensator.motion_compensate_points(
                sensor_id, xyz[:100], np.full(100, end_us, dtype=np.uint64), start_us, end_us
            )
            np.testing.assert_allclose(single.xyz_e_reftime, xyz[:100], rtol=0, atol=1e-4)

    def test_static_path_and_zero_length_frame(self):
        """Paths of static edges only (constant transformations, no pose timestamps) and frames without duration
        (start == end) match the exact method"""
        timestamps_us = np.array([0, 1_000_000], dtype=np.uint64)
        pose_graph = PoseGraphInterpolator(
            [
                PoseGraphInterpolator.Edge(
                    "rig", "world", np.stack([np.eye(4), get_SE3(np.array([10.0, 0.0, 0.0]))]), timestamps_us
                ),
                PoseGraphInterpolator.Edge("lidar", "rig", get_SE3(np.array([1.0, 0.0, 2.0])), None),
            ]
        )
        self.assertEqual(pose_graph.get_path_timestamps_us("lidar", "rig"), [])
        rng = np.random.default_rng(4)
        xyz = rng.normal(scale=20.0, size=(1000, 3)).astype(np.float32)
        for anchor, start_us, end_us in (("rig", 100_000, 200_000), ("world", 150_000, 150_000)):
            timestamp_us = rng.integers(start_us, end_us + 1, size=1000).astype(np.uint64)
            expected, actual = (
                MotionCompensator(pose_graph, parameters=MotionCompensator.Parameters(method)).motion_compensate_points(
                    "lidar", xyz, timestamp_us, start_us, end_us, anchor_frame_id=anchor
                )
                for method in (EXACT, INTERPOLATED)
            )
            np.testing.assert_allclose(actual.xyz_e_reftime, expected.xyz_e_reftime, rtol=0, atol=1e-5)
            np.testing.assert_allclose(actual.xyz_s_reftime, expected.xyz_s_reftime, rtol=0, atol=1e-5)

    def test_sequence_loader_forwards_method(self):
        """SequenceLoaderV4 point clouds use the requested method"""
        _test_data_path = unpack_optional(
            _RUNFILES.Rlocation(
                "test-data-v4/c9b05cf4-afb9-11ec-b3c2-00044bf65fcb@1648597318700123-1648599151600035.json"
            )
        )
        reader = SequenceComponentGroupsReader([Path(_test_data_path)])
        exact_loader = SequenceLoaderV4(reader)
        interpolated_loader = SequenceLoaderV4(
            reader, motion_compensator_parameters=MotionCompensator.Parameters(INTERPOLATED)
        )

        exact_pc = exact_loader.get_lidar_sensor("lidar_gt_top_p128_v4p5").get_frame_point_cloud(
            0, motion_compensation=True, with_start_points=True
        )
        interpolated_pc = interpolated_loader.get_lidar_sensor("lidar_gt_top_p128_v4p5").get_frame_point_cloud(
            0, motion_compensation=True, with_start_points=True
        )
        np.testing.assert_allclose(interpolated_pc.xyz_m_end, exact_pc.xyz_m_end, rtol=0, atol=1e-4)
        # the methods differ (the interpolation is not the exact evaluation)
        self.assertFalse(np.array_equal(interpolated_pc.xyz_m_end, exact_pc.xyz_m_end))

        # the parameters are passed through to the sensors' motion compensators
        parameters = MotionCompensator.Parameters(INTERPOLATED, interpolation_tolerance=3e-4)
        loader = SequenceLoaderV4(reader, motion_compensator_parameters=parameters)
        lidar, lidar_again = (loader.get_lidar_sensor("lidar_gt_top_p128_v4p5") for _ in range(2))
        assert isinstance(lidar, SequenceLoaderV4.LidarSensor) and isinstance(lidar_again, SequenceLoaderV4.LidarSensor)
        self.assertIs(lidar._motion_compensator.parameters, parameters)  # noqa: SLF001
        # the sensors share the loader's motion compensator
        self.assertIs(lidar._motion_compensator, lidar_again._motion_compensator)  # noqa: SLF001


def _reference_motion_compensation(
    pose_graph: PoseGraphInterpolator,
    sensor_id: str,
    xyz: np.ndarray,
    timestamp_us: np.ndarray,
    reference_timestamp_us: int,
    anchor_frame_id: str,
):
    """Reference (previous) implementation of MotionCompensator: one 4x4 transformation per point, at the distinct
    point timestamps

    Returns:
        compensated end / start points, decompensated points
    """
    timestamp_unique, inverse = np.unique(timestamp_us, return_inverse=True)
    T_anchor_sensor_reftime = pose_graph.evaluate_poses(
        anchor_frame_id, sensor_id, np.array(reference_timestamp_us, dtype=np.uint64)
    )
    T_compensate = (
        T_anchor_sensor_reftime @ pose_graph.evaluate_poses(sensor_id, anchor_frame_id, timestamp_unique)
    ).astype(np.float32)
    T_sensor_reftime_anchor = pose_graph.evaluate_poses(
        sensor_id, anchor_frame_id, np.array(reference_timestamp_us, dtype=np.uint64)
    )
    T_decompensate = (
        pose_graph.evaluate_poses(anchor_frame_id, sensor_id, timestamp_unique) @ T_sensor_reftime_anchor
    ).astype(np.float32)
    return (
        transform_point_cloud(xyz[:, np.newaxis, :], T_compensate[inverse]).squeeze(1),
        T_compensate[inverse, :3, -1],
        transform_point_cloud(xyz[:, np.newaxis, :], T_decompensate[inverse]).squeeze(1),
    )


class TestMotionCompensatorRegression(unittest.TestCase):
    """Both methods against the previous implementation, for the inputs of all current callers (sequence loaders,
    data converters, tools) and edge cases: ``EXACT`` reproduces it up to the rounding of the summation order,
    ``INTERPOLATED`` within its tolerance (plus float32 rounding)"""

    # Epoch microseconds as in recorded data (the interpolation weights are relative to the grid timestamps)
    T0 = 1_648_597_318_700_123

    def _pose_graph(self) -> PoseGraphInterpolator:
        """sensor -(static)-> rig -(dynamic, turning and accelerating)-> world -(static)-> world_global"""
        timestamps_us = self.T0 + np.array([0, 100_000, 200_000, 300_000], dtype=np.uint64)
        T_rig_worlds = np.stack(
            [
                np.block([[R.from_euler("z", yaw).as_matrix(), np.array(t).reshape(3, 1)], [np.array([0, 0, 0, 1])]])
                for yaw, t in ((0.0, (0, 0, 0)), (0.05, (1, 0.1, 0)), (0.2, (3, 0.5, 0.1)), (0.3, (6, 1.5, 0.1)))
            ]
        )
        T_sensor_rig = np.block(
            [
                [R.from_euler("xyz", (0.01, -0.02, 0.5)).as_matrix(), np.array([[1.0], [0.2], [1.8]])],
                [np.array([0, 0, 0, 1])],
            ]
        )
        return PoseGraphInterpolator(
            [
                PoseGraphInterpolator.Edge("rig", "world", T_rig_worlds, timestamps_us),
                PoseGraphInterpolator.Edge("lidar", "rig", T_sensor_rig, None),
                PoseGraphInterpolator.Edge("world", "world_global", get_SE3(np.array([1e3, -2e3, 50.0])), None),
            ]
        )

    def _cases(self):
        """(name, xyz, timestamps, frame start, frame end, keyword arguments) covering the callers' inputs"""
        rng = np.random.default_rng(5)
        n = 4000
        start, end = self.T0 + 50_000, self.T0 + 150_000  # contains the pose timestamp at T0 + 100 ms
        xyz = rng.normal(scale=40.0, size=(n, 3)).astype(np.float32)
        # spinning lidar: ~1 timestamp per 8 points, shuffled order (as stored per ray)
        timestamp_us = start + np.repeat(np.arange(n // 8, dtype=np.uint64) * ((end - start) // (n // 8)), 8)
        rng.shuffle(timestamp_us)
        nonfinite = xyz.copy()
        nonfinite[[1, 2, 3]] = (np.nan, np.inf, -np.inf)
        sparse = np.array([start, start + 1, end - 1, end, (start + end) // 2], dtype=np.uint64)
        # timestamps spanning fewer microseconds than ~4x the points (lookup tables instead of sorting / searching)
        dense = start + rng.integers(0, 5_000, size=n).astype(np.uint64)
        return [
            ("spinning lidar", xyz, timestamp_us, start, end, {}),
            ("sorted timestamps", xyz, np.sort(timestamp_us), start, end, {}),
            ("int64 timestamps", xyz, timestamp_us.astype(np.int64), start, end, {}),
            ("dense timestamps", xyz, dense, start, end, {}),
            ("dense int64 timestamps", xyz, dense.astype(np.int64), start, end, {}),
            ("float64 points", xyz.astype(np.float64), timestamp_us, start, end, {}),
            ("reference at start", xyz, timestamp_us, start, end, {"reference_timestamp_us": start}),
            ("interior reference", xyz, timestamp_us, start, end, {"reference_timestamp_us": start + 12_345}),
            ("static anchor offset", xyz, timestamp_us, start, end, {"anchor_frame_id": "world_global"}),
            ("sparse timestamps", xyz[:5], sparse, start, end, {}),
            ("single point", xyz[:1], timestamp_us[:1], start, end, {}),
            ("single timestamp", xyz[:50], np.full(50, start + 7, dtype=np.uint64), start, end, {}),
            ("instant frame", xyz[:50], np.full(50, start, dtype=np.uint64), start, start, {}),
            (
                "points within wider frame",
                xyz,
                timestamp_us,
                self.T0,
                self.T0 + 300_000,
                {"reference_timestamp_us": end},
            ),
            (
                "frame across pose timestamps",
                xyz,
                self.T0 + np.sort(timestamp_us - start) * 3,
                self.T0,
                self.T0 + 300_000,
                {},
            ),
            ("non-finite points", nonfinite, timestamp_us, start, end, {}),
        ]

    def test_methods_against_previous_implementation(self):
        pose_graph = self._pose_graph()
        compensators = {
            method: MotionCompensator(pose_graph, parameters=MotionCompensator.Parameters(method))
            for method in (EXACT, INTERPOLATED)
        }
        for name, xyz, timestamp_us, start, end, kwargs in self._cases():
            reference_us = kwargs.get("reference_timestamp_us", end)
            anchor = kwargs.get("anchor_frame_id", "world")
            expected_e, expected_s, expected_decompensated = _reference_motion_compensation(
                pose_graph, "lidar", xyz, timestamp_us, reference_us, anchor
            )
            # float32 rounding of the transformed points: each coordinate sums 3 products and a translation, rounded
            # by up to eps/2 relative each (eps = 2^-23); summed in a different order (einsum / matmul, numpy version)
            # the results differ by a few eps of the largest coordinates. For INTERPOLATED plus the interpolation
            # tolerance.
            rounding = 8 * float(np.finfo(np.float32).eps) * float(np.nanmax(np.abs(xyz[np.isfinite(xyz)])))
            atol = 1e-5 + rounding
            for method, compensator in compensators.items():
                with self.subTest(case=name, method=method.name):
                    result = compensator.motion_compensate_points("lidar", xyz, timestamp_us, start, end, **kwargs)
                    decompensation = compensator.motion_decompensate_points(
                        "lidar", xyz, timestamp_us, start, end, **kwargs
                    )
                    self.assertIsInstance(decompensation, MotionCompensator.MotionDecompensationResult)
                    decompensated = decompensation.xyz_pointtime
                    for frames in (result, decompensation):
                        self.assertEqual(
                            (frames.reference_frame_id, frames.reference_timestamp_us, frames.anchor_frame_id),
                            ("lidar", reference_us, anchor),
                        )
                    for actual, expected in (
                        (result.xyz_e_reftime, expected_e),
                        (result.xyz_s_reftime, expected_s),
                        (decompensated, expected_decompensated),
                    ):
                        self.assertEqual(actual.shape, expected.shape)
                        self.assertEqual(actual.dtype, expected.dtype)
                        np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
                        if method is EXACT:
                            # the same transformations, up to the rounding of the summation order (depends on the
                            # numpy version)
                            finite = np.isfinite(expected)
                            np.testing.assert_allclose(actual[finite], expected[finite], rtol=0, atol=rounding)
                        else:
                            finite = np.isfinite(expected)
                            np.testing.assert_allclose(actual[finite], expected[finite], rtol=0, atol=atol)

    def test_invalid_inputs_raise_for_both_methods(self):
        pose_graph = self._pose_graph()
        start, end = self.T0 + 50_000, self.T0 + 150_000
        xyz = np.zeros((3, 3), dtype=np.float32)
        timestamp_us = np.array([start, start + 10, end], dtype=np.uint64)
        for method in (EXACT, INTERPOLATED):
            compensator = MotionCompensator(pose_graph, parameters=MotionCompensator.Parameters(method))
            for function in (compensator.motion_compensate_points, compensator.motion_decompensate_points):
                with self.subTest(method=method.name, function=function.__name__):
                    with self.assertRaisesRegex(ValueError, "must have the same length, got 3 and 2"):
                        function("lidar", xyz, timestamp_us[:2], start, end)
                    with self.assertRaisesRegex(ValueError, "Point timestamps .* not in frame time bounds"):
                        function("lidar", xyz, timestamp_us, start + 1, end)
                    with self.assertRaisesRegex(ValueError, "Reference timestamp .* not in frame time bounds"):
                        function("lidar", xyz, timestamp_us, start, end, reference_timestamp_us=end + 1)
                    with self.assertRaises(ValueError):  # point timestamps beyond the poses
                        late = self.T0 + 400_000
                        function("lidar", xyz, np.array([start, end, late], dtype=np.uint64), start, late)
                    with self.assertRaises(KeyError):  # unknown sensor
                        function("radar", xyz, timestamp_us, start, end)
            empty = compensator.motion_compensate_points(
                "lidar", np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.uint64), start, end
            )
            empty_decompensation = compensator.motion_decompensate_points(
                "lidar", np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.uint64), start, end
            )
            for points in (empty.xyz_e_reftime, empty.xyz_s_reftime, empty_decompensation.xyz_pointtime):
                self.assertEqual((points.shape, points.dtype), ((0, 3), np.float32))
            for frames in (empty, empty_decompensation):
                self.assertEqual(
                    (frames.reference_frame_id, frames.reference_timestamp_us, frames.anchor_frame_id),
                    ("lidar", end, "world"),
                )

    def test_multiple_time_dependent_edges(self):
        """Pose timestamps of all time-dependent edges on the path are grid breakpoints (here a moving sensor on a
        moving rig, with differently timed poses)"""
        rig_timestamps_us = self.T0 + np.array([0, 100_000, 200_000], dtype=np.uint64)
        sensor_timestamps_us = self.T0 + np.array([0, 33_000, 66_000, 200_000], dtype=np.uint64)
        pose_graph = PoseGraphInterpolator(
            [
                PoseGraphInterpolator.Edge(
                    "rig",
                    "world",
                    np.stack([get_SE3(np.array(t, dtype=float)) for t in ((0, 0, 0), (2, 0, 0), (2, 3, 0))]),
                    rig_timestamps_us,
                ),
                PoseGraphInterpolator.Edge(
                    "lidar",
                    "rig",
                    np.stack([get_SE3(np.array(t, dtype=float)) for t in ((0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1))]),
                    sensor_timestamps_us,
                ),
            ]
        )
        self.assertEqual(
            sorted(t.tolist() for t in pose_graph.get_path_timestamps_us("lidar", "world")),
            sorted([rig_timestamps_us.tolist(), sensor_timestamps_us.tolist()]),
        )
        rng = np.random.default_rng(6)
        xyz = rng.normal(scale=10.0, size=(3000, 3)).astype(np.float32)
        start, end = int(self.T0) + 10_000, int(self.T0) + 190_000
        timestamp_us = rng.integers(start, end + 1, size=3000).astype(np.uint64)
        expected, _, _ = _reference_motion_compensation(pose_graph, "lidar", xyz, timestamp_us, end, "world")
        actual = MotionCompensator(
            pose_graph, parameters=MotionCompensator.Parameters(INTERPOLATED)
        ).motion_compensate_points("lidar", xyz, timestamp_us, start, end)
        # the piecewise linear translations are reproduced up to float32 rounding (no refinement needed)
        np.testing.assert_allclose(actual.xyz_e_reftime, expected, rtol=0, atol=1e-5)

    def test_unique_timestamps_match_numpy(self):
        rng = np.random.default_rng(7)
        for name, timestamp_us in (
            ("dense", self.T0 + rng.integers(0, 100_000, 50_000).astype(np.uint64)),
            ("sparse", self.T0 + rng.integers(0, 10**9, 1_000).astype(np.uint64)),
            ("single", np.array([self.T0], dtype=np.uint64)),
            ("int64", rng.integers(10**6, 10**6 + 500, 2_000)),
        ):
            with self.subTest(case=name):
                unique, inverse = MotionCompensator._unique_timestamps(timestamp_us)  # noqa: SLF001
                expected_unique, expected_inverse = np.unique(timestamp_us, return_inverse=True)
                np.testing.assert_array_equal(unique, expected_unique)
                np.testing.assert_array_equal(inverse, expected_inverse)
                self.assertEqual(unique.dtype, timestamp_us.dtype)

    def test_parameters(self):
        default = MotionCompensator.Parameters()
        self.assertEqual((default.method, default.interpolation_tolerance), (EXACT, 1e-5))
        self.assertEqual(
            MotionCompensator.Parameters(INTERPOLATED, 1e-3), MotionCompensator.Parameters(INTERPOLATED, 1e-3)
        )
        with self.assertRaises(AttributeError):  # frozen
            setattr(default, "method", INTERPOLATED)
        compensator = MotionCompensator.from_sensor_rig(
            "lidar",
            get_SE3(np.zeros(3)),
            np.stack([np.eye(4), get_SE3(np.ones(3))]),
            np.array([0, 1000], dtype=np.uint64),
            parameters=MotionCompensator.Parameters(INTERPOLATED),
        )
        self.assertIs(compensator.parameters.method, INTERPOLATED)


class TestMotionCompensatorAnchorFrame(unittest.TestCase):
    """Anchor-frame parameterization on a small synthetic pose graph (no external data)."""

    def _build_graph(self) -> PoseGraphInterpolator:
        # sensor -(static)-> rig -(dynamic)-> world -(static)-> world_global.
        # "world" and "world_global" differ only by a static transform, so either is a
        # valid anchor that preserves the within-frame motion; "rig" is NOT (the
        # sensor is static relative to it, so it would see no motion).
        timestamps_us = np.array([1000, 2000], dtype=np.uint64)
        T_rig_world = np.stack([get_SE3(np.array([0.0, 0.0, 0.0])), get_SE3(np.array([1.0, 2.0, 3.0]))])
        T_sensor_rig = get_SE3(np.array([0.5, -0.5, 0.25]))
        T_world_world_global = get_SE3(np.array([10.0, -20.0, 5.0]))
        return PoseGraphInterpolator(
            [
                PoseGraphInterpolator.Edge("rig", "world", T_rig_world, timestamps_us),
                PoseGraphInterpolator.Edge("lidar", "rig", T_sensor_rig, None),
                PoseGraphInterpolator.Edge("world", "world_global", T_world_world_global, None),
            ]
        )

    def test_compensation_independent_of_static_anchor_choice(self):
        """Anchors related by a static transform (world / world_global) give identical results."""
        compensator = MotionCompensator(self._build_graph())

        rng = np.random.default_rng(0)
        xyz = rng.normal(size=(64, 3)).astype(np.float32)
        timestamp_us = np.linspace(1000, 2000, 64).astype(np.uint64)

        via_world = compensator.motion_compensate_points(
            "lidar", xyz, timestamp_us, 1000, 2000, anchor_frame_id="world"
        )
        via_global = compensator.motion_compensate_points(
            "lidar", xyz, timestamp_us, 1000, 2000, anchor_frame_id="world_global"
        )

        np.testing.assert_array_almost_equal(via_world.xyz_e_reftime, via_global.xyz_e_reftime, decimal=4)
        self.assertEqual(via_world.anchor_frame_id, "world")
        self.assertEqual(via_global.anchor_frame_id, "world_global")

        # Round-trip with a non-default anchor recovers the input.
        roundtrip = compensator.motion_decompensate_points(
            "lidar", via_global.xyz_e_reftime, timestamp_us, 1000, 2000, anchor_frame_id="world_global"
        ).xyz_pointtime
        np.testing.assert_array_almost_equal(roundtrip, xyz, decimal=4)

    def test_from_sensor_rig_custom_frame_names(self):
        """from_sensor_rig honors custom anchor/rig frame node names."""
        timestamps_us = np.array([1000, 2000], dtype=np.uint64)
        T_rig_anchor = np.stack([get_SE3(np.array([0.0, 0.0, 0.0])), get_SE3(np.array([2.0, 0.0, -1.0]))])
        T_sensor_rig = get_SE3(np.array([0.1, 0.2, 0.3]))

        compensator = MotionCompensator.from_sensor_rig(
            sensor_id="lidar",
            T_sensor_rig=T_sensor_rig,
            T_rig_worlds=T_rig_anchor,
            T_rig_worlds_timestamps_us=timestamps_us,
            anchor_frame_id="ego_world",
            rig_frame_id="chassis",
        )

        rng = np.random.default_rng(1)
        xyz = rng.normal(size=(32, 3)).astype(np.float32)
        timestamp_us = np.linspace(1000, 2000, 32).astype(np.uint64)

        compensated = compensator.motion_compensate_points(
            "lidar", xyz, timestamp_us, 1000, 2000, anchor_frame_id="ego_world"
        )
        roundtrip = compensator.motion_decompensate_points(
            "lidar", compensated.xyz_e_reftime, timestamp_us, 1000, 2000, anchor_frame_id="ego_world"
        ).xyz_pointtime
        np.testing.assert_array_almost_equal(roundtrip, xyz, decimal=4)


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

        with self.assertRaisesRegex(ValueError, r"static edge must have shape \(4, 4\)"):
            # invalid edge
            PoseGraphInterpolator.Edge("V3", "V4", np.stack([np.eye(4), get_SE3(np.array([0, 0, 1]))]), None)

        with self.assertRaisesRegex(ValueError, r"dynamic edge must have shape \[2, 4, 4\]"):
            # invalid edge
            PoseGraphInterpolator.Edge("V3", "V4", get_SE3(np.array([0, 1, 0])), np.array([0, 10], dtype=np.uint64))

        with self.assertRaisesRegex(ValueError, r"dynamic edge must have shape \[n, 4, 4\]"):
            # invalid edge
            PoseGraphInterpolator.Edge("V3", "V4", np.zeros((2, 3, 4)), np.array([0, 10], dtype=np.uint64))

        with self.assertRaisesRegex(ValueError, "timestamps_us must be of integer type, got float64"):
            # non-integer evaluation timestamps
            PoseGraphInterpolator.Edge("V3", "V4", np.eye(4), None).get_T("V3", "V4", np.array([0.5]))

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


class TestPoseInterpolator(unittest.TestCase):
    def test_internals_are_private(self):
        """Only the poses / timestamps properties and the interpolation methods are public"""
        poses = np.stack([get_SE3(np.array([float(t), 0.0, 0.0])) for t in (0, 1, 2)])
        interpolator = PoseInterpolator(poses, np.array([0, 100, 200], dtype=np.uint64))
        for name in ("slerp", "f_t", "last_row"):
            self.assertFalse(hasattr(interpolator, name), name)
        self.assertEqual(
            sorted(name for name in dir(interpolator) if not name.startswith("_")),
            ["extrapolate_to_timestamps", "in_range", "interpolate_to_timestamps", "poses", "timestamps"],
        )

    def test_invalid_inputs_raise(self):
        poses = np.stack([get_SE3(np.array([float(t), 0.0, 0.0])) for t in (0, 1)])
        with self.assertRaisesRegex(ValueError, r"poses must be a floating point \[m, 4, 4\] array"):
            PoseInterpolator(poses[:, :3], np.array([0, 100]))
        with self.assertRaisesRegex(ValueError, r"poses must be a floating point \[m, 4, 4\] array"):
            PoseInterpolator(poses.astype(np.int64), np.array([0, 100]))
        with self.assertRaisesRegex(ValueError, "timestamps must be a 1-dimensional array of at least 2 elements"):
            PoseInterpolator(poses[:1], np.array([0]))

    def test_accepts_array_likes(self):
        """Poses and timestamps may be given as sequences"""
        poses = [get_SE3(np.array([0.0, 0.0, 0.0])), get_SE3(np.array([2.0, 0.0, 0.0]))]
        interpolator = PoseInterpolator(poses, [0, 100])
        np.testing.assert_allclose(
            interpolator.interpolate_to_timestamps(np.array([50]))[0], get_SE3(np.array([1.0, 0.0, 0.0]))
        )
        self.assertIsInstance(interpolator.poses, np.ndarray)
        self.assertIsInstance(interpolator.timestamps, np.ndarray)


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
            angle_diff = _angle_between(result_rot, expected_rot)
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
            angle_diff = _angle_between(result_rot, expected_rot)
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
        angle_diff = _angle_between(result_rot, expected_rotation)
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
                angle_diff = _angle_between(result_rot, expected_rot)

                tolerance = 1e-4 if test_dtype == np.float32 else 1e-10
                self.assertLess(angle_diff, tolerance, f"Rotation mismatch for dtype {test_dtype}")
