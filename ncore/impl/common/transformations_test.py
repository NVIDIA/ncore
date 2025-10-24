# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import unittest

from pathlib import Path

import numpy as np
import parameterized

from python.runfiles import Runfiles

from ncore.impl.common.transformations import MotionCompensator, PoseGraphInterpolator
from ncore.impl.data import types
from ncore.impl.data.data3 import ShardDataLoader


_RUNFILES = Runfiles.Create()


class TestMotionCompensator(unittest.TestCase):
    def setUp(self):
        # Make printed errors more representable numerically
        np.set_printoptions(floatmode="unique", linewidth=200, suppress=True)

        # Load a lidar sensor as a source for motion-compensated point cloud data
        all_shards = sorted(
            [
                str(p)
                for p in Path(
                    _RUNFILES.Rlocation(
                        "test-data-v3-shards/c9b05cf4-afb9-11ec-b3c2-00044bf65fcb@1648597318700123-1648599151600035_0-3.zarr.itar"
                    )
                ).parent.iterdir()
                if p.match("*.itar")
            ]
        )

        loader = ShardDataLoader(all_shards)
        self.poses = loader.get_poses()
        self.lidar_sensor = ShardDataLoader(all_shards).get_lidar_sensor("lidar_gt_top_p128_v4p5")

    def test_idempotence(self):
        """Test to verify compensation / decompensation are symmetric"""

        motion_compensator = MotionCompensator.from_sensor_rig(
            self.lidar_sensor.get_sensor_id(),
            self.lidar_sensor.get_T_sensor_rig(),
            self.poses.T_rig_worlds,
            self.poses.T_rig_world_timestamps_us,
        )

        # Check on a few frames only
        for frame_idx in range(1, 3):
            # Load motion-compensated reference point cloud
            xyz_s = self.lidar_sensor.get_frame_data(frame_idx, "xyz_s")
            xyz_e = self.lidar_sensor.get_frame_data(frame_idx, "xyz_e")
            timestamp_us = self.lidar_sensor.get_frame_data(frame_idx, "timestamp_us")

            frame_start_timestamps_us = self.lidar_sensor.get_frame_timestamp_us(frame_idx, types.FrameTimepoint.START)
            frame_end_timestamps_us = self.lidar_sensor.get_frame_timestamp_us(frame_idx, types.FrameTimepoint.END)

            # Run decompensation
            xyz_pointtime = motion_compensator.motion_decompensate_points(
                self.lidar_sensor.get_sensor_id(),
                xyz_e,
                timestamp_us,
                frame_start_timestamps_us,
                frame_end_timestamps_us,
            )

            # Re-run compensation on non-compensated points
            motion_compensation_result = motion_compensator.motion_compensate_points(
                self.lidar_sensor.get_sensor_id(),
                xyz_pointtime,
                timestamp_us,
                frame_start_timestamps_us,
                frame_end_timestamps_us,
            )

            # Check for consistency
            self.assertIsNone(
                np.testing.assert_array_almost_equal(xyz_s, motion_compensation_result.xyz_s_sensorend),
                f"frame_idx {frame_idx}",
            )
            self.assertIsNone(
                np.testing.assert_array_almost_equal(
                    xyz_e,
                    motion_compensation_result.xyz_e_sensorend,
                    # lower-precision check only because of numerical errors building up + not doing
                    # *repeated* linear interpolatins in MotionCompensator for poses as in the source test data
                    decimal=2,
                ),
                f"frame_idx {frame_idx}",
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
            PoseGraphInterpolator.Edge("V1", "V2", np.stack([np.eye(4), get_SE3(np.array([1, 0, 0]))]), timestamps_us),
            PoseGraphInterpolator.Edge("V1", "V7", get_SE3(np.array([0, 1, 0])), None),
            PoseGraphInterpolator.Edge("V1", "V3", get_SE3(np.array([0, 1, 0])), None),
            PoseGraphInterpolator.Edge("V1", "V4", np.stack([np.eye(4), get_SE3(np.array([2, 0, 0]))]), timestamps_us),
            PoseGraphInterpolator.Edge("V2", "V5", get_SE3(np.array([0, 1, 0])), None),
            PoseGraphInterpolator.Edge("V6", "V2", np.stack([np.eye(4), get_SE3(np.array([0, 0, 1]))]), timestamps_us),
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

    @parameterized.parameterized.expand(
        [
            (np.float32,),
            (np.float64,),
        ]
    )
    def test_interpolation(self, dtype: np.dtype):
        """Test to verify pose interpolation along different paths through a graph"""

        graph = PoseGraphInterpolator(self.edges)

        with self.assertRaises(KeyError):
            # non-existing start node
            graph.evaluate_poses("foo", "V1", np.array([0], dtype=np.uint64), dtype=dtype)

        with self.assertRaises(KeyError):
            # non-existing end node
            graph.evaluate_poses("V1", "foo", np.array([0], dtype=np.uint64), dtype=dtype)

        # self-pose is identity
        self.assertTrue(
            np.array_equal(
                graph.evaluate_poses("V1", "V1", np.array([0], dtype=np.uint64), dtype=dtype),
                np.eye(4)[np.newaxis],
            )
        )

        # verify different timestamp dtypes and empty batch shape on self / static edges
        self.assertTrue(
            np.array_equal(
                graph.evaluate_poses("V1", "V1", np.empty((), dtype=np.int8), dtype=dtype),
                np.eye(4),
            )
        )
        self.assertTrue(
            np.array_equal(
                graph.evaluate_poses("V7", "V1", np.empty((), dtype=np.int32), dtype=dtype),
                get_SE3(np.array([0, -1, 0])),
            )
        )

        # V7 < - V1 -> V3  is static
        self.assertTrue(
            # one hop
            np.array_equal(
                graph.evaluate_poses("V1", "V3", np.array([0, 5, 10], dtype=np.uint64), dtype=dtype),
                np.repeat(get_SE3(np.array([0, 1, 0]))[np.newaxis], 3, axis=0),
            )
        )

        self.assertTrue(
            # two hops, one inverse
            np.array_equal(
                graph.evaluate_poses("V7", "V3", np.array([0, 5, 10], dtype=np.uint64), dtype=dtype),
                np.repeat(get_SE3(np.array([0, 0, 0]))[np.newaxis], 3, axis=0),
            )
        )

        # V1 -> V4 is dynamic
        self.assertTrue(
            np.array_equal(
                graph.evaluate_poses("V1", "V4", np.array([0, 5, 10], dtype=np.uint64), dtype=dtype),
                np.stack([get_SE3(np.array([0, 0, 0])), get_SE3(np.array([1, 0, 0])), get_SE3(np.array([2, 0, 0]))]),
            )
        )

        # V1 -> V2 <- V6 is dynamic
        self.assertTrue(
            np.array_equal(
                graph.evaluate_poses("V1", "V6", np.array([0, 5, 10], dtype=np.uint64), dtype=dtype),
                np.stack(
                    [get_SE3(np.array([0, 0, 0])), get_SE3(np.array([0.5, 0, -0.5])), get_SE3(np.array([1, 0, -1]))]
                ),
            )
        )

        # V7 <- V1 -> V2 <- V6 is mixed static / dynamic
        self.assertTrue(
            np.array_equal(
                graph.evaluate_poses("V7", "V6", np.array([0, 5, 10], dtype=np.uint64), dtype=dtype),
                np.stack(
                    [get_SE3(np.array([0, -1, 0])), get_SE3(np.array([0.5, -1, -0.5])), get_SE3(np.array([1, -1, -1]))]
                ),
            )
        )

        # verify batch-shape handling
        self.assertTrue(
            np.array_equal(
                graph.evaluate_poses(
                    "V7", "V6", np.array([0, 5, 10], dtype=np.uint64).reshape((1, 3, 1, 1)), dtype=dtype
                ),
                np.stack(
                    [get_SE3(np.array([0, -1, 0])), get_SE3(np.array([0.5, -1, -0.5])), get_SE3(np.array([1, -1, -1]))]
                ).reshape((1, 3, 1, 1, 4, 4)),
            )
        )
