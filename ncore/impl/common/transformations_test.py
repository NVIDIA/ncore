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

from python.runfiles import Runfiles

from ncore.impl.data import types
from ncore.impl.common.transformations import MotionCompensator
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

        motion_compensator = MotionCompensator(
            self.lidar_sensor.get_T_sensor_rig(), self.poses.T_rig_worlds, self.poses.T_rig_world_timestamps_us
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
                xyz_e, timestamp_us, frame_start_timestamps_us, frame_end_timestamps_us
            )

            # Re-run compensation on non-compensated points
            motion_compensation_result = motion_compensator.motion_compensate_points(
                xyz_pointtime, timestamp_us, frame_start_timestamps_us, frame_end_timestamps_us
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
