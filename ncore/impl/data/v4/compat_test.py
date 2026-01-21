# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from ncore.impl.common.util import unpack_optional

from .compat import SequenceLoaderV4
from .components import SequenceComponentGroupsReader


_RUNFILES = Runfiles.Create()


class TestCompatV4(unittest.TestCase):
    """Test to verify SequenceLoaderV4 compatibility layer"""

    def setUp(self):
        # Make printed errors more representable numerically
        np.set_printoptions(floatmode="unique", linewidth=200, suppress=True)

        # Load V4 data
        self.loader = SequenceLoaderV4(
            SequenceComponentGroupsReader(
                [
                    _RUNFILES.Rlocation(
                        "test-data-v4/c9b05cf4-afb9-11ec-b3c2-00044bf65fcb@1648597318700123-1648599151600035.json"
                    )
                ],
            )
        )

    def test_sequence_properties(self):
        """Test sequence-level properties through V4 compat layer"""
        # Test sequence_id
        seq_id = self.loader.sequence_id
        self.assertIsInstance(seq_id, str)
        self.assertGreater(len(seq_id), 0)

        # Test generic_meta_data
        meta_data = self.loader.generic_meta_data
        self.assertIsInstance(meta_data, dict)

        # Test sequence_timestamp_interval_us
        interval = self.loader.sequence_timestamp_interval_us
        self.assertIsNotNone(interval)
        self.assertGreater(interval.stop, interval.start)

    def test_sensor_enumeration(self):
        """Test sensor ID enumeration in V4"""
        # Test camera_ids
        camera_ids = self.loader.camera_ids
        self.assertEqual(len(camera_ids), 12)
        self.assertIn("camera_front_wide_120fov", camera_ids)

        # Test lidar_ids
        lidar_ids = self.loader.lidar_ids
        self.assertEqual(len(lidar_ids), 1)
        self.assertIn("lidar_gt_top_p128_v4p5", lidar_ids)

        # Test radar_ids
        radar_ids = self.loader.radar_ids
        self.assertEqual(len(radar_ids), 18)

    def test_pose_graph(self):
        """Test pose graph access in V4"""
        pose_graph = self.loader.pose_graph
        self.assertIsNotNone(pose_graph)

    def test_camera_sensor_basic(self):
        """Test basic camera sensor properties in V4"""
        camera_id = "camera_front_wide_120fov"
        camera = self.loader.get_camera_sensor(camera_id)

        # Test sensor_id
        self.assertEqual(camera.sensor_id, camera_id)

        # Test frames_count
        frames_count = camera.frames_count
        self.assertGreater(frames_count, 0)

        # Test frames_timestamps_us
        timestamps = camera.frames_timestamps_us
        self.assertEqual(timestamps.shape[0], frames_count)
        self.assertEqual(timestamps.shape[1], 2)

    def test_camera_sensor_frames(self):
        """Test camera frame data access in V4"""
        camera = self.loader.get_camera_sensor("camera_front_wide_120fov")

        # Test first frame
        frame_idx = 0

        # Test get_frame_handle
        handle = camera.get_frame_handle(frame_idx)
        self.assertIsNotNone(handle)

        # Test get_frame_image
        image = camera.get_frame_image(frame_idx)
        self.assertIsNotNone(image)

        # Test get_frame_image_array
        image_array = camera.get_frame_image_array(frame_idx)
        self.assertIsInstance(image_array, np.ndarray)

    def test_sequence_paths(self):
        """Test sequence_paths property"""
        paths = self.loader.sequence_paths
        self.assertIsInstance(paths, list)
        self.assertEqual(len(paths), 35)
        for path in paths:
            self.assertTrue(
                path.name.startswith("c9b05cf4-afb9-11ec-b3c2-00044bf65fcb@1648597318700123-1648599151600035")
            )

    def test_reload_resources(self):
        """Test reload_resources in V4"""
        # Should not raise an exception
        self.loader.reload_resources()

        # Should still be able to access data after reload
        camera = self.loader.get_camera_sensor("camera_front_wide_120fov")
        self.assertGreater(camera.frames_count, 0)

    def test_get_closest_frame_index_relative_frame_time_v4(self):
        """Test get_closest_frame_index with various relative_frame_time values for V4 data"""
        camera = self.loader.get_camera_sensor("camera_front_wide_120fov")

        # Get frame timestamps
        timestamps = camera.frames_timestamps_us
        self.assertGreater(timestamps.shape[0], 0)

        # Test with relative_frame_time = 0.0 (start of frame)
        # Should find the closest frame based on frame start time
        test_frame_idx = 0
        test_start_timestamp = timestamps[test_frame_idx, 0]
        found_idx = camera.get_closest_frame_index(test_start_timestamp, relative_frame_time=0.0)
        self.assertEqual(found_idx, test_frame_idx)

        # Test with relative_frame_time = 1.0 (end of frame, default)
        # Should find the closest frame based on frame end time
        test_end_timestamp = timestamps[test_frame_idx, 1]
        found_idx = camera.get_closest_frame_index(test_end_timestamp, relative_frame_time=1.0)
        self.assertEqual(found_idx, test_frame_idx)

        # Test with relative_frame_time = 0.5 (middle of frame)
        # Should find the closest frame based on frame midpoint
        mid_timestamp = (timestamps[test_frame_idx, 0] + timestamps[test_frame_idx, 1]) // 2
        found_idx = camera.get_closest_frame_index(mid_timestamp, relative_frame_time=0.5)
        self.assertEqual(found_idx, test_frame_idx)

        # Test boundary values
        if timestamps.shape[0] > 1:
            test_frame_idx = 1
            test_start_timestamp = timestamps[test_frame_idx, 0]
            found_idx = camera.get_closest_frame_index(test_start_timestamp, relative_frame_time=0.0)
            self.assertEqual(found_idx, test_frame_idx)

    def test_lidar_sensor_basic(self):
        """Test basic lidar sensor properties in V4"""
        lidar_id = "lidar_gt_top_p128_v4p5"
        lidar = self.loader.get_lidar_sensor(lidar_id)

        # Test sensor_id
        self.assertEqual(lidar.sensor_id, lidar_id)

        # Test frames_count
        frames_count = lidar.frames_count
        self.assertGreater(frames_count, 0)

        # Test frames_timestamps_us
        timestamps = lidar.frames_timestamps_us
        self.assertEqual(timestamps.shape[0], frames_count)
        self.assertEqual(timestamps.shape[1], 2)

        # Test T_sensor_rig
        T_sensor_rig = lidar.T_sensor_rig
        self.assertIsNotNone(T_sensor_rig)
        self.assertEqual(unpack_optional(T_sensor_rig).shape, (4, 4))

    def test_lidar_sensor_point_cloud(self):
        """Test lidar point cloud access in V4"""
        lidar = self.loader.get_lidar_sensor("lidar_gt_top_p128_v4p5")

        # Test first frame
        frame_idx = 0

        # Test get_frame_point_cloud
        point_cloud = lidar.get_frame_point_cloud(frame_idx, motion_compensation=True, with_start_points=False)
        self.assertIsNotNone(point_cloud)

        # Verify point cloud structure
        self.assertIsNotNone(point_cloud.xyz_m_end)
        self.assertGreater(len(point_cloud.xyz_m_end), 0)
        self.assertEqual(point_cloud.xyz_m_end.shape[1], 3)

    def test_lidar_sensor_ray_bundle(self):
        """Test lidar ray bundle access in V4"""
        lidar = self.loader.get_lidar_sensor("lidar_gt_top_p128_v4p5")

        frame_idx = 0

        # Test get_frame_ray_bundle_count
        count = lidar.get_frame_ray_bundle_count(frame_idx)
        self.assertGreater(count, 0)

        # Test get_frame_ray_bundle_timestamp_us
        timestamps = lidar.get_frame_ray_bundle_timestamp_us(frame_idx)
        self.assertEqual(timestamps.shape, (count,))

        # Test get_frame_ray_bundle_return_count
        return_count = lidar.get_frame_ray_bundle_return_count(frame_idx)
        self.assertGreaterEqual(return_count, 1)

        # Test get_frame_ray_bundle_return_distance
        distances = lidar.get_frame_ray_bundle_return_distance(frame_idx)
        self.assertEqual(distances.shape[0], count)

        # Test get_frame_ray_bundle_return_intensity
        intensities = lidar.get_frame_ray_bundle_return_intensity(frame_idx)
        self.assertEqual(intensities.shape[0], count)

    def test_lidar_sensor_transforms(self):
        """Test lidar sensor transformations in V4"""
        lidar = self.loader.get_lidar_sensor("lidar_gt_top_p128_v4p5")

        # Test get_frames_T_sensor_target
        self.assertEqual(lidar.get_frames_T_sensor_target("world", 0).shape, (4, 4))
        self.assertEqual(lidar.get_frames_T_sensor_target("world", np.array([0, 1, 2])).shape, (3, 4, 4))
        self.assertEqual(lidar.get_frames_T_sensor_target("world", 1, frame_timepoint=None).shape, (2, 4, 4))
        self.assertEqual(
            lidar.get_frames_T_sensor_target("world", np.array([1, 2, 3]), frame_timepoint=None).shape, (3, 2, 4, 4)
        )

        # Test get_frames_T_source_sensor
        self.assertEqual(lidar.get_frames_T_source_sensor("world", 0).shape, (4, 4))
        self.assertEqual(lidar.get_frames_T_source_sensor("world", np.array([0, 1, 2])).shape, (3, 4, 4))
        self.assertEqual(lidar.get_frames_T_source_sensor("world", 1, frame_timepoint=None).shape, (2, 4, 4))
        self.assertEqual(
            lidar.get_frames_T_source_sensor("world", np.array([1, 2, 3]), frame_timepoint=None).shape, (3, 2, 4, 4)
        )
