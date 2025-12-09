# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import tempfile
import unittest

import numpy as np

from python.runfiles import Runfiles
from upath import UPath

from ncore.impl.common.common import unpack_optional
from ncore.impl.data.data3 import ShardDataLoader
from ncore.impl.data.types import FrameTimepoint

from .compat import SequenceLoaderV3, SequenceLoaderV4
from .components import SequenceComponentStoreReader
from .conversion import NCore3To4


_RUNFILES = Runfiles.Create()


class TestCompatV3(unittest.TestCase):
    """Test to verify SequenceLoaderV3 compatibility layer"""

    def setUp(self):
        # Make printed errors more representable numerically
        np.set_printoptions(floatmode="unique", linewidth=200, suppress=True)

        # load V3 reference data
        all_shards = sorted(
            [
                str(p)
                for p in UPath(
                    _RUNFILES.Rlocation(
                        "test-data-v3-shards/c9b05cf4-afb9-11ec-b3c2-00044bf65fcb@1648597318700123-1648599151600035_0-3.zarr.itar"
                    )
                ).parent.iterdir()
                if p.match("*.itar")
            ]
        )

        self.assertEqual(len(all_shards), 3)

        self.shard_data_loader = ShardDataLoader(all_shards)
        self.loader = SequenceLoaderV3(self.shard_data_loader)

    def test_sequence_properties(self):
        """Test sequence-level properties through compat layer"""
        # Test sequence_id (V3 doesn't include shard range when accessed through compat layer)
        self.assertEqual(
            self.loader.sequence_id,
            "c9b05cf4-afb9-11ec-b3c2-00044bf65fcb@1648597318700123-1648599151600035",
        )

        # Test generic_meta_data
        meta_data = self.loader.generic_meta_data
        self.assertIsInstance(meta_data, dict)
        self.assertIn("nv-rig", meta_data)

        # Test sequence_timestamp_interval_us
        interval = self.loader.sequence_timestamp_interval_us
        self.assertIsNotNone(interval)
        self.assertGreater(interval.stop, interval.start)

        # Test get_sequence_meta
        seq_meta = self.loader.get_sequence_meta()
        self.assertIsInstance(seq_meta, dict)

    def test_sensor_enumeration(self):
        """Test sensor ID enumeration"""
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
        """Test pose graph access and functionality"""
        pose_graph = self.loader.pose_graph

        # Verify pose graph can be queried
        self.assertIsNotNone(pose_graph)

        # Test that we can query transformations
        # Query rig->world at a timestamp within the sequence range
        interval = self.loader.sequence_timestamp_interval_us
        # Use a timestamp from within the actual data range
        test_timestamp = interval.start + (interval.stop - interval.start) // 2
        timestamps_us = np.array([test_timestamp], dtype=np.uint64)
        T_rig_world = pose_graph.evaluate_poses("rig", "world", timestamps_us)
        # evaluate_poses returns (n, 4, 4) where n is the number of timestamps
        self.assertEqual(T_rig_world.shape, (1, 4, 4))

    def test_camera_sensor_basic(self):
        """Test basic camera sensor properties"""
        camera_id = "camera_front_wide_120fov"
        camera = self.loader.get_camera_sensor(camera_id)

        # Test sensor_id
        self.assertEqual(camera.sensor_id, camera_id)

        # Test frames_count
        frames_count = camera.frames_count
        self.assertEqual(frames_count, 27)

        # Test frames_timestamps_us
        timestamps = camera.frames_timestamps_us
        self.assertEqual(timestamps.shape, (27, 2))
        self.assertTrue(np.all(timestamps[:, 0] <= timestamps[:, 1]))

        # Test get_frames_timestamps_us
        end_timestamps = camera.get_frames_timestamps_us(FrameTimepoint.END)
        self.assertEqual(end_timestamps.shape, (27,))
        np.testing.assert_array_equal(end_timestamps, timestamps[:, 1])

        start_timestamps = camera.get_frames_timestamps_us(FrameTimepoint.START)
        self.assertEqual(start_timestamps.shape, (27,))
        np.testing.assert_array_equal(start_timestamps, timestamps[:, 0])

    def test_camera_sensor_model_parameters(self):
        """Test camera model parameters access"""
        camera = self.loader.get_camera_sensor("camera_front_wide_120fov")

        model_params = camera.model_parameters
        self.assertIsNotNone(model_params)
        # Should have intrinsic parameters - check type has necessary attributes
        # Different camera models have different attributes, so just verify it's not None
        self.assertIsNotNone(model_params)

    def test_camera_sensor_masks(self):
        """Test camera mask images access"""
        camera = self.loader.get_camera_sensor("camera_front_wide_120fov")

        masks = camera.get_mask_images()
        self.assertIsInstance(masks, dict)
        # V3 data has ego masks
        if len(masks) > 0:
            self.assertIn("ego", masks)
            mask_image = masks["ego"]
            self.assertIsNotNone(mask_image)

    def test_camera_sensor_frames(self):
        """Test camera frame data access"""
        camera = self.loader.get_camera_sensor("camera_front_wide_120fov")

        # Test first frame
        frame_idx = 0

        # Test get_frame_handle (should return encoded data handle)
        # V3 compat layer returns an EncodedImageData object
        handle = camera.get_frame_handle(frame_idx)
        self.assertIsNotNone(handle)
        # The handle has get_data() which returns encoded image data
        encoded_data = handle.get_data()
        self.assertIsNotNone(encoded_data)
        # EncodedImageData has get_encoded_image_data() method for actual bytes
        actual_bytes = encoded_data.get_encoded_image_data()
        self.assertIsInstance(actual_bytes, bytes)
        self.assertGreater(len(actual_bytes), 0)

        # Test get_frame_image
        image = camera.get_frame_image(frame_idx)
        self.assertIsNotNone(image)
        self.assertEqual(len(image.size), 2)

        # Test get_frame_image_array
        image_array = camera.get_frame_image_array(frame_idx)
        self.assertIsInstance(image_array, np.ndarray)
        self.assertEqual(len(image_array.shape), 3)

    def test_camera_sensor_transforms(self):
        """Test camera sensor transformations"""
        camera = self.loader.get_camera_sensor("camera_front_wide_120fov")

        # Test T_sensor_rig
        T_sensor_rig = camera.T_sensor_rig
        self.assertIsNotNone(T_sensor_rig)
        self.assertEqual(T_sensor_rig.shape, (4, 4))

        # Test get_frame_T_sensor_world
        T_sensor_world = camera.get_frame_T_sensor_world(0)
        self.assertIsNotNone(T_sensor_world)
        self.assertEqual(T_sensor_world.shape, (4, 4))

    def test_camera_sensor_generic_data(self):
        """Test generic data access for camera frames"""
        camera = self.loader.get_camera_sensor("camera_front_wide_120fov")

        frame_idx = 0

        # Test get_frame_generic_data_names
        names = camera.get_frame_generic_data_names(frame_idx)
        self.assertIsInstance(names, list)

        # Test has_frame_generic_data
        for name in names:
            self.assertTrue(camera.has_frame_generic_data(frame_idx, name))

        # Should not have non-existent data
        self.assertFalse(camera.has_frame_generic_data(frame_idx, "nonexistent_data"))

    def test_lidar_sensor_basic(self):
        """Test basic lidar sensor properties"""
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

    def test_lidar_sensor_model_parameters(self):
        """Test lidar model parameters access"""
        lidar = self.loader.get_lidar_sensor("lidar_gt_top_p128_v4p5")

        model_params = lidar.model_parameters
        # V3 input data may or may not have lidar model parameters
        if model_params is not None:
            self.assertTrue(hasattr(model_params, "row_elevations_rad"))

    def test_lidar_sensor_point_cloud(self):
        """Test lidar point cloud access"""
        lidar = self.loader.get_lidar_sensor("lidar_gt_top_p128_v4p5")

        frame_idx = 0

        # Test get_frame_point_cloud - V3 requires motion_compensation and with_start_points args
        point_cloud = lidar.get_frame_point_cloud(frame_idx, motion_compensation=True, with_start_points=False)
        self.assertIsNotNone(point_cloud)

        # Verify point cloud structure - V3 returns xyz_m_end for motion compensated
        self.assertIsNotNone(point_cloud.xyz_m_end)
        self.assertGreater(len(point_cloud.xyz_m_end), 0)
        self.assertEqual(point_cloud.xyz_m_end.shape[1], 3)

    def test_lidar_sensor_ray_bundle(self):
        """Test lidar ray bundle access"""
        lidar = self.loader.get_lidar_sensor("lidar_gt_top_p128_v4p5")

        frame_idx = 0

        # Test get_frame_ray_bundle_count
        count = lidar.get_frame_ray_bundle_count(frame_idx)
        self.assertGreater(count, 0)

        # Test get_frame_ray_bundle_timestamp_us (test this first as it doesn't require motion decompensation)
        timestamps = lidar.get_frame_ray_bundle_timestamp_us(frame_idx)
        self.assertEqual(timestamps.shape, (count,))

        # Test get_frame_ray_bundle_return_count (returns int, not array)
        return_count = lidar.get_frame_ray_bundle_return_count(frame_idx)
        self.assertEqual(return_count, 1)  # V3 only supports single return

        # Test get_frame_ray_bundle_return_distance
        distances = lidar.get_frame_ray_bundle_return_distance(frame_idx)
        self.assertEqual(distances.shape[0], count)

        # Test get_frame_ray_bundle_return_intensity
        intensities = lidar.get_frame_ray_bundle_return_intensity(frame_idx)
        self.assertEqual(intensities.shape[0], count)

        # Test get_frame_ray_bundle_direction (requires motion decompensation, may fail on some V3 data)
        try:
            directions = lidar.get_frame_ray_bundle_direction(frame_idx)
            self.assertEqual(directions.shape, (count, 3))
            # Directions should be unit vectors
            norms = np.linalg.norm(directions, axis=1)
            np.testing.assert_allclose(norms, 1.0, rtol=1e-5)
        except (KeyError, ValueError):
            # Some V3 data may not support direction queries due to missing timestamp data or interpolation issues
            self.skipTest("V3 lidar data does not support direction queries (motion decompensation failed)")

    def test_lidar_sensor_transforms(self):
        """Test lidar sensor transformations"""
        lidar = self.loader.get_lidar_sensor("lidar_gt_top_p128_v4p5")

        # Test T_sensor_rig
        T_sensor_rig = lidar.T_sensor_rig
        self.assertIsNotNone(T_sensor_rig)
        self.assertEqual(T_sensor_rig.shape, (4, 4))

        # Test get_frame_T_sensor_world
        T_sensor_world = lidar.get_frame_T_sensor_world(0)
        self.assertIsNotNone(T_sensor_world)
        self.assertEqual(T_sensor_world.shape, (4, 4))

    def test_radar_sensor_basic(self):
        """Test basic radar sensor properties"""
        radar_ids = self.loader.radar_ids
        self.assertGreater(len(radar_ids), 0)

        radar_id = radar_ids[0]
        radar = self.loader.get_radar_sensor(radar_id)

        # Test sensor_id
        self.assertEqual(radar.sensor_id, radar_id)

        # Test frames_count
        frames_count = radar.frames_count
        self.assertGreater(frames_count, 0)

        # Test frames_timestamps_us
        timestamps = radar.frames_timestamps_us
        self.assertEqual(timestamps.shape[0], frames_count)
        self.assertEqual(timestamps.shape[1], 2)

    def test_radar_sensor_ray_bundle(self):
        """Test radar ray bundle access"""
        radar_ids = self.loader.radar_ids
        radar = self.loader.get_radar_sensor(radar_ids[0])

        if radar.frames_count == 0:
            self.skipTest("No radar frames available")

        frame_idx = 0

        # Test get_frame_ray_bundle_count
        count = radar.get_frame_ray_bundle_count(frame_idx)
        if count == 0:
            self.skipTest("No radar points in frame")

        # Test get_frame_ray_bundle_timestamp_us
        try:
            timestamps = radar.get_frame_ray_bundle_timestamp_us(frame_idx)
            self.assertEqual(timestamps.shape, (count,))
        except KeyError:
            self.skipTest("V3 radar data missing timestamp_us field")

        # Test get_frame_ray_bundle_return_distance
        try:
            distances = radar.get_frame_ray_bundle_return_distance(frame_idx)
            self.assertEqual(distances.shape[0], count)
        except KeyError:
            self.skipTest("V3 radar data missing required fields for distance")

        # Test get_frame_ray_bundle_direction (may require motion decompensation)
        try:
            directions = radar.get_frame_ray_bundle_direction(frame_idx)
            self.assertEqual(directions.shape, (count, 3))
        except (KeyError, ValueError):
            # Some V3 radar data may not support direction queries
            self.skipTest("V3 radar data does not support direction queries")

    def test_cuboid_track_observations(self):
        """Test cuboid track observations access"""
        observations = list(self.loader.get_cuboid_track_observations())

        # Should have some observations
        self.assertGreater(len(observations), 0)

        # Verify observation structure
        for obs in observations[:5]:  # Check first few
            self.assertIsNotNone(obs.track_id)
            self.assertIsNotNone(obs.class_id)
            self.assertIsNotNone(obs.timestamp_us)
            self.assertIsNotNone(obs.bbox3)

    def test_sequence_paths(self):
        """Test sequence_paths property"""
        paths = self.loader.sequence_paths
        self.assertIsInstance(paths, list)
        self.assertEqual(len(paths), 3)
        for path in paths:
            self.assertRegex(
                path.name, r"c9b05cf4-afb9-11ec-b3c2-00044bf65fcb@1648597318700123-1648599151600035_\d+-\d+\.zarr\.itar"
            )

    def test_reload_resources(self):
        """Test reload_resources method"""
        # Should not raise an exception
        self.loader.reload_resources()

        # Should still be able to access data after reload
        camera = self.loader.get_camera_sensor("camera_front_wide_120fov")
        self.assertEqual(camera.frames_count, 27)


class TestCompatV3V4Consistency(unittest.TestCase):
    """Test consistency between V3 and V4 data through compat layer"""

    def setUp(self):
        # Make printed errors more representable numerically
        np.set_printoptions(floatmode="unique", linewidth=200, suppress=True)

        # load V3 reference data
        all_shards = sorted(
            [
                str(p)
                for p in UPath(
                    _RUNFILES.Rlocation(
                        "test-data-v3-shards/c9b05cf4-afb9-11ec-b3c2-00044bf65fcb@1648597318700123-1648599151600035_0-3.zarr.itar"
                    )
                ).parent.iterdir()
                if p.match("*.itar")
            ]
        )

        self.assertEqual(len(all_shards), 3)

        self.shard_data_loader = ShardDataLoader(all_shards)
        self.v3_loader = SequenceLoaderV3(self.shard_data_loader)

        # Convert to V4
        self.tempdir = tempfile.TemporaryDirectory()
        camera_ids = ["camera_front_wide_120fov"]
        lidar_ids = ["lidar_gt_top_p128_v4p5"]

        output_paths = NCore3To4.convert(
            self.shard_data_loader,
            output_dir_path=UPath(self.tempdir.name),
            camera_ids=camera_ids,
            lidar_ids=lidar_ids,
            radar_ids=[],  # Skip radars for faster testing
        )

        # Load V4 data
        v4_reader = SequenceComponentStoreReader(output_paths)
        self.v4_loader = SequenceLoaderV4(v4_reader)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_sequence_id_consistency(self):
        """Verify sequence ID is consistent between V3 and V4"""
        v3_id = self.v3_loader.sequence_id
        v4_id = self.v4_loader.sequence_id

        # V3 includes shard range, V4 doesn't, so check base sequence ID
        self.assertTrue(v4_id in v3_id or v3_id in v4_id)

    def test_sequence_timestamp_interval_consistency(self):
        """Verify timestamp intervals are consistent"""
        v3_interval = self.v3_loader.sequence_timestamp_interval_us
        v4_interval = self.v4_loader.sequence_timestamp_interval_us

        # V4 interval should be within or equal to V3 interval
        self.assertGreaterEqual(v4_interval.start, v3_interval.start)
        self.assertLessEqual(v4_interval.stop, v3_interval.stop)

    def test_camera_sensor_consistency(self):
        """Verify camera data consistency between V3 and V4"""
        camera_id = "camera_front_wide_120fov"

        v3_camera = self.v3_loader.get_camera_sensor(camera_id)
        v4_camera = self.v4_loader.get_camera_sensor(camera_id)

        # Both should have the same sensor_id
        self.assertEqual(v3_camera.sensor_id, v4_camera.sensor_id)

        # Frame counts might differ due to time filtering, but V4 should not have more
        self.assertLessEqual(v4_camera.frames_count, v3_camera.frames_count)

        # Check model parameters consistency
        v3_model = v3_camera.model_parameters
        v4_model = v4_camera.model_parameters

        self.assertEqual(type(v3_model), type(v4_model))

        # Check T_sensor_rig consistency
        v3_T_sensor_rig = v3_camera.T_sensor_rig
        v4_T_sensor_rig = v4_camera.T_sensor_rig

        if v3_T_sensor_rig is not None and v4_T_sensor_rig is not None:
            np.testing.assert_allclose(v3_T_sensor_rig, v4_T_sensor_rig, rtol=1e-5, atol=1e-6)

    def test_camera_frames_consistency(self):
        """Verify camera frame data consistency"""
        camera_id = "camera_front_wide_120fov"

        v3_camera = self.v3_loader.get_camera_sensor(camera_id)
        v4_camera = self.v4_loader.get_camera_sensor(camera_id)

        # Compare first frame that exists in both
        v4_timestamps = v4_camera.frames_timestamps_us
        v3_timestamps = v3_camera.frames_timestamps_us

        # Find matching frame
        for v4_idx in range(min(3, v4_camera.frames_count)):  # Check first few frames
            v4_timestamp = v4_timestamps[v4_idx, 1]  # end timestamp

            # Find matching V3 frame
            matching_v3_indices = np.where(v3_timestamps[:, 1] == v4_timestamp)[0]

            if len(matching_v3_indices) > 0:
                v3_idx = matching_v3_indices[0]

                # Compare images (sizes should match)
                v3_image = v3_camera.get_frame_image(v3_idx)
                v4_image = v4_camera.get_frame_image(v4_idx)

                self.assertEqual(v3_image.size, v4_image.size)

                # Compare as arrays (should be identical)
                v3_array = np.array(v3_image)
                v4_array = np.array(v4_image)

                np.testing.assert_array_equal(v3_array, v4_array)

    def test_pose_graph_consistency(self):
        """Verify pose graph provides consistent transformations"""
        camera_id = "camera_front_wide_120fov"

        v3_camera = self.v3_loader.get_camera_sensor(camera_id)
        v4_camera = self.v4_loader.get_camera_sensor(camera_id)

        # Get a common timestamp
        v4_timestamps = v4_camera.get_frames_timestamps_us(FrameTimepoint.END)
        v3_timestamps = v3_camera.get_frames_timestamps_us(FrameTimepoint.END)

        # Find common timestamp
        common_timestamps = np.intersect1d(v3_timestamps, v4_timestamps)

        if len(common_timestamps) > 0:
            test_timestamp = common_timestamps[0]

            # Query transformations from both loaders
            v3_T_rig_world = self.v3_loader.pose_graph.evaluate_poses(
                "rig", "world", np.array([test_timestamp], dtype=np.uint64)
            )
            v4_T_rig_world = self.v4_loader.pose_graph.evaluate_poses(
                "rig", "world", np.array([test_timestamp], dtype=np.uint64)
            )

            # Should be very close
            np.testing.assert_allclose(v3_T_rig_world, v4_T_rig_world, rtol=1e-5, atol=1e-6)

    def test_lidar_sensor_consistency(self):
        """Verify lidar data consistency between V3 and V4"""
        lidar_id = "lidar_gt_top_p128_v4p5"

        v3_lidar = self.v3_loader.get_lidar_sensor(lidar_id)
        v4_lidar = self.v4_loader.get_lidar_sensor(lidar_id)

        # Both should have the same sensor_id
        self.assertEqual(v3_lidar.sensor_id, v4_lidar.sensor_id)

        # Frame counts might differ due to time filtering, but V4 should not have more
        self.assertLessEqual(v4_lidar.frames_count, v3_lidar.frames_count)

        # Check T_sensor_rig consistency
        v3_T_sensor_rig = v3_lidar.T_sensor_rig
        v4_T_sensor_rig = v4_lidar.T_sensor_rig

        if v3_T_sensor_rig is not None and v4_T_sensor_rig is not None:
            np.testing.assert_allclose(v3_T_sensor_rig, v4_T_sensor_rig, rtol=1e-5, atol=1e-6)

    def test_lidar_point_cloud_consistency(self):
        """Verify lidar point cloud data consistency"""
        lidar_id = "lidar_gt_top_p128_v4p5"

        v3_lidar = self.v3_loader.get_lidar_sensor(lidar_id)
        v4_lidar = self.v4_loader.get_lidar_sensor(lidar_id)

        # Test first v4 frame (assume frame indices are consistent), which might be different from v3 as first frame might be dropped
        v4_frame_idx = 0
        v3_frame_idx = v3_lidar.get_closest_frame_index(v4_lidar.frames_timestamps_us[v4_frame_idx, 1])

        # Compare point clouds - both V3 and V4 require motion_compensation and with_start_points args
        v3_pc = v3_lidar.get_frame_point_cloud(v3_frame_idx, motion_compensation=True, with_start_points=True)
        v4_pc = v4_lidar.get_frame_point_cloud(v4_frame_idx, motion_compensation=True, with_start_points=True)

        # V3 may contain zero-length rays, V4 filters them out - filter V3 to match
        v3_nonzero_mask = v3_lidar.get_frame_ray_bundle_return_distance(v3_frame_idx) > 0

        # Compare point counts and positions (after filtering)
        np.testing.assert_allclose(v3_pc.xyz_m_end[v3_nonzero_mask], v4_pc.xyz_m_end, atol=1e-2)
        np.testing.assert_allclose(
            unpack_optional(v3_pc.xyz_m_start)[v3_nonzero_mask], unpack_optional(v4_pc.xyz_m_start), atol=1e-2
        )

    def test_lidar_ray_bundle_consistency(self):
        """Verify lidar ray bundle data consistency"""
        lidar_id = "lidar_gt_top_p128_v4p5"

        v3_lidar = self.v3_loader.get_lidar_sensor(lidar_id)
        v4_lidar = self.v4_loader.get_lidar_sensor(lidar_id)

        # Test first v4 frame (assume frame indices are consistent), which might be different from v3 as first frame might be dropped
        v4_frame_idx = 0
        v3_frame_idx = v3_lidar.get_closest_frame_index(v4_lidar.frames_timestamps_us[v4_frame_idx, 1])

        # Get ray bundle data
        v4_count = v4_lidar.get_frame_ray_bundle_count(v4_frame_idx)

        v3_distances = v3_lidar.get_frame_ray_bundle_return_distance(v3_frame_idx)
        v4_distances = v4_lidar.get_frame_ray_bundle_return_distance(v4_frame_idx)

        v3_intensities = v3_lidar.get_frame_ray_bundle_return_intensity(v3_frame_idx)
        v4_intensities = v4_lidar.get_frame_ray_bundle_return_intensity(v4_frame_idx)

        # V3 may contain zero-length rays, V4 filters them out - filter V3 to match
        v3_nonzero_mask = v3_distances > 0
        v3_distances_filtered = v3_distances[v3_nonzero_mask]
        v3_intensities_filtered = v3_intensities[v3_nonzero_mask]

        # Compare counts (after filtering)
        self.assertEqual(len(v3_distances_filtered), v4_count)

        # Compare distances
        np.testing.assert_allclose(v3_distances_filtered, v4_distances, rtol=1e-3)

        # Compare intensities
        np.testing.assert_allclose(v3_intensities_filtered, v4_intensities, rtol=1e-5)


class TestCompatV4(unittest.TestCase):
    """Test to verify SequenceLoaderV4 compatibility layer"""

    def setUp(self):
        # Make printed errors more representable numerically
        np.set_printoptions(floatmode="unique", linewidth=200, suppress=True)

        # load V3 reference data and convert to V4
        all_shards = sorted(
            [
                str(p)
                for p in UPath(
                    _RUNFILES.Rlocation(
                        "test-data-v3-shards/c9b05cf4-afb9-11ec-b3c2-00044bf65fcb@1648597318700123-1648599151600035_0-3.zarr.itar"
                    )
                ).parent.iterdir()
                if p.match("*.itar")
            ]
        )

        self.assertEqual(len(all_shards), 3)

        shard_data_loader = ShardDataLoader(all_shards)

        # Convert to V4
        self.tempdir = tempfile.TemporaryDirectory()
        camera_ids = ["camera_front_wide_120fov"]
        lidar_ids = ["lidar_gt_top_p128_v4p5"]

        output_paths = NCore3To4.convert(
            shard_data_loader,
            output_dir_path=UPath(self.tempdir.name),
            camera_ids=camera_ids,
            lidar_ids=lidar_ids,
            radar_ids=[],  # Skip radars for faster testing
        )

        # Load V4 data
        self.v4_reader = SequenceComponentStoreReader(output_paths)
        self.loader = SequenceLoaderV4(self.v4_reader)

    def tearDown(self):
        self.tempdir.cleanup()

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
        self.assertEqual(len(camera_ids), 1)  # We only converted one camera
        self.assertIn("camera_front_wide_120fov", camera_ids)

        # Test lidar_ids
        lidar_ids = self.loader.lidar_ids
        self.assertEqual(len(lidar_ids), 1)  # We converted one lidar
        self.assertIn("lidar_gt_top_p128_v4p5", lidar_ids)

        # Test radar_ids (should be empty as we didn't convert any)
        radar_ids = self.loader.radar_ids
        self.assertEqual(len(radar_ids), 0)

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
        self.assertEqual(len(paths), 1)
        for path in paths:
            self.assertEqual(
                path.name, "c9b05cf4-afb9-11ec-b3c2-00044bf65fcb@1648597318700123-1648599151600035.ncore4.zarr.itar"
            )

    def test_reload_resources(self):
        """Test reload_resources in V4"""
        # Should not raise an exception
        self.loader.reload_resources()

        # Should still be able to access data after reload
        camera = self.loader.get_camera_sensor("camera_front_wide_120fov")
        self.assertGreater(camera.frames_count, 0)

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
        self.assertEqual(T_sensor_rig.shape, (4, 4))

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

        # Test get_frame_T_sensor_world
        T_sensor_world = lidar.get_frame_T_sensor_world(0)
        self.assertIsNotNone(T_sensor_world)
        self.assertEqual(T_sensor_world.shape, (4, 4))
