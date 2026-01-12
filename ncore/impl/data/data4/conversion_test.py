# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import json
import tempfile
import unittest

from pathlib import Path

import numpy as np
import parameterized

from python.runfiles import Runfiles  # ty:ignore[unresolved-import]
from typing_extensions import Literal
from upath import UPath

from ncore.impl.data.data3 import ShardDataLoader
from ncore.impl.data.types import FrameTimepoint

from .compat import SequenceLoaderV3, SequenceLoaderV4
from .components import SequenceComponentGroupsReader
from .conversion import NCore3To4


_RUNFILES = Runfiles.Create()


class TestData3Converter(unittest.TestCase):
    """Test to verify functionality of V3->V4 data converter and consistency via compat layer"""

    def setUp(self):
        # Make printed errors more representable numerically
        np.set_printoptions(floatmode="unique", linewidth=200, suppress=True)

        # load V3 reference data
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

        self.assertEqual(len(all_shards), 3)

        self.shard_data_loader = ShardDataLoader(all_shards)
        self.v3_loader = SequenceLoaderV3(self.shard_data_loader)

    @parameterized.parameterized.expand([("default",), ("separate-sensors",), ("separate-all",)])
    def test_convert_profiles(self, profile: Literal["default", "separate-sensors", "separate-all"]):
        """Test basic conversion from V3 to V4 with different profiles"""
        tempdir = tempfile.TemporaryDirectory()

        # Convert reference sequence with single camera
        output_paths = NCore3To4.convert(
            self.shard_data_loader,
            output_dir_path=UPath(tempdir.name),
            camera_ids=["camera_front_wide_120fov"],
            lidar_ids=[],
            radar_ids=[],
            component_groups=NCore3To4.create_component_groups(
                source_data_loader=self.shard_data_loader, profile=profile
            ),
        )

        # Verify output paths were created
        self.assertGreater(len(output_paths), 0)
        for path in output_paths:
            self.assertTrue(path.exists())

        tempdir.cleanup()

    @parameterized.parameterized.expand([("itar",), ("directory",)])
    def test_convert_store_types(self, store_type: Literal["itar", "directory"]):
        """Test conversion with different store types"""
        tempdir = tempfile.TemporaryDirectory()

        output_paths = NCore3To4.convert(
            self.shard_data_loader,
            output_dir_path=UPath(tempdir.name),
            camera_ids=["camera_front_wide_120fov"],
            lidar_ids=[],
            radar_ids=[],
            store_type=store_type,
        )

        self.assertGreater(len(output_paths), 0)

        # Verify we can load the converted data
        reader = SequenceComponentGroupsReader(output_paths)
        loader = SequenceLoaderV4(reader)
        self.assertEqual(len(loader.camera_ids), 1)

        tempdir.cleanup()

    def test_convert_with_cameras(self):
        """Test conversion with camera data and verify consistency"""
        tempdir = tempfile.TemporaryDirectory()

        camera_ids = ["camera_front_wide_120fov", "camera_cross_left_120fov"]

        # Convert
        output_paths = NCore3To4.convert(
            self.shard_data_loader,
            output_dir_path=UPath(tempdir.name),
            camera_ids=camera_ids,
            lidar_ids=[],
            radar_ids=[],
        )

        # Load converted data
        v4_reader = SequenceComponentGroupsReader(output_paths)
        v4_loader = SequenceLoaderV4(v4_reader)

        # Verify cameras were converted
        self.assertEqual(set(v4_loader.camera_ids), set(camera_ids))

        # Verify each camera through compat layer
        for camera_id in camera_ids:
            v3_camera = self.v3_loader.get_camera_sensor(camera_id)
            v4_camera = v4_loader.get_camera_sensor(camera_id)

            # Check sensor ID
            self.assertEqual(v3_camera.sensor_id, v4_camera.sensor_id)

            # Check that V4 has frames
            self.assertGreater(v4_camera.frames_count, 0)
            self.assertLessEqual(v4_camera.frames_count, v3_camera.frames_count)

            # Check model parameters type consistency
            v3_model = v3_camera.model_parameters
            v4_model = v4_camera.model_parameters
            self.assertEqual(type(v3_model), type(v4_model))

            # Check extrinsics consistency
            v3_T = v3_camera.T_sensor_rig
            v4_T = v4_camera.T_sensor_rig
            if v3_T is not None and v4_T is not None:
                np.testing.assert_allclose(v3_T, v4_T, rtol=1e-5, atol=1e-6)

        tempdir.cleanup()

    def test_convert_camera_frames_consistency(self):
        """Test that converted camera frames match original V3 data"""
        tempdir = tempfile.TemporaryDirectory()

        camera_id = "camera_front_wide_120fov"

        # Convert
        output_paths = NCore3To4.convert(
            self.shard_data_loader,
            output_dir_path=UPath(tempdir.name),
            camera_ids=[camera_id],
            lidar_ids=[],
            radar_ids=[],
        )

        # Load
        v4_reader = SequenceComponentGroupsReader(output_paths)
        v4_loader = SequenceLoaderV4(v4_reader)

        v3_camera = self.v3_loader.get_camera_sensor(camera_id)
        v4_camera = v4_loader.get_camera_sensor(camera_id)

        # Get timestamps
        v3_timestamps = v3_camera.frames_timestamps_us
        v4_timestamps = v4_camera.frames_timestamps_us

        # Check that V4 timestamps are a subset of V3 timestamps (due to time filtering)
        self.assertLessEqual(v4_camera.frames_count, v3_camera.frames_count)

        # Find where V4 frames start in V3
        v3_start_idx = np.where(v3_timestamps[:, 1] == v4_timestamps[0, 1])[0]
        self.assertGreater(len(v3_start_idx), 0, "V4 first frame not found in V3")
        v3_start_idx = v3_start_idx[0]

        # Verify all V4 timestamps match corresponding V3 timestamps
        v3_subset = v3_timestamps[v3_start_idx : v3_start_idx + v4_camera.frames_count]
        np.testing.assert_array_equal(v3_subset, v4_timestamps, err_msg="Timestamps don't match")

        # Now compare frames by index (since timestamps agree)
        for frame_idx in range(min(3, v4_camera.frames_count)):
            v3_idx = v3_start_idx + frame_idx

            # Compare images - should be pixel-perfect
            v3_image = v3_camera.get_frame_image(v3_idx)
            v4_image = v4_camera.get_frame_image(frame_idx)

            self.assertEqual(v3_image.size, v4_image.size)

            v3_array = np.array(v3_image)
            v4_array = np.array(v4_image)
            np.testing.assert_array_equal(v3_array, v4_array)

        tempdir.cleanup()

    def test_convert_with_radars(self):
        """Test conversion with radar data and verify consistency"""
        tempdir = tempfile.TemporaryDirectory()

        # Select a subset of radars
        radar_ids = self.v3_loader.radar_ids[:2]

        # Convert
        output_paths = NCore3To4.convert(
            self.shard_data_loader,
            output_dir_path=UPath(tempdir.name),
            camera_ids=[],
            lidar_ids=[],
            radar_ids=radar_ids,
        )

        # Load
        v4_reader = SequenceComponentGroupsReader(output_paths)
        v4_loader = SequenceLoaderV4(v4_reader)

        # Verify radars were converted
        self.assertEqual(set(v4_loader.radar_ids), set(radar_ids))

        # Check each radar
        for radar_id in radar_ids:
            v3_radar = self.v3_loader.get_radar_sensor(radar_id)
            v4_radar = v4_loader.get_radar_sensor(radar_id)

            self.assertEqual(v3_radar.sensor_id, v4_radar.sensor_id)
            self.assertGreater(v4_radar.frames_count, 0)

            # Check extrinsics
            v3_T = v3_radar.T_sensor_rig
            v4_T = v4_radar.T_sensor_rig
            if v3_T is not None and v4_T is not None:
                np.testing.assert_allclose(v3_T, v4_T, rtol=1e-5, atol=1e-6)

        tempdir.cleanup()

    def test_convert_with_lidars(self):
        """Test conversion with lidar data and verify consistency"""
        tempdir = tempfile.TemporaryDirectory()

        # Select lidar sensor
        lidar_ids = self.v3_loader.lidar_ids

        # Convert
        output_paths = NCore3To4.convert(
            self.shard_data_loader,
            output_dir_path=UPath(tempdir.name),
            camera_ids=[],
            lidar_ids=lidar_ids,
            radar_ids=[],
        )

        # Load
        v4_reader = SequenceComponentGroupsReader(output_paths)
        v4_loader = SequenceLoaderV4(v4_reader)

        # Verify lidars were converted
        self.assertEqual(set(v4_loader.lidar_ids), set(lidar_ids))

        # Check each lidar
        for lidar_id in lidar_ids:
            v3_lidar = self.v3_loader.get_lidar_sensor(lidar_id)
            v4_lidar = v4_loader.get_lidar_sensor(lidar_id)

            self.assertEqual(v3_lidar.sensor_id, v4_lidar.sensor_id)
            self.assertGreater(v4_lidar.frames_count, 0)

            # Check extrinsics
            v3_T = v3_lidar.T_sensor_rig
            v4_T = v4_lidar.T_sensor_rig
            if v3_T is not None and v4_T is not None:
                np.testing.assert_allclose(v3_T, v4_T, rtol=1e-5, atol=1e-6)

        tempdir.cleanup()

    def test_convert_lidar_frames_consistency(self):
        """Test that converted lidar frames match original V3 data"""
        tempdir = tempfile.TemporaryDirectory()

        lidar_id = self.v3_loader.lidar_ids[0]

        # Convert
        output_paths = NCore3To4.convert(
            self.shard_data_loader,
            output_dir_path=UPath(tempdir.name),
            camera_ids=[],
            lidar_ids=[lidar_id],
            radar_ids=[],
        )

        # Load
        v4_reader = SequenceComponentGroupsReader(output_paths)
        v4_loader = SequenceLoaderV4(v4_reader)

        v3_lidar = self.v3_loader.get_lidar_sensor(lidar_id)
        v4_lidar = v4_loader.get_lidar_sensor(lidar_id)

        # Get timestamps
        v3_timestamps = v3_lidar.frames_timestamps_us
        v4_timestamps = v4_lidar.frames_timestamps_us

        # Check that V4 timestamps are a subset of V3 timestamps (due to time filtering)
        self.assertLessEqual(v4_lidar.frames_count, v3_lidar.frames_count)

        # Find where V4 frames start in V3
        v3_start_idx = np.where(v3_timestamps[:, 1] == v4_timestamps[0, 1])[0]
        self.assertGreater(len(v3_start_idx), 0, "V4 first frame not found in V3")
        v3_start_idx = v3_start_idx[0]

        # Verify all V4 timestamps match corresponding V3 timestamps
        v3_subset = v3_timestamps[v3_start_idx : v3_start_idx + v4_lidar.frames_count]
        np.testing.assert_array_equal(v3_subset, v4_timestamps, err_msg="Frame timestamps don't match")

        # Now compare frames by index (since timestamps agree)
        for frame_idx in range(min(3, v4_lidar.frames_count)):
            v3_idx = v3_start_idx + frame_idx

            # Get point counts
            v3_count = v3_lidar.get_frame_ray_bundle_count(v3_idx)
            v4_count = v4_lidar.get_frame_ray_bundle_count(frame_idx)

            if v3_count == 0 or v4_count == 0:
                continue

            # V3 and V4 should have the same number of points (no filtering in conversion)
            self.assertEqual(v3_count, v4_count, f"Point count mismatch in frame {frame_idx}")

            # Compare timestamps
            v3_ts_data = v3_lidar.get_frame_ray_bundle_timestamp_us(v3_idx)
            v4_ts_data = v4_lidar.get_frame_ray_bundle_timestamp_us(frame_idx)
            np.testing.assert_array_equal(v3_ts_data, v4_ts_data, err_msg="Point timestamps don't match")

            # Compare distances (V3 stores motion-compensated xyz, V4 does motion decompensation)
            v3_distances = v3_lidar.get_frame_ray_bundle_return_distance(v3_idx)
            v4_distances = v4_lidar.get_frame_ray_bundle_return_distance(frame_idx)

            # Distances should match closely (some numerical differences from motion compensation/decompensation)
            # Motion decompensation can introduce larger numerical differences, so use relaxed tolerance
            np.testing.assert_allclose(
                v3_distances, v4_distances, rtol=0.01, atol=0.01, err_msg=f"Distance mismatch in frame {frame_idx}"
            )

            # Compare intensities
            v3_intensities = v3_lidar.get_frame_ray_bundle_return_intensity(v3_idx)
            v4_intensities = v4_lidar.get_frame_ray_bundle_return_intensity(frame_idx)

            # Intensities should match exactly (no transformation)
            np.testing.assert_allclose(
                v3_intensities, v4_intensities, rtol=1e-7, atol=1e-8, err_msg=f"Intensity mismatch in frame {frame_idx}"
            )

            # Verify V3 directions are unit vectors
            v3_directions = v3_lidar.get_frame_ray_bundle_direction(v3_idx)
            self.assertEqual(v3_directions.shape, (v3_count, 3))
            v3_norms = np.linalg.norm(v3_directions, axis=1)
            np.testing.assert_allclose(v3_norms, 1.0, rtol=1e-5, err_msg="V3 directions are not unit vectors")

            # Verify V4 directions are unit vectors
            v4_directions = v4_lidar.get_frame_ray_bundle_direction(frame_idx)
            self.assertEqual(v4_directions.shape, (v4_count, 3))
            v4_norms = np.linalg.norm(v4_directions, axis=1)
            np.testing.assert_allclose(v4_norms, 1.0, rtol=1e-5, err_msg="V4 directions are not unit vectors")

        tempdir.cleanup()

    def test_convert_radar_frames_consistency(self):
        """Test that converted radar frames match original V3 data"""
        tempdir = tempfile.TemporaryDirectory()

        radar_id = self.v3_loader.radar_ids[0]

        # Convert
        output_paths = NCore3To4.convert(
            self.shard_data_loader,
            output_dir_path=UPath(tempdir.name),
            camera_ids=[],
            lidar_ids=[],
            radar_ids=[radar_id],
        )

        # Load
        v4_reader = SequenceComponentGroupsReader(output_paths)
        v4_loader = SequenceLoaderV4(v4_reader)

        v3_radar = self.v3_loader.get_radar_sensor(radar_id)
        v4_radar = v4_loader.get_radar_sensor(radar_id)

        # Get timestamps
        v3_timestamps = v3_radar.frames_timestamps_us
        v4_timestamps = v4_radar.frames_timestamps_us

        # Check that V4 timestamps are a subset of V3 timestamps (due to time filtering)
        self.assertLessEqual(v4_radar.frames_count, v3_radar.frames_count)

        # Find where V4 frames start in V3
        v3_start_idx = np.where(v3_timestamps[:, 1] == v4_timestamps[0, 1])[0]
        self.assertGreater(len(v3_start_idx), 0, "V4 first frame not found in V3")
        v3_start_idx = v3_start_idx[0]

        # Verify all V4 timestamps match corresponding V3 timestamps
        v3_subset = v3_timestamps[v3_start_idx : v3_start_idx + v4_radar.frames_count]
        np.testing.assert_array_equal(v3_subset, v4_timestamps, err_msg="Frame timestamps don't match")

        # Now compare frames by index (since timestamps agree)
        for frame_idx in range(min(3, v4_radar.frames_count)):
            v3_idx = v3_start_idx + frame_idx

            # Get point counts
            v3_count = v3_radar.get_frame_ray_bundle_count(v3_idx)
            v4_count = v4_radar.get_frame_ray_bundle_count(frame_idx)

            if v3_count == 0 or v4_count == 0:
                continue

            # V3 and V4 should have the same number of points (no filtering in conversion)
            self.assertEqual(v3_count, v4_count, f"Point count mismatch in frame {frame_idx}")

            # Compare timestamps
            v3_ts_data = v3_radar.get_frame_ray_bundle_timestamp_us(v3_idx)
            v4_ts_data = v4_radar.get_frame_ray_bundle_timestamp_us(frame_idx)
            np.testing.assert_array_equal(v3_ts_data, v4_ts_data, err_msg="Point timestamps don't match")

            # Compare distances (V3 stores xyz_e, V4 converts properly)
            v3_distances = v3_radar.get_frame_ray_bundle_return_distance(v3_idx)
            v4_distances = v4_radar.get_frame_ray_bundle_return_distance(frame_idx)

            # Distances should match closely (some numerical differences from conversion)
            np.testing.assert_allclose(
                v3_distances, v4_distances, rtol=1e-5, atol=1e-6, err_msg=f"Distance mismatch in frame {frame_idx}"
            )

            # Verify V4 directions are unit vectors
            v4_directions = v4_radar.get_frame_ray_bundle_direction(frame_idx)
            self.assertEqual(v4_directions.shape, (v4_count, 3))
            norms = np.linalg.norm(v4_directions, axis=1)
            np.testing.assert_allclose(norms, 1.0, rtol=1e-5, err_msg="Directions are not unit vectors")

        tempdir.cleanup()

    def test_convert_poses_consistency(self):
        """Test that pose graphs are consistent between V3 and V4"""
        tempdir = tempfile.TemporaryDirectory()

        # Convert with one camera
        output_paths = NCore3To4.convert(
            self.shard_data_loader,
            output_dir_path=UPath(tempdir.name),
            camera_ids=["camera_front_wide_120fov"],
            lidar_ids=[],
            radar_ids=[],
        )

        # Load
        v4_reader = SequenceComponentGroupsReader(output_paths)
        v4_loader = SequenceLoaderV4(v4_reader)

        # Get common timestamp
        camera_id = "camera_front_wide_120fov"
        v3_camera = self.v3_loader.get_camera_sensor(camera_id)
        v4_camera = v4_loader.get_camera_sensor(camera_id)

        v3_ts = v3_camera.get_frames_timestamps_us(FrameTimepoint.END)
        v4_ts = v4_camera.get_frames_timestamps_us(FrameTimepoint.END)

        common_ts = np.intersect1d(v3_ts, v4_ts)
        self.assertGreater(len(common_ts), 0)

        # Compare rig->world transformations
        test_ts = common_ts[:5].astype(np.uint64)  # Test first 5 common timestamps, ensure uint64 dtype

        v3_poses = self.v3_loader.pose_graph.evaluate_poses("rig", "world", test_ts)
        v4_poses = v4_loader.pose_graph.evaluate_poses("rig", "world", test_ts)

        np.testing.assert_allclose(v3_poses, v4_poses, rtol=1e-5, atol=1e-6)

        tempdir.cleanup()

    def test_convert_with_time_range(self):
        """Test conversion with time range selection"""
        tempdir = tempfile.TemporaryDirectory()

        # Get time range from V3 data
        v3_interval = self.v3_loader.sequence_timestamp_interval_us
        mid_time = (v3_interval.start + v3_interval.stop) // 2

        # Convert only second half
        output_paths = NCore3To4.convert(
            self.shard_data_loader,
            output_dir_path=UPath(tempdir.name),
            camera_ids=["camera_front_wide_120fov"],
            lidar_ids=[],
            radar_ids=[],
            start_timestamp_us=mid_time,
        )

        # Load
        v4_reader = SequenceComponentGroupsReader(output_paths)
        v4_loader = SequenceLoaderV4(v4_reader)

        # Check that V4 interval is within specified range
        v4_interval = v4_loader.sequence_timestamp_interval_us
        self.assertGreaterEqual(v4_interval.start, mid_time)

        # Check that camera has fewer frames
        v3_camera = self.v3_loader.get_camera_sensor("camera_front_wide_120fov")
        v4_camera = v4_loader.get_camera_sensor("camera_front_wide_120fov")
        self.assertLess(v4_camera.frames_count, v3_camera.frames_count)

        tempdir.cleanup()

    def test_convert_generic_metadata(self):
        """Test that generic metadata is preserved and can be extended"""
        tempdir = tempfile.TemporaryDirectory()

        custom_metadata = {"custom_key": "custom_value", "test_number": 42}

        # Convert with custom metadata
        output_paths = NCore3To4.convert(
            self.shard_data_loader,
            output_dir_path=UPath(tempdir.name),
            camera_ids=["camera_front_wide_120fov"],
            lidar_ids=[],
            radar_ids=[],
            generic_meta_data=custom_metadata,
        )

        # Load
        v4_reader = SequenceComponentGroupsReader(output_paths)
        v4_loader = SequenceLoaderV4(v4_reader)

        # Check that custom metadata is present
        v4_metadata = v4_loader.generic_meta_data
        self.assertEqual(v4_metadata["custom_key"], "custom_value")
        self.assertEqual(v4_metadata["test_number"], 42)

        # Check that original V3 metadata is also present
        v3_metadata = self.v3_loader.generic_meta_data
        for key in v3_metadata:
            if key not in custom_metadata:
                self.assertIn(key, v4_metadata)

        tempdir.cleanup()

    def test_convert_sequence_meta(self):
        """Test that sequence metadata is accessible through compat layer"""
        tempdir = tempfile.TemporaryDirectory()

        # Convert
        output_paths = NCore3To4.convert(
            self.shard_data_loader,
            output_dir_path=UPath(tempdir.name),
            camera_ids=["camera_front_wide_120fov"],
            lidar_ids=[],
            radar_ids=[],
        )

        # Load
        v4_reader = SequenceComponentGroupsReader(output_paths)
        v4_loader = SequenceLoaderV4(v4_reader)

        # Get sequence meta from both loaders
        v3_meta = self.v3_loader.get_sequence_meta()
        v4_meta = v4_loader.get_sequence_meta()

        # Both should return dict-like objects
        self.assertIsNotNone(v3_meta)
        self.assertIsNotNone(v4_meta)

        # Reload V4 from serialized sequence meta path
        seq_meta_path = UPath(tempdir.name) / f"{v4_loader.sequence_id}.ncore4.json"
        with seq_meta_path.open("w") as f:
            json.dump(v4_loader.get_sequence_meta(), f, indent=2)

        v4_reader_json = SequenceComponentGroupsReader([seq_meta_path])

        self.assertEqual(v4_reader.get_sequence_meta(), v4_reader_json.get_sequence_meta())

        tempdir.cleanup()

    def test_convert_all_sensors(self):
        """Test conversion with all sensor types"""
        tempdir = tempfile.TemporaryDirectory()

        # Convert all sensors (this may take longer)
        output_paths = NCore3To4.convert(
            self.shard_data_loader,
            output_dir_path=UPath(tempdir.name),
            camera_ids=["camera_front_wide_120fov"],  # Just one camera for speed
            lidar_ids=self.v3_loader.lidar_ids[:1],  # Subset of lidars
            radar_ids=self.v3_loader.radar_ids[:2],  # Subset of radars
        )

        # Load
        v4_reader = SequenceComponentGroupsReader(output_paths)
        v4_loader = SequenceLoaderV4(v4_reader)

        # Verify all converted sensors are accessible
        self.assertEqual(len(v4_loader.camera_ids), 1)
        self.assertEqual(len(v4_loader.lidar_ids), 1)
        self.assertEqual(len(v4_loader.radar_ids), 2)

        # Verify we can access each sensor
        for camera_id in v4_loader.camera_ids:
            camera = v4_loader.get_camera_sensor(camera_id)
            self.assertIsNotNone(camera)

        for lidar_id in v4_loader.lidar_ids:
            lidar = v4_loader.get_lidar_sensor(lidar_id)
            self.assertIsNotNone(lidar)

        for radar_id in v4_loader.radar_ids:
            radar = v4_loader.get_radar_sensor(radar_id)
            self.assertIsNotNone(radar)

        tempdir.cleanup()
