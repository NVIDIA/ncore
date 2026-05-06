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

import tempfile
import unittest

from typing import Tuple

import numpy as np

from python.runfiles import Runfiles  # pyright: ignore[reportMissingImports] # ty:ignore[unresolved-import]
from scipy.spatial.transform import Rotation as R
from upath import UPath

from ncore.impl.common.transformations import HalfClosedInterval
from ncore.impl.common.util import unpack_optional
from ncore.impl.data.types import PointCloud, RowOffsetStructuredSpinningLidarModelParameters
from ncore.impl.data.v4.compat import SequenceLoaderV4
from ncore.impl.data.v4.components import (
    IntrinsicsComponent,
    LidarSensorComponent,
    PointCloudsComponent,
    PosesComponent,
    SequenceComponentGroupsReader,
    SequenceComponentGroupsWriter,
)


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

        # Test get_frame_ray_bundle_return_valid_mask
        valid_masks = lidar.get_frame_ray_bundle_return_valid_mask(frame_idx)
        self.assertEqual(valid_masks.shape, (count,))
        self.assertTrue(valid_masks.dtype == np.bool_)
        self.assertTrue(np.all(valid_masks))

        # Test get_frame_ray_bundle_return_distance_m
        distances_m = lidar.get_frame_ray_bundle_return_distance_m(frame_idx)
        self.assertEqual(distances_m.shape[0], count)

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


class TestCompatV4ReferenceValues(unittest.TestCase):
    """Test V4 data against known reference values"""

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

    def test_sensor_extrinsics_reference_values(self):
        """Test T_sensor_rig matches known reference values for camera_front_wide_120fov"""
        sensor = self.loader.get_camera_sensor("camera_front_wide_120fov")

        # Reference T_sensor_rig values
        reference_T_sensor_rig = np.array(
            [
                [-0.01506471, -0.0072778263, 0.99986, 1.774368],
                [-0.9998305, 0.010698613, -0.014986393, 0.0035241419],
                [-0.010588046, -0.9999163, -0.0074377647, 1.4483173],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        np.testing.assert_array_equal(unpack_optional(sensor.T_sensor_rig), reference_T_sensor_rig)

    def test_sensor_frame_count_reference_values(self):
        """Test frames_count matches expected value"""
        sensor = self.loader.get_camera_sensor("camera_front_wide_120fov")
        self.assertEqual(sensor.frames_count, 26)

    def test_sensor_timestamps_reference_values(self):
        """Test that frame timestamps match expected values"""
        sensor = self.loader.get_camera_sensor("camera_front_wide_120fov")

        # Reference timestamps: (frame_idx, start_timestamp_us, end_timestamp_us)
        reference_frame_timestamps = [
            (0, 1648597318809370, 1648597318840981),
            (3, 1648597318909357, 1648597318940968),
            (4, 1648597318942767, 1648597318974378),
            (7, 1648597319042761, 1648597319074372),
        ]

        for frame_idx, expected_start, expected_end in reference_frame_timestamps:
            actual_start = sensor.frames_timestamps_us[frame_idx, 0]
            actual_end = sensor.frames_timestamps_us[frame_idx, 1]
            self.assertEqual(actual_start, expected_start, f"Frame {frame_idx} start timestamp mismatch")
            self.assertEqual(actual_end, expected_end, f"Frame {frame_idx} end timestamp mismatch")

    def test_sensor_poses_by_timestamp_reference_values(self):
        """Test T_rig_world at known timestamps matches reference values."""
        # Reference poses keyed by timestamp_us
        reference_poses = {
            # Frame 0 START
            1648597318809370: np.array(
                [
                    [0.994072, 0.108689055, -0.0027365193, -0.10483271],
                    [-0.1083891, 0.99267316, 0.05340203, -7.33575],
                    [0.008520685, -0.052788854, 0.99856937, 0.4586895],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            # Frame 0 END
            1648597318840981: np.array(
                [
                    [0.9943994, 0.10563379, -0.0033572735, -0.055768613],
                    [-0.105360806, 0.9933232, 0.046990618, -7.133101],
                    [0.008298654, -0.046373717, 0.9988897, 0.43542957],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            # Frame 3 START
            1648597318909357: np.array(
                [
                    [0.9947975, 0.10178284, -0.004263375, 0.008754347],
                    [-0.10156045, 0.99415636, 0.036586877, -6.863315],
                    [0.007962378, -0.035963543, 0.9993214, 0.40074944],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            # Frame 3 END
            1648597318940968: np.array(
                [
                    [0.9941264, 0.10816693, -0.0035539374, -0.08399431],
                    [-0.107923664, 0.993271, 0.042014558, -7.2351894],
                    [0.008074609, -0.041384228, 0.9991107, 0.43077266],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            # Frame 4 START
            1648597318942767: np.array(
                [
                    [0.994087, 0.10853001, -0.0035125145, -0.089272685],
                    [-0.10828564, 0.9932185, 0.042323276, -7.2563534],
                    [0.00808204, -0.041692663, 0.9990978, 0.4324813],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            # Frame 4 END
            1648597318974378: np.array(
                [
                    [0.9933726, 0.11490519, -0.0027662509, -0.18202133],
                    [-0.114643395, 0.9922587, 0.047744706, -7.6282277],
                    [0.008230952, -0.04711115, 0.99885577, 0.4625045],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            # Frame 7 START
            1648597319042761: np.array(
                [
                    [0.99407494, 0.10865563, -0.0029973236, -0.116565935],
                    [-0.10847275, 0.9934174, 0.036817934, -7.1801877],
                    [0.0069780694, -0.036274657, 0.99931747, 0.39714238],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            # Frame 7 END
            1648597319074372: np.array(
                [
                    [0.99496245, 0.100186095, -0.0035222045, -0.012373716],
                    [-0.10006499, 0.994655, 0.025464071, -6.625067],
                    [0.0060545243, -0.024983346, 0.99966955, 0.33072114],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
        }

        # Query poses from pose graph
        timestamps = np.array(list(reference_poses.keys()), dtype=np.uint64)
        actual_poses = self.loader.pose_graph.evaluate_poses("rig", "world", timestamps)

        # Verify each pose matches (use almost_equal due to float32 interpolation precision)
        for i, ts in enumerate(timestamps):
            np.testing.assert_array_almost_equal(
                actual_poses[i],
                reference_poses[ts],
                decimal=6,
                err_msg=f"T_rig_world mismatch at timestamp {ts}",
            )

    def test_closest_frame_index_reference_values(self):
        """Test get_closest_frame_index returns expected frame indices"""
        sensor = self.loader.get_camera_sensor("camera_front_wide_120fov")

        # Reference: (query_timestamp, expected_frame_idx)
        reference_queries = [
            (1648597318840981, 0),  # Frame 0 END timestamp
            (1648597318940968, 3),  # Frame 3 END timestamp
            (1648597318974378, 4),  # Frame 4 END timestamp
            (1648597319074372, 7),  # Frame 7 END timestamp
        ]

        for query_ts, expected_idx in reference_queries:
            actual_idx = sensor.get_closest_frame_index(query_ts)
            self.assertEqual(actual_idx, expected_idx, f"get_closest_frame_index({query_ts}) mismatch")

    def test_rig_world_poses_for_all_sensors(self):
        """Test that rig-world poses can be evaluated for all frame start/end times for all sensors"""
        # Collect all sensors: cameras, lidars, radars
        sensors = []
        for camera_id in self.loader.camera_ids:
            sensors.append(("camera", camera_id, self.loader.get_camera_sensor(camera_id)))
        for lidar_id in self.loader.lidar_ids:
            sensors.append(("lidar", lidar_id, self.loader.get_lidar_sensor(lidar_id)))
        for radar_id in self.loader.radar_ids:
            sensors.append(("radar", radar_id, self.loader.get_radar_sensor(radar_id)))

        for sensor_type, sensor_id, sensor in sensors:
            with self.subTest(sensor_type=sensor_type, sensor_id=sensor_id):
                timestamps = sensor.frames_timestamps_us

                # Should be able to evaluate poses for all frame timestamps without error
                poses = self.loader.pose_graph.evaluate_poses("rig", "world", timestamps)

                # Verify shape: (N, 2, 4, 4) transformation matrices
                self.assertEqual(poses.shape, (len(timestamps), 2, 4, 4))

                # Verify all poses are valid transformation matrices (last row should be [0, 0, 0, 1])
                self.assertTrue(np.allclose(poses[:, :, 3, :], np.array([0.0, 0.0, 0.0, 1.0])))

    def test_cuboid_observations_reference_values(self):
        """Test cuboid track observations match reference values"""
        observations = list(self.loader.get_cuboid_track_observations())

        # Reference values: total observation count
        self.assertEqual(len(observations), 148)

        # Reference values for first observation
        first_obs = observations[0]
        self.assertEqual(first_obs.track_id, "b95fd6f978e83165e0a065230bf00ea8f41a1d2f")
        self.assertEqual(first_obs.class_id, "automobile")
        self.assertEqual(first_obs.timestamp_us, 1648597318800163)
        self.assertEqual(first_obs.reference_frame_id, "lidar_gt_top_p128_v4p5")
        self.assertEqual(first_obs.reference_frame_timestamp_us, 1648597318900083)

        # BBox3 values for first observation (use almost_equal for float comparison)
        np.testing.assert_array_almost_equal(
            first_obs.bbox3.centroid,
            (-12.650735855102539, -1.3851573467254639, -0.9924717545509338),
            decimal=5,
        )
        np.testing.assert_array_almost_equal(
            first_obs.bbox3.dim,
            (4.159830093383789, 1.8550034761428833, 1.6977629661560059),
            decimal=5,
        )
        np.testing.assert_array_almost_equal(
            first_obs.bbox3.rot,
            (-0.01817314885556698, -0.010435115545988083, 0.06443758308887482),
            decimal=5,
        )

        # Reference values for second observation (different track)
        second_obs = observations[1]
        self.assertEqual(second_obs.track_id, "443fd5dc167479746ef5ac285af078505d00be53")
        self.assertEqual(second_obs.class_id, "automobile")
        self.assertEqual(second_obs.timestamp_us, 1648597318803918)

        # Reference values for third observation
        third_obs = observations[2]
        self.assertEqual(third_obs.track_id, "718b491f98662e3f820a72966e5f6e1ed314aba4")
        self.assertEqual(third_obs.class_id, "automobile")
        self.assertEqual(third_obs.timestamp_us, 1648597318806465)

    def test_cuboid_observations_timestamp_filtering(self):
        """Test that timestamp_interval_us correctly filters cuboid track observations"""

        # Get all observations to establish reference data
        all_observations = list(self.loader.get_cuboid_track_observations())
        self.assertEqual(len(all_observations), 148)

        all_timestamps = sorted(obs.timestamp_us for obs in all_observations)
        min_ts = all_timestamps[0]
        max_ts = all_timestamps[-1]

        # None (default) returns all observations
        all_observations_none = list(self.loader.get_cuboid_track_observations(timestamp_interval_us=None))
        self.assertEqual(len(all_observations_none), 148)

        # Interval covering the full timestamp range returns all observations
        full_interval = HalfClosedInterval.from_start_end(min_ts, max_ts)
        full_observations = list(self.loader.get_cuboid_track_observations(timestamp_interval_us=full_interval))
        self.assertEqual(len(full_observations), 148)

        # Interval strictly between second and third observation timestamps (half-closed [start, stop))
        # First observation has timestamp 1648597318800163, second 1648597318803918, third 1648597318806465
        # Use an interval [second_ts, third_ts) which should include the second but exclude the third
        second_ts = all_timestamps[1]
        third_ts = all_timestamps[2]
        narrow_interval = HalfClosedInterval(second_ts, third_ts)
        narrow_observations = list(self.loader.get_cuboid_track_observations(timestamp_interval_us=narrow_interval))
        # All returned observations must have timestamp_us in the interval
        for obs in narrow_observations:
            self.assertGreaterEqual(obs.timestamp_us, narrow_interval.start)
            self.assertLess(obs.timestamp_us, narrow_interval.stop)
        # Must include the second observation (start is inclusive)
        self.assertTrue(any(obs.timestamp_us == second_ts for obs in narrow_observations))
        # Must exclude the third observation (stop is exclusive)
        self.assertFalse(any(obs.timestamp_us == third_ts for obs in narrow_observations))
        # Must be fewer than all observations
        self.assertLess(len(narrow_observations), 148)
        self.assertGreater(len(narrow_observations), 0)

        # Disjoint interval (far in the future) returns 0 observations
        disjoint_interval = HalfClosedInterval(max_ts + 1000000, max_ts + 2000000)
        disjoint_observations = list(self.loader.get_cuboid_track_observations(timestamp_interval_us=disjoint_interval))
        self.assertEqual(len(disjoint_observations), 0)

        # Interval that starts exactly at an observation's timestamp includes it (half-closed start)
        start_boundary = HalfClosedInterval(min_ts, min_ts + 1)
        start_observations = list(self.loader.get_cuboid_track_observations(timestamp_interval_us=start_boundary))
        self.assertTrue(any(obs.timestamp_us == min_ts for obs in start_observations))

        # Interval that stops exactly at an observation's timestamp excludes it (half-closed stop)
        stop_boundary = HalfClosedInterval(min_ts - 1, min_ts)
        stop_observations = list(self.loader.get_cuboid_track_observations(timestamp_interval_us=stop_boundary))
        self.assertFalse(any(obs.timestamp_us == min_ts for obs in stop_observations))


class TestPointCloudsSourceIntegration(unittest.TestCase):
    """Integration tests for PointCloudsSourceProtocol via SequenceLoaderV4.

    Writes a REAL V4 dataset (poses, intrinsics, lidar, point_clouds) using Writers,
    finalizes, then loads through SequenceLoaderV4 and verifies the point-clouds
    source protocol works correctly for both native and lidar-adapted sources.
    """

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_directions(vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        norms = np.linalg.norm(vectors, axis=1)
        return vectors / norms[:, np.newaxis], norms

    # ── setUp ────────────────────────────────────────────────────────────────

    def setUp(self):
        np.set_printoptions(floatmode="unique", linewidth=200, suppress=True)
        np.random.seed(42)

        self._tempdir = tempfile.TemporaryDirectory()

        ref_sequence_id = "pc-source-test"
        ref_ts_interval = HalfClosedInterval(0, 1_000_001)

        store_writer = SequenceComponentGroupsWriter(
            output_dir_path=UPath(self._tempdir.name),
            store_base_name=ref_sequence_id,
            sequence_id=ref_sequence_id,
            sequence_timestamp_interval_us=ref_ts_interval,
            store_type="directory",
            generic_meta_data={},
        )

        # ── Poses ────────────────────────────────────────────────────────
        T_rig_worlds = np.stack(
            [
                np.block(
                    [
                        [
                            R.from_euler("xyz", [0, a, 0], degrees=True).as_matrix(),
                            np.array([a * 0.01, 0, 0]).reshape(3, 1),
                        ],
                        [np.array([0, 0, 0, 1])],
                    ]
                )
                for a in [0.0, 0.5, 1.0]
            ]
        )
        T_rig_world_timestamps_us = np.array([0, 500_000, 1_000_000], dtype=np.uint64)

        T_lidar_rig = np.block(
            [
                [R.from_euler("xyz", [1, 2, 3], degrees=True).as_matrix(), np.array([0.5, 0.1, -0.2]).reshape(3, 1)],
                [np.array([0, 0, 0, 1])],
            ]
        ).astype(np.float32)

        poses_writer = store_writer.register_component_writer(
            PosesComponent.Writer,
            "default",
            group_name=None,
        )
        poses_writer.store_dynamic_pose(
            source_frame_id="rig",
            target_frame_id="world",
            poses=T_rig_worlds,
            timestamps_us=T_rig_world_timestamps_us,
        )
        poses_writer.store_static_pose(
            source_frame_id="test_lidar",
            target_frame_id="rig",
            pose=T_lidar_rig,
        )

        # ── Intrinsics ──────────────────────────────────────────────────
        intrinsics_writer = store_writer.register_component_writer(
            IntrinsicsComponent.Writer,
            "default",
            "intrinsics",
        )
        self.ref_lidar_intrinsics = RowOffsetStructuredSpinningLidarModelParameters(
            spinning_frequency_hz=10.0,
            spinning_direction="ccw",
            n_rows=32,
            n_columns=360,
            row_elevations_rad=np.linspace(0.25, -0.43, 32, dtype=np.float32),
            column_azimuths_rad=np.linspace(-3.14, 3.14, 360, dtype=np.float32),
            row_azimuth_offsets_rad=np.zeros(32, dtype=np.float32),
        )
        intrinsics_writer.store_lidar_intrinsics("test_lidar", self.ref_lidar_intrinsics)

        # ── Lidar frames ────────────────────────────────────────────────
        lidar_writer = store_writer.register_component_writer(
            LidarSensorComponent.Writer,
            "test_lidar",
            "lidars",
        )

        # Frame 0: 5 rays, 1 return, with model_element, generic_data={"rgb": (5,3)}
        self.ref_lidar_dir0, lidar_dist0 = self._normalize_directions(np.random.rand(5, 3).astype(np.float32) + 0.1)
        self.ref_lidar_ts0 = np.linspace(0, 500_000, num=5, dtype=np.uint64)
        self.ref_lidar_model_element0 = np.arange(10, dtype=np.uint16).reshape(5, 2)
        self.ref_lidar_distance_m0 = lidar_dist0[np.newaxis, :]
        self.ref_lidar_intensity0 = np.random.rand(1, 5).astype(np.float32)
        self.ref_lidar_rgb0 = np.random.randint(0, 256, size=(5, 3), dtype=np.uint8)
        self.ref_lidar_frame_ts0 = np.array([0, 500_000], dtype=np.uint64)

        lidar_writer.store_frame(
            self.ref_lidar_dir0,
            self.ref_lidar_ts0,
            self.ref_lidar_model_element0,
            self.ref_lidar_distance_m0,
            self.ref_lidar_intensity0,
            self.ref_lidar_frame_ts0,
            generic_data={"rgb": self.ref_lidar_rgb0},
            generic_meta_data={"frame": 0},
        )

        # Frame 1: 8 rays, 2 returns, no model_element, generic_data={"rgb": (8,3)}
        self.ref_lidar_dir1, lidar_dist1 = self._normalize_directions(np.random.rand(8, 3).astype(np.float32) + 0.1)
        self.ref_lidar_ts1 = np.linspace(500_001, 1_000_000, num=8, dtype=np.uint64)
        self.ref_lidar_rgb1 = np.random.randint(0, 256, size=(8, 3), dtype=np.uint8)
        self.ref_lidar_frame_ts1 = np.array([500_001, 1_000_000], dtype=np.uint64)

        # Build 2-return distances and intensities with some NaN for return 1
        absent_mask = np.zeros((2, 8), dtype=bool)
        absent_mask[1, 3:6] = True  # three absent entries in second return

        self.ref_lidar_distance_m1 = np.stack((lidar_dist1, lidar_dist1 + 0.1)).astype(np.float32)
        self.ref_lidar_distance_m1[absent_mask] = np.nan

        self.ref_lidar_intensity1 = np.random.rand(2, 8).astype(np.float32)
        self.ref_lidar_intensity1[absent_mask] = np.nan

        self.ref_lidar_valid_mask1 = ~absent_mask

        lidar_writer.store_frame(
            self.ref_lidar_dir1,
            self.ref_lidar_ts1,
            None,  # no model_element
            self.ref_lidar_distance_m1,
            self.ref_lidar_intensity1,
            self.ref_lidar_frame_ts1,
            generic_data={"rgb": self.ref_lidar_rgb1},
            generic_meta_data={"frame": 1},
        )

        # ── PointCloudsComponent ("sfm_points") ────────────────────────
        self.ref_pc_xyz = np.random.rand(10, 3).astype(np.float32)
        self.ref_pc_rgb = np.random.randint(0, 256, size=(10, 3), dtype=np.uint8)
        self.ref_pc_score = np.random.rand(2).astype(np.float32)

        pc_writer = store_writer.register_component_writer(
            PointCloudsComponent.Writer,
            "sfm_points",
            coordinate_unit=PointCloud.CoordinateUnit.METERS,
            attribute_schemas={
                "rgb": PointCloudsComponent.AttributeSchema(
                    transform_type=PointCloud.AttributeTransformType.INVARIANT,
                    dtype=np.dtype("uint8"),
                    shape_suffix=(3,),
                ),
            },
        )
        pc_writer.store_pc(
            xyz=self.ref_pc_xyz,
            reference_frame_id="world",
            reference_frame_timestamp_us=500_000,
            attributes={"rgb": self.ref_pc_rgb},
            generic_data={"score": self.ref_pc_score},
            generic_meta_data={"source": "sfm"},
        )

        # ── Finalize & Load ─────────────────────────────────────────────
        store_paths = store_writer.finalize()

        reader = SequenceComponentGroupsReader(store_paths, open_consolidated=False)
        self.loader = SequenceLoaderV4(
            reader,
            masks_component_group_name=None,
            cuboids_component_group_name=None,
        )

    def tearDown(self):
        self._tempdir.cleanup()

    # ── Tests ────────────────────────────────────────────────────────────────

    def test_point_clouds_ids(self):
        """point_clouds_ids returns only native sources, not lidar/radar."""
        ids = self.loader.point_clouds_ids
        self.assertEqual(ids, ["sfm_points"])
        self.assertNotIn("test_lidar", ids)

    def test_native_point_clouds_source(self):
        """Verify native source: xyz, attributes, reference frame, coordinate_unit, generic_data."""
        src = self.loader.get_point_clouds_source("sfm_points")

        self.assertEqual(src.point_clouds_source_id, "sfm_points")
        self.assertEqual(src.pcs_count, 1)
        np.testing.assert_array_equal(src.pc_timestamps_us, np.array([500_000], dtype=np.uint64))

        pc = src.get_pc(0)
        np.testing.assert_array_almost_equal(pc.xyz, self.ref_pc_xyz)
        self.assertEqual(pc.reference_frame_id, "world")
        self.assertEqual(pc.reference_frame_timestamp_us, 500_000)
        self.assertEqual(pc.coordinate_unit, PointCloud.CoordinateUnit.METERS)

        # Schema attribute "rgb" should be in attribute_names
        self.assertIn("rgb", pc.attribute_names)
        np.testing.assert_array_equal(pc.get_attribute("rgb"), self.ref_pc_rgb)

        # generic_data "score" should NOT be in attribute_names but accessible via get_pc_generic_data
        self.assertNotIn("score", pc.attribute_names)
        self.assertTrue(src.has_pc_generic_data(0, "score"))
        np.testing.assert_array_almost_equal(src.get_pc_generic_data(0, "score"), self.ref_pc_score)
        self.assertEqual(sorted(src.get_pc_generic_data_names(0)), ["score"])
        self.assertEqual(src.get_pc_generic_meta_data(0), {"source": "sfm"})

    def test_lidar_adapted_source(self):
        """Verify adapter: pcs_count matches frames_count, PointCloud has intensity/timestamp_us/valid_mask.
        Sensor generic_data (rgb) NOT in PointCloud attributes but accessible via get_pc_generic_data.
        """
        src = self.loader.get_point_clouds_source("test_lidar")

        self.assertEqual(src.point_clouds_source_id, "test_lidar")
        self.assertEqual(src.pcs_count, 2)

        # Check timestamps (end-of-frame)
        np.testing.assert_array_equal(
            src.pc_timestamps_us,
            np.array([500_000, 1_000_000], dtype=np.uint64),
        )

        # Frame 0
        pc0 = src.get_pc(0)
        self.assertEqual(pc0.points_count, 5)
        self.assertEqual(pc0.reference_frame_id, "test_lidar")
        self.assertEqual(pc0.reference_frame_timestamp_us, 500_000)
        self.assertEqual(pc0.coordinate_unit, PointCloud.CoordinateUnit.METERS)

        # Attributes should include timestamp_us, valid_mask, intensity (lidar)
        self.assertIn("timestamp_us", pc0.attribute_names)
        self.assertIn("valid_mask", pc0.attribute_names)
        self.assertIn("intensity", pc0.attribute_names)

        np.testing.assert_array_equal(pc0.get_attribute("timestamp_us"), self.ref_lidar_ts0)
        np.testing.assert_array_equal(
            pc0.get_attribute("valid_mask"),
            np.ones(5, dtype=bool),
        )
        np.testing.assert_array_almost_equal(
            pc0.get_attribute("intensity"),
            self.ref_lidar_intensity0[0],
        )

        # Sensor generic_data (rgb) is auto-promoted to a PointCloud attribute
        # (default GenericDataPromotion matches "rgb" with shape (N, 3))
        self.assertIn("rgb", pc0.attribute_names)
        np.testing.assert_array_equal(pc0.get_attribute("rgb"), self.ref_lidar_rgb0)
        self.assertEqual(pc0.get_attribute_transform_type("rgb"), PointCloud.AttributeTransformType.INVARIANT)

        # Also still accessible via get_pc_generic_data (forwarded from sensor)
        self.assertTrue(src.has_pc_generic_data(0, "rgb"))
        np.testing.assert_array_equal(src.get_pc_generic_data(0, "rgb"), self.ref_lidar_rgb0)
        self.assertEqual(src.get_pc_generic_data_names(0), ["rgb"])

    def test_lidar_adapter_model_element(self):
        """model_element present for frame 0, absent for frame 1."""
        src = self.loader.get_point_clouds_source("test_lidar")

        pc0 = src.get_pc(0)
        self.assertIn("model_element", pc0.attribute_names)
        np.testing.assert_array_equal(
            pc0.get_attribute("model_element"),
            self.ref_lidar_model_element0,
        )

        pc1 = src.get_pc(1)
        self.assertNotIn("model_element", pc1.attribute_names)

    def test_lidar_adapter_multi_return(self):
        """Use return_index=0 vs return_index=1 for frame 1:
        - Different xyz values (different distances)
        - Different intensity values
        - Different valid_mask (return 0 all valid, return 1 has some invalid)
        """
        src0 = self.loader.get_point_clouds_source("test_lidar", return_index=0)
        src1 = self.loader.get_point_clouds_source("test_lidar", return_index=1)

        pc0 = src0.get_pc(1)  # frame 1, return 0
        pc1 = src1.get_pc(1)  # frame 1, return 1

        # Different xyz (different distances lead to different point positions)
        self.assertFalse(np.allclose(pc0.xyz, pc1.xyz))

        # Different intensity values
        intensity0 = pc0.get_attribute("intensity")
        intensity1 = pc1.get_attribute("intensity")
        # Note: where return 1 is absent, its intensity is NaN
        valid_both = self.ref_lidar_valid_mask1[0] & self.ref_lidar_valid_mask1[1]
        if np.any(valid_both):
            # For valid entries, intensities may differ (random)
            pass  # Just check they're different arrays
        self.assertFalse(np.array_equal(intensity0, intensity1))

        # Different valid_mask
        mask0 = pc0.get_attribute("valid_mask")
        mask1 = pc1.get_attribute("valid_mask")

        # Return 0: all valid
        np.testing.assert_array_equal(mask0, self.ref_lidar_valid_mask1[0])
        self.assertTrue(np.all(mask0))

        # Return 1: some invalid
        np.testing.assert_array_equal(mask1, self.ref_lidar_valid_mask1[1])
        self.assertFalse(np.all(mask1))

    def test_lidar_adapter_generic_meta_data(self):
        """Adapter forwards per-pc generic metadata from the sensor."""
        src = self.loader.get_point_clouds_source("test_lidar")
        self.assertEqual(src.get_pc_generic_meta_data(0), {"frame": 0})
        self.assertEqual(src.get_pc_generic_meta_data(1), {"frame": 1})

    def test_lidar_adapter_missing_generic_data(self):
        """has_pc_generic_data returns False for non-existent names."""
        src = self.loader.get_point_clouds_source("test_lidar")
        self.assertFalse(src.has_pc_generic_data(0, "nonexistent"))

    def test_unknown_id_raises_key_error(self):
        """Requesting a non-existent source raises KeyError."""
        with self.assertRaises(KeyError):
            self.loader.get_point_clouds_source("nonexistent_source")

    def test_pc_index_range(self):
        """Strided access works for both native and adapted sources."""
        # Native source: 1 pc
        native_src = self.loader.get_point_clouds_source("sfm_points")
        self.assertEqual(list(native_src.get_pc_index_range()), [0])
        self.assertEqual(list(native_src.get_pc_index_range(0, 1)), [0])
        self.assertEqual(list(native_src.get_pc_index_range(0, 0)), [])

        # Adapted lidar source: 2 pcs (frames)
        lidar_src = self.loader.get_point_clouds_source("test_lidar")
        self.assertEqual(list(lidar_src.get_pc_index_range()), [0, 1])
        self.assertEqual(list(lidar_src.get_pc_index_range(0, 2, 2)), [0])
        self.assertEqual(list(lidar_src.get_pc_index_range(1, 2)), [1])
        self.assertEqual(list(lidar_src.get_pc_index_range(step=1)), [0, 1])
