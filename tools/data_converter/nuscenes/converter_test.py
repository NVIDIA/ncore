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

"""Integration tests for the nuScenes data converter (V4 format).

Requires the NUSCENES_DIR environment variable pointing to a nuScenes dataset root
directory (e.g. /data/nuscenes). Works with any version but v1.0-mini is recommended
for CI since it is small (~4GB).

Set NUSCENES_VERSION to override the default (v1.0-mini for tests).
"""

import os
import tempfile
import unittest

from typing import Literal, cast

import numpy as np
import torch

from parameterized import parameterized_class
from upath import UPath

from ncore.impl.data.types import IdealPinholeCameraModelParameters, RowOffsetStructuredSpinningLidarModelParameters
from ncore.impl.data.v4.components import (
    CameraSensorComponent,
    CuboidsComponent,
    IntrinsicsComponent,
    LidarSensorComponent,
    PosesComponent,
    RadarSensorComponent,
    SequenceComponentGroupsReader,
)
from ncore.impl.sensors.lidar import StructuredLidarModel
from tools.data_converter.nuscenes.converter import NuScenesConverter4, NuScenesConverter4Config
from tools.data_converter.nuscenes.utils import get_nuscenes


@parameterized_class(
    ("store_type",),
    [
        ("itar",),
        ("directory",),
    ],
)
class TestNuScenesConverter(unittest.TestCase):
    """Integration tests for nuScenes data converter.

    Requires NUSCENES_DIR environment variable pointing to a nuScenes dataset root.
    Uses the first scene in the dataset for testing.
    """

    store_type: Literal["itar", "directory"]

    @classmethod
    def setUpClass(cls):
        cls.nuscenes_dir = os.environ.get("NUSCENES_DIR")
        if cls.nuscenes_dir is None:
            raise unittest.SkipTest("NUSCENES_DIR not set -- skipping nuScenes integration tests")

        cls.nuscenes_version = os.environ.get("NUSCENES_VERSION", "v1.0-mini")

        cls._tempdir = tempfile.TemporaryDirectory(prefix="nuscenes_test_")
        cls.output_dir = cls._tempdir.name

        # Run the converter for the first scene only

        nusc = get_nuscenes(version=cls.nuscenes_version, dataroot=cls.nuscenes_dir)
        cls.scene_token = nusc.scene[0]["token"]
        cls.scene_name = nusc.scene[0]["name"]

        config = NuScenesConverter4Config(
            root_dir=cls.nuscenes_dir,
            output_dir=cls.output_dir,
            no_cameras=False,
            camera_ids=None,
            no_lidars=False,
            lidar_ids=None,
            no_radars=False,
            radar_ids=None,
            verbose=False,
            debug=False,
            debug_port=5678,
            version=cls.nuscenes_version,
            scene_token=cls.scene_token,
            scene_name=None,
            store_type=cls.store_type,
            component_group_profile="separate-sensors",
            store_sequence_meta=True,
        )
        NuScenesConverter4.convert(config)

        # Find output sequence directory (named after scene_name)
        seq_dirs = [d for d in UPath(cls.output_dir).iterdir() if d.is_dir()]
        assert len(seq_dirs) == 1, f"Expected 1 sequence dir, found {len(seq_dirs)}: {seq_dirs}"
        cls.seq_dir = seq_dirs[0]

        # Open reader via the sequence meta JSON file
        meta_files = list(cls.seq_dir.glob("*.json"))
        assert len(meta_files) == 1, f"Expected 1 meta JSON, found {len(meta_files)}"
        cls.reader = SequenceComponentGroupsReader([meta_files[0]])

    @classmethod
    def tearDownClass(cls):
        cls._tempdir.cleanup()

    # --- Poses ----------------------------------------------------------------

    def test_sequence_has_dynamic_rig_to_world_pose(self):
        """Verify dynamic rig -> world ego pose exists."""
        poses_readers = self.reader.open_component_readers(PosesComponent.Reader)
        self.assertEqual(len(poses_readers), 1)
        poses_reader = list(poses_readers.values())[0]

        poses, timestamps = poses_reader.get_dynamic_pose("rig", "world")
        self.assertEqual(poses.shape[1:], (4, 4))
        self.assertGreater(poses.shape[0], 0)
        self.assertEqual(timestamps.shape[0], poses.shape[0])

    def test_sequence_has_static_world_to_world_global(self):
        """Verify static world -> world_global pose exists."""
        poses_readers = self.reader.open_component_readers(PosesComponent.Reader)
        poses_reader = list(poses_readers.values())[0]

        static_poses = dict(poses_reader.get_static_poses())
        self.assertIn(("world", "world_global"), static_poses)
        pose = static_poses[("world", "world_global")]
        self.assertEqual(pose.shape, (4, 4))

    def test_first_pose_near_identity(self):
        """Verify the first actual sweep pose is near identity (local origin)."""
        poses_readers = self.reader.open_component_readers(PosesComponent.Reader)
        poses_reader = list(poses_readers.values())[0]

        poses, _ = poses_reader.get_dynamic_pose("rig", "world")
        # poses[0] is the extrapolated boundary; poses[1] is the first real sweep
        second_pose = poses[1]
        np.testing.assert_array_almost_equal(second_pose, np.eye(4, dtype=np.float32), decimal=3)

    # --- Cameras --------------------------------------------------------------

    def test_six_cameras_exist(self):
        """Verify all 6 camera readers exist with frames."""
        camera_readers = self.reader.open_component_readers(CameraSensorComponent.Reader)
        expected_ids = {
            "camera_front",
            "camera_front_left",
            "camera_front_right",
            "camera_back",
            "camera_back_left",
            "camera_back_right",
        }
        self.assertEqual(set(camera_readers.keys()), expected_ids)
        for cam_id, cam_reader in camera_readers.items():
            self.assertGreater(cam_reader.frames_count, 0, f"{cam_id} should have frames")

    def test_camera_intrinsics_ideal_pinhole(self):
        """Verify camera intrinsics are distortion-free ideal pinholes.

        nuScenes images are already undistorted, so the converter stores an ideal
        (distortion-free) pinhole model -- there are no distortion coefficients to
        check, only valid focal length and principal point.
        """
        intrinsics_readers = self.reader.open_component_readers(IntrinsicsComponent.Reader)
        self.assertEqual(len(intrinsics_readers), 1)
        intrinsics_reader = list(intrinsics_readers.values())[0]

        for cam_id in [
            "camera_front",
            "camera_front_left",
            "camera_front_right",
            "camera_back",
            "camera_back_left",
            "camera_back_right",
        ]:
            params = intrinsics_reader.get_camera_model_parameters(cam_id)
            self.assertIsInstance(params, IdealPinholeCameraModelParameters)
            params = cast(IdealPinholeCameraModelParameters, params)
            self.assertTrue(np.all(params.focal_length > 0))
            self.assertTrue(np.all(params.principal_point > 0))

    def test_camera_extrinsics_stored_as_static_poses(self):
        """Verify each camera has a static sensor -> rig extrinsic pose."""
        poses_readers = self.reader.open_component_readers(PosesComponent.Reader)
        poses_reader = list(poses_readers.values())[0]

        static_poses = dict(poses_reader.get_static_poses())
        for cam_id in [
            "camera_front",
            "camera_front_left",
            "camera_front_right",
            "camera_back",
            "camera_back_left",
            "camera_back_right",
        ]:
            self.assertIn((cam_id, "rig"), static_poses, f"Missing static pose for {cam_id}")
            pose = static_poses[(cam_id, "rig")]
            self.assertEqual(pose.shape, (4, 4))

    # --- Lidar ----------------------------------------------------------------

    def test_lidar_exists_with_frames(self):
        """Verify lidar reader exists with frames."""
        lidar_readers = self.reader.open_component_readers(LidarSensorComponent.Reader)
        self.assertIn("lidar_top", lidar_readers)
        lidar_reader = lidar_readers["lidar_top"]
        self.assertGreater(lidar_reader.frames_count, 0)

    def test_lidar_extrinsic_stored_as_static_pose(self):
        """Verify lidar has a static sensor -> rig extrinsic pose."""
        poses_readers = self.reader.open_component_readers(PosesComponent.Reader)
        poses_reader = list(poses_readers.values())[0]

        static_poses = dict(poses_reader.get_static_poses())
        self.assertIn(("lidar_top", "rig"), static_poses)
        pose = static_poses[("lidar_top", "rig")]
        self.assertEqual(pose.shape, (4, 4))

    def test_lidar_has_structured_model(self):
        """Verify lidar intrinsics contain a structured spinning model."""

        intrinsics_readers = self.reader.open_component_readers(IntrinsicsComponent.Reader)
        intrinsics_reader = list(intrinsics_readers.values())[0]

        params = intrinsics_reader.get_lidar_model_parameters("lidar_top")
        self.assertIsNotNone(params)
        self.assertIsInstance(params, RowOffsetStructuredSpinningLidarModelParameters)
        assert isinstance(params, RowOffsetStructuredSpinningLidarModelParameters)  # narrow type

        # HDL-32E: 32 rows, spinning CW at ~20Hz
        self.assertEqual(params.n_rows, 32)
        self.assertGreater(params.n_columns, 1000)  # typically ~1084
        self.assertEqual(params.spinning_direction, "cw")
        self.assertAlmostEqual(params.spinning_frequency_hz, 20.0, delta=1.0)

        # Elevation angles: should span roughly -30 to +10 deg for HDL-32E
        self.assertEqual(len(params.row_elevations_rad), 32)
        min_elev_deg = np.degrees(params.row_elevations_rad.min())
        max_elev_deg = np.degrees(params.row_elevations_rad.max())
        self.assertLess(min_elev_deg, -20.0)
        self.assertGreater(max_elev_deg, 5.0)

        # Column azimuths: should span nearly 360 degrees
        azimuth_span = params.column_azimuths_rad.max() - params.column_azimuths_rad.min()
        self.assertGreater(np.degrees(azimuth_span), 300.0)

    def test_lidar_model_reproduces_point_cloud(self):
        """Verify that model-based points match native direction*distance points across frames.

        For each lidar frame, computes points two ways:
        1. Native: direction * distance (stored ray-bundle data)
        2. Model-based: elements_to_sensor_points(model_element, distance)

        The model is derived from median statistics, so we expect small angular
        deviations (~0.1 deg) but the overall structure should match closely.
        """

        intrinsics_readers = self.reader.open_component_readers(IntrinsicsComponent.Reader)
        intrinsics_reader = list(intrinsics_readers.values())[0]
        params = intrinsics_reader.get_lidar_model_parameters("lidar_top")
        assert isinstance(params, RowOffsetStructuredSpinningLidarModelParameters)

        lidar_model = StructuredLidarModel.maybe_from_parameters(params, device="cpu")
        assert lidar_model is not None

        lidar_readers = self.reader.open_component_readers(LidarSensorComponent.Reader)
        lidar_reader = lidar_readers["lidar_top"]

        # Get frame end timestamps for indexing (reader API uses end-of-frame timestamp as key)
        frame_timestamps = lidar_reader.frames_timestamps_us  # [N, 2] (start, end)
        frame_end_timestamps = frame_timestamps[:, 1]
        n_frames = len(frame_end_timestamps)

        # Check a subset of frames (every 20th to keep test fast)
        frame_indices = list(range(0, n_frames, max(1, n_frames // 20)))

        max_angular_errors_deg = []
        mean_angular_errors_deg = []

        for idx in frame_indices:
            ts = int(frame_end_timestamps[idx])

            # Native: direction * distance
            direction = lidar_reader.get_frame_ray_bundle_data(ts, "direction")
            distance_2d = np.array(lidar_reader._get_ray_bundle_returns_group(ts)["distance_m"])  # [R, N]
            distance = distance_2d[0]  # first return

            # Model element
            model_element_data = lidar_reader.get_frame_ray_bundle_data(ts, "model_element")

            # Filter valid (finite, positive distance)
            valid = np.isfinite(distance) & (distance > 0)
            if not valid.any():
                continue

            native_pts = direction[valid] * distance[valid, np.newaxis]

            # Model-based: elements_to_sensor_points
            model_pts = (
                lidar_model.elements_to_sensor_points(
                    model_element_data[valid],
                    distance[valid],
                )
                .cpu()
                .numpy()
            )

            # Compare via angular error between the two point sets
            native_norms = np.linalg.norm(native_pts, axis=1, keepdims=True)
            model_norms = np.linalg.norm(model_pts, axis=1, keepdims=True)
            native_dirs = native_pts / np.maximum(native_norms, 1e-8)
            model_dirs = model_pts / np.maximum(model_norms, 1e-8)
            cos_angle = np.clip(np.sum(native_dirs * model_dirs, axis=1), -1.0, 1.0)
            angular_error_deg = np.degrees(np.arccos(cos_angle))

            max_angular_errors_deg.append(float(angular_error_deg.max()))
            mean_angular_errors_deg.append(float(angular_error_deg.mean()))

        # Expect: mean angular error < 0.5 deg overall, < 0.2 deg for >20m range.
        # The model uses a linear fit of far-range azimuths + iterative timestamp refinement.
        # Deviations come from MC distortion (translational) which scales inversely with range.
        overall_mean = np.mean(mean_angular_errors_deg)
        overall_max = np.max(max_angular_errors_deg)
        self.assertLess(overall_mean, 0.5, f"Mean angular error too large: {overall_mean:.3f} deg")
        self.assertLess(overall_max, 5.0, f"Max angular error too large: {overall_max:.3f} deg")

    # --- Radars ---------------------------------------------------------------

    def test_five_radars_exist(self):
        """Verify all 5 radar readers exist with frames."""
        radar_readers = self.reader.open_component_readers(RadarSensorComponent.Reader)
        expected_ids = {
            "radar_front",
            "radar_front_left",
            "radar_front_right",
            "radar_back_left",
            "radar_back_right",
        }
        self.assertEqual(set(radar_readers.keys()), expected_ids)
        for radar_id, radar_reader in radar_readers.items():
            self.assertGreater(radar_reader.frames_count, 0, f"{radar_id} should have frames")

    def test_radar_extrinsics_stored_as_static_poses(self):
        """Verify each radar has a static sensor -> rig extrinsic pose."""
        poses_readers = self.reader.open_component_readers(PosesComponent.Reader)
        poses_reader = list(poses_readers.values())[0]

        static_poses = dict(poses_reader.get_static_poses())
        for radar_id in [
            "radar_front",
            "radar_front_left",
            "radar_front_right",
            "radar_back_left",
            "radar_back_right",
        ]:
            self.assertIn((radar_id, "rig"), static_poses, f"Missing static pose for {radar_id}")
            pose = static_poses[(radar_id, "rig")]
            self.assertEqual(pose.shape, (4, 4))

    # --- Cuboids (Annotations) ------------------------------------------------

    def test_cuboid_observations_exist(self):
        """Verify cuboid track observations were stored from annotations."""
        cuboid_readers = self.reader.open_component_readers(CuboidsComponent.Reader)
        # v1.0-mini has annotations, v1.0-test does not
        if not cuboid_readers:
            self.skipTest("No cuboid component (possibly test split with no annotations)")

        self.assertEqual(len(cuboid_readers), 1)
        cuboid_reader = list(cuboid_readers.values())[0]

        observations = list(cuboid_reader.get_observations())
        self.assertGreater(len(observations), 0)

        # Check first observation has expected fields
        obs = observations[0]
        self.assertIsInstance(obs.track_id, str)
        self.assertIsInstance(obs.class_id, str)
        self.assertEqual(obs.reference_frame_id, "world_global")

    # --- Sequence Meta --------------------------------------------------------

    def test_sequence_meta_file_exists(self):
        """Verify sequence meta JSON file was written."""
        meta_files = list(self.seq_dir.glob("*.json"))
        self.assertEqual(len(meta_files), 1)

    def test_sequence_id_matches_scene_name(self):
        """Verify the sequence ID matches the scene name."""
        self.assertEqual(self.reader.sequence_id, self.scene_name)


if __name__ == "__main__":
    unittest.main()
