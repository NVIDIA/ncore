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

"""Integration tests for the KITTI raw data converter (V4 format)."""

import os
import tempfile
import unittest

from typing import Literal, cast

import numpy as np

from parameterized import parameterized_class
from upath import UPath

from ncore.impl.data.types import OpenCVPinholeCameraModelParameters
from ncore.impl.data.v4.components import (
    CameraSensorComponent,
    CuboidsComponent,
    IntrinsicsComponent,
    LidarSensorComponent,
    PosesComponent,
    SequenceComponentGroupsReader,
)
from tools.data_converter.kitti.converter import KittiConverter4, KittiConverter4Config


@parameterized_class(
    ("store_type"),
    [
        ("itar",),
        ("directory",),
    ],
)
class TestKittiConverter(unittest.TestCase):
    """Integration tests for KITTI raw data converter.

    Requires KITTI_RAW_DIR environment variable pointing to a KITTI raw date
    directory (e.g. /tmp/kitti_test_data/2011_09_26) containing at least one
    drive sequence with calibration files and tracklet labels.
    """

    store_type: Literal["itar", "directory"]

    @classmethod
    def setUpClass(cls):
        cls.kitti_dir = os.environ.get("KITTI_RAW_DIR")
        if cls.kitti_dir is None:
            raise unittest.SkipTest("KITTI_RAW_DIR not set -- skipping KITTI integration tests")

        cls._tempdir = tempfile.TemporaryDirectory(prefix="kitti_test_")
        cls.output_dir = cls._tempdir.name

        # Run the converter once for all tests
        config = KittiConverter4Config(
            root_dir=cls.kitti_dir,
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
            store_type=cls.store_type,
            component_group_profile="separate-sensors",
            store_sequence_meta=True,
        )
        KittiConverter4.convert(config)

        # Find output sequence directory
        seq_dirs = list(UPath(cls.output_dir).glob("*_drive_*_sync"))
        assert len(seq_dirs) == 1, f"Expected 1 sequence, found {len(seq_dirs)}"
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
        """Verify dynamic rig-> world ego pose exists."""
        poses_readers = self.reader.open_component_readers(PosesComponent.Reader)
        self.assertEqual(len(poses_readers), 1)
        poses_reader = list(poses_readers.values())[0]

        poses, timestamps = poses_reader.get_dynamic_pose("rig", "world")
        self.assertEqual(poses.shape[1:], (4, 4))
        self.assertGreater(poses.shape[0], 0)
        self.assertEqual(timestamps.shape[0], poses.shape[0])

    def test_sequence_has_static_world_to_world_global(self):
        """Verify static world-> world_global pose exists."""
        poses_readers = self.reader.open_component_readers(PosesComponent.Reader)
        poses_reader = list(poses_readers.values())[0]

        static_poses = dict(poses_reader.get_static_poses())
        self.assertIn(("world", "world_global"), static_poses)
        pose = static_poses[("world", "world_global")]
        self.assertEqual(pose.shape, (4, 4))

    def test_first_pose_near_identity(self):
        """Verify first ego pose is near identity (local origin)."""
        poses_readers = self.reader.open_component_readers(PosesComponent.Reader)
        poses_reader = list(poses_readers.values())[0]

        poses, _ = poses_reader.get_dynamic_pose("rig", "world")
        first_pose = poses[0]
        np.testing.assert_array_almost_equal(first_pose, np.eye(4), decimal=3)

    def test_oxts_generic_data(self):
        """Verify raw OXTS data stored as generic data on poses component."""
        poses_readers = self.reader.open_component_readers(PosesComponent.Reader)
        poses_reader = list(poses_readers.values())[0]

        # Check generic data arrays exist
        self.assertTrue(poses_reader.has_generic_data("oxts_data"))
        self.assertTrue(poses_reader.has_generic_data("oxts_timestamps_us"))

        oxts_data = poses_reader.get_generic_data("oxts_data")
        self.assertEqual(oxts_data.ndim, 2)
        self.assertGreater(oxts_data.shape[0], 0)
        self.assertEqual(oxts_data.shape[1], 30)  # 30 OXTS fields

        # Check field names in metadata
        meta = poses_reader.generic_meta_data
        self.assertIn("oxts_field_names", meta)
        oxts_field_names = cast(list, meta["oxts_field_names"])
        self.assertEqual(len(oxts_field_names), 30)
        self.assertEqual(oxts_field_names[0], "lat")

    # --- Cameras --------------------------------------------------------------

    def test_four_cameras_exist(self):
        """Verify all 4 camera readers exist with frames."""
        camera_readers = self.reader.open_component_readers(CameraSensorComponent.Reader)
        expected_ids = {"camera_gray_left", "camera_gray_right", "camera_color_left", "camera_color_right"}
        self.assertEqual(set(camera_readers.keys()), expected_ids)
        for cam_id, cam_reader in camera_readers.items():
            self.assertGreater(cam_reader.frames_count, 0, f"{cam_id} should have frames")

    def test_camera_intrinsics_zero_distortion(self):
        """Verify camera intrinsics have zero distortion (rectified images)."""
        intrinsics_readers = self.reader.open_component_readers(IntrinsicsComponent.Reader)
        self.assertEqual(len(intrinsics_readers), 1)
        intrinsics_reader = list(intrinsics_readers.values())[0]

        for cam_id in ["camera_gray_left", "camera_gray_right", "camera_color_left", "camera_color_right"]:
            params = intrinsics_reader.get_camera_model_parameters(cam_id)
            self.assertIsInstance(params, OpenCVPinholeCameraModelParameters)
            params = cast(OpenCVPinholeCameraModelParameters, params)
            # Distortion should be zero for rectified images
            np.testing.assert_array_equal(params.radial_coeffs, np.zeros(6, dtype=np.float32))
            np.testing.assert_array_equal(params.tangential_coeffs, np.zeros(2, dtype=np.float32))
            # Focal length should be positive
            self.assertTrue(np.all(params.focal_length > 0))

    def test_camera_extrinsics_stored_as_static_poses(self):
        """Verify each camera has a static sensor-> rig extrinsic pose."""
        poses_readers = self.reader.open_component_readers(PosesComponent.Reader)
        poses_reader = list(poses_readers.values())[0]

        static_poses = dict(poses_reader.get_static_poses())
        for cam_id in ["camera_gray_left", "camera_gray_right", "camera_color_left", "camera_color_right"]:
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
        """Verify lidar has a static sensor-> rig extrinsic pose."""
        poses_readers = self.reader.open_component_readers(PosesComponent.Reader)
        poses_reader = list(poses_readers.values())[0]

        static_poses = dict(poses_reader.get_static_poses())
        self.assertIn(("lidar_top", "rig"), static_poses)
        pose = static_poses[("lidar_top", "rig")]
        self.assertEqual(pose.shape, (4, 4))

    # --- Cuboids (Tracklets) -------------------------------------------------

    def test_cuboid_observations_exist(self):
        """Verify cuboid track observations were stored from tracklets."""
        cuboid_readers = self.reader.open_component_readers(CuboidsComponent.Reader)
        self.assertEqual(len(cuboid_readers), 1)
        cuboid_reader = list(cuboid_readers.values())[0]

        observations = list(cuboid_reader.get_observations())
        self.assertGreater(len(observations), 0)

        # Check first observation has expected fields
        obs = observations[0]
        self.assertIsInstance(obs.track_id, str)
        self.assertIsInstance(obs.class_id, str)
        self.assertEqual(obs.reference_frame_id, "lidar_top")

    # --- Sequence Meta --------------------------------------------------------

    def test_sequence_meta_file_exists(self):
        """Verify sequence meta JSON file was written."""
        meta_files = list(self.seq_dir.glob("*.json"))
        self.assertEqual(len(meta_files), 1)

    def test_sequence_id_matches_drive_name(self):
        """Verify the sequence ID matches the drive directory name."""
        self.assertIn("drive", self.reader.sequence_id)
        self.assertIn("sync", self.reader.sequence_id)


if __name__ == "__main__":
    unittest.main()
