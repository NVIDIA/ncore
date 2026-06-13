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

"""Integration tests for the Argoverse 2 data converter (V4 format).

Requires the AV2_DIR environment variable pointing to an Argoverse 2 Sensor
Dataset root directory organised as ``{AV2_DIR}/{split}/{log_id}/...``.

Set AV2_SPLIT to override the default split (``val``). The first log in the split
is used for testing.
"""

import os
import tempfile
import unittest

from typing import Literal, cast

import numpy as np

from parameterized import parameterized_class
from upath import UPath

from ncore.impl.data.types import (
    IdealPinholeCameraModelParameters,
    RowOffsetStructuredSpinningLidarModelParameters,
    ShutterType,
)
from ncore.impl.data.v4.components import (
    CameraSensorComponent,
    CuboidsComponent,
    IntrinsicsComponent,
    LidarSensorComponent,
    PosesComponent,
    RadarSensorComponent,
    SequenceComponentGroupsReader,
)
from tools.data_converter.argoverse2.converter import Argoverse2Converter4, Argoverse2Converter4Config
from tools.data_converter.argoverse2.utils import CAMERA_NAMES, LIDAR_NAMES, list_log_ids


@parameterized_class(
    ("store_type",),
    [
        ("itar",),
        ("directory",),
    ],
)
class TestArgoverse2Converter(unittest.TestCase):
    """Integration tests for the Argoverse 2 data converter.

    Requires AV2_DIR environment variable pointing to an Argoverse 2 Sensor Dataset
    root. Uses the first log in the split for testing.
    """

    store_type: Literal["itar", "directory"]

    @classmethod
    def setUpClass(cls):
        cls.av2_dir = os.environ.get("AV2_DIR")
        if cls.av2_dir is None:
            raise unittest.SkipTest("AV2_DIR not set -- skipping Argoverse 2 integration tests")

        cls.split = os.environ.get("AV2_SPLIT", "val")

        log_ids = list_log_ids(UPath(cls.av2_dir) / cls.split)
        assert log_ids, f"No logs found under {cls.av2_dir}/{cls.split}"
        cls.log_id = log_ids[0]

        cls._tempdir = tempfile.TemporaryDirectory(prefix="argoverse2_test_")
        cls.output_dir = cls._tempdir.name

        config = Argoverse2Converter4Config(
            root_dir=cls.av2_dir,
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
            split=cls.split,
            log_id=cls.log_id,
            store_type=cls.store_type,
            component_group_profile="separate-sensors",
            store_sequence_meta=True,
        )
        Argoverse2Converter4.convert(config)

        seq_dirs = [d for d in UPath(cls.output_dir).iterdir() if d.is_dir()]
        assert len(seq_dirs) == 1, f"Expected 1 sequence dir, found {len(seq_dirs)}: {seq_dirs}"
        cls.seq_dir = seq_dirs[0]

        meta_files = list(cls.seq_dir.glob("*.json"))
        assert len(meta_files) == 1, f"Expected 1 meta JSON, found {len(meta_files)}"
        cls.reader = SequenceComponentGroupsReader([meta_files[0]])

    @classmethod
    def tearDownClass(cls):
        cls._tempdir.cleanup()

    # --- Poses ----------------------------------------------------------------

    def test_sequence_has_dynamic_rig_to_world_pose(self):
        poses_readers = self.reader.open_component_readers(PosesComponent.Reader)
        self.assertEqual(len(poses_readers), 1)
        poses_reader = list(poses_readers.values())[0]

        poses, timestamps = poses_reader.get_dynamic_pose("rig", "world")
        self.assertEqual(poses.shape[1:], (4, 4))
        self.assertGreater(poses.shape[0], 0)
        self.assertEqual(timestamps.shape[0], poses.shape[0])

    def test_sequence_has_static_world_to_world_global(self):
        """world_global is the AV2 city frame; verify the static anchor exists."""
        poses_readers = self.reader.open_component_readers(PosesComponent.Reader)
        poses_reader = list(poses_readers.values())[0]

        static_poses = dict(poses_reader.get_static_poses())
        self.assertIn(("world", "world_global"), static_poses)
        self.assertEqual(static_poses[("world", "world_global")].shape, (4, 4))

    def test_first_real_pose_near_identity(self):
        """The anchored ego pose is stored as relative identity in the trajectory.

        The first pose's city_SE3_egovehicle is the world_global anchor, so its
        relative rig -> world pose must be (near) identity. Boundary extrapolation
        may prepend an extra pose, so we locate the identity pose rather than
        assuming a fixed index.
        """
        poses_readers = self.reader.open_component_readers(PosesComponent.Reader)
        poses_reader = list(poses_readers.values())[0]

        poses, _ = poses_reader.get_dynamic_pose("rig", "world")
        deviations = np.linalg.norm(poses - np.eye(4, dtype=np.float32), axis=(1, 2))
        np.testing.assert_array_almost_equal(poses[int(np.argmin(deviations))], np.eye(4, dtype=np.float32), decimal=3)

    # --- Cameras --------------------------------------------------------------

    def test_nine_cameras_exist(self):
        camera_readers = self.reader.open_component_readers(CameraSensorComponent.Reader)
        self.assertEqual(set(camera_readers.keys()), set(CAMERA_NAMES))
        for cam_id, cam_reader in camera_readers.items():
            self.assertGreater(cam_reader.frames_count, 0, f"{cam_id} should have frames")

    def test_original_distortion_coefficients_preserved_in_metadata(self):
        # The released imagery is undistorted (so the stored model is distortion-free),
        # but the original lens k1/k2/k3 are preserved per camera as provenance.
        camera_readers = self.reader.open_component_readers(CameraSensorComponent.Reader)
        for cam_id, cam_reader in camera_readers.items():
            meta = cam_reader.generic_meta_data
            self.assertIn("av2_original_distortion", meta, f"{cam_id} missing distortion provenance")
            distortion = meta["av2_original_distortion"]
            self.assertEqual(set(distortion), {"k1", "k2", "k3"}, f"{cam_id} distortion keys")
            for key, value in distortion.items():
                self.assertIsInstance(value, float, f"{cam_id} {key} should be a float")

    def test_camera_intrinsics_ideal_pinhole_global_shutter(self):
        intrinsics_readers = self.reader.open_component_readers(IntrinsicsComponent.Reader)
        self.assertEqual(len(intrinsics_readers), 1)
        intrinsics_reader = list(intrinsics_readers.values())[0]

        for cam_id in CAMERA_NAMES:
            params = intrinsics_reader.get_camera_model_parameters(cam_id)
            # AV2 imagery is shipped undistorted, so an ideal (distortion-free)
            # pinhole is the exact model.
            self.assertIsInstance(params, IdealPinholeCameraModelParameters)
            params = cast(IdealPinholeCameraModelParameters, params)
            self.assertEqual(params.shutter_type, ShutterType.GLOBAL)
            self.assertTrue(np.all(params.focal_length > 0))

    def test_camera_extrinsics_stored_as_static_poses(self):
        poses_readers = self.reader.open_component_readers(PosesComponent.Reader)
        poses_reader = list(poses_readers.values())[0]

        static_poses = dict(poses_reader.get_static_poses())
        for cam_id in CAMERA_NAMES:
            self.assertIn((cam_id, "rig"), static_poses)

    # --- Lidar ----------------------------------------------------------------

    def test_two_lidar_units_exist(self):
        lidar_readers = self.reader.open_component_readers(LidarSensorComponent.Reader)
        self.assertEqual(set(lidar_readers.keys()), set(LIDAR_NAMES))
        for lidar_id, lidar_reader in lidar_readers.items():
            self.assertGreater(lidar_reader.frames_count, 0, f"{lidar_id} should have frames")

    def test_lidar_extrinsics_stored_as_static_poses(self):
        poses_readers = self.reader.open_component_readers(PosesComponent.Reader)
        poses_reader = list(poses_readers.values())[0]

        static_poses = dict(poses_reader.get_static_poses())
        for lidar_id in LIDAR_NAMES:
            self.assertIn((lidar_id, "rig"), static_poses)

    def test_lidar_directions_unit_norm(self):
        lidar_readers = self.reader.open_component_readers(LidarSensorComponent.Reader)
        lidar_reader = lidar_readers["up_lidar"]
        ts = int(lidar_reader.frames_timestamps_us[0, 1])  # end-of-frame timestamp key
        direction = lidar_reader.get_frame_ray_bundle_data(ts, "direction")
        norms = np.linalg.norm(direction, axis=1)
        # Zero-distance rays may have zero direction; check the populated ones.
        nonzero = norms > 0
        np.testing.assert_allclose(norms[nonzero], 1.0, atol=1e-4)

    def test_lidar_unit_split_recovered_from_geometry(self):
        """The two units carry comparable point counts (~half the sweep each).

        Each VLP-32C contributes 32 of the 64 beams, so a correct split yields
        roughly balanced point counts per unit (allowing for differing FOV
        occupancy).
        """
        lidar_readers = self.reader.open_component_readers(LidarSensorComponent.Reader)
        counts = {}
        for unit in ("up_lidar", "down_lidar"):
            reader = lidar_readers[unit]
            ts = int(reader.frames_timestamps_us[0, 1])
            counts[unit] = len(reader.get_frame_ray_bundle_data(ts, "direction"))
        ratio = min(counts.values()) / max(counts.values())
        self.assertGreater(ratio, 0.5, f"Lidar unit point counts unbalanced: {counts}")

    # --- Lidar structured model -----------------------------------------------

    def test_lidar_intrinsics_vlp32c_model_per_unit(self):
        """Each unit stores a VLP-32C structured model (32 rows) as intrinsics."""
        intrinsics_reader = list(self.reader.open_component_readers(IntrinsicsComponent.Reader).values())[0]
        for unit in LIDAR_NAMES:
            model = intrinsics_reader.get_lidar_model_parameters(unit)
            self.assertIsInstance(model, RowOffsetStructuredSpinningLidarModelParameters)
            model = cast(RowOffsetStructuredSpinningLidarModelParameters, model)
            self.assertEqual(model.n_rows, 32)
            self.assertGreater(model.n_columns, 100)
            self.assertIn(model.spinning_direction, ("cw", "ccw"))

    def test_lidar_model_elements_in_bounds(self):
        """Stored per-point model elements index valid (row, column) cells."""
        intrinsics_reader = list(self.reader.open_component_readers(IntrinsicsComponent.Reader).values())[0]
        lidar_readers = self.reader.open_component_readers(LidarSensorComponent.Reader)
        for unit in LIDAR_NAMES:
            model = cast(
                RowOffsetStructuredSpinningLidarModelParameters,
                intrinsics_reader.get_lidar_model_parameters(unit),
            )
            reader = lidar_readers[unit]
            ts = int(reader.frames_timestamps_us[0, 1])
            elem = reader.get_frame_ray_bundle_data(ts, "model_element")
            self.assertEqual(elem.shape[1], 2)
            self.assertEqual(elem.dtype, np.uint16)
            self.assertTrue(np.all(elem[:, 0] < model.n_rows), "row index out of bounds")
            self.assertTrue(np.all(elem[:, 1] < model.n_columns), "column index out of bounds")

    def test_lidar_model_reconstructs_directions(self):
        """Model-predicted directions match stored native directions (far-range).

        Validates the firing-pattern reconstruction: the structured model, indexed
        by the stored per-point (row, column), should reproduce the stored ray
        directions to within a small angular error for far-range returns.
        """
        from ncore.impl.sensors.lidar import StructuredLidarModel

        intrinsics_reader = list(self.reader.open_component_readers(IntrinsicsComponent.Reader).values())[0]
        lidar_readers = self.reader.open_component_readers(LidarSensorComponent.Reader)
        for unit in LIDAR_NAMES:
            model_params = cast(
                RowOffsetStructuredSpinningLidarModelParameters,
                intrinsics_reader.get_lidar_model_parameters(unit),
            )
            reader = lidar_readers[unit]
            ts = int(reader.frames_timestamps_us[0, 1])
            direction = reader.get_frame_ray_bundle_data(ts, "direction")
            elem = reader.get_frame_ray_bundle_data(ts, "model_element")
            distance = np.asarray(reader._get_ray_bundle_returns_group(ts)["distance_m"])[0]

            far = np.isfinite(distance) & (distance > 20.0) & (np.linalg.norm(direction, axis=1) > 0)
            self.assertGreater(int(far.sum()), 100, f"{unit}: too few far returns to validate")

            model = StructuredLidarModel.maybe_from_parameters(model_params, device="cpu")
            assert model is not None
            predicted = model.elements_to_sensor_points(elem[far], np.ones(int(far.sum()), dtype=np.float32))
            predicted = predicted.cpu().numpy()
            predicted /= np.linalg.norm(predicted, axis=1, keepdims=True)
            cos = np.clip(np.sum(predicted * direction[far], axis=1), -1.0, 1.0)
            median_err_deg = float(np.degrees(np.median(np.arccos(cos))))
            # The structured model is reconstructed from offset_ns + laser_number on the
            # decompensated cloud, with empirical per-row azimuth offsets. This yields
            # sub-0.1 deg far-range reconstruction (on par with native-column sensors).
            self.assertLess(median_err_deg, 0.2, f"{unit}: model direction error {median_err_deg:.3f} deg too high")

    # --- No radar -------------------------------------------------------------

    def test_no_radar(self):
        radar_readers = self.reader.open_component_readers(RadarSensorComponent.Reader)
        self.assertEqual(len(radar_readers), 0)

    # --- Cuboids --------------------------------------------------------------

    def test_cuboids_in_rig_frame(self):
        """Cuboids are stored in the native ``rig`` frame (no ego pose baked in)."""
        cuboid_readers = self.reader.open_component_readers(CuboidsComponent.Reader)
        if not cuboid_readers:
            self.skipTest("No cuboids (test split)")
        cuboid_reader = list(cuboid_readers.values())[0]
        observations = list(cuboid_reader.get_observations())
        self.assertGreater(len(observations), 0)
        for obs in observations[:50]:
            self.assertEqual(obs.reference_frame_id, "rig")

    def test_cuboids_align_with_lidar(self):
        """A reasonable fraction of lidar points fall inside annotated cuboids.

        This is the regression guard for the lidar decompensation reference bug: if
        the points were decompensated against the wrong reference (or the cuboids
        mis-referenced), almost no points would land inside the boxes. We transform
        the stored (decompensated) first-frame lidar points to ``world`` via their
        own per-point rig pose, transform the active cuboids from ``rig`` at the
        sweep timestamp to ``world``, and count points inside.
        """
        from ncore.impl.common.transformations import is_within_3d_bboxes, transform_bbox

        cuboid_readers = self.reader.open_component_readers(CuboidsComponent.Reader)
        if not cuboid_readers:
            self.skipTest("No cuboids (test split)")

        poses_reader = list(self.reader.open_component_readers(PosesComponent.Reader).values())[0]
        lidar_reader = self.reader.open_component_readers(LidarSensorComponent.Reader)["up_lidar"]
        frame_start_us, frame_end_us = (int(v) for v in lidar_reader.frames_timestamps_us[0])
        ts = frame_end_us  # reader frame key is the end-of-frame timestamp

        # The cuboid reference timestamp is the AV2 sweep reference time, which is
        # the start of the point window (offset_ns runs forward from it).
        cuboid_ts = frame_start_us

        # Each lidar point is in its own per-point-time sensor frame; transform via
        # sensor -> rig (static) and rig -> world at the point's own timestamp.
        static = dict(poses_reader.get_static_poses())
        T_up_rig = static[("up_lidar", "rig")]
        rig_poses, pose_ts = poses_reader.get_dynamic_pose("rig", "world")

        direction = lidar_reader.get_frame_ray_bundle_data(ts, "direction")
        distance = np.asarray(lidar_reader._get_ray_bundle_returns_group(ts)["distance_m"])[0]
        point_ts = lidar_reader.get_frame_ray_bundle_data(ts, "timestamp_us").astype(np.int64)
        valid = np.isfinite(distance) & (distance > 0)
        pts_sensor = direction[valid] * distance[valid, None]
        pts_rig = (T_up_rig[:3, :3] @ pts_sensor.T).T + T_up_rig[:3, 3]
        # Per-point rig -> world using the nearest stored pose (poses are dense).
        nearest = np.searchsorted(pose_ts.astype(np.int64), point_ts[valid]).clip(0, len(rig_poses) - 1)
        T_pts = rig_poses[nearest]
        pts_world = np.einsum("nij,nj->ni", T_pts[:, :3, :3], pts_rig) + T_pts[:, :3, 3]

        observations = list(list(cuboid_readers.values())[0].get_observations())
        # cuboids active at this sweep (cuboid ref ts == sweep start)
        active = [o for o in observations if abs(o.reference_frame_timestamp_us - cuboid_ts) < 2000]
        self.assertGreater(len(active), 0, "no cuboids active at first lidar frame")

        # Cuboids are in rig at the sweep timestamp; bring them to world via the
        # rig pose at that timestamp.
        cuboid_pose_idx = int(np.argmin(np.abs(pose_ts.astype(np.int64) - cuboid_ts)))
        T_rig_world_cuboid = rig_poses[cuboid_pose_idx]
        boxes = np.stack([transform_bbox(o.bbox3.to_array().astype(np.float64), T_rig_world_cuboid) for o in active])
        inside = is_within_3d_bboxes(pts_world.astype(np.float64), boxes.astype(np.float64))
        n_inside = int(inside.any(axis=1).sum())
        self.assertGreater(
            n_inside, 50, f"only {n_inside} lidar points inside any cuboid -- likely a frame/timestamp shift"
        )
