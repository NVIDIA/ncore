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

from __future__ import annotations

import json
import os
import unittest

from typing import Any, Dict, Tuple

import numpy as np
import parameterized
import torch

from scipy.spatial.transform import Rotation

from ncore.impl.data.types import (
    FThetaCameraModelParameters,
    RowOffsetStructuredSpinningLidarModelParameters,
    ShutterType,
)
from ncore.impl.sensors.camera import FThetaCameraModel
from ncore.impl.sensors.common import (
    RollingShutterSolver,
    rotmat_to_unitquat,
    to_torch,
    unitquat_slerp,
    unitquat_to_rotmat,
)
from ncore.impl.sensors.lidar import RowOffsetStructuredSpinningLidarModel


# =============================================================================
# Device selection: run on CPU always, and on CUDA when available and not disabled.
# =============================================================================


def _get_test_devices() -> Tuple[torch.device, ...]:
    """Return the devices to test based on NCORE_NO_GPU_TESTS environment variable."""
    if os.environ.get("NCORE_NO_GPU_TESTS", "0") in ("1", "true", "True", "TRUE"):
        return (torch.device("cpu"),)
    if torch.version.cuda is None:  # ty: ignore[possibly-missing-submodule]
        return (torch.device("cpu"),)
    return (torch.device("cpu"), torch.device("cuda"))


# --- Mock projectors for Section 1 ---


class _MockProjector:
    """Mock projector that converges immediately (stable output).

    project(): returns input rays' first 2 columns as "projected", all valid.
    relative_frame_time(): returns projected[:, 0] clamped to [0, 1].
    """

    def __init__(self) -> None:
        self.call_count = 0

    def project(self, sensor_points: torch.Tensor) -> RollingShutterSolver.ProjectionResult:
        projected = sensor_points[:, :2].clone()
        valid_flag = torch.ones(sensor_points.shape[0], dtype=torch.bool, device=sensor_points.device)
        return RollingShutterSolver.ProjectionResult(projected=projected, valid_flag=valid_flag)

    def relative_frame_time(self, projected: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        return projected[:, 0].clamp(0.0, 1.0)


class _NeverConvergesProjector:
    """Mock projector that never converges (time shifts each iteration)."""

    def __init__(self) -> None:
        self.call_count = 0

    def project(self, sensor_points: torch.Tensor) -> RollingShutterSolver.ProjectionResult:
        projected = sensor_points[:, :2].clone()
        valid_flag = torch.ones(sensor_points.shape[0], dtype=torch.bool, device=sensor_points.device)
        return RollingShutterSolver.ProjectionResult(projected=projected, valid_flag=valid_flag)

    def relative_frame_time(self, projected: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        return (projected[:, 0] + 0.1 * self.call_count).clamp(0.0, 1.0)


class _AllInvalidProjector:
    """Mock projector where all projections are invalid."""

    def __init__(self) -> None:
        self.call_count = 0

    def project(self, sensor_points: torch.Tensor) -> RollingShutterSolver.ProjectionResult:
        projected = sensor_points[:, :2].clone()
        valid_flag = torch.zeros(sensor_points.shape[0], dtype=torch.bool, device=sensor_points.device)
        return RollingShutterSolver.ProjectionResult(projected=projected, valid_flag=valid_flag)

    def relative_frame_time(self, projected: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        return projected[:, 0].clamp(0.0, 1.0)


def _make_ftheta_camera(device: torch.device, dtype: torch.dtype) -> FThetaCameraModel:
    """Create a simple FTheta camera model for testing."""
    params = FThetaCameraModelParameters(
        resolution=np.array([1920, 1080], dtype=np.uint64),
        shutter_type=ShutterType.ROLLING_TOP_TO_BOTTOM,
        principal_point=np.array([959.5, 539.5], dtype=np.float32),
        reference_poly=FThetaCameraModelParameters.PolynomialType.PIXELDIST_TO_ANGLE,
        pixeldist_to_angle_poly=np.array([0.0, 1.0 / 500.0], dtype=np.float32),
        angle_to_pixeldist_poly=np.array([0.0, 500.0], dtype=np.float32),
        max_angle=float(np.radians(80)),
    )
    return FThetaCameraModel(params, device=device, dtype=dtype)


def _make_pose(rotvec_deg: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Create a 4x4 pose matrix from a rotation vector (in degrees) and translation."""
    R = Rotation.from_rotvec(np.radians(rotvec_deg)).as_matrix().astype(np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# =============================================================================
# Section 1: Solver unit tests (mock projector)
# Parameterized over cpu and cuda devices.
# =============================================================================


@parameterized.parameterized_class(("device",), [(d,) for d in _get_test_devices()])
class TestRollingShutterSolver(unittest.TestCase):
    """Unit tests for RollingShutterSolver.solve using mock projectors."""

    device: torch.device

    def setUp(self) -> None:
        self.n_points = 10
        # World points in front of sensor (z > 0)
        torch.manual_seed(42)
        self.world_points = torch.randn(self.n_points, 3, device=self.device, dtype=torch.float32)
        self.world_points[:, 2] = self.world_points[:, 2].abs() + 1.0

        # Non-trivial poses: 5-degree rotation + translation motion between start and end
        self.T_start = torch.from_numpy(
            _make_pose(np.array([1.0, -2.0, 0.5]), np.array([0.0, 0.0, 0.0], dtype=np.float32))
        ).to(device=self.device)
        self.T_end = torch.from_numpy(
            _make_pose(np.array([2.0, 3.0, -1.0]), np.array([0.05, -0.02, 0.01], dtype=np.float32))
        ).to(device=self.device)

    def test_convergence_early_stop(self) -> None:
        """Verify solver stops before max_iterations when convergence is reached."""
        projector = _MockProjector()
        result = RollingShutterSolver.solve(
            world_points=self.world_points,
            T_world_sensor_start=self.T_start,
            T_world_sensor_end=self.T_end,
            max_iterations=20,
            stop_t_delta=1e-6,
            projector=projector,
        )

        self.assertIsInstance(result, RollingShutterSolver.Result)
        self.assertTrue(result.projection_valid.all())
        # The mock projector produces stable output, so it should converge well before max_iterations.
        # compute_relative_frame_time is called once for initialization + once per iteration.
        self.assertLess(projector.call_count, 15)

    def test_max_iterations(self) -> None:
        """Verify solver runs exactly max_iterations when it never converges."""
        max_iter = 7
        projector = _NeverConvergesProjector()
        result = RollingShutterSolver.solve(
            world_points=self.world_points,
            T_world_sensor_start=self.T_start,
            T_world_sensor_end=self.T_end,
            max_iterations=max_iter,
            stop_t_delta=1e-10,
            projector=projector,
        )

        self.assertIsInstance(result, RollingShutterSolver.Result)
        # compute_relative_frame_time is called once for initialization + once per iteration
        self.assertEqual(projector.call_count, max_iter + 1)

    def test_result_shapes(self) -> None:
        """Verify that result tensors have correct shapes matching input n."""
        projector = _MockProjector()
        result = RollingShutterSolver.solve(
            world_points=self.world_points,
            T_world_sensor_start=self.T_start,
            T_world_sensor_end=self.T_end,
            max_iterations=5,
            stop_t_delta=1e-6,
            projector=projector,
        )

        # All points are valid with the mock projector
        self.assertEqual(result.projection_valid.shape, (self.n_points,))
        self.assertTrue(result.projection_valid.all())
        self.assertEqual(result.rot_rs.shape, (self.n_points, 3, 3))
        self.assertEqual(result.trans_rs.shape, (self.n_points, 3))
        self.assertEqual(result.t.shape, (self.n_points,))
        self.assertEqual(result.projection.shape, (self.n_points, 2))
        self.assertEqual(result.projection_init.shape, (self.n_points, 2))

    def test_pose_interpolation_midpoint(self) -> None:
        """With t=0.5, verify interpolated translation is midpoint of start/end poses."""
        T_start = torch.from_numpy(
            _make_pose(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0], dtype=np.float32))
        ).to(device=self.device)
        T_end = torch.from_numpy(_make_pose(np.array([0.0, 0.0, 0.0]), np.array([2.0, 4.0, 6.0], dtype=np.float32))).to(
            device=self.device
        )

        # Projector that always returns t=0.5
        class _HalfTimeProjector:
            def __init__(self) -> None:
                self.call_count = 0

            def project(self, sensor_points: torch.Tensor) -> RollingShutterSolver.ProjectionResult:
                projected = sensor_points[:, :2].clone()
                valid_flag = torch.ones(sensor_points.shape[0], dtype=torch.bool, device=sensor_points.device)
                return RollingShutterSolver.ProjectionResult(projected=projected, valid_flag=valid_flag)

            def relative_frame_time(self, projected: torch.Tensor) -> torch.Tensor:
                self.call_count += 1
                return torch.full((projected.shape[0],), 0.5, device=projected.device, dtype=projected.dtype)

        projector = _HalfTimeProjector()
        result = RollingShutterSolver.solve(
            world_points=self.world_points,
            T_world_sensor_start=T_start,
            T_world_sensor_end=T_end,
            max_iterations=1,
            stop_t_delta=1e-10,
            projector=projector,
        )

        # Translation should be midpoint: [1, 2, 3]
        expected_trans = torch.tensor([1.0, 2.0, 3.0], device=self.device, dtype=torch.float32)
        for i in range(self.n_points):
            torch.testing.assert_close(result.trans_rs[i], expected_trans)

        # Rotation should remain identity (both quats are identity)
        expected_rot = torch.eye(3, device=self.device, dtype=torch.float32)
        for i in range(self.n_points):
            torch.testing.assert_close(result.rot_rs[i], expected_rot, atol=1e-6, rtol=1e-5)

    def test_all_invalid_projections(self) -> None:
        """Verify solver returns early when all initial projections are invalid."""
        projector = _AllInvalidProjector()
        result = RollingShutterSolver.solve(
            world_points=self.world_points,
            T_world_sensor_start=self.T_start,
            T_world_sensor_end=self.T_end,
            max_iterations=20,
            stop_t_delta=1e-6,
            projector=projector,
        )

        self.assertIsInstance(result, RollingShutterSolver.Result)
        # When all initial projections are invalid, projection_valid is all-False and solver returns early
        self.assertFalse(result.projection_valid.any())
        self.assertEqual(result.rot_rs.shape, (0, 3, 3))
        self.assertEqual(result.trans_rs.shape, (0, 3))
        self.assertEqual(result.t.shape, (0,))
        # compute_relative_frame_time is never called (no valid points)
        self.assertEqual(projector.call_count, 0)

    def test_relative_times_in_unit_range(self) -> None:
        """Verify output relative frame times are within [0, 1]."""
        projector = _MockProjector()
        result = RollingShutterSolver.solve(
            world_points=self.world_points,
            T_world_sensor_start=self.T_start,
            T_world_sensor_end=self.T_end,
            max_iterations=10,
            stop_t_delta=1e-6,
            projector=projector,
        )

        self.assertTrue((result.t >= 0.0).all())
        self.assertTrue((result.t <= 1.0).all())


# =============================================================================
# Section 2: Camera integration tests (real FThetaCameraModel)
# Camera uses ROLLING_TOP_TO_BOTTOM shutter: relative frame time t = row / (height - 1).
# Parameterized over cpu and cuda devices.
# =============================================================================


@parameterized.parameterized_class(("device",), [(d,) for d in _get_test_devices()])
class TestCameraRollingShutterIntegration(unittest.TestCase):
    """Integration tests for rolling-shutter projection using a real FThetaCameraModel.

    The camera uses a ROLLING_TOP_TO_BOTTOM shutter, so relative frame time t = row / (height - 1).
    """

    device: torch.device

    def setUp(self) -> None:
        self.dtype = torch.float32
        self.camera = _make_ftheta_camera(self.device, self.dtype)
        self.height = 1080
        self.width = 1920
        # Common timestamps for a 1-second frame
        self.start_timestamp_us = 0
        self.end_timestamp_us = 1000000

    def test_self_consistency_fixed_point(self) -> None:
        """Project world points with shutter compensation and verify fixed-point property.

        Shutter direction: ROLLING_TOP_TO_BOTTOM (t increases with row index).

        For each valid output point (u, v) with time t*, the fundamental correctness
        check is: t* == v / (height - 1) within tolerance.
        Also verifies return_timestamps and return_all_projections outputs.
        """
        # Points spread across the field of view at moderate depth
        torch.manual_seed(123)
        n_world_points = 50
        world_points = torch.randn(n_world_points, 3, device=self.device, dtype=self.dtype)
        world_points[:, 2] = world_points[:, 2].abs() + 5.0  # ensure in front of camera

        T_start = _make_pose(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0], dtype=np.float32))
        T_end = _make_pose(
            np.array([0.0, 5.0, 0.0]),
            np.array([0.1, 0.0, 0.0], dtype=np.float32),
        )

        result = self.camera.world_points_to_image_points_shutter_pose(
            world_points,
            T_start,
            T_end,
            max_iterations=20,
            stop_t_delta=1e-7,
            start_timestamp_us=self.start_timestamp_us,
            end_timestamp_us=self.end_timestamp_us,
            return_valid_indices=True,
            return_timestamps=True,
        )

        image_points = result.image_points
        valid_indices = result.valid_indices
        timestamps_us = result.timestamps_us
        assert valid_indices is not None
        assert timestamps_us is not None

        self.assertGreater(len(image_points), 0, "Should have at least one valid projection")

        # Verify valid_indices: length matches image_points and indices are in [0, n_world_points)
        self.assertEqual(len(valid_indices), len(image_points))
        self.assertTrue((valid_indices >= 0).all())
        self.assertTrue((valid_indices < n_world_points).all())

        # For each valid image point, compute expected relative frame time from row
        rows = image_points[:, 1]  # v coordinate
        expected_t = rows / (self.height - 1)

        # Verify via the relative frame time API
        actual_t = self.camera.image_points_relative_frame_times(image_points)
        torch.testing.assert_close(actual_t, expected_t, atol=2e-3, rtol=2e-3)

        # Verify timestamps are consistent with t values
        # timestamps_us = start + t * (end - start) = t * 1000000
        expected_timestamps = (expected_t * (self.end_timestamp_us - self.start_timestamp_us)).to(torch.int64)
        torch.testing.assert_close(timestamps_us, expected_timestamps, atol=2000, rtol=0)

        # Also test return_all_projections: output shape should be [n_world_points, 2]
        result_all = self.camera.world_points_to_image_points_shutter_pose(
            world_points,
            T_start,
            T_end,
            max_iterations=20,
            stop_t_delta=1e-7,
            return_valid_indices=True,
            return_all_projections=True,
        )
        self.assertEqual(result_all.image_points.shape, (n_world_points, 2))
        # Valid indices from return_all_projections should match
        assert result_all.valid_indices is not None
        self.assertEqual(result_all.valid_indices.shape[0], result.image_points.shape[0])

    def test_zero_motion_matches_static(self) -> None:
        """With T_start == T_end == identity, rolling-shutter result matches static projection.

        Shutter direction: ROLLING_TOP_TO_BOTTOM.
        Includes points behind the camera (z < 0) to verify invalid points are excluded consistently.
        """
        torch.manual_seed(456)
        # Mix of valid (z > 0) and invalid (z < 0) points
        world_points = torch.randn(30, 3, device=self.device, dtype=self.dtype)
        world_points[:25, 2] = world_points[:25, 2].abs() + 3.0  # valid: in front
        world_points[25:, 2] = -world_points[25:, 2].abs() - 1.0  # invalid: behind camera

        T_identity = np.eye(4, dtype=np.float32)

        # Rolling-shutter projection with no motion
        rs_result = self.camera.world_points_to_image_points_shutter_pose(
            world_points,
            T_identity,
            T_identity,
            max_iterations=10,
            stop_t_delta=1e-6,
            return_valid_indices=True,
        )

        # Static projection: transform points to camera frame then project
        static_result = self.camera.camera_rays_to_image_points(world_points)

        # Compare valid points from both methods
        static_valid_points = static_result.image_points[static_result.valid_flag]

        # Both should produce the same valid set
        self.assertEqual(rs_result.image_points.shape[0], static_valid_points.shape[0])
        torch.testing.assert_close(rs_result.image_points, static_valid_points, atol=1e-5, rtol=1e-5)

        # Verify the invalid points (z < 0) are excluded
        # valid_indices should only reference the first 25 points (those in front)
        assert rs_result.valid_indices is not None
        self.assertTrue((rs_result.valid_indices < 25).all())

    def test_single_point(self) -> None:
        """Project a single world point to verify no shape bugs.

        Shutter direction: ROLLING_TOP_TO_BOTTOM.
        Checks all return values: image_points, valid_indices, timestamps_us.
        """
        world_point = torch.tensor([[0.0, 0.0, 5.0]], device=self.device, dtype=self.dtype)
        T_start = np.eye(4, dtype=np.float32)
        T_end = _make_pose(np.array([0.0, 0.0, 0.0]), np.array([0.01, 0.0, 0.0], dtype=np.float32))

        result = self.camera.world_points_to_image_points_shutter_pose(
            world_point,
            T_start,
            T_end,
            max_iterations=10,
            stop_t_delta=1e-6,
            start_timestamp_us=self.start_timestamp_us,
            end_timestamp_us=self.end_timestamp_us,
            return_valid_indices=True,
            return_timestamps=True,
        )

        # Single point on optical axis should be valid
        self.assertEqual(result.image_points.shape[0], 1)
        self.assertEqual(result.image_points.shape[1], 2)

        # valid_indices should be [0] for the single valid point
        assert result.valid_indices is not None
        self.assertEqual(result.valid_indices.shape[0], 1)
        self.assertEqual(result.valid_indices[0].item(), 0)

        # Timestamp should be consistent with the projected row
        assert result.timestamps_us is not None
        row = result.image_points[0, 1]
        expected_t = row / (self.height - 1)
        expected_timestamp = int(expected_t.item() * (self.end_timestamp_us - self.start_timestamp_us))
        self.assertAlmostEqual(result.timestamps_us[0].item(), expected_timestamp, delta=2000)

    def test_all_points_behind_camera(self) -> None:
        """All points with z < 0 should produce empty valid output for all return values.

        Shutter direction: ROLLING_TOP_TO_BOTTOM.
        """
        # Points behind the camera (negative z in camera frame with identity pose)
        world_points = torch.tensor(
            [
                [0.0, 0.0, -5.0],
                [1.0, 1.0, -3.0],
                [-1.0, -1.0, -10.0],
            ],
            device=self.device,
            dtype=self.dtype,
        )
        T_identity = np.eye(4, dtype=np.float32)

        result = self.camera.world_points_to_image_points_shutter_pose(
            world_points,
            T_identity,
            T_identity,
            max_iterations=10,
            stop_t_delta=1e-6,
            start_timestamp_us=self.start_timestamp_us,
            end_timestamp_us=self.end_timestamp_us,
            return_T_world_sensors=True,
            return_valid_indices=True,
            return_timestamps=True,
        )

        # No valid projections -- all outputs should be empty
        self.assertEqual(result.image_points.shape[0], 0)
        assert result.valid_indices is not None
        self.assertEqual(result.valid_indices.shape[0], 0)
        assert result.timestamps_us is not None
        self.assertEqual(result.timestamps_us.shape[0], 0)
        assert result.T_world_sensors is not None
        self.assertEqual(result.T_world_sensors.shape[0], 0)

    def test_point_valid_at_start_invalid_at_end(self) -> None:
        """A point inside FOV at start pose but outside at end pose is still processed.

        Shutter direction: ROLLING_TOP_TO_BOTTOM.
        Verifies validity by local projection with start and end poses independently.
        """
        # Point at moderate angle from optical axis
        world_point = torch.tensor([[2.0, 0.0, 5.0]], device=self.device, dtype=self.dtype)

        T_start = np.eye(4, dtype=np.float32)
        # Large rotation around Y: 60 degrees -- the point will be outside FOV at end
        T_end = _make_pose(np.array([0.0, 60.0, 0.0]), np.zeros(3, dtype=np.float32))

        # Verify by local projection: point should be valid at start pose
        T_start_torch = torch.from_numpy(T_start).to(device=self.device, dtype=self.dtype)
        cam_point_start = (T_start_torch[:3, :3] @ world_point[0] + T_start_torch[:3, 3]).unsqueeze(0)
        start_proj = self.camera.camera_rays_to_image_points(cam_point_start)
        self.assertTrue(start_proj.valid_flag[0].item(), "Point should be valid at start pose")

        # Verify by local projection: point should be invalid at end pose
        T_end_torch = torch.from_numpy(T_end).to(device=self.device, dtype=self.dtype)
        cam_point_end = (T_end_torch[:3, :3] @ world_point[0] + T_end_torch[:3, 3]).unsqueeze(0)
        end_proj = self.camera.camera_rays_to_image_points(cam_point_end)
        self.assertFalse(end_proj.valid_flag[0].item(), "Point should be invalid at end pose")

        result = self.camera.world_points_to_image_points_shutter_pose(
            world_point,
            T_start,
            T_end,
            max_iterations=10,
            stop_t_delta=1e-6,
            return_valid_indices=True,
            return_timestamps=True,
            start_timestamp_us=self.start_timestamp_us,
            end_timestamp_us=self.end_timestamp_us,
        )

        # The union logic should include this point since it's valid at start.
        # After iteration it may converge to a valid or invalid state depending on its row.
        assert result.valid_indices is not None
        if result.image_points.shape[0] > 0:
            self.assertEqual(result.valid_indices.shape[0], result.image_points.shape[0])
            self.assertEqual(result.valid_indices[0].item(), 0)
            # Verify timestamp is consistent with row
            assert result.timestamps_us is not None
            row = result.image_points[0, 1]
            expected_t = row / (self.height - 1)
            expected_timestamp = int(expected_t.item() * (self.end_timestamp_us - self.start_timestamp_us))
            self.assertAlmostEqual(result.timestamps_us[0].item(), expected_timestamp, delta=2000)
        else:
            # If the solver determined the point is ultimately invalid, valid_indices is empty
            self.assertEqual(result.valid_indices.shape[0], 0)

    def test_point_invalid_at_start_valid_at_end(self) -> None:
        """A point outside FOV at start but inside at end pose is still processed.

        Shutter direction: ROLLING_TOP_TO_BOTTOM.
        """
        # Large rotation around Y: -60 degrees at start relative to point
        # Place point such that it's behind/outside at identity but valid after rotation
        T_start = _make_pose(np.array([0.0, -60.0, 0.0]), np.zeros(3, dtype=np.float32))
        T_end = np.eye(4, dtype=np.float32)

        # This point is on the optical axis for identity pose -- valid at end
        world_point = torch.tensor([[0.0, 0.0, 5.0]], device=self.device, dtype=self.dtype)

        result = self.camera.world_points_to_image_points_shutter_pose(
            world_point,
            T_start,
            T_end,
            max_iterations=10,
            stop_t_delta=1e-6,
            return_valid_indices=True,
            return_timestamps=True,
            start_timestamp_us=self.start_timestamp_us,
            end_timestamp_us=self.end_timestamp_us,
        )

        # The union logic should include this point since it's valid at end
        self.assertIsNotNone(result.image_points)
        # The point should be in the valid set
        self.assertGreaterEqual(result.image_points.shape[0], 1)

        # Verify valid_indices and timestamps consistency
        assert result.valid_indices is not None
        self.assertEqual(result.valid_indices.shape[0], result.image_points.shape[0])
        assert result.timestamps_us is not None
        self.assertEqual(result.timestamps_us.shape[0], result.image_points.shape[0])

        # Verify timestamps are consistent with row-based relative time
        if result.image_points.shape[0] > 0:
            row = result.image_points[0, 1]
            expected_t = row / (self.height - 1)
            expected_timestamp = int(expected_t.item() * (self.end_timestamp_us - self.start_timestamp_us))
            self.assertAlmostEqual(result.timestamps_us[0].item(), expected_timestamp, delta=2000)

    def test_large_rotation_convergence(self) -> None:
        """30-degree rotation between start/end still converges with valid output.

        Shutter direction: ROLLING_TOP_TO_BOTTOM.
        Verifies self-consistency, valid_indices, timestamps, and return_timestamps output.
        """
        torch.manual_seed(789)
        world_points = torch.randn(20, 3, device=self.device, dtype=self.dtype)
        world_points[:, 2] = world_points[:, 2].abs() + 5.0

        T_start = np.eye(4, dtype=np.float32)
        T_end = _make_pose(
            np.array([0.0, 30.0, 0.0]),
            np.array([0.5, 0.0, 0.0], dtype=np.float32),
        )

        result = self.camera.world_points_to_image_points_shutter_pose(
            world_points,
            T_start,
            T_end,
            max_iterations=20,
            stop_t_delta=1e-6,
            start_timestamp_us=self.start_timestamp_us,
            end_timestamp_us=self.end_timestamp_us,
            return_valid_indices=True,
            return_timestamps=True,
        )

        # Should have at least some valid projections
        self.assertGreater(result.image_points.shape[0], 0)

        # Verify valid_indices is consistent
        valid_indices = result.valid_indices
        assert valid_indices is not None
        self.assertEqual(valid_indices.shape[0], result.image_points.shape[0])
        self.assertTrue((valid_indices >= 0).all())
        self.assertTrue((valid_indices < 20).all())

        # Self-consistency: for valid output points, t* == row / (height - 1)
        image_points = result.image_points
        rows = image_points[:, 1]
        expected_t = rows / (self.height - 1)
        actual_t = self.camera.image_points_relative_frame_times(image_points)

        torch.testing.assert_close(actual_t, expected_t, atol=2e-3, rtol=2e-3)

        # Verify timestamps are consistent with row-based expected_t (return_timestamps output)
        timestamps_us = result.timestamps_us
        assert timestamps_us is not None
        expected_timestamps = (expected_t * (self.end_timestamp_us - self.start_timestamp_us)).to(torch.int64)
        torch.testing.assert_close(timestamps_us, expected_timestamps, atol=2000, rtol=0)

    def test_mixed_valid_invalid_points(self) -> None:
        """Test with a mix of valid and behind-camera points to exercise validity logic.

        Shutter direction: ROLLING_TOP_TO_BOTTOM.
        Ensures invalid points are excluded from valid_indices and all returned tensors
        have consistent lengths.
        """
        torch.manual_seed(999)
        n_total = 40
        world_points = torch.randn(n_total, 3, device=self.device, dtype=self.dtype)
        # First 30 points in front, last 10 behind
        world_points[:30, 2] = world_points[:30, 2].abs() + 4.0
        world_points[30:, 2] = -world_points[30:, 2].abs() - 2.0

        T_start = _make_pose(np.array([0.0, 0.0, 0.0]), np.zeros(3, dtype=np.float32))
        T_end = _make_pose(np.array([0.0, 3.0, 0.0]), np.array([0.05, 0.0, 0.0], dtype=np.float32))

        result = self.camera.world_points_to_image_points_shutter_pose(
            world_points,
            T_start,
            T_end,
            max_iterations=15,
            stop_t_delta=1e-6,
            start_timestamp_us=self.start_timestamp_us,
            end_timestamp_us=self.end_timestamp_us,
            return_valid_indices=True,
            return_timestamps=True,
            return_T_world_sensors=True,
        )

        n_valid = result.image_points.shape[0]
        self.assertGreater(n_valid, 0)
        # At minimum some points should be valid from the front group
        self.assertLessEqual(n_valid, 30)  # cannot have more valid than in-front points

        # Check all return values have consistent sizes
        assert result.valid_indices is not None
        self.assertEqual(result.valid_indices.shape[0], n_valid)
        # All valid indices should refer to the first 30 points
        self.assertTrue((result.valid_indices < 30).all())

        assert result.timestamps_us is not None
        self.assertEqual(result.timestamps_us.shape[0], n_valid)

        assert result.T_world_sensors is not None
        self.assertEqual(result.T_world_sensors.shape, (n_valid, 4, 4))

        # Verify self-consistency of timestamps with row
        rows = result.image_points[:, 1]
        expected_t = rows / (self.height - 1)
        expected_timestamps = (expected_t * (self.end_timestamp_us - self.start_timestamp_us)).to(torch.int64)
        torch.testing.assert_close(result.timestamps_us, expected_timestamps, atol=2000, rtol=0)

    def test_return_all_projections_with_timestamps(self) -> None:
        """Verify return_all_projections fills output for all n input points and timestamps work.

        Shutter direction: ROLLING_TOP_TO_BOTTOM.
        Uses return_timestamps to obtain relative frame time from the API directly.
        """
        torch.manual_seed(321)
        n_world_points = 25
        world_points = torch.randn(n_world_points, 3, device=self.device, dtype=self.dtype)
        world_points[:, 2] = world_points[:, 2].abs() + 5.0

        T_start = _make_pose(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0], dtype=np.float32))
        T_end = _make_pose(np.array([2.0, 3.0, 0.0]), np.array([0.1, 0.0, 0.0], dtype=np.float32))

        result = self.camera.world_points_to_image_points_shutter_pose(
            world_points,
            T_start,
            T_end,
            max_iterations=20,
            stop_t_delta=1e-7,
            start_timestamp_us=self.start_timestamp_us,
            end_timestamp_us=self.end_timestamp_us,
            return_valid_indices=True,
            return_timestamps=True,
            return_all_projections=True,
        )

        # With return_all_projections, image_points is [n_world_points, 2]
        self.assertEqual(result.image_points.shape, (n_world_points, 2))

        # Valid indices still indicates which subset converged properly
        assert result.valid_indices is not None
        n_valid = result.valid_indices.shape[0]
        self.assertGreater(n_valid, 0)

        # Timestamps length corresponds to the valid subset (not all points)
        assert result.timestamps_us is not None
        self.assertEqual(result.timestamps_us.shape[0], n_valid)

        # Verify timestamps consistency: use return_timestamps relative time
        valid_image_points = result.image_points[result.valid_indices]
        rows = valid_image_points[:, 1]
        expected_t = rows / (self.height - 1)
        expected_timestamps = (expected_t * (self.end_timestamp_us - self.start_timestamp_us)).to(torch.int64)
        torch.testing.assert_close(result.timestamps_us, expected_timestamps, atol=2000, rtol=0)


# =============================================================================
# Section 3: Lidar integration test for return_all_projections
# =============================================================================


def _make_lidar(device: torch.device, dtype: torch.dtype):
    """Load the default row-offset spinning lidar model for testing."""
    with open("ncore/impl/sensors/test_data/row-offset-spinning-lidar-model-parameters.json", "r") as fp:
        params = RowOffsetStructuredSpinningLidarModelParameters.from_dict(json.load(fp))
    return RowOffsetStructuredSpinningLidarModel(params, angles_to_columns_map_init=False, device=device, dtype=dtype)


@parameterized.parameterized_class(("device",), [(d,) for d in _get_test_devices()])
class TestLidarRollingShutterReturnAllProjections(unittest.TestCase):
    """Test return_all_projections for lidar rolling-shutter projection."""

    device: torch.device

    def setUp(self) -> None:
        self.lidar = _make_lidar(self.device, torch.float32)

    def test_return_all_projections_shape(self) -> None:
        """With return_all_projections, sensor_angles has shape [n_world_points, 2]."""
        torch.manual_seed(42)
        n = 50
        # Points distributed around the vehicle
        angles_az = torch.rand(n) * 2.0 * np.pi - np.pi
        angles_el = (torch.rand(n) - 0.5) * 0.4
        ranges = torch.rand(n) * 20.0 + 5.0
        world_points = torch.zeros(n, 3, device=self.device, dtype=torch.float32)
        world_points[:, 0] = ranges * torch.cos(angles_el) * torch.cos(angles_az)
        world_points[:, 1] = ranges * torch.cos(angles_el) * torch.sin(angles_az)
        world_points[:, 2] = ranges * torch.sin(angles_el)

        T_start = np.eye(4, dtype=np.float32)
        T_end = _make_pose(np.array([0.1, 0.0, 2.0]), np.array([1.0, 0.05, 0.0], dtype=np.float32))

        # Without return_all_projections: returns only valid subset
        result_valid_only = self.lidar.world_points_to_sensor_angles_shutter_pose(
            world_points,
            T_start,
            T_end,
            max_iterations=10,
            stop_t_delta=1e-4,
            return_valid_indices=True,
        )
        n_valid = result_valid_only.sensor_angles.shape[0]
        self.assertGreater(n_valid, 0)
        self.assertLessEqual(n_valid, n)

        # With return_all_projections: returns all n points
        result_all = self.lidar.world_points_to_sensor_angles_shutter_pose(
            world_points,
            T_start,
            T_end,
            max_iterations=10,
            stop_t_delta=1e-4,
            return_valid_indices=True,
            return_all_projections=True,
        )
        self.assertEqual(result_all.sensor_angles.shape, (n, 2))

        # Valid indices should have same count
        assert result_all.valid_indices is not None
        self.assertEqual(result_all.valid_indices.shape[0], n_valid)

        # Valid points in return_all_projections should match the valid-only result
        valid_from_all = result_all.sensor_angles[result_all.valid_indices]
        torch.testing.assert_close(valid_from_all, result_valid_only.sensor_angles, atol=1e-5, rtol=1e-5)

    def test_return_all_projections_with_timestamps(self) -> None:
        """Verify return_all_projections + return_timestamps gives consistent outputs."""
        torch.manual_seed(123)
        n = 40
        angles_az = torch.rand(n) * 2.0 * np.pi - np.pi
        angles_el = (torch.rand(n) - 0.5) * 0.4
        ranges = torch.rand(n) * 20.0 + 5.0
        world_points = torch.zeros(n, 3, device=self.device, dtype=torch.float32)
        world_points[:, 0] = ranges * torch.cos(angles_el) * torch.cos(angles_az)
        world_points[:, 1] = ranges * torch.cos(angles_el) * torch.sin(angles_az)
        world_points[:, 2] = ranges * torch.sin(angles_el)

        T_start = np.eye(4, dtype=np.float32)
        T_end = _make_pose(np.array([0.2, -0.1, 5.0]), np.array([0.8, 0.15, 0.0], dtype=np.float32))

        start_ts = 0
        end_ts = 100000  # 100ms frame

        result = self.lidar.world_points_to_sensor_angles_shutter_pose(
            world_points,
            T_start,
            T_end,
            max_iterations=10,
            stop_t_delta=1e-4,
            start_timestamp_us=start_ts,
            end_timestamp_us=end_ts,
            return_valid_indices=True,
            return_timestamps=True,
            return_all_projections=True,
        )

        # sensor_angles has shape [n, 2] with return_all_projections
        self.assertEqual(result.sensor_angles.shape, (n, 2))

        # timestamps correspond to valid subset
        assert result.valid_indices is not None
        assert result.timestamps_us is not None
        n_valid = result.valid_indices.shape[0]
        self.assertGreater(n_valid, 0)
        self.assertEqual(result.timestamps_us.shape[0], n_valid)

        # Timestamps should be within [start, end]
        self.assertTrue((result.timestamps_us >= start_ts).all())
        self.assertTrue((result.timestamps_us <= end_ts).all())


# =============================================================================
# Section 3: Equivalence test -- new time-delta solver vs. legacy px-distance solver
# =============================================================================


def _legacy_camera_rolling_shutter(
    camera: FThetaCameraModel,
    world_points: torch.Tensor,
    T_world_sensor_start: torch.Tensor,
    T_world_sensor_end: torch.Tensor,
    max_iterations: int = 10,
    stop_mean_error_px: float = 1e-3,
    stop_delta_mean_error_px: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Re-implementation of legacy px-distance-based rolling-shutter solver.

    This is the logic that existed before the refactor into RollingShutterSolver.
    Returns (image_points_valid, valid_mask_over_n, relative_times_valid).
    """

    # Project from start and end poses
    sensor_points_start = (T_world_sensor_start[:3, :3] @ world_points.T + T_world_sensor_start[:3, 3, None]).T
    sensor_points_end = (T_world_sensor_end[:3, :3] @ world_points.T + T_world_sensor_end[:3, 3, None]).T

    image_points_start = camera.camera_rays_to_image_points(sensor_points_start)
    image_points_end = camera.camera_rays_to_image_points(sensor_points_end)

    valid = image_points_start.valid_flag | image_points_end.valid_flag
    init_image_points = image_points_end.image_points.clone()
    init_image_points[image_points_start.valid_flag] = image_points_start.image_points[image_points_start.valid_flag]

    if not valid.any():
        return (
            torch.empty((0, 2), dtype=world_points.dtype, device=world_points.device),
            valid,
            torch.empty(0, dtype=world_points.dtype, device=world_points.device),
        )

    n_valid = int(valid.sum().item())
    world_sensor_s_quat = rotmat_to_unitquat(T_world_sensor_start[None, :3, :3])
    world_sensor_e_quat = rotmat_to_unitquat(T_world_sensor_end[None, :3, :3])
    s_quat_expanded = world_sensor_s_quat.expand(n_valid, -1)
    e_quat_expanded = world_sensor_e_quat.expand(n_valid, -1)
    transl_start = T_world_sensor_start[:3, 3]
    transl_end = T_world_sensor_end[:3, 3]

    image_points_rs_prev = init_image_points[valid, :]
    mean_error_px = 1e12

    for _ in range(max_iterations):
        t = camera.image_points_relative_frame_times(image_points_rs_prev)
        rot_rs = unitquat_to_rotmat(unitquat_slerp(s_quat_expanded, e_quat_expanded, t))
        trans_rs = (1 - t)[..., None] * transl_start + t[..., None] * transl_end
        cam_rays_rs = (torch.bmm(rot_rs, world_points[valid, :, None]) + trans_rs[..., None]).squeeze(-1)
        image_points_rs = camera.camera_rays_to_image_points(cam_rays_rs)

        new_mean_error_px = torch.linalg.norm(
            image_points_rs.image_points[image_points_rs.valid_flag] - image_points_rs_prev[image_points_rs.valid_flag],
            dim=1,
        ).mean()

        if abs(mean_error_px - new_mean_error_px) <= stop_delta_mean_error_px:
            break
        if new_mean_error_px <= stop_mean_error_px:
            mean_error_px = new_mean_error_px
            break

        mean_error_px = new_mean_error_px
        image_points_rs_prev = image_points_rs.image_points.clone()

    # Build output mask same as legacy
    final_valid = valid.clone()
    final_valid[torch.argwhere(valid).squeeze()] = image_points_rs.valid_flag

    result_image_points = image_points_rs.image_points[image_points_rs.valid_flag]
    result_times = t[image_points_rs.valid_flag]

    return result_image_points, final_valid, result_times


@parameterized.parameterized_class(("device",), [(d,) for d in _get_test_devices()])
class TestNewSolverEquivalenceToLegacy(unittest.TestCase):
    """Verify that the new time-delta-based solver produces results equivalent to the
    legacy pixel-distance-based solver for camera rolling-shutter projection.

    The legacy solver converged based on mean pixel displacement between iterations.
    The new solver converges based on mean absolute change in relative frame time.
    Both should reach the same fixed-point solution since convergence in either metric
    implies convergence in the other.
    """

    device: torch.device

    def setUp(self) -> None:
        self.dtype = torch.float32
        self.camera = _make_ftheta_camera(self.device, self.dtype)
        self.height = 1080

    def _run_comparison(
        self,
        world_points: torch.Tensor,
        T_start: np.ndarray,
        T_end: np.ndarray,
        label: str,
    ) -> Dict[str, Any]:
        """Run both solvers and return comparison metrics."""

        T_start_t = to_torch(T_start, device=self.device, dtype=self.dtype)
        T_end_t = to_torch(T_end, device=self.device, dtype=self.dtype)

        # New solver (time-delta convergence)
        new_result = self.camera.world_points_to_image_points_shutter_pose(
            world_points,
            T_start,
            T_end,
            max_iterations=20,
            stop_t_delta=1e-7,
            return_valid_indices=True,
        )

        # Legacy solver (px-distance convergence)
        legacy_image_points, legacy_valid_mask, legacy_times = _legacy_camera_rolling_shutter(
            self.camera,
            world_points,
            T_start_t,
            T_end_t,
            max_iterations=20,
            stop_mean_error_px=1e-5,
            stop_delta_mean_error_px=1e-8,
        )

        # Both should produce the same valid set
        new_valid_mask = torch.zeros(world_points.shape[0], dtype=torch.bool, device=self.device)
        if new_result.valid_indices is not None and new_result.valid_indices.numel() > 0:
            new_valid_mask[new_result.valid_indices] = True

        n_new = int(new_valid_mask.sum().item())
        n_orig = int(legacy_valid_mask.sum().item())

        # Compare on the intersection of valid points
        both_valid = new_valid_mask & legacy_valid_mask
        n_both = int(both_valid.sum().item())

        metrics: Dict[str, Any] = {
            "label": label,
            "n_points": world_points.shape[0],
            "n_new_valid": n_new,
            "n_legacy_valid": n_orig,
            "n_both_valid": n_both,
            "valid_set_matches": torch.equal(new_valid_mask, legacy_valid_mask),
        }

        if n_both > 0:
            # Get image points for the shared valid set
            # For the new solver, we need to map from valid_indices to the both_valid subset
            new_indices_in_both = torch.zeros(world_points.shape[0], dtype=torch.long, device=self.device)
            if new_result.valid_indices is not None:
                for i, idx in enumerate(new_result.valid_indices):
                    new_indices_in_both[idx] = i

            # Gather image points for both_valid points
            both_valid_indices = torch.where(both_valid)[0]
            new_pts = new_result.image_points[
                torch.tensor([new_indices_in_both[idx] for idx in both_valid_indices], device=self.device)
            ]

            # For legacy: legacy_image_points is indexed by positions in legacy_valid_mask
            legacy_cumsum = torch.cumsum(legacy_valid_mask.long(), dim=0) - 1
            legacy_pts = legacy_image_points[
                torch.tensor([legacy_cumsum[idx] for idx in both_valid_indices], device=self.device)
            ]

            # Pixel distance between new and legacy
            px_diff = torch.linalg.norm(new_pts - legacy_pts, dim=1)
            metrics["max_px_diff"] = px_diff.max().item()
            metrics["mean_px_diff"] = px_diff.mean().item()
            metrics["median_px_diff"] = px_diff.median().item()

            # Time difference
            new_times = self.camera.image_points_relative_frame_times(new_pts)
            legacy_times_both = self.camera.image_points_relative_frame_times(legacy_pts)
            t_diff = (new_times - legacy_times_both).abs()
            metrics["max_t_diff"] = t_diff.max().item()
            metrics["mean_t_diff"] = t_diff.mean().item()
        else:
            metrics["max_px_diff"] = 0.0
            metrics["mean_px_diff"] = 0.0
            metrics["median_px_diff"] = 0.0
            metrics["max_t_diff"] = 0.0
            metrics["mean_t_diff"] = 0.0

        return metrics

    def test_equivalence_small_motion(self) -> None:
        """Compare solvers with small motion (5-degree rotation + 0.1m translation)."""
        torch.manual_seed(100)
        n = 100
        world_points = torch.randn(n, 3, device=self.device, dtype=self.dtype)
        world_points[:, 2] = world_points[:, 2].abs() + 5.0

        T_start = np.eye(4, dtype=np.float32)
        T_end = _make_pose(np.array([0.0, 5.0, 0.0]), np.array([0.1, 0.0, 0.0], dtype=np.float32))

        metrics = self._run_comparison(world_points, T_start, T_end, "small_motion")

        # Both solvers should agree on validity
        self.assertTrue(
            metrics["valid_set_matches"],
            f"Valid sets differ: new={metrics['n_new_valid']}, orig={metrics['n_legacy_valid']}",
        )
        # With small motion, both should converge to essentially the same point
        self.assertLess(metrics["max_px_diff"], 0.5, f"Max pixel diff {metrics['max_px_diff']:.6f} exceeds 0.5 px")
        self.assertLess(metrics["mean_px_diff"], 0.1)

    def test_equivalence_moderate_motion(self) -> None:
        """Compare solvers with moderate motion (15-degree rotation + 0.3m translation)."""
        torch.manual_seed(200)
        n = 100
        world_points = torch.randn(n, 3, device=self.device, dtype=self.dtype)
        world_points[:, 2] = world_points[:, 2].abs() + 5.0

        T_start = np.eye(4, dtype=np.float32)
        T_end = _make_pose(np.array([2.0, 15.0, 1.0]), np.array([0.3, -0.1, 0.05], dtype=np.float32))

        metrics = self._run_comparison(world_points, T_start, T_end, "moderate_motion")

        self.assertTrue(
            metrics["valid_set_matches"],
            f"Valid sets differ: new={metrics['n_new_valid']}, orig={metrics['n_legacy_valid']}",
        )
        # The two solvers use different convergence criteria and may stop at different
        # points. The key requirement is sub-pixel agreement (< 0.5 px).
        self.assertLess(metrics["max_px_diff"], 0.5, f"Max pixel diff {metrics['max_px_diff']:.6f} exceeds 0.5 px")
        self.assertLess(metrics["mean_px_diff"], 0.1)

    def test_equivalence_large_motion(self) -> None:
        """Compare solvers with large motion (30-degree rotation + 0.5m translation)."""
        torch.manual_seed(300)
        n = 100
        world_points = torch.randn(n, 3, device=self.device, dtype=self.dtype)
        world_points[:, 2] = world_points[:, 2].abs() + 8.0  # further away for stability

        T_start = np.eye(4, dtype=np.float32)
        T_end = _make_pose(np.array([5.0, 30.0, -3.0]), np.array([0.5, -0.2, 0.1], dtype=np.float32))

        metrics = self._run_comparison(world_points, T_start, T_end, "large_motion")

        self.assertTrue(
            metrics["valid_set_matches"],
            f"Valid sets differ: new={metrics['n_new_valid']}, orig={metrics['n_legacy_valid']}",
        )
        self.assertLess(metrics["max_px_diff"], 0.5, f"Max pixel diff {metrics['max_px_diff']:.6f} exceeds 0.5 px")
        self.assertLess(metrics["mean_px_diff"], 0.1)

    def test_equivalence_mixed_validity(self) -> None:
        """Compare solvers with points that have mixed start/end validity."""
        torch.manual_seed(400)
        n = 80
        world_points = torch.randn(n, 3, device=self.device, dtype=self.dtype)
        # Some very close, some far -- creates validity differences between start/end
        world_points[:40, 2] = world_points[:40, 2].abs() + 3.0
        world_points[40:60, 2] = world_points[40:60, 2].abs() + 10.0
        world_points[60:, 2] = -world_points[60:, 2].abs() - 1.0  # behind camera

        T_start = _make_pose(np.array([0.0, -10.0, 0.0]), np.array([0.0, 0.0, 0.0], dtype=np.float32))
        T_end = _make_pose(np.array([0.0, 10.0, 0.0]), np.array([0.2, 0.0, 0.0], dtype=np.float32))

        metrics = self._run_comparison(world_points, T_start, T_end, "mixed_validity")

        self.assertTrue(metrics["valid_set_matches"])
        self.assertLess(metrics["max_px_diff"], 0.5)

    def test_equivalence_pure_translation(self) -> None:
        """Compare solvers with pure translation (no rotation)."""
        torch.manual_seed(500)
        n = 50
        world_points = torch.randn(n, 3, device=self.device, dtype=self.dtype)
        world_points[:, 2] = world_points[:, 2].abs() + 5.0

        T_start = np.eye(4, dtype=np.float32)
        T_end = _make_pose(np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.3, -0.1], dtype=np.float32))

        metrics = self._run_comparison(world_points, T_start, T_end, "pure_translation")

        self.assertTrue(metrics["valid_set_matches"])
        self.assertLess(metrics["max_px_diff"], 0.5)

    def test_equivalence_pure_rotation(self) -> None:
        """Compare solvers with pure rotation (no translation)."""
        torch.manual_seed(600)
        n = 50
        world_points = torch.randn(n, 3, device=self.device, dtype=self.dtype)
        world_points[:, 2] = world_points[:, 2].abs() + 6.0

        T_start = np.eye(4, dtype=np.float32)
        T_end = _make_pose(np.array([3.0, 10.0, -2.0]), np.array([0.0, 0.0, 0.0], dtype=np.float32))

        metrics = self._run_comparison(world_points, T_start, T_end, "pure_rotation")

        self.assertTrue(metrics["valid_set_matches"])
        self.assertLess(metrics["max_px_diff"], 0.5)


# =============================================================================
# Section 4: FOV gap analysis -- points visible only at intermediate times
# =============================================================================


class TestFovGapAnalysis(unittest.TestCase):
    """Demonstrate that a point can be outside FOV at t=0 and t=1 but inside at intermediate t.

    This proves that the validity pre-filter `valid = proj_start.valid_flag | proj_end.valid_flag`
    is an approximation that can miss points visible only at intermediate rolling-shutter times.

    The geometric construction:
    - Camera with 80-deg max_angle (conical FOV)
    - Rotation of 50 degrees about y-axis between start and end poses
    - Point placed in the y-z plane at 79 degrees from z-axis
    - At t=0 and t=1, rotation pushes the point to ~80.04 deg (outside FOV)
    - At t=0.5 (identity rotation), point is at 79 deg (inside FOV)

    The key insight: when the rotation axis is perpendicular to the plane containing
    the optical axis and the point direction, the great-circle arc of the optical axis
    passes closest to the point at the midpoint. If both endpoints put the point just
    outside the FOV cone, the midpoint can put it inside.
    """

    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.camera = _make_ftheta_camera(self.device, self.dtype)
        self.max_angle = np.radians(80.0)

    def test_point_invisible_at_endpoints_visible_at_midpoint(self) -> None:
        """Prove a point can be invalid at t=0, t=1 but valid at t=0.5."""

        # 50-degree total rotation about x-axis (25 deg each side of identity)
        # Point in x-z plane at 79 degrees from z -- projects HORIZONTALLY (within image bounds)
        #
        # Geometry: R_x(a) @ [sin(theta), 0, cos(theta)] = [sin(theta), -sin(a)*cos(theta), cos(a)*cos(theta)]
        # Angle from z at t=0.5 (identity): theta = 79 deg (< 80 max_angle)
        # Angle from z at t=0/1 (R_x(+-25)): arccos(cos(25)*cos(79)) = 80.04 deg (> 80 max_angle)
        alpha_rad = np.radians(25.0)
        delta_rad = np.radians(79.0)
        distance = 10.0

        P_world = np.array([np.sin(delta_rad), 0, np.cos(delta_rad)], dtype=np.float32) * distance
        world_points = torch.from_numpy(P_world[None, :]).to(device=self.device, dtype=self.dtype)

        # Poses: rotation about x-axis by +-alpha
        R_start = Rotation.from_rotvec([alpha_rad, 0, 0]).as_matrix().astype(np.float32)
        R_end = Rotation.from_rotvec([-alpha_rad, 0, 0]).as_matrix().astype(np.float32)
        T_start = torch.eye(4, device=self.device, dtype=self.dtype)
        T_end = torch.eye(4, device=self.device, dtype=self.dtype)
        T_start[:3, :3] = torch.from_numpy(R_start)
        T_end[:3, :3] = torch.from_numpy(R_end)

        # Project from start and end poses
        sp_start = (T_start[:3, :3] @ world_points.T + T_start[:3, 3, None]).T
        sp_end = (T_end[:3, :3] @ world_points.T + T_end[:3, 3, None]).T

        proj_start = self.camera.camera_rays_to_image_points(sp_start)
        proj_end = self.camera.camera_rays_to_image_points(sp_end)

        # Current validity check: OR of start and end
        valid_current = proj_start.valid_flag | proj_end.valid_flag

        # Assert: point is INVALID at both endpoints
        self.assertFalse(proj_start.valid_flag[0].item(), "Point should be outside FOV at t=0")
        self.assertFalse(proj_end.valid_flag[0].item(), "Point should be outside FOV at t=1")
        self.assertFalse(valid_current[0].item(), "OR of endpoints should be False")

        # Now check at t=0.5 -- the midpoint where the point IS visible
        quat_start = rotmat_to_unitquat(T_start[None, :3, :3])
        quat_end = rotmat_to_unitquat(T_end[None, :3, :3])
        t_mid = torch.tensor([0.5], device=self.device, dtype=self.dtype)
        rot_mid = unitquat_to_rotmat(unitquat_slerp(quat_start, quat_end, t_mid))
        trans_mid = 0.5 * T_start[:3, 3] + 0.5 * T_end[:3, 3]

        sp_mid = (rot_mid @ world_points[:, :, None] + trans_mid[None, :, None]).squeeze(-1)
        proj_mid = self.camera.camera_rays_to_image_points(sp_mid)

        # Assert: point IS valid at midpoint
        self.assertTrue(proj_mid.valid_flag[0].item(), "Point should be inside FOV at t=0.5")

        # Verify the solver misses this point (uses public API)
        result = self.camera.world_points_to_image_points_shutter_pose(
            world_points,
            T_start.numpy(),
            T_end.numpy(),
            max_iterations=10,
            stop_t_delta=1e-4,
            return_valid_indices=True,
        )

        # The solver excludes this point due to the validity pre-filter
        assert result.valid_indices is not None
        self.assertEqual(result.valid_indices.shape[0], 0, "Solver misses this point (expected)")
        self.assertEqual(result.image_points.shape[0], 0, "No points projected (expected)")

    def test_practical_impact_small_rotation(self) -> None:
        """Show that for typical AV rotations (1-5 deg), the gap is sub-pixel."""
        # For a 5-degree rotation, find the critical point angle
        alpha_rad = np.radians(2.5)  # 5 deg total
        # The critical angle gamma where cos(gamma) = cos(max_angle) / cos(alpha)
        cos_gamma = np.cos(self.max_angle) / np.cos(alpha_rad)

        if cos_gamma > 1.0:
            # No gap possible
            self.skipTest("No gap possible for this rotation")

        gamma = np.arccos(cos_gamma)
        margin_rad = self.max_angle - gamma

        # The gap only affects points within margin_deg of the FOV boundary
        # For 500 px/rad focal length, this is:
        margin_px = margin_rad * 500.0

        # Assert: for 5-degree rotation, the gap is sub-pixel (< 0.1 pixels)
        self.assertLess(margin_px, 0.1, f"For 5 deg rotation, gap should be sub-pixel but got {margin_px:.3f} px")

    def test_minimum_rotation_for_significant_gap(self) -> None:
        """Determine the minimum rotation for the gap to affect >= 1 pixel."""
        # For a gap of 1 pixel at 500 px/rad focal length:
        # margin = 1/500 rad = 0.00200 rad
        # cos(max_angle - margin) * cos(phi) = cos(max_angle)
        # cos(phi) = cos(max_angle) / cos(max_angle - margin)

        target_margin_rad = 1.0 / 500.0  # 1 pixel
        cos_phi = np.cos(self.max_angle) / np.cos(self.max_angle - target_margin_rad)
        phi = np.arccos(np.clip(cos_phi, -1, 1))
        total_rotation_deg = 2 * np.degrees(phi)

        # Assert: need > 15 degrees of rotation for a 1-pixel gap
        self.assertGreater(total_rotation_deg, 15.0, "Should need substantial rotation for 1-pixel gap")
