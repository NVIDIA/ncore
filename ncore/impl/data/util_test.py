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

import unittest

import numpy as np
import torch

from numpy.polynomial.polynomial import Polynomial

from ncore.impl.data.util import closest_index_sorted, compute_max_angle_with_monotonicity, relative_angle


class TestClosestIndexSorted(unittest.TestCase):
    """Test to verify functionality of closest_index_sorted"""

    def test_empty(self):
        with self.assertRaises(ValueError):
            closest_index_sorted(np.array([], dtype=np.uint64), 5)  # empty array -> raises exception

    def test_regular(self):
        def check(sorted_array, value, expected_index: int):
            assert closest_index_sorted(sorted_array, value) == expected_index

        sorted_timestamp_array = [
            1624564702900262,
            1624564703000172,
            1624564703100110,
            1624564703200048,
            1624564703299986,
            1624564703399952,
        ]

        check(sorted_timestamp_array, sorted_timestamp_array[0], 0)  # exact first
        check(sorted_timestamp_array, sorted_timestamp_array[0] - 1, 0)  # slightly smaller than first
        check(sorted_timestamp_array, sorted_timestamp_array[0] + 1, 0)  # slightly larger than first

        check(sorted_timestamp_array, sorted_timestamp_array[-1], len(sorted_timestamp_array) - 1)  # exact last
        check(
            sorted_timestamp_array, sorted_timestamp_array[-1] - 1, len(sorted_timestamp_array) - 1
        )  # slightly smaller than last
        check(
            sorted_timestamp_array, sorted_timestamp_array[-1] + 1, len(sorted_timestamp_array) - 1
        )  # slightly larger than last

        for idx in range(len(sorted_timestamp_array)):
            check(sorted_timestamp_array, sorted_timestamp_array[idx], idx)  # exact hit
            check(sorted_timestamp_array, sorted_timestamp_array[idx] - 1, idx)  # inexact hit
            check(sorted_timestamp_array, sorted_timestamp_array[idx] + 1, idx)  # inexact hit


class TestComputeMaxAngleWithMonotonicity(unittest.TestCase):
    """Tests for the generic compute_max_angle_with_monotonicity helper."""

    def test_identity_polynomial_stops_at_max_radius(self):
        """r(theta) = theta (identity) should stop at max_radius."""
        # Polynomial: r = theta -> coeffs [0, 1] (c0=0, c1=1)
        poly = np.array([0.0, 1.0])
        max_radius = 1.5
        angle = compute_max_angle_with_monotonicity(poly, max_radius)
        self.assertAlmostEqual(angle, max_radius, places=5)

    def test_cubic_with_fold_stops_at_monotonicity_limit(self):
        """r(theta) = theta - theta^3 has dr/dtheta = 0 at theta = 1/sqrt(3)."""
        # Polynomial: r = theta - theta^3 -> coeffs [0, 1, 0, -1]
        poly = np.array([0.0, 1.0, 0.0, -1.0])
        max_radius = 10.0  # large enough to not be the limiting factor
        angle = compute_max_angle_with_monotonicity(poly, max_radius)
        expected = 1.0 / np.sqrt(3.0)  # ~0.577 rad
        self.assertAlmostEqual(angle, expected, places=5)

    def test_monotone_polynomial_reaches_max_radius(self):
        """A well-behaved polynomial should stop at max_radius, not monotonicity."""
        # Polynomial: r = theta + 0.1*theta^3 (always increasing for theta > 0)
        poly = np.array([0.0, 1.0, 0.0, 0.1])
        max_radius = 1.0
        angle = compute_max_angle_with_monotonicity(poly, max_radius)
        # Verify the polynomial at the returned angle is close to max_radius
        r = Polynomial(poly)(angle)
        self.assertAlmostEqual(r, max_radius, places=4)

    def test_derivative_positive_up_to_returned_angle(self):
        """The forward polynomial derivative must be positive for all theta in [0, angle]."""
        # Use a polynomial that folds: r = theta + 0.5*theta^2 - 2*theta^3
        poly = np.array([0.0, 1.0, 0.5, -2.0])
        max_radius = 10.0
        angle = compute_max_angle_with_monotonicity(poly, max_radius)

        # Derivative: d/dtheta [c0 + c1*t + c2*t^2 + c3*t^3] = c1 + 2*c2*t + 3*c3*t^2
        d_poly = Polynomial(poly).deriv()
        # Sample many points in [0, angle] and verify derivative > 0
        thetas = np.linspace(0, angle, 100)
        for t in thetas:
            dr = d_poly(t)
            self.assertGreaterEqual(dr, 0.0, f"Derivative negative at theta={t}")


class TestRelativeAngle(unittest.TestCase):
    """Tests for relative_angle, including float32 precision robustness."""

    def test_self_reference_is_zero_float64(self) -> None:
        """The relative angle of the reference element to itself is exactly 0."""
        angles = np.array([1.0, 0.5, 0.0, -0.5], dtype=np.float64)
        rel = relative_angle(angles[0], angles, "cw")
        self.assertEqual(float(rel.relative_angle_rad[0]), 0.0)

    def test_self_reference_is_zero_float32_near_pi(self) -> None:
        """Self-reference must be 0 even for a float32 array starting near -pi.

        Regression: relative_angle used to reduce the (float32) reference
        scalar with `% 2pi` in float64 while reducing the (float32) array in
        float32. The two reductions of the same value disagreed by ~1 ULP, so
        the self-distance at element 0 wrapped to ~2*pi instead of 0. With a
        strictly-decreasing CW sweep this made np.diff(relative_angle) negative
        at index 0 and broke monotonicity checks (e.g. the structured lidar
        model constructor) for otherwise-valid azimuths.
        """
        n = 4340
        span = 2.0 * np.pi * (1.0 - 1e-4)
        # Strictly-decreasing CW sweep whose first element sits just above -pi.
        azimuths = (-np.pi + 1e-3 - np.linspace(0.0, span, n)).astype(np.float32)

        rel = relative_angle(azimuths[0], azimuths, "cw")

        self.assertEqual(float(rel.relative_angle_rad[0]), 0.0)
        # A strictly-decreasing CW sweep has strictly-increasing relative angles.
        self.assertTrue(np.all(np.diff(rel.relative_angle_rad) > 0))
        self.assertTrue(np.all(~rel.wrap_around_flag))

    def test_matches_float64_reference(self) -> None:
        """float32 and float64 inputs agree to float32 precision."""
        n = 256
        span = 2.0 * np.pi * (1.0 - 1e-3)
        az64 = -np.pi + 1e-2 - np.linspace(0.0, span, n)
        rel64 = relative_angle(az64[0], az64, "cw")
        rel32 = relative_angle(az64.astype(np.float32)[0], az64.astype(np.float32), "cw")
        np.testing.assert_allclose(rel32.relative_angle_rad.astype(np.float64), rel64.relative_angle_rad, atol=1e-5)

    def test_self_reference_is_zero_float32_scalar(self) -> None:
        """A float32 numpy scalar reference also yields 0 self-distance.

        relative_angle is called with numpy scalars (e.g. arr[-1]) as well as
        arrays; subtracting before reducing keeps the float32 dtype so the
        self-distance reduces to exactly 0.
        """
        azimuths = (-np.pi + 1e-3 - np.linspace(0.0, 2.0 * np.pi * (1.0 - 1e-4), 16)).astype(np.float32)
        # angle_rad is a single float32 scalar equal to the reference.
        rel = relative_angle(azimuths[0], azimuths[0], "cw")
        self.assertEqual(float(rel.relative_angle_rad), 0.0)

    def test_self_reference_is_zero_torch_float32_near_pi(self) -> None:
        """The dtype-agnostic reduction also holds for torch float32 tensors.

        relative_angle is generic over numpy and torch (it does not import
        torch). Subtracting the python-scalar reference before reducing keeps the
        tensor's float32 dtype, so the self-distance at the reference reduces to
        exactly 0 and a strictly-decreasing CW sweep yields strictly-increasing
        relative angles -- same as for numpy.
        """
        n = 4340
        span = 2.0 * np.pi * (1.0 - 1e-4)
        azimuths = torch.tensor(-np.pi + 1e-3 - np.linspace(0.0, span, n), dtype=torch.float32)

        rel = relative_angle(float(azimuths[0].item()), azimuths, "cw")

        self.assertEqual(float(rel.relative_angle_rad[0].item()), 0.0)
        self.assertTrue(bool((torch.diff(rel.relative_angle_rad) > 0).all()))
        self.assertTrue(bool((~rel.wrap_around_flag).all()))
