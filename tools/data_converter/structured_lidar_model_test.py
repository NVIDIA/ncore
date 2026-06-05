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

from tools.data_converter.structured_lidar_model import (
    HDL32E_ELEVATIONS_RAD,
    HDL32E_FIRING_PAIR_INTERVAL_US,
    HDL32E_N_BEAMS,
    HDL32E_N_COLUMNS,
    HDL32E_SCAN_DURATION_US,
    AlignedFrameData,
    ColumnAlignment,
    assign_model_columns,
    compute_column_alignment,
    compute_frame_timestamps,
    compute_intra_column_firing_offsets,
    compute_model_consistency,
    derive_nominal_hdl32e,
    extract_column_azimuths,
    optimize_model,
    upsample_model,
)


class TestStructuredLidarModel(unittest.TestCase):
    def setUp(self) -> None:
        self.model = derive_nominal_hdl32e()

    # --- compute_column_alignment tests ----------------------------------------

    def test_compute_column_alignment_exact_match(self) -> None:
        """Exact match: spin azimuths identical to model -> shift=0, near-zero error."""
        n_cols = 100
        step = -2 * np.pi / n_cols
        model_azimuths = np.arange(n_cols, dtype=np.float64) * step
        spin_azimuths = model_azimuths.copy()

        alignment = compute_column_alignment(spin_azimuths, model_azimuths)

        self.assertEqual(alignment.spin_column_range.start, alignment.static_column_range.start)
        self.assertLess(alignment.mean_alignment_error_rad, 1e-6)

    def test_compute_column_alignment_with_shift(self) -> None:
        """Spin is a subset of model starting 5 columns in."""
        n_cols = 100
        step = -2 * np.pi / n_cols
        model_azimuths = np.arange(n_cols, dtype=np.float64) * step
        spin_azimuths = model_azimuths[5:95].copy()

        alignment = compute_column_alignment(spin_azimuths, model_azimuths)

        self.assertEqual(alignment.static_column_range.start, 5)
        self.assertEqual(alignment.spin_column_range.start, 0)
        self.assertEqual(len(alignment.spin_column_range), len(alignment.static_column_range))
        self.assertAlmostEqual(len(alignment.spin_column_range), 90, delta=2)
        self.assertLess(alignment.mean_alignment_error_rad, 1e-6)

    def test_compute_column_alignment_fewer_spin_cols(self) -> None:
        """Model has 100 cols, spin has 95 cols offset by 3."""
        n_cols = 100
        step = -2 * np.pi / n_cols
        model_azimuths = np.arange(n_cols, dtype=np.float64) * step
        # Spin starts at column 3, has 95 columns
        spin_azimuths = (3 + np.arange(95, dtype=np.float64)) * step

        alignment = compute_column_alignment(spin_azimuths, model_azimuths)

        self.assertEqual(alignment.static_column_range.start, 3)
        self.assertEqual(alignment.spin_column_range.start, 0)
        self.assertLess(alignment.mean_alignment_error_rad, 1e-6)

    # --- extract_column_azimuths tests -----------------------------------------

    def test_extract_column_azimuths_synthetic(self) -> None:
        """Verify extraction from synthetic point cloud with known per-column azimuths."""
        n_cols = 20
        n_beams = 4
        n_points = n_cols * n_beams
        r = 30.0  # above default min_range_m=20

        expected_azimuths = np.linspace(0, np.pi, n_cols, endpoint=False)

        xyz = np.zeros((n_points, 3), dtype=np.float64)
        col_idx = np.zeros(n_points, dtype=np.int64)

        for c in range(n_cols):
            az = expected_azimuths[c]
            for b in range(n_beams):
                idx = c * n_beams + b
                xyz[idx, 0] = np.cos(az) * r
                xyz[idx, 1] = np.sin(az) * r
                xyz[idx, 2] = 0.0
                col_idx[idx] = c

        result = extract_column_azimuths(xyz, col_idx, n_cols, min_range_m=20.0, min_points_per_col=3)

        valid_mask = ~np.isnan(result)
        self.assertTrue(valid_mask.all())
        np.testing.assert_allclose(result, expected_azimuths, atol=0.001)

    # --- assign_model_columns tests --------------------------------------------

    def test_assign_model_columns_native(self) -> None:
        """Native resolution (1x): produces 1:1 mapping from alignment."""
        model = self.model
        n_spin = 1065
        spin_start = 5
        static_start = 3

        # Create spin azimuths matching model positions
        spin_col_azimuths = model.column_azimuths_rad[static_start : static_start + n_spin].astype(np.float64)

        alignment = ColumnAlignment(
            spin_column_range=range(spin_start, spin_start + n_spin),
            static_column_range=range(static_start, static_start + n_spin),
            mean_alignment_error_rad=0.0,
        )

        result = assign_model_columns(spin_col_azimuths, model, alignment, resolution_factor=1)

        # Within the overlap, should be static_start + (c - spin_start) for each c
        expected = np.zeros(len(spin_col_azimuths), dtype=np.int64)
        for c in range(len(spin_col_azimuths)):
            if alignment.spin_column_range.start <= c < alignment.spin_column_range.stop:
                expected[c] = static_start + (c - spin_start)
            elif c < alignment.spin_column_range.start:
                expected[c] = static_start
            else:
                expected[c] = min(
                    static_start + (alignment.spin_column_range.stop - 1 - spin_start),
                    model.n_columns - 1,
                )

        np.testing.assert_array_equal(result, expected)

    def test_assign_model_columns_4x_resolution(self) -> None:
        """4x resolution: picks nearest sub-column, not just coarse position."""
        model_4x = upsample_model(self.model, 4)
        n_spin = 100
        spin_start = 0
        static_start = 10

        alignment = ColumnAlignment(
            spin_column_range=range(spin_start, n_spin),
            static_column_range=range(static_start, static_start + n_spin),
            mean_alignment_error_rad=0.0,
        )

        # Create spin azimuths that are offset by 0.5 native columns from coarse positions
        # Each native column spans 4 model columns in the upsampled model
        model_az = model_4x.column_azimuths_rad.astype(np.float64)
        spin_col_azimuths = np.zeros(n_spin, dtype=np.float64)
        for c in range(n_spin):
            coarse_idx = (static_start + c) * 4
            # Offset by ~2 sub-columns (0.5 native column)
            target_idx = min(coarse_idx + 2, model_4x.n_columns - 1)
            spin_col_azimuths[c] = model_az[target_idx]

        result = assign_model_columns(spin_col_azimuths, model_4x, alignment, resolution_factor=4)

        # Each result should be close to coarse_idx + 2 (the offset sub-column)
        for c in range(n_spin):
            coarse_idx = (static_start + c) * 4
            target_idx = min(coarse_idx + 2, model_4x.n_columns - 1)
            self.assertAlmostEqual(result[c], target_idx, delta=1)

    # --- compute_frame_timestamps tests ----------------------------------------

    def test_compute_frame_timestamps_linearity(self) -> None:
        """Timestamps are linear with column index."""
        model_col = np.array([0, 500, 1000], dtype=np.int64)
        n_model_cols = 1000
        start = 0
        end = 50000

        result = compute_frame_timestamps(model_col, n_model_cols, start, end)

        expected = np.array([0, 25000, 50000], dtype=np.uint64)
        np.testing.assert_array_equal(result, expected)

    def test_compute_frame_timestamps_fencepost(self) -> None:
        """Last column (n-1) gets timestamp < frame_end (not equal)."""
        n = 1085
        model_col = np.array([0, n - 1], dtype=np.int64)
        start = 0
        end = 50000

        result = compute_frame_timestamps(model_col, n, start, end)

        self.assertEqual(result[0], 0)
        # Column n-1 out of n: fraction = (n-1)/n < 1, so timestamp < end
        self.assertLess(result[1], end)
        expected_last = int((n - 1) / n * end)
        self.assertEqual(result[1], expected_last)

    # --- upsample_model tests --------------------------------------------------

    def test_upsample_model_doubles_columns(self) -> None:
        """Upsampling by 2 doubles column count and preserves CW monotonicity."""
        model_2x = upsample_model(self.model, 2)

        self.assertEqual(model_2x.n_columns, HDL32E_N_COLUMNS * 2)
        self.assertEqual(len(model_2x.column_azimuths_rad), HDL32E_N_COLUMNS * 2)

        # CW: strictly decreasing
        diffs = np.diff(model_2x.column_azimuths_rad.astype(np.float64))
        self.assertTrue(np.all(diffs < 0))

        # First and last azimuths close to original
        orig_az = self.model.column_azimuths_rad.astype(np.float64)
        up_az = model_2x.column_azimuths_rad.astype(np.float64)
        self.assertAlmostEqual(up_az[0], orig_az[0], places=4)
        self.assertAlmostEqual(up_az[-1], orig_az[-1], places=2)

    def test_upsample_model_identity(self) -> None:
        """Upsampling by 1 returns unchanged model."""
        result = upsample_model(self.model, 1)
        self.assertIs(result, self.model)

    def test_upsample_model_preserves_monotonicity(self) -> None:
        """Upsampling by 4 maintains strictly decreasing azimuths (CW)."""
        model_4x = upsample_model(self.model, 4)

        diffs = np.diff(model_4x.column_azimuths_rad.astype(np.float64))
        self.assertTrue(np.all(diffs < 0))

    # --- optimize_model tests --------------------------------------------------

    def test_optimize_model_reduces_residual(self) -> None:
        """Optimization reduces residual when given a systematic offset."""
        model = self.model
        n_points = model.n_columns * model.n_rows

        # Create synthetic observations: model directions + systematic per-column offset
        model_cols = np.repeat(np.arange(model.n_columns, dtype=np.int64), model.n_rows)
        model_rows = np.tile(np.arange(model.n_rows, dtype=np.int64), model.n_columns)

        # "True" azimuths = model azimuths + 0.001 rad systematic offset
        offset = 0.001
        true_azimuths = (
            model.column_azimuths_rad[model_cols].astype(np.float64)
            + model.row_azimuth_offsets_rad[model_rows].astype(np.float64)
            + offset
        )

        distances = np.full(n_points, 30.0, dtype=np.float64)

        # Initial residual
        initial_predicted = model.column_azimuths_rad[model_cols].astype(np.float64) + model.row_azimuth_offsets_rad[
            model_rows
        ].astype(np.float64)
        initial_residual = np.abs(
            np.arctan2(
                np.sin(true_azimuths - initial_predicted),
                np.cos(true_azimuths - initial_predicted),
            )
        ).mean()

        # Optimize
        optimized = optimize_model(
            model,
            frame_azimuths=[true_azimuths],
            frame_model_cols=[model_cols],
            frame_model_rows=[model_rows],
            frame_distances=[distances],
            min_range_m=10.0,
            n_iterations=1,
        )

        # Compute residual after optimization
        opt_predicted = optimized.column_azimuths_rad[model_cols].astype(
            np.float64
        ) + optimized.row_azimuth_offsets_rad[model_rows].astype(np.float64)
        opt_residual = np.abs(
            np.arctan2(
                np.sin(true_azimuths - opt_predicted),
                np.cos(true_azimuths - opt_predicted),
            )
        ).mean()

        self.assertLess(opt_residual, initial_residual)
        # Should be near zero after 1 iteration with clean data
        self.assertLess(opt_residual, 1e-5)

    # --- derive_nominal_hdl32e tests -------------------------------------------

    def test_derive_nominal_hdl32e_dimensions(self) -> None:
        """Nominal HDL-32E model has correct dimensions and direction."""
        model = self.model

        self.assertEqual(model.n_rows, 32)
        self.assertEqual(model.n_columns, 1085)
        self.assertEqual(model.spinning_direction, "cw")
        np.testing.assert_array_equal(model.row_elevations_rad, HDL32E_ELEVATIONS_RAD)
        self.assertEqual(len(model.column_azimuths_rad), 1085)
        self.assertEqual(len(model.row_azimuth_offsets_rad), 32)

    def test_derive_nominal_hdl32e_uniform_azimuths(self) -> None:
        """Column azimuths are uniformly spaced (after unwrap)."""
        model = self.model
        az_unwrapped = np.unwrap(model.column_azimuths_rad.astype(np.float64))
        diffs = np.diff(az_unwrapped)

        expected_step = -2 * np.pi / HDL32E_N_COLUMNS
        np.testing.assert_allclose(diffs, expected_step, atol=1e-5)

    # --- compute_intra_column_firing_offsets tests -----------------------------

    def test_compute_intra_column_firing_offsets_range(self) -> None:
        """Offsets have expected total angular range and correct dtype/shape."""
        offsets = compute_intra_column_firing_offsets(
            n_beams=32,
            beam_pair_interval_us=1.152,
            scan_duration_us=50000,
            spinning_direction="cw",
        )

        self.assertEqual(offsets.dtype, np.float32)
        self.assertEqual(len(offsets), 32)

        # Total range: 2 banks of 16, max beam_in_bank=15
        # time span per bank = 15 * 1.152 * 2 = 34.56 us
        # angular range = time_span * (2*pi / 50000)
        angular_rate = 2.0 * np.pi / 50000
        max_time_us = 15 * 1.152 * 2
        expected_range_rad = max_time_us * angular_rate
        actual_range = float(offsets.max() - offsets.min())

        # The range should be close to expected (mean-subtraction doesn't change range)
        self.assertAlmostEqual(actual_range, expected_range_rad, places=5)

    def test_compute_intra_column_firing_offsets_symmetry(self) -> None:
        """Offsets are mean-subtracted (sum ~0). Two banks have similar patterns."""
        offsets = compute_intra_column_firing_offsets(
            n_beams=32,
            beam_pair_interval_us=1.152,
            scan_duration_us=50000,
            spinning_direction="cw",
        )

        # Mean-subtracted: mean should be ~0
        self.assertAlmostEqual(float(offsets.mean()), 0.0, places=6)

        # Two banks (in ring order before reversal): even rings and odd rings
        # In model order (reversed ring), first 16 and last 16
        bank1 = offsets[:16].astype(np.float64)
        bank2 = offsets[16:].astype(np.float64)

        # Both banks should span similar ranges (same timing pattern)
        range1 = bank1.max() - bank1.min()
        range2 = bank2.max() - bank2.min()
        self.assertAlmostEqual(range1, range2, places=5)

    # --- compute_model_consistency tests ---------------------------------------

    def test_compute_model_consistency_perfect(self) -> None:
        """Perfect consistency: stored directions match model predictions exactly."""
        model = self.model
        n_points = 1000

        # Create random valid model element indices
        rng = np.random.default_rng(42)
        model_rows = rng.integers(0, model.n_rows, size=n_points).astype(np.uint16)
        model_cols = rng.integers(0, model.n_columns, size=n_points).astype(np.uint16)
        model_element = np.stack([model_rows, model_cols], axis=1)

        # Compute model-predicted directions
        model_az = model.column_azimuths_rad[model_cols].astype(np.float64) + model.row_azimuth_offsets_rad[
            model_rows
        ].astype(np.float64)
        model_el = model.row_elevations_rad[model_rows].astype(np.float64)
        cos_el = np.cos(model_el)
        directions = np.stack(
            [cos_el * np.cos(model_az), cos_el * np.sin(model_az), np.sin(model_el)],
            axis=1,
        ).astype(np.float32)

        distances = np.full(n_points, 30.0, dtype=np.float32)

        mean_err_all, mean_err_far, mean_az_shift = compute_model_consistency(
            directions, model_element, distances, model
        )

        self.assertAlmostEqual(mean_err_all, 0.0, places=1)  # float32 precision ~0.005 deg
        self.assertAlmostEqual(mean_err_far, 0.0, places=1)
        self.assertAlmostEqual(mean_az_shift, 0.0, places=2)

    def test_compute_model_consistency_with_offset(self) -> None:
        """Systematic azimuth offset produces expected error magnitude."""
        model = self.model
        n_points = 2000
        az_offset_rad = 0.01

        rng = np.random.default_rng(123)
        model_rows = rng.integers(0, model.n_rows, size=n_points).astype(np.uint16)
        model_cols = rng.integers(0, model.n_columns, size=n_points).astype(np.uint16)
        model_element = np.stack([model_rows, model_cols], axis=1)

        # Compute directions WITH systematic azimuth offset
        model_az = (
            model.column_azimuths_rad[model_cols].astype(np.float64)
            + model.row_azimuth_offsets_rad[model_rows].astype(np.float64)
            + az_offset_rad
        )
        model_el = model.row_elevations_rad[model_rows].astype(np.float64)
        cos_el = np.cos(model_el)
        directions = np.stack(
            [cos_el * np.cos(model_az), cos_el * np.sin(model_az), np.sin(model_el)],
            axis=1,
        ).astype(np.float32)

        distances = np.full(n_points, 30.0, dtype=np.float32)

        mean_err_all, mean_err_far, mean_az_shift = compute_model_consistency(
            directions, model_element, distances, model
        )

        expected_deg = np.degrees(az_offset_rad)  # ~0.573 deg
        # The angular error won't be exactly equal to the az offset due to elevation,
        # but the azimuth shift metric should be close
        self.assertAlmostEqual(mean_az_shift, expected_deg, places=1)
        # Total angular error should also be in the right ballpark
        self.assertAlmostEqual(mean_err_all, expected_deg, delta=0.1)


if __name__ == "__main__":
    unittest.main()
