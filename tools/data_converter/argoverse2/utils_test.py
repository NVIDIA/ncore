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

"""Data-free unit tests for the Argoverse 2 VLP-32C lidar-model derivation.

These run in CI (no external dataset needed): a synthetic spinning-lidar sweep is
generated with a known firing geometry, the model is derived from it, and the
reconstruction accuracy is asserted -- the same property the eval tool measures.
This guards the firing-pattern reconstruction (and in particular the
more-than-one-revolution column wrap) against regressions.
"""

from __future__ import annotations

import tempfile
import unittest

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

from upath import UPath

from ncore.impl.data.types import IdealPinholeCameraModelParameters, ShutterType
from ncore.impl.sensors.lidar import StructuredLidarModel
from tools.data_converter.argoverse2.utils import (
    VLP32C_N_BEAMS,
    VLP32C_SCAN_DURATION_US,
    build_vlp32c_model,
    derive_vlp32c_geometry,
    read_intrinsics,
    reconstruct_model_elements,
)


# A VLP-32C-like, deliberately non-uniform elevation table (degrees), and a
# laser-number -> physical order that is NOT elevation-sorted, so the test
# exercises the laser->row recovery.
_TRUE_ELEVATIONS_DEG = np.linspace(15.0, -25.0, VLP32C_N_BEAMS)
_LASER_PERMUTATION = np.array(
    [(i * 7) % VLP32C_N_BEAMS for i in range(VLP32C_N_BEAMS)], dtype=np.int64
)  # laser_number -> index into the (elevation-sorted) beam list


def _synthesize_sweep(
    spinning_direction: str,
    revolutions: float,
    column_period_ns: float = 55296.0,
    seed: int = 0,
    start_az_rad: float | None = None,
    az_rate_factor: float = 1.0,
):
    """Build a synthetic decompensated VLP-32C sweep with a known geometry.

    Returns (xyz_decompensated [N,3], laser_number_in_unit [N], offset_ns [N],
    true_elevations_rad [32] in laser order, true_row_offsets_rad [32] in laser order).
    """
    rng = np.random.default_rng(seed)
    sign = -1.0 if spinning_direction == "cw" else 1.0

    # Per-laser true elevation (in laser-number order) and a per-laser azimuth
    # offset spanning several degrees (VLP-32C beams in a column are not co-azimuthal).
    elev_sorted_rad = np.radians(_TRUE_ELEVATIONS_DEG)  # highest..lowest
    true_elev_by_laser = np.empty(VLP32C_N_BEAMS)
    true_elev_by_laser[np.arange(VLP32C_N_BEAMS)] = elev_sorted_rad[_LASER_PERMUTATION]
    true_offset_by_laser = np.radians(np.linspace(-4.0, 4.0, VLP32C_N_BEAMS))[_LASER_PERMUTATION]

    revolution_ns = VLP32C_SCAN_DURATION_US * 1000.0
    n_cols = int(round(revolution_ns / column_period_ns))
    total_cols = int(round(n_cols * revolutions))

    laser_list = []
    offset_list = []
    xyz_list = []
    base_az = rng.uniform(-np.pi, np.pi) if start_az_rad is None else start_az_rad
    for c in range(total_cols):
        # column reference azimuth advances with the spin. az_rate_factor != 1
        # models an intra-sweep spin-rate drift relative to the offset_ns timing
        # (the azimuth advances slightly faster/slower than the nominal column rate).
        col_az = base_az + sign * 2.0 * np.pi * c * az_rate_factor / n_cols
        for laser in range(VLP32C_N_BEAMS):
            # drop ~10% of returns to mimic sparsity; keep enough far points
            if rng.random() < 0.1:
                continue
            az = col_az + true_offset_by_laser[laser]
            el = true_elev_by_laser[laser]
            rng_m = rng.uniform(25.0, 60.0)  # far-range so azimuth fitting is clean
            x = rng_m * np.cos(el) * np.cos(az)
            y = rng_m * np.cos(el) * np.sin(az)
            z = rng_m * np.sin(el)
            xyz_list.append((x, y, z))
            laser_list.append(laser)
            offset_list.append(int(round(c * column_period_ns)))

    xyz = np.array(xyz_list, dtype=np.float64)
    laser = np.array(laser_list, dtype=np.int64)
    offset = np.array(offset_list, dtype=np.int64)
    return xyz, laser, offset, true_elev_by_laser, true_offset_by_laser


class TestVlp32cModelDerivation(unittest.TestCase):
    def _check(self, spinning_direction: str, revolutions: float) -> None:
        xyz, laser, offset, _, _ = _synthesize_sweep(spinning_direction, revolutions)

        geometry = derive_vlp32c_geometry(xyz, laser, offset)
        self.assertEqual(geometry.spinning_direction, spinning_direction)
        # Columns are upsampled for sub-column alignment precision.
        self.assertGreater(geometry.resolution_factor, 1)
        self.assertEqual(geometry.n_columns, len(geometry.column_azimuths_rad))

        model = build_vlp32c_model(geometry)
        self.assertEqual(model.n_rows, VLP32C_N_BEAMS)
        # The column-azimuth ramp must stay strictly below one revolution; the model
        # constructor enforces this, so building it at all proves the >1-rev wrap held.
        self.assertLess(abs(float(model.column_azimuths_rad[-1] - model.column_azimuths_rad[0])), 2.0 * np.pi)

        elem = reconstruct_model_elements(laser, offset, geometry, xyz)
        self.assertEqual(elem.dtype, np.uint16)
        self.assertTrue(np.all(elem[:, 0] < model.n_rows))
        self.assertTrue(np.all(elem[:, 1] < model.n_columns))

        sm = StructuredLidarModel.maybe_from_parameters(model, device="cpu")
        assert sm is not None
        predicted = sm.elements_to_sensor_points(elem, np.ones(len(elem), dtype=np.float32)).cpu().numpy()
        predicted /= np.linalg.norm(predicted, axis=1, keepdims=True)
        direction = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
        cos = np.clip(np.sum(predicted * direction, axis=1), -1.0, 1.0)
        err_deg = np.degrees(np.arccos(cos))
        median_err = float(np.median(err_deg))
        self.assertLess(
            median_err,
            0.1,
            f"{spinning_direction} {revolutions}rev: reconstruction error {median_err:.4f} deg too high",
        )

    def test_cw_single_revolution(self) -> None:
        self._check("cw", revolutions=1.0)

    def test_cw_more_than_one_revolution(self) -> None:
        # The regression case: an AV2 sweep covers ~1.02 revolutions, so the column
        # ramp would exceed 2*pi unless columns are wrapped modulo one revolution.
        self._check("cw", revolutions=1.05)

    def test_ccw_more_than_one_revolution(self) -> None:
        # The two stacked units spin oppositely in their own frames; cover the ccw case.
        self._check("ccw", revolutions=1.05)

    def test_model_generalizes_across_phase_shifted_frames(self) -> None:
        """The model derived from one sweep must reconstruct other sweeps accurately.

        The sensor's spin phase at a given ``offset_ns`` drifts between sweeps, so a
        sweep other than the one the model was derived from is rigidly rotated in
        azimuth. ``reconstruct_model_elements`` must re-align per frame; without it,
        these frames are off by ~1 deg (the multi-scene failure this guards against).
        """
        # Derive the model from a reference sweep at one phase.
        ref_xyz, ref_laser, ref_offset, _, _ = _synthesize_sweep("cw", revolutions=1.05, start_az_rad=0.3)
        geometry = derive_vlp32c_geometry(ref_xyz, ref_laser, ref_offset)
        model = build_vlp32c_model(geometry)
        sm = StructuredLidarModel.maybe_from_parameters(model, device="cpu")
        assert sm is not None

        # Reconstruct several sweeps captured at different spin phases.
        for shift_deg in (-30.0, -1.0, 1.0, 30.0):
            xyz, laser, offset, _, _ = _synthesize_sweep(
                "cw", revolutions=1.05, seed=1, start_az_rad=0.3 + np.radians(shift_deg)
            )
            elem = reconstruct_model_elements(laser, offset, geometry, xyz)
            self.assertTrue(np.all(elem[:, 1] < model.n_columns))
            predicted = sm.elements_to_sensor_points(elem, np.ones(len(elem), dtype=np.float32)).cpu().numpy()
            predicted /= np.linalg.norm(predicted, axis=1, keepdims=True)
            direction = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
            cos = np.clip(np.sum(predicted * direction, axis=1), -1.0, 1.0)
            median_err = float(np.degrees(np.median(np.arccos(cos))))
            self.assertLess(
                median_err,
                0.1,
                f"phase shift {shift_deg} deg: reconstruction error {median_err:.4f} deg too high",
            )

    def test_model_handles_intra_sweep_rate_drift(self) -> None:
        """A frame whose spin rate drifts vs offset_ns is reconstructed accurately.

        On some scenes the azimuth advances slightly faster/slower than the nominal
        ``offset_ns`` column rate within a sweep. A single rigid phase shift cannot
        correct this (it leaves ~0.25 deg); the affine (phase + linear) per-frame
        alignment in ``reconstruct_model_elements`` does. This guards that fix.
        """
        ref_xyz, ref_laser, ref_offset, _, _ = _synthesize_sweep("cw", revolutions=1.05, start_az_rad=0.2)
        geometry = derive_vlp32c_geometry(ref_xyz, ref_laser, ref_offset)
        model = build_vlp32c_model(geometry)
        sm = StructuredLidarModel.maybe_from_parameters(model, device="cpu")
        assert sm is not None

        # A sweep whose azimuth advances 0.3% faster than the nominal column rate
        # (~1 deg of drift accumulated over the revolution) plus a phase offset.
        xyz, laser, offset, _, _ = _synthesize_sweep(
            "cw", revolutions=1.05, seed=2, start_az_rad=0.2 + np.radians(5.0), az_rate_factor=1.003
        )
        elem = reconstruct_model_elements(laser, offset, geometry, xyz)
        predicted = sm.elements_to_sensor_points(elem, np.ones(len(elem), dtype=np.float32)).cpu().numpy()
        predicted /= np.linalg.norm(predicted, axis=1, keepdims=True)
        direction = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
        cos = np.clip(np.sum(predicted * direction, axis=1), -1.0, 1.0)
        median_err = float(np.degrees(np.median(np.arccos(cos))))
        self.assertLess(median_err, 0.1, f"intra-sweep drift: reconstruction error {median_err:.4f} deg too high")


class TestReadIntrinsics(unittest.TestCase):
    """``read_intrinsics`` builds an ideal pinhole and preserves k1/k2/k3."""

    def _write_intrinsics(self, log_dir: Path) -> None:
        (log_dir / "calibration").mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "sensor_name": ["ring_front_center", "ring_rear_left"],
                "fx_px": [1685.0, 1683.0],
                "fy_px": [1685.5, 1683.5],
                "cx_px": [775.0, 773.0],
                "cy_px": [1023.0, 1021.0],
                "k1": [-0.27, -0.26],
                "k2": [0.11, 0.10],
                "k3": [-0.018, -0.017],
                "height_px": [2048, 1550],
                "width_px": [1550, 2048],
            }
        )
        feather.write_feather(table, str(log_dir / "calibration" / "intrinsics.feather"))

    def test_model_is_ideal_pinhole_global_shutter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            self._write_intrinsics(log_dir)
            intrinsics = read_intrinsics(UPath(log_dir))

        self.assertEqual(set(intrinsics), {"ring_front_center", "ring_rear_left"})
        cam = intrinsics["ring_front_center"]
        self.assertIsInstance(cam.model, IdealPinholeCameraModelParameters)
        # Global shutter is assumed because the released imagery is already undistorted.
        self.assertEqual(cam.model.shutter_type, ShutterType.GLOBAL)
        self.assertIsNone(cam.model.external_distortion_parameters)
        np.testing.assert_array_equal(cam.model.resolution, np.array([1550, 2048], dtype=np.uint64))
        np.testing.assert_allclose(cam.model.focal_length, np.array([1685.0, 1685.5], dtype=np.float32))
        np.testing.assert_allclose(cam.model.principal_point, np.array([775.0, 1023.0], dtype=np.float32))

    def test_original_distortion_coefficients_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            self._write_intrinsics(log_dir)
            intrinsics = read_intrinsics(UPath(log_dir))

        # The raw lens coefficients are kept verbatim (not applied to the images).
        self.assertEqual(intrinsics["ring_front_center"].original_distortion_k1k2k3, (-0.27, 0.11, -0.018))
        self.assertEqual(intrinsics["ring_rear_left"].original_distortion_k1k2k3, (-0.26, 0.10, -0.017))


if __name__ == "__main__":
    unittest.main()
