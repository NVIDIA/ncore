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

import os
import unittest

from typing import Tuple

import numpy as np
import parameterized
import torch

from ncore.impl.data.types import (
    ConcreteCameraModelParametersUnion,
    FThetaCameraModelParameters,
    IdealPinholeCameraModelParameters,
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
    ShutterType,
)
from ncore.impl.sensors.camera import CameraModel
from ncore.impl.sensors.rectification import Rectificator


def _get_test_devices() -> Tuple[torch.device, ...]:
    if os.environ.get("NCORE_NO_GPU_TESTS"):
        return (torch.device("cpu"),)
    if torch.cuda.is_available():
        return (torch.device("cpu"), torch.device("cuda"))
    return (torch.device("cpu"),)


def _distorted_pinhole_params() -> OpenCVPinholeCameraModelParameters:
    return OpenCVPinholeCameraModelParameters(
        resolution=np.array([640, 480], dtype=np.uint64),
        shutter_type=ShutterType.GLOBAL,
        principal_point=np.array([320.0, 240.0], dtype=np.float32),
        focal_length=np.array([300.0, 300.0], dtype=np.float32),
        radial_coeffs=np.array([-0.2, 0.05, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        tangential_coeffs=np.zeros(2, dtype=np.float32),
        thin_prism_coeffs=np.zeros(4, dtype=np.float32),
    )


def _fisheye_params() -> OpenCVFisheyeCameraModelParameters:
    return OpenCVFisheyeCameraModelParameters(
        resolution=np.array([640, 480], dtype=np.uint64),
        shutter_type=ShutterType.GLOBAL,
        principal_point=np.array([320.0, 240.0], dtype=np.float32),
        focal_length=np.array([250.0, 250.0], dtype=np.float32),
        radial_coeffs=np.array([-0.03, -0.005, 0.0, 0.0], dtype=np.float32),
        max_angle=np.deg2rad(70.0),
    )


def _ftheta_params() -> FThetaCameraModelParameters:
    return FThetaCameraModelParameters(
        resolution=np.array([3848, 2168], dtype=np.uint64),
        shutter_type=ShutterType.GLOBAL,
        principal_point=np.array([1904.948486328125, 1090.5164794921875], dtype=np.float32),
        reference_poly=FThetaCameraModelParameters.PolynomialType.PIXELDIST_TO_ANGLE,
        pixeldist_to_angle_poly=np.array(
            [
                0.0,
                0.0005380856455303729,
                -1.2021251771798802e-09,
                4.5657002484267295e-12,
                -5.581118088908714e-16,
                0.0,
            ],
            dtype=np.float32,
        ),
        angle_to_pixeldist_poly=np.array(
            [0.0, 1858.59228515625, 6.894773483276367, -53.92193603515625, 14.201756477355957, 0.0],
            dtype=np.float32,
        ),
        max_angle=1.2292176485061646,
    )


# Source parameter factories exercising every distorted camera model
_SOURCE_PARAM_FACTORIES = {
    "opencv-pinhole": _distorted_pinhole_params,
    "opencv-fisheye": _fisheye_params,
    "ftheta": _ftheta_params,
}


def _model(params: ConcreteCameraModelParametersUnion, device, dtype) -> CameraModel:
    return CameraModel.from_parameters(params, device=device, dtype=dtype)


def _ideal_target_params(focal=300.0, resolution=(640, 480)) -> IdealPinholeCameraModelParameters:
    return IdealPinholeCameraModelParameters(
        resolution=np.array(resolution, dtype=np.uint64),
        shutter_type=ShutterType.GLOBAL,
        principal_point=np.array([resolution[0] / 2.0, resolution[1] / 2.0], dtype=np.float32),
        focal_length=np.array([focal, focal], dtype=np.float32),
    )


@parameterized.parameterized_class(("device",), [(d,) for d in _get_test_devices()])
class TestRectificator(unittest.TestCase):
    device: torch.device
    dtype = torch.float32

    def _build(self, source_kind: str = "opencv-pinhole", target_fov=None) -> Rectificator:
        source_params = _SOURCE_PARAM_FACTORIES[source_kind]()
        source = _model(source_params, self.device, self.dtype)
        target_params = IdealPinholeCameraModelParameters.from_source(source_params, target_fov=target_fov)
        target = _model(target_params, self.device, self.dtype)
        return Rectificator(source, target)

    def test_map_shapes(self):
        rect = self._build()
        self.assertEqual(tuple(rect.sample_map.shape), (480, 640, 2))
        self.assertEqual(tuple(rect.valid_mask.shape), (480, 640))
        self.assertEqual(rect.valid_mask.dtype, torch.bool)

    @parameterized.parameterized.expand(sorted(_SOURCE_PARAM_FACTORIES.keys()))
    def test_map_matches_explicit_projection(self, source_kind: str):
        # The sample map for a target pixel must equal the source projection of that
        # target pixel's unprojected ray - validated across all distorted source models
        rect = self._build(source_kind)
        target = rect.target
        source = rect.source

        # Probe a few central target pixels (valid for all source models)
        for u, v in [(320, 240), (280, 200), (360, 280)]:
            ray = target.image_points_to_camera_rays(np.array([[u + 0.5, v + 0.5]], dtype=np.float32))
            proj = source.camera_rays_to_image_points(ray)
            if not bool(proj.valid_flag[0]):
                continue
            expected = proj.image_points[0].cpu().numpy()
            actual = rect.sample_map[v, u].cpu().numpy()
            np.testing.assert_array_almost_equal(actual, expected, decimal=3)

    @parameterized.parameterized.expand(sorted(_SOURCE_PARAM_FACTORIES.keys()))
    def test_point_roundtrip(self, source_kind: str):
        rect = self._build(source_kind)
        pts = torch.tensor([[320.5, 240.5], [300.0, 220.0], [350.0, 260.0]], device=self.device)
        target = rect.source_points_to_target(pts)
        source = rect.target_points_to_source(target.image_points)
        valid = target.valid_flag & source.valid_flag
        np.testing.assert_array_almost_equal(
            source.image_points[valid].cpu().numpy(), pts[valid].cpu().numpy(), decimal=3
        )

    def test_apply_float_and_integer(self):
        rect = self._build()

        # float HWC
        img = torch.arange(640 * 480 * 3, dtype=torch.float32, device=self.device).reshape(480, 640, 3)
        out = rect.apply(img)
        self.assertEqual(tuple(out.shape), (480, 640, 3))
        self.assertTrue(out.is_floating_point())

        # integer (uint8) HWC preserves dtype
        imgu = (torch.rand(480, 640, 3, device=self.device) * 255).to(torch.uint8)
        outu = rect.apply(imgu)
        self.assertEqual(outu.dtype, torch.uint8)
        self.assertEqual(tuple(outu.shape), (480, 640, 3))

        # integer (int32) HWC preserves dtype
        imgi = (torch.rand(480, 640, 3, device=self.device) * 1000).to(torch.int32)
        outi = rect.apply(imgi)
        self.assertEqual(outi.dtype, torch.int32)

        # grayscale HW
        img2 = torch.rand(480, 640, device=self.device)
        out2 = rect.apply(img2)
        self.assertEqual(tuple(out2.shape), (480, 640))

        # batched NHWC
        imgn = torch.rand(2, 480, 640, 3, device=self.device)
        outn = rect.apply(imgn)
        self.assertEqual(tuple(outn.shape), (2, 480, 640, 3))

    def test_target_fov_pure_narrow_widen(self):
        # The rectification target FOV is controlled by from_source(target_fov=...):
        # pure (default), narrowed and widened relative to the natural FOV.
        source_params = _distorted_pinhole_params()
        natural = IdealPinholeCameraModelParameters.natural_fov(source_params)

        pure = self._build("opencv-pinhole")
        narrow = self._build("opencv-pinhole", target_fov=0.6 * natural)
        widen = self._build("opencv-pinhole", target_fov=1.4 * natural)

        # All produce a same-resolution map (resolution is unchanged by from_source)
        for rect in (pure, narrow, widen):
            self.assertEqual(tuple(rect.sample_map.shape), (480, 640, 2))

        # Narrowing keeps everything within the source image (fully valid); widening
        # beyond the source FOV introduces invalid (black-border) regions.
        self.assertTrue(bool(narrow.valid_mask.all()))
        self.assertFalse(bool(widen.valid_mask.all()))

        # apply() works across all variants
        img = torch.rand(480, 640, 3, device=self.device)
        for rect in (pure, narrow, widen):
            self.assertEqual(tuple(rect.apply(img).shape), (480, 640, 3))

    def test_apply_zeroes_invalid(self):
        # Build a target that is wider than the source so some target pixels fall
        # outside the source image and must be zeroed
        source = _model(_distorted_pinhole_params(), self.device, self.dtype)
        wide_target = _model(
            IdealPinholeCameraModelParameters(
                resolution=np.array([640, 480], dtype=np.uint64),
                shutter_type=ShutterType.GLOBAL,
                principal_point=np.array([320.0, 240.0], dtype=np.float32),
                focal_length=np.array([120.0, 120.0], dtype=np.float32),  # very wide -> spills out
            ),
            self.device,
            self.dtype,
        )
        rect = Rectificator(source, wide_target)
        self.assertFalse(bool(rect.valid_mask.all()))

        img = torch.ones(480, 640, 3, device=self.device)
        out = rect.apply(img)
        invalid = ~rect.valid_mask
        self.assertTrue(bool((out[invalid] == 0).all()))

    def test_identity_rectification(self):
        # Rectifying with identical source and target should reproduce the image
        target = _model(_ideal_target_params(), self.device, self.dtype)
        rect = Rectificator(target, target)
        img = torch.rand(480, 640, 3, device=self.device)
        out = rect.apply(img)
        # Interior pixels (away from borders) should match closely
        np.testing.assert_array_almost_equal(
            out[10:-10, 10:-10].cpu().numpy(), img[10:-10, 10:-10].cpu().numpy(), decimal=3
        )


if __name__ == "__main__":
    unittest.main()
