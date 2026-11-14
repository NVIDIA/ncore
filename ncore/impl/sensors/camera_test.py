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

import copy
import dataclasses
import inspect
import itertools
import math
import os
import unittest

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import cv2
import numpy as np
import parameterized
import scipy
import scipy.linalg
import torch
import torch.version

from numpy.polynomial.polynomial import Polynomial
from typing_extensions import Self

from ncore.impl.common.util import unpack_optional
from ncore.impl.data.types import (
    EXTERNAL_DISTORTION_TYPE_KEY,
    BivariateWindshieldModelParameters,
    CameraModelParameters,
    ConcreteCameraModelParametersUnion,
    ExternalDistortionParameters,
    FThetaCameraModelParameters,
    IdealPinholeCameraModelParameters,
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
    ParaxialPinholeGeometry,
    ReferencePolynomial,
    ShutterType,
    decode_camera_model_parameters,
    encode_camera_model_parameters,
)
from ncore.impl.sensors.camera import (
    BivariateWindshieldModel,
    CameraModel,
    ExternalDistortionModel,
    FThetaCameraModel,
    IdealPinholeCameraModel,
    OpenCVFisheyeCameraModel,
    OpenCVPinholeCameraModel,
    camera_model_from_parameters,
    external_distortion_model_from_parameters,
    register_camera_model,
    register_external_distortion_model,
    to_torch,
)


# =============================================================================
# GPU Test Detection
# =============================================================================
# Tests that require CUDA can be skipped via the NCORE_NO_GPU_TESTS environment variable.
# This is typically used in CI environments without GPU access.
#
# Usage:
#   bazel test --config=no-gpu ...
#
# This allows the same test suite to run on both:
# - Local execution / GPU CI runners - all tests run
# - CPU-only CI runners - GPU tests skipped via NCORE_NO_GPU_TESTS
# =============================================================================
def _get_test_devices() -> Tuple[torch.device, ...]:
    """Return the devices to test based on NCORE_NO_GPU_TESTS environment variable - will always contain CPU and conditionally GPU."""
    if os.environ.get("NCORE_NO_GPU_TESTS", "0") in ("1", "true", "True", "TRUE"):
        return (torch.device("cpu"),)
    if torch.version.cuda is None:
        # CPU-only torch build (e.g., Python 3.8 with torch+cpu)
        return (torch.device("cpu"),)
    return (torch.device("cpu"), torch.device("cuda"))


ConcreteCameraModelUnion = Union[
    FThetaCameraModel, IdealPinholeCameraModel, OpenCVPinholeCameraModel, OpenCVFisheyeCameraModel
]


def _ftheta_front_tele_params() -> FThetaCameraModelParameters:
    """Real-world narrow FTheta camera with tangential distortion (front tele 30fov)"""
    return FThetaCameraModelParameters(
        resolution=np.array([3840, 2160], dtype=np.uint64),
        shutter_type=ShutterType.ROLLING_TOP_TO_BOTTOM,
        principal_point=np.array([1897.9122, 1077.6737], dtype=np.float32),
        reference_poly=FThetaCameraModelParameters.PolynomialType.ANGLE_TO_PIXELDIST,
        angle_to_pixeldist_poly=np.array(
            [0.0, 7284.8965, -130.64005, 1932.2605, -2034.5332, -1009.2933], dtype=np.float32
        ),
        pixeldist_to_angle_poly=np.array(
            [0.0, 1.3726648e-04, 3.5653702e-10, -7.2147897e-13, 1.2644619e-16, 1.5034910e-21], dtype=np.float32
        ),
        max_angle=0.302192,
        tangential_coeffs=np.array([-3.598739, -3.6424482], dtype=np.float32),
    )


def _ftheta_cross_right_params() -> FThetaCameraModelParameters:
    """Real-world wide FTheta camera with tangential distortion (cross right 120fov)"""
    return FThetaCameraModelParameters(
        resolution=np.array([3840, 2160], dtype=np.uint64),
        shutter_type=ShutterType.ROLLING_TOP_TO_BOTTOM,
        principal_point=np.array([1919.6964, 1513.8091], dtype=np.float32),
        reference_poly=FThetaCameraModelParameters.PolynomialType.ANGLE_TO_PIXELDIST,
        angle_to_pixeldist_poly=np.array([0.0, 1870.7732, -5.2363844, -52.2381, -31.12245, 18.9092], dtype=np.float32),
        pixeldist_to_angle_poly=np.array(
            [0.0, 5.3350365e-04, 4.6422204e-09, -1.1631308e-12, 4.6077542e-15, -9.3253013e-19], dtype=np.float32
        ),
        max_angle=1.398486,
        tangential_coeffs=np.array([-0.096198946, -0.08372834], dtype=np.float32),
    )


def _ftheta_fisheye_params() -> FThetaCameraModelParameters:
    """FTheta fisheye beyond a hemisphere with explicit zero tangential coefficients

    The forward polynomial is a real-world front fisheye 200fov calibration, the backward polynomial a least-squares
    fit of its inverse on [0, 2] rad. The backward polynomial is the reference as it is monotonic over the whole
    image (unlike the forward polynomial, which has its maximum at 2.47 rad, within the image corners).

    This is a test fixture, not a real calibration: the fitted backward polynomial deviates from the exact inverse
    of the forward polynomial by up to about 1.3e-3 rad."""
    return FThetaCameraModelParameters(
        resolution=np.array([1920, 1536], dtype=np.uint64),
        shutter_type=ShutterType.ROLLING_TOP_TO_BOTTOM,
        principal_point=np.array([958.93066, 761.12213], dtype=np.float32),
        reference_poly=FThetaCameraModelParameters.PolynomialType.PIXELDIST_TO_ANGLE,
        angle_to_pixeldist_poly=np.array(
            [0.0, 506.00974, 29.93178, -43.35096, 49.284027, -15.206235], dtype=np.float32
        ),
        pixeldist_to_angle_poly=np.array(
            [0.0, 1.9737487e-03, -1.7467720e-07, 3.5128320e-10, -8.0337608e-13, 4.8121265e-16], dtype=np.float32
        ),
        max_angle=1.9,
        tangential_coeffs=np.zeros(2, dtype=np.float32),
    )


class ReferenceFThetaCamera:
    _FORWARD_POLYNOMIAL_ACCURACY = 0.01

    def __init__(self, imageSize, principalPoint, backwardPolynomial):
        assert (imageSize[0] > principalPoint[0]) and (imageSize[1] > principalPoint[1])
        assert backwardPolynomial[0] == 0
        assert 1 < len(backwardPolynomial)
        self._imageSize = np.array(imageSize)
        self._principalPoint = np.array(principalPoint)
        self._maxRadius = self._calculateMaxRadius()
        self._backwardPolynomial = backwardPolynomial
        # forward polynomial to use only as start value for newton iterations
        self._forwardPolynomial = self._determineForwardPolynomial(self._maxRadius)

    def isVisible(self, point2d):
        # potential different design decision:
        # a single pixel has a width of 1 pixel and a height of 1 pixel
        # the pixel index points to the center of the pixel
        # accordingly, the upper left corner of the upper left pixel in the image
        # has the coordinates [-0.5, -0.5]
        # potentially, also the coordinate [-0.5, -0.5] would still be
        # considered visible

        lastPixel = self._imageSize - np.array([1, 1])
        return (0 <= point2d[0]) and (point2d[0] <= lastPixel[0]) and (0 <= point2d[1]) and (point2d[1] <= lastPixel[1])

    def setBackwardPolynomial(self, backwardPolynomial):
        self._backwardPolynomial = backwardPolynomial

    def rays2imagePointsIfVisible(self, point3d):
        """map a 3d ray to the visible part of the image. return [] if the mapping
        is not within the image boundaries.
        """
        imagePoints2d = self.rays2imagePoints(point3d)
        if 0 < len(imagePoints2d) and self.isVisible(imagePoints2d):
            return imagePoints2d
        else:
            return []

    def rays2imagePoints(self, points3d):
        # project to unit sphere
        rays3d = np.array(points3d, dtype=float).T
        rays3d_norm = np.linalg.norm(rays3d, axis=0)
        rays3d /= rays3d_norm

        # project ray to equatorial plane and rescale radius according to
        # camera model
        directions2d = rays3d[0:2]
        # ensure directions2d_norm to be an array to allow for array masks
        directions2d_norm = np.array(np.linalg.norm(directions2d, axis=0))

        # compute spherical coordinates polar angle
        polars = np.arctan2(directions2d_norm, rays3d[2])

        # apply lens distortion
        radii = self._angles2radiiNewton(polars)

        directions2d_norm[directions2d_norm < np.finfo(float).eps] = 1.0
        offsets2d = directions2d * (radii / directions2d_norm)

        # add principal point. for rays with vanishing polar angle round to principal point
        polar_mask = np.broadcast_to(np.finfo(float).eps < polars, offsets2d.shape).T
        offsets2d = offsets2d.T
        imagePoints2d = np.full_like(offsets2d, self._principalPoint)
        imagePoints2d[polar_mask] += offsets2d[polar_mask]

        return imagePoints2d

    def imagePoints2rays(self, imagePoints2d):
        offsets2d = np.array(imagePoints2d) - self._principalPoint
        return self._offsets2rays(offsets2d)

    def _offsets2rays(self, offset2d):
        offset = np.array(offset2d)
        radius = np.linalg.norm(offset, axis=offset.ndim - 1, keepdims=True)
        theta = self._radius2angle(radius)
        s, c = np.sin(theta), np.cos(theta)
        radius[radius < np.finfo(float).eps] = 1.0
        ray = np.append(offset * s / radius, c, axis=offset.ndim - 1)
        return ray

    def _determineForwardPolynomial(self, maxRadius):
        linearSystemMatrix, linearSystemVector = self._getForwardPolynomialLinearSystem(maxRadius)
        coefficients = _solveLinearEquation(linearSystemMatrix, linearSystemVector)
        return np.concatenate(([0.0], coefficients))

    def _getForwardPolynomialLinearSystem(self, maxRadius):
        samplesRadius = np.array(range(1, int(np.ceil(maxRadius))))
        samplesAngle = np.array([self._radius2angle(r) for r in samplesRadius])
        transposedSystemMatrix = [samplesAngle**p for p in range(1, len(self._backwardPolynomial))]
        return np.transpose(transposedSystemMatrix), samplesRadius

    def _calculateMaxRadius(self):
        corners = np.array([[0, 0], [self._imageSize[0] - 1, 0], [0, self._imageSize[1] - 1], self._imageSize - [1, 1]])
        radiusAtCorners = [np.linalg.norm(corner - self._principalPoint) for corner in corners]
        return np.max(np.array(radiusAtCorners))

    def _radius2angle(self, radius):
        theta = np.zeros_like(radius)
        for c in reversed(self._backwardPolynomial):
            theta = c + radius * theta
        return theta

    def _dradius2angle(self, radius):
        """d/dr _radius2angle(r)"""
        theta = np.zeros_like(radius)
        dpolynomial = [i * c for i, c in enumerate(self._backwardPolynomial)]
        for c in reversed(dpolynomial[1:]):
            theta = c + radius * theta
        return theta

    def _angles2radiiNewton(self, thetas):
        # allows for scalars and vectors as arguments

        # currently, 6 iterations are the minimum to used this function
        # for any minimization based on numerical derivatives.
        MAX_ITERATIONS = 6
        THRESHOLD_RESIDUAL = np.finfo(float).eps * 100

        radii = np.array(self._angle2radiusApproximation(thetas))

        residuals = self._radius2angle(radii) - thetas

        iterCount = 0
        notConvergedMask = np.abs(residuals) > THRESHOLD_RESIDUAL
        while iterCount < MAX_ITERATIONS and np.any(notConvergedMask):
            derivatives = self._dradius2angle(radii)

            radii[notConvergedMask] -= residuals[notConvergedMask] / derivatives[notConvergedMask]

            residuals = self._radius2angle(radii) - thetas

            notConvergedMask = np.abs(residuals) > THRESHOLD_RESIDUAL
            iterCount += 1

        radii[notConvergedMask] = None

        return radii

    def _angle2radiusApproximation(self, theta):
        radius = np.zeros_like(theta)
        for c in reversed(self._forwardPolynomial):
            radius = c + theta * radius
        return radius


class CudaCheck(unittest.TestCase):
    @unittest.skipIf(len(_get_test_devices()) == 1, "GPU tests disabled via NCORE_NO_GPU_TESTS")
    def test_cuda_available(self):
        """
        Some camera tests explicitly check cuda-based computations
        (while internally falling back to CPU if cuda is not available).

        This test asserts that a cuda device is actually available to torch if not
        skipped via NCORE_NO_GPU_TESTS.
        """

        self.assertTrue(torch.cuda.is_available())


class CommonTestCase(unittest.TestCase):
    def _compareVector(self, a, b):
        self.assertEqual(len(a), len(b))
        self.assertIsNone(np.testing.assert_array_almost_equal(a, b))


# NOTE: Uses _get_test_devices() to skip GPU tests when NCORE_NO_GPU_TESTS is set
@parameterized.parameterized_class(
    ("device", "dtype"), itertools.product(_get_test_devices(), (torch.float32, torch.float64))
)
class TestReferenceFThetaCamera(CommonTestCase):
    """Parameterized test cases validating both the reference implementation and the torch-based camera model"""

    device: str
    dtype: torch.dtype

    def test_imagePoints2rays(self):
        """test backward polynomial coefficients from r**1, r**2, ... r**4"""
        for orderPolynomial in range(1, 5):
            self._test_imagePoints2rays_orderPolynomial(orderPolynomial)

    def _test_imagePoints2rays_orderPolynomial(self, orderPolynomial):
        """test backward polynomial coefficients up to r**orderPolynomial"""
        baseAngle = np.radians(45)
        resolution = 1000
        principalPoint = (resolution - 1) / 2
        backwardPolynomial = _generateBackwardPolynomial(resolution, baseAngle, orderPolynomial)
        cumulativeAngle = _computeCumulativeAngleAtImageBorder(baseAngle, orderPolynomial)

        camera = ReferenceFThetaCamera([resolution, resolution], [principalPoint, principalPoint], backwardPolynomial)

        self._executeImagePoints2RaysTestCase(camera, [[principalPoint, principalPoint]], [[0, 0, 1]])
        self._executeImagePoints2RaysTestCase(
            camera, [[resolution - 1, principalPoint]], [[np.sin(cumulativeAngle), 0, np.cos(cumulativeAngle)]]
        )
        self._executeImagePoints2RaysTestCase(
            camera, [[principalPoint, resolution - 1]], [[0, np.sin(cumulativeAngle), np.cos(cumulativeAngle)]]
        )

        self._executeImagePoints2RaysTestCase(
            camera,
            [[principalPoint, principalPoint], [resolution - 1, principalPoint], [principalPoint, resolution - 1]],
            [
                [0, 0, 1],
                [np.sin(cumulativeAngle), 0, np.cos(cumulativeAngle)],
                [0, np.sin(cumulativeAngle), np.cos(cumulativeAngle)],
            ],
        )

    def test_imagePoints2ray_shiftedPrincipalPoint(self):
        """test principal point shift for camera without radial distortions"""
        fov = np.radians(90)
        resolution = np.array([1000, 1000])
        camera = ReferenceFThetaCamera(resolution, [10, 10], [0, fov / resolution[0]])
        self._executeImagePoints2RaysTestCase(
            camera, [[10 + resolution[0] / 2, 10]], [[np.sin(fov / 2), 0, np.cos(fov / 2)]]
        )
        self._executeImagePoints2RaysTestCase(
            camera, [[10, 10 + resolution[1] / 2]], [[0, np.sin(fov / 2), np.cos(fov / 2)]]
        )

    def _executeImagePoints2RaysTestCase(self, camera, imagePoints2d, rays3dExpected):
        # Reference
        for a, e in zip(camera.imagePoints2rays(imagePoints2d), rays3dExpected):
            self._compareVector(a, e)

        # Torch-version (image points are within the field of view of the reference cameras)
        camera_rays = ftheta_from_reference(camera, self.device, self.dtype).image_points_to_camera_rays(
            np.array(imagePoints2d, ndmin=2)
        )
        self.assertTrue(bool(camera_rays.valid_flag.all()))
        for a, e in zip(np.array(camera_rays.camera_rays.cpu()), np.array(rays3dExpected, ndmin=2)):
            self._compareVector(a, e)

    def test_rays2imagePoints(self):
        for orderPolynomial in range(1, 5):
            self._test_rays2imagePoints_orderPolynomial(orderPolynomial)

    def _test_rays2imagePoints_orderPolynomial(self, orderPolynomial):
        baseAngle = np.radians(35)
        # note: accuracy of 10^-7 not reached for baseAngle= 30deg
        # baseAngle= np.radians(30)
        resolution = 1000
        principalPoint = (resolution - 1) / 2
        backwardPolynomial = _generateBackwardPolynomial(resolution, baseAngle, orderPolynomial)
        cumulativeAngle = _computeCumulativeAngleAtImageBorder(baseAngle, orderPolynomial)
        camera = ReferenceFThetaCamera([resolution, resolution], [principalPoint, principalPoint], backwardPolynomial)

        opticalAxesRay = [0, 0, 1]
        rightRay = [np.sin(cumulativeAngle), 0, np.cos(cumulativeAngle)]
        bottomRay = [0, np.sin(cumulativeAngle), np.cos(cumulativeAngle)]

        self._executeRays2ImagePointsTestCase(camera, [opticalAxesRay], [[principalPoint, principalPoint]])
        self._executeRays2ImagePointsTestCase(camera, [rightRay], [[resolution - 1, principalPoint]])
        self._executeRays2ImagePointsTestCase(camera, [bottomRay], [[principalPoint, resolution - 1]])

        rays3d = np.array([opticalAxesRay, rightRay, bottomRay])
        imagePoints2dExpected = np.array(
            [[principalPoint, principalPoint], [resolution - 1, principalPoint], [principalPoint, resolution - 1]]
        )
        self._executeRays2ImagePointsTestCase(camera, rays3d, imagePoints2dExpected)

    def _executeRays2ImagePointsTestCase(self, camera, rays3d, imagePoints2dExpected):
        # Reference
        for a, e in zip(camera.rays2imagePoints(rays3d), imagePoints2dExpected):
            self._compareVector(a, e)

        # Torch-version
        a = ftheta_from_reference(camera, self.device, self.dtype).camera_rays_to_image_points(
            np.array(rays3d, ndmin=2)
        )
        e = np.array(imagePoints2dExpected, ndmin=2)

        self._compareVector(np.array(a.image_points.cpu()), e)

    def test_imagePoints2rays_rays2imagePoints_consistency(self):
        """Tests self-consistency of both the reference camera and torch-based FTheta cameras, as well as
        cross-consistency of both cameras"""
        MAX_DEVIATION_IN_PIXEL = 0.001
        MAX_DEVIATION_RAY = 0.001
        size2d = np.array([1000, 1000])
        principalPoint = size2d / 2
        focalLengthPixel = 500.0
        backwardPolynomial = [
            0.0,
            0.4 / focalLengthPixel,
            (0.4 / focalLengthPixel) ** 2,
            (0.4 / focalLengthPixel) ** 3,
            (0.4 / focalLengthPixel) ** 4,
        ]
        camera_ref = ReferenceFThetaCamera(size2d, principalPoint, backwardPolynomial)

        camera_ftheta = ftheta_from_reference(
            camera_ref, self.device, self.dtype
        )  # instantiate a corresponding torch-based camera

        # for p in [0, px]:
        for p in range(int(principalPoint[0])):
            with self.subTest(p=p):
                expectedPoint2d = np.array([[p, p]])

                # Evaluate reference camera
                ray3d_ref = camera_ref.imagePoints2rays(expectedPoint2d)

                # Evaluate torch-camera
                camera_rays = camera_ftheta.image_points_to_camera_rays(
                    to_torch(expectedPoint2d, device=camera_ftheta.device, dtype=camera_ftheta.dtype)
                )
                ray3d = camera_rays.camera_rays

                # test that the computed rays of both cameras agree
                self.assertLessEqual(np.linalg.norm(ray3d_ref - np.array(ray3d.cpu())).item(), MAX_DEVIATION_RAY)

                with self.subTest(angle=np.degrees(np.arccos(ray3d_ref[0][2]))):
                    # Verify reference camera's result
                    actualPoint2d_ref = camera_ref.rays2imagePoints(ray3d_ref)
                    self.assertLessEqual(
                        np.linalg.norm(expectedPoint2d - actualPoint2d_ref).item(), MAX_DEVIATION_IN_PIXEL
                    )

                    # Verify torch-camera's result: image points are within the field of view, so their rays are
                    # valid, and the valid flag of the camera rays is combined in the projection (the image corner
                    # p=0 lies on the boundary of the field of view / image, so is not checked)
                    image_points = camera_ftheta.camera_rays_to_image_points(camera_rays)
                    if p > 0:
                        self.assertTrue(bool(camera_rays.valid_flag[0]))
                        self.assertTrue(bool(image_points.valid_flag[0]))
                    self.assertLessEqual(
                        np.linalg.norm(expectedPoint2d - np.array(image_points.image_points.cpu())).item(),
                        MAX_DEVIATION_IN_PIXEL,
                    )

    def test_imagePoints2rays_rays2imagePoints_consistency_linear(self):
        """Tests self-consistency of torch-based FTheta cameras given a non-trivial linear term"""
        MAX_DEVIATION_IN_PIXEL = 0.001
        size2d = np.array([1000, 1000])
        principalPoint = size2d / 2
        focalLengthPixel = 500.0
        backwardPolynomial = [
            0.0,
            0.4 / focalLengthPixel,
            (0.4 / focalLengthPixel) ** 2,
            (0.4 / focalLengthPixel) ** 3,
            (0.4 / focalLengthPixel) ** 4,
        ]

        camera_model_parameters = ftheta_parameters_from_reference(
            ReferenceFThetaCamera(size2d, principalPoint, backwardPolynomial)
        )

        # add non-identity linear term to the camera model
        camera_model_parameters = dataclasses.replace(
            camera_model_parameters, linear_cde=np.array([1.2, 0.1, 0.2], dtype=np.float32)
        )

        camera_ftheta = FThetaCameraModel(
            camera_model_parameters=camera_model_parameters, device=self.device, dtype=self.dtype
        )

        # for p in [0, px]:
        for p in range(int(principalPoint[0])):
            with self.subTest(p=p):
                expectedPoint2d = np.array([[p, p]])

                # Evaluate torch-camera
                camera_rays = camera_ftheta.image_points_to_camera_rays(
                    to_torch(expectedPoint2d, device=camera_ftheta.device, dtype=camera_ftheta.dtype)
                )
                ray3d = camera_rays.camera_rays

                with self.subTest(angle=np.degrees(np.arccos(ray3d.cpu()[0][2]))):
                    # Verify torch-camera's result: image points are within the field of view, so their rays are
                    # valid, and the valid flag of the camera rays is combined in the projection (the image corner
                    # p=0 lies on the boundary of the field of view / image, so is not checked)
                    image_points = camera_ftheta.camera_rays_to_image_points(camera_rays)
                    if p > 0:
                        self.assertTrue(bool(camera_rays.valid_flag[0]))
                        self.assertTrue(bool(image_points.valid_flag[0]))
                    self.assertLessEqual(
                        np.linalg.norm(expectedPoint2d - np.array(image_points.image_points.cpu())).item(),
                        MAX_DEVIATION_IN_PIXEL,
                    )

    def test_imagePoints2rays_rays2imagePoints_consistency_fwpoly(self):
        """Tests self-consistency of torch-based FTheta cameras using forward reference polynomials"""
        MAX_DEVIATION_IN_PIXEL = 0.001

        # A none trivial forward polynomial (using ANGLE_TO_PIXELDIST as reference) camera model
        camera_ftheta = FThetaCameraModel(
            camera_model_parameters=(
                camera_model_parameters := FThetaCameraModelParameters(
                    resolution=np.array([3848, 2168], dtype=np.uint64),
                    shutter_type=ShutterType.ROLLING_TOP_TO_BOTTOM,
                    principal_point=np.array([1909.3092041015625, 1103.27880859375], dtype=np.float32),
                    reference_poly=FThetaCameraModelParameters.PolynomialType.ANGLE_TO_PIXELDIST,
                    pixeldist_to_angle_poly=np.array(
                        [
                            0.0,
                            0.00031855489942245185,
                            -5.4367417234857385e-09,
                            4.775631279319015e-12,
                            -1.0283620548333567e-15,
                            -1.1274463994279525e-19,
                        ],
                        dtype=np.float32,
                    ),
                    angle_to_pixeldist_poly=np.array(
                        [
                            0.0,
                            3139.48583984375,
                            164.5725860595703,
                            -442.12896728515625,
                            259.5827331542969,
                            153.66644287109375,
                        ],
                        dtype=np.float32,
                    ),
                    max_angle=0.7037167544041137,
                    linear_cde=np.array(
                        [1.0000840425491333, -2.8000000384054147e-05, -7.300000288523734e-05], dtype=np.float32
                    ),
                )
            ),
            device=self.device,
            dtype=self.dtype,
        )

        # for p in [0, px]:
        for p in range(int(camera_model_parameters.principal_point[0])):
            with self.subTest(p=p):
                expectedPoint2d = np.array([[p, p]])

                # Evaluate torch-camera
                camera_rays = camera_ftheta.image_points_to_camera_rays(
                    to_torch(expectedPoint2d, device=camera_ftheta.device, dtype=camera_ftheta.dtype)
                )
                ray3d = camera_rays.camera_rays

                with self.subTest(angle=np.degrees(np.arccos(ray3d.cpu()[0][2]))):
                    # Verify torch-camera's result: image points are within the field of view, so their rays are
                    # valid, and the valid flag of the camera rays is combined in the projection (the image corner
                    # p=0 lies on the boundary of the field of view / image, so is not checked)
                    image_points = camera_ftheta.camera_rays_to_image_points(camera_rays)
                    if p > 0:
                        self.assertTrue(bool(camera_rays.valid_flag[0]))
                        self.assertTrue(bool(image_points.valid_flag[0]))
                    self.assertLessEqual(
                        np.linalg.norm(expectedPoint2d - np.array(image_points.image_points.cpu())).item(),
                        MAX_DEVIATION_IN_PIXEL,
                    )

    def test_calculateMaxRadius(self):
        size2d = np.array([10, 5])
        max2d = size2d - [1, 1]
        principalPoint2d = np.array([0, 0])
        self._test_calculateMaxRadiusTestCase(size2d, principalPoint2d, np.linalg.norm(max2d))
        principalPoint2d = np.array([1, 2])
        self._test_calculateMaxRadiusTestCase(size2d, principalPoint2d, np.linalg.norm(max2d - principalPoint2d))
        principalPoint2d = np.array([max2d[0], 0])
        self._test_calculateMaxRadiusTestCase(size2d, principalPoint2d, np.linalg.norm(max2d))
        principalPoint2d = np.array([0, max2d[1]])
        self._test_calculateMaxRadiusTestCase(size2d, principalPoint2d, np.linalg.norm(max2d))
        principalPoint2d = max2d
        self._test_calculateMaxRadiusTestCase(size2d, principalPoint2d, np.linalg.norm(max2d))

    def _test_calculateMaxRadiusTestCase(self, size2d, principalPoint2d, expectedMaxRadius):
        camera = ReferenceFThetaCamera(size2d, principalPoint2d, [0, 1])
        actualMaxRadius = camera._maxRadius
        self.assertAlmostEqual(actualMaxRadius, expectedMaxRadius)

    def test_rays2imagePoints_rays2Pixels_consistency(self):
        resolution = 1000
        principalPoint = (resolution - 1) / 2
        baseAngle = np.radians(35)
        backwardPolynomial = _generateBackwardPolynomial(resolution, baseAngle, 4)
        cumulativeAngle = _computeCumulativeAngleAtImageBorder(baseAngle, 4)
        camera = ReferenceFThetaCamera([resolution, resolution], [principalPoint, principalPoint], backwardPolynomial)

        ftheta_cam = ftheta_from_reference(camera, self.device, self.dtype)

        # Points to test
        opticalAxesRay = [0, 0, 1]
        rightRay = [np.sin(cumulativeAngle), 0, np.cos(cumulativeAngle)]
        bottomRay = [0, np.sin(cumulativeAngle), np.cos(cumulativeAngle)]

        self._test_rays2imagePoints_rays2Pixels_consistencyTestCase(ftheta_cam, opticalAxesRay)
        self._test_rays2imagePoints_rays2Pixels_consistencyTestCase(ftheta_cam, rightRay)
        self._test_rays2imagePoints_rays2Pixels_consistencyTestCase(ftheta_cam, bottomRay)

    def _test_rays2imagePoints_rays2Pixels_consistencyTestCase(self, ftheta_cam, cam_ray):
        image_points = ftheta_cam.camera_rays_to_image_points(np.array(cam_ray, ndmin=2))
        pixels = ftheta_cam.camera_rays_to_pixels(np.array(cam_ray, ndmin=2))
        self._compareVector(torch.floor(image_points.image_points.cpu()), pixels.pixels.cpu().float())

    def test_imagePoints2rays_pixels2Rays_consistency(self):
        resolution = 1000
        principalPoint = (resolution - 1) / 2
        baseAngle = np.radians(35)
        backwardPolynomial = _generateBackwardPolynomial(resolution, baseAngle, 4)
        camera = ReferenceFThetaCamera([resolution, resolution], [principalPoint, principalPoint], backwardPolynomial)

        ftheta_cam = ftheta_from_reference(camera, self.device, self.dtype)

        # Points to test
        pixel_idxs = np.random.default_rng(seed=0).choice(resolution - 1, (100, 2))

        pixel_rays = ftheta_cam.pixels_to_camera_rays(pixel_idxs.astype(np.int32)).camera_rays.cpu()
        image_point_rays = ftheta_cam.image_points_to_camera_rays(
            (pixel_idxs + 0.5).astype(np.float32)
        ).camera_rays.cpu()

        self._compareVector(pixel_rays, image_point_rays)

    def test_empty_single_more_pixels(self):
        resolution = 1000
        principalPoint = (resolution - 1) / 2
        baseAngle = np.radians(35)
        backwardPolynomial = _generateBackwardPolynomial(resolution, baseAngle, 4)
        camera = ReferenceFThetaCamera([resolution, resolution], [principalPoint, principalPoint], backwardPolynomial)

        ftheta_cam = ftheta_from_reference(camera, self.device, self.dtype)

        def check(pixel_idxs: np.ndarray) -> None:
            camera_rays = ftheta_cam.pixels_to_camera_rays(pixel_idxs.astype(np.int32))

            world_rays = ftheta_cam.pixels_to_world_rays_shutter_pose(
                pixel_idxs=pixel_idxs,
                T_sensor_world_start=np.eye(4, 4, dtype=np.float32),
                T_sensor_world_end=np.eye(4, 4, dtype=np.float32),
                start_timestamp_us=0,
                end_timestamp_us=10,
                camera_rays=camera_rays,
                return_T_sensor_worlds=True,
                return_timestamps=True,
            )

            assert len(world_rays.world_rays) == len(pixel_idxs)

            assert world_rays.T_sensor_worlds is not None
            assert len(world_rays.T_sensor_worlds) == len(pixel_idxs)

            assert world_rays.timestamps_us is not None
            assert len(world_rays.timestamps_us) == len(pixel_idxs)

        # single pixel
        check(np.random.default_rng(seed=0).choice(resolution - 1, (1, 2)))

        # no pixel
        check(np.random.default_rng(seed=0).choice(resolution - 1, (0, 2)))

        # more pixel
        check(np.random.default_rng(seed=0).choice(resolution - 1, (10, 2)))

    def test_return_all_projections(self):
        resolution = 1000
        principalPoint = (resolution - 1) / 2
        baseAngle = np.radians(5)
        backwardPolynomial = _generateBackwardPolynomial(resolution, baseAngle, 4)

        camera = ReferenceFThetaCamera([resolution, resolution], [principalPoint, principalPoint], backwardPolynomial)

        T_world_sensor_start = np.eye(4)
        T_world_sensor_end = np.eye(4)

        ftheta_cam = ftheta_from_reference(camera, self.device, self.dtype)

        # Points to test (two 2,3 are invalid)
        world_points = np.array([[0, 0, 10], [0, 0, 20], [50, 5, 10], [0, 0, -10], [0, 0, 30]])

        # Test shutter pose projection
        image_points = ftheta_cam.world_points_to_image_points_shutter_pose(
            world_points, T_world_sensor_start, T_world_sensor_end
        )
        image_points_all = ftheta_cam.world_points_to_image_points_shutter_pose(
            world_points,
            T_world_sensor_start,
            T_world_sensor_end,
            return_valid_indices=True,
            return_all_projections=True,
        )
        self._compareVector(
            image_points.image_points.cpu(), image_points_all.image_points[image_points_all.valid_indices].cpu()
        )

        image_point_invalid = ftheta_cam.world_points_to_image_points_shutter_pose(
            np.array([[0, 0, -1]]),
            T_world_sensor_start,
            T_world_sensor_end,
            return_valid_indices=True,
            return_all_projections=True,
        )
        self.assertTrue(len(unpack_optional(image_point_invalid.valid_indices)) == 0)
        self.assertTrue(len(unpack_optional(image_point_invalid.image_points)) > 0)

        # Test single pose projection
        image_points = ftheta_cam.world_points_to_image_points_static_pose(world_points, T_world_sensor_start)
        image_points_all = ftheta_cam.world_points_to_image_points_static_pose(
            world_points, T_world_sensor_start, return_valid_indices=True, return_all_projections=True
        )

        self._compareVector(
            image_points.image_points.cpu(), image_points_all.image_points[image_points_all.valid_indices].cpu()
        )

        image_point_invalid = ftheta_cam.world_points_to_image_points_static_pose(
            np.array([[0, 0, -1]]), T_world_sensor_start, return_valid_indices=True, return_all_projections=True
        )
        self.assertTrue(len(unpack_optional(image_point_invalid.valid_indices)) == 0)
        self.assertTrue(len(image_point_invalid.image_points) > 0)

    def test_inputs_and_input_types(self):
        camera = ReferenceFThetaCamera(np.array([1000, 1000]), [10, 10], [0, np.radians(90) / 1000])
        ftheta_cam = ftheta_from_reference(camera, self.device, self.dtype)

        pixel = np.array([100, 100]).reshape(1, 2)
        ray = np.array([0, 1, 0]).reshape(1, 3)

        # Test invalid inputs
        with self.assertRaisesRegex(ValueError, "image_points must be floating point values, got dtype torch.int32"):
            ftheta_cam.image_points_to_camera_rays(pixel.astype(np.int32))
        with self.assertRaisesRegex(ValueError, "Pixel indices must be integers, got dtype torch.float32"):
            ftheta_cam.pixels_to_camera_rays(pixel.astype(np.float32))
        with self.assertRaisesRegex(ValueError, "Image points must be floating point values, got dtype torch.int32"):
            ftheta_cam.image_points_to_pixels(pixel.astype(np.int32))

        with self.assertRaisesRegex(ValueError, "start_timestamp_us is required when return_timestamps=True"):
            ftheta_cam.world_points_to_image_points_shutter_pose(ray, np.eye(4), np.eye(4), return_timestamps=True)
        with self.assertRaisesRegex(ValueError, "end_timestamp_us is required when return_timestamps=True"):
            ftheta_cam.world_points_to_image_points_shutter_pose(
                ray, np.eye(4), np.eye(4), start_timestamp_us=90, return_timestamps=True
            )
        with self.assertRaisesRegex(ValueError, "End timestamp must be larger or equal to the start timestamp"):
            ftheta_cam.world_points_to_image_points_shutter_pose(
                ray, np.eye(4), np.eye(4), start_timestamp_us=100, end_timestamp_us=90, return_timestamps=True
            )
        with self.assertRaisesRegex(ValueError, "End timestamp must be larger or equal to the start timestamp"):
            ftheta_cam.world_points_to_pixels_shutter_pose(
                ray, np.eye(4), np.eye(4), start_timestamp_us=100, end_timestamp_us=90, return_timestamps=True
            )
        with self.assertRaisesRegex(ValueError, "End timestamp must be larger or equal to the start timestamp"):
            ftheta_cam.world_points_to_image_points_mean_pose(
                ray, np.eye(4), np.eye(4), start_timestamp_us=100, end_timestamp_us=90, return_timestamps=True
            )
        with self.assertRaisesRegex(ValueError, "End timestamp must be larger or equal to the start timestamp"):
            ftheta_cam.image_points_to_world_rays_mean_pose(
                pixel.astype(np.float32), np.eye(4), np.eye(4), 100, 90, return_timestamps=True
            )
        with self.assertRaisesRegex(ValueError, "End timestamp must be larger or equal to the start timestamp"):
            ftheta_cam.image_points_to_world_rays_shutter_pose(
                pixel.astype(np.float32), np.eye(4), np.eye(4), 100, 90, return_timestamps=True
            )
        with self.assertRaisesRegex(ValueError, "timestamp_us is required when return_timestamps=True"):
            ftheta_cam.world_points_to_image_points_static_pose(ray, np.eye(4), return_timestamps=True)
        with self.assertRaisesRegex(ValueError, "timestamp_us is required when return_timestamps=True"):
            ftheta_cam.image_points_to_world_rays_static_pose(
                pixel.astype(np.float32), np.eye(4), return_timestamps=True
            )

        with self.assertRaisesRegex(ValueError, r"T_world_sensor_start must have shape \(4, 4\), got shape \(3, 4\)"):
            ftheta_cam.world_points_to_image_points_shutter_pose(ray, np.eye(4)[:3], np.eye(4))
        with self.assertRaisesRegex(ValueError, r"world_points must have shape \[n, 3\], got shape \(1, 2\)"):
            ftheta_cam.world_points_to_image_points_shutter_pose(ray[:, :2], np.eye(4), np.eye(4))
        with self.assertRaisesRegex(ValueError, "max_iterations must be positive, got 0"):
            ftheta_cam.world_points_to_image_points_shutter_pose(ray, np.eye(4), np.eye(4), max_iterations=0)
        with self.assertRaisesRegex(ValueError, "max_iterations must be an int, got float"):
            ftheta_cam.world_points_to_image_points_shutter_pose(
                ray,
                np.eye(4),
                np.eye(4),
                max_iterations=cast(int, 1.0),
            )
        with self.assertRaisesRegex(ValueError, "world_points must be 2-dimensional, got shape"):
            ftheta_cam.world_points_to_image_points_static_pose(ray[0], np.eye(4))
        with self.assertRaisesRegex(ValueError, r"T_world_sensor must have shape \(4, 4\)"):
            ftheta_cam.world_points_to_image_points_static_pose(ray, np.eye(3))
        with self.assertRaisesRegex(ValueError, r"image_points must have shape \[n, 2\], got shape \(1, 3\)"):
            ftheta_cam.image_points_to_world_rays_static_pose(ray.astype(np.float32), np.eye(4))
        with self.assertRaisesRegex(ValueError, r"T_sensor_world_end must have shape \(4, 4\)"):
            ftheta_cam.image_points_to_world_rays_shutter_pose(pixel.astype(np.float32), np.eye(4), np.eye(3))

        # Test valid inputs
        ftheta_cam.image_points_to_camera_rays(pixel.astype(np.float32))
        ftheta_cam.pixels_to_camera_rays(pixel.astype(np.int32))
        ftheta_cam.world_points_to_image_points_shutter_pose(
            ray, np.eye(4), np.eye(4), start_timestamp_us=90, end_timestamp_us=100, return_timestamps=True
        )


def _solveLinearEquation(linearSystemMatrix, linearSystemVector):
    solution, _, _, _ = scipy.linalg.lstsq(linearSystemMatrix, linearSystemVector)
    return solution


def _generateBackwardPolynomial(resolution, baseAngle, orderPolynomial):
    firstToLastPixelDistance = resolution - 1
    backwardPolynomial = [0]
    for j in range(1, orderPolynomial + 1):
        backwardPolynomial.append(baseAngle / ((0.5 * firstToLastPixelDistance) ** j))
    return backwardPolynomial


def _computeCumulativeAngleAtImageBorder(baseAngle, orderPolynomial):
    return baseAngle * orderPolynomial


def ftheta_parameters_from_reference(reference_camera: ReferenceFThetaCamera) -> FThetaCameraModelParameters:
    return FThetaCameraModelParameters(
        resolution=reference_camera._imageSize.astype(np.uint64),
        shutter_type=ShutterType.ROLLING_TOP_TO_BOTTOM,
        # Subtract the principal offset to align the image coordinate system conventions
        # (offset will be added back during the initialization of the class)
        principal_point=reference_camera._principalPoint.astype(np.float32) - 0.5,
        reference_poly=FThetaCameraModelParameters.PolynomialType.PIXELDIST_TO_ANGLE,
        pixeldist_to_angle_poly=np.array(reference_camera._backwardPolynomial, dtype=np.float32),
        angle_to_pixeldist_poly=np.array(reference_camera._forwardPolynomial, dtype=np.float32),
        max_angle=reference_camera._radius2angle(reference_camera._maxRadius).astype(np.float32),
    )


def ftheta_from_reference(
    reference_camera: ReferenceFThetaCamera, device: str, dtype: torch.dtype
) -> FThetaCameraModel:
    return FThetaCameraModel(
        camera_model_parameters=ftheta_parameters_from_reference(reference_camera), device=device, dtype=dtype
    )


@parameterized.parameterized_class(
    ("device", "dtype"), itertools.product(_get_test_devices(), (torch.float32, torch.float64))
)
class TestPinholeCamera(CommonTestCase):
    device: str
    dtype: torch.dtype

    def test_imagePoints2rays_rays2imagePoints_consistency(self):
        """Tests self-consistency of torch-based Pinhole camera model"""

        # Waymo camera parameters
        cam_model_params = OpenCVPinholeCameraModelParameters(
            resolution=np.array([1920, 1280], dtype=np.uint64),
            shutter_type=ShutterType.ROLLING_RIGHT_TO_LEFT,
            principal_point=np.array([935.1248081874216, 635.052474560227], dtype=np.float32),
            focal_length=np.array(
                [
                    2059.0471439559833,
                    2059.0471439559833,
                ],
                dtype=np.float32,
            ),
            radial_coeffs=np.array(
                [
                    0.04239636827428756,
                    -0.34165672675852826,
                    0,
                    0,
                    0,
                    0,
                ],
                dtype=np.float32,
            ),
            tangential_coeffs=np.array([0.001805535524580487, -0.00005530628187935031], dtype=np.float32),
            thin_prism_coeffs=np.array([0, 0, 0, 0], dtype=np.float32),
        )

        # add additional arbitrary radial and thin-prism coeffs for this test only to guarantee code-coverage
        cam_model_params.radial_coeffs[2:] = [0.01, 0.02, -0.01, 0.02]
        cam_model_params.thin_prism_coeffs[:] = [0.01, 0.02, 0.02, 0.01]

        cam_model = OpenCVPinholeCameraModel(cam_model_params, device=self.device, dtype=self.dtype)

        MAX_DEVIATION_IN_IMAGE_COORDINATES = 0.001

        # for p in [0, px] with stepsize
        STEPSIZE = 20
        for p in range(0, int(cam_model_params.principal_point[0]), STEPSIZE):
            with self.subTest(p=p):
                # very idempotence of imagePoints2rays(rays2imagePoints([p,p]))
                expectedPoint2d = np.array([[p, p]])

                # Verify torch-camera's result (the valid flag of the camera rays is combined in the projection)
                camera_rays = cam_model.image_points_to_camera_rays(
                    to_torch(expectedPoint2d, device=cam_model.device, dtype=cam_model.dtype)
                )
                image_points = cam_model.camera_rays_to_image_points(camera_rays)

                self.assertTrue(camera_rays.valid_flag)
                self.assertTrue(image_points.valid_flag)
                self.assertLessEqual(
                    np.linalg.norm(expectedPoint2d - np.array(image_points.image_points.cpu())).item(),
                    MAX_DEVIATION_IN_IMAGE_COORDINATES,
                )

    def test_opencv_reference(self):
        """Validates the torch OpenCV-pinhole projection against the ``cv2.projectPoints`` reference"""

        np_dtype = np.float64 if self.dtype == torch.float64 else np.float32

        cam_model_params = OpenCVPinholeCameraModelParameters(
            resolution=np.array([1920, 1280], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([935.1248081874216, 635.052474560227], dtype=np.float32),
            focal_length=np.array([2059.0471439559833, 2049.0471439559833], dtype=np.float32),
            # rational radial [k1..k6], tangential [p1,p2], thin-prism [s1..s4]
            radial_coeffs=np.array([0.0424, -0.3417, 0.01, 0.02, -0.01, 0.02], dtype=np.float32),
            tangential_coeffs=np.array([0.001805535524580487, -0.00005530628187935031], dtype=np.float32),
            thin_prism_coeffs=np.array([0.01, 0.02, 0.02, 0.01], dtype=np.float32),
        )
        cam_model = OpenCVPinholeCameraModel(cam_model_params, device=self.device, dtype=self.dtype)

        # cv2 distortion vector order: [k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4]
        K = np.array(
            [
                [cam_model_params.focal_length[0], 0, cam_model_params.principal_point[0]],
                [0, cam_model_params.focal_length[1], cam_model_params.principal_point[1]],
                [0, 0, 1],
            ],
            dtype=np_dtype,
        )
        r = cam_model_params.radial_coeffs
        p = cam_model_params.tangential_coeffs
        s = cam_model_params.thin_prism_coeffs
        distortion = np.array([r[0], r[1], p[0], p[1], r[2], r[3], r[4], r[5], s[0], s[1], s[2], s[3]], dtype=np_dtype)
        rvec = np.zeros(3, dtype=np_dtype)
        tvec = np.zeros(3, dtype=np_dtype)

        MAX_DEVIATION_IN_IMAGE_COORDINATES = 0.01

        # A spread of rays in front of the camera (moderate angles to stay within the
        # rational model's valid radial range)
        for x in np.linspace(-0.3, 0.3, num=7):
            for y in np.linspace(-0.2, 0.2, num=5):
                with self.subTest(x=x, y=y):
                    ray = np.array([[x, y, 1.0]], dtype=np_dtype)

                    ours = cam_model.camera_rays_to_image_points(
                        to_torch(ray, device=cam_model.device, dtype=cam_model.dtype)
                    )

                    reference, _ = cv2.projectPoints(ray.reshape(1, 1, 3), rvec, tvec, K, distortion)
                    reference = reference.reshape(1, 2)

                    self.assertLessEqual(
                        np.linalg.norm(reference - np.array(ours.image_points.cpu())).item(),
                        MAX_DEVIATION_IN_IMAGE_COORDINATES,
                    )


class ReferenceSimplePinholeCamera:
    """Simple reference pinhole camera with symbolic evaluations (supporting k1,k2,k3,p1,p2)"""

    def __init__(self, params: OpenCVPinholeCameraModelParameters, dtype: np.dtype):
        self.params = params
        self.dtype = dtype

        assert not np.any(self.params.radial_coeffs[3:]), "only supporting non-zero k1,k2,k3"
        assert not np.any(self.params.thin_prism_coeffs), "not supporting thin-prism coeffs"

    def _distortion(self, uvN):
        """Computes the radial + tangential distortion given the camera rays"""

        # Helper variables for primary function evaluation
        u0u0 = uvN[0] * uvN[0]
        u1u1 = uvN[1] * uvN[1]
        r_2 = u0u0 + u1u1
        uv_prod = uvN[0] * uvN[1]
        a1 = 2 * uv_prod
        a2 = r_2 + 2 * u0u0
        a3 = r_2 + 2 * u1u1

        icD = 1.0 + r_2 * (
            self.params.radial_coeffs[0] + r_2 * (self.params.radial_coeffs[1] + r_2 * self.params.radial_coeffs[2])
        )

        delta_x = self.params.tangential_coeffs[0] * a1 + self.params.tangential_coeffs[1] * a2
        delta_y = self.params.tangential_coeffs[0] * a3 + self.params.tangential_coeffs[1] * a1

        uvND = uvN * icD + np.array([[delta_x, delta_y]], dtype=self.dtype)

        # Helper variables for symbolic Jacobian evaluation
        b1 = self.params.radial_coeffs[1] + self.params.radial_coeffs[2] * r_2
        b11 = 2 * (self.params.radial_coeffs[0] + b1 * r_2) + r_2 * (2 * self.params.radial_coeffs[2] * r_2 + 2 * b1)
        b2 = uvN[0] * b11
        b3 = uvN[1] * b11
        b4 = (self.params.radial_coeffs[0] + b1 * r_2) * r_2 + 1.0

        J_uvND = np.array(
            [
                [
                    2 * self.params.tangential_coeffs[0] * uvN[1]
                    + 6 * self.params.tangential_coeffs[1] * uvN[0]
                    + uvN[0] * b2
                    + b4,
                    2 * self.params.tangential_coeffs[0] * uvN[0]
                    + 2 * self.params.tangential_coeffs[1] * uvN[1]
                    + uvN[0] * b3,
                ],
                [
                    2 * self.params.tangential_coeffs[0] * uvN[0]
                    + 2 * self.params.tangential_coeffs[1] * uvN[1]
                    + uvN[1] * b2,
                    6 * self.params.tangential_coeffs[0] * uvN[1]
                    + 2 * self.params.tangential_coeffs[1] * uvN[0]
                    + uvN[1] * b3
                    + b4,
                ],
            ]
        )

        return uvND, J_uvND

    def _perspective_normalization(self, x: np.ndarray):
        uvN = np.array([x[0] / x[2], x[1] / x[2]], dtype=self.dtype)
        J_uvN = np.array([[1 / x[2], 0, -x[0] / x[2] ** 2], [0, 1 / x[2], -x[1] / x[2] ** 2]], dtype=self.dtype)

        return uvN, J_uvN

    def _perspective_projection(self, uvND: np.ndarray):
        uv = uvND * self.params.focal_length + self.params.principal_point
        J_uv = np.array([[self.params.focal_length[0], 0], [0, self.params.focal_length[1]]], dtype=self.dtype)

        return uv, J_uv

    def camera_ray_to_image_points(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Assumes ray is a valid projection / returns image point + Jacobian"""

        uvN, J_uvN = self._perspective_normalization(x)

        uvND, J_uvND = self._distortion(uvN)

        uv, J_uv = self._perspective_projection(uvND)

        return uv.squeeze(), J_uv @ J_uvND @ J_uvN  # Assemble full transformation's Jacobian according to chain-rule


@parameterized.parameterized_class(
    ("device", "dtype"), itertools.product(_get_test_devices(), (torch.float32, torch.float64))
)
class TestJacobian(CommonTestCase):
    device: str
    dtype: torch.dtype

    def test_pinhole_reference(self):
        """Tests consistency of camera model Jacobians with reference implementation"""

        # Distorted pinhole camera model with "simple" k1,k2,k3,p1,p2 parametrization only
        cam_model_params = OpenCVPinholeCameraModelParameters(
            resolution=np.array([1920, 1280], dtype=np.uint64),
            shutter_type=ShutterType.ROLLING_RIGHT_TO_LEFT,
            principal_point=np.array([935.1248081874216, 635.052474560227], dtype=np.float32),
            focal_length=np.array(
                [
                    2059.0471439559833,
                    2059.4231439559833,
                ],
                dtype=np.float32,
            ),
            radial_coeffs=np.array(
                [
                    0.04239636827428756,
                    -0.34165672675852826,
                    0.01,
                    0,
                    0,
                    0,
                ],
                dtype=np.float32,
            ),
            tangential_coeffs=np.array([0.001805535524580487, -0.00005530628187935031], dtype=np.float32),
            thin_prism_coeffs=np.array([0, 0, 0, 0], dtype=np.float32),
        )

        cam_model_ref = ReferenceSimplePinholeCamera(
            cam_model_params, cast(np.dtype, {torch.float32: np.float32, torch.float64: np.float64}[self.dtype])
        )
        cam_model = camera_model_from_parameters(cam_model_params, device=self.device, dtype=self.dtype)

        camera_rays = cam_model.image_points_to_camera_rays(torch.Tensor([[20, 40], [11, 12], [15, 20], [500, 500]]))
        self.assertTrue(bool(camera_rays.valid_flag.all()))
        rays3d = camera_rays.camera_rays  # valid rays only

        for ray3d in rays3d:
            pref, Jref = cam_model_ref.camera_ray_to_image_points(ray3d.cpu().numpy())

            proj = cam_model.camera_rays_to_image_points(ray3d.unsqueeze(1).transpose(1, 0), return_jacobians=True)

            np.testing.assert_array_almost_equal(pref, proj.image_points.detach()[0].cpu().numpy())
            np.testing.assert_array_almost_equal(
                Jref,
                unpack_optional(proj.jacobians).detach()[0].cpu().numpy(),
                decimal=6 if self.dtype == torch.float64 else 3,
            )

    def test_jacobian_consistency(self):
        """Tests consistency of camera model Jacobians with autograd results"""

        cam_models = [
            # Ideal pinhole camera parameters
            camera_model_from_parameters(
                OpenCVPinholeCameraModelParameters(
                    resolution=np.array([1920, 1280], dtype=np.uint64),
                    shutter_type=ShutterType.ROLLING_RIGHT_TO_LEFT,
                    principal_point=np.array([935.1248081874216, 635.052474560227], dtype=np.float32),
                    focal_length=np.array(
                        [
                            2059.0471439559833,
                            2059.0471439559833,
                        ],
                        dtype=np.float32,
                    ),
                    radial_coeffs=np.array(
                        [
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ],
                        dtype=np.float32,
                    ),
                    tangential_coeffs=np.array([0, 0], dtype=np.float32),
                    thin_prism_coeffs=np.array([0, 0, 0, 0], dtype=np.float32),
                ),
                device=self.device,
                dtype=self.dtype,
            ),
            # Waymo camera parameters
            camera_model_from_parameters(
                OpenCVPinholeCameraModelParameters(
                    resolution=np.array([1920, 1280], dtype=np.uint64),
                    shutter_type=ShutterType.ROLLING_RIGHT_TO_LEFT,
                    principal_point=np.array([935.1248081874216, 635.052474560227], dtype=np.float32),
                    focal_length=np.array(
                        [
                            2059.0471439559833,
                            2059.0471439559833,
                        ],
                        dtype=np.float32,
                    ),
                    radial_coeffs=np.array(
                        [
                            0.04239636827428756,
                            -0.34165672675852826,
                            0,
                            0,
                            0,
                            0,
                        ],
                        dtype=np.float32,
                    ),
                    tangential_coeffs=np.array([0.001805535524580487, -0.00005530628187935031], dtype=np.float32),
                    thin_prism_coeffs=np.array([0, 0, 0, 0], dtype=np.float32),
                ),
                device=self.device,
                dtype=self.dtype,
            ),
            # NV 120deg instance
            camera_model_from_parameters(
                FThetaCameraModelParameters(
                    resolution=np.array([3848, 2168], dtype=np.uint64),
                    shutter_type=ShutterType.ROLLING_TOP_TO_BOTTOM,
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
                ),
                device=self.device,
                dtype=self.dtype,
            ),
            # NV FTheta instances with tangential distortion
            camera_model_from_parameters(_ftheta_front_tele_params(), device=self.device, dtype=self.dtype),
            camera_model_from_parameters(_ftheta_cross_right_params(), device=self.device, dtype=self.dtype),
            # External costumer fisheye model
            camera_model_from_parameters(
                OpenCVFisheyeCameraModelParameters(
                    resolution=np.array([3840, 2160], dtype=np.uint64),
                    shutter_type=ShutterType.ROLLING_TOP_TO_BOTTOM,
                    principal_point=np.array([1928.184506, 1083.862789], dtype=np.float32),
                    focal_length=np.array(
                        [
                            1913.76478,
                            1913.99708,
                        ],
                        dtype=np.float32,
                    ),
                    radial_coeffs=np.array(
                        [
                            -0.030093122,
                            -0.005103817,
                            -0.000849622,
                            0.001079542,
                        ],
                        dtype=np.float32,
                    ),
                    max_angle=np.deg2rad(140 / 2),
                ),
                device=self.device,
            ),
        ]

        for cam_model in cam_models:

            def projection_wrapper(x):
                return cam_model.camera_rays_to_image_points(x[None, :]).image_points.squeeze()

            valid_camera_rays = cam_model.image_points_to_camera_rays(
                torch.Tensor([[20, 40], [11, 12], [15, 20], [500, 500]])
            )
            self.assertTrue(bool(valid_camera_rays.valid_flag.all()), msg=f"{type(cam_model)} returned invalid rays")
            valid_rays3d = valid_camera_rays.camera_rays  # valid rays
            self.assertLessEqual(
                (torch.linalg.norm(valid_rays3d, axis=1, keepdims=True) - torch.ones_like(valid_rays3d[:, :1]))
                .abs()
                .max()
                .item(),
                # A few ulps (sin / cos of an angle and a separately rounded unit direction are normalized only up
                # to rounding, for purely radial FTheta models as well)
                4 * torch.finfo(valid_rays3d.dtype).eps,
                msg=f"{type(cam_model)} failed to return normalized rays",
            )  # make sure all camera models return *normalized* rays
            principal_direction_rays3d = torch.Tensor([[0, 0, 1], [0, 0, 5], [0, 0, 0.1]]).to(
                valid_rays3d
            )  # rays along the principal direction
            invalid_rays3d = torch.Tensor([[1, 2, -5], [1, 2, 0], [0, 0, 0]]).to(
                valid_rays3d
            )  # some "invalid" rays (behind camera / on the center of projection plane but ouf of FOV / zero)
            rays3d = torch.cat([valid_rays3d, principal_direction_rays3d, invalid_rays3d])

            # evaluate projection with jacobians
            proj = cam_model.camera_rays_to_image_points(rays3d, return_jacobians=True)

            for i, ray3d in enumerate(rays3d):
                Jref = torch.autograd.functional.jacobian(
                    projection_wrapper, ray3d, strict=True, strategy="reverse-mode"
                )

                # Make sure API-computed Jacobian coincides with autograd result
                np.testing.assert_array_almost_equal(
                    Jref.cpu().numpy(), unpack_optional(proj.jacobians)[i].cpu().numpy()
                )

                self.assertTrue(
                    proj.valid_flag[i] if i < len(rays3d) - len(invalid_rays3d) else not proj.valid_flag[i]
                )  # First rays should be flagged as valid, others should be invalid


@parameterized.parameterized_class(
    ("device", "dtype"), itertools.product(_get_test_devices(), (torch.float32, torch.float64))
)
class TestFisheyeCamera(CommonTestCase):
    device: torch.device
    dtype: torch.dtype

    MAX_DEVIATION_IN_IMAGE_COORDINATES = 0.001
    MAX_DEVIATION_IN_RAY_COORDINATES = 0.001

    def setUp(self):
        # Make printed errors more representable numerically
        np.set_printoptions(floatmode="unique", linewidth=200, suppress=True)

        # Real-world customer camera parameters
        self.cam_model_params = OpenCVFisheyeCameraModelParameters(
            resolution=np.array([3840, 2160], dtype=np.uint64),
            shutter_type=ShutterType.ROLLING_TOP_TO_BOTTOM,
            principal_point=np.array([1928.184506, 1083.862789], dtype=np.float32),
            focal_length=np.array(
                [
                    1913.76478,
                    1913.99708,
                ],
                dtype=np.float32,
            ),
            radial_coeffs=np.array(
                [
                    -0.030093122,
                    -0.005103817,
                    -0.000849622,
                    0.001079542,
                ],
                dtype=np.float32,
            ),
            max_angle=np.deg2rad(140 / 2),
        )

        self.cam_model = OpenCVFisheyeCameraModel(self.cam_model_params, device=self.device, dtype=self.dtype)

        if self.dtype == torch.float64:
            self.np_dtype = np.float64
        elif self.dtype == torch.float32:
            self.np_dtype = np.float32

    def test_special_cases(self):
        """Validate a few special cases"""

        # make sure the principal point gets unprojected to the principal axis, as a valid ray
        camera_rays = self.cam_model.image_points_to_camera_rays(
            torch.from_numpy(self.cam_model_params.principal_point).reshape(1, 2)
        )
        ray3d = camera_rays.camera_rays
        self.assertTrue(bool(camera_rays.valid_flag[0]))

        self.assertLessEqual(
            np.linalg.norm(ray3d.cpu().numpy() - np.array([0, 0, 1])).item(), self.MAX_DEVIATION_IN_RAY_COORDINATES
        )

    def test_opencv_reference(self):
        """Tests self-consistency of torch-based fisheye camera model, as well as consistency with OpenCV reference implementation"""

        def ray_to_image_point_opencv(
            ray: Union[np.ndarray, List[float]], cam_model_params: OpenCVFisheyeCameraModelParameters
        ):
            """Evaluate OpenCV's 'fisheye' model for a single ray-to-image projection"""

            ray = np.array(ray, dtype=self.np_dtype)
            assert ray.size == 3

            # Parameterizing identity extrinsics
            rvec = np.array([0.0, 0.0, 0.0], dtype=self.np_dtype)
            tvec = np.array([0.0, 0.0, 0.0], dtype=self.np_dtype)

            # Camera matrix
            K = np.array(
                [
                    [cam_model_params.focal_length[0], 0, cam_model_params.principal_point[0]],
                    [0, cam_model_params.focal_length[1], cam_model_params.principal_point[1]],
                    [0, 0, 1],
                ],
                dtype=self.np_dtype,
            )
            d = cam_model_params.radial_coeffs.astype(self.np_dtype)  # distortion parameters [k1, k2, k3, k4]
            alpha = 0.0  # skew factor

            p, _ = cv2.fisheye.projectPoints(
                ray.astype(self.np_dtype).reshape(1, 1, 3), rvec, tvec, K, d, None, alpha
            )  # second returned value are Jacobians, can't be disabled

            return p.reshape(1, 2)

        # for p in [0, px]
        for i, p in enumerate(np.linspace(0.0, self.cam_model_params.principal_point[0], num=50, endpoint=True)):
            with self.subTest(p=p):
                # 1. very idempotence imagePoints2rays(rays2imagePoints([p,p])) torch-camera's result
                expectedPoint2d = np.array([[p, p]])

                camera_rays = self.cam_model.image_points_to_camera_rays(
                    to_torch(expectedPoint2d, device=self.cam_model.device, dtype=self.dtype)
                )
                ray3d = camera_rays.camera_rays
                image_points = self.cam_model.camera_rays_to_image_points(camera_rays)

                # all sampled points are within the field of view (max_angle)
                self.assertTrue(bool(camera_rays.valid_flag[0]))
                if i > 0:
                    # avoid 'valid' prevision issues if points get re-projected right onto each side of the image boundary for p=[0,0]
                    self.assertTrue(image_points.valid_flag)
                self.assertLessEqual(
                    np.linalg.norm(expectedPoint2d - np.array(image_points.image_points.cpu())).item(),
                    self.MAX_DEVIATION_IN_IMAGE_COORDINATES,
                )

                # 2. verify consistency with OpenCV reference (one-way is sufficient)
                image_point_opencv = ray_to_image_point_opencv(ray3d.cpu().numpy(), self.cam_model_params)
                self.assertLessEqual(
                    np.linalg.norm(image_point_opencv - np.array(image_points.image_points.cpu())).item(),
                    self.MAX_DEVIATION_IN_IMAGE_COORDINATES,
                )


@parameterized.parameterized_class(
    ("device", "dtype"), itertools.product(_get_test_devices(), (torch.float32, torch.float64))
)
class TestFisheyeCameraInverse(CommonTestCase):
    """Accuracy of the OpenCV fisheye inverse (image points to camera rays) for calibrations where
    a plain Newton iteration from the equidistant starting point fails"""

    device: torch.device
    dtype: torch.dtype

    def _model(
        self,
        radial_coeffs: List[float],
        resolution: Tuple[int, int],
        principal_point: Tuple[float, float],
        focal_length: float,
        max_angle: float,
    ) -> OpenCVFisheyeCameraModel:
        return OpenCVFisheyeCameraModel(
            OpenCVFisheyeCameraModelParameters(
                resolution=np.array(resolution, dtype=np.uint64),
                shutter_type=ShutterType.GLOBAL,
                principal_point=np.array(principal_point, dtype=np.float32),
                focal_length=np.array([focal_length, focal_length], dtype=np.float32),
                radial_coeffs=np.array(radial_coeffs, dtype=np.float32),
                max_angle=max_angle,
            ),
            device=self.device,
            dtype=self.dtype,
        )

    def _cubic_model(self) -> OpenCVFisheyeCameraModel:
        # r(theta) = theta - theta^3/3 has its only stationary point at theta = 1, where r = 2/3, and the
        # closed-form inverse theta = 2*sin(asin(3r/2)/3) on [0, 1]. With max_angle = 2 the equidistant
        # starting point theta_0 = 2r lands exactly on the stationary point for r = 0.5.
        return self._model([-1.0 / 3.0, 0.0, 0.0, 0.0], (1280, 576), (640.0, 288.0), 640.0, 2.0)

    @staticmethod
    def _cubic_inverse(k1: float, radii: np.ndarray) -> np.ndarray:
        # Closed-form inverse of theta + k1*theta^3 (k1 < 0) on its increasing branch
        a = np.sqrt(-3.0 * k1)
        return 2.0 / a * np.sin(np.arcsin(1.5 * radii * a) / 3.0)

    @staticmethod
    def _monotonic_inverse(radial_coeffs: List[float], radii: np.ndarray, theta_max: float) -> np.ndarray:
        # float64 bisection reference on [0, theta_max], valid if r(theta) is increasing on that interval
        k = np.array(radial_coeffs, dtype=np.float32).astype(np.float64)
        poly = Polynomial([0.0, 1.0, 0.0, k[0], 0.0, k[1], 0.0, k[2], 0.0, k[3]])
        lower, upper = np.zeros_like(radii), np.full_like(radii, theta_max)
        for _ in range(100):
            middle = 0.5 * (lower + upper)
            below = poly(middle) < radii
            lower, upper = np.where(below, middle, lower), np.where(below, upper, middle)
        return 0.5 * (lower + upper)

    def _rays_along_x(self, cam_model: OpenCVFisheyeCameraModel, radii: np.ndarray) -> torch.Tensor:
        principal_point = cam_model.principal_point.cpu().numpy().astype(np.float64)
        focal_length = cam_model.focal_length.cpu().numpy().astype(np.float64)
        image_points = np.stack(
            [principal_point[0] + radii * focal_length[0], np.full_like(radii, principal_point[1])], 1
        )
        return cam_model.image_points_to_camera_rays(
            to_torch(image_points, device=self.device, dtype=self.dtype)
        ).camera_rays

    def _assert_angles(self, rays: torch.Tensor, expected_thetas: np.ndarray, atol: float):
        rays_np = rays.detach().cpu().numpy().astype(np.float64)
        self.assertTrue(np.isfinite(rays_np).all(), f"non-finite rays {rays_np}")
        thetas = np.arctan2(np.linalg.norm(rays_np[:, :2], axis=1), rays_np[:, 2])
        np.testing.assert_allclose(thetas, expected_thetas, rtol=0.0, atol=atol)

    @property
    def _angle_atol(self) -> float:
        # Accuracy of converged Newton iterations at the floating-point resolution of the polynomial evaluation
        return 3e-7 if self.dtype == torch.float32 else 1e-14

    def test_inverse_at_stationary_starting_point(self):
        """Recovers the analytic inverse when Newton's starting point has a vanishing derivative"""

        cam_model = self._cubic_model()
        radii = np.array([0.1, 0.3, 0.49999, 0.5, 0.50001, 0.6, 0.66])
        k1 = float(np.float32(-1.0 / 3.0))
        self._assert_angles(self._rays_along_x(cam_model, radii), self._cubic_inverse(k1, radii), self._angle_atol)

    def test_inverse_converges_over_full_image(self):
        """Every pixel of a calibration whose equidistant starting points are far off converges with the default
        iteration settings"""

        radial_coeffs = [0.1, 0.2, 0.3, 0.4]
        cam_model = self._model(radial_coeffs, (960, 540), (960.0, 540.0), 1000.0, float(np.deg2rad(70.0)))
        pixels = np.stack(np.meshgrid(np.arange(960), np.arange(540), indexing="xy"), -1).reshape(-1, 2) + 0.5
        rays = cam_model.image_points_to_camera_rays(to_torch(pixels, device=self.device, dtype=self.dtype)).camera_rays
        radii = np.linalg.norm((pixels - np.array([960.0, 540.0])) / 1000.0, axis=1)
        # r(theta) is increasing everywhere for positive coefficients
        self._assert_angles(rays, self._monotonic_inverse(radial_coeffs, radii, np.pi), self._angle_atol)

    def test_inverse_extrapolates_beyond_max_angle(self):
        """Image points beyond max_angle but within the increasing part of the polynomial keep their exact
        (extrapolated) inverse, they are not clamped to max_angle"""

        radial_coeffs = [-0.030093122, -0.005103817, -0.000849622, 0.001079542]
        cam_model = self._model(radial_coeffs, (3840, 2160), (1920.0, 1080.0), 1913.8, float(np.deg2rad(40.0)))
        radii = np.array([0.2, 0.6, 0.9, 1.2, 1.5])
        expected = self._monotonic_inverse(radial_coeffs, radii, np.pi)
        self.assertGreater(expected[-1], np.deg2rad(40.0))
        self._assert_angles(self._rays_along_x(cam_model, radii), expected, self._angle_atol)

    def test_inverse_beyond_stationary_point_is_boundary_ray(self):
        """Image points beyond the maximum of the polynomial have no inverse and map to the finite
        boundary ray at the stationary angle"""

        cam_model = self._cubic_model()
        radii = np.array([0.67, 0.7, 0.9, 1.5, 3.0])
        self._assert_angles(self._rays_along_x(cam_model, radii), np.ones_like(radii), 1e-6)

    def test_inverse_gradients_follow_implicit_derivative(self):
        """Gradients with respect to image points are the derivatives of the exact inverse, including at
        the stationary starting point"""

        cam_model = self._cubic_model()
        image_points = torch.tensor(
            [[960.0, 288.0], [832.0, 544.0], [700.0, 100.0]], device=self.device, dtype=self.dtype, requires_grad=True
        )
        weights = torch.tensor([[0.3, -0.7, 0.9], [-0.2, 0.6, 0.4], [0.5, 0.1, -0.8]], dtype=torch.float64)
        rays = cam_model.image_points_to_camera_rays(image_points).camera_rays
        (rays.double().cpu() * weights).sum().backward()

        # Reference: differentiate the closed-form inverse in float64
        k1 = float(np.float32(-1.0 / 3.0))
        ref_points = image_points.detach().double().cpu().requires_grad_()
        uv = (ref_points - torch.tensor([640.0, 288.0], dtype=torch.float64)) / 640.0
        radii = uv.norm(dim=1)
        a = math.sqrt(-3.0 * k1)
        thetas = 2.0 / a * torch.sin(torch.asin(1.5 * radii * a) / 3.0)
        ref_rays = torch.cat([uv * (torch.sin(thetas) / radii)[:, None], torch.cos(thetas)[:, None]], 1)
        (ref_rays * weights).sum().backward()

        rtol = 1e-4 if self.dtype == torch.float32 else 1e-9
        torch.testing.assert_close(
            unpack_optional(image_points.grad).double().cpu(), unpack_optional(ref_points.grad), rtol=rtol, atol=rtol
        )


@parameterized.parameterized_class(
    ("device", "dtype"), itertools.product(_get_test_devices(), (torch.float32, torch.float64))
)
class TestCameraRaysValidFlag(CommonTestCase):
    """Valid flags of image-point unprojections: a ray is valid iff the model has a well-defined ray for the image
    point within its calibrated field of view (independently of the image bounds)"""

    device: torch.device
    dtype: torch.dtype

    def _image_points(self, points: List[List[float]]) -> torch.Tensor:
        return torch.tensor(points, device=self.device, dtype=self.dtype)

    def _assert_valid_flag(self, rays: CameraModel.CameraRaysReturn, expected: List[bool]):
        self.assertEqual(rays.camera_rays.shape, (len(expected), 3))
        self.assertEqual(rays.valid_flag.dtype, torch.bool)
        self.assertEqual(rays.valid_flag.device.type, self.device.type)
        self.assertEqual(rays.valid_flag.cpu().tolist(), expected)

    def test_fisheye_valid_within_max_angle(self):
        """Fisheye rays are valid strictly within max_angle, also beyond the image bounds, and invalid for
        extrapolated rays beyond max_angle and boundary rays beyond the maximum of the polynomial"""

        # r(theta) = theta - theta^3/3 is increasing up to theta = 1 (r = 2/3), max_angle = 0.8 corresponds to
        # r = 0.6293 (403 pixels from the principal point at 640, 288)
        cam_model = OpenCVFisheyeCameraModel(
            OpenCVFisheyeCameraModelParameters(
                resolution=np.array([1280, 576], dtype=np.uint64),
                shutter_type=ShutterType.GLOBAL,
                principal_point=np.array([640.0, 288.0], dtype=np.float32),
                focal_length=np.array([640.0, 640.0], dtype=np.float32),
                radial_coeffs=np.array([-1.0 / 3.0, 0.0, 0.0, 0.0], dtype=np.float32),
                max_angle=0.8,
            ),
            device=self.device,
            dtype=self.dtype,
        )
        rays = cam_model.image_points_to_camera_rays(
            self._image_points(
                [
                    [640.0, 288.0],  # principal point
                    [640.0 + 0.5 * 640.0, 288.0],  # within max_angle
                    [640.0, 288.0 + 0.6 * 640.0],  # within max_angle, outside of the image
                    [640.0 + 0.65 * 640.0, 288.0],  # beyond max_angle, extrapolated
                    [640.0 + 0.9 * 640.0, 288.0],  # beyond the maximum of the polynomial
                ]
            )
        )
        self._assert_valid_flag(rays, [True, True, True, False, False])

    def test_ftheta_valid_within_max_angle(self):
        """FTheta rays are valid strictly within max_angle"""

        cam_model = FThetaCameraModel(
            FThetaCameraModelParameters(
                resolution=np.array([1000, 1000], dtype=np.uint64),
                shutter_type=ShutterType.GLOBAL,
                principal_point=np.array([499.5, 499.5], dtype=np.float32),
                reference_poly=FThetaCameraModelParameters.PolynomialType.PIXELDIST_TO_ANGLE,
                pixeldist_to_angle_poly=np.array([0.0, 1e-3, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                angle_to_pixeldist_poly=np.array([0.0, 1e3, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                max_angle=0.6,
                linear_cde=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            ),
            device=self.device,
            dtype=self.dtype,
        )
        # theta = 1e-3 * pixel distance from the principal point (at 500, 500 in image point coordinates)
        rays = cam_model.image_points_to_camera_rays(
            self._image_points([[500.0, 500.0], [800.0, 500.0], [500.0, 1050.0], [500.0, 1150.0]])
        )
        self._assert_valid_flag(rays, [True, True, True, False])

    def test_ftheta_invalid_for_backward_inverse(self):
        """FTheta rays are invalid if the inverse of the forward polynomial yields a negative angle"""

        # angle_to_pixeldist = 1000*theta - 200*theta^3 has its maximum at theta = 1.29 (r = 860.7 pixels), beyond
        # which the Newton-based inverse yields negative angles (backward-facing rays)
        cam_model = FThetaCameraModel(
            FThetaCameraModelParameters(
                resolution=np.array([1000, 1000], dtype=np.uint64),
                shutter_type=ShutterType.GLOBAL,
                principal_point=np.array([499.5, 499.5], dtype=np.float32),
                reference_poly=FThetaCameraModelParameters.PolynomialType.ANGLE_TO_PIXELDIST,
                pixeldist_to_angle_poly=np.array([0.0, 1e-3, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                angle_to_pixeldist_poly=np.array([0.0, 1e3, 0.0, -200.0, 0.0, 0.0], dtype=np.float32),
                max_angle=0.6,
                linear_cde=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            ),
            device=self.device,
            dtype=self.dtype,
        )
        rays = cam_model.image_points_to_camera_rays(self._image_points([[500.0, 800.0], [500.0, 3000.0]]))
        self.assertLess(float(rays.camera_rays[1, 2]), 0.0)
        self._assert_valid_flag(rays, [True, False])

    def test_camera_rays_to_image_points_combines_valid_flags(self):
        """Projecting a CameraRaysReturn combines its valid flag with the validity of the projection, both for image
        points and pixels, while plain ray tensors keep the projection's validity"""

        cam_model = IdealPinholeCameraModel(
            IdealPinholeCameraModelParameters(
                resolution=np.array([640, 480], dtype=np.uint64),
                shutter_type=ShutterType.GLOBAL,
                principal_point=np.array([320.0, 240.0], dtype=np.float32),
                focal_length=np.array([500.0, 500.0], dtype=np.float32),
            ),
            device=self.device,
            dtype=self.dtype,
        )
        rays = torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [5.0, 0.0, 1.0]], device=self.device, dtype=self.dtype)
        expected_projection_valid = [True, True, False]  # the last ray projects outside of the image

        self.assertEqual(
            cam_model.camera_rays_to_image_points(rays).valid_flag.cpu().tolist(), expected_projection_valid
        )

        camera_rays = CameraModel.CameraRaysReturn(
            camera_rays=rays, valid_flag=torch.tensor([False, True, True], device=self.device)
        )
        self.assertEqual(
            cam_model.camera_rays_to_image_points(camera_rays).valid_flag.cpu().tolist(), [False, True, False]
        )
        self.assertEqual(cam_model.camera_rays_to_pixels(camera_rays).valid_flag.cpu().tolist(), [False, True, False])

        # Jacobians are unaffected by the combination of the flags
        with_jacobians = cam_model.camera_rays_to_image_points(camera_rays, return_jacobians=True)
        self.assertIsNotNone(with_jacobians.jacobians)
        self.assertEqual(with_jacobians.valid_flag.cpu().tolist(), [False, True, False])

    def test_all_valid_camera_rays(self):
        """All-valid camera rays have a valid flag of True per ray, on the rays' device, and are combined with the
        validity of projections"""
        rays = torch.tensor([[0.0, 0.0, 1.0], [5.0, 0.0, 1.0]], device=self.device, dtype=self.dtype)
        all_valid = CameraModel.CameraRaysReturn.all_valid(rays)
        self.assertIs(all_valid.camera_rays, rays)
        self.assertEqual(all_valid.valid_flag.dtype, torch.bool)
        self.assertEqual(all_valid.valid_flag.shape, rays.shape[:-1])
        self.assertEqual(all_valid.valid_flag.device, rays.device)
        self.assertTrue(bool(all_valid.valid_flag.all()))

        cam_model = IdealPinholeCameraModel(
            IdealPinholeCameraModelParameters(
                resolution=np.array([100, 100], dtype=np.uint64),
                shutter_type=ShutterType.GLOBAL,
                principal_point=np.array([50.0, 50.0], dtype=np.float32),
                focal_length=np.array([50.0, 50.0], dtype=np.float32),
            ),
            device=self.device,
            dtype=self.dtype,
        )
        # (the second ray projects outside of the image)
        self.assertEqual(cam_model.camera_rays_to_image_points(all_valid).valid_flag.cpu().tolist(), [True, False])

    def test_ideal_pinhole_always_valid(self):
        """Ideal pinhole rays are valid for all image points"""

        cam_model = IdealPinholeCameraModel(
            IdealPinholeCameraModelParameters(
                resolution=np.array([640, 480], dtype=np.uint64),
                shutter_type=ShutterType.GLOBAL,
                principal_point=np.array([320.0, 240.0], dtype=np.float32),
                focal_length=np.array([500.0, 500.0], dtype=np.float32),
            ),
            device=self.device,
            dtype=self.dtype,
        )
        rays = cam_model.image_points_to_camera_rays(self._image_points([[320.0, 240.0], [-1e4, 1e4]]))
        self._assert_valid_flag(rays, [True, True])

    def _opencv_pinhole(self, radial_coeffs: List[float]) -> OpenCVPinholeCameraModel:
        return OpenCVPinholeCameraModel(
            OpenCVPinholeCameraModelParameters(
                resolution=np.array([640, 480], dtype=np.uint64),
                shutter_type=ShutterType.GLOBAL,
                principal_point=np.array([320.0, 240.0], dtype=np.float32),
                focal_length=np.array([500.0, 500.0], dtype=np.float32),
                radial_coeffs=np.array(radial_coeffs, dtype=np.float32),
                tangential_coeffs=np.zeros(2, dtype=np.float32),
                thin_prism_coeffs=np.zeros(4, dtype=np.float32),
            ),
            device=self.device,
            dtype=self.dtype,
        )

    def test_opencv_pinhole_valid_if_undistortion_converges(self):
        """OpenCV pinhole rays are valid iff the iterative undistortion of the image point converges"""

        # Mild barrel distortion: the undistortion converges within the image and slightly beyond
        rays = self._opencv_pinhole([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0]).image_points_to_camera_rays(
            self._image_points([[320.0, 240.0], [600.0, 400.0], [700.0, 500.0]])
        )
        self._assert_valid_flag(rays, [True, True, True])

        # Strong pincushion distortion: the fixed-point undistortion diverges far from the principal point
        rays = self._opencv_pinhole([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]).image_points_to_camera_rays(
            self._image_points([[320.0, 240.0], [330.0, 245.0], [2000.0, 2000.0]])
        )
        self._assert_valid_flag(rays, [True, True, False])

    def test_distortion_free_opencv_pinhole_always_valid(self):
        """Distortion-free OpenCV pinholes take the closed-form path and are valid for all image points"""

        rays = self._opencv_pinhole([0.0] * 6).image_points_to_camera_rays(
            self._image_points([[320.0, 240.0], [-1e4, 1e4]])
        )
        self._assert_valid_flag(rays, [True, True])

    def test_pixels_and_world_rays_forward_valid_flag(self):
        """pixels_to_camera_rays and all world-ray unprojections forward the valid flag of the camera rays,
        including camera rays that are passed in"""

        cam_model = OpenCVFisheyeCameraModel(
            OpenCVFisheyeCameraModelParameters(
                resolution=np.array([1280, 576], dtype=np.uint64),
                shutter_type=ShutterType.ROLLING_TOP_TO_BOTTOM,
                principal_point=np.array([640.0, 288.0], dtype=np.float32),
                focal_length=np.array([640.0, 640.0], dtype=np.float32),
                radial_coeffs=np.array([-1.0 / 3.0, 0.0, 0.0, 0.0], dtype=np.float32),
                max_angle=0.8,
            ),
            device=self.device,
            dtype=self.dtype,
        )
        # Pixel 320 to the right of the principal point is within max_angle (r = 0.5), pixel 600 is beyond (r = 0.94)
        pixel_idxs = torch.tensor([[640, 288], [960, 288], [1240, 288]], dtype=torch.int32, device=self.device)
        expected = [True, True, False]

        camera_rays = cam_model.pixels_to_camera_rays(pixel_idxs)
        self._assert_valid_flag(camera_rays, expected)

        T_start = torch.eye(4, device=self.device, dtype=self.dtype)
        T_end = T_start.clone()
        T_end[0, 3] = 1.0

        for passed_camera_rays in (None, camera_rays):
            for name, world_rays in (
                (
                    "static",
                    cam_model.pixels_to_world_rays_static_pose(pixel_idxs, T_start, camera_rays=passed_camera_rays),
                ),
                (
                    "mean",
                    cam_model.pixels_to_world_rays_mean_pose(
                        pixel_idxs, T_start, T_end, camera_rays=passed_camera_rays
                    ),
                ),
                (
                    "shutter",
                    cam_model.pixels_to_world_rays_shutter_pose(
                        pixel_idxs, T_start, T_end, camera_rays=passed_camera_rays
                    ),
                ),
            ):
                with self.subTest(pose=name, passed_camera_rays=passed_camera_rays is not None):
                    self.assertEqual(world_rays.world_rays.shape, (3, 6))
                    self.assertEqual(world_rays.valid_flag.cpu().tolist(), expected)

        # The flag of passed camera rays is forwarded as is
        passed = CameraModel.CameraRaysReturn(
            camera_rays=camera_rays.camera_rays,
            valid_flag=torch.tensor([False, True, True], device=self.device),
        )
        world_rays = cam_model.pixels_to_world_rays_shutter_pose(pixel_idxs, T_start, T_end, camera_rays=passed)
        self.assertEqual(world_rays.valid_flag.cpu().tolist(), [False, True, True])


class CameraModelsBaseTestCase(CommonTestCase):
    def setUp(self) -> None:
        # Make printed errors more representable numerically
        np.set_printoptions(floatmode="unique", linewidth=200, suppress=True)

        # Real-world customer camera parameters to test
        self.cam_model_params: List[ConcreteCameraModelParametersUnion] = [
            # fw-based ftheta camera model
            FThetaCameraModelParameters(
                resolution=np.array([3848, 2168], dtype=np.uint64),
                shutter_type=ShutterType.ROLLING_TOP_TO_BOTTOM,
                principal_point=np.array([1909.3092041015625, 1103.27880859375], dtype=np.float32),
                reference_poly=FThetaCameraModelParameters.PolynomialType.ANGLE_TO_PIXELDIST,
                pixeldist_to_angle_poly=np.array(
                    [
                        0.0,
                        0.00031855489942245185,
                        -5.4367417234857385e-09,
                        4.775631279319015e-12,
                        -1.0283620548333567e-15,
                        -1.1274463994279525e-19,
                    ],
                    dtype=np.float32,
                ),
                angle_to_pixeldist_poly=np.array(
                    [
                        0.0,
                        3139.48583984375,
                        164.5725860595703,
                        -442.12896728515625,
                        259.5827331542969,
                        153.66644287109375,
                    ],
                    dtype=np.float32,
                ),
                max_angle=0.7037167544041137,
                linear_cde=np.array([1.1, -0.1, 0.2], dtype=np.float32),  # updated from [1,0,0] to be more significant
            ),
            # bw-based ftheta camera model
            FThetaCameraModelParameters(
                resolution=np.array([3848, 2168], dtype=np.uint64),
                shutter_type=ShutterType.ROLLING_TOP_TO_BOTTOM,
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
            ),
            OpenCVPinholeCameraModelParameters(
                resolution=np.array([1920, 1280], dtype=np.uint64),
                shutter_type=ShutterType.ROLLING_RIGHT_TO_LEFT,
                principal_point=np.array([935.1248081874216, 635.052474560227], dtype=np.float32),
                focal_length=np.array(
                    [
                        2059.0471439559833,
                        2059.0471439559833,
                    ],
                    dtype=np.float32,
                ),
                radial_coeffs=np.array(
                    [
                        0.04239636827428756,
                        -0.34165672675852826,
                        0,
                        0,
                        0,
                        0,
                    ],
                    dtype=np.float32,
                ),
                tangential_coeffs=np.array([0.001805535524580487, -0.00005530628187935031], dtype=np.float32),
                thin_prism_coeffs=np.array([0, 0, 0, 0], dtype=np.float32),
            ),
            # opencv pinhole camera model with rational radial and thin-prism, but without tangential distortion
            OpenCVPinholeCameraModelParameters(
                resolution=np.array([1920, 1280], dtype=np.uint64),
                shutter_type=ShutterType.GLOBAL,
                principal_point=np.array([958.3, 641.7], dtype=np.float32),
                focal_length=np.array([2061.5, 2060.25], dtype=np.float32),
                radial_coeffs=np.array([0.0424, -0.3417, 0.01, 0.02, -0.01, 0.02], dtype=np.float32),
                tangential_coeffs=np.array([0, 0], dtype=np.float32),
                thin_prism_coeffs=np.array([0.01, 0.02, 0.02, 0.01], dtype=np.float32),
            ),
            OpenCVFisheyeCameraModelParameters(
                resolution=np.array([3840, 2160], dtype=np.uint64),
                shutter_type=ShutterType.ROLLING_TOP_TO_BOTTOM,
                principal_point=np.array([1928.184506, 1083.862789], dtype=np.float32),
                focal_length=np.array(
                    [
                        1913.76478,
                        1913.99708,
                    ],
                    dtype=np.float32,
                ),
                radial_coeffs=np.array(
                    [
                        -0.030093122,
                        -0.005103817,
                        -0.000849622,
                        0.001079542,
                    ],
                    dtype=np.float32,
                ),
                max_angle=np.deg2rad(140 / 2),
            ),
            IdealPinholeCameraModelParameters(
                resolution=np.array([1920, 1280], dtype=np.uint64),
                shutter_type=ShutterType.GLOBAL,
                principal_point=np.array([959.5, 639.5], dtype=np.float32),
                focal_length=np.array([1480.0, 1480.0], dtype=np.float32),
            ),
            # ftheta camera models with tangential distortion
            _ftheta_front_tele_params(),
            _ftheta_cross_right_params(),
        ]

        # Add an arbitrary dummy windshield model
        horizontal_poly = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        vertical_poly = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        windshield_model_parameters = BivariateWindshieldModelParameters(
            ReferencePolynomial.FORWARD,
            horizontal_poly,
            vertical_poly,
            horizontal_poly,
            vertical_poly,
        )

        self.cam_model_params_wsd = []
        for cam_model_params in self.cam_model_params:
            self.cam_model_params_wsd.append(
                dataclasses.replace(cam_model_params, external_distortion_parameters=windshield_model_parameters)
            )

        # ftheta fisheye beyond a hemisphere with explicit zero tangential distortion (without the dummy windshield
        # model, whose float32 ray z-component sqrt(1 - x^2 - y^2) is too imprecise for rays close to 90 degrees)
        self.cam_model_params.append(_ftheta_fisheye_params())


@parameterized.parameterized_class(
    ("device", "dtype"), itertools.product(_get_test_devices(), (torch.float32, torch.float64))
)
class TestParameterIO(CameraModelsBaseTestCase):
    device: torch.device
    dtype: torch.dtype

    def test_model_parameters_roundtrip(self):
        """Validate model parameters obtained from torch model instances are correctly mapped back to the input versions
        between device transfers"""

        for cam_model_params in self.cam_model_params + self.cam_model_params_wsd:
            with self.subTest(cam_model_params=cam_model_params):
                cam_model = camera_model_from_parameters(cam_model_params, device=self.device, dtype=self.dtype)

                self.assertEqual(
                    cam_model.resolution.device.type, self.device.type
                )  # make sure original device is correct

                cam_model = cast(ConcreteCameraModelUnion, cam_model)

                # make sure retrieved parameters correspond to reference
                self.assertEqual(cam_model_params.to_json(), cam_model.get_parameters().to_json())

                # flip flop device using nn.Module magic
                if self.device.type == "cpu" and len(_get_test_devices()) == 1:
                    # When on CPU and GPU tests are disabled, we can't flip to CUDA
                    # Just verify CPU -> CPU works
                    cam_model.to(device=torch.device("cpu"))
                    new_device_str = "cpu"
                else:
                    new_device_str = "cuda" if self.device.type == "cpu" else "cpu"
                    cam_model.to(device=torch.device(new_device_str))
                self.assertEqual(
                    cam_model.resolution.device.type, new_device_str
                )  # make sure the new device is correct

                # make sure retrieved parameters still correspond to reference
                self.assertEqual(cam_model_params.to_json(), cam_model.get_parameters().to_json())

    def test_encode_decode_roundtrip(self):
        """Validate encode/decode of camera model parameters preserves all model types"""

        for cam_model_params in self.cam_model_params + self.cam_model_params_wsd:
            with self.subTest(cam_model_params=cam_model_params):
                encoded = encode_camera_model_parameters(cam_model_params)
                self.assertEqual(encoded["camera_model_type"], cam_model_params.type())

                decoded = decode_camera_model_parameters(encoded)
                self.assertIs(type(decoded), type(cam_model_params))
                self.assertEqual(decoded.to_json(), cam_model_params.to_json())


@parameterized.parameterized_class(
    ("device", "dtype"), itertools.product(_get_test_devices(), (torch.float32, torch.float64))
)
class TestCameraRaysValidFlagConsistency(CameraModelsBaseTestCase):
    """Calibration-independent consistency of the valid flags of image-point unprojections for all camera models"""

    device: torch.device
    dtype: torch.dtype

    MAX_DEVIATION_IN_IMAGE_COORDINATES = 0.01
    GRID_STEP = 16

    def _image_point_grid(self, cam_model: CameraModel, margin: int) -> torch.Tensor:
        # Grid covering the image and a margin around it (to also exercise points outside of the image)
        width, height = (int(v) for v in cam_model.resolution.cpu())
        xs = torch.arange(-margin, width + margin, self.GRID_STEP, dtype=self.dtype) + 0.5
        ys = torch.arange(-margin, height + margin, self.GRID_STEP, dtype=self.dtype) + 0.5
        return torch.stack(torch.meshgrid(xs, ys, indexing="xy"), dim=-1).reshape(-1, 2).to(self.device)

    def test_valid_rays_reproject_consistently(self):
        """Valid rays are finite, normalized, and reproject to their image points, both directly and through
        camera_rays_to_image_points with the CameraRaysReturn (whose valid flag is combined with the projection's)"""

        for cam_model_params in self.cam_model_params + self.cam_model_params_wsd:
            cam_model = camera_model_from_parameters(cam_model_params, device=self.device, dtype=self.dtype)
            name = f"{type(cam_model).__name__}{'+wsd' if cam_model.external_distortion is not None else ''}"
            with self.subTest(model=name):
                image_points = self._image_point_grid(cam_model, margin=int(cam_model.resolution.max()) // 4)
                camera_rays = cam_model.image_points_to_camera_rays(image_points)
                valid = camera_rays.valid_flag

                self.assertEqual(valid.shape, (len(image_points),))
                self.assertEqual(valid.dtype, torch.bool)
                self.assertGreater(int(valid.sum()), 0, msg="Expected valid rays for image points within the image")

                # Valid rays are finite and normalized
                valid_rays = camera_rays.camera_rays[valid]
                self.assertTrue(bool(torch.isfinite(valid_rays).all()))
                torch.testing.assert_close(
                    torch.linalg.norm(valid_rays, dim=1), torch.ones_like(valid_rays[:, 0]), atol=1e-5, rtol=0.0
                )

                # Projections of camera rays given as CameraRaysReturn are never valid for invalid rays
                projection = cam_model.camera_rays_to_image_points(camera_rays)
                self.assertFalse(bool((projection.valid_flag & ~valid).any()))

                # All valid rays reproject to their image points, also outside of the image (where the projection
                # itself is invalid only because of the image bounds)
                deviation = torch.linalg.norm(projection.image_points[valid] - image_points[valid], dim=1)
                self.assertLessEqual(float(deviation.max()), self.MAX_DEVIATION_IN_IMAGE_COORDINATES)

                # Valid rays of image points within the image also reproject validly
                width, height = cam_model.resolution.cpu().tolist()
                in_image = (
                    (image_points[:, 0] >= 0)
                    & (image_points[:, 0] < width)
                    & (image_points[:, 1] >= 0)
                    & (image_points[:, 1] < height)
                )
                self.assertTrue(bool(projection.valid_flag[valid & in_image].all()))


@parameterized.parameterized_class(
    ("device", "dtype"), itertools.product(_get_test_devices(), (torch.float32, torch.float64))
)
class TestProjectionValidIndices(CameraModelsBaseTestCase):
    """valid_indices of world-point projections for all camera models and shutter types"""

    device: torch.device
    dtype: torch.dtype

    def test_valid_indices_are_1d_and_ordered(self):
        """valid_indices stays 1-D and ordered for zero, one, or more valid projections on every projection path"""
        T_world_sensor = np.eye(4)

        # Points on the optical axis project to the principal point of every camera model
        in_front = [0, 0, 10]
        behind = [0, 0, -10]
        further_in_front = [0, 0, 20]

        def check(cam_model: CameraModel, world_points: List[List[int]], expected_valid_indices: List[int]) -> None:
            # Shutter-pose projection takes a separate code path for global and rolling shutters
            shutter_pose = cam_model.world_points_to_image_points_shutter_pose(
                np.array(world_points), T_world_sensor, T_world_sensor, return_valid_indices=True
            )
            self.assertEqual(unpack_optional(shutter_pose.valid_indices).tolist(), expected_valid_indices)

            static_pose = cam_model.world_points_to_image_points_static_pose(
                np.array(world_points), T_world_sensor, return_valid_indices=True
            )
            self.assertEqual(unpack_optional(static_pose.valid_indices).tolist(), expected_valid_indices)

        for cam_model_params in self.cam_model_params + self.cam_model_params_wsd:
            # Covers every shutter type so new types are tested automatically
            for shutter_type in ShutterType:
                cam_model = camera_model_from_parameters(
                    dataclasses.replace(cam_model_params, shutter_type=shutter_type),
                    device=self.device,
                    dtype=self.dtype,
                )
                name = f"{type(cam_model).__name__}{'+wsd' if cam_model.external_distortion is not None else ''}"
                with self.subTest(model=name, shutter_type=shutter_type):
                    # No valid points: valid_indices is empty
                    check(cam_model, [behind], expected_valid_indices=[])

                    # Single valid point: valid_indices must be [0], not a 0-dim scalar
                    check(cam_model, [in_front, behind], expected_valid_indices=[0])

                    # Multiple valid points: indices of the valid points, in input order
                    check(cam_model, [in_front, behind, further_in_front], expected_valid_indices=[0, 2])


@parameterized.parameterized_class(
    ("device", "dtype"), itertools.product(_get_test_devices(), (torch.float32, torch.float64))
)
class TestProjectionValidFlag(CameraModelsBaseTestCase):
    """valid_flag of world-point projections for all camera models, shutter types and projection methods"""

    device: torch.device
    dtype: torch.dtype

    def _projections(self, cam_model: CameraModel, world_points: np.ndarray, **kwargs):
        T_world_sensor = np.eye(4)
        return {
            "image_points_shutter": cam_model.world_points_to_image_points_shutter_pose(
                world_points, T_world_sensor, T_world_sensor, **kwargs
            ),
            "image_points_static": cam_model.world_points_to_image_points_static_pose(
                world_points, T_world_sensor, **kwargs
            ),
            "image_points_mean": cam_model.world_points_to_image_points_mean_pose(
                world_points, T_world_sensor, T_world_sensor, **kwargs
            ),
            "pixels_shutter": cam_model.world_points_to_pixels_shutter_pose(
                world_points, T_world_sensor, T_world_sensor, **kwargs
            ),
            "pixels_static": cam_model.world_points_to_pixels_static_pose(world_points, T_world_sensor, **kwargs),
            "pixels_mean": cam_model.world_points_to_pixels_mean_pose(
                world_points, T_world_sensor, T_world_sensor, **kwargs
            ),
        }

    def test_valid_flag(self):
        """The valid flag covers all input points, with and without return_all_projections"""
        # Points on the optical axis project to the principal point of every camera model
        world_points = np.array([[0, 0, 10], [0, 0, -10], [0, 0, 20], [0, 0, -5]])
        expected = [True, False, True, False]

        for cam_model_params in self.cam_model_params + self.cam_model_params_wsd:
            for shutter_type in (ShutterType.GLOBAL, ShutterType.ROLLING_TOP_TO_BOTTOM):
                cam_model = camera_model_from_parameters(
                    dataclasses.replace(cam_model_params, shutter_type=shutter_type),
                    device=self.device,
                    dtype=self.dtype,
                )
                for return_all_projections in (False, True):
                    projections = self._projections(
                        cam_model, world_points, return_all_projections=return_all_projections
                    )
                    for name, projection in projections.items():
                        with self.subTest(
                            model=type(cam_model).__name__,
                            shutter_type=shutter_type,
                            method=name,
                            return_all_projections=return_all_projections,
                        ):
                            self.assertEqual(projection.valid_flag.dtype, torch.bool)
                            self.assertEqual(projection.valid_flag.cpu().tolist(), expected)
                            points = (
                                projection.pixels
                                if isinstance(projection, CameraModel.WorldPointsToPixelsReturn)
                                else projection.image_points
                            )
                            self.assertEqual(len(points), len(world_points) if return_all_projections else 2)

    def test_return_flags_are_keyword_only(self):
        cam_model = camera_model_from_parameters(self.cam_model_params[0], device=self.device, dtype=self.dtype)
        world_points = np.array([[0, 0, 10]])
        T = np.eye(4)
        # (untyped: the positional calls don't type check)
        for method, args in (
            (cam_model.world_points_to_image_points_shutter_pose, (world_points, T, T, 0, 1, 10, 1e-4, True)),
            (cam_model.world_points_to_pixels_shutter_pose, (world_points, T, T, 0, 1, 10, 1e-4, True)),
            (cam_model.world_points_to_image_points_static_pose, (world_points, T, 0, True)),
            (cam_model.world_points_to_pixels_static_pose, (world_points, T, 0, True)),
            (cam_model.world_points_to_image_points_mean_pose, (world_points, T, T, 0, 1, True)),
            (cam_model.world_points_to_pixels_mean_pose, (world_points, T, T, 0, 1, True)),
        ):
            with self.subTest(method=method.__name__):
                with self.assertRaisesRegex(TypeError, "positional argument"):
                    cast(Any, method)(*args)


@parameterized.parameterized_class(
    ("device", "dtype"), itertools.product(_get_test_devices(), (torch.float32, torch.float64))
)
class TestTransformParameters(CameraModelsBaseTestCase):
    device: torch.device
    dtype: torch.dtype

    MAX_DEVIATION_IN_IMAGE_COORDINATES = 0.001

    def test_image_domain_transform(self):
        """Validate image up- / down-scaling and offsetting"""

        SCALE_FACTORS = [
            # isotropic scaling
            1.0,  # no scaling
            0.25,  # 4x downscale
            0.5,  # 2x downscale
            2.0,  # 2x upscale
            # anisotropic scaling
            (1.0, 1.0),  # no scaling
            (0.5, 0.5),  # 2x downscale
            (0.5, 1.0),  # 2x downscale in x
            (1.0, 0.5),  # 2x downscale in y
            (2.0, 1.0),  # 2x upscale in x
            (1.0, 2.0),  # 2x upscale in y
            (0.5, 0.25),  # 2x downscale in x, 4x downscale in y
        ]

        OFFSETS = [
            # no offset
            (0.0, 0.0),
            # some offset
            (20.0, 10.0),
        ]

        IMAGE_POINTS = np.array(
            [[150.2, 200.1], [500.1, 500.5], [867.4, 321.7]], dtype=np.float32
        )  # some image coordinates to use for evaluation [should be in the original image domains of all tested camera models incl. (scaled) offsets]

        for scale_factor in SCALE_FACTORS:
            with self.subTest(msg=f"scale_factor {scale_factor}", scale_factor=scale_factor):
                for offset in OFFSETS:
                    with self.subTest(msg=f"offset {offset}", offset=offset):
                        for cam_model_params in self.cam_model_params:
                            with self.subTest(cam_model_params=cam_model_params):
                                cam_model = camera_model_from_parameters(
                                    cam_model_params, device=self.device, dtype=self.dtype
                                )

                                cam_model_transformed = camera_model_from_parameters(
                                    cam_model_params_transformed := cam_model_params.transform(
                                        image_domain_scale=scale_factor,
                                        image_domain_offset=offset,
                                    ),
                                    device=self.device,
                                    dtype=self.dtype,
                                )

                                # Make sure types are preserved
                                self.assertEqual(
                                    type(cam_model_transformed), type(cam_model), msg="Camera model type mismatch"
                                )
                                self.assertEqual(
                                    type(cam_model_params_transformed),
                                    type(cam_model_params),
                                    msg="Camera model parameters type mismatch",
                                )

                                # Validate original image domain -> 3d -> transformed image domain round-trip
                                camera_rays = cam_model.image_points_to_camera_rays(IMAGE_POINTS)
                                self.assertTrue(camera_rays.valid_flag.all(), msg="All rays need to be valid")
                                image_points_transformed = cam_model_transformed.camera_rays_to_image_points(
                                    camera_rays
                                )

                                self.assertTrue(
                                    image_points_transformed.valid_flag.all(),
                                    msg="All point projections need to be valid for scale verification",
                                )

                                self.assertLessEqual(
                                    np.linalg.norm(
                                        (image_points_transformed_ref := (IMAGE_POINTS * scale_factor - offset))
                                        - image_points_transformed.image_points.cpu().numpy()
                                    ).item(),
                                    self.MAX_DEVIATION_IN_IMAGE_COORDINATES,
                                )

                                # Validate transformed image-domain -> 3d -> untransformed image-domain round-trip
                                camera_rays_transformed = cam_model_transformed.image_points_to_camera_rays(
                                    image_points_transformed_ref
                                )

                                image_points_untransformed = cam_model.camera_rays_to_image_points(
                                    camera_rays_transformed
                                )

                                self.assertTrue(
                                    image_points_untransformed.valid_flag.all(),
                                    msg="All point projections need to be valid for transformation verification",
                                )

                                self.assertLessEqual(
                                    np.linalg.norm(
                                        IMAGE_POINTS - image_points_untransformed.image_points.cpu().numpy()
                                    ).item(),
                                    self.MAX_DEVIATION_IN_IMAGE_COORDINATES,
                                )


@parameterized.parameterized_class(
    ("device", "dtype"), itertools.product(_get_test_devices(), (torch.float32, torch.float64))
)
class TestFThetaTangential(CameraModelsBaseTestCase):
    """Tangential distortion of the FTheta camera model"""

    device: torch.device
    dtype: torch.dtype

    def _tolerance(self) -> float:
        return 1e-6 if self.dtype == torch.float64 else 1e-3

    def _roundtrip_tolerance(self) -> float:
        """Image point round trip bound [px] (float32: the image coordinates themselves are only resolved to
        ~2.4e-4 px at x = 3840)"""
        return 1e-3 if self.dtype == torch.float64 else 2e-3

    def _tangential_params(self) -> List[FThetaCameraModelParameters]:
        return [_ftheta_front_tele_params(), _ftheta_cross_right_params()]

    def _zero_tangential_ftheta_params(self) -> List[FThetaCameraModelParameters]:
        return [
            p
            for p in self.cam_model_params + self.cam_model_params_wsd
            if isinstance(p, FThetaCameraModelParameters) and not p.has_tangential_distortion
        ]

    def _ray_grid(self, max_angle: float, num_thetas: int = 32, num_azimuths: int = 64) -> np.ndarray:
        """Unit camera rays with angles in (0, max_angle] to the principal axis, for all azimuths"""
        thetas, azimuths = np.meshgrid(
            np.linspace(max_angle / num_thetas, max_angle, num_thetas),
            np.linspace(0.0, 2 * np.pi, num_azimuths, endpoint=False),
            indexing="ij",
        )
        thetas, azimuths = thetas.reshape(-1), azimuths.reshape(-1)
        return np.stack([np.sin(thetas) * np.cos(azimuths), np.sin(thetas) * np.sin(azimuths), np.cos(thetas)], axis=1)

    @staticmethod
    def _reference_projection(params: FThetaCameraModelParameters, rays: np.ndarray) -> np.ndarray:
        """Independent float64 numpy reference of the FTheta projection with tangential distortion and
        linear term (in image coordinates with the origin at the top-left corner of the first pixel)"""
        rays = rays.astype(np.float64)
        X, Y, Z = rays[:, 0], rays[:, 1], rays[:, 2]
        rho = np.hypot(X, Y)
        theta = np.arctan2(rho, Z)
        delta = np.polynomial.polynomial.polyval(theta, params.angle_to_pixeldist_poly.astype(np.float64))
        x, y = X / Z, Y / Z
        r2 = x * x + y * y
        t0, t1 = (float(t) for t in params.tangential_coeffs)
        u = delta * X / rho + 2 * t0 * x * y + t1 * (r2 + 2 * x * x)
        v = delta * Y / rho + 2 * t1 * x * y + t0 * (r2 + 2 * y * y)
        c, d, e = (float(v) for v in params.linear_cde)
        return np.stack([c * u + d * v, e * u + v], axis=1) + params.principal_point.astype(np.float64) + 0.5

    def test_eval_tangential_distortion_matches_reference_formula(self):
        xy = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.3, -0.2]], dtype=self.dtype, device=self.device)
        t = torch.tensor([0.7, -0.4], dtype=self.dtype, device=self.device)
        out = FThetaCameraModel.eval_tangential_distortion(xy, t).cpu().to(torch.float64)
        # Axis-aligned unit inputs: T(1, 0) = (3 * t1, t0) and T(0, 1) = (t1, 3 * t0)
        torch.testing.assert_close(out[0], torch.tensor([3 * -0.4, 0.7], dtype=torch.float64))
        torch.testing.assert_close(out[1], torch.tensor([-0.4, 3 * 0.7], dtype=torch.float64))
        x, y = 0.3, -0.2
        r2 = x * x + y * y
        torch.testing.assert_close(
            out[2],
            torch.tensor(
                [2 * 0.7 * x * y + -0.4 * (r2 + 2 * x * x), 2 * -0.4 * x * y + 0.7 * (r2 + 2 * y * y)],
                dtype=torch.float64,
            ),
        )

    def test_projection_matches_reference(self):
        """Projections of in-FOV rays coincide with an independent numpy reference"""
        # Also with a non-identity linear term, which is applied after the tangential offsets
        all_params = [
            variant
            for params in self._tangential_params()
            for variant in (params, dataclasses.replace(params, linear_cde=np.array([1.1, -0.1, 0.2], np.float32)))
        ]
        for params in all_params:
            with self.subTest(principal_point=params.principal_point.tolist(), linear_cde=params.linear_cde.tolist()):
                cam_model = camera_model_from_parameters(params, device=self.device, dtype=self.dtype)

                # Evaluate the reference on the rays as represented in the model's precision
                rays = to_torch(self._ray_grid(0.95 * params.max_angle), device=self.device, dtype=self.dtype)
                expected = self._reference_projection(params, rays.cpu().numpy())

                projection = cam_model.camera_rays_to_image_points(rays)
                actual = projection.image_points.cpu().numpy().astype(np.float64)

                self.assertTrue(bool(projection.valid_flag.any()))
                self.assertLessEqual(float(np.abs(actual - expected).max()), self._tolerance())

    def test_zero_tangentials_take_radial_only_path(self):
        """Models with zero tangential coefficients never evaluate the tangential term (NaN coefficients do not
        affect any output), so their outputs are exactly those of the purely radial model"""
        for params in self._zero_tangential_ftheta_params():
            with self.subTest(params=params):
                reference = camera_model_from_parameters(params, device=self.device, dtype=self.dtype)
                # Model buffers may share memory with the parameters' arrays, so poison a copy only
                poisoned = camera_model_from_parameters(copy.deepcopy(params), device=self.device, dtype=self.dtype)
                poisoned.tangential_coeffs.fill_(float("nan"))

                legacy_dict = params.to_dict()
                legacy_dict.pop("tangential_coeffs", None)
                legacy = camera_model_from_parameters(
                    FThetaCameraModelParameters.from_dict(legacy_dict), device=self.device, dtype=self.dtype
                )

                rays = to_torch(self._ray_grid(1.2 * params.max_angle), device=self.device, dtype=self.dtype)
                image_points = rays.new_tensor([[10.5, 20.5], [500.0, 300.0], [1000.0, 700.0], [1700.0, 1100.0]])

                expected_projection = reference.camera_rays_to_image_points(rays, return_jacobians=True)
                expected_rays = reference.image_points_to_camera_rays(image_points)

                for name, model in (("nan", poisoned), ("legacy", legacy)):
                    with self.subTest(model=name):
                        projection = model.camera_rays_to_image_points(rays, return_jacobians=True)
                        camera_rays = model.image_points_to_camera_rays(image_points)
                        for actual, expected in (
                            (projection.image_points, expected_projection.image_points),
                            (projection.valid_flag, expected_projection.valid_flag),
                            (projection.jacobians, expected_projection.jacobians),
                            (camera_rays.camera_rays, expected_rays.camera_rays),
                            (camera_rays.valid_flag, expected_rays.valid_flag),
                        ):
                            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_identity_linear_term_skips_matrix(self):
        """Models with an identity linear term never evaluate the linear matrices (NaN matrices do not affect any
        output, which are exactly those of the unmodified model), models with a non-identity linear term do"""
        ftheta_params = self._tangential_params() + [
            p for p in self.cam_model_params + self.cam_model_params_wsd if isinstance(p, FThetaCameraModelParameters)
        ]
        identity_params = [p for p in ftheta_params if not p.has_linear_term]
        non_identity_params = [p for p in ftheta_params if p.has_linear_term]
        # Cover both the tangential and the purely radial code paths, and the non-identity case
        self.assertTrue(any(p.has_tangential_distortion for p in identity_params))
        self.assertTrue(any(not p.has_tangential_distortion for p in identity_params))
        self.assertTrue(non_identity_params)

        def evaluate(model: FThetaCameraModel, max_angle: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            rays = to_torch(self._ray_grid(0.95 * max_angle), device=self.device, dtype=self.dtype)
            image_points = rays.new_tensor([[10.5, 20.5], [500.0, 300.0], [1000.0, 700.0], [1700.0, 1100.0]])
            projection = model.camera_rays_to_image_points(rays, return_jacobians=True)
            camera_rays = model.image_points_to_camera_rays(image_points)
            assert projection.jacobians is not None
            return projection.image_points, projection.jacobians, camera_rays.camera_rays

        for params, is_identity in [(p, True) for p in identity_params] + [(p, False) for p in non_identity_params]:
            with self.subTest(params=params):
                reference = camera_model_from_parameters(params, device=self.device, dtype=self.dtype)
                # Model buffers may share memory with the parameters' arrays, so poison a copy only
                poisoned = camera_model_from_parameters(copy.deepcopy(params), device=self.device, dtype=self.dtype)
                assert isinstance(reference, FThetaCameraModel) and isinstance(poisoned, FThetaCameraModel)
                poisoned.A.fill_(float("nan"))
                poisoned.Ainv.fill_(float("nan"))

                expected_outputs = evaluate(reference, params.max_angle)
                actual_outputs = evaluate(poisoned, params.max_angle)
                for actual, expected in zip(actual_outputs, expected_outputs):
                    if is_identity:
                        self.assertTrue(bool(torch.isfinite(actual).all()))
                        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
                    else:
                        self.assertTrue(bool(torch.isnan(actual).any()))

    def test_get_parameters_roundtrips_tangentials(self):
        for params in self._tangential_params():
            with self.subTest(principal_point=params.principal_point.tolist()):
                cam_model = camera_model_from_parameters(params, device=self.device, dtype=self.dtype)
                self.assertEqual(cam_model.get_parameters().to_json(), params.to_json())
                np.testing.assert_array_equal(cam_model.get_parameters().tangential_coeffs, params.tangential_coeffs)

    def test_out_of_fov_rays_finite(self):
        """Rays beyond the FOV (incl. at 90 degrees and behind the camera) project to finite, invalid points with
        finite Jacobians, on the cone of the maximum angle"""
        params = _ftheta_cross_right_params()
        cam_model = camera_model_from_parameters(params, device=self.device, dtype=self.dtype)

        azimuth = 0.7
        direction = np.array([np.cos(azimuth), np.sin(azimuth)])

        def ray(theta: float) -> List[float]:
            return [*(np.sin(theta) * direction), np.cos(theta)]

        rays = to_torch(
            np.array(
                [
                    ray(params.max_angle + 0.01),
                    [*direction, 0.0],  # exactly at 90 degrees
                    ray(2.0),  # behind the camera
                    [*(1e-3 * direction), -1.0],  # almost opposite to the principal axis
                    [0.0, 0.0, -1.0],  # opposite to the principal axis
                    [0.0, 0.0, 1.0],  # principal axis
                ]
            ),
            device=self.device,
            dtype=self.dtype,
        )

        projection = cam_model.camera_rays_to_image_points(rays, return_jacobians=True)

        self.assertTrue(bool(torch.isfinite(projection.image_points).all()))
        self.assertTrue(bool(torch.isfinite(unpack_optional(projection.jacobians)).all()))
        self.assertEqual(projection.valid_flag.cpu().tolist(), [False] * 5 + [True])

        # The principal axis maps to the principal point exactly
        torch.testing.assert_close(projection.image_points[-1], cam_model.principal_point, rtol=0.0, atol=0.0)

        # Out-of-FOV rays with the same azimuth are clamped to the cone of the maximum angle
        boundary = cam_model.camera_rays_to_image_points(
            to_torch(np.array([ray(params.max_angle)]), device=self.device, dtype=self.dtype)
        ).image_points
        torch.testing.assert_close(
            projection.image_points[:4],
            boundary.expand(4, 2),
            rtol=0.0,
            atol=self._tolerance() if self.dtype == torch.float64 else 1e-2,
        )

    def test_transform_behavioural(self):
        """Projections (unprojections) of transformed parameters coincide with the transformed projections
        (unprojections of the transformed image points) of the original ones"""
        scales_offsets: List[Tuple[Union[float, Tuple[float, float]], Tuple[float, float]]] = [
            (0.5, (0.0, 0.0)),
            ((0.5, 0.25), (0.0, 0.0)),
            (1.0, (13.0, 7.0)),
            ((0.5, 0.25), (13.0, 7.0)),
        ]
        for params in self._tangential_params() + [_ftheta_fisheye_params()]:
            rays = to_torch(self._ray_grid(0.95 * params.max_angle), device=self.device, dtype=self.dtype)
            cam_model = camera_model_from_parameters(params, device=self.device, dtype=self.dtype)
            p0 = cam_model.camera_rays_to_image_points(rays)
            image_points = self._image_grid(params, num_x=25, num_y=15)
            r0 = cam_model.image_points_to_camera_rays(image_points)
            for scale, offset in scales_offsets:
                with self.subTest(principal_point=params.principal_point.tolist(), scale=scale, offset=offset):
                    transformed = camera_model_from_parameters(
                        params.transform(image_domain_scale=scale, image_domain_offset=offset),
                        device=self.device,
                        dtype=self.dtype,
                    )
                    p1 = transformed.camera_rays_to_image_points(rays).image_points

                    # Image coordinates have their origin at the top-left corner of the first pixel, so the image
                    # domain transformation is a plain scale followed by the offset
                    expected = p0.image_points * p0.image_points.new_tensor(
                        scale if isinstance(scale, tuple) else (scale, scale)
                    ) - p0.image_points.new_tensor(offset)
                    self.assertLessEqual(float((p1 - expected).abs().max()), self._tolerance())

                    # Unprojections of the transformed image points by the transformed parameters coincide with the
                    # unprojections of the original image points
                    scale_xy = image_points.new_tensor(scale if isinstance(scale, tuple) else (scale, scale))
                    r1 = transformed.image_points_to_camera_rays(
                        image_points * scale_xy - image_points.new_tensor(offset)
                    )
                    both_valid = r0.valid_flag & r1.valid_flag
                    self.assertGreater(int(both_valid.sum()), 0)
                    self.assertTrue(bool((r0.valid_flag == r1.valid_flag).all()))
                    rays0 = r0.camera_rays[both_valid].to(torch.float64)
                    rays1 = r1.camera_rays[both_valid].to(torch.float64)
                    angles = torch.atan2(
                        torch.linalg.norm(torch.linalg.cross(rays0, rays1), dim=1), (rays0 * rays1).sum(dim=1)
                    )
                    # Same bound for float32 (the power-of-two scales and integer offsets transform exactly)
                    self.assertLessEqual(float(angles.max()), 1e-5)

    def _image_grid(self, params: FThetaCameraModelParameters, num_x: int = 97, num_y: int = 55) -> torch.Tensor:
        """Regular grid of image points over [0, W] x [0, H], incl. the image corners"""
        width, height = (float(v) for v in params.resolution)
        xs, ys = np.meshgrid(np.linspace(0.0, width, num_x), np.linspace(0.0, height, num_y))
        return to_torch(np.stack([xs.reshape(-1), ys.reshape(-1)], axis=1), device=self.device, dtype=self.dtype)

    def _roundtrip_params(self) -> List[FThetaCameraModelParameters]:
        """Tangential models with either polynomial as reference (with the backward polynomial as reference it is the
        exact unprojection, the forward polynomial only its approximate inverse)"""
        return [
            dataclasses.replace(params, reference_poly=reference_poly)
            for params in self._tangential_params()
            for reference_poly in (
                FThetaCameraModelParameters.PolynomialType.ANGLE_TO_PIXELDIST,
                FThetaCameraModelParameters.PolynomialType.PIXELDIST_TO_ANGLE,
            )
        ]

    def test_unprojection_roundtrip_full_image(self):
        """Unprojected image points over the full image reproject to themselves"""
        tolerance = self._roundtrip_tolerance()
        for params in self._roundtrip_params():
            with self.subTest(principal_point=params.principal_point.tolist(), reference_poly=params.reference_poly):
                cam_model = camera_model_from_parameters(params, device=self.device, dtype=self.dtype)
                image_points = self._image_grid(params)

                camera_rays = cam_model.image_points_to_camera_rays(image_points)
                valid = camera_rays.valid_flag
                # All image points are in the FOV, except for a single image corner just beyond max_angle (by less
                # than a pixel): bottom-right for the front tele, top-right for the cross right camera
                width, height = (float(v) for v in params.resolution)
                is_front_tele = np.array_equal(params.principal_point, _ftheta_front_tele_params().principal_point)
                invalid_corner = image_points.new_tensor((width, height) if is_front_tele else (width, 0.0))
                expected_valid = (image_points != invalid_corner).any(dim=1)
                self.assertEqual(int((~expected_valid).sum()), 1)
                self.assertTrue(bool((valid == expected_valid).all()))

                reprojected = cam_model.camera_rays_to_image_points(camera_rays.camera_rays[valid]).image_points
                self.assertLessEqual(float((reprojected - image_points[valid]).abs().max()), tolerance)

    def test_unprojection_out_of_fov_finite(self):
        """Image points beyond the FOV unproject to finite, invalid rays with finite gradients, and image points within
        the FOV still reproject to themselves"""
        # Reduced FOV so that the image corners (at ~1.25 rad) are beyond the FOV, for both reference polynomials (the
        # backward polynomial of the cross-right camera turns negative at ~7100 px from the principal point)
        for reference_poly in (
            FThetaCameraModelParameters.PolynomialType.ANGLE_TO_PIXELDIST,
            FThetaCameraModelParameters.PolynomialType.PIXELDIST_TO_ANGLE,
        ):
            params = dataclasses.replace(_ftheta_cross_right_params(), max_angle=0.9, reference_poly=reference_poly)
            with self.subTest(reference_poly=reference_poly):
                cam_model = camera_model_from_parameters(params, device=self.device, dtype=self.dtype)
                width, height = (float(v) for v in params.resolution)

                # Image corners and radial sweeps from the principal point to far beyond the image, whose radial
                # angles reach beyond 90 degrees (forward reference) or below zero (backward reference)
                corners = np.array([[0.0, 0.0], [width, 0.0], [0.0, height], [width, height]])
                dists, azimuths = np.meshgrid(
                    np.linspace(0.0, 10000.0, 1001), np.linspace(0.0, 2 * np.pi, 16, endpoint=False)
                )
                sweeps = np.stack([dists * np.cos(azimuths), dists * np.sin(azimuths)], axis=-1).reshape(-1, 2) + (
                    params.principal_point.astype(np.float64) + 0.5
                )
                image_points = to_torch(np.concatenate([corners, sweeps]), device=self.device, dtype=self.dtype)
                image_points.requires_grad_(True)

                camera_rays = cam_model.image_points_to_camera_rays(image_points)
                camera_rays.camera_rays.sum().backward()

                self.assertTrue(bool(torch.isfinite(camera_rays.camera_rays).all()))
                self.assertTrue(bool(torch.isfinite(unpack_optional(image_points.grad)).all()))
                self.assertEqual(camera_rays.valid_flag[:4].cpu().tolist(), [False] * 4)

                # Valid rays within 6000 px of the principal point reproject to their image points (out-of-FOV image
                # points are not misclassified as valid there; further out, the polynomials themselves wrap around,
                # which purely radial models are equally subject to)
                within = np.concatenate([np.ones(4, dtype=bool), dists.reshape(-1) <= 6000.0])
                valid = camera_rays.valid_flag & to_torch(within, device=self.device)
                self.assertGreater(int(valid.sum()), 0)
                reprojected = cam_model.camera_rays_to_image_points(
                    camera_rays.camera_rays.detach()[valid]
                ).image_points
                tolerance = self._roundtrip_tolerance()
                self.assertLessEqual(float((reprojected - image_points.detach()[valid]).abs().max()), tolerance)

    def test_unprojection_principal_point(self):
        """The principal point unprojects exactly to the principal axis"""
        for params in self._roundtrip_params():
            with self.subTest(principal_point=params.principal_point.tolist(), reference_poly=params.reference_poly):
                cam_model = camera_model_from_parameters(params, device=self.device, dtype=self.dtype)
                camera_rays = cam_model.image_points_to_camera_rays(cam_model.principal_point[None, :])
                torch.testing.assert_close(
                    camera_rays.camera_rays,
                    camera_rays.camera_rays.new_tensor([[0.0, 0.0, 1.0]]),
                    rtol=0.0,
                    atol=0.0,
                )
                self.assertEqual(camera_rays.valid_flag.cpu().tolist(), [True])


@parameterized.parameterized_class(
    ("device", "dtype"), itertools.product(_get_test_devices(), (torch.float32, torch.float64))
)
class TestOpenCVPinholeZeroTerms(CameraModelsBaseTestCase):
    """Zero tangential / thin-prism terms of the OpenCV pinhole camera model are skipped"""

    device: torch.device
    dtype: torch.dtype

    def _opencv_pinhole_params(self) -> List[OpenCVPinholeCameraModelParameters]:
        params = [
            p
            for p in self.cam_model_params + self.cam_model_params_wsd
            if isinstance(p, OpenCVPinholeCameraModelParameters)
        ]
        # Radial-only variants of all fixtures
        params += [
            dataclasses.replace(
                p, tangential_coeffs=np.zeros(2, dtype=np.float32), thin_prism_coeffs=np.zeros(4, dtype=np.float32)
            )
            for p in params
        ]
        return params

    def _evaluate(self, model: CameraModel) -> Tuple[torch.Tensor, ...]:
        width, height = (float(v) for v in model.resolution.cpu())
        xs, ys = np.meshgrid(np.linspace(0.0, width, 25), np.linspace(0.0, height, 17))
        image_points = to_torch(
            np.stack([xs.reshape(-1), ys.reshape(-1)], axis=1), device=self.device, dtype=self.dtype
        )
        camera_rays = model.image_points_to_camera_rays(image_points)
        # Also project rays beyond the radial distortion limits
        rays = torch.cat(
            [camera_rays.camera_rays, camera_rays.camera_rays.new_tensor([[2.0, 1.0, 1.0], [-3.0, -2.0, 1.0]])]
        )
        projection = model.camera_rays_to_image_points(rays, return_jacobians=True)
        return (
            camera_rays.camera_rays,
            camera_rays.valid_flag,
            projection.image_points,
            projection.valid_flag,
            unpack_optional(projection.jacobians),
        )

    def test_flags(self):
        for params in self._opencv_pinhole_params():
            with self.subTest(params=params):
                model = camera_model_from_parameters(params, device=self.device, dtype=self.dtype)
                assert isinstance(model, OpenCVPinholeCameraModel)
                self.assertEqual(model._has_tangential_distortion, bool(np.any(params.tangential_coeffs)))
                self.assertEqual(model._has_thin_prism_distortion, bool(np.any(params.thin_prism_coeffs)))

    def test_zero_terms_bit_identical_to_full_evaluation(self):
        """Skipping zero terms gives exactly the outputs of evaluating all terms"""
        for params in self._opencv_pinhole_params():
            with self.subTest(params=params):
                fast = camera_model_from_parameters(params, device=self.device, dtype=self.dtype)
                full = camera_model_from_parameters(params, device=self.device, dtype=self.dtype)
                assert isinstance(full, OpenCVPinholeCameraModel)
                full._has_tangential_distortion = True
                full._has_thin_prism_distortion = True
                for actual, expected in zip(self._evaluate(fast), self._evaluate(full)):
                    self.assertTrue(torch.equal(actual, expected))

    def test_zero_terms_skip_coefficients(self):
        """Zero tangential / thin-prism coefficients are never read (NaN coefficients do not affect any output, which
        are exactly those of the unmodified model), non-zero ones are"""
        params_list = self._opencv_pinhole_params()
        # Cover zero and non-zero tangential and thin-prism terms
        self.assertTrue(any(np.any(p.tangential_coeffs) for p in params_list))
        self.assertTrue(any(not np.any(p.tangential_coeffs) for p in params_list))
        self.assertTrue(any(np.any(p.thin_prism_coeffs) for p in params_list))
        self.assertTrue(any(not np.any(p.thin_prism_coeffs) for p in params_list))

        for params in params_list:
            for name in ("tangential_coeffs", "thin_prism_coeffs"):
                with self.subTest(params=params, poisoned=name):
                    reference = camera_model_from_parameters(params, device=self.device, dtype=self.dtype)
                    # Model buffers may share memory with the parameters' arrays, so poison a copy only
                    poisoned = camera_model_from_parameters(copy.deepcopy(params), device=self.device, dtype=self.dtype)
                    getattr(poisoned, name).fill_(float("nan"))

                    expected_outputs = self._evaluate(reference)
                    actual_outputs = self._evaluate(poisoned)
                    if np.any(getattr(params, name)):
                        self.assertTrue(bool(torch.isnan(actual_outputs[0]).any()))
                    else:
                        for actual, expected in zip(actual_outputs, expected_outputs):
                            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


class TestCameraModelConventions(unittest.TestCase):
    """Buffers, code path flags and extension hooks shared by the camera models"""

    _ftheta = staticmethod(_ftheta_front_tele_params)

    @staticmethod
    def _opencv() -> OpenCVPinholeCameraModelParameters:
        return OpenCVPinholeCameraModelParameters(
            resolution=np.array([1920, 1280], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([958.3, 641.7], dtype=np.float32),
            focal_length=np.array([2061.5, 2060.25], dtype=np.float32),
            radial_coeffs=np.array([0.0424, -0.3417, 0.01, 0.02, -0.01, 0.02], dtype=np.float32),
            tangential_coeffs=np.array([0.0018, -0.00005], dtype=np.float32),
            thin_prism_coeffs=np.array([0.01, 0.02, 0.02, 0.01], dtype=np.float32),
        )

    def test_coefficient_buffers_are_persistent(self):
        """Coefficient buffers of all camera models are part of the state dict, and restore a model"""
        params_list: List[Union[FThetaCameraModelParameters, OpenCVPinholeCameraModelParameters]] = [
            self._ftheta(),
            self._opencv(),
        ]
        for params, names in zip(
            params_list,
            (
                ("fw_poly", "bw_poly", "A", "Ainv", "tangential_coeffs"),
                ("radial_coeffs", "tangential_coeffs", "thin_prism_coeffs"),
            ),
        ):
            with self.subTest(model=type(params).__name__):
                model = camera_model_from_parameters(params, device="cpu", dtype=torch.float64)
                state_dict = model.state_dict()
                for name in names:
                    self.assertIn(name, state_dict)
                    self.assertTrue(torch.equal(state_dict[name], getattr(model, name)))

                restored = camera_model_from_parameters(params, device="cpu", dtype=torch.float64)
                getattr(restored, "tangential_coeffs").mul_(2.0)
                restored.load_state_dict(state_dict)
                self.assertTrue(
                    torch.equal(getattr(restored, "tangential_coeffs"), getattr(model, "tangential_coeffs"))
                )

    def test_ftheta_loads_state_dicts_without_tangential_coeffs(self):
        """FTheta state dicts without tangential coefficients load strictly, keeping the constructed coefficients"""
        params = self._ftheta()
        model = camera_model_from_parameters(params, device="cpu", dtype=torch.float64)
        state_dict = model.state_dict()
        del state_dict["tangential_coeffs"]

        restored = camera_model_from_parameters(params, device="cpu", dtype=torch.float64)
        restored.load_state_dict(state_dict)
        np.testing.assert_array_equal(restored.get_parameters().tangential_coeffs, params.tangential_coeffs)
        # (the given state dict is not modified)
        self.assertNotIn("tangential_coeffs", state_dict)

    def test_code_path_flags(self):
        ftheta = camera_model_from_parameters(self._ftheta(), device="cpu")
        self.assertTrue(ftheta._has_tangential_distortion)
        opencv = camera_model_from_parameters(self._opencv(), device="cpu")
        self.assertTrue(opencv._has_tangential_distortion)
        self.assertTrue(opencv._has_thin_prism_distortion)
        for model in (ftheta, opencv):
            for name in ("_has_tangentials", "_has_tangential", "_has_thin_prism"):
                self.assertFalse(hasattr(model, name), name)

    def test_no_name_mangled_methods(self):
        for model_type in (
            CameraModel,
            FThetaCameraModel,
            IdealPinholeCameraModel,
            OpenCVPinholeCameraModel,
            OpenCVFisheyeCameraModel,
        ):
            with self.subTest(model=model_type.__name__):
                mangled = [name for name in vars(model_type) if name.startswith(f"_{model_type.__name__}__")]
                self.assertEqual(mangled, [])

    def test_ftheta_tangential_iterations(self):
        params = self._ftheta()
        self.assertEqual(FThetaCameraModel(params, "cpu").tangential_iterations, 4)
        with self.assertRaisesRegex(TypeError, "positional argument"):
            # (untyped: the positional call doesn't type check)
            cast(Any, FThetaCameraModel)(params, "cpu", torch.float64, 3, 1e-6, 2)
        with self.assertRaises(ValueError):
            FThetaCameraModel(params, "cpu", tangential_iterations=-1)

        # The unprojection inverts the tangential distortion by the given number of iterations
        reference = FThetaCameraModel(params, "cpu", torch.float64, tangential_iterations=20)
        rays = reference.image_points_to_camera_rays(np.array([[100.5, 80.5], [3700.5, 2100.5]]))
        errors = []
        for iterations in (0, 1, 4):
            model = FThetaCameraModel(params, "cpu", torch.float64, tangential_iterations=iterations)
            self.assertEqual(model.tangential_iterations, iterations)
            camera_rays = model.image_points_to_camera_rays(np.array([[100.5, 80.5], [3700.5, 2100.5]]))
            errors.append(float((camera_rays.camera_rays - rays.camera_rays).abs().max()))
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])

    def test_opencv_pinhole_distortion_hook(self):
        """Projection and undistortion evaluate the distortion through the overridable _compute_distortion"""

        class RadialOnlyModel(OpenCVPinholeCameraModel):
            def _compute_distortion(self, xy):
                icD, _, r_2 = super()._compute_distortion(xy)
                return icD, None, r_2

        params = self._opencv()
        radial_only = dataclasses.replace(
            params, tangential_coeffs=np.zeros(2, dtype=np.float32), thin_prism_coeffs=np.zeros(4, dtype=np.float32)
        )
        overridden = RadialOnlyModel(params, "cpu", torch.float64)
        expected = OpenCVPinholeCameraModel(radial_only, "cpu", torch.float64)

        rays = torch.tensor([[0.1, 0.05, 1.0], [-0.2, 0.15, 1.0]], dtype=torch.float64)
        image_points = overridden.camera_rays_to_image_points(rays).image_points
        torch.testing.assert_close(image_points, expected.camera_rays_to_image_points(rays).image_points)
        torch.testing.assert_close(
            overridden.image_points_to_camera_rays(image_points).camera_rays,
            expected.image_points_to_camera_rays(image_points).camera_rays,
        )

    def test_opencv_pinhole_undistort_hook(self):
        class CountingModel(OpenCVPinholeCameraModel):
            calls = 0

            def _iterative_undistort(self, image_points, **kwargs):
                CountingModel.calls += 1
                return super()._iterative_undistort(image_points, max_iterations=20)

        model = CountingModel(self._opencv(), "cpu", torch.float64)
        model.image_points_to_camera_rays(np.array([[100.0, 200.0]]))
        self.assertEqual(CountingModel.calls, 1)
        with self.assertRaisesRegex(TypeError, "positional argument"):
            # (untyped: the positional call doesn't type check)
            cast(Any, OpenCVPinholeCameraModel._iterative_undistort)(model, torch.zeros((1, 2)), 1e-12)

    def test_opencv_fisheye_inverse_max_angle_is_private(self):
        model = camera_model_from_parameters(TestIdealPinholeFromSource._fisheye(), device="cpu")
        self.assertFalse(hasattr(model, "inverse_max_angle"))
        self.assertGreater(model._inverse_max_angle, model.max_angle)

    def test_camera_rays_keyword(self):
        """The camera rays argument of the projections is named camera_rays"""
        model = camera_model_from_parameters(self._opencv(), device="cpu", dtype=torch.float64)
        rays = torch.tensor([[0.1, 0.05, 1.0]], dtype=torch.float64)
        expected = model.camera_rays_to_image_points(rays)
        torch.testing.assert_close(
            model.camera_rays_to_image_points(camera_rays=rays).image_points, expected.image_points
        )
        self.assertTrue(
            torch.equal(
                model.camera_rays_to_pixels(camera_rays=rays).pixels,
                model.image_points_to_pixels(expected.image_points),
            )
        )
        self.assertTrue(
            torch.equal(
                model._camera_rays_to_image_points_impl(camera_rays=rays, return_jacobians=False).image_points,
                expected.image_points,
            )
        )


class TestOpenCVPinholeMaxNormalizedRadiusSquared(unittest.TestCase):
    """Bound on the squared radius of normalized coordinates of the OpenCV pinhole radial distortion"""

    @staticmethod
    def _distorted_radius(radial_coeffs, r2: np.ndarray) -> np.ndarray:
        k1, k2, k3, k4, k5, k6 = radial_coeffs
        d = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
        return np.sqrt(r2) * d

    def _assert_first_extremum(self, radial_coeffs, max_r2: float) -> None:
        """The distorted radius is increasing below the bound, and decreasing just above it"""
        r2 = np.linspace(0.0, max_r2, 10001)
        self.assertTrue(np.all(np.diff(self._distorted_radius(radial_coeffs, r2)) > 0))
        above = self._distorted_radius(radial_coeffs, np.array([max_r2, max_r2 * 1.01]))
        self.assertLess(above[1], above[0])

    def test_radial_k1(self):
        # The distorted radius r (1 + k1 r^2) has its extremum at r^2 = -1 / (3 k1) (where d = 2/3)
        for k1 in (-0.05, -0.1, -0.3, -1.0):
            with self.subTest(k1=k1):
                coeffs = [k1, 0, 0, 0, 0, 0]
                max_r2 = OpenCVPinholeCameraModelParameters.compute_max_normalized_radius_squared(np.array(coeffs))
                self.assertAlmostEqual(max_r2, -1.0 / (3.0 * k1), places=12)
                self._assert_first_extremum(coeffs, max_r2)

    def test_monotonic_distortions(self):
        for coeffs in (
            [0, 0, 0, 0, 0, 0],
            [0.2, 0.05, 0.001, 0, 0, 0],  # pincushion
            [-0.1, 0.01, 0, 0, 0, 0],  # barrel, increasing again before an extremum
            [0.1, 0, 0, 0.05, 0, 0],
        ):
            with self.subTest(coeffs=coeffs):
                self.assertEqual(
                    OpenCVPinholeCameraModelParameters.compute_max_normalized_radius_squared(np.array(coeffs)), math.inf
                )

    def test_pole(self):
        # D(s) = 1 - 0.5 s has its root at s = 2 (with an increasing distorted radius before it)
        self.assertAlmostEqual(
            OpenCVPinholeCameraModelParameters.compute_max_normalized_radius_squared(np.array([0.1, 0, 0, -0.5, 0, 0])),
            2.0,
            places=12,
        )

    def test_rational_first_extremum(self):
        # The distorted radius of this rational distortion has a maximum, a minimum, and increases again beyond
        coeffs = [-0.3, 0.05, 0, 0.2, 0, 0]
        max_r2 = OpenCVPinholeCameraModelParameters.compute_max_normalized_radius_squared(np.array(coeffs))
        self._assert_first_extremum(coeffs, max_r2)
        r2 = np.linspace(max_r2, 20.0, 10001)
        self.assertTrue(np.any(np.diff(self._distorted_radius(coeffs, r2)) > 0))

    def test_double_root(self):
        # The derivative 1 + 3 k1 s + 5 k2 s^2 only touches zero at s = -3 k1 / (10 k2) for (3 k1)^2 = 20 k2 (its
        # double root is computed as a nearly real complex pair for the second coefficients)
        for k1, k2 in ((-0.2, 0.018), (-0.3, 0.0405)):
            with self.subTest(k1=k1, k2=k2):
                max_r2 = OpenCVPinholeCameraModelParameters.compute_max_normalized_radius_squared(
                    np.array([k1, k2, 0, 0, 0, 0])
                )
                self.assertAlmostEqual(max_r2, -3.0 * k1 / (10.0 * k2), places=5)

    def test_waymo(self):
        # Extremum of the radial distortion of the Waymo fixture
        coeffs = [0.04239636827428756, -0.34165672675852826, 0, 0, 0, 0]
        max_r2 = OpenCVPinholeCameraModelParameters.compute_max_normalized_radius_squared(np.array(coeffs))
        self._assert_first_extremum(coeffs, max_r2)


@parameterized.parameterized_class(
    ("device", "dtype"), itertools.product(_get_test_devices(), (torch.float32, torch.float64))
)
class TestOpenCVPinholeRadialValidity(CommonTestCase):
    """Validity of OpenCV pinhole projections / unprojections by the bound on the squared normalized radius"""

    device: str
    dtype: torch.dtype

    K1 = -0.1  # extremum of the distorted radius at r^2 = 10 / 3 (radial distortion factor 2/3)
    FOCAL_LENGTH = 500.0
    PRINCIPAL_POINT = (960.0, 640.0)

    def _params(self, k1: float = K1) -> OpenCVPinholeCameraModelParameters:
        return OpenCVPinholeCameraModelParameters(
            resolution=np.array([1920, 1280], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array(self.PRINCIPAL_POINT, dtype=np.float32),
            focal_length=np.array([self.FOCAL_LENGTH, self.FOCAL_LENGTH], dtype=np.float32),
            radial_coeffs=np.array([k1, 0, 0, 0, 0, 0], dtype=np.float32),
            tangential_coeffs=np.zeros(2, dtype=np.float32),
            thin_prism_coeffs=np.zeros(4, dtype=np.float32),
        )

    def _rays(self, r2s) -> torch.Tensor:
        # Rays along the image's diagonal direction with the given squared normalized radii
        direction = np.array([0.8, 0.6])
        return torch.tensor([[*(direction * math.sqrt(r2)), 1.0] for r2 in r2s], device=self.device, dtype=self.dtype)

    def test_derived_bound(self):
        params = self._params()
        model = OpenCVPinholeCameraModel(params, device=self.device, dtype=self.dtype)
        expected_max_r2 = OpenCVPinholeCameraModelParameters.compute_max_normalized_radius_squared(params.radial_coeffs)
        self.assertEqual(model.max_normalized_radius_squared, expected_max_r2)
        self.assertAlmostEqual(expected_max_r2, 10.0 / 3.0, places=5)
        # (the generic factory derives the bound)
        from_parameters = camera_model_from_parameters(params, device=self.device, dtype=self.dtype)
        assert isinstance(from_parameters, OpenCVPinholeCameraModel)
        self.assertEqual(from_parameters.max_normalized_radius_squared, expected_max_r2)

    def test_projection_validity(self):
        model = OpenCVPinholeCameraModel(self._params(), device=self.device, dtype=self.dtype)
        max_r2 = model.max_normalized_radius_squared
        # Within the image: r^2 = 2.0 (radial distortion factor 0.8, the distorted point at 566 px from the principal
        # point), 0.95 max_r2 (factor 0.68, which the former fixed limits (0.8, 1.2) rejected), and 1.05 max_r2 (beyond
        # the extremum; the point would still project into the image, but not one-to-one)
        projection = model.camera_rays_to_image_points(self._rays([2.0, 0.95 * max_r2, 1.05 * max_r2]))
        self.assertEqual(projection.valid_flag.tolist(), [True, True, False])

        # Valid rays project with the distortion model
        for index, r2 in enumerate([2.0, 0.95 * max_r2]):
            radius = math.sqrt(r2) * (1 + self.K1 * r2) * self.FOCAL_LENGTH
            expected = torch.tensor(
                [self.PRINCIPAL_POINT[0] + 0.8 * radius, self.PRINCIPAL_POINT[1] + 0.6 * radius],
                device=self.device,
                dtype=self.dtype,
            )
            torch.testing.assert_close(projection.image_points[index], expected, rtol=0.0, atol=1e-3)

        # The ray beyond the extremum is moved out of the image
        self.assertGreater(
            float(
                torch.linalg.norm(projection.image_points[2] - projection.image_points.new_tensor(self.PRINCIPAL_POINT))
            ),
            1000.0,
        )

    def test_unprojection_validity(self):
        model = OpenCVPinholeCameraModel(self._params(), device=self.device, dtype=self.dtype)
        bounded = OpenCVPinholeCameraModel(
            self._params(), device=self.device, dtype=self.dtype, max_normalized_radius_squared=0.75
        )
        # (rays where the fixed-point undistortion converges, which is slow towards the extremum)
        for r2 in (0.5, 1.0):
            with self.subTest(r2=r2):
                ray = self._rays([r2])
                image_point = model.camera_rays_to_image_points(ray).image_points
                unprojection = model.image_points_to_camera_rays(image_point)
                self.assertTrue(bool(unprojection.valid_flag[0]))
                torch.testing.assert_close(
                    unprojection.camera_rays, ray / torch.linalg.norm(ray, dim=1, keepdim=True), rtol=0.0, atol=1e-5
                )
                # A bound below the ray's radius invalidates its unprojection (whose undistortion converged)
                self.assertEqual(bool(bounded.image_points_to_camera_rays(image_point).valid_flag[0]), r2 < 0.75)

    def test_explicit_bound(self):
        params = self._params()
        max_r2 = OpenCVPinholeCameraModelParameters.compute_max_normalized_radius_squared(params.radial_coeffs)
        rays = self._rays([1.0, 2.0, 1.05 * max_r2])
        for bound, expected_valid in (
            (1.5, [True, False, False]),  # stricter than the derived bound
            (1.1 * max_r2, [True, True, True]),  # beyond the extremum
            (math.inf, [True, True, True]),
        ):
            with self.subTest(bound=bound):
                model = OpenCVPinholeCameraModel(
                    params, device=self.device, dtype=self.dtype, max_normalized_radius_squared=bound
                )
                self.assertEqual(model.max_normalized_radius_squared, bound)
                self.assertEqual(model.camera_rays_to_image_points(rays).valid_flag.tolist(), expected_valid)

    def test_explicit_bound_of_distortion_free_model(self):
        """Distortion-free models project like ideal pinholes, irrespective of the bound"""
        params = self._params(k1=0.0)
        bounded = OpenCVPinholeCameraModel(
            params, device=self.device, dtype=self.dtype, max_normalized_radius_squared=1.5
        )
        unbounded = OpenCVPinholeCameraModel(params, device=self.device, dtype=self.dtype)
        self.assertEqual(unbounded.max_normalized_radius_squared, math.inf)

        rays = self._rays([1.0, 2.0])
        bounded_projection = bounded.camera_rays_to_image_points(rays)
        self.assertEqual(bounded_projection.valid_flag.tolist(), [True, True])
        self.assertTrue(
            torch.equal(bounded_projection.image_points, unbounded.camera_rays_to_image_points(rays).image_points)
        )
        bounded_rays = bounded.image_points_to_camera_rays(bounded_projection.image_points)
        self.assertEqual(bounded_rays.valid_flag.tolist(), [True, True])
        self.assertTrue(
            torch.equal(
                bounded_rays.camera_rays,
                unbounded.image_points_to_camera_rays(bounded_projection.image_points).camera_rays,
            )
        )

    def test_invalid_bound(self):
        for bound in (0.0, -1.0, math.nan):
            with self.subTest(bound=bound):
                with self.assertRaises(ValueError):
                    OpenCVPinholeCameraModel(
                        self._params(), device=self.device, dtype=self.dtype, max_normalized_radius_squared=bound
                    )


@parameterized.parameterized_class(("device",), [(device,) for device in _get_test_devices()])
class TestOpenCVPinholeUndistortionPeriphery(CameraModelsBaseTestCase):
    """Float32 unprojections of image points up to the image corners are valid and accurate (also unprojected
    together with points closer to the principal point)"""

    device: torch.device

    def _image_points(self, resolution: np.ndarray) -> np.ndarray:
        # Grid over the full image, including its borders and corners
        width, height = (float(v) for v in resolution)
        xs, ys = np.meshgrid(np.linspace(0.0, width, 49), np.linspace(0.0, height, 33))
        return np.stack([xs.reshape(-1), ys.reshape(-1)], axis=1)

    def test_float32_periphery_rays_converge(self):
        params_list = [p for p in self.cam_model_params if isinstance(p, OpenCVPinholeCameraModelParameters)]
        self.assertGreater(len(params_list), 0)
        for params in params_list:
            with self.subTest(params=params):
                model32 = camera_model_from_parameters(params, device=self.device, dtype=torch.float32)
                model64 = camera_model_from_parameters(params, device=self.device, dtype=torch.float64)
                image_points = self._image_points(params.resolution)

                reference = model64.image_points_to_camera_rays(image_points)
                self.assertTrue(bool(reference.valid_flag.all()))

                # (one by one, and all at once: points converging faster than others don't stop the iteration)
                for image_point, reference_ray in zip(image_points, reference.camera_rays):
                    rays = model32.image_points_to_camera_rays(image_point[None])
                    self.assertTrue(bool(rays.valid_flag[0]), f"invalid ray for image point {image_point}")
                    torch.testing.assert_close(
                        rays.camera_rays[0].to(torch.float64), reference_ray, rtol=0.0, atol=1e-6
                    )
                rays = model32.image_points_to_camera_rays(image_points)
                self.assertTrue(bool(rays.valid_flag.all()))
                torch.testing.assert_close(
                    rays.camera_rays.to(torch.float64), reference.camera_rays, rtol=0.0, atol=1e-6
                )


class TestCameraModelConstructorValidation(unittest.TestCase):
    """Constructor validation (on CPU, so it runs without a GPU)"""

    device = torch.device("cpu")

    def test_min_2d_norm_must_be_positive(self):
        ftheta_parameters = ftheta_parameters_from_reference(
            ReferenceFThetaCamera(np.array([1000, 1000]), [10, 10], [0, np.radians(90) / 1000])
        )
        fisheye_parameters = OpenCVFisheyeCameraModelParameters(
            resolution=np.array([640, 480], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([320.0, 240.0], dtype=np.float32),
            focal_length=np.array([300.0, 300.0], dtype=np.float32),
            radial_coeffs=np.zeros(4, dtype=np.float32),
            max_angle=np.deg2rad(60),
        )
        for min_2d_norm in (0.0, -1.0, float("nan")):
            with self.subTest(min_2d_norm=min_2d_norm):
                with self.assertRaisesRegex(ValueError, "min_2d_norm must be positive"):
                    FThetaCameraModel(ftheta_parameters, self.device, min_2d_norm=min_2d_norm)
                with self.assertRaisesRegex(ValueError, "min_2d_norm must be positive"):
                    OpenCVFisheyeCameraModel(fisheye_parameters, self.device, min_2d_norm=min_2d_norm)

    def test_newton_iterations_must_be_non_negative(self):
        ftheta_parameters = ftheta_parameters_from_reference(
            ReferenceFThetaCamera(np.array([1000, 1000]), [10, 10], [0, np.radians(90) / 1000])
        )
        fisheye_parameters = OpenCVFisheyeCameraModelParameters(
            resolution=np.array([640, 480], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([320.0, 240.0], dtype=np.float32),
            focal_length=np.array([300.0, 300.0], dtype=np.float32),
            radial_coeffs=np.zeros(4, dtype=np.float32),
            max_angle=np.deg2rad(60),
        )
        with self.assertRaisesRegex(ValueError, "newton_iterations must be non-negative, got -1"):
            FThetaCameraModel(ftheta_parameters, self.device, newton_iterations=-1)
        with self.assertRaisesRegex(ValueError, "newton_iterations must be non-negative, got -1"):
            OpenCVFisheyeCameraModel(fisheye_parameters, self.device, newton_iterations=-1)
        # (zero iterations are valid)
        self.assertEqual(FThetaCameraModel(ftheta_parameters, self.device, newton_iterations=0).newton_iterations, 0)

    def test_reused_camera_rays_are_validated(self):
        """Camera rays passed to the world-ray unprojections and projections must match the image points / rays"""
        model = IdealPinholeCameraModel(
            IdealPinholeCameraModelParameters(
                resolution=np.array([640, 480], dtype=np.uint64),
                shutter_type=ShutterType.GLOBAL,
                principal_point=np.array([320.0, 240.0], dtype=np.float32),
                focal_length=np.array([500.0, 500.0], dtype=np.float32),
            ),
            self.device,
        )
        image_points = torch.tensor([[320.0, 240.0], [100.0, 100.0]])
        rays = model.image_points_to_camera_rays(image_points)
        eye = np.eye(4, dtype=np.float32)
        for camera_rays, message in (
            (
                cast(CameraModel.CameraRaysReturn, rays.camera_rays),
                "camera_rays must be a CameraModel.CameraRaysReturn",
            ),
            (
                CameraModel.CameraRaysReturn(camera_rays=rays.camera_rays[:1], valid_flag=rays.valid_flag[:1]),
                r"camera_rays must have shape \(2, 3\) \(one ray per image point\), got shape \(1, 3\)",
            ),
            (
                CameraModel.CameraRaysReturn(camera_rays=rays.camera_rays, valid_flag=rays.valid_flag[:1]),
                r"The valid flag of camera_rays must have shape \(2,\), got shape \(1,\)",
            ),
            (
                CameraModel.CameraRaysReturn(camera_rays=rays.camera_rays, valid_flag=rays.valid_flag.int()),
                "The valid flag of camera_rays must be of dtype torch.bool",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    model.image_points_to_world_rays_static_pose(image_points, eye, camera_rays=camera_rays)
        with self.assertRaisesRegex(
            ValueError, r"The valid flag of camera_rays must have shape \(2,\) \(one flag per ray\), got shape \(1,\)"
        ):
            model.camera_rays_to_image_points(
                CameraModel.CameraRaysReturn(camera_rays=rays.camera_rays, valid_flag=rays.valid_flag[:1])
            )

    def test_to_torch_rejects_unsigned_overflow(self):
        for dtype, signed in ((np.uint16, np.int16), (np.uint32, np.int32), (np.uint64, np.int64)):
            with self.subTest(dtype=dtype.__name__):
                in_range = np.array([0, np.iinfo(signed).max], dtype=dtype)
                self.assertEqual(to_torch(in_range, self.device).tolist(), [0, np.iinfo(signed).max])
                with self.assertRaisesRegex(ValueError, "exceeds max range"):
                    to_torch(np.array([np.iinfo(signed).max + 1], dtype=dtype), self.device)

    def test_invalid_shutter_type_is_rejected(self):
        parameters = IdealPinholeCameraModelParameters(
            resolution=np.array([640, 480], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([320.0, 240.0], dtype=np.float32),
            focal_length=np.array([500.0, 500.0], dtype=np.float32),
        )
        parameters.shutter_type = cast(ShutterType, 42)
        with self.assertRaisesRegex(ValueError, "Unsupported shutter type 42"):
            IdealPinholeCameraModel(parameters, self.device)


@parameterized.parameterized_class(
    ("device", "dtype"), itertools.product(_get_test_devices(), (torch.float32, torch.float64))
)
class TestExternalDistortion(CommonTestCase):
    device: str
    dtype: torch.dtype

    def test_from_parameters(self):
        # Verify that, when provided BivariateWindshieldModelParameters, a BivariateWindshieldModel object is returned
        horizontal_poly = np.zeros((3), dtype=np.float32)
        vertical_poly = np.zeros_like(horizontal_poly)
        horizontal_poly_inverse = np.zeros_like(horizontal_poly)
        vertical_poly_inverse = np.zeros_like(horizontal_poly)
        windshield_params = BivariateWindshieldModelParameters(
            ReferencePolynomial.FORWARD,
            horizontal_poly,
            vertical_poly,
            horizontal_poly_inverse,
            vertical_poly_inverse,
        )
        res_ws = external_distortion_model_from_parameters(windshield_params, self.device, self.dtype)
        self.assertTrue(isinstance(res_ws, BivariateWindshieldModel))


@parameterized.parameterized_class(
    ("device", "dtype"), itertools.product(_get_test_devices(), (torch.float32, torch.float64))
)
class TestBivariateWindshieldModel(CommonTestCase):
    device: torch.device
    dtype: torch.dtype

    def test_init(self):
        """Tests initialization of BivariateWindshieldModel"""

        horizontal_poly = np.zeros((3), dtype=np.float32)
        vertical_poly = np.zeros_like(horizontal_poly)
        horizontal_poly_inverse = np.zeros_like(horizontal_poly)
        vertical_poly_inverse = np.zeros_like(horizontal_poly)
        windshield_params = BivariateWindshieldModelParameters(
            ReferencePolynomial.FORWARD,
            horizontal_poly,
            vertical_poly,
            horizontal_poly_inverse,
            vertical_poly_inverse,
        )
        windshield_distortion = BivariateWindshieldModel(windshield_params, self.device, self.dtype)
        self.assertTrue(isinstance(windshield_distortion, BivariateWindshieldModel))

    def test_poly_eval_2d(self):
        """Tests evaluation of 2d polynomials"""

        coeffs = torch.zeros(3, dtype=self.dtype, device=self.device)
        x = torch.tensor([-1.0, 2.0, 3.0], dtype=self.dtype, device=self.device)
        y = torch.tensor([-2.0, 1.0, 3.0, 5.0], dtype=self.dtype, device=self.device)
        with self.assertRaises(ValueError) as _:
            BivariateWindshieldModel.poly_eval_2d(coeffs, x, y, order=1)

        coeffs = torch.zeros(3, dtype=self.dtype, device=self.device)
        x = torch.tensor([-1.0, 2.0, 3.0, 4.0], dtype=self.dtype, device=self.device)
        y = torch.tensor([-2.0, 1.0, 3.0, 5.0], dtype=self.dtype, device=self.device)
        res = BivariateWindshieldModel.poly_eval_2d(coeffs, x, y, order=1)
        expected_value = torch.zeros_like(x)
        torch.testing.assert_close(res, expected_value)

        # Oracle test
        coeffs = torch.tensor(
            [0.90113, 0.77499, 0.55887, 0.77048, 0.47019, 0.84775, 0.68832, 0.77690, 0.92327, 0.83983],
            dtype=self.dtype,
            device=self.device,
        )
        x = torch.tensor([1.2, 1.2], dtype=self.dtype, device=self.device)
        y = torch.tensor([0.4, 0.4], dtype=self.dtype, device=self.device)
        res = BivariateWindshieldModel.poly_eval_2d(coeffs, x, y, order=3)
        expected_value = torch.tensor([5.31406952, 5.31406952], dtype=self.dtype, device=self.device)
        torch.testing.assert_close(res, expected_value)

    def test_distort_rays(self):
        """Tests distortion of rays using a bivariate polynomial"""

        # Create a polynomial evaluation function that always returns sqrt(2)/2. This way, we expect to
        # see the sqrt(2)/2 in both x and y outputs, and 0 in z
        def poly_eval_func(coeffs, x, y, _):
            return torch.asin(np.sqrt(2.0) / 2.0 * torch.ones_like(x))

        horizontal_poly = torch.tensor([-1.0, 2.0, 3.0], dtype=self.dtype, device=self.device)
        vertical_poly = torch.tensor([1.0, 3.0, 6.0], dtype=self.dtype, device=self.device)
        order = 1
        rays = torch.tensor([[-1.0, 2.0, 3.0], [1.0, 3.0, 6.0]], dtype=self.dtype, device=self.device)
        res = BivariateWindshieldModel.distort_rays(rays, horizontal_poly, vertical_poly, order, order, poly_eval_func)
        expected_value = np.sqrt(2.0) / 2.0 * torch.ones_like(rays)
        expected_value[:, 2] = 0
        torch.testing.assert_close(res, expected_value, rtol=1e-4, atol=1e-3)

    def test_distort_camera_rays(self):
        """Tests distortion / undistortion using full WSD model"""

        rt2_2 = np.sqrt(2.0) / 2.0
        horizontal_poly = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        vertical_poly = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        rays = torch.tensor([[rt2_2, rt2_2, 0.0], [-rt2_2, rt2_2, 0.0]], dtype=self.dtype, device=self.device)

        windshield_params = BivariateWindshieldModelParameters(
            ReferencePolynomial.FORWARD,
            horizontal_poly,
            vertical_poly,
            horizontal_poly,
            vertical_poly,
        )
        windshield_distortion = BivariateWindshieldModel(windshield_params, self.device, self.dtype)
        res = windshield_distortion.distort_camera_rays(rays)
        expected_value = rays.clone()
        expected_value[:, :2] *= -1.0
        torch.testing.assert_close(res, expected_value)

        # Expect distort and undistort to give the same results when provided with the same coefficients
        windshield_params = BivariateWindshieldModelParameters(
            ReferencePolynomial.FORWARD,
            horizontal_poly,
            vertical_poly,
            horizontal_poly,
            vertical_poly,
        )
        windshield_distortion = BivariateWindshieldModel(windshield_params, self.device, self.dtype)
        res_w = windshield_distortion.undistort_camera_rays(rays)
        torch.testing.assert_close(res, res_w)

        windshield_horizontal_polynomial = np.array(
            [
                -0.000475919834570959,
                0.99944007396698,
                0.000166745347087272,
                0.000205887947231531,
                0.0055195577442646,
                0.000861024134792387,
            ],
            dtype=np.float32,
        )
        windshield_vertical_polynomial = np.array(
            [
                0.00152770057320595,
                -0.000532537756953388,
                -5.65027039556298e-05,
                -4.02410341848736e-06,
                0.000608163303695619,
                1.0094313621521,
                -0.00125278066843748,
                0.00823396816849709,
                -0.000293767458060756,
                0.0185473654419184,
                -0.003074218519032,
                0.00599765172228217,
                0.0172030478715897,
                -0.00364979170262814,
                0.0069147446192801,
            ],
            dtype=np.float32,
        )
        windshield_horizontal_polynomial_inv = np.array(
            [0.0004770369, 1.0005774, -0.00016896478, -0.00020207358, -0.0054899976, -0.0008536868], dtype=np.float32
        )
        windshield_vertical_polynomial_inv = np.array(
            [
                -0.0015191488,
                0.00052959577,
                7.882431e-05,
                -6.966009e-06,
                -0.00059701066,
                0.9906775,
                0.00116782,
                -0.007893825,
                0.00026140467,
                -0.017767625,
                0.0027627628,
                -0.00544897,
                -0.015480865,
                0.0033684247,
                -0.0057964055,
            ],
            dtype=np.float32,
        )
        r = 1.0
        phi = 0.05
        theta = 0.02
        rays = torch.nn.functional.normalize(
            torch.tensor(
                [
                    [0.0, 0.0, 1.0],
                    [r * np.sin(phi) * np.cos(theta), r * np.sin(phi) * np.sin(theta), r * np.cos(theta)],
                    [r * np.sin(phi) * np.cos(theta), r * np.sin(phi) * np.sin(theta), -r * np.cos(theta)],
                ],
                dtype=self.dtype,
                device=self.device,
            ),
            dim=-1,
        )
        windshield_params = BivariateWindshieldModelParameters(
            ReferencePolynomial.FORWARD,
            windshield_horizontal_polynomial,
            windshield_vertical_polynomial,
            windshield_horizontal_polynomial_inv,
            windshield_vertical_polynomial_inv,
        )
        windshield_distortion = BivariateWindshieldModel(windshield_params, self.device, self.dtype)
        res = windshield_distortion.distort_camera_rays(rays)
        res_w = windshield_distortion.undistort_camera_rays(res)
        torch.testing.assert_close(res_w, rays, rtol=1e-4, atol=1e-5)


class TestOpenCVFisheyeMaxAngle(unittest.TestCase):
    """Tests for OpenCVFisheyeCameraModelParameters.compute_max_angle."""

    # Shared camera intrinsics (real-world ScanNet++ DSLR values)
    RESOLUTION = np.array([1752, 1168], dtype=np.uint64)
    FOCAL_LENGTH = np.array([789.28, 789.46], dtype=np.float32)
    PRINCIPAL_POINT = np.array([883.03, 581.78], dtype=np.float32)
    RADIAL_COEFFS = np.array([-0.0542, 0.0301, -0.0229, 0.0064], dtype=np.float32)

    def test_compute_max_angle_basic(self):
        """compute_max_angle returns a plausible angle for typical fisheye intrinsics."""
        angle = OpenCVFisheyeCameraModelParameters.compute_max_angle(
            self.RESOLUTION, self.FOCAL_LENGTH, self.PRINCIPAL_POINT, self.RADIAL_COEFFS
        )
        # Should be a reasonable fisheye half-FOV (roughly 60-100 degrees)
        self.assertGreater(angle, np.deg2rad(60))
        self.assertLess(angle, np.deg2rad(100))

    def test_compute_max_angle_zero_distortion(self):
        """With zero distortion the model is equidistant: r = f * theta, so theta = r/f."""
        resolution = np.array([2000, 2000], dtype=np.uint64)
        focal_length = np.array([1000.0, 1000.0], dtype=np.float32)
        principal_point = np.array([1000.0, 1000.0], dtype=np.float32)  # centred
        radial_coeffs = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        angle = OpenCVFisheyeCameraModelParameters.compute_max_angle(
            resolution, focal_length, principal_point, radial_coeffs
        )
        # Farthest corner is at (0,0) or (2000,2000), distance = sqrt(1^2 + 1^2) = sqrt(2)
        expected = np.sqrt(2.0)  # normalised distance = corner distance / f = 1000/1000 * sqrt(2)
        self.assertAlmostEqual(angle, expected, places=5)

    def test_compute_max_angle_forward_inverse_consistency(self):
        """The computed angle, when passed through the forward polynomial, should give the max corner distance."""
        angle = OpenCVFisheyeCameraModelParameters.compute_max_angle(
            self.RESOLUTION, self.FOCAL_LENGTH, self.PRINCIPAL_POINT, self.RADIAL_COEFFS
        )

        # Forward polynomial: r(theta) = theta * (1 + k1*t^2 + k2*t^4 + k3*t^6 + k4*t^8)
        k1, k2, k3, k4 = self.RADIAL_COEFFS.astype(np.float64)
        t2 = angle**2
        r_forward = angle * (1.0 + k1 * t2 + k2 * t2**2 + k3 * t2**3 + k4 * t2**4)

        # Max normalised corner distance
        corners = np.array([[0, 0], [1752, 0], [0, 1168], [1752, 1168]], dtype=np.float64)
        normalised = (corners - self.PRINCIPAL_POINT.astype(np.float64)) / self.FOCAL_LENGTH.astype(np.float64)
        max_r = float(np.max(np.linalg.norm(normalised, axis=1)))

        self.assertAlmostEqual(r_forward, max_r, places=6)

    def test_explicit_max_angle_preserved(self):
        """An explicitly provided max_angle is stored as-is on the parameters."""
        explicit = 1.234
        params = OpenCVFisheyeCameraModelParameters(
            resolution=self.RESOLUTION.copy(),
            shutter_type=ShutterType.GLOBAL,
            principal_point=self.PRINCIPAL_POINT.copy(),
            focal_length=self.FOCAL_LENGTH.copy(),
            radial_coeffs=self.RADIAL_COEFFS.copy(),
            external_distortion_parameters=None,
            max_angle=explicit,
        )
        self.assertAlmostEqual(params.max_angle, explicit, places=6)

    def test_json_round_trip(self):
        """Serialise and deserialise with a computed max_angle; the value should survive the round-trip."""
        max_angle = OpenCVFisheyeCameraModelParameters.compute_max_angle(
            self.RESOLUTION, self.FOCAL_LENGTH, self.PRINCIPAL_POINT, self.RADIAL_COEFFS
        )
        original = OpenCVFisheyeCameraModelParameters(
            resolution=self.RESOLUTION.copy(),
            shutter_type=ShutterType.GLOBAL,
            principal_point=self.PRINCIPAL_POINT.copy(),
            focal_length=self.FOCAL_LENGTH.copy(),
            radial_coeffs=self.RADIAL_COEFFS.copy(),
            external_distortion_parameters=None,
            max_angle=max_angle,
        )
        json_dict = original.to_dict()
        restored = OpenCVFisheyeCameraModelParameters.from_dict(json_dict)
        self.assertAlmostEqual(restored.max_angle, max_angle, places=5)

    def test_asymmetric_principal_point(self):
        """max_angle picks the farthest corner, not the nearest."""
        # Principal point near top-left corner -> farthest corner is bottom-right
        angle_tl = OpenCVFisheyeCameraModelParameters.compute_max_angle(
            self.RESOLUTION, self.FOCAL_LENGTH, np.array([100.0, 100.0], dtype=np.float32), self.RADIAL_COEFFS
        )
        # Principal point near centre
        angle_c = OpenCVFisheyeCameraModelParameters.compute_max_angle(
            self.RESOLUTION, self.FOCAL_LENGTH, np.array([876.0, 584.0], dtype=np.float32), self.RADIAL_COEFFS
        )
        # Off-centre principal point should give a larger max_angle
        self.assertGreater(angle_tl, angle_c)


class TestOpenCVFisheyeMaxAngleMonotonicity(unittest.TestCase):
    """Tests for the monotonicity-aware compute_max_angle on OpenCVFisheyeCameraModel."""

    def test_non_monotone_polynomial_gives_reasonable_angle(self):
        """The reported bug case: corners outside FOV should not produce angle > pi."""
        # These intrinsics have a polynomial that folds before reaching the image corners
        max_angle = OpenCVFisheyeCameraModelParameters.compute_max_angle(
            np.array([1920, 1536], dtype=np.uint64),
            np.array([449.191, 448.985], dtype=np.float32),
            np.array([959.917, 767.244], dtype=np.float32),
            np.array([0.145375, -0.0673299, 0.0201295, -0.00245998], dtype=np.float32),
        )
        # The old implementation returned 7.87 rad which is nonsense (> 2*pi)
        # The angle must be less than pi (180 degrees) for any physical camera
        self.assertLess(max_angle, np.pi)
        # Should be a reasonable fisheye FOV (at least 45 degrees)
        self.assertGreater(max_angle, np.deg2rad(45))

    def test_non_monotone_polynomial_derivative_positive(self):
        """The returned angle should be within the monotone region of the polynomial."""
        radial_coeffs = np.array([0.145375, -0.0673299, 0.0201295, -0.00245998], dtype=np.float32)
        max_angle = OpenCVFisheyeCameraModelParameters.compute_max_angle(
            np.array([1920, 1536], dtype=np.uint64),
            np.array([449.191, 448.985], dtype=np.float32),
            np.array([959.917, 767.244], dtype=np.float32),
            radial_coeffs,
        )
        # Build derivative polynomial for OpenCV fisheye:
        # r(theta) = theta*(1 + k1*t^2 + k2*t^4 + k3*t^6 + k4*t^8)
        # r'(theta) = 1 + 3*k1*t^2 + 5*k2*t^4 + 7*k3*t^6 + 9*k4*t^8
        k1, k2, k3, k4 = radial_coeffs.astype(np.float64)
        dfw_coeffs = np.array([1.0, 0.0, 3.0 * k1, 0.0, 5.0 * k2, 0.0, 7.0 * k3, 0.0, 9.0 * k4])
        d_poly = Polynomial(dfw_coeffs)

        # Verify derivative is non-negative at the returned angle (tolerance for floating point)
        self.assertGreaterEqual(float(d_poly(max_angle)), -1e-10)

    def test_well_behaved_polynomial_still_works(self):
        """ScanNet++ intrinsics (well-behaved) should still give a plausible angle."""
        angle = OpenCVFisheyeCameraModelParameters.compute_max_angle(
            np.array([1752, 1168], dtype=np.uint64),
            np.array([789.28, 789.46], dtype=np.float32),
            np.array([883.03, 581.78], dtype=np.float32),
            np.array([-0.0542, 0.0301, -0.0229, 0.0064], dtype=np.float32),
        )
        # Should still be in the 60-100 degree range (same as existing test)
        self.assertGreater(angle, np.deg2rad(60))
        self.assertLess(angle, np.deg2rad(100))

    def test_monotonicity_guarantee_multiple_cameras(self):
        """For various coefficient sets, verify monotonicity up to the returned angle."""
        test_cases = [
            # (resolution, focal, pp, radial_coeffs)
            (
                np.array([1920, 1536], dtype=np.uint64),
                np.array([449.191, 448.985], dtype=np.float32),
                np.array([959.917, 767.244], dtype=np.float32),
                np.array([0.145375, -0.0673299, 0.0201295, -0.00245998], dtype=np.float32),
            ),
            (
                np.array([1752, 1168], dtype=np.uint64),
                np.array([789.28, 789.46], dtype=np.float32),
                np.array([883.03, 581.78], dtype=np.float32),
                np.array([-0.0542, 0.0301, -0.0229, 0.0064], dtype=np.float32),
            ),
            (
                np.array([2000, 2000], dtype=np.uint64),
                np.array([1000.0, 1000.0], dtype=np.float32),
                np.array([1000.0, 1000.0], dtype=np.float32),
                np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ),
        ]

        for resolution, focal, pp, radial in test_cases:
            max_angle = OpenCVFisheyeCameraModelParameters.compute_max_angle(resolution, focal, pp, radial)
            k1, k2, k3, k4 = radial.astype(np.float64)
            dfw_coeffs = np.array([1.0, 0.0, 3.0 * k1, 0.0, 5.0 * k2, 0.0, 7.0 * k3, 0.0, 9.0 * k4])
            d_poly = Polynomial(dfw_coeffs)
            # Sample derivative at many points up to max_angle
            thetas = np.linspace(0, max_angle, 200)
            for t in thetas:
                self.assertGreaterEqual(
                    float(d_poly(t)), -1e-10, f"Derivative negative at theta={t:.4f} for radial={radial}"
                )


@parameterized.parameterized_class(
    ("device", "dtype"), itertools.product(_get_test_devices(), (torch.float32, torch.float64))
)
class TestIdealPinholeCamera(CommonTestCase):
    """Tests for the ideal (distortion-free) pinhole camera model"""

    device: torch.device
    dtype: torch.dtype

    def _make_ideal(self) -> IdealPinholeCameraModel:
        params = IdealPinholeCameraModelParameters(
            resolution=np.array([640, 480], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([320.0, 240.0], dtype=np.float32),
            focal_length=np.array([500.0, 510.0], dtype=np.float32),
        )
        model = camera_model_from_parameters(params, device=self.device, dtype=self.dtype)
        assert isinstance(model, IdealPinholeCameraModel)
        return model

    def _make_zero_coeff_opencv(self) -> OpenCVPinholeCameraModel:
        params = OpenCVPinholeCameraModelParameters(
            resolution=np.array([640, 480], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([320.0, 240.0], dtype=np.float32),
            focal_length=np.array([500.0, 510.0], dtype=np.float32),
            radial_coeffs=np.zeros(6, dtype=np.float32),
            tangential_coeffs=np.zeros(2, dtype=np.float32),
            thin_prism_coeffs=np.zeros(4, dtype=np.float32),
        )
        model = camera_model_from_parameters(params, device=self.device, dtype=self.dtype)
        assert isinstance(model, OpenCVPinholeCameraModel)
        return model

    def test_opencv_reference(self):
        """Validates the ideal (distortion-free) pinhole projection against ``cv2.projectPoints``"""

        np_dtype = np.float64 if self.dtype == torch.float64 else np.float32
        model = self._make_ideal()
        params = model.get_parameters()

        K = np.array(
            [
                [params.focal_length[0], 0, params.principal_point[0]],
                [0, params.focal_length[1], params.principal_point[1]],
                [0, 0, 1],
            ],
            dtype=np_dtype,
        )
        # An ideal pinhole has no distortion
        distortion = np.zeros(5, dtype=np_dtype)
        rvec = np.zeros(3, dtype=np_dtype)
        tvec = np.zeros(3, dtype=np_dtype)

        for x in np.linspace(-0.4, 0.4, num=7):
            for y in np.linspace(-0.3, 0.3, num=5):
                with self.subTest(x=x, y=y):
                    ray = np.array([[x, y, 1.0]], dtype=np_dtype)
                    ours = model.camera_rays_to_image_points(to_torch(ray, device=model.device, dtype=model.dtype))
                    reference, _ = cv2.projectPoints(ray.reshape(1, 1, 3), rvec, tvec, K, distortion)
                    reference = reference.reshape(1, 2)
                    self.assertLessEqual(np.linalg.norm(reference - np.array(ours.image_points.cpu())).item(), 0.01)

    def test_dispatch_and_type(self):
        model = self._make_ideal()
        self.assertEqual(IdealPinholeCameraModelParameters.type(), "ideal-pinhole")
        self.assertIsInstance(model, IdealPinholeCameraModel)

    def test_opencv_is_not_ideal_instance(self):
        # An OpenCV pinhole must NOT be an instance of the ideal pinhole params/model
        self.assertFalse(issubclass(OpenCVPinholeCameraModelParameters, IdealPinholeCameraModelParameters))
        self.assertFalse(issubclass(OpenCVPinholeCameraModel, IdealPinholeCameraModel))

    def test_roundtrip(self):
        model = self._make_ideal()
        pixels = torch.tensor([[100, 80], [320, 240], [500, 400]], dtype=torch.int32, device=self.device)
        rays = model.pixels_to_camera_rays(pixels)
        self.assertTrue(bool(rays.valid_flag.all()))
        result = model.camera_rays_to_pixels(rays)
        self.assertTrue(bool(result.valid_flag.all()))
        self._compareVector(result.pixels.cpu().numpy(), pixels.cpu().numpy())

    def test_parity_with_zero_coeff_opencv(self):
        ideal = self._make_ideal()
        opencv = self._make_zero_coeff_opencv()

        image_points = ideal.pixels_to_image_points(
            torch.tensor([[100, 80], [320, 240], [500, 400]], dtype=torch.int32, device=self.device)
        )
        rays_ideal = ideal.image_points_to_camera_rays(image_points)
        rays_opencv = opencv.image_points_to_camera_rays(image_points)
        self.assertTrue(bool(rays_ideal.valid_flag.all()) and bool(rays_opencv.valid_flag.all()))
        self.assertIsNone(
            np.testing.assert_array_almost_equal(
                rays_ideal.camera_rays.cpu().numpy(), rays_opencv.camera_rays.cpu().numpy(), decimal=5
            )
        )

        proj_ideal = ideal.camera_rays_to_image_points(rays_ideal)
        proj_opencv = opencv.camera_rays_to_image_points(rays_opencv)
        self.assertIsNone(
            np.testing.assert_array_almost_equal(
                proj_ideal.image_points.cpu().numpy(), proj_opencv.image_points.cpu().numpy(), decimal=4
            )
        )

    def test_distortion_free_flag(self):
        self.assertTrue(self._make_zero_coeff_opencv()._is_distortion_free)

        distorted = OpenCVPinholeCameraModelParameters(
            resolution=np.array([640, 480], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([320.0, 240.0], dtype=np.float32),
            focal_length=np.array([500.0, 500.0], dtype=np.float32),
            radial_coeffs=np.array([-0.2, 0.05, 0, 0, 0, 0], dtype=np.float32),
            tangential_coeffs=np.zeros(2, dtype=np.float32),
            thin_prism_coeffs=np.zeros(4, dtype=np.float32),
        )
        self.assertFalse(distorted.is_distortion_free)
        model = camera_model_from_parameters(distorted, device=self.device, dtype=self.dtype)
        assert isinstance(model, OpenCVPinholeCameraModel)
        self.assertFalse(model._is_distortion_free)

    def test_jacobian_path(self):
        model = self._make_ideal()
        rays = model.pixels_to_camera_rays(
            torch.tensor([[100, 80], [320, 240]], dtype=torch.int32, device=self.device)
        ).camera_rays
        result = model.camera_rays_to_image_points(rays, return_jacobians=True)
        jacobians = result.jacobians
        self.assertIsNotNone(jacobians)
        assert jacobians is not None
        self.assertEqual(tuple(jacobians.shape), (2, 2, 3))


class TestIdealPinholeFromSource(unittest.TestCase):
    """Tests for IdealPinholeCameraModelParameters.from_source() / natural_fov() / fov()"""

    @staticmethod
    def _ideal(focal=(500.0, 500.0), resolution=(640, 480), principal_point=(320.0, 240.0)):
        return IdealPinholeCameraModelParameters(
            resolution=np.array(resolution, dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array(principal_point, dtype=np.float32),
            focal_length=np.array(focal, dtype=np.float32),
        )

    @staticmethod
    def _opencv(focal=(400.0, 410.0)):
        return OpenCVPinholeCameraModelParameters(
            resolution=np.array([640, 480], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([300.0, 250.0], dtype=np.float32),
            focal_length=np.array(focal, dtype=np.float32),
            radial_coeffs=np.array([-0.2, 0.05, 0, 0, 0, 0], dtype=np.float32),
            tangential_coeffs=np.zeros(2, dtype=np.float32),
            thin_prism_coeffs=np.zeros(4, dtype=np.float32),
        )

    @staticmethod
    def _fisheye(focal=(300.0, 300.0), max_angle_deg=70.0):
        return OpenCVFisheyeCameraModelParameters(
            resolution=np.array([640, 480], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([320.0, 240.0], dtype=np.float32),
            focal_length=np.array(focal, dtype=np.float32),
            radial_coeffs=np.zeros(4, dtype=np.float32),
            max_angle=np.radians(max_angle_deg),
        )

    @staticmethod
    def _ftheta(c1=1250.0, c=1.0, max_angle_deg=45.0, principal_point=(960.0, 540.0)):
        return FThetaCameraModelParameters(
            resolution=np.array([1920, 1080], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array(principal_point, dtype=np.float32),
            reference_poly=FThetaCameraModelParameters.PolynomialType.ANGLE_TO_PIXELDIST,
            pixeldist_to_angle_poly=np.array([0.0, 0.0008, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            angle_to_pixeldist_poly=np.array([0.0, c1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            max_angle=np.radians(max_angle_deg),
            linear_cde=np.array([c, 0.0, 0.0], dtype=np.float32),
        )

    def test_default_ideal_identity(self):
        params = self._ideal()
        ideal = IdealPinholeCameraModelParameters.from_source(params)
        self.assertEqual(ideal.type(), "ideal-pinhole")
        np.testing.assert_allclose(ideal.focal_length, params.focal_length, rtol=1e-5)
        np.testing.assert_array_equal(ideal.principal_point, params.principal_point)
        np.testing.assert_array_equal(ideal.resolution, params.resolution)

    def test_default_opencv_drops_distortion_preserves_anisotropy(self):
        params = self._opencv(focal=(400.0, 410.0))
        ideal = IdealPinholeCameraModelParameters.from_source(params)
        self.assertIsInstance(ideal, IdealPinholeCameraModelParameters)
        # focal (incl. anisotropic fx != fy) preserved, distortion dropped
        np.testing.assert_allclose(ideal.focal_length, params.focal_length, rtol=1e-5)
        np.testing.assert_array_equal(ideal.principal_point, params.principal_point)

    def test_default_fisheye_uses_focal(self):
        params = self._fisheye(focal=(300.0, 300.0))
        ideal = IdealPinholeCameraModelParameters.from_source(params)
        np.testing.assert_allclose(ideal.focal_length, params.focal_length, rtol=1e-5)

    def test_ftheta_focal_from_forward_poly_with_linear_term(self):
        # focal = [c1 * c, c1] (forward poly first-order coeff, x scaled by linear term c)
        params = self._ftheta(c1=1250.0, c=1.002)
        ideal = IdealPinholeCameraModelParameters.from_source(params)
        self.assertAlmostEqual(float(ideal.focal_length[0]), 1250.0 * 1.002, places=2)
        self.assertAlmostEqual(float(ideal.focal_length[1]), 1250.0, places=2)

    def test_ftheta_non_positive_focal_raises(self):
        params = self._ftheta(c1=0.0)
        with self.assertRaises(ValueError):
            IdealPinholeCameraModelParameters.from_source(params)

    @parameterized.parameterized.expand(_get_test_devices())
    def test_ftheta_principal_point_pixel_center_shift(self, device: torch.device):
        # F-Theta stores the principal point in the pixel-center convention; the ideal
        # pinhole uses the image-coordinate (top-left) convention, so from_source must
        # apply a +0.5 shift. Verify both the value and that the principal ray projects
        # to the same image point through the source and the ideal models.
        params = self._ftheta(principal_point=(959.5, 539.5))
        ideal = IdealPinholeCameraModelParameters.from_source(params)
        np.testing.assert_allclose(ideal.principal_point, np.array([960.0, 540.0], dtype=np.float32), atol=1e-4)

        source_model = camera_model_from_parameters(params, device=device, dtype=torch.float64)
        ideal_model = camera_model_from_parameters(ideal, device=device, dtype=torch.float64)
        principal_ray = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        src_pt = source_model.camera_rays_to_image_points(principal_ray).image_points[0].cpu().numpy()
        ideal_pt = ideal_model.camera_rays_to_image_points(principal_ray).image_points[0].cpu().numpy()
        np.testing.assert_allclose(src_pt, ideal_pt, atol=1e-3)

    def test_natural_fov_and_roundtrip(self):
        params = self._opencv(focal=(400.0, 410.0))
        nat = IdealPinholeCameraModelParameters.natural_fov(params)
        self.assertEqual(nat.shape, (2,))

        # fov() of the default-converted model equals natural_fov(source)
        ideal = IdealPinholeCameraModelParameters.from_source(params)
        np.testing.assert_allclose(ideal.fov(), nat, rtol=1e-5)

        # passing natural_fov back as a per-axis target reproduces the default
        roundtrip = IdealPinholeCameraModelParameters.from_source(params, target_fov=nat)
        np.testing.assert_allclose(roundtrip.focal_length, ideal.focal_length, rtol=1e-4)

    def test_scalar_target_fov_narrow_and_widen(self):
        params = self._ideal(focal=(500.0, 500.0))
        default_focal = float(IdealPinholeCameraModelParameters.from_source(params).focal_length[0])

        narrow = IdealPinholeCameraModelParameters.from_source(params, target_fov=np.radians(40.0))
        wide = IdealPinholeCameraModelParameters.from_source(params, target_fov=np.radians(120.0))

        # narrower FOV -> larger focal; wider FOV -> smaller focal
        self.assertGreater(float(narrow.focal_length[0]), default_focal)
        self.assertLess(float(wide.focal_length[0]), default_focal)

    def test_scalar_target_fov_preserves_focal_aspect_ratio(self):
        # Anisotropic source + non-square resolution: a scalar target_fov must scale the
        # focal isotropically, preserving fx:fy (no squashing).
        params = self._ideal(focal=(900.0, 950.0), resolution=(1920, 1080), principal_point=(960.0, 540.0))
        out = IdealPinholeCameraModelParameters.from_source(params, target_fov=np.radians(60.0))
        self.assertAlmostEqual(float(out.focal_length[0]) / float(out.focal_length[1]), 900.0 / 950.0, places=4)
        # The binding (tighter) axis lands at the requested FOV, the other stays within it
        half = IdealPinholeCameraModelParameters._max_corner_half_extent(out.resolution, out.principal_point)
        fov = 2.0 * np.arctan2(half, out.focal_length)
        self.assertLessEqual(float(fov.max()), np.radians(60.0) + 1e-4)
        self.assertAlmostEqual(float(fov.max()), np.radians(60.0), places=3)

    def test_per_axis_target_fov(self):
        params = self._ideal(focal=(500.0, 500.0), resolution=(1920, 1080), principal_point=(960.0, 540.0))
        target = np.array([np.radians(90.0), np.radians(50.0)], dtype=np.float64)
        out = IdealPinholeCameraModelParameters.from_source(params, target_fov=target)
        half = IdealPinholeCameraModelParameters._max_corner_half_extent(out.resolution, out.principal_point)
        fov = 2.0 * np.arctan2(half, out.focal_length)
        np.testing.assert_allclose(fov, target, atol=1e-4)

    def test_wide_fisheye_default_succeeds_as_central_crop(self):
        # A wide F-Theta fisheye yields a valid (always < 180deg) pinhole by default: the
        # paraxial focal maps a narrow rectilinear central window. A wider rectilinear
        # view is obtained with an explicit target_fov.
        params = self._ftheta(c1=490.0, max_angle_deg=120.0)
        params = dataclasses.replace(
            params,
            resolution=np.array([1920, 1536], dtype=np.uint64),
            principal_point=np.array([960.0, 768.0], dtype=np.float32),
        )
        default = IdealPinholeCameraModelParameters.from_source(params)
        self.assertIsInstance(default, IdealPinholeCameraModelParameters)
        # default focal == paraxial focal [c1, c1] (c == 1 here)
        np.testing.assert_allclose(default.focal_length, np.array([490.0, 490.0], dtype=np.float32), rtol=1e-5)
        # any pinhole FOV is strictly below 180 degrees
        self.assertTrue(np.all(default.fov() < np.pi))

        # widening to an explicit target FOV reduces the focal
        wide = IdealPinholeCameraModelParameters.from_source(params, target_fov=np.radians(150.0))
        self.assertLess(float(wide.focal_length[0]), float(default.focal_length[0]))

    def test_invalid_target_fov_raises(self):
        params = self._ideal()
        for bad in (0.0, -1.0, np.pi, 4.0):
            with self.assertRaises(ValueError):
                IdealPinholeCameraModelParameters.from_source(params, target_fov=bad)
        with self.assertRaises(ValueError):
            IdealPinholeCameraModelParameters.from_source(params, target_fov=np.array([1.0, 1.0, 1.0]))

    def test_abstract_typed_source_is_accepted(self):
        """A source held under the abstract base is usable, without narrowing at the call site.

        This is the contract the signatures promise: callers holding a `CameraModelParameters`,
        which is what the decoders and the `model_parameters` accessors hand out, can reach these
        helpers directly. Annotating the local is the point of the test, so keep it.
        """
        for concrete in (
            self._ideal(),
            self._opencv(),
            self._fisheye(),
            self._ftheta(),
        ):
            with self.subTest(model=type(concrete).__name__):
                source: CameraModelParameters = concrete

                fov = IdealPinholeCameraModelParameters.natural_fov(source)
                self.assertEqual(fov.shape, (2,))
                self.assertTrue(np.all(fov > 0.0))

                ideal = IdealPinholeCameraModelParameters.from_source(source)
                self.assertIsInstance(ideal, IdealPinholeCameraModelParameters)
                np.testing.assert_array_equal(ideal.resolution, concrete.resolution)

    def test_out_of_tree_model_opts_in_by_implementing_the_geometry(self):
        """An out-of-tree model becomes usable by `from_source` / `natural_fov` via one method.

        This is the point of the extension point: nothing in this repository knows about the class
        below, yet both helpers work on it because it declares its own paraxial pinhole.
        """

        @dataclass
        class _OptedInCameraModelParameters(CameraModelParameters):
            """An out-of-tree model that declares its paraxial pinhole"""

            focal: float = 800.0

            @staticmethod
            def type() -> str:
                return "opted-in-test-model"

            def transform(
                self,
                image_domain_scale: Union[float, Tuple[float, float]],
                image_domain_offset: Tuple[float, float] = (0.0, 0.0),
                new_resolution: Optional[Tuple[int, int]] = None,
            ) -> Self:
                return self

            def paraxial_pinhole_geometry(self) -> ParaxialPinholeGeometry:
                return ParaxialPinholeGeometry(
                    np.array([self.focal, self.focal], dtype=np.float32),
                    np.array([320.0, 240.0], dtype=np.float32),
                    self.resolution,
                )

        source = _OptedInCameraModelParameters(
            resolution=np.array((640, 480), dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
        )

        ideal = IdealPinholeCameraModelParameters.from_source(source)
        self.assertIsInstance(ideal, IdealPinholeCameraModelParameters)
        np.testing.assert_allclose(ideal.focal_length, np.array([800.0, 800.0], dtype=np.float32))
        np.testing.assert_array_equal(ideal.resolution, source.resolution)

        # and the derived quantities follow from it, with no further per-model knowledge
        fov = IdealPinholeCameraModelParameters.natural_fov(source)
        self.assertEqual(fov.shape, (2,))
        self.assertTrue(np.all(fov > 0.0))

    def test_out_of_tree_model_may_opt_out_by_raising(self):
        """A model with no meaningful paraxial pinhole opts out, and both helpers surface that.

        The method is abstract, so a model cannot inherit a silent refusal: declining is a
        deliberate `TypeError` the implementation writes itself, which then travels out through
        `from_source` and `natural_fov` unchanged.
        """

        @dataclass
        class _NoPinholeCameraModelParameters(CameraModelParameters):
            """An out-of-tree model that has no paraxial pinhole and says so"""

            @staticmethod
            def type() -> str:
                return "no-pinhole-test-model"

            def transform(
                self,
                image_domain_scale: Union[float, Tuple[float, float]],
                image_domain_offset: Tuple[float, float] = (0.0, 0.0),
                new_resolution: Optional[Tuple[int, int]] = None,
            ) -> Self:
                return self

            def paraxial_pinhole_geometry(self) -> ParaxialPinholeGeometry:
                raise TypeError(f"{type(self).__name__} has no paraxial pinhole")

        source = _NoPinholeCameraModelParameters(
            resolution=np.array((640, 480), dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
        )

        with self.assertRaises(TypeError) as ctx:
            IdealPinholeCameraModelParameters.from_source(source)
        self.assertIn("_NoPinholeCameraModelParameters", str(ctx.exception))

        with self.assertRaises(TypeError):
            IdealPinholeCameraModelParameters.natural_fov(source)

    def test_model_omitting_the_geometry_cannot_be_instantiated(self):
        """Omitting it is not a way to decline: the class is abstract and fails at construction.

        This is what makes the requirement real rather than advisory, and it is the reason adding
        the method is a breaking change for pre-existing out-of-tree subclasses.
        """

        @dataclass
        class _IncompleteCameraModelParameters(CameraModelParameters):
            @staticmethod
            def type() -> str:
                return "incomplete-test-model"

            def transform(
                self,
                image_domain_scale: Union[float, Tuple[float, float]],
                image_domain_offset: Tuple[float, float] = (0.0, 0.0),
                new_resolution: Optional[Tuple[int, int]] = None,
            ) -> Self:
                return self

        with self.assertRaises(TypeError) as ctx:
            _IncompleteCameraModelParameters(
                resolution=np.array((640, 480), dtype=np.uint64),
                shutter_type=ShutterType.GLOBAL,
            )
        self.assertIn("paraxial_pinhole_geometry", str(ctx.exception))

    def test_paraxial_geometry_matches_the_in_tree_models(self):
        """The geometry each in-tree model reports is the one `from_source` then builds from."""
        for concrete in (self._ideal(), self._opencv(), self._fisheye(), self._ftheta()):
            with self.subTest(model=type(concrete).__name__):
                geometry = concrete.paraxial_pinhole_geometry()
                ideal = IdealPinholeCameraModelParameters.from_source(concrete)

                np.testing.assert_allclose(ideal.focal_length, geometry.focal_length, rtol=1e-6)
                np.testing.assert_allclose(ideal.principal_point, geometry.principal_point, rtol=1e-6)
                np.testing.assert_array_equal(ideal.resolution, geometry.resolution)


class TestIdealPinholeParameterIO(unittest.TestCase):
    """Serialization round-trip for the ideal pinhole parameters"""

    def test_encode_decode(self):
        params = IdealPinholeCameraModelParameters(
            resolution=np.array([640, 480], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([320.0, 240.0], dtype=np.float32),
            focal_length=np.array([500.0, 500.0], dtype=np.float32),
        )
        encoded = encode_camera_model_parameters(params)
        self.assertEqual(encoded["camera_model_type"], "ideal-pinhole")
        decoded = decode_camera_model_parameters(encoded)
        self.assertIsInstance(decoded, IdealPinholeCameraModelParameters)
        assert isinstance(decoded, IdealPinholeCameraModelParameters)
        np.testing.assert_array_equal(decoded.resolution, params.resolution)
        np.testing.assert_array_equal(decoded.focal_length, params.focal_length)
        np.testing.assert_array_equal(decoded.principal_point, params.principal_point)

    def test_transform(self):
        params = IdealPinholeCameraModelParameters(
            resolution=np.array([640, 480], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([320.0, 240.0], dtype=np.float32),
            focal_length=np.array([500.0, 500.0], dtype=np.float32),
        )
        transformed = params.transform(0.5)
        np.testing.assert_array_equal(transformed.resolution, np.array([320, 240], dtype=np.uint64))
        np.testing.assert_array_equal(transformed.focal_length, np.array([250.0, 250.0], dtype=np.float32))
        np.testing.assert_array_equal(transformed.principal_point, np.array([160.0, 120.0], dtype=np.float32))


class TestCameraModelFactory(unittest.TestCase):
    """Tests for the free camera_model_from_parameters() factory and its open registry"""

    # Reuse the parameter builders of the from_source test case
    _ideal = staticmethod(TestIdealPinholeFromSource._ideal)
    _opencv = staticmethod(TestIdealPinholeFromSource._opencv)
    _fisheye = staticmethod(TestIdealPinholeFromSource._fisheye)
    _ftheta = staticmethod(TestIdealPinholeFromSource._ftheta)

    def test_dispatches_to_concrete_model(self):
        for params, expected in (
            (self._ftheta(), FThetaCameraModel),
            (self._ideal(), IdealPinholeCameraModel),
            (self._opencv(), OpenCVPinholeCameraModel),
            (self._fisheye(), OpenCVFisheyeCameraModel),
        ):
            with self.subTest(model=expected.__name__):
                model = camera_model_from_parameters(params, device="cpu")
                self.assertIsInstance(model, expected)
                # The parameters round-trip back out through the abstract interface
                self.assertEqual(model.get_parameters().type(), params.type())

    def test_static_factory_removed(self):
        self.assertFalse(hasattr(CameraModel, "from_parameters"))

    def test_forwards_model_kwargs(self):
        """Keyword arguments after dtype reach the concrete model's constructor"""
        model = camera_model_from_parameters(self._ftheta(), device="cpu", newton_iterations=7, tangential_iterations=2)
        self.assertEqual((model.newton_iterations, model.tangential_iterations), (7, 2))
        model = camera_model_from_parameters(self._fisheye(), device="cpu", newton_iterations=9, min_2d_norm=1e-4)
        self.assertEqual(model.newton_iterations, 9)
        self.assertAlmostEqual(float(model.min_2d_norm), 1e-4)
        model = camera_model_from_parameters(self._opencv(), device="cpu", max_normalized_radius_squared=0.5)
        self.assertEqual(model.max_normalized_radius_squared, 0.5)

    def test_rejects_unknown_model_kwargs(self):
        for params in (self._ftheta(), self._ideal(), self._opencv(), self._fisheye()):
            with self.subTest(params=type(params).__name__):
                with self.assertRaisesRegex(TypeError, "unexpected keyword argument 'not_an_argument'"):
                    camera_model_from_parameters(params, device="cpu", not_an_argument=1)

    def test_unregistered_parameters_raise(self):
        not_a_camera = cast(ConcreteCameraModelParametersUnion, object())
        with self.assertRaises(TypeError):
            camera_model_from_parameters(not_a_camera, device="cpu")

    def test_out_of_tree_registration(self):
        @dataclasses.dataclass
        class CustomCameraModelParameters(IdealPinholeCameraModelParameters):
            @staticmethod
            def type() -> str:
                return "custom"

        class CustomCameraModel(IdealPinholeCameraModel):
            pass

        # Without a registration, dispatch falls back to the base's factory along the MRO
        params = CustomCameraModelParameters(
            resolution=np.array([640, 480], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([320.0, 240.0], dtype=np.float32),
            focal_length=np.array([500.0, 500.0], dtype=np.float32),
        )
        self.assertIsInstance(camera_model_from_parameters(params, device="cpu"), IdealPinholeCameraModel)

        received_kwargs: List[dict] = []

        @register_camera_model
        def _(
            cam_model_parameters: CustomCameraModelParameters,
            device: Union[str, torch.device] = torch.device("cuda"),
            dtype: torch.dtype = torch.float32,
            **model_kwargs,
        ) -> CustomCameraModel:
            received_kwargs.append(model_kwargs)
            return CustomCameraModel(cam_model_parameters, device, dtype)

        # The registry is process-global and singledispatch offers no unregister; keying it on a
        # method-local parameter type keeps the registration unreachable from other tests
        self.assertIsInstance(camera_model_from_parameters(params, device="cpu"), CustomCameraModel)
        # (keyword arguments reach registered factories)
        camera_model_from_parameters(params, device="cpu", custom_option=3)
        self.assertEqual(received_kwargs, [{}, {"custom_option": 3}])

    def test_camera_model_parameters_type_is_covariant(self):
        # A heterogeneous collection of concrete camera models must still have `CameraModel` as a
        # common static supertype. With an invariant parameter type it would not: a type checker
        # joining the element types would fall back past `CameraModel` to its own bases, and every
        # call to a `CameraModel` method on the joined type would be an error. This assignment only
        # type-checks while `CameraModelParametersT_co` stays covariant, and the calls below keep
        # the check honest at runtime too.
        models: List[CameraModel[CameraModelParameters]] = [
            camera_model_from_parameters(self._ftheta(), device="cpu"),
            camera_model_from_parameters(self._ideal(), device="cpu"),
            camera_model_from_parameters(self._opencv(), device="cpu"),
            camera_model_from_parameters(self._fisheye(), device="cpu"),
        ]
        for model in models:
            self.assertEqual(tuple(model.resolution.shape), (2,))
            self.assertIsInstance(model.get_parameters(), CameraModelParameters)

    def test_abstract_base_declares_serialization(self):
        # Parameters can be serialized through the abstract type without narrowing
        def encode(parameters: CameraModelParameters) -> dict:
            return {"camera_model_type": parameters.type(), "camera_model_parameters": parameters.to_dict()}

        self.assertEqual(encode(self._ideal())["camera_model_type"], "ideal-pinhole")

        # ... but the base itself has no identifier of its own, and requires one from every
        # concrete subclass rather than defaulting
        self.assertTrue(inspect.isabstract(CameraModelParameters))
        self.assertIn("type", CameraModelParameters.__abstractmethods__)


class TestExternalDistortionModelFactory(unittest.TestCase):
    """Tests for the free external_distortion_model_from_parameters() factory and its registry"""

    @staticmethod
    def _windshield() -> BivariateWindshieldModelParameters:
        return BivariateWindshieldModelParameters(
            reference_poly=ReferencePolynomial.FORWARD,
            horizontal_poly=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            vertical_poly=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            horizontal_poly_inverse=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            vertical_poly_inverse=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )

    @classmethod
    def _camera_with_distortion(cls) -> IdealPinholeCameraModelParameters:
        return IdealPinholeCameraModelParameters(
            resolution=np.array([640, 480], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([320.0, 240.0], dtype=np.float32),
            focal_length=np.array([500.0, 500.0], dtype=np.float32),
            external_distortion_parameters=cls._windshield(),
        )

    def test_dispatches_to_concrete_model(self):
        parameters = self._windshield()
        model = external_distortion_model_from_parameters(parameters, device="cpu")
        self.assertIsInstance(model, BivariateWindshieldModel)
        # The parameters round-trip back out through the abstract interface
        self.assertEqual(model.get_parameters().type(), parameters.type())

    def test_static_factory_removed(self):
        self.assertFalse(hasattr(ExternalDistortionModel, "from_parameters"))

    def test_rejects_unknown_model_kwargs(self):
        with self.assertRaisesRegex(TypeError, "unexpected keyword argument 'not_an_argument'"):
            external_distortion_model_from_parameters(self._windshield(), device="cpu", not_an_argument=1)

    def test_unregistered_parameters_raise(self):
        with self.assertRaises(TypeError):
            external_distortion_model_from_parameters(cast(ExternalDistortionParameters, object()), device="cpu")

    def test_out_of_tree_registration(self):
        @dataclasses.dataclass
        class CustomDistortionParameters(BivariateWindshieldModelParameters):
            @staticmethod
            def type() -> str:
                return "custom"

        class CustomDistortionModel(BivariateWindshieldModel):
            pass

        parameters = CustomDistortionParameters(**dataclasses.asdict(self._windshield()))

        # Without a registration, dispatch falls back to the base's factory along the MRO
        self.assertIsInstance(
            external_distortion_model_from_parameters(parameters, device="cpu"), BivariateWindshieldModel
        )

        @register_external_distortion_model
        def _(
            external_distortion_parameters: CustomDistortionParameters,
            device: Union[str, torch.device] = torch.device("cuda"),
            dtype: torch.dtype = torch.float32,
        ) -> CustomDistortionModel:
            return CustomDistortionModel(external_distortion_parameters, device, dtype)

        # The registry is process-global and singledispatch offers no unregister; keying it on a
        # method-local parameter type keeps the registration unreachable from other tests
        self.assertIsInstance(
            external_distortion_model_from_parameters(parameters, device="cpu"), CustomDistortionModel
        )

    def test_abstract_base_declares_serialization(self):
        # Parameters can be serialized through the abstract type without narrowing
        def encode(parameters: ExternalDistortionParameters) -> dict:
            return {"external_distortion_type": parameters.type(), "parameters": parameters.to_dict()}

        self.assertEqual(encode(self._windshield())["external_distortion_type"], "bivariate-windshield")

        # ... but the base itself has no identifier of its own, and requires one from every
        # concrete subclass rather than defaulting
        self.assertTrue(inspect.isabstract(ExternalDistortionParameters))
        self.assertIn("type", ExternalDistortionParameters.__abstractmethods__)

    def test_from_dict_reconstructs_the_concrete_distortion_type(self):
        # `dataclasses_json` constructs whatever type the field is annotated with, so with the
        # field declared against the abstract base this only works because the serialized form
        # carries the concrete type (EXTERNAL_DISTORTION_TYPE_KEY) and the field's decoder
        # dispatches on it. Without that, the nested dict deserializes into the base itself and the
        # failure only surfaces later, where the value is used.
        camera = IdealPinholeCameraModelParameters(
            resolution=np.array([640, 480], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([320.0, 240.0], dtype=np.float32),
            focal_length=np.array([500.0, 500.0], dtype=np.float32),
            external_distortion_parameters=self._windshield(),
        )
        restored = IdealPinholeCameraModelParameters.from_dict(camera.to_dict())
        self.assertIsInstance(restored.external_distortion_parameters, BivariateWindshieldModelParameters)
        self.assertEqual(restored.to_json(), camera.to_json())

    def test_serialized_form_carries_the_concrete_type(self):
        # The discriminator has to sit inside the nested object: that is what a field declared
        # against the abstract base has to dispatch on when reconstructing.
        camera = self._camera_with_distortion()
        nested: Dict = dict(cast(Dict, camera.to_dict()["external_distortion_parameters"]))
        self.assertEqual(nested[EXTERNAL_DISTORTION_TYPE_KEY], "bivariate-windshield")

    def test_untagged_legacy_payload_still_decodes(self):
        # Payloads written before the nested type key exist in stored data; they are unambiguous
        # because the bivariate windshield was the only concrete type at the time.
        camera = self._camera_with_distortion()
        encoded: Dict = dict(camera.to_dict())
        legacy: Dict = dict(cast(Dict, encoded["external_distortion_parameters"]))
        legacy.pop(EXTERNAL_DISTORTION_TYPE_KEY)
        encoded["external_distortion_parameters"] = legacy
        restored = IdealPinholeCameraModelParameters.from_dict(encoded)
        self.assertIsInstance(restored.external_distortion_parameters, BivariateWindshieldModelParameters)
        self.assertEqual(restored.to_json(), camera.to_json())

    def test_legacy_camera_level_type_is_migrated_by_the_decoder(self):
        # The old encoded layout stored the type beside the camera parameters, with an untagged
        # nested object. decode_camera_model_parameters has to move it inside.
        camera = self._camera_with_distortion()
        encoded = encode_camera_model_parameters(camera)
        camera_parameters: Dict = dict(cast(Dict, encoded["camera_model_parameters"]))
        nested: Dict = dict(cast(Dict, camera_parameters["external_distortion_parameters"]))
        nested.pop(EXTERNAL_DISTORTION_TYPE_KEY)
        camera_parameters["external_distortion_parameters"] = nested
        legacy: Dict = dict(encoded)
        legacy["camera_model_parameters"] = camera_parameters
        decoded = decode_camera_model_parameters(legacy)
        self.assertIsInstance(decoded.external_distortion_parameters, BivariateWindshieldModelParameters)

    def test_decode_accepts_old_and_new_payload_structures(self):
        # The two serialized layouts must decode to the same parameters:
        #
        #   old: the type sits beside the camera parameters, the nested object is untagged
        #   new: the type sits inside the nested object (both are written, for older readers)
        #
        # Stored data contains the old layout, so this equivalence is what lets the field be
        # declared against the abstract base without a migration.
        camera = self._camera_with_distortion()
        new_payload = encode_camera_model_parameters(camera)

        camera_parameters: Dict = dict(cast(Dict, new_payload["camera_model_parameters"]))
        nested: Dict = dict(cast(Dict, camera_parameters["external_distortion_parameters"]))
        self.assertIn(EXTERNAL_DISTORTION_TYPE_KEY, nested)
        nested.pop(EXTERNAL_DISTORTION_TYPE_KEY)
        camera_parameters["external_distortion_parameters"] = nested
        old_payload: Dict = dict(new_payload)
        old_payload["camera_model_parameters"] = camera_parameters

        decoded_old = decode_camera_model_parameters(old_payload)
        decoded_new = decode_camera_model_parameters(new_payload)

        for decoded in (decoded_old, decoded_new):
            self.assertIsInstance(decoded.external_distortion_parameters, BivariateWindshieldModelParameters)
        self.assertEqual(decoded_old.to_json(), decoded_new.to_json())
        self.assertEqual(decoded_old.to_json(), camera.to_json())

    def test_unknown_serialized_type_raises(self):
        camera = self._camera_with_distortion()
        encoded: Dict = dict(camera.to_dict())
        nested: Dict = dict(cast(Dict, encoded["external_distortion_parameters"]))
        nested[EXTERNAL_DISTORTION_TYPE_KEY] = "not-a-real-distortion"
        encoded["external_distortion_parameters"] = nested
        with self.assertRaises(ValueError):
            IdealPinholeCameraModelParameters.from_dict(encoded)

    def test_abstract_base_is_not_instantiable(self):
        # `ABC` does not prevent instantiation without an abstract method, and `type()` is
        # deliberately non-abstract, so the base guards itself explicitly.
        with self.assertRaises(TypeError):
            ExternalDistortionParameters()

    def test_encode_decode_roundtrip_with_distortion(self):
        # The round-trip through the encode/decode helpers, which carry the type beside the camera
        # parameters as well as inside them.
        camera = IdealPinholeCameraModelParameters(
            resolution=np.array([640, 480], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([320.0, 240.0], dtype=np.float32),
            focal_length=np.array([500.0, 500.0], dtype=np.float32),
            external_distortion_parameters=self._windshield(),
        )
        encoded = encode_camera_model_parameters(camera)
        self.assertEqual(encoded["external_distortion_type"], "bivariate-windshield")
        decoded = decode_camera_model_parameters(encoded)
        self.assertEqual(decoded.to_json(), camera.to_json())


if __name__ == "__main__":
    unittest.main()
