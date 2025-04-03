# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

import dataclasses
import unittest
import itertools

from typing import Tuple, Union, List

import numpy as np
import scipy
import scipy.linalg
import parameterized
import torch
import cv2

from ncore.impl.data.types import (
    FThetaCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
    OpenCVFisheyeCameraModelParameters,
    ShutterType,
    ReferencePolynomial,
    BivariateWindshieldModelParameters,
)
from ncore.impl.sensors.camera import (
    CameraModel,
    FThetaCameraModel,
    OpenCVPinholeCameraModel,
    OpenCVFisheyeCameraModel,
    ExternalDistortionModel,
    BivariateWindshieldModel,
    to_torch,
)
from ncore.impl.sensors.common import to_torch


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
    def test_cuda_available(self):
        """
        Some camera tests explicitly check cuda-based computations
        (while internally falling back to CPU if cuda is not available).

        This test asserts that a cuda device is actually available to torch
        """

        self.assertTrue(torch.cuda.is_available())


class CommonTestCase(unittest.TestCase):
    def _compareVector(self, a, b):
        self.assertEqual(len(a), len(b))
        self.assertIsNone(np.testing.assert_array_almost_equal(a, b))


@parameterized.parameterized_class(
    ("device", "dtype"), itertools.product(("cpu", "cuda"), (torch.float32, torch.float64))
)
class TestReferenceFThetaCamera(CommonTestCase):
    """Parameterized test cases validating both the reference implementation and the torch-based camera model"""

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

        # Torch-version
        for a, e in zip(
            np.array(
                ftheta_from_reference(camera, self.device, self.dtype)
                .image_points_to_camera_rays(np.array(imagePoints2d, ndmin=2))
                .cpu()
            ),
            np.array(rays3dExpected, ndmin=2),
        ):
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
                ray3d = camera_ftheta.image_points_to_camera_rays(
                    to_torch(expectedPoint2d, device=camera_ftheta.device, dtype=camera_ftheta.dtype)
                )

                # test that the computed rays of both cameras agree
                self.assertLessEqual(np.linalg.norm(ray3d_ref - np.array(ray3d.cpu())), MAX_DEVIATION_RAY)

                with self.subTest(angle=np.degrees(np.arccos(ray3d_ref[0][2]))):
                    # Verify reference camera's result
                    actualPoint2d_ref = camera_ref.rays2imagePoints(ray3d_ref)
                    self.assertLessEqual(np.linalg.norm(expectedPoint2d - actualPoint2d_ref), MAX_DEVIATION_IN_PIXEL)

                    # Verify torch-camera's result
                    image_points = camera_ftheta.camera_rays_to_image_points(ray3d)
                    self.assertLessEqual(
                        np.linalg.norm(expectedPoint2d - np.array(image_points.image_points.cpu())),
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
                ray3d = camera_ftheta.image_points_to_camera_rays(
                    to_torch(expectedPoint2d, device=camera_ftheta.device, dtype=camera_ftheta.dtype)
                )

                with self.subTest(angle=np.degrees(np.arccos(ray3d.cpu()[0][2]))):
                    # Verify torch-camera's result
                    image_points = camera_ftheta.camera_rays_to_image_points(ray3d)
                    self.assertLessEqual(
                        np.linalg.norm(expectedPoint2d - np.array(image_points.image_points.cpu())),
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
                ray3d = camera_ftheta.image_points_to_camera_rays(
                    to_torch(expectedPoint2d, device=camera_ftheta.device, dtype=camera_ftheta.dtype)
                )

                with self.subTest(angle=np.degrees(np.arccos(ray3d.cpu()[0][2]))):
                    # Verify torch-camera's result
                    image_points = camera_ftheta.camera_rays_to_image_points(ray3d)
                    self.assertLessEqual(
                        np.linalg.norm(expectedPoint2d - np.array(image_points.image_points.cpu())),
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

        pixel_rays = ftheta_cam.pixels_to_camera_rays(pixel_idxs.astype(np.int32)).cpu()
        image_point_rays = ftheta_cam.image_points_to_camera_rays((pixel_idxs + 0.5).astype(np.float32)).cpu()

        self._compareVector(pixel_rays, image_point_rays)

    def test_empty_single_more_pixels(self):
        resolution = 1000
        principalPoint = (resolution - 1) / 2
        baseAngle = np.radians(35)
        backwardPolynomial = _generateBackwardPolynomial(resolution, baseAngle, 4)
        camera = ReferenceFThetaCamera([resolution, resolution], [principalPoint, principalPoint], backwardPolynomial)

        ftheta_cam = ftheta_from_reference(camera, self.device, self.dtype)

        def check(pixel_idxs: np.ndarray) -> None:
            camera_rays = ftheta_cam.pixels_to_camera_rays(pixel_idxs.astype(np.int32)).cpu()

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
        self.assertTrue(len(image_point_invalid.valid_indices) == 0)
        self.assertTrue(len(image_point_invalid.image_points) > 0)

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
        self.assertTrue(len(image_point_invalid.valid_indices) == 0)
        self.assertTrue(len(image_point_invalid.image_points) > 0)

    def test_inputs_and_input_types(self):
        camera = ReferenceFThetaCamera(np.array([1000, 1000]), [10, 10], [0, np.radians(90) / 1000])
        ftheta_cam = ftheta_from_reference(camera, self.device, self.dtype)

        pixel = np.array([100, 100]).reshape(1, 2)
        ray = np.array([0, 1, 0]).reshape(1, 3)

        # Test invalid inputs
        self.assertRaises(AssertionError, ftheta_cam.image_points_to_camera_rays, pixel.astype(np.int32))
        self.assertRaises(AssertionError, ftheta_cam.pixels_to_camera_rays, pixel.astype(np.float32))

        self.assertRaises(
            AssertionError,
            ftheta_cam.world_points_to_image_points_shutter_pose,
            ray,
            np.eye(4),
            np.eye(4),
            **{"return_timestamps": True},
        )
        self.assertRaises(
            AssertionError,
            ftheta_cam.world_points_to_image_points_shutter_pose,
            ray,
            np.eye(4),
            np.eye(4),
            **{"start_timestamp_us": 100, "end_timestamp_us": 90, "return_timestamps": True},
        )

        # Test valid inputs
        ftheta_cam.image_points_to_camera_rays(pixel.astype(np.float32))
        ftheta_cam.pixels_to_camera_rays(pixel.astype(np.int32))
        ftheta_cam.world_points_to_image_points_shutter_pose(
            ray, np.eye(4), np.eye(4), **{"start_timestamp_us": 90, "end_timestamp_us": 100, "return_timestamps": True}
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
    ("device", "dtype"), itertools.product(("cpu", "cuda"), (torch.float32, torch.float64))
)
class TestPinholeCamera(CommonTestCase):
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

                # Verify torch-camera's result
                ray3d = cam_model.image_points_to_camera_rays(
                    to_torch(expectedPoint2d, device=cam_model.device, dtype=cam_model.dtype)
                )
                image_points = cam_model.camera_rays_to_image_points(ray3d)

                self.assertTrue(image_points.valid_flag)
                self.assertLessEqual(
                    np.linalg.norm(expectedPoint2d - np.array(image_points.image_points.cpu())),
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
    ("device", "dtype"), itertools.product(("cpu", "cuda"), (torch.float32, torch.float64))
)
class TestJacobian(CommonTestCase):
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
            cam_model_params, {torch.float32: np.float32, torch.float64: np.float64}[self.dtype]
        )
        cam_model = CameraModel.from_parameters(cam_model_params, device=self.device, dtype=self.dtype)

        rays3d = cam_model.image_points_to_camera_rays(
            torch.Tensor([[20, 40], [11, 12], [15, 20], [500, 500]])
        )  # valid rays only

        for ray3d in rays3d:
            pref, Jref = cam_model_ref.camera_ray_to_image_points(ray3d.cpu().numpy())

            proj = cam_model.camera_rays_to_image_points(ray3d.unsqueeze(1).transpose(1, 0), return_jacobians=True)

            np.testing.assert_array_almost_equal(pref, proj.image_points.detach()[0].cpu().numpy())
            np.testing.assert_array_almost_equal(
                Jref, proj.jacobians.detach()[0].cpu().numpy(), decimal=6 if self.dtype == torch.float64 else 3
            )

    def test_jacobian_consistency(self):
        """Tests consistency of camera model Jacobians with autograd results"""

        cam_models = [
            # Ideal pinhole camera parameters
            CameraModel.from_parameters(
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
            CameraModel.from_parameters(
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
            CameraModel.from_parameters(
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
            # External costumer fisheye model
            CameraModel.from_parameters(
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
                )
            ),
        ]

        for cam_model in cam_models:

            def projection_wrapper(x):
                return cam_model.camera_rays_to_image_points(x[None, :]).image_points.squeeze()

            valid_rays3d = cam_model.image_points_to_camera_rays(
                torch.Tensor([[20, 40], [11, 12], [15, 20], [500, 500]])
            )  # valid rays
            self.assertLessEqual(
                (torch.linalg.norm(valid_rays3d, axis=1, keepdims=True) - torch.ones_like(valid_rays3d[:, :1]))
                .abs()
                .max()
                .item(),
                1e-07,
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
                np.testing.assert_array_almost_equal(Jref.cpu().numpy(), proj.jacobians[i].cpu().numpy())

                self.assertTrue(
                    proj.valid_flag[i] if i < len(rays3d) - len(invalid_rays3d) else not proj.valid_flag[i]
                )  # First rays should be flagged as valid, others should be invalid


@parameterized.parameterized_class(
    ("device", "dtype"), itertools.product(("cpu", "cuda"), (torch.float32, torch.float64))
)
class TestFisheyeCamera(CommonTestCase):
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

        # make sure the principal point gets unprojected to the principal axis
        ray3d = self.cam_model.image_points_to_camera_rays(
            torch.from_numpy(self.cam_model_params.principal_point).reshape(1, 2)
        )

        self.assertLessEqual(
            np.linalg.norm(ray3d.cpu().numpy() - np.array([0, 0, 1])), self.MAX_DEVIATION_IN_RAY_COORDINATES
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

                ray3d = self.cam_model.image_points_to_camera_rays(
                    to_torch(expectedPoint2d, device=self.cam_model.device, dtype=self.dtype)
                )
                image_points = self.cam_model.camera_rays_to_image_points(ray3d)

                if i > 0:
                    # avoid 'valid' prevision issues if points get re-projected right onto each side of the image boundary for p=[0,0]
                    self.assertTrue(image_points.valid_flag)
                self.assertLessEqual(
                    np.linalg.norm(expectedPoint2d - np.array(image_points.image_points.cpu())),
                    self.MAX_DEVIATION_IN_IMAGE_COORDINATES,
                )

                # 2. verify consistency with OpenCV reference (one-way is sufficient)
                image_point_opencv = ray_to_image_point_opencv(ray3d.cpu().numpy(), self.cam_model_params)
                self.assertLessEqual(
                    np.linalg.norm(image_point_opencv - np.array(image_points.image_points.cpu())),
                    self.MAX_DEVIATION_IN_IMAGE_COORDINATES,
                )


class CameraModelsBaseTestCase(CommonTestCase):
    def setUp(self):
        # Make printed errors more representable numerically
        np.set_printoptions(floatmode="unique", linewidth=200, suppress=True)

        # Real-world customer camera parameters to test
        self.cam_model_params = [
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


@parameterized.parameterized_class(
    ("device", "dtype"), itertools.product(("cpu", "cuda"), (torch.float32, torch.float64))
)
class TestParameterIO(CameraModelsBaseTestCase):
    def test_model_parameters_roundtrip(self):
        """Validate model parameters obtained from torch model instances are correctly mapped back to the input versions
        between device transfers"""

        for cam_model_params in self.cam_model_params + self.cam_model_params_wsd:
            with self.subTest(cam_model_params=cam_model_params):
                cam_model = CameraModel.from_parameters(cam_model_params, device=self.device, dtype=self.dtype)

                expected_device = self.device
                if expected_device == "cuda" and not torch.cuda.is_available():
                    # fall back to cpu if cuda is not available in unit test machine
                    expected_device = "cpu"
                self.assertEqual(
                    cam_model.resolution.device.type, expected_device
                )  # make sure original device is correct

                # make sure retrieved parameters correspond to reference
                self.assertEqual(cam_model_params.to_json(), cam_model.get_parameters().to_json())

                # flip flop device using nn.Module magic
                new_device_str = "cuda" if self.device == "cpu" else "cpu"
                if new_device_str == "cuda" and not torch.cuda.is_available():
                    # fall back to cpu if cuda is not available in unit test machine
                    new_device_str = "cpu"
                cam_model.to(device=new_device_str)
                self.assertEqual(
                    cam_model.resolution.device.type, new_device_str
                )  # make sure the new device is correct

                # make sure retrieved parameters still correspond to reference
                self.assertEqual(cam_model_params.to_json(), cam_model.get_parameters().to_json())


@parameterized.parameterized_class(
    ("device", "dtype"), itertools.product(("cpu", "cuda"), (torch.float32, torch.float64))
)
class TestTransformParameters(CameraModelsBaseTestCase):
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
                                cam_model = CameraModel.from_parameters(
                                    cam_model_params, device=self.device, dtype=self.dtype
                                )

                                cam_model_transformed = CameraModel.from_parameters(
                                    cam_model_params.transform(
                                        image_domain_scale=scale_factor, image_domain_offset=offset
                                    ),
                                    device=self.device,
                                    dtype=self.dtype,
                                )

                                # Validate original image domain -> 3d -> transformed image domain round-trip
                                ray3d = cam_model.image_points_to_camera_rays(IMAGE_POINTS)
                                image_points_transformed = cam_model_transformed.camera_rays_to_image_points(ray3d)

                                self.assertTrue(
                                    image_points_transformed.valid_flag.all(),
                                    msg="All point projections need to be valid for scale verification",
                                )

                                self.assertLessEqual(
                                    np.linalg.norm(
                                        (image_points_transformed_ref := (IMAGE_POINTS * scale_factor - offset))
                                        - image_points_transformed.image_points.cpu().numpy()
                                    ),
                                    self.MAX_DEVIATION_IN_IMAGE_COORDINATES,
                                )

                                # Validate transformed image-domain -> 3d -> untransformed image-domain round-trip
                                ray3d_transformed = cam_model_transformed.image_points_to_camera_rays(
                                    image_points_transformed_ref
                                )

                                image_points_untransformed = cam_model.camera_rays_to_image_points(ray3d_transformed)

                                self.assertTrue(
                                    image_points_untransformed.valid_flag.all(),
                                    msg="All point projections need to be valid for transformation verification",
                                )

                                self.assertLessEqual(
                                    np.linalg.norm(
                                        IMAGE_POINTS - image_points_untransformed.image_points.cpu().numpy()
                                    ),
                                    self.MAX_DEVIATION_IN_IMAGE_COORDINATES,
                                )


@parameterized.parameterized_class(
    ("device", "dtype"), itertools.product(("cpu", "cuda"), (torch.float32, torch.float64))
)
class TestExternalDistortion(CommonTestCase):
    def test_from_parameters(self):
        # Verify that "None" parameters returns a "None" object
        res_none = ExternalDistortionModel.maybe_from_parameters(None, self.device, self.dtype)
        self.assertIsNone(res_none)

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
        res_ws = ExternalDistortionModel.maybe_from_parameters(windshield_params, self.device, self.dtype)
        self.assertTrue(isinstance(res_ws, BivariateWindshieldModel))


@parameterized.parameterized_class(
    ("device", "dtype"), itertools.product(("cpu", "cuda"), (torch.float32, torch.float64))
)
class TestBivariateWindshieldModel(CommonTestCase):
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
        with self.assertRaises(ValueError) as context:
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


if __name__ == "__main__":
    unittest.main()
