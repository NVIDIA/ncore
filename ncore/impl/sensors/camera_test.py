# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

import unittest
import itertools

from typing import Tuple

import numpy as np
import scipy
import scipy.linalg
import parameterized
import torch

from ncore.impl.data.types import FThetaCameraModelParameters, PinholeCameraModelParameters, ShutterType
from ncore.impl.sensors.camera import CameraModel, FThetaCameraModel, PinholeCameraModel

class ReferenceFThetaCamera():
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
        return ((0 <= point2d[0]) and (point2d[0] <= lastPixel[0]) and (0 <= point2d[1])
                and (point2d[1] <= lastPixel[1]))


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

        directions2d_norm[directions2d_norm < np.finfo(float).eps] = 1.
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
        radius[radius < np.finfo(float).eps] = 1.
        ray = np.append(offset * s / radius, c, axis=offset.ndim - 1)
        return ray

    def _determineForwardPolynomial(self, maxRadius):
        linearSystemMatrix, linearSystemVector = self._getForwardPolynomialLinearSystem(maxRadius)
        coefficients = _solveLinearEquation(linearSystemMatrix, linearSystemVector)
        return np.concatenate(([0.], coefficients))

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


class CommonTestCase(unittest.TestCase):
    def _compareVector(self, a, b):
        self.assertEqual(len(a), len(b))
        self.assertIsNone(np.testing.assert_array_almost_equal(a, b))


@parameterized.parameterized_class(('device', 'dtype'),
                                   itertools.product(('cpu', 'cuda'), (torch.float32, torch.float64)))
class TestReferenceFThetaCamera(CommonTestCase):
    ''' Parameterized test cases validating both the reference implementation and the torch-based camera model '''
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
            camera, [[resolution - 1, principalPoint]],
            [[np.sin(cumulativeAngle), 0, np.cos(cumulativeAngle)]])
        self._executeImagePoints2RaysTestCase(
            camera, [[principalPoint, resolution - 1]],
            [[0, np.sin(cumulativeAngle), np.cos(cumulativeAngle)]])

        self._executeImagePoints2RaysTestCase(
            camera,
            [[principalPoint, principalPoint], [resolution - 1, principalPoint], [principalPoint, resolution - 1]],
            [[0, 0, 1], [np.sin(cumulativeAngle), 0, np.cos(cumulativeAngle)],
             [0, np.sin(cumulativeAngle), np.cos(cumulativeAngle)]])

    def test_imagePoints2ray_shiftedPrincipalPoint(self):
        """test principal point shift for camera without radial distortions"""
        fov = np.radians(90)
        resolution = np.array([1000, 1000])
        camera = ReferenceFThetaCamera(resolution, [10, 10], [0, fov / resolution[0]])
        self._executeImagePoints2RaysTestCase(camera, [[ 10 + resolution[0] / 2,  10]], [[np.sin(fov / 2), 0, np.cos(fov / 2)]])
        self._executeImagePoints2RaysTestCase(camera, [[10, 10 + resolution[1] / 2]], [[0, np.sin(fov / 2), np.cos(fov / 2)]])

    def _executeImagePoints2RaysTestCase(self, camera, imagePoints2d, rays3dExpected):
        # Reference
        for a, e in zip(camera.imagePoints2rays(imagePoints2d), rays3dExpected):
            self._compareVector(a, e)

        # Torch-version
        for a, e in zip(
                np.array(
                    ftheta_from_reference(camera, self.device,
                                          self.dtype).image_points_to_camera_rays(np.array(imagePoints2d, ndmin=2)).cpu()),
                np.array(rays3dExpected, ndmin=2)):
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
        imagePoints2dExpected = np.array([[principalPoint, principalPoint], [resolution - 1, principalPoint],
                                     [principalPoint, resolution - 1]])
        self._executeRays2ImagePointsTestCase(camera, rays3d, imagePoints2dExpected)

    def _executeRays2ImagePointsTestCase(self, camera, rays3d, imagePoints2dExpected):
        # Reference
        for a, e in zip(camera.rays2imagePoints(rays3d), imagePoints2dExpected):
            self._compareVector(a, e)

        # Torch-version
        a = ftheta_from_reference(camera, self.device, self.dtype).camera_rays_to_image_points(np.array(rays3d, ndmin=2))
        e = np.array(imagePoints2dExpected, ndmin=2)

        self._compareVector(np.array(a.image_points.cpu()), e)

    def test_imagePoints2rays_rays2imagePoints_consistency(self):
        ''' Tests self-consistency of both the reference camera and torch-based FTheta cameras, as well as
            cross-consistency of both cameras '''
        MAX_DEVIATION_IN_PIXEL = 0.001
        MAX_DEVIATION_RAY = 0.001
        size2d = np.array([1000, 1000])
        principalPoint = size2d / 2
        focalLengthPixel = 500.
        backwardPolynomial = [
            0., 0.4 / focalLengthPixel, (0.4 / focalLengthPixel)**2, (0.4 / focalLengthPixel)**3,
            (0.4 / focalLengthPixel)**4
        ]
        camera_ref = ReferenceFThetaCamera(size2d, principalPoint, backwardPolynomial)

        camera_ftheta = ftheta_from_reference(camera_ref, self.device,
                                              self.dtype)  # instantiate a corresponding torch-based camera

        # for p in [0, px]:
        for p in range(int(principalPoint[0])):
            with self.subTest(p=p):
                expectedPoint2d = np.array([[p, p]])

                # Evaluate reference camera
                ray3d_ref = camera_ref.imagePoints2rays(expectedPoint2d)

                # Evaluate torch-camera
                ray3d = camera_ftheta.image_points_to_camera_rays(
                    camera_ftheta.to_torch(expectedPoint2d).to(camera_ftheta.dtype))

                # test that the computed rays of both cameras agree
                self.assertLessEqual(np.linalg.norm(ray3d_ref - np.array(ray3d.cpu())), MAX_DEVIATION_RAY)

                with self.subTest(angle=np.degrees(np.arccos(ray3d_ref[0][2]))):
                    # Verify reference camera's result
                    actualPoint2d_ref = camera_ref.rays2imagePoints(ray3d_ref)
                    self.assertLessEqual(np.linalg.norm(expectedPoint2d - actualPoint2d_ref), MAX_DEVIATION_IN_PIXEL)

                    # Verify torch-camera's result
                    image_points = camera_ftheta.camera_rays_to_image_points(ray3d)
                    self.assertLessEqual(np.linalg.norm(expectedPoint2d - np.array(image_points.image_points.cpu())),
                                         MAX_DEVIATION_IN_PIXEL)

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
        pixel_idxs = np.random.default_rng(seed=0).choice(resolution-1, (100,2))

        pixel_rays = ftheta_cam.pixels_to_camera_rays(pixel_idxs.astype(np.int32)).cpu()
        image_point_rays = ftheta_cam.image_points_to_camera_rays((pixel_idxs + 0.5).astype(np.float32)).cpu()

        self._compareVector(pixel_rays, image_point_rays)

    def test_return_all_projections(self):

        resolution = 1000
        principalPoint = (resolution - 1) / 2
        baseAngle = np.radians(5)
        backwardPolynomial = _generateBackwardPolynomial(resolution, baseAngle, 4)
        cumulativeAngle = _computeCumulativeAngleAtImageBorder(baseAngle, 4)

        camera = ReferenceFThetaCamera([resolution, resolution], [principalPoint, principalPoint], backwardPolynomial)

        T_world_sensor_start = np.eye(4)
        T_world_sensor_end = np.eye(4)

        ftheta_cam = ftheta_from_reference(camera, self.device,self.dtype)

        # Points to test (two 2,3 are invalid)
        world_points = np.array([[0,0,10], [0,0,20], [50,5,10], [0,0,-10], [0,0,30]])

        # Test shutter pose projection
        image_points = ftheta_cam.world_points_to_image_points_shutter_pose(world_points, T_world_sensor_start, T_world_sensor_end)
        image_points_all = ftheta_cam.world_points_to_image_points_shutter_pose(world_points, T_world_sensor_start,
                                                                                T_world_sensor_end,
                                                                                return_valid_indices=True,
                                                                                return_all_projections=True)
        self._compareVector(image_points.image_points.cpu(), image_points_all.image_points[image_points_all.valid_indices].cpu())

        # Test single pose projection
        image_points = ftheta_cam.world_points_to_image_points_static_pose(world_points, T_world_sensor_start)
        image_points_all = ftheta_cam.world_points_to_image_points_static_pose(world_points, T_world_sensor_start,
                                                                                return_valid_indices=True,
                                                                                return_all_projections=True)

        self._compareVector(image_points.image_points.cpu(), image_points_all.image_points[image_points_all.valid_indices].cpu())

    def test_inputs_and_input_types(self):
        camera = ReferenceFThetaCamera(np.array([1000, 1000]), [10, 10], [0, np.radians(90) / 1000])
        ftheta_cam = ftheta_from_reference(camera, self.device,self.dtype)

        pixel = np.array([100, 100]).reshape(1,2)
        ray = np.array([0,1,0]).reshape(1,3)

        # Test invalid inputs
        self.assertRaises(AssertionError, ftheta_cam.image_points_to_camera_rays, pixel.astype(np.int32))
        self.assertRaises(AssertionError, ftheta_cam.pixels_to_camera_rays, pixel.astype(np.float32))

        self.assertRaises(AssertionError, ftheta_cam.world_points_to_image_points_shutter_pose, ray, np.eye(4), np.eye(4), **{'return_timestamps': True})
        self.assertRaises(AssertionError, ftheta_cam.world_points_to_image_points_shutter_pose, ray, np.eye(4), np.eye(4),
                                **{'start_timestamp_us': 100, 'end_timestamp_us': 90, 'return_timestamps': True})

        # Test valid inputs
        ftheta_cam.image_points_to_camera_rays(pixel.astype(np.float32))
        ftheta_cam.pixels_to_camera_rays(pixel.astype(np.int32))
        ftheta_cam.world_points_to_image_points_shutter_pose(ray, np.eye(4), np.eye(4), **{'start_timestamp_us': 90, 'end_timestamp_us': 100, 'return_timestamps': True})

def _solveLinearEquation(linearSystemMatrix, linearSystemVector):
    solution, _, _, _ = scipy.linalg.lstsq(linearSystemMatrix, linearSystemVector)
    return solution


def _generateBackwardPolynomial(resolution, baseAngle, orderPolynomial):
    firstToLastPixelDistance = (resolution - 1)
    backwardPolynomial = [0]
    for j in range(1, orderPolynomial + 1):
        backwardPolynomial.append(baseAngle / ((0.5 * firstToLastPixelDistance)**j))
    return backwardPolynomial


def _computeCumulativeAngleAtImageBorder(baseAngle, orderPolynomial):
    return baseAngle * orderPolynomial


def ftheta_from_reference(reference_camera: ReferenceFThetaCamera, device: str,
                          dtype: torch.dtype) -> FThetaCameraModel:

    parameters = FThetaCameraModelParameters(
        resolution=reference_camera._imageSize.astype(np.uint64),
        shutter_type=ShutterType.ROLLING_TOP_TO_BOTTOM,
        # Subtract the principal offset to align the image coordinate system conventions
        # (offset will be added back during the initialization of the class)
        principal_point=reference_camera._principalPoint.astype(np.float32) - 0.5,
        reference_poly=FThetaCameraModelParameters.PolynomialType.PIXELDIST_TO_ANGLE,
        pixeldist_to_angle_poly=np.array(reference_camera._backwardPolynomial, dtype=np.float32),
        angle_to_pixeldist_poly=np.array(reference_camera._forwardPolynomial, dtype=np.float32),
        max_angle=reference_camera._radius2angle(reference_camera._maxRadius).astype(np.float32))

    return FThetaCameraModel(camera_model_parameters=parameters, device=device, dtype=dtype)


@parameterized.parameterized_class(('device', 'dtype'),
                                   itertools.product(('cpu', 'cuda'), (torch.float32, torch.float64)))
class TestPinholeCamera(CommonTestCase):


    def test_imagePoints2rays_rays2imagePoints_consistency(self):
        ''' Tests self-consistency of torch-based Pinhole camera model '''

        # Waymo camera parameters
        cam_model_params = PinholeCameraModelParameters(resolution=np.array([1920, 1280], dtype=np.uint64),
                                                        shutter_type=ShutterType.ROLLING_RIGHT_TO_LEFT,
                                                        principal_point=np.array([935.1248081874216, 635.052474560227],
                                                                                 dtype=np.float32),
                                                        focal_length=np.array([
                                                            2059.0471439559833,
                                                            2059.0471439559833,
                                                        ],
                                                                              dtype=np.float32),
                                                        radial_coeffs=np.array([
                                                            0.04239636827428756,
                                                            -0.34165672675852826,
                                                            0,
                                                            0,
                                                            0,
                                                            0,
                                                        ],
                                                                               dtype=np.float32),
                                                        tangential_coeffs=np.array(
                                                            [0.001805535524580487, -0.00005530628187935031],
                                                            dtype=np.float32),
                                                        thin_prism_coeffs=np.array([0, 0, 0, 0], dtype=np.float32))

        # add additional arbitrary radial and thin-prism coeffs for this test only to guarantee code-coverage
        cam_model_params.radial_coeffs[2:] = [0.01, 0.02, -0.01, 0.02]
        cam_model_params.thin_prism_coeffs[:] = [0.01, 0.02, 0.02, 0.01]

        cam_model = PinholeCameraModel(cam_model_params, device=self.device, dtype=self.dtype)

        MAX_DEVIATION_IN_IMAGE_COORDINATES = 0.001

        # for p in [0, px] with stepsize
        STEPSIZE = 20
        for p in range(0, int(cam_model_params.principal_point[0]), STEPSIZE):
            with self.subTest(p=p):
                # very idempotence of imagePoints2rays(rays2imagePoints([p,p]))
                expectedPoint2d = np.array([[p, p]])

                # Verify torch-camera's result
                ray3d = cam_model.image_points_to_camera_rays(cam_model.to_torch(expectedPoint2d).to(cam_model.dtype))
                image_points = cam_model.camera_rays_to_image_points(ray3d)

                self.assertTrue(image_points.valid_flag)
                self.assertLessEqual(np.linalg.norm(expectedPoint2d - np.array(image_points.image_points.cpu())),
                                     MAX_DEVIATION_IN_IMAGE_COORDINATES)


class ReferenceSimplePinholeCamera():
    ''' Simple reference pinhole camera with symbolic evaluations (supporting k1,k2,k3,p1,p2) '''
    def __init__(self, params: PinholeCameraModelParameters, dtype: np.dtype):
        self.params = params
        self.dtype = dtype

        assert not np.any(self.params.radial_coeffs[3:]), "only supporting non-zero k1,k2,k3"
        assert not np.any(self.params.thin_prism_coeffs), "not supporting thin-prism coeffs"

    def _distortion(self, uvN):
        ''' Computes the radial + tangential distortion given the camera rays '''

        # Helper variables for primary function evaluation
        u0u0 = uvN[0] * uvN[0]
        u1u1 = uvN[1] * uvN[1]
        r_2 = u0u0 + u1u1
        uv_prod = uvN[0] * uvN[1]
        a1 = 2 * uv_prod
        a2 = r_2 + 2 * u0u0
        a3 = r_2 + 2 * u1u1

        icD = 1.0 + r_2 * (self.params.radial_coeffs[0] + r_2 *
                           (self.params.radial_coeffs[1] + r_2 * self.params.radial_coeffs[2]))

        delta_x = self.params.tangential_coeffs[0] * a1 + self.params.tangential_coeffs[1] * a2
        delta_y = self.params.tangential_coeffs[0] * a3 + self.params.tangential_coeffs[1] * a1

        uvND = uvN * icD + np.array([[delta_x, delta_y]], dtype=self.dtype)

        # Helper variables for symbolic Jacobian evaluation
        b1 = self.params.radial_coeffs[1] + self.params.radial_coeffs[2] * r_2
        b11 = 2 * (self.params.radial_coeffs[0] + b1 * r_2) + r_2 * (2 * self.params.radial_coeffs[2] * r_2 + 2 * b1)
        b2 = uvN[0] * b11
        b3 = uvN[1] * b11
        b4 = (self.params.radial_coeffs[0] + b1 * r_2) * r_2 + 1.0

        J_uvND = np.array([[
            2 * self.params.tangential_coeffs[0] * uvN[1] + 6 * self.params.tangential_coeffs[1] * uvN[0] +
            uvN[0] * b2 + b4,
            2 * self.params.tangential_coeffs[0] * uvN[0] + 2 * self.params.tangential_coeffs[1] * uvN[1] + uvN[0] * b3
        ],
                           [
                               2 * self.params.tangential_coeffs[0] * uvN[0] +
                               2 * self.params.tangential_coeffs[1] * uvN[1] + uvN[1] * b2,
                               6 * self.params.tangential_coeffs[0] * uvN[1] +
                               2 * self.params.tangential_coeffs[1] * uvN[0] + uvN[1] * b3 + b4
                           ]])

        return uvND, J_uvND

    def _perspective_normalization(self, x: np.ndarray):
        uvN = np.array([x[0] / x[2], x[1] / x[2]], dtype=self.dtype)
        J_uvN = np.array([[1 / x[2], 0, -x[0] / x[2]**2], [0, 1 / x[2], -x[1] / x[2]**2]], dtype=self.dtype)

        return uvN, J_uvN

    def _perspective_projection(self, uvND: np.ndarray):
        uv = uvND * self.params.focal_length + self.params.principal_point
        J_uv = np.array([[self.params.focal_length[0], 0], [0, self.params.focal_length[1]]], dtype=self.dtype)

        return uv, J_uv

    def camera_ray_to_image_points(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ''' Assumes ray is a valid projection / returns image point + Jacobian'''

        uvN, J_uvN = self._perspective_normalization(x)

        uvND, J_uvND = self._distortion(uvN)

        uv, J_uv = self._perspective_projection(uvND)

        return uv.squeeze(), J_uv @ J_uvND @ J_uvN  # Assemble full transformation's Jacobian according to chain-rule


@parameterized.parameterized_class(('device', 'dtype'),
                                   itertools.product(('cpu', 'cuda'), (torch.float32, torch.float64)))
class TestJacobian(CommonTestCase):
    def test_pinhole_reference(self):
        ''' Tests consistency of camera model Jacobians with reference implementation '''

        # Distorted pinhole camera model with "simple" k1,k2,k3,p1,p2 parametrization only
        cam_model_params = PinholeCameraModelParameters(resolution=np.array([1920, 1280], dtype=np.uint64),
                                             shutter_type=ShutterType.ROLLING_RIGHT_TO_LEFT,
                                             principal_point=np.array([935.1248081874216, 635.052474560227],
                                                                      dtype=np.float32),
                                             focal_length=np.array([
                                                 2059.0471439559833,
                                                 2059.4231439559833,
                                             ],
                                                                   dtype=np.float32),
                                             radial_coeffs=np.array([
                                                 0.04239636827428756,
                                                 -0.34165672675852826,
                                                 0.01,
                                                 0,
                                                 0,
                                                 0,
                                             ], dtype=np.float32),
                                             tangential_coeffs=np.array([0.001805535524580487, -0.00005530628187935031], dtype=np.float32),
                                             thin_prism_coeffs=np.array([0, 0, 0, 0], dtype=np.float32))

        cam_model_ref = ReferenceSimplePinholeCamera(cam_model_params, {torch.float32 : np.float32, torch.float64 : np.float64}[self.dtype])
        cam_model = CameraModel.from_parameters(cam_model_params, device=self.device, dtype=self.dtype)

        rays3d = cam_model.image_points_to_camera_rays(torch.Tensor([[20, 40], [11, 12], [15, 20],
                                                                     [500, 500]]))  # valid rays only

        for ray3d in rays3d:
            pref, Jref = cam_model_ref.camera_ray_to_image_points(ray3d.cpu().numpy())

            proj = cam_model.camera_rays_to_image_points(ray3d.unsqueeze(1).transpose(1,0), return_jacobians=True)

            np.testing.assert_array_almost_equal(pref, proj.image_points.detach()[0].cpu().numpy())
            np.testing.assert_array_almost_equal(Jref, proj.jacobians.detach()[0].cpu().numpy(), decimal=6 if self.dtype == torch.float64 else 4)


    def test_jacobian_consistency(self):
        ''' Tests consistency of camera model Jacobians with autograd results '''

        cam_models = [
            # Ideal pinhole camera parameters
            CameraModel.from_parameters(
                PinholeCameraModelParameters(resolution=np.array([1920, 1280], dtype=np.uint64),
                                             shutter_type=ShutterType.ROLLING_RIGHT_TO_LEFT,
                                             principal_point=np.array([935.1248081874216, 635.052474560227],
                                                                      dtype=np.float32),
                                             focal_length=np.array([
                                                 2059.0471439559833,
                                                 2059.0471439559833,
                                             ],
                                                                   dtype=np.float32),
                                             radial_coeffs=np.array([
                                                 0,
                                                 0,
                                                 0,
                                                 0,
                                                 0,
                                                 0,
                                             ], dtype=np.float32),
                                             tangential_coeffs=np.array([0, 0], dtype=np.float32),
                                             thin_prism_coeffs=np.array([0, 0, 0, 0], dtype=np.float32)),
                                             device=self.device, dtype=self.dtype),
            # Waymo camera parameters
            CameraModel.from_parameters(
                PinholeCameraModelParameters(resolution=np.array([1920, 1280], dtype=np.uint64),
                                             shutter_type=ShutterType.ROLLING_RIGHT_TO_LEFT,
                                             principal_point=np.array([935.1248081874216, 635.052474560227],
                                                                      dtype=np.float32),
                                             focal_length=np.array([
                                                 2059.0471439559833,
                                                 2059.0471439559833,
                                             ],
                                                                   dtype=np.float32),
                                             radial_coeffs=np.array([
                                                 0.04239636827428756,
                                                 -0.34165672675852826,
                                                 0,
                                                 0,
                                                 0,
                                                 0,
                                             ],
                                                                    dtype=np.float32),
                                             tangential_coeffs=np.array([0.001805535524580487, -0.00005530628187935031],
                                                                        dtype=np.float32),
                                             thin_prism_coeffs=np.array([0, 0, 0, 0], dtype=np.float32)),
                                             device=self.device, dtype=self.dtype),

            # NV 120deg instance
            CameraModel.from_parameters(
                FThetaCameraModelParameters(
                    resolution=np.array([3848, 2168], dtype=np.uint64),
                    shutter_type=ShutterType.ROLLING_TOP_TO_BOTTOM,
                    principal_point=np.array([1904.948486328125, 1090.5164794921875], dtype=np.float32),
                    reference_poly=FThetaCameraModelParameters.PolynomialType.PIXELDIST_TO_ANGLE,
                    pixeldist_to_angle_poly=np.array([
                        0.0, 0.0005380856455303729, -1.2021251771798802e-09, 4.5657002484267295e-12,
                        -5.581118088908714e-16, 0.0
                    ],
                                                     dtype=np.float32),
                    angle_to_pixeldist_poly=np.array(
                        [0.0, 1858.59228515625, 6.894773483276367, -53.92193603515625, 14.201756477355957, 0.0],
                        dtype=np.float32),
                    max_angle=1.2292176485061646),
                    device=self.device, dtype=self.dtype)
        ]

        for cam_model in cam_models:

            def projection_wrapper(x):
                return cam_model.camera_rays_to_image_points(x[None, :]).image_points.squeeze()

            valid_rays3d = cam_model.image_points_to_camera_rays(
                torch.Tensor([[20, 40], [11, 12], [15, 20], [500, 500]]))  # valid rays
            principal_direction_rays3d = torch.Tensor([[0, 0, 1], 
                                                       [0, 0, 5],
                                                       [0, 0, 0.1]]).to(valid_rays3d)  # rays along the principal direction
            invalid_rays3d = torch.Tensor([[1, 2, -5], [1, 2, 0], [0, 0, 0]]).to(
                valid_rays3d
            )  # some "invalid" rays (behind camera / on the center of projection plane but ouf of FOV / zero)
            rays3d = torch.cat([valid_rays3d, principal_direction_rays3d, invalid_rays3d])

            # evaluate projection with jacobians
            proj = cam_model.camera_rays_to_image_points(rays3d, return_jacobians=True)

            for i, ray3d in enumerate(rays3d):
                Jref = torch.autograd.functional.jacobian(projection_wrapper,
                                                          ray3d,
                                                          strict=True,
                                                          strategy='reverse-mode')

                # Make sure API-computed Jacobian coincides with autograd result
                np.testing.assert_array_almost_equal(Jref.cpu().numpy(), proj.jacobians[i].cpu().numpy())

                self.assertTrue(
                    proj.valid_flag[i] if i < len(rays3d) - len(invalid_rays3d) else
                    not proj.valid_flag[i])  # First rays should be flagged as valid, others should be invalid

if __name__ == '__main__':
    unittest.main()
