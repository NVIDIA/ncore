# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

import unittest
import abc
import copy

import numpy as np
import scipy
import parameterized
import torch

from dsai.impl.data.types import FThetaCameraModelParameters, ShutterType
from .camera import FThetaCameraModel

## Reference camera implementation
class ReferenceCamera(metaclass=abc.ABCMeta):

    def project(self, point3d):
        return self.ray2pixel(point3d)

    @abc.abstractmethod
    def ray2pixelIfVisible(self, point3d):
        pass

    @abc.abstractmethod
    def ray2pixel(self, point3d):
        pass

    @abc.abstractmethod
    def pixel2ray(self, pixel2d):
        pass


class ReferenceFThetaCamera(ReferenceCamera):
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

    def getImageSize(self):
        return copy.deepcopy(self._imageSize)

    def isVisible(self, point2d):
        # potential different design decision:
        # - a single pixel has a width of 1 pixel and a height of 1 pixel
        # - the image coordinates point to the center of a pixel
        # - accordingly, the upper left corner of the upper left pixel in the image
        #   has the coordinates [-0.5, -0.5]
        # - potentially, also the coordinate [-0.5, -0.5] would still be
        #   considered visible
        lastPixel = self._imageSize - np.array([1, 1])
        return ((0 <= point2d[0]) and (point2d[0] <= lastPixel[0])
                and (0 <= point2d[1]) and (point2d[1] <= lastPixel[1]))

    def getPrincipalPoint(self):
        return copy.deepcopy(self._principalPoint)

    def getBackwardPolynomial(self):
        return copy.deepcopy(self._backwardPolynomial)

    def setBackwardPolynomial(self, backwardPolynomial):
        self._backwardPolynomial = backwardPolynomial

    def ray2pixelIfVisible(self, point3d):
        """map a 3d ray to the visible part of the image. return [] if the mapping
        is not within the image boundaries.
        """
        point2d = self.ray2pixel(point3d)
        if 0 < len(point2d) and self.isVisible(point2d):
            return point2d
        else:
            return []

    def ray2pixel(self, point3d):
        return self.rays2pixels(point3d)

    def rays2pixels(self, points3d):
        # vectorized version of ray2pixel
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
        points2d = np.full_like(offsets2d, self._principalPoint)
        points2d[polar_mask] += offsets2d[polar_mask]

        return points2d

    def pixel2ray(self, pixel2d):
        offset2d = np.array(pixel2d) - self._principalPoint
        return self._offset2ray(offset2d)

    def _offset2ray(self, offset2d):
        offset = np.array(offset2d)
        radius = np.linalg.norm(offset, axis=offset.ndim-1, keepdims=True)
        theta = self._radius2angle(radius)
        s, c = np.sin(theta), np.cos(theta)
        radius[radius < np.finfo(float).eps] = 1.
        ray = np.append(offset * s / radius, c, axis=offset.ndim-1)
        return ray

    def _determineForwardPolynomial(self, maxRadius):
        linearSystemMatrix, linearSystemVector = self._getForwardPolynomialLinearSystem(maxRadius)
        coefficients = _solveLinearEquation(linearSystemMatrix, linearSystemVector)
        # self._forwardPolynomial= coefficients
        return np.concatenate(([0.], coefficients))

    def _getForwardPolynomialLinearSystem(self, maxRadius):
        samplesRadius = np.array(range(1, int(np.ceil(maxRadius))))
        samplesAngle = np.array([self._radius2angle(r) for r in samplesRadius])
        transposedSystemMatrix = [samplesAngle ** p for p in range(1, len(self._backwardPolynomial))]
        return np.transpose(transposedSystemMatrix), samplesRadius

    def _calculateMaxRadius(self):
        corners = np.array([[0, 0], [self._imageSize[0] - 1, 0],
                               [0, self._imageSize[1] - 1], self._imageSize - [1, 1]])
        radiusAtCorners = [np.linalg.norm(corner - self._principalPoint)
                           for corner in corners]
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

class TestReferenceFThetaCamera(unittest.TestCase):

    def _compareVector(self, a, b):
        self.assertEqual(len(a), len(b))
        for c in zip(a, b):
            self.assertAlmostEqual(c[0], c[1], delta=c[1] * 10 ** -7)

    def test_pixel2ray(self):
        """test backward polynomial coefficients from r**1, r**2, ... r**4"""
        for orderPolynomial in range(1, 5):
            # print("powerOfRadius= ", powerOfRadius)
            self._test_pixel2ray_orderPolynomial(orderPolynomial)

    def _test_pixel2ray_orderPolynomial(self, orderPolynomial):
        """test backward polynomial coefficients up to r**orderPolynomial"""
        baseAngle = np.radians(45)
        resolution = 1000
        principalPoint = (resolution - 1) / 2
        backwardPolynomial = _generateBackwardPolynomial(resolution, baseAngle, orderPolynomial)
        cumulativeAngle = _computeCumulativeAngleAtImageBorder(baseAngle, orderPolynomial)
        camera = ReferenceFThetaCamera([resolution, resolution], [principalPoint, principalPoint], backwardPolynomial)
        self._executePixel2RayTestCase(camera, [principalPoint, principalPoint], [0, 0, 1])
        self._executePixel2RayTestCase(camera, [resolution - 1, principalPoint],
                                       [np.sin(cumulativeAngle), 0, np.cos(cumulativeAngle)])
        self._executePixel2RayTestCase(camera, [principalPoint, resolution - 1],
                                       [0, np.sin(cumulativeAngle), np.cos(cumulativeAngle)])

    def test_pixel2ray_shiftedPrincipalPoint(self):
        """test principal point shift for camera without radial distortions"""
        fov = np.radians(90)
        resolution = np.array([1000, 1000])
        camera = ReferenceFThetaCamera(resolution, [0, 0], [0, fov / resolution[0]])
        self._executePixel2RayTestCase(camera, [resolution[0] / 2, 0], [np.sin(fov / 2), 0, np.cos(fov / 2)])
        self._executePixel2RayTestCase(camera, [0, resolution[1] / 2], [0, np.sin(fov / 2), np.cos(fov / 2)])

    def _executePixel2RayTestCase(self, camera, pixel2d, ray3dExpected):
        rayActual = camera.pixel2ray(pixel2d)
        # print("pixel2d= ", pixel2d, "actual3d= ", rayActual, "expected3d= ", ray3dExpected)
        self._compareVector(rayActual, ray3dExpected)

    def test_ray2pixel(self):
        for orderPolynomial in range(1, 5):
            self._test_ray2pixel_orderPolynomial(orderPolynomial)

    def _test_ray2pixel_orderPolynomial(self, orderPolynomial):
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

        self._executeRay2PixelTestCase(camera, opticalAxesRay, [principalPoint, principalPoint])
        self._executeRay2PixelTestCase(camera, rightRay, [resolution - 1, principalPoint])
        self._executeRay2PixelTestCase(camera, bottomRay, [principalPoint, resolution - 1])

        rays3d = np.array([opticalAxesRay, rightRay, bottomRay])
        pixels2dExpected = np.array(
            [[principalPoint, principalPoint], [resolution - 1, principalPoint], [principalPoint, resolution - 1]])
        self._executeRays2PixelsTestCase(camera, rays3d, pixels2dExpected)

    def _executeRay2PixelTestCase(self, camera, ray3d, pixel2dExpected):
        if len(pixel2dExpected) == 0:
            self._executeRay2PixelFail(camera, ray3d)
        else:
            self._executeRay2PixelSuccess(camera, ray3d, pixel2dExpected)

    def _executeRay2PixelFail(self, camera, ray3d):
        pixel2dActual = camera.ray2pixel(ray3d)
        self.assertEqual(pixel2dActual, [])
        # self.assertFalse(pixel2dActual.success)

    def _executeRay2PixelSuccess(self, camera, ray3d, pixel2dExpected):
        pixel2dActual = camera.ray2pixel(ray3d)
        self.assertTrue(not np.array_equal(pixel2dActual, []))
        # self.assertTrue(pixel2dActual.success)
        # print("ray3d= ", ray3d, "actual2d= ", pixel2dActual, "expected2d= ", pixel2dExpected)
        self._compareVector(pixel2dActual, pixel2dExpected)

    def _executeRays2PixelsTestCase(self, camera, rays3d, pixels2dExpected):
        pixels2dActual = camera.rays2pixels(rays3d)

        for a, e in zip(pixels2dActual, pixels2dExpected):
            self._compareVector(a, e)

    @parameterized.parameterized.expand([(
        "cpu-based evaluation, f32",
        'cpu', torch.float32
    ),
    (
        "cpu-based evaluation, f64",
        'cpu', torch.float64
    ),
    (
        "cuda-based evaluation, f32",
        'cuda', torch.float32
    ),
    (
        "cuda-based evaluation, f64",
        'cuda', torch.float64
    )])
    def test_pixel2ray_ray2pixel_consistency(self, _, device, dtype):
        ''' Tests self-consistency of both the reference camera and torch-based FTheta cameras, as well as
            cross-consistency of both cameras '''
        MAX_DEVIATION_IN_PIXEL = 0.001
        MAX_DEVIATION_RAY = 0.001
        size2d = np.array([1000, 1000])
        principalPoint = size2d / 2
        focalLengthPixel = 500.
        backwardPolynomial = [0., 0.4 / focalLengthPixel, (0.4 / focalLengthPixel) ** 2, (0.4 / focalLengthPixel) ** 3,
                              (0.4 / focalLengthPixel) ** 4]
        camera_ref = ReferenceFThetaCamera(size2d, principalPoint, backwardPolynomial)

        camera_ftheta = ftheta_from_reference(camera_ref, device, dtype) # instantiate a corresponding torch-based camera

        # for p in [0, px]:
        for p in range(int(principalPoint[0])):
            with self.subTest(p=p):
                expectedPoint2d = np.array([[p, p]])

                # Evaluate reference camera
                ray3d_ref = camera_ref.pixel2ray(expectedPoint2d)

                # Evaluate torch-camera
                ray3d = camera_ftheta.pixel_to_camera_ray(camera_ftheta.to_torch(expectedPoint2d).to(camera_ftheta.dtype))

                # test that the computed rays of both cameras agree
                self.assertLessEqual(np.linalg.norm(ray3d_ref - np.array(ray3d.cpu())), MAX_DEVIATION_RAY)

                with self.subTest(angle=np.degrees(np.arccos(ray3d_ref[0][2]))):
                    # Verify reference camera's result
                    actualPoint2d_ref = camera_ref.ray2pixel(ray3d_ref)
                    self.assertLessEqual(np.linalg.norm(expectedPoint2d - actualPoint2d_ref), MAX_DEVIATION_IN_PIXEL)

                    # Verify torch-camera's result
                    actualPoint2d, _ = camera_ftheta.camera_ray_to_pixel(ray3d)
                    self.assertLessEqual(np.linalg.norm(expectedPoint2d - np.array(actualPoint2d.cpu())), MAX_DEVIATION_IN_PIXEL)

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

def _solveLinearEquation(linearSystemMatrix, linearSystemVector):
    solution, residues, rank, singularValues = scipy.linalg.lstsq(linearSystemMatrix, linearSystemVector)
    # print("solution= ", solution, "max(residues)= ", numpy.max(residues))
    return solution

def _generateBackwardPolynomial(resolution, baseAngle, orderPolynomial):
    firstToLastPixelDistance = (resolution - 1)
    backwardPolynomial = [0]
    for j in range(1, orderPolynomial + 1):
        backwardPolynomial.append(baseAngle / ((0.5 * firstToLastPixelDistance) ** j))
    return backwardPolynomial

def _computeCumulativeAngleAtImageBorder(baseAngle, orderPolynomial):
    return baseAngle * orderPolynomial


def ftheta_from_reference(reference_camera: ReferenceFThetaCamera, device: str, dtype: torch.dtype) -> FThetaCameraModel:
    parameters = FThetaCameraModelParameters(
        resolution=reference_camera._imageSize.astype(np.uint64),
        shutter_type=ShutterType.ROLLING_TOP_TO_BOTTOM,
        exposure_time_us=np.uint64(1641.58),
        principal_point=reference_camera._principalPoint.astype(np.float32),
        reference_poly=FThetaCameraModelParameters.PolynomialType.PIXELDIST_TO_ANGLE,
        pixeldist_to_angle_poly=np.array(reference_camera._backwardPolynomial, dtype=np.float32),
        angle_to_pixeldist_poly=np.array(reference_camera._forwardPolynomial, dtype=np.float32),
        max_angle=reference_camera._maxRadius.astype(np.float32))
    return FThetaCameraModel(camera_model_parameters=parameters, device=device, dtype=dtype)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
