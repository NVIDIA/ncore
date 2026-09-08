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

import io
import unittest

from typing import List, Optional, Tuple, TypeVar, Union

import numpy as np
import numpy.testing as npt
import PIL.Image as PILImage

from ncore.impl.common.transformations import PoseGraphInterpolator
from ncore.impl.data.types import (
    CameraModelParameters,
    EncodedImageData,
    FThetaCameraModelParameters,
    IdealPinholeCameraModelParameters,
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
    PointCloud,
    ShutterType,
)


#: A caller-side generic helper over camera model parameters. Bound to the abstract base, this is
#: the shape downstream code needs in order to transform parameters while keeping the caller's
#: concrete type; it only type-checks because every `transform` override returns `Self`.
_CameraModelParametersT = TypeVar("_CameraModelParametersT", bound=CameraModelParameters)


def _downscale(
    parameters: _CameraModelParametersT,
    image_domain_scale: Union[float, Tuple[float, float]],
) -> _CameraModelParametersT:
    return parameters.transform(image_domain_scale)


class TestPointCloud(unittest.TestCase):
    """Tests for the PointCloud dataclass."""

    @staticmethod
    def _make_pc(
        xyz: np.ndarray,
        attributes: Optional[dict] = None,
        coordinate_unit: PointCloud.CoordinateUnit = PointCloud.CoordinateUnit.METERS,
        reference_frame_id: str = "sensor",
        reference_frame_timestamp_us: int = 0,
    ) -> PointCloud:
        """Build a PointCloud with sensible defaults."""
        return PointCloud(
            _xyz=xyz,
            reference_frame_id=reference_frame_id,
            reference_frame_timestamp_us=reference_frame_timestamp_us,
            coordinate_unit=coordinate_unit,
            _attributes=attributes or {},
        )

    @staticmethod
    def _make_pose_graph_with_translation(
        source: str, target: str, tx: float, ty: float, tz: float
    ) -> PoseGraphInterpolator:
        """Create a pose graph with a single static translation edge."""
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = [tx, ty, tz]
        return PoseGraphInterpolator(
            [
                PoseGraphInterpolator.Edge(
                    source_node=source, target_node=target, T_source_target=T, timestamps_us=None
                ),
            ]
        )

    @staticmethod
    def _make_pose_graph_with_rotation_z_90(
        source: str, target: str, tx: float = 0.0, ty: float = 0.0, tz: float = 0.0
    ) -> PoseGraphInterpolator:
        """Create a pose graph with a single static 90-degree Z rotation + optional translation."""
        T = np.eye(4, dtype=np.float64)
        T[0, 0] = 0.0
        T[0, 1] = -1.0
        T[1, 0] = 1.0
        T[1, 1] = 0.0
        T[:3, 3] = [tx, ty, tz]
        return PoseGraphInterpolator(
            [
                PoseGraphInterpolator.Edge(
                    source_node=source, target_node=target, T_source_target=T, timestamps_us=None
                ),
            ]
        )

    def test_points_count(self):
        """points_count is derived from xyz shape."""
        pc = self._make_pc(np.zeros((5, 3), dtype=np.float32))
        self.assertEqual(pc.points_count, 5)

    def test_xyz_no_transform(self):
        """xyz returns raw data unchanged when no transform is applied."""
        xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        pc = self._make_pc(xyz)
        npt.assert_array_equal(pc.xyz, xyz)

    def test_xyz_with_transform(self):
        """xyz applies the accumulated pose graph transform lazily."""
        xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        pg = self._make_pose_graph_with_translation("sensor", "world", 10.0, 20.0, 30.0)
        pc = self._make_pc(xyz).transform("world", 0, pg)
        npt.assert_allclose(pc.xyz, [[11.0, 22.0, 33.0]], atol=1e-5)

    def test_xyz_with_f32_coordinates(self):
        """xyz works correctly with float32 input coordinates."""
        xyz = np.array([[1.5, -2.5, 3.5]], dtype=np.float32)
        pg = self._make_pose_graph_with_translation("sensor", "world", 0.5, 0.5, 0.5)
        pc = self._make_pc(xyz).transform("world", 0, pg)
        npt.assert_allclose(pc.xyz, [[2.0, -2.0, 4.0]], atol=1e-5)

    def test_attribute_invariant(self):
        """An INVARIANT attribute (e.g. rgb) is unchanged by a rigid transform."""
        rgb = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        attrs = {
            "rgb": PointCloud.Attribute(loader=lambda: rgb, transform_type=PointCloud.AttributeTransformType.INVARIANT),
        }
        pg = self._make_pose_graph_with_rotation_z_90("sensor", "world", 1.0, 2.0, 3.0)
        pc = self._make_pc(np.zeros((3, 3), dtype=np.float32), attrs).transform("world", 0, pg)
        npt.assert_array_equal(pc.get_attribute("rgb"), rgb)

    def test_attribute_direction(self):
        """A DIRECTION attribute (e.g. normal) is rotated but NOT translated."""
        normal = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        attrs = {
            "normal": PointCloud.Attribute(
                loader=lambda: normal, transform_type=PointCloud.AttributeTransformType.DIRECTION
            ),
        }
        # 90-degree Z rotation with large translation -- translation must NOT affect direction
        pg = self._make_pose_graph_with_rotation_z_90("sensor", "world", 99.0, 99.0, 99.0)
        pc = self._make_pc(np.zeros((1, 3), dtype=np.float32), attrs).transform("world", 0, pg)
        # 90-deg Z rotation: (1,0,0) -> (0,1,0)
        npt.assert_allclose(pc.get_attribute("normal"), [[0.0, 1.0, 0.0]], atol=1e-6)

    def test_attribute_point(self):
        """A POINT attribute (e.g. secondary xyz) gets the full rigid transform."""
        secondary = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        attrs = {
            "secondary": PointCloud.Attribute(
                loader=lambda: secondary, transform_type=PointCloud.AttributeTransformType.POINT
            ),
        }
        pg = self._make_pose_graph_with_rotation_z_90("sensor", "world", 10.0, 20.0, 30.0)
        pc = self._make_pc(np.zeros((1, 3), dtype=np.float32), attrs).transform("world", 0, pg)
        # rotation: (1,0,0) -> (0,1,0), then + (10,20,30) = (10,21,30)
        npt.assert_allclose(pc.get_attribute("secondary"), [[10.0, 21.0, 30.0]], atol=1e-5)

    def test_has_attribute(self):
        """has_attribute returns True for existing and False for missing attributes."""
        attrs = {
            "rgb": PointCloud.Attribute(
                loader=lambda: np.zeros((1, 3)), transform_type=PointCloud.AttributeTransformType.INVARIANT
            ),
        }
        pc = self._make_pc(np.zeros((1, 3), dtype=np.float32), attrs)
        self.assertTrue(pc.has_attribute("rgb"))
        self.assertFalse(pc.has_attribute("normal"))

    def test_attribute_names(self):
        """attribute_names returns the set of all registered attribute names."""
        attrs = {
            "rgb": PointCloud.Attribute(
                loader=lambda: np.zeros((1, 3)), transform_type=PointCloud.AttributeTransformType.INVARIANT
            ),
            "normal": PointCloud.Attribute(
                loader=lambda: np.zeros((1, 3)), transform_type=PointCloud.AttributeTransformType.DIRECTION
            ),
        }
        pc = self._make_pc(np.zeros((1, 3), dtype=np.float32), attrs)
        self.assertEqual(set(pc.attribute_names), {"rgb", "normal"})

    def test_coordinate_unit_preserved(self):
        """CoordinateUnit is preserved through transform()."""
        pg = self._make_pose_graph_with_translation("sensor", "world", 1.0, 2.0, 3.0)
        pc = self._make_pc(
            np.zeros((1, 3), dtype=np.float32), coordinate_unit=PointCloud.CoordinateUnit.UNITLESS
        ).transform("world", 0, pg)
        self.assertEqual(pc.coordinate_unit, PointCloud.CoordinateUnit.UNITLESS)

    def test_lazy_loading(self):
        """Attribute loader is NOT called until get_attribute()."""
        call_count = 0

        def _loader():
            nonlocal call_count
            call_count += 1
            return np.zeros((1, 3), dtype=np.float32)

        attrs = {
            "lazy": PointCloud.Attribute(loader=_loader, transform_type=PointCloud.AttributeTransformType.INVARIANT),
        }
        pc = self._make_pc(np.zeros((1, 3), dtype=np.float32), attrs)
        self.assertEqual(call_count, 0, "loader must not be called during construction")

        pc.get_attribute("lazy")
        self.assertEqual(call_count, 1, "loader should be called once by get_attribute()")

    def test_empty_point_cloud(self):
        """A point cloud with 0 points has correct count and shape."""
        pc = self._make_pc(np.zeros((0, 3), dtype=np.float32))
        self.assertEqual(pc.points_count, 0)
        self.assertEqual(pc.xyz.shape, (0, 3))

    def test_get_attribute_unknown_raises(self):
        """Accessing a non-existent attribute raises KeyError."""
        pc = self._make_pc(np.zeros((1, 3), dtype=np.float32))
        with self.assertRaises(KeyError):
            pc.get_attribute("does_not_exist")

    def test_get_attribute_transform_type(self):
        """get_attribute_transform_type returns the correct enum for each attribute."""
        attrs = {
            "rgb": PointCloud.Attribute(
                loader=lambda: np.zeros((1, 3)), transform_type=PointCloud.AttributeTransformType.INVARIANT
            ),
            "normal": PointCloud.Attribute(
                loader=lambda: np.zeros((1, 3)), transform_type=PointCloud.AttributeTransformType.DIRECTION
            ),
        }
        pc = self._make_pc(np.zeros((1, 3), dtype=np.float32), attrs)
        self.assertEqual(pc.get_attribute_transform_type("rgb"), PointCloud.AttributeTransformType.INVARIANT)
        self.assertEqual(pc.get_attribute_transform_type("normal"), PointCloud.AttributeTransformType.DIRECTION)

    def test_transform_same_frame_is_noop(self):
        """Transforming to the same frame and timestamp returns self unchanged."""
        pc = self._make_pc(
            np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            reference_frame_id="world",
            reference_frame_timestamp_us=100,
        )
        # Pose graph is not consulted because the short-circuit fires first
        pg = self._make_pose_graph_with_translation("world", "dummy", 999.0, 999.0, 999.0)
        pc2 = pc.transform("world", 100, pg)
        self.assertIs(pc2, pc)

    def test_accumulated_transform(self):
        """Chaining two transform() calls accumulates correctly.

        sensor_a -> world translates by (+10, 0, 0).
        sensor_b -> world translates by (0, +20, 0).

        Point (1, 0, 0) in sensor_a:
          -> world: (11, 0, 0)
          -> sensor_b: world point minus sensor_b offset = (11, -20, 0)
        """
        T_a_world = np.eye(4, dtype=np.float64)
        T_a_world[0, 3] = 10.0  # sensor_a -> world: +10 in X

        T_b_world = np.eye(4, dtype=np.float64)
        T_b_world[1, 3] = 20.0  # sensor_b -> world: +20 in Y

        pg = PoseGraphInterpolator(
            [
                PoseGraphInterpolator.Edge("sensor_a", "world", T_a_world, timestamps_us=None),
                PoseGraphInterpolator.Edge("sensor_b", "world", T_b_world, timestamps_us=None),
            ]
        )

        pc = self._make_pc(
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            reference_frame_id="sensor_a",
        )

        # First transform: sensor_a -> world
        pc_world = pc.transform("world", 0, pg)
        npt.assert_allclose(pc_world.xyz, [[11.0, 0.0, 0.0]], atol=1e-5)
        self.assertEqual(pc_world.reference_frame_id, "world")

        # Second transform: world -> sensor_b (accumulated on the same raw data)
        pc_b = pc_world.transform("sensor_b", 0, pg)
        npt.assert_allclose(pc_b.xyz, [[11.0, -20.0, 0.0]], atol=1e-5)
        self.assertEqual(pc_b.reference_frame_id, "sensor_b")

    def test_accumulated_transform_with_direction_attribute(self):
        """Accumulated transforms correctly rotate direction attributes but do not translate them."""
        T_a_world = np.eye(4, dtype=np.float64)
        # sensor_a -> world: 90-degree Z rotation + translation
        T_a_world[0, 0] = 0.0
        T_a_world[0, 1] = -1.0
        T_a_world[1, 0] = 1.0
        T_a_world[1, 1] = 0.0
        T_a_world[0, 3] = 100.0

        T_b_world = np.eye(4, dtype=np.float64)
        # sensor_b -> world: 90-degree Z rotation in opposite direction
        T_b_world[0, 0] = 0.0
        T_b_world[0, 1] = 1.0
        T_b_world[1, 0] = -1.0
        T_b_world[1, 1] = 0.0

        pg = PoseGraphInterpolator(
            [
                PoseGraphInterpolator.Edge("sensor_a", "world", T_a_world, timestamps_us=None),
                PoseGraphInterpolator.Edge("sensor_b", "world", T_b_world, timestamps_us=None),
            ]
        )

        normal = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        attrs = {
            "normal": PointCloud.Attribute(
                loader=lambda: normal, transform_type=PointCloud.AttributeTransformType.DIRECTION
            ),
        }
        pc = self._make_pc(
            np.zeros((1, 3), dtype=np.float32),
            attrs,
            reference_frame_id="sensor_a",
        )

        # sensor_a -> world: rotates (1,0,0) by +90° Z -> (0,1,0)
        pc_world = pc.transform("world", 0, pg)
        npt.assert_allclose(pc_world.get_attribute("normal"), [[0.0, 1.0, 0.0]], atol=1e-6)

        # world -> sensor_b: sensor_b->world is -90° Z, so world->sensor_b is +90° Z
        # (0,1,0) rotated +90° Z -> (-1,0,0)
        pc_b = pc_world.transform("sensor_b", 0, pg)
        npt.assert_allclose(pc_b.get_attribute("normal"), [[-1.0, 0.0, 0.0]], atol=1e-6)


def _encode_png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    PILImage.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


class TestEncodedImageData(unittest.TestCase):
    """Tests for EncodedImageData decode."""

    @staticmethod
    def _sample_png() -> bytes:
        rng = np.random.default_rng(0)
        return _encode_png(rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8))

    def test_decode_roundtrip(self) -> None:
        """get_decoded_image() returns a usable image of the expected size."""
        image = EncodedImageData(self._sample_png(), "PNG").get_decoded_image()
        self.assertEqual(np.asarray(image.convert("RGB")).shape, (8, 8, 3))

    def test_decode_is_memoized(self) -> None:
        """Repeated calls return the same decoded object (per-instance memoization)."""
        image_data = EncodedImageData(self._sample_png(), "PNG")
        self.assertIs(image_data.get_decoded_image(), image_data.get_decoded_image())

    def test_memoization_survives_interleaved_instances(self) -> None:
        """Memoization is stable across interleaved use of other instances.

        Guards against regressing to a method-level ``lru_cache(maxsize=1)``,
        whose single global slot would be evicted by the intervening
        ``other`` call, so ``first.get_decoded_image()`` would re-decode and
        return a different object on the second call.
        """
        png = self._sample_png()
        first = EncodedImageData(png, "PNG")
        other = EncodedImageData(png, "PNG")
        a1 = first.get_decoded_image()
        other.get_decoded_image()  # would evict a method-level maxsize=1 cache
        a2 = first.get_decoded_image()
        self.assertIs(a1, a2)


class TestCameraModelParametersTransformSelf(unittest.TestCase):
    """`transform` must preserve the concrete parameter type, statically and at runtime"""

    _RESOLUTION = np.array([640, 480], dtype=np.uint64)
    _PRINCIPAL_POINT = np.array([320.0, 240.0], dtype=np.float32)
    _FOCAL_LENGTH = np.array([500.0, 500.0], dtype=np.float32)

    @classmethod
    def _ftheta(cls) -> FThetaCameraModelParameters:
        return FThetaCameraModelParameters(
            resolution=cls._RESOLUTION,
            shutter_type=ShutterType.GLOBAL,
            principal_point=cls._PRINCIPAL_POINT,
            reference_poly=FThetaCameraModelParameters.PolynomialType.ANGLE_TO_PIXELDIST,
            pixeldist_to_angle_poly=np.array([0.0, 0.0008, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            angle_to_pixeldist_poly=np.array([0.0, 1250.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            max_angle=float(np.radians(45.0)),
            linear_cde=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )

    @classmethod
    def _all_parameters(cls) -> List[CameraModelParameters]:
        resolution = cls._RESOLUTION
        principal_point = cls._PRINCIPAL_POINT
        focal_length = cls._FOCAL_LENGTH
        return [
            cls._ftheta(),
            IdealPinholeCameraModelParameters(
                resolution=resolution,
                shutter_type=ShutterType.GLOBAL,
                principal_point=principal_point,
                focal_length=focal_length,
            ),
            OpenCVPinholeCameraModelParameters(
                resolution=resolution,
                shutter_type=ShutterType.GLOBAL,
                principal_point=principal_point,
                focal_length=focal_length,
                radial_coeffs=np.zeros(6, dtype=np.float32),
                tangential_coeffs=np.zeros(2, dtype=np.float32),
                thin_prism_coeffs=np.zeros(4, dtype=np.float32),
            ),
            OpenCVFisheyeCameraModelParameters(
                resolution=resolution,
                shutter_type=ShutterType.GLOBAL,
                principal_point=principal_point,
                focal_length=focal_length,
                radial_coeffs=np.zeros(4, dtype=np.float32),
                max_angle=float(np.radians(70.0)),
            ),
        ]

    def test_transform_preserves_concrete_type(self):
        # Statically, `_downscale` is generic over the abstract base and returns the caller's own
        # type; that only holds while every override returns `Self` rather than its own class.
        for parameters in self._all_parameters():
            with self.subTest(model=parameters.type()):
                transformed = _downscale(parameters, 0.5)
                self.assertIs(type(transformed), type(parameters))
                npt.assert_array_equal(transformed.resolution, np.array([320, 240], dtype=np.uint64))

    def test_generic_helper_keeps_the_concrete_static_type(self):
        # Assigning to the concrete type is what pins the `Self` behaviour for the type checker;
        # with a non-`Self` override this narrows to the declaring class and fails to check. This
        # needs a statically concrete input, so it takes `_ftheta()` rather than an element of
        # `_all_parameters()`, which is typed as the abstract base.
        ftheta: FThetaCameraModelParameters = _downscale(self._ftheta(), 0.5)
        self.assertIsInstance(ftheta.angle_to_pixeldist_poly, np.ndarray)


if __name__ == "__main__":
    unittest.main()
