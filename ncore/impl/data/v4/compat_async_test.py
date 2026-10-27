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

"""Asynchronous reads of SequenceLoaderV4 (AsyncSequenceLoaderProtocol): each ``*_async`` method returns the same data
as its synchronous method, and reads it via the asynchronous array reads (not via the synchronous ones)."""

from __future__ import annotations

import asyncio
import contextlib
import io
import tempfile
import unittest

from typing import Any, Awaitable, Callable, Iterator, List, Literal, Optional, Tuple, TypeVar
from unittest import mock

import numpy as np
import PIL.Image as PILImage

from parameterized import parameterized_class
from python.runfiles import Runfiles
from upath import UPath

from ncore.impl.common.transformations import HalfClosedInterval
from ncore.impl.data import nodes
from ncore.impl.data.compat import (
    AsyncCameraLabelsProtocol,
    AsyncCameraSensorProtocol,
    AsyncLidarSensorProtocol,
    AsyncPointCloudsSourceProtocol,
    AsyncRadarSensorProtocol,
    AsyncSequenceLoaderProtocol,
    RayBundleSensorProtocol,
)
from ncore.impl.data.types import (
    CameraLabelDescriptor,
    EncodedImageData,
    LabelEncoding,
    LabelSchema,
    LabelSource,
    LabelType,
    PointCloud,
)
from ncore.impl.data.v4.compat import SequenceLoaderV4
from ncore.impl.data.v4.components import (
    CameraLabelsComponent,
    IntrinsicsComponent,
    LidarSensorComponent,
    PointCloudsComponent,
    PosesComponent,
    SequenceComponentGroupsReader,
    SequenceComponentGroupsWriter,
)


_RUNFILES = Runfiles.Create()
assert _RUNFILES is not None

_TEST_DATA_PATH = UPath(
    _RUNFILES.Rlocation("test-data-v4/c9b05cf4-afb9-11ec-b3c2-00044bf65fcb@1648597318700123-1648599151600035.json")
    or ""
)

T = TypeVar("T")


@contextlib.contextmanager
def _array_reads() -> Iterator[List[str]]:
    """Records the array reads issued while active: "async" for asynchronous reads, "sync" for synchronous reads"""
    reads: List[str] = []
    read, read_async = nodes.Array.read, nodes.Array.read_async
    read_bytes, read_bytes_async = nodes.Array.read_bytes, nodes.Array.read_bytes_async

    def sync(f: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            reads.append("sync")
            return f(*args, **kwargs)

        return wrapped

    def asynchronous(f: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            reads.append("async")
            return await f(*args, **kwargs)

        return wrapped

    with contextlib.ExitStack() as patches:
        for name, wrapper in (
            ("read", sync(read)),
            ("read_async", asynchronous(read_async)),
            ("read_bytes", sync(read_bytes)),
            ("read_bytes_async", asynchronous(read_bytes_async)),
        ):
            patches.enter_context(mock.patch.object(nodes.Array, name, wrapper))
        yield reads


class _AsyncReadsTestCase(unittest.TestCase):
    def run_async(self, read: Awaitable[T], expected_reads: Optional[int] = None) -> T:
        """Awaits an asynchronous read, asserting that it reads data via asynchronous array reads only (and the number
        of them, if given)"""

        async def run() -> T:
            return await read

        with _array_reads() as reads:
            result = asyncio.run(run())
        self.assertNotIn("sync", reads, "asynchronous read issued synchronous array reads")
        self.assertIn("async", reads, "asynchronous read issued no asynchronous array reads")
        if expected_reads is not None:
            self.assertEqual(len(reads), expected_reads)
        return result

    def assertFramePointCloudEqual(
        self, a: RayBundleSensorProtocol.FramePointCloud, b: RayBundleSensorProtocol.FramePointCloud
    ) -> None:
        self.assertEqual(a.motion_compensation, b.motion_compensation)
        np.testing.assert_array_equal(a.xyz_m_end, b.xyz_m_end)
        if a.xyz_m_start is None or b.xyz_m_start is None:
            self.assertIs(a.xyz_m_start, b.xyz_m_start)
        else:
            np.testing.assert_array_equal(a.xyz_m_start, b.xyz_m_start)

    def assertPointCloudEqual(self, a: PointCloud, b: PointCloud) -> None:
        np.testing.assert_array_equal(a.xyz, b.xyz)
        self.assertEqual(a.reference_frame_id, b.reference_frame_id)
        self.assertEqual(a.reference_frame_timestamp_us, b.reference_frame_timestamp_us)
        self.assertEqual(a.coordinate_unit, b.coordinate_unit)
        self.assertEqual(a.attribute_names, b.attribute_names)
        for name in a.attribute_names:
            np.testing.assert_array_equal(a.get_attribute(name), b.get_attribute(name))
            self.assertEqual(a.get_attribute_transform_type(name), b.get_attribute_transform_type(name))


class TestSequenceLoaderV4AsyncReads(_AsyncReadsTestCase):
    """Asynchronous reads of cameras, lidars and radars of a recorded sequence"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.loader = SequenceLoaderV4(SequenceComponentGroupsReader([_TEST_DATA_PATH]))

    def test_protocols(self) -> None:
        loader = self.loader
        self.assertIsInstance(loader, AsyncSequenceLoaderProtocol)
        self.assertIsInstance(loader.get_camera_sensor(loader.camera_ids[0]), AsyncCameraSensorProtocol)
        self.assertIsInstance(loader.get_lidar_sensor(loader.lidar_ids[0]), AsyncLidarSensorProtocol)
        self.assertIsInstance(loader.get_radar_sensor(loader.radar_ids[0]), AsyncRadarSensorProtocol)
        self.assertIsInstance(
            loader.get_point_clouds_source(loader.lidar_ids[0]),
            AsyncPointCloudsSourceProtocol,
        )

    def test_camera_frame_data(self) -> None:
        camera = self.loader.get_camera_sensor("camera_front_wide_120fov")
        for frame_index in (0, camera.frames_count - 1):
            with self.subTest(frame_index=frame_index):
                expected = camera.get_frame_data(frame_index)
                data = self.run_async(camera.get_frame_data_async(frame_index), expected_reads=1)
                self.assertIsInstance(data, EncodedImageData)
                self.assertEqual(data.get_encoded_image_data(), expected.get_encoded_image_data())
                self.assertEqual(data.get_encoded_image_format(), expected.get_encoded_image_format())
                handle_data = self.run_async(camera.get_frame_handle(frame_index).get_data_async(), expected_reads=1)
                self.assertEqual(handle_data.get_encoded_image_data(), expected.get_encoded_image_data())

    def test_camera_frames_concurrently(self) -> None:
        camera = self.loader.get_camera_sensor("camera_front_wide_120fov")
        frame_indices = list(range(camera.frames_count))

        async def read_all() -> Tuple[EncodedImageData, ...]:
            return tuple(await asyncio.gather(*(camera.get_frame_data_async(i) for i in frame_indices)))

        frames = self.run_async(read_all(), expected_reads=len(frame_indices))
        for frame_index, data in zip(frame_indices, frames):
            self.assertEqual(data.get_encoded_image_data(), camera.get_frame_data(frame_index).get_encoded_image_data())

    def test_camera_mask_images(self) -> None:
        camera = self.loader.get_camera_sensor("camera_front_wide_120fov")
        expected = camera.get_mask_images()
        self.assertTrue(expected)
        masks = self.run_async(camera.get_mask_images_async(), expected_reads=len(expected))
        self.assertEqual(list(masks), list(expected))
        for name, image in masks.items():
            np.testing.assert_array_equal(np.asarray(image), np.asarray(expected[name]))

    def test_sensor_generic_data(self) -> None:
        radar = self.loader.get_radar_sensor(self.loader.radar_ids[0])
        names = radar.get_frame_generic_data_names(0)
        self.assertTrue(names)
        for name in names:
            with self.subTest(name=name):
                np.testing.assert_array_equal(
                    self.run_async(radar.get_frame_generic_data_async(0, name), expected_reads=1),
                    radar.get_frame_generic_data(0, name),
                )

    def test_ray_bundle_reads(self) -> None:
        for sensor in (
            self.loader.get_lidar_sensor(self.loader.lidar_ids[0]),
            self.loader.get_radar_sensor(self.loader.radar_ids[0]),
        ):
            for frame_index in (0, sensor.frames_count - 1):
                with self.subTest(sensor=sensor.sensor_id, frame_index=frame_index):
                    np.testing.assert_array_equal(
                        self.run_async(sensor.get_frame_ray_bundle_direction_async(frame_index), expected_reads=1),
                        sensor.get_frame_ray_bundle_direction(frame_index),
                    )
                    np.testing.assert_array_equal(
                        self.run_async(sensor.get_frame_ray_bundle_timestamp_us_async(frame_index), expected_reads=1),
                        sensor.get_frame_ray_bundle_timestamp_us(frame_index),
                    )
                    np.testing.assert_array_equal(
                        self.run_async(
                            sensor.get_frame_ray_bundle_return_distance_m_async(frame_index), expected_reads=1
                        ),
                        sensor.get_frame_ray_bundle_return_distance_m(frame_index),
                    )
                    np.testing.assert_array_equal(
                        self.run_async(
                            sensor.get_frame_ray_bundle_return_valid_mask_async(frame_index), expected_reads=1
                        ),
                        sensor.get_frame_ray_bundle_return_valid_mask(frame_index),
                    )

    def test_lidar_reads(self) -> None:
        lidar = self.loader.get_lidar_sensor(self.loader.lidar_ids[0])
        np.testing.assert_array_equal(
            self.run_async(lidar.get_frame_ray_bundle_return_intensity_async(0), expected_reads=1),
            lidar.get_frame_ray_bundle_return_intensity(0),
        )
        # (the recorded lidar has no model elements: returns None without reading)
        self.assertIsNone(lidar.get_frame_ray_bundle_model_element(0))
        self.assertIsNone(asyncio.run(lidar.get_frame_ray_bundle_model_element_async(0)))

    def test_frame_point_cloud(self) -> None:
        for sensor in (
            self.loader.get_lidar_sensor(self.loader.lidar_ids[0]),
            self.loader.get_radar_sensor(self.loader.radar_ids[0]),
        ):
            for motion_compensation in (False, True):
                for with_start_points in (False, True):
                    with self.subTest(
                        sensor=sensor.sensor_id,
                        motion_compensation=motion_compensation,
                        with_start_points=with_start_points,
                    ):
                        self.assertFramePointCloudEqual(
                            self.run_async(
                                sensor.get_frame_point_cloud_async(1, motion_compensation, with_start_points),
                                # direction + distance (+ per-ray timestamps for motion compensation)
                                expected_reads=3 if motion_compensation else 2,
                            ),
                            sensor.get_frame_point_cloud(1, motion_compensation, with_start_points),
                        )


class TestSequenceLoaderV4OpenAsync(unittest.TestCase):
    """SequenceLoaderV4.open_async opens the same sequence as the synchronous constructor, also many sequences
    concurrently"""

    def test_open_async(self) -> None:
        expected = SequenceLoaderV4(SequenceComponentGroupsReader([_TEST_DATA_PATH]))
        loader = asyncio.run(SequenceLoaderV4.open_async([_TEST_DATA_PATH]))
        self.assertIsInstance(loader, AsyncSequenceLoaderProtocol)
        self.assertEqual(loader.sequence_id, expected.sequence_id)
        self.assertEqual(loader.camera_ids, expected.camera_ids)
        self.assertEqual(loader.lidar_ids, expected.lidar_ids)
        self.assertEqual(loader.radar_ids, expected.radar_ids)
        camera = loader.get_camera_sensor("camera_front_wide_120fov")
        np.testing.assert_array_equal(
            camera.frames_timestamps_us, expected.get_camera_sensor("camera_front_wide_120fov").frames_timestamps_us
        )
        self.assertEqual(
            asyncio.run(camera.get_frame_data_async(0)).get_encoded_image_data(),
            expected.get_camera_sensor("camera_front_wide_120fov").get_frame_data(0).get_encoded_image_data(),
        )

    def test_open_many_concurrently(self) -> None:
        async def open_all() -> List[SequenceLoaderV4]:
            limit = asyncio.Semaphore(8)
            return list(
                await asyncio.gather(
                    *(SequenceLoaderV4.open_async([_TEST_DATA_PATH], max_concurrent_requests=limit) for _ in range(4))
                )
            )

        loaders = asyncio.run(open_all())
        self.assertEqual(len(loaders), 4)
        self.assertEqual({loader.sequence_id for loader in loaders}, {loaders[0].sequence_id})

    def test_open_async_index_tail_retention(self) -> None:
        """The index tail retention is passed to the reader (RELEASE_IF_CONSOLIDATED by default)"""
        with mock.patch.object(
            SequenceComponentGroupsReader, "open_async", wraps=SequenceComponentGroupsReader.open_async
        ) as open_async:
            asyncio.run(SequenceLoaderV4.open_async([_TEST_DATA_PATH]))
            asyncio.run(SequenceLoaderV4.open_async([_TEST_DATA_PATH], itar_index_tail_retention="keep"))
        self.assertEqual(
            [call.kwargs["itar_index_tail_retention"] for call in open_async.call_args_list],
            ["release_if_consolidated", "keep"],
        )


@parameterized_class(("store_type",), [("itar",), ("directory",)])
class TestSequenceLoaderV4AsyncPointCloudsAndLabels(_AsyncReadsTestCase):
    """Asynchronous reads of native and lidar-adapted point-clouds sources (multi-return lidar with model elements),
    and of camera labels (raw and image-encoded)"""

    store_type: Literal["itar", "directory"]

    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(42)

        store_writer = SequenceComponentGroupsWriter(
            output_dir_path=UPath(self._tempdir.name),
            store_base_name="async-compat-test",
            sequence_id="async-compat-test",
            sequence_timestamp_interval_us=HalfClosedInterval(0, 1_000_001),
            store_type=self.store_type,
            generic_meta_data={},
        )

        # (a moving rig, so motion compensation transforms the points)
        T_rig_worlds = np.stack([np.eye(4, dtype=np.float64)] * 3)
        T_rig_worlds[:, 0, 3] = [0.0, 1.0, 2.0]
        poses_writer = store_writer.register_component_writer(PosesComponent.Writer, "default", group_name=None)
        poses_writer.store_dynamic_pose(
            source_frame_id="rig",
            target_frame_id="world",
            poses=T_rig_worlds,
            timestamps_us=np.array([0, 500_000, 1_000_000], dtype=np.uint64),
        )
        poses_writer.store_static_pose(source_frame_id="lidar", target_frame_id="rig", pose=np.eye(4, dtype=np.float32))

        store_writer.register_component_writer(IntrinsicsComponent.Writer, "default", "intrinsics")

        # 2 frames: 5 rays / 1 return / model elements, 8 rays / 2 returns (some absent) / no model elements
        lidar_writer = store_writer.register_component_writer(LidarSensorComponent.Writer, "lidar", "lidars")
        for frame_index, (n_rays, n_returns, with_model_element) in enumerate(((5, 1, True), (8, 2, False))):
            start_us, end_us = (0, 500_000) if frame_index == 0 else (500_001, 1_000_000)
            directions = rng.random((n_rays, 3)).astype(np.float32) + 0.1
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            distance_m = rng.uniform(1, 50, size=(n_returns, n_rays)).astype(np.float32)
            intensity = rng.random((n_returns, n_rays)).astype(np.float32)
            if n_returns > 1:
                distance_m[1, 3:6] = intensity[1, 3:6] = np.nan  # (absent returns)
            lidar_writer.store_frame(
                directions,
                np.linspace(start_us, end_us, num=n_rays, dtype=np.uint64),
                np.arange(n_rays * 2, dtype=np.uint16).reshape(n_rays, 2) if with_model_element else None,
                distance_m,
                intensity,
                np.array([start_us, end_us], dtype=np.uint64),
                generic_data={"rgb": rng.integers(0, 256, size=(n_rays, 3), dtype=np.uint8)},
                generic_meta_data={},
            )

        pc_writer = store_writer.register_component_writer(
            PointCloudsComponent.Writer,
            "sfm",
            coordinate_unit=PointCloud.CoordinateUnit.METERS,
            attribute_schemas={
                "rgb": PointCloudsComponent.AttributeSchema(
                    transform_type=PointCloud.AttributeTransformType.INVARIANT,
                    dtype=np.dtype("uint8"),
                    shape_suffix=(3,),
                )
            },
        )
        pc_writer.store_pc(
            xyz=rng.normal(size=(20, 3)).astype(np.float32),
            reference_frame_id="world",
            reference_frame_timestamp_us=500_000,
            attributes={"rgb": rng.integers(0, 255, size=(20, 3), dtype=np.uint8)},
            generic_data={"score": rng.uniform(size=(20,)).astype(np.float32)},
            generic_meta_data={},
        )

        # camera labels: raw depth, PNG-encoded segmentation
        PILImage.fromarray(rng.integers(0, 20, size=(12, 16), dtype=np.uint8), mode="L").save(
            png := io.BytesIO(), format="PNG"
        )
        for label_type, schema, data in (
            (
                LabelType.DEPTH_Z_M,
                LabelSchema(dtype=np.dtype("float32"), shape_suffix=(), encoding=LabelEncoding.RAW),
                rng.random((12, 16), dtype=np.float32),
            ),
            (
                LabelType.SEGMENTATION_SEMANTIC,
                LabelSchema(dtype=np.dtype("uint8"), encoding=LabelEncoding.IMAGE_ENCODED, encoded_format="png"),
                png.getvalue(),
            ),
        ):
            descriptor = CameraLabelDescriptor(
                camera_id="front", label_type=label_type, label_schema=schema, label_source=LabelSource.AUTOLABEL
            )
            labels_writer = store_writer.register_component_writer(
                CameraLabelsComponent.Writer,
                descriptor.default_instance_name,
                generic_meta_data={},
                descriptor=descriptor,
            )
            labels_writer.store_label(data=data, timestamp_us=500_000)

        reader = SequenceComponentGroupsReader(store_writer.finalize())
        self.loader = SequenceLoaderV4(reader, masks_component_group_name=None, cuboids_component_group_name=None)

    def tearDown(self) -> None:
        self._tempdir.cleanup()

    def test_protocols(self) -> None:
        # (in a test of its own: isinstance checks of runtime-checkable protocols evaluate properties, which may load
        # data the read tests must not have loaded before)
        self.assertIsInstance(self.loader.get_point_clouds_source("sfm"), AsyncPointCloudsSourceProtocol)
        self.assertIsInstance(self.loader.get_point_clouds_source("lidar"), AsyncPointCloudsSourceProtocol)
        for camera_labels in self.loader.query_camera_labels("front"):
            self.assertIsInstance(camera_labels, AsyncCameraLabelsProtocol)
            self.assertIsInstance(
                self.loader.get_camera_labels(camera_labels.label_descriptor.default_instance_name),
                AsyncCameraLabelsProtocol,
            )

    def test_multi_return_reads(self) -> None:
        lidar = self.loader.get_lidar_sensor("lidar")
        for return_index in (0, 1):
            with self.subTest(return_index=return_index):
                np.testing.assert_array_equal(
                    self.run_async(lidar.get_frame_ray_bundle_return_distance_m_async(1, return_index)),
                    lidar.get_frame_ray_bundle_return_distance_m(1, return_index),
                )
                np.testing.assert_array_equal(
                    self.run_async(lidar.get_frame_ray_bundle_return_intensity_async(1, return_index)),
                    lidar.get_frame_ray_bundle_return_intensity(1, return_index),
                )
                np.testing.assert_array_equal(
                    self.run_async(lidar.get_frame_ray_bundle_return_valid_mask_async(1, return_index)),
                    lidar.get_frame_ray_bundle_return_valid_mask(1, return_index),
                )
                self.assertFramePointCloudEqual(
                    self.run_async(lidar.get_frame_point_cloud_async(1, True, True, return_index)),
                    lidar.get_frame_point_cloud(1, True, True, return_index),
                )
        # the returns differ (the test reads the requested return)
        self.assertFalse(
            np.array_equal(
                lidar.get_frame_ray_bundle_return_distance_m(1, 0), lidar.get_frame_ray_bundle_return_distance_m(1, 1)
            )
        )

    def test_model_element(self) -> None:
        lidar = self.loader.get_lidar_sensor("lidar")
        model_element = unpack(self.run_async(lidar.get_frame_ray_bundle_model_element_async(0), expected_reads=1))
        np.testing.assert_array_equal(model_element, unpack(lidar.get_frame_ray_bundle_model_element(0)))
        self.assertIsNone(asyncio.run(lidar.get_frame_ray_bundle_model_element_async(1)))

    def test_native_point_clouds_source(self) -> None:
        source = self.loader.get_point_clouds_source("sfm")
        # (the source's timestamps are metadata, loaded synchronously on first use: like a sensor's frame timestamps)
        self.assertEqual(source.pcs_count, 1)
        self.assertPointCloudEqual(self.run_async(source.get_pc_async(0), expected_reads=1), source.get_pc(0))
        np.testing.assert_array_equal(
            self.run_async(source.get_pc_generic_data_async(0, "score"), expected_reads=1),
            source.get_pc_generic_data(0, "score"),
        )

    def test_lidar_adapted_point_clouds_source(self) -> None:
        for return_index in (0, 1):
            source = self.loader.get_point_clouds_source("lidar", return_index=return_index)
            for pc_index in range(source.pcs_count):
                if return_index >= self.loader.get_lidar_sensor("lidar").get_frame_ray_bundle_return_count(pc_index):
                    continue
                with self.subTest(return_index=return_index, pc_index=pc_index):
                    self.assertPointCloudEqual(self.run_async(source.get_pc_async(pc_index)), source.get_pc(pc_index))
                    np.testing.assert_array_equal(
                        self.run_async(source.get_pc_generic_data_async(pc_index, "rgb"), expected_reads=1),
                        source.get_pc_generic_data(pc_index, "rgb"),
                    )
        # (model elements are part of the adapted point cloud of frame 0)
        self.assertIn(
            "model_element", asyncio.run(self.loader.get_point_clouds_source("lidar").get_pc_async(0)).attribute_names
        )

    def test_camera_labels(self) -> None:
        labels = self.loader.query_camera_labels("front")
        self.assertEqual(len(labels), 2)
        for camera_labels in labels:
            handle = camera_labels.get_label(500_000)
            with self.subTest(label_type=camera_labels.label_descriptor.label_type):
                np.testing.assert_array_equal(
                    self.run_async(handle.get_data_async(), expected_reads=1), handle.get_data()
                )
                encoded = handle.get_encoded_data()
                if encoded is None:  # (raw labels: no encoded data, nothing read)
                    self.assertIsNone(asyncio.run(handle.get_encoded_data_async()))
                else:
                    self.assertEqual(self.run_async(handle.get_encoded_data_async(), expected_reads=1), encoded)


def unpack(value: Optional[T]) -> T:
    assert value is not None
    return value


if __name__ == "__main__":
    unittest.main()
