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

"""Cross-format / cross-runtime consistency of NCore V4 test data

Compares the zarr format v2 and zarr format v3 serializations of the same (time-trimmed) test sequence, as well as
re-serializations performed by the zarr-python version under test.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest

from typing import Any, Dict, List, Literal, Optional, cast

import numpy as np
import parameterized

from python.runfiles import Runfiles
from upath import UPath

from ncore.impl.data import nodes
from ncore.impl.data.v4.components import (
    CameraSensorComponent,
    CuboidsComponent,
    IntrinsicsComponent,
    LidarSensorComponent,
    MasksComponent,
    PosesComponent,
    RadarSensorComponent,
    SequenceComponentGroupsReader,
    SequenceComponentGroupsWriter,
)


_RUNFILES = Runfiles.Create()
assert _RUNFILES is not None
_RUNFILES_: Runfiles = _RUNFILES

_SEQUENCE = "c9b05cf4-afb9-11ec-b3c2-00044bf65fcb@1648597318700123-1648599151600035"


def _dataset(name: str) -> UPath:
    path = UPath(_RUNFILES_.Rlocation(f"test-data-v4-trimmed/{name}/{_SEQUENCE}.json") or "")
    assert path.exists(), f"test data {name} not found (missing data dependency?)"
    return path


_TRIMMED_V2 = _dataset("zarr-v2")
_TRIMMED_V3 = _dataset("zarr-v3")


#: Frame subsampling of the reserialization test (every N-th frame, plus the last frame of each sensor)
_FRAME_STRIDE = 5


def _subsampled(frames_timestamps_us: np.ndarray) -> np.ndarray:
    """Subsampled frames (every _FRAME_STRIDE-th frame, plus the last frame)"""
    indices = sorted(set(range(0, len(frames_timestamps_us), _FRAME_STRIDE)) | {len(frames_timestamps_us) - 1})
    return frames_timestamps_us[[i for i in indices if i >= 0]]


def _summary(reader: SequenceComponentGroupsReader, subsampled: bool = False, frames: bool = True) -> Dict[str, Any]:
    """Collects a comparable summary of all data of a sequence (optionally of the subsampled frames only, or without
    masks and sensor frames)"""

    def selected(timestamps_us: np.ndarray) -> np.ndarray:
        return _subsampled(timestamps_us) if subsampled else timestamps_us

    out: Dict[str, Any] = {
        "sequence_id": reader.sequence_id,
        "interval": (reader.sequence_timestamp_interval_us.start, reader.sequence_timestamp_interval_us.stop),
        "generic_meta_data": reader.generic_meta_data,
    }

    for name, poses in sorted(reader.open_component_readers(PosesComponent.Reader).items()):
        out[f"poses/{name}/static"] = {k: v.tolist() for k, v in poses.get_static_poses()}
        out[f"poses/{name}/dynamic"] = {k: (p.tolist(), t.tolist()) for k, (p, t) in poses.get_dynamic_poses()}

    for name, intrinsics in sorted(reader.open_component_readers(IntrinsicsComponent.Reader).items()):
        out[f"intrinsics/{name}"] = (intrinsics._group.group("cameras").attrs, intrinsics._group.group("lidars").attrs)

    for name, cuboids in sorted(reader.open_component_readers(CuboidsComponent.Reader).items()):
        out[f"cuboids/{name}"] = [obs.to_dict() for obs in cuboids.get_observations()]

    if not frames:
        return out

    for name, masks in sorted(reader.open_component_readers(MasksComponent.Reader).items()):
        out[f"masks/{name}"] = {
            camera_id: {
                mask_name: np.asarray(image).tobytes() for mask_name, image in masks.get_camera_mask_images(camera_id)
            }
            for camera_id in masks._group.group("cameras").members()
        }

    for name, cameras in sorted(reader.open_component_readers(CameraSensorComponent.Reader).items()):
        frames: List[object] = []
        for ts in selected(cameras.frames_timestamps_us)[:, 1]:
            ts = int(ts)
            data = cameras.get_frame_data(ts)
            frames.append(
                (
                    ts,
                    data.get_encoded_image_format(),
                    data.get_encoded_image_data(),
                    {
                        n: cameras.get_frame_generic_data(ts, n).tobytes()
                        for n in sorted(cameras.get_frame_generic_data_names(ts))
                    },
                    cameras.get_frame_generic_meta_data(ts),
                )
            )
        out[f"cameras/{name}"] = frames

    for component in (LidarSensorComponent, RadarSensorComponent):
        for name, sensor in sorted(reader.open_component_readers(component.Reader).items()):
            sensor_frames = []
            for ts in selected(sensor.frames_timestamps_us)[:, 1]:
                ts = int(ts)
                sensor_frames.append(
                    (
                        ts,
                        {
                            n: sensor.get_frame_ray_bundle_data(ts, n).tobytes()
                            for n in sorted(sensor.get_frame_ray_bundle_data_names(ts))
                        },
                        {
                            n: sensor.get_frame_ray_bundle_return_data(ts, n, None).tobytes()
                            for n in sorted(sensor.get_frame_ray_bundle_return_data_names(ts))
                        },
                        sensor.get_frame_ray_bundle_return_valid_mask(ts).tobytes(),
                        {
                            n: sensor.get_frame_generic_data(ts, n).tobytes()
                            for n in sorted(sensor.get_frame_generic_data_names(ts))
                        },
                        sensor.get_frame_generic_meta_data(ts),
                    )
                )
            out[f"{component.COMPONENT_NAME}/{name}"] = sensor_frames

    return out


async def _summary_async(reader: SequenceComponentGroupsReader) -> Dict[str, Any]:
    """:func:`_summary` via the asynchronous reader APIs, all frames of all sensors read concurrently"""
    out = _summary(reader, subsampled=False, frames=False)

    async def masks(masks: MasksComponent.Reader) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for camera_id in masks._group.group("cameras").members():
            out[camera_id] = {}
            for mask_name in masks.get_camera_mask_names(camera_id):
                image = await masks.get_camera_mask_image_async(camera_id, mask_name)
                out[camera_id][mask_name] = np.asarray(image).tobytes()
        return out

    async def camera_frame(cameras: CameraSensorComponent.Reader, ts: int) -> object:
        data, generic = await asyncio.gather(
            cameras.get_frame_data_async(ts),
            asyncio.gather(
                *(cameras.get_frame_generic_data_async(ts, n) for n in sorted(cameras.get_frame_generic_data_names(ts)))
            ),
        )
        names = sorted(cameras.get_frame_generic_data_names(ts))
        return (
            ts,
            data.get_encoded_image_format(),
            data.get_encoded_image_data(),
            {n: value.tobytes() for n, value in zip(names, generic)},
            cameras.get_frame_generic_meta_data(ts),
        )

    async def sensor_frame(sensor: Any, ts: int) -> object:
        ray_names = sorted(sensor.get_frame_ray_bundle_data_names(ts))
        return_names = sorted(sensor.get_frame_ray_bundle_return_data_names(ts))
        generic_names = sorted(sensor.get_frame_generic_data_names(ts))
        rays, returns, mask, generic = await asyncio.gather(
            asyncio.gather(*(sensor.get_frame_ray_bundle_data_async(ts, n) for n in ray_names)),
            asyncio.gather(*(sensor.get_frame_ray_bundle_return_data_async(ts, n, None) for n in return_names)),
            sensor.get_frame_ray_bundle_return_valid_mask_async(ts),
            asyncio.gather(*(sensor.get_frame_generic_data_async(ts, n) for n in generic_names)),
        )
        return (
            ts,
            {n: v.tobytes() for n, v in zip(ray_names, rays)},
            {n: v.tobytes() for n, v in zip(return_names, returns)},
            mask.tobytes(),
            {n: v.tobytes() for n, v in zip(generic_names, generic)},
            sensor.get_frame_generic_meta_data(ts),
        )

    for name, masks_reader in sorted(reader.open_component_readers(MasksComponent.Reader).items()):
        out[f"masks/{name}"] = await masks(masks_reader)
    for name, cameras in sorted(reader.open_component_readers(CameraSensorComponent.Reader).items()):
        out[f"cameras/{name}"] = list(
            await asyncio.gather(*(camera_frame(cameras, int(ts)) for ts in cameras.frames_timestamps_us[:, 1]))
        )
    for component in (LidarSensorComponent, RadarSensorComponent):
        for name, sensor in sorted(reader.open_component_readers(component.Reader).items()):
            out[f"{component.COMPONENT_NAME}/{name}"] = list(
                await asyncio.gather(*(sensor_frame(sensor, int(ts)) for ts in sensor.frames_timestamps_us[:, 1]))
            )
    return out


class TestZarrFormats(unittest.TestCase):
    """zarr format v2 and v3 serializations of a sequence need to be identical in content"""

    reference: Dict[str, Any]  # summary of the zarr format v2 test data
    reference_subsampled: Dict[str, Any]  # summary of the subsampled frames of the zarr format v2 test data

    @classmethod
    def setUpClass(cls) -> None:
        cls.reference = _summary(SequenceComponentGroupsReader([_TRIMMED_V2]))
        cls.reference_subsampled = _summary(SequenceComponentGroupsReader([_TRIMMED_V2]), subsampled=True)

    def test_v2_v3_consistency(self):
        if 3 not in nodes.SUPPORTED_ZARR_FORMATS:
            # zarr format v3 data can't be read with zarr-python 2
            with self.assertRaises(Exception):
                SequenceComponentGroupsReader([_TRIMMED_V3])
            return

        _assert_summaries_equal(self, self.reference, _summary(SequenceComponentGroupsReader([_TRIMMED_V3])))

    @parameterized.parameterized.expand([(None,), (0,), (1,), (16,)])
    def test_node_cache_sizes(self, node_cache_size: Optional[int]):
        """All node cache configurations of the readers return the same data, and caches respect their bounds"""
        reader = SequenceComponentGroupsReader([_TRIMMED_V2], node_cache_size=node_cache_size)
        _assert_summaries_equal(self, self.reference, _summary(reader))

        lidar = next(iter(reader.open_component_readers(LidarSensorComponent.Reader).values()))
        frames = lidar.frames_timestamps_us[:, 1]
        for _ in range(2):
            for ts in frames:
                lidar.get_frame_ray_bundle_data(int(ts), "direction")
        if node_cache_size == 0:
            self.assertIsNone(lidar._group.cache)
        else:
            assert (cache := lidar._group.cache) is not None
            self.assertEqual(cache.max_entries, node_cache_size)
            self.assertLessEqual(len(cache), node_cache_size or len(cache))
            frame_entries = [key for key in cache.keys() if key.endswith("/ray_bundle/direction")]
            self.assertEqual(len(frame_entries), min(len(frames), node_cache_size or len(frames)))

    @parameterized.parameterized.expand([("v2",), ("v3",)])
    def test_async_reads(self, data: str):
        """Asynchronous reader APIs (all frames read concurrently) return the same data as the synchronous APIs"""
        if data == "v3" and 3 not in nodes.SUPPORTED_ZARR_FORMATS:
            self.skipTest("zarr format v3 data requires zarr-python 3")
        reader = SequenceComponentGroupsReader([_TRIMMED_V2 if data == "v2" else _TRIMMED_V3])
        _assert_summaries_equal(self, self.reference, asyncio.run(_summary_async(reader)))

    def test_invalid_node_cache_size(self):
        with self.assertRaises(ValueError):
            SequenceComponentGroupsReader([_TRIMMED_V2], node_cache_size=-1)

    @parameterized.parameterized.expand([(f,) for f in nodes.SUPPORTED_ZARR_FORMATS])
    def test_reserialization(self, zarr_format: int):
        """Re-serializing (subsampled frames of) all components of the test data with the installed zarr-python version
        via the public component APIs preserves all content"""
        reader = SequenceComponentGroupsReader([_TRIMMED_V2])

        with tempfile.TemporaryDirectory() as output_dir:
            writer = SequenceComponentGroupsWriter.from_reader(
                UPath(output_dir), "copy", reader, zarr_format=cast(Literal[2, 3], zarr_format)
            )
            _copy_components(reader, writer)
            copy = _summary(SequenceComponentGroupsReader(writer.finalize()))  # (contains the subsampled frames only)

        self.assertEqual(self.reference_subsampled.keys(), copy.keys())
        for component in ("poses/", "intrinsics/", "masks/", "cameras/", "lidars/", "radars/", "cuboids/"):
            self.assertTrue(any(key.startswith(component) for key in copy), f"no {component} data in the test data")
        _assert_summaries_equal(self, self.reference_subsampled, copy)


def _assert_summaries_equal(test: unittest.TestCase, expected: Dict[str, Any], actual: Dict[str, Any]) -> None:
    """Compares sequence summaries per component / frame / field (without expensive diffs of large payloads)"""
    test.assertEqual(sorted(expected), sorted(actual))
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if not isinstance(expected_value, list):
            test.assertTrue(expected_value == actual_value, key)
            continue
        test.assertEqual(len(expected_value), len(actual_value), key)
        for index, (expected_frame, actual_frame) in enumerate(zip(expected_value, actual_value)):
            if not isinstance(expected_frame, tuple):
                test.assertTrue(expected_frame == actual_frame, f"{key}[{index}]")
                continue
            for field, (expected_field, actual_field) in enumerate(zip(expected_frame, actual_frame)):
                if isinstance(expected_field, dict):
                    expected_items = cast(Dict[str, object], expected_field)
                    actual_items = cast(Dict[str, object], actual_field)
                    test.assertEqual(sorted(expected_items), sorted(actual_items), f"{key}[{index}][{field}]")
                    for name, expected_item in expected_items.items():
                        test.assertTrue(expected_item == actual_items[name], f"{key}[{index}][{field}][{name}] differs")
                else:
                    test.assertTrue(expected_field == actual_field, f"{key}[{index}][{field}] differs")


def _copy_components(reader: SequenceComponentGroupsReader, writer: SequenceComponentGroupsWriter) -> None:
    """Copies (subsampled frames of) all components of a sequence via the public component APIs"""

    for name, poses in reader.open_component_readers(PosesComponent.Reader).items():
        poses_writer = writer.register_component_writer(PosesComponent.Writer, name, name, poses.generic_meta_data)
        for (source, target), pose in poses.get_static_poses():
            poses_writer.store_static_pose(source, target, pose)
        for (source, target), (trajectory, timestamps_us) in poses.get_dynamic_poses():
            poses_writer.store_dynamic_pose(source, target, trajectory, timestamps_us)

    for name, intrinsics in reader.open_component_readers(IntrinsicsComponent.Reader).items():
        intrinsics_writer = writer.register_component_writer(
            IntrinsicsComponent.Writer, name, name, intrinsics.generic_meta_data
        )
        for camera_id in intrinsics._group.group("cameras").members():
            intrinsics_writer.store_camera_intrinsics(camera_id, intrinsics.get_camera_model_parameters(camera_id))
        for lidar_id in intrinsics._group.group("lidars").members():
            if (lidar_model := intrinsics.get_lidar_model_parameters(lidar_id)) is not None:
                intrinsics_writer.store_lidar_intrinsics(lidar_id, lidar_model)

    for name, masks in reader.open_component_readers(MasksComponent.Reader).items():
        masks_writer = writer.register_component_writer(MasksComponent.Writer, name, name, masks.generic_meta_data)
        for camera_id in masks._group.group("cameras").members():
            masks_writer.store_camera_masks(camera_id, dict(masks.get_camera_mask_images(camera_id)))

    for name, cameras in reader.open_component_readers(CameraSensorComponent.Reader).items():
        camera_writer = writer.register_component_writer(
            CameraSensorComponent.Writer, name, name, cameras.generic_meta_data
        )
        for frame_timestamps_us in _subsampled(cameras.frames_timestamps_us):
            ts = int(frame_timestamps_us[1])
            data = cameras.get_frame_data(ts)
            camera_writer.store_frame(
                data.get_encoded_image_data(),
                data.get_encoded_image_format(),
                frame_timestamps_us,
                {n: cameras.get_frame_generic_data(ts, n) for n in cameras.get_frame_generic_data_names(ts)},
                cameras.get_frame_generic_meta_data(ts),
            )

    for name, lidar in reader.open_component_readers(LidarSensorComponent.Reader).items():
        lidar_writer = writer.register_component_writer(
            LidarSensorComponent.Writer, name, name, lidar.generic_meta_data
        )
        for frame_timestamps_us in _subsampled(lidar.frames_timestamps_us):
            ts = int(frame_timestamps_us[1])
            ray_names = lidar.get_frame_ray_bundle_data_names(ts)
            lidar_writer.store_frame(
                direction=lidar.get_frame_ray_bundle_data(ts, "direction"),
                timestamp_us=lidar.get_frame_ray_bundle_data(ts, "timestamp_us"),
                model_element=(
                    lidar.get_frame_ray_bundle_data(ts, "model_element") if "model_element" in ray_names else None
                ),
                distance_m=lidar.get_frame_ray_bundle_return_data(ts, "distance_m", None),
                intensity=lidar.get_frame_ray_bundle_return_data(ts, "intensity", None),
                frame_timestamps_us=frame_timestamps_us,
                generic_data={n: lidar.get_frame_generic_data(ts, n) for n in lidar.get_frame_generic_data_names(ts)},
                generic_meta_data=lidar.get_frame_generic_meta_data(ts),
            )

    for name, radar in reader.open_component_readers(RadarSensorComponent.Reader).items():
        radar_writer = writer.register_component_writer(
            RadarSensorComponent.Writer, name, name, radar.generic_meta_data
        )
        for frame_timestamps_us in _subsampled(radar.frames_timestamps_us):
            ts = int(frame_timestamps_us[1])
            radar_writer.store_frame(
                direction=radar.get_frame_ray_bundle_data(ts, "direction"),
                timestamp_us=radar.get_frame_ray_bundle_data(ts, "timestamp_us"),
                distance_m=radar.get_frame_ray_bundle_return_data(ts, "distance_m", None),
                frame_timestamps_us=frame_timestamps_us,
                generic_data={n: radar.get_frame_generic_data(ts, n) for n in radar.get_frame_generic_data_names(ts)},
                generic_meta_data=radar.get_frame_generic_meta_data(ts),
            )

    for name, cuboids in reader.open_component_readers(CuboidsComponent.Reader).items():
        writer.register_component_writer(
            CuboidsComponent.Writer, name, name, cuboids.generic_meta_data
        ).store_observations(list(cuboids.get_observations()))


if __name__ == "__main__":
    unittest.main()
