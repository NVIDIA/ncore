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

import dataclasses
import tempfile
import unittest

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import parameterized

from python.runfiles import Runfiles
from upath import UPath

from ncore.impl.data import nodes
from ncore.impl.data.json_types import JsonDocument
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


@dataclasses.dataclass
class CameraFrame:
    timestamp_us: int
    image_format: str
    image_data: bytes
    generic_data: Dict[str, bytes]
    generic_meta_data: JsonDocument


@dataclasses.dataclass
class SensorFrame:
    """A lidar / radar frame"""

    timestamp_us: int
    ray_bundle_data: Dict[str, bytes]
    ray_bundle_return_data: Dict[str, bytes]
    ray_bundle_return_valid_mask: bytes
    generic_data: Dict[str, bytes]
    generic_meta_data: JsonDocument


@dataclasses.dataclass
class SequenceSummary:
    """The (comparable) data of a sequence, by component instance name"""

    sequence_id: str
    interval_us: Tuple[int, int]
    generic_meta_data: JsonDocument
    static_poses: Dict[str, Dict[Tuple[str, str], List[List[float]]]] = dataclasses.field(default_factory=dict)
    dynamic_poses: Dict[str, Dict[Tuple[str, str], Tuple[List[List[List[float]]], List[int]]]] = dataclasses.field(
        default_factory=dict
    )
    intrinsics: Dict[str, Tuple[JsonDocument, JsonDocument]] = dataclasses.field(default_factory=dict)
    cuboids: Dict[str, List[JsonDocument]] = dataclasses.field(default_factory=dict)
    #: mask images by camera id and mask name
    masks: Dict[str, Dict[str, Dict[str, bytes]]] = dataclasses.field(default_factory=dict)
    cameras: Dict[str, List[CameraFrame]] = dataclasses.field(default_factory=dict)
    #: lidar and radar frames (by component name / instance name)
    sensors: Dict[str, List[SensorFrame]] = dataclasses.field(default_factory=dict)


def _camera_frame(cameras: CameraSensorComponent.Reader, ts: int) -> CameraFrame:
    data = cameras.get_frame_data(ts)
    return CameraFrame(
        ts,
        data.get_encoded_image_format(),
        data.get_encoded_image_data(),
        {n: cameras.get_frame_generic_data(ts, n).tobytes() for n in sorted(cameras.get_frame_generic_data_names(ts))},
        cameras.get_frame_generic_meta_data(ts),
    )


def _sensor_frame(sensor: Union[LidarSensorComponent.Reader, RadarSensorComponent.Reader], ts: int) -> SensorFrame:
    return SensorFrame(
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
        {n: sensor.get_frame_generic_data(ts, n).tobytes() for n in sorted(sensor.get_frame_generic_data_names(ts))},
        sensor.get_frame_generic_meta_data(ts),
    )


def _sensor_readers(
    reader: SequenceComponentGroupsReader,
) -> List[Tuple[str, Union[LidarSensorComponent.Reader, RadarSensorComponent.Reader]]]:
    """Lidar and radar readers, by component name / instance name"""
    readers: List[Tuple[str, Union[LidarSensorComponent.Reader, RadarSensorComponent.Reader]]] = []
    for name, lidar in sorted(reader.open_component_readers(LidarSensorComponent.Reader).items()):
        readers.append((f"{LidarSensorComponent.COMPONENT_NAME}/{name}", lidar))
    for name, radar in sorted(reader.open_component_readers(RadarSensorComponent.Reader).items()):
        readers.append((f"{RadarSensorComponent.COMPONENT_NAME}/{name}", radar))
    return readers


def _summary(reader: SequenceComponentGroupsReader, subsampled: bool = False, frames: bool = True) -> SequenceSummary:
    """Collects a comparable summary of all data of a sequence (optionally of the subsampled frames only, or without
    masks and sensor frames)"""

    def selected(timestamps_us: np.ndarray) -> np.ndarray:
        return _subsampled(timestamps_us) if subsampled else timestamps_us

    interval = reader.sequence_timestamp_interval_us
    out = SequenceSummary(reader.sequence_id, (interval.start, interval.stop), reader.generic_meta_data)

    for name, poses in sorted(reader.open_component_readers(PosesComponent.Reader).items()):
        out.static_poses[name] = {k: v.tolist() for k, v in poses.get_static_poses()}
        out.dynamic_poses[name] = {k: (p.tolist(), t.tolist()) for k, (p, t) in poses.get_dynamic_poses()}

    for name, intrinsics in sorted(reader.open_component_readers(IntrinsicsComponent.Reader).items()):
        out.intrinsics[name] = (intrinsics._group.group("cameras").attrs, intrinsics._group.group("lidars").attrs)

    for name, cuboids in sorted(reader.open_component_readers(CuboidsComponent.Reader).items()):
        out.cuboids[name] = [obs.to_dict() for obs in cuboids.get_observations()]

    if not frames:
        return out

    for name, masks in sorted(reader.open_component_readers(MasksComponent.Reader).items()):
        out.masks[name] = {
            camera_id: {
                mask_name: np.asarray(image).tobytes() for mask_name, image in masks.get_camera_mask_images(camera_id)
            }
            for camera_id in masks._group.group("cameras").members()
        }

    for name, cameras in sorted(reader.open_component_readers(CameraSensorComponent.Reader).items()):
        out.cameras[name] = [_camera_frame(cameras, int(ts)) for ts in selected(cameras.frames_timestamps_us)[:, 1]]

    for name, sensor in _sensor_readers(reader):
        out.sensors[name] = [_sensor_frame(sensor, int(ts)) for ts in selected(sensor.frames_timestamps_us)[:, 1]]

    return out


class TestZarrFormats(unittest.TestCase):
    """zarr format v2 and v3 serializations of a sequence need to be identical in content"""

    reference: SequenceSummary  # summary of the zarr format v2 test data
    reference_subsampled: SequenceSummary  # summary of the subsampled frames of the zarr format v2 test data

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

    def test_invalid_node_cache_size(self):
        with self.assertRaises(ValueError):
            SequenceComponentGroupsReader([_TRIMMED_V2], node_cache_size=-1)

    @parameterized.parameterized.expand([(f,) for f in nodes.SUPPORTED_ZARR_FORMATS])
    def test_reserialization(self, zarr_format: nodes.ZarrFormat):
        """Re-serializing (subsampled frames of) all components of the test data with the installed zarr-python version
        via the public component APIs preserves all content"""
        reader = SequenceComponentGroupsReader([_TRIMMED_V2])

        with tempfile.TemporaryDirectory() as output_dir:
            writer = SequenceComponentGroupsWriter.from_reader(
                UPath(output_dir), "copy", reader, zarr_format=zarr_format
            )
            _copy_components(reader, writer)
            copy = _summary(SequenceComponentGroupsReader(writer.finalize()))  # (contains the subsampled frames only)

        for component, data in dataclasses.asdict(copy).items():
            self.assertTrue(data, f"no {component} data in the test data")
        self.assertTrue(any(name.startswith("radars/") for name in copy.sensors), "no radar data in the test data")
        _assert_summaries_equal(self, self.reference_subsampled, copy)


def _assert_summaries_equal(test: unittest.TestCase, expected: SequenceSummary, actual: SequenceSummary) -> None:
    """Compares sequence summaries per component / frame / field (without expensive diffs of large payloads)"""
    for field in dataclasses.fields(SequenceSummary):
        expected_value, actual_value = getattr(expected, field.name), getattr(actual, field.name)
        if field.name not in ("cameras", "sensors"):
            test.assertTrue(expected_value == actual_value, field.name)
            continue
        test.assertEqual(sorted(expected_value), sorted(actual_value), field.name)
        for name, expected_frames in expected_value.items():
            actual_frames = actual_value[name]
            test.assertEqual(len(expected_frames), len(actual_frames), f"{field.name}/{name}")
            for index, (expected_frame, actual_frame) in enumerate(zip(expected_frames, actual_frames)):
                for frame_field in dataclasses.fields(expected_frame):
                    test.assertTrue(
                        getattr(expected_frame, frame_field.name) == getattr(actual_frame, frame_field.name),
                        f"{field.name}/{name}[{index}].{frame_field.name} differs",
                    )


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
