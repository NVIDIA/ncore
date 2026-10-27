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

"""Structure of the asynchronous compat protocols: each extends its synchronous protocol, adds ``*_async`` variants of
the synchronous data reads, and narrows accessors returning other protocol types to their asynchronous protocols."""

from __future__ import annotations

import inspect
import re
import unittest

from typing import Any, Dict, List, Set, Tuple, Type

from ncore.impl.data import compat


# Asynchronous protocol -> its synchronous protocol
ASYNC_PROTOCOLS: Dict[Type[Any], Type[Any]] = {
    compat.AsyncSequenceLoaderProtocol: compat.SequenceLoaderProtocol,
    compat.AsyncSensorProtocol: compat.SensorProtocol,
    compat.AsyncCameraSensorProtocol: compat.CameraSensorProtocol,
    compat.AsyncCameraSensorProtocol.EncodedImageDataHandleProtocol: (
        compat.CameraSensorProtocol.EncodedImageDataHandleProtocol
    ),
    compat.AsyncRayBundleSensorProtocol: compat.RayBundleSensorProtocol,
    compat.AsyncLidarSensorProtocol: compat.LidarSensorProtocol,
    compat.AsyncRadarSensorProtocol: compat.RadarSensorProtocol,
    compat.AsyncPointCloudsSourceProtocol: compat.PointCloudsSourceProtocol,
    compat.AsyncCameraLabelsProtocol: compat.CameraLabelsProtocol,
    compat.AsyncCameraLabelsProtocol.CameraLabelHandleProtocol: compat.CameraLabelsProtocol.CameraLabelHandleProtocol,
}

# Synchronous methods reading data from storage, which have asynchronous ``*_async`` variants
SYNC_READS: Dict[Type[Any], List[str]] = {
    compat.SensorProtocol: ["get_frame_generic_data"],
    compat.CameraSensorProtocol: ["get_frame_data", "get_mask_images"],
    compat.CameraSensorProtocol.EncodedImageDataHandleProtocol: ["get_data"],
    compat.RayBundleSensorProtocol: [
        "get_frame_ray_bundle_direction",
        "get_frame_ray_bundle_timestamp_us",
        "get_frame_ray_bundle_return_distance_m",
        "get_frame_ray_bundle_return_valid_mask",
        "get_frame_point_cloud",
    ],
    compat.LidarSensorProtocol: [
        "get_frame_ray_bundle_model_element",
        "get_frame_ray_bundle_return_intensity",
    ],
    compat.PointCloudsSourceProtocol: ["get_pc", "get_pc_generic_data"],
    compat.CameraLabelsProtocol.CameraLabelHandleProtocol: ["get_data", "get_encoded_data"],
}

# Accessors of synchronous protocols returning other protocol types: (async protocol, method) -> narrowed return type
NARROWED_ACCESSORS: Dict[Tuple[Type[Any], str], str] = {
    (compat.AsyncSequenceLoaderProtocol, "get_camera_sensor"): "AsyncCameraSensorProtocol",
    (compat.AsyncSequenceLoaderProtocol, "get_lidar_sensor"): "AsyncLidarSensorProtocol",
    (compat.AsyncSequenceLoaderProtocol, "get_radar_sensor"): "AsyncRadarSensorProtocol",
    (compat.AsyncSequenceLoaderProtocol, "get_point_clouds_source"): "AsyncPointCloudsSourceProtocol",
    (compat.AsyncSequenceLoaderProtocol, "get_camera_labels"): "AsyncCameraLabelsProtocol",
    (compat.AsyncSequenceLoaderProtocol, "query_camera_labels"): "Sequence[AsyncCameraLabelsProtocol]",
    (compat.AsyncCameraSensorProtocol, "get_frame_handle"): "AsyncCameraSensorProtocol.EncodedImageDataHandleProtocol",
    (compat.AsyncCameraLabelsProtocol, "get_label"): "AsyncCameraLabelsProtocol.CameraLabelHandleProtocol",
}


def _protocol_members(cls: Type[Any]) -> Set[str]:
    """The public methods / properties a protocol (and its base protocols) declares"""
    return {
        name
        for base in cls.__mro__
        if base is not object and "Protocol" in base.__name__
        for name, value in vars(base).items()
        if not name.startswith("_") and (inspect.isfunction(value) or isinstance(value, property))
    }


def _return_annotation(cls: Type[Any], method: str) -> str:
    """The (unevaluated, ``from __future__ import annotations``) return annotation of a method as defined on `cls`"""
    for base in cls.__mro__:
        if method in vars(base):
            return str(inspect.signature(vars(base)[method]).return_annotation)
    raise AttributeError(f"{cls.__qualname__} has no {method}")


class TestAsyncCompatProtocols(unittest.TestCase):
    def test_async_protocols_extend_sync_protocols(self) -> None:
        for async_protocol, sync_protocol in ASYNC_PROTOCOLS.items():
            with self.subTest(async_protocol.__qualname__):
                self.assertIn(sync_protocol, async_protocol.__mro__)
                self.assertTrue(_protocol_members(async_protocol) >= _protocol_members(sync_protocol))

    def test_every_sync_read_has_an_async_variant(self) -> None:
        for sync_protocol, methods in SYNC_READS.items():
            async_protocols = [a for a in ASYNC_PROTOCOLS if sync_protocol in a.__mro__]
            self.assertTrue(async_protocols, sync_protocol.__qualname__)
            for async_protocol in async_protocols:
                for method in methods:
                    with self.subTest(f"{async_protocol.__qualname__}.{method}_async"):
                        async_method = getattr(async_protocol, f"{method}_async")
                        self.assertTrue(inspect.iscoroutinefunction(async_method))
                        # same parameters as the synchronous read
                        self.assertEqual(
                            list(inspect.signature(async_method).parameters),
                            list(inspect.signature(getattr(sync_protocol, method)).parameters),
                        )

    def test_async_methods_are_coroutines(self) -> None:
        for async_protocol in ASYNC_PROTOCOLS:
            for name in _protocol_members(async_protocol):
                if name.endswith("_async"):
                    with self.subTest(f"{async_protocol.__qualname__}.{name}"):
                        self.assertTrue(inspect.iscoroutinefunction(getattr(async_protocol, name)))

    def test_accessors_are_narrowed_to_async_protocols(self) -> None:
        for (async_protocol, method), expected in NARROWED_ACCESSORS.items():
            with self.subTest(f"{async_protocol.__qualname__}.{method}"):
                self.assertEqual(_return_annotation(async_protocol, method), expected)

    def test_all_protocol_valued_accessors_are_narrowed(self) -> None:
        """Every member of an asynchronous protocol returning a synchronous compat protocol type is narrowed to the
        asynchronous protocol, so asynchronous reads are reachable from AsyncSequenceLoaderProtocol without casts"""
        # (protocol types only: e.g. RayBundleSensorProtocol.FramePointCloud is a dataclass, shared by both APIs)
        sync_protocol_names = {p.__qualname__ for p in ASYNC_PROTOCOLS.values()}
        for async_protocol in ASYNC_PROTOCOLS:
            for name in _protocol_members(async_protocol):
                if not inspect.isfunction(getattr(async_protocol, name)):
                    continue
                annotation = _return_annotation(async_protocol, name)
                # (dotted names in the annotation, e.g. "AsyncCameraLabelsProtocol.CameraLabelHandleProtocol")
                referenced = set(re.findall(r"[A-Za-z_][\w.]*", annotation))
                with self.subTest(f"{async_protocol.__qualname__}.{name} -> {annotation}"):
                    self.assertFalse(
                        referenced & sync_protocol_names,
                        "returns a synchronous protocol type, narrow it to the asynchronous protocol",
                    )


if __name__ == "__main__":
    unittest.main()
