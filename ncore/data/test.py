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
# limitations under the


def test_import_package():
    """Make sure package can be imported"""

    import ncore.data  # noqa: F401


def test_json_types():
    """The package exposes the JSON value types of the implementation (the same objects)"""
    import ncore.data as public

    from ncore.impl.data import json_types

    for name in ("Json", "JsonDocument", "JsonLike", "JsonDocumentLike"):
        assert name in public.__all__, name
        assert getattr(public, name) is getattr(json_types, name), name


def test_public_names():
    """The package exposes the data types and protocols of the implementation (the same objects)"""
    import ncore.data as public

    from ncore.impl.data import compat, json_types, types, util

    expected = {
        "LabelSource": types.LabelSource,
        "FrameTimepoint": types.FrameTimepoint,
        "ShutterType": types.ShutterType,
        "CuboidTrackObservation": types.CuboidTrackObservation,
        "BBox3": types.BBox3,
        "ReferencePolynomial": types.ReferencePolynomial,
        "BivariateWindshieldModelParameters": types.BivariateWindshieldModelParameters,
        "FThetaCameraModelParameters": types.FThetaCameraModelParameters,
        "OpenCVPinholeCameraModelParameters": types.OpenCVPinholeCameraModelParameters,
        "OpenCVFisheyeCameraModelParameters": types.OpenCVFisheyeCameraModelParameters,
        "IdealPinholeCameraModelParameters": types.IdealPinholeCameraModelParameters,
        "RowOffsetStructuredSpinningLidarModelParameters": types.RowOffsetStructuredSpinningLidarModelParameters,
        "LidarModelParameters": types.LidarModelParameters,
        "ParaxialPinholeGeometry": types.ParaxialPinholeGeometry,
        "SpinningLidarModelParameters": types.SpinningLidarModelParameters,
        "StructuredSpinningLidarModelParameters": types.StructuredSpinningLidarModelParameters,
        "EncodedImageData": types.EncodedImageData,
        "EncodedImageHandle": compat.CameraSensorProtocol.EncodedImageDataHandleProtocol,
        "CameraModelParameters": types.CameraModelParameters,
        "ConcreteCameraModelParametersUnion": types.ConcreteCameraModelParametersUnion,
        "ConcreteExternalDistortionParametersUnion": types.ConcreteExternalDistortionParametersUnion,
        "ExternalDistortionParameters": types.ExternalDistortionParameters,
        "register_external_distortion_parameters": types.register_external_distortion_parameters,
        "register_lidar_model_parameters": types.register_lidar_model_parameters,
        "register_camera_model_parameters": types.register_camera_model_parameters,
        "ConcreteLidarModelParametersUnion": types.ConcreteLidarModelParametersUnion,
        "PointCloud": types.PointCloud,
        "LabelCategory": types.LabelCategory,
        "LabelUnit": types.LabelUnit,
        "LabelEncoding": types.LabelEncoding,
        "LabelType": types.LabelType,
        "QuantizationParams": types.QuantizationParams,
        "LabelSchema": types.LabelSchema,
        "CameraLabelDescriptor": types.CameraLabelDescriptor,
        "FOV": util.FOV,
        "Json": json_types.Json,
        "JsonDocument": json_types.JsonDocument,
        "JsonLike": json_types.JsonLike,
        "JsonDocumentLike": json_types.JsonDocumentLike,
        "SequenceLoaderProtocol": compat.SequenceLoaderProtocol,
        "SensorProtocol": compat.SensorProtocol,
        "CameraSensorProtocol": compat.CameraSensorProtocol,
        "LidarSensorProtocol": compat.LidarSensorProtocol,
        "RadarSensorProtocol": compat.RadarSensorProtocol,
        "RayBundleSensorProtocol": compat.RayBundleSensorProtocol,
        "PointCloudsSourceProtocol": compat.PointCloudsSourceProtocol,
        "RayBundleSensorPointCloudsSourceAdapter": compat.RayBundleSensorPointCloudsSourceAdapter,
        "CameraLabelsProtocol": compat.CameraLabelsProtocol,
        "AsyncSequenceLoaderProtocol": compat.AsyncSequenceLoaderProtocol,
        "AsyncSensorProtocol": compat.AsyncSensorProtocol,
        "AsyncCameraSensorProtocol": compat.AsyncCameraSensorProtocol,
        "AsyncRayBundleSensorProtocol": compat.AsyncRayBundleSensorProtocol,
        "AsyncLidarSensorProtocol": compat.AsyncLidarSensorProtocol,
        "AsyncRadarSensorProtocol": compat.AsyncRadarSensorProtocol,
        "AsyncPointCloudsSourceProtocol": compat.AsyncPointCloudsSourceProtocol,
        "AsyncRayBundleSensorPointCloudsSourceAdapter": compat.AsyncRayBundleSensorPointCloudsSourceAdapter,
        "AsyncCameraLabelsProtocol": compat.AsyncCameraLabelsProtocol,
    }
    assert len(public.__all__) == len(set(public.__all__))
    assert sorted(public.__all__) == sorted(expected)
    for name, value in expected.items():
        assert getattr(public, name) is value, name


def test_point_clouds_source_adapter_keyword_only_parameters():
    """The optional parameters of the ray bundle sensor point-cloud source adapters are keyword-only"""
    import inspect

    import ncore.data as public

    for adapter in (
        public.RayBundleSensorPointCloudsSourceAdapter,
        public.AsyncRayBundleSensorPointCloudsSourceAdapter,
    ):
        parameters = inspect.signature(adapter.__init__).parameters.values()
        assert [p.name for p in parameters if p.kind is p.KEYWORD_ONLY] == ["return_index", "promote_generic_data"]
