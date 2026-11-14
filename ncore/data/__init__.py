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

"""Package exposing methods related to NCore's basic data types and abstract APIs"""

from ncore.impl.data.compat import (
    AsyncCameraLabelsProtocol,
    AsyncCameraSensorProtocol,
    AsyncLidarSensorProtocol,
    AsyncPointCloudsSourceProtocol,
    AsyncRadarSensorProtocol,
    AsyncRayBundleSensorPointCloudsSourceAdapter,
    AsyncRayBundleSensorProtocol,
    AsyncSensorProtocol,
    AsyncSequenceLoaderProtocol,
    CameraLabelsProtocol,
    CameraSensorProtocol,
    EncodedImageHandle,
    LidarSensorProtocol,
    PointCloudsSourceProtocol,
    RadarSensorProtocol,
    RayBundleSensorPointCloudsSourceAdapter,
    RayBundleSensorProtocol,
    SensorProtocol,
    SequenceLoaderProtocol,
)
from ncore.impl.data.json_types import Json, JsonDocument, JsonDocumentLike, JsonLike
from ncore.impl.data.types import (
    BBox3,
    BivariateWindshieldModelParameters,
    CameraLabelDescriptor,
    CameraModelParameters,
    ConcreteCameraModelParametersUnion,
    ConcreteExternalDistortionParametersUnion,
    ConcreteLidarModelParametersUnion,
    CuboidTrackObservation,
    EncodedImageData,
    ExternalDistortionParameters,
    FrameTimepoint,
    FThetaCameraModelParameters,
    IdealPinholeCameraModelParameters,
    LabelCategory,
    LabelEncoding,
    LabelSchema,
    LabelSource,
    LabelType,
    LabelUnit,
    LidarModelParameters,
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
    ParaxialPinholeGeometry,
    PointCloud,
    QuantizationParams,
    ReferencePolynomial,
    RowOffsetStructuredSpinningLidarModelParameters,
    ShutterType,
    SpinningLidarModelParameters,
    StructuredSpinningLidarModelParameters,
    register_camera_model_parameters,
    register_external_distortion_parameters,
    register_lidar_model_parameters,
)
from ncore.impl.data.util import FOV


__all__ = [
    # regular data types
    "LabelSource",
    "FrameTimepoint",
    "ShutterType",
    "CuboidTrackObservation",
    "BBox3",
    "ReferencePolynomial",
    "BivariateWindshieldModelParameters",
    "FThetaCameraModelParameters",
    "OpenCVPinholeCameraModelParameters",
    "OpenCVFisheyeCameraModelParameters",
    "IdealPinholeCameraModelParameters",
    "RowOffsetStructuredSpinningLidarModelParameters",
    "LidarModelParameters",
    "ParaxialPinholeGeometry",
    "SpinningLidarModelParameters",
    "StructuredSpinningLidarModelParameters",
    "EncodedImageData",
    "EncodedImageHandle",
    "CameraModelParameters",
    "ConcreteCameraModelParametersUnion",
    "ConcreteExternalDistortionParametersUnion",
    "ExternalDistortionParameters",
    "register_external_distortion_parameters",
    "register_lidar_model_parameters",
    "register_camera_model_parameters",
    "ConcreteLidarModelParametersUnion",
    "PointCloud",
    "LabelCategory",
    "LabelUnit",
    "LabelEncoding",
    "LabelType",
    "QuantizationParams",
    "LabelSchema",
    "CameraLabelDescriptor",
    "FOV",
    # JSON values of attributes / meta data (as read, and accepted for writing)
    "Json",
    "JsonDocument",
    "JsonLike",
    "JsonDocumentLike",
    # compat protocols
    "SequenceLoaderProtocol",
    "SensorProtocol",
    "CameraSensorProtocol",
    "LidarSensorProtocol",
    "RadarSensorProtocol",
    "RayBundleSensorProtocol",
    "PointCloudsSourceProtocol",
    "RayBundleSensorPointCloudsSourceAdapter",
    "CameraLabelsProtocol",
    # asynchronous compat protocols
    "AsyncSequenceLoaderProtocol",
    "AsyncSensorProtocol",
    "AsyncCameraSensorProtocol",
    "AsyncRayBundleSensorProtocol",
    "AsyncLidarSensorProtocol",
    "AsyncRadarSensorProtocol",
    "AsyncPointCloudsSourceProtocol",
    "AsyncRayBundleSensorPointCloudsSourceAdapter",
    "AsyncCameraLabelsProtocol",
]
