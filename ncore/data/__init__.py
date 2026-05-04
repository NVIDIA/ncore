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
    CameraLabelsProtocol,
    CameraSensorProtocol,
    LidarSensorProtocol,
    PointCloudsSourceProtocol,
    RadarSensorProtocol,
    RayBundleSensorPointCloudsSourceAdapter,
    SensorProtocol,
    SequenceLoaderProtocol,
)
from ncore.impl.data.types import (
    BBox3,
    BivariateWindshieldModelParameters,
    CameraLabelDescriptor,
    ConcreteCameraModelParametersUnion,
    ConcreteExternalDistortionParametersUnion,
    ConcreteLidarModelParametersUnion,
    CuboidTrackObservation,
    EncodedImageData,
    EncodedImageHandle,
    FrameTimepoint,
    FThetaCameraModelParameters,
    LabelCategory,
    LabelEncoding,
    LabelSchema,
    LabelSource,
    LabelType,
    LabelUnit,
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
    PointCloud,
    QuantizationParams,
    ReferencePolynomial,
    RowOffsetStructuredSpinningLidarModelParameters,
    ShutterType,
)


__all__ = [
    # regular data types
    "LabelSource",
    "LabelCategory",
    "LabelType",
    "LabelUnit",
    "LabelEncoding",
    "LabelSchema",
    "QuantizationParams",
    "CameraLabelDescriptor",
    "FrameTimepoint",
    "ShutterType",
    "CuboidTrackObservation",
    "BBox3",
    "ReferencePolynomial",
    "BivariateWindshieldModelParameters",
    "FThetaCameraModelParameters",
    "OpenCVPinholeCameraModelParameters",
    "OpenCVFisheyeCameraModelParameters",
    "RowOffsetStructuredSpinningLidarModelParameters",
    "EncodedImageData",
    "EncodedImageHandle",
    "ConcreteCameraModelParametersUnion",
    "ConcreteExternalDistortionParametersUnion",
    "ConcreteLidarModelParametersUnion",
    "PointCloud",
    # compat protocols
    "SequenceLoaderProtocol",
    "SensorProtocol",
    "CameraSensorProtocol",
    "LidarSensorProtocol",
    "RadarSensorProtocol",
    "PointCloudsSourceProtocol",
    "CameraLabelsProtocol",
    "RayBundleSensorPointCloudsSourceAdapter",
]
