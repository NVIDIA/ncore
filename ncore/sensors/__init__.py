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

"""Package exposing methods related to NCore's sensor types"""

from ncore.impl.sensors.camera import (
    BivariateWindshieldModel,
    CameraModel,
    ExternalDistortionModel,
    FThetaCameraModel,
    IdealPinholeCameraModel,
    OpenCVFisheyeCameraModel,
    OpenCVPinholeCameraModel,
    PinholeCameraModel,
    camera_model_from_parameters,
    external_distortion_model_from_parameters,
    register_camera_model,
    register_external_distortion_model,
)
from ncore.impl.sensors.lidar import (
    LidarModel,
    RowOffsetStructuredSpinningLidarModel,
    StructuredLidarModel,
    lidar_model_from_parameters,
    maybe_lidar_model_from_parameters,
    register_lidar_model,
)
from ncore.impl.sensors.rectification import Rectificator


__all__ = [
    "CameraModel",
    "camera_model_from_parameters",
    "register_camera_model",
    "FThetaCameraModel",
    "PinholeCameraModel",
    "IdealPinholeCameraModel",
    "OpenCVPinholeCameraModel",
    "OpenCVFisheyeCameraModel",
    "ExternalDistortionModel",
    "external_distortion_model_from_parameters",
    "register_external_distortion_model",
    "BivariateWindshieldModel",
    "LidarModel",
    "lidar_model_from_parameters",
    "maybe_lidar_model_from_parameters",
    "register_lidar_model",
    "StructuredLidarModel",
    "RowOffsetStructuredSpinningLidarModel",
    "Rectificator",
]
