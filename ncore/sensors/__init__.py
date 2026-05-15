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

"""Package exposing methods related to NCore's sensor types.

This module requires PyTorch. Install it with: pip install ncore[sensors]
"""

from __future__ import annotations

import importlib

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ncore.impl.sensors.camera import (
        BivariateWindshieldModel as BivariateWindshieldModel,
        CameraModel as CameraModel,
        ExternalDistortionModel as ExternalDistortionModel,
        FThetaCameraModel as FThetaCameraModel,
        OpenCVFisheyeCameraModel as OpenCVFisheyeCameraModel,
        OpenCVPinholeCameraModel as OpenCVPinholeCameraModel,
    )
    from ncore.impl.sensors.lidar import (
        LidarModel as LidarModel,
        RowOffsetStructuredSpinningLidarModel as RowOffsetStructuredSpinningLidarModel,
        StructuredLidarModel as StructuredLidarModel,
    )


__all__ = [
    "CameraModel",
    "FThetaCameraModel",
    "OpenCVPinholeCameraModel",
    "OpenCVFisheyeCameraModel",
    "ExternalDistortionModel",
    "BivariateWindshieldModel",
    "LidarModel",
    "StructuredLidarModel",
    "RowOffsetStructuredSpinningLidarModel",
]


def __getattr__(name: str):
    if name in __all__:
        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError(
                "torch is required for sensor model evaluation. "
                "Install it with: pip install ncore[sensors]"
            ) from None

        if name in (
            "BivariateWindshieldModel",
            "CameraModel",
            "ExternalDistortionModel",
            "FThetaCameraModel",
            "OpenCVFisheyeCameraModel",
            "OpenCVPinholeCameraModel",
        ):
            module = importlib.import_module("ncore.impl.sensors.camera")
        else:
            module = importlib.import_module("ncore.impl.sensors.lidar")

        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
