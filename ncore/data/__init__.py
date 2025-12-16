# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


"""Package exposing methods related to NCORE's data types"""

from ncore.impl.data.types import (
    BBox3,
    BivariateWindshieldModelParameters,
    CameraModelParameters,
    FrameTimepoint,
    FThetaCameraModelParameters,
    LabelSource,
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
    ReferencePolynomial,
    RowOffsetStructuredSpinningLidarModelParameters,
    ShutterType,
)
from ncore.impl.data.util import padded_index_string


__all__ = [
    # generic types
    "FrameTimepoint",
    "CameraModelParameters",
    "ShutterType",
    "FThetaCameraModelParameters",
    "OpenCVPinholeCameraModelParameters",
    "OpenCVFisheyeCameraModelParameters",
    "ReferencePolynomial",
    "BivariateWindshieldModelParameters",
    "RowOffsetStructuredSpinningLidarModelParameters",
    "BBox3",
    "LabelSource",
    # util
    "padded_index_string",
]
