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
    DynamicFlagState,
    FrameLabel3,
    FrameTimepoint,
    FThetaCameraModelParameters,
    LabelSource,
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
    Poses,
    ReferencePolynomial,
    RowOffsetStructuredSpinningLidarModelParameters,
    ShutterType,
    TrackLabel,
    Tracks,
)
from ncore.impl.data.util import padded_index_string


__all__ = [
    # types
    "Poses",
    "FrameTimepoint",
    "CameraModelParameters",
    "ShutterType",
    "FThetaCameraModelParameters",
    "OpenCVPinholeCameraModelParameters",
    "OpenCVFisheyeCameraModelParameters",
    "ReferencePolynomial",
    "BivariateWindshieldModelParameters",
    "RowOffsetStructuredSpinningLidarModelParameters",
    "Tracks",
    "TrackLabel",
    "FrameLabel3",
    "BBox3",
    "LabelSource",
    "DynamicFlagState",
    # util
    "padded_index_string",
]
