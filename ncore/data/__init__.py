# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


"""Package exposing methods related to NCore's basic data types"""

from ncore.impl.data.types import (
    BBox3,
    BivariateWindshieldModelParameters,
    ConcreteCameraModelParametersUnion,
    ConcreteExternalDistortionParametersUnion,
    ConcreteLidarModelParametersUnion,
    CuboidTrackObservation,
    EncodedImageData,
    EncodedImageHandle,
    FrameTimepoint,
    FThetaCameraModelParameters,
    LabelSource,
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
    ReferencePolynomial,
    RowOffsetStructuredSpinningLidarModelParameters,
    ShutterType,
)


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
    "RowOffsetStructuredSpinningLidarModelParameters",
    "EncodedImageData",
    "EncodedImageHandle",
]
