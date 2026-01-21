# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


"""Package exposing methods related to NCORE's V4 data interaction APIs"""

from ncore.impl.data.v4.compat import SequenceLoaderV3, SequenceLoaderV4
from ncore.impl.data.v4.components import (
    CameraSensorComponent,
    ComponentReader,
    ComponentWriter,
    CuboidsComponent,
    IntrinsicsComponent,
    LidarSensorComponent,
    MasksComponent,
    PosesComponent,
    RadarSensorComponent,
    SequenceComponentGroupsReader,
    SequenceComponentGroupsWriter,
)
from ncore.impl.data.v4.conversion import NCore3To4
from ncore.impl.data.v4.types import CuboidTrackObservation


__all__ = [
    # component APIs
    "SequenceComponentGroupsWriter",
    "SequenceComponentGroupsReader",
    "ComponentWriter",
    "ComponentReader",
    "PosesComponent",
    "IntrinsicsComponent",
    "MasksComponent",
    "CameraSensorComponent",
    "LidarSensorComponent",
    "RadarSensorComponent",
    "CuboidsComponent",
    # compat APIs
    "SequenceLoaderV3",
    "SequenceLoaderV4",
    # conversion APIs
    "NCore3To4",
    # types
    "CuboidTrackObservation",
]
