# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


"""Package exposing methods related to NCORE's V3 data interaction APIs"""

from ncore.impl.data.v3.compat import SequenceLoaderV3
from ncore.impl.data.v3.shards import (
    CameraSensor,
    FrameLabel3,
    LidarSensor,
    PointCloudSensor,
    Poses,
    RadarSensor,
    Sensor,
    ShardDataLoader,
    TrackLabel,
    Tracks,
)


__all__ = [
    # shard APIs
    "ShardDataLoader",
    "Sensor",
    "CameraSensor",
    "PointCloudSensor",
    "LidarSensor",
    "RadarSensor",
    "FrameLabel3",
    "Poses",
    "TrackLabel",
    "Tracks",
    # compat APIs
    "SequenceLoaderV3",
]
