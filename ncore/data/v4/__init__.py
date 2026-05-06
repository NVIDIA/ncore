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

"""Package exposing methods related to NCore's V4 data interaction APIs"""

from ncore.impl.data.v4.compat import SequenceLoaderV4
from ncore.impl.data.v4.components import (
    CameraLabelsComponent,
    CameraSensorComponent,
    ComponentReader,
    ComponentWriter,
    CuboidsComponent,
    IntrinsicsComponent,
    LidarSensorComponent,
    MasksComponent,
    PointCloudsComponent,
    PosesComponent,
    RadarSensorComponent,
    SequenceComponentGroupsReader,
    SequenceComponentGroupsWriter,
)


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
    "PointCloudsComponent",
    "CameraLabelsComponent",
    # compat APIs
    "SequenceLoaderV4",
]
