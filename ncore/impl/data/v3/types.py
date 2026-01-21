# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import dataclasses_json
import numpy as np

from ncore.impl.data.types import BBox3, LabelSource
from ncore.impl.data.util import enum_field


@dataclass
class Poses:
    """Represents a collection of timestamped poses (rig-to-local-world transformation)"""

    T_rig_world_base: np.ndarray  #: Base rig-to-global-world SE3 transformation (float64, [4,4])
    T_rig_worlds: np.ndarray  #: All rig-to-local-world SE3 transformations of the trajectory (float64, [N,4,4])
    T_rig_world_timestamps_us: (
        np.ndarray
    )  #: All rig-to-local-world transformation timestamps of the trajectory (uint64, [N,])

    def __post_init__(self):
        # Sanity checks
        assert self.T_rig_world_base.shape == (4, 4)
        assert self.T_rig_world_base.dtype == np.dtype("float64")

        assert self.T_rig_worlds.shape[1:] == (4, 4)
        assert self.T_rig_worlds.dtype == np.dtype("float64")

        assert self.T_rig_world_timestamps_us.ndim == 1
        assert self.T_rig_world_timestamps_us.dtype == np.dtype("uint64")

        assert self.T_rig_worlds.shape[0] == self.T_rig_world_timestamps_us.shape[0]


@dataclass
class FrameLabel3(dataclasses_json.DataClassJsonMixin):
    """Description of a 3D frame-associated label"""

    label_id: str  #: Identifier of the current frame label (unique among all labels)
    track_id: str  #: Unique identifier of the object's track this label is associated with
    label_class: str  #: String-representation of the class associated with this label
    bbox3: BBox3  #: Bounding-box coordinates of the object relative to the frame's end-of-frame coordinate system
    global_speed: float  #: Instantaneous global speed [m/s] of the object
    timestamp_us: int  #: The timestamp associated with the centroid of the label (possibly an accurate in-frame time)
    confidence: Optional[float]  #: If available, the confidence score of the label [0..1]
    source: LabelSource = enum_field(LabelSource)  #: The source for the current label
    source_version: Optional[str] = (
        None  #: If provided, the unique version ID of the source for the current label (to distinguish between different versions of the same source)
    )

    def __post_init__(self):
        # Sanity checks
        assert isinstance(self.label_id, str)
        assert isinstance(self.track_id, str)
        assert isinstance(self.label_class, str)
        assert isinstance(self.bbox3, BBox3)
        assert isinstance(self.global_speed, float)
        assert isinstance(self.timestamp_us, int)
        assert isinstance(self.confidence, (type(None), float))

        if not isinstance(self.source, LabelSource):
            self.source = LabelSource(self.source)
        assert self.source in LabelSource.__members__.values()

        assert isinstance(self.source_version, (type(None), str))


@dataclass
class TrackLabel(dataclasses_json.DataClassJsonMixin):
    """Description of an individual object-specific track"""

    sensors: Dict[
        str, List[int]
    ]  #: Represents all frame-timestamps (map values) of the object's observations in different sensors (map keys)


@dataclass
class Tracks(dataclasses_json.DataClassJsonMixin):
    """Represents a collection of tracks"""

    track_labels: Dict[
        str, TrackLabel
    ]  #: Represents individual object tracks (map values) referenced by `track_id`'s (map keys, same as in `FrameLabel3`)
