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
from typing import List, Optional

import dataclasses_json

from ncore.impl.data import types, util


@dataclass
class CuboidTrack(dataclasses_json.DataClassJsonMixin):
    """Cuboid track instance associated with a specific reference frame"""

    @dataclass
    class Observation(dataclasses_json.DataClassJsonMixin):
        """Individual cuboid track observation relative to the reference frame"""

        observation_id: str  #: Identifier of the current observation (unique among all observations)
        timestamp_us: (
            int  #: The timestamp associated with the centroid of the observation (possibly an accurate in-frame time)
        )
        reference_frame_timestamp_us: int  #: The timestamp of the reference frame
        bbox3: (
            types.BBox3
        )  #: Bounding-box coordinates of the object relative to the reference frame's coordinate system

        def __post_init__(self):
            # Sanity checks
            assert isinstance(self.observation_id, str)
            assert isinstance(self.reference_frame_timestamp_us, int)
            assert isinstance(self.bbox3, types.BBox3)
            assert isinstance(self.timestamp_us, int)

    track_id: str  #: Unique identifier of the object's track this observation is associated with
    label_class: str  #: String-representation of the labeled class associated with this observation
    reference_frame_name: str  #: String-identifier of the reference frame (e.g., sensor name)
    observations: List["Observation"]  #: All observations associated with this track
    source: types.LabelSource = util.enum_field(types.LabelSource)  #: The source for the current label
    source_version: Optional[str] = (
        None  #: If provided, the unique version ID of the source for the current label (to distinguish between different versions of the same source)
    )

    def __post_init__(self):
        # Sanity checks
        assert isinstance(self.track_id, str)
        assert isinstance(self.label_class, str)
        assert isinstance(self.reference_frame_name, str)

        assert isinstance(self.observations, List)
        self.observations = [
            CuboidTrack.Observation.from_dict(obs) if isinstance(obs, dict) else obs for obs in self.observations
        ]

        if not isinstance(self.source, types.LabelSource):
            self.source = types.LabelSource(self.source)
        assert self.source in types.LabelSource.__members__.values()

        assert isinstance(self.source_version, (type(None), str))
