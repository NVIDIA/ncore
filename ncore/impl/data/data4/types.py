# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

import sys

from dataclasses import dataclass, replace
from typing import Optional

import dataclasses_json
import numpy as np

from ncore.impl.common.transformations import PoseGraphInterpolator, transform_bbox
from ncore.impl.data import types, util


if sys.version_info >= (3, 11):
    # Older python versions have issues with type-hints for nested types in
    # combination with typing.get_type_hints() (used by, e.g., 'dataclasses_json')
    # - alias these globally as a workaround
    from typing import Self


@dataclass
class CuboidTrackObservation(dataclasses_json.DataClassJsonMixin):
    """Individual cuboid track observation relative to a reference frame"""

    track_id: str  #: Unique identifier of the object's track this observation is associated with
    class_id: str  #: String-representation of the labeled class of the object

    timestamp_us: (
        int  #: The timestamp associated with the centroid of the observation (possibly an accurate in-frame time)
    )

    reference_frame_id: str  #: String-identifier of the reference frame (e.g., sensor name)
    reference_frame_timestamp_us: int  #: The timestamp of the reference frame

    bbox3: types.BBox3  #: Bounding-box coordinates of the object relative to the reference frame's coordinate system

    source: types.LabelSource = util.enum_field(types.LabelSource)  #: The source for the current label
    source_version: Optional[str] = (
        None  #: If provided, the unique version ID of the source for the current label (to distinguish between different versions of the same source)
    )

    def transform(
        self,
        target_frame_id: str,
        target_frame_timestamp_us: int,
        pose_graph: PoseGraphInterpolator,
        anchor_frame_id: str = "world",
    ) -> "Self":
        """Transform the observation's bounding box to a different reference frame.

        Args:
            target_frame_id: ID of the target reference frame
            target_frame_timestamp_us: Timestamp of the target reference frame
            pose_graph: PoseGraphInterpolator to perform the evaluation of transformations
            anchor_frame_id: ID of the common anchor frame for transformations (default: "world")

        Returns:
            A CuboidTrackObservation instance with the transformed bounding box and updated reference frame info
        """

        if (
            self.reference_frame_id == target_frame_id
            and self.reference_frame_timestamp_us == target_frame_timestamp_us
        ):
            # Skip transformation if already in correct target frame
            return self

        # Transform observation from reference frame at observation time to target frame at target time via world
        T_reference_world = pose_graph.evaluate_poses(
            self.reference_frame_id,
            anchor_frame_id,
            np.array(self.reference_frame_timestamp_us, dtype=np.int64),
        )
        T_world_target = pose_graph.evaluate_poses(
            anchor_frame_id,
            target_frame_id,
            np.array(target_frame_timestamp_us, dtype=np.int64),
        )

        T_reference_target = T_world_target @ T_reference_world

        return replace(
            self,
            bbox3=types.BBox3.from_array(transform_bbox(self.bbox3.to_array(), T_reference_target)),
            reference_frame_id=target_frame_id,
            reference_frame_timestamp_us=target_frame_timestamp_us,
        )

    def __post_init__(self):
        # Sanity checks
        assert isinstance(self.track_id, str)
        assert isinstance(self.class_id, str)
        assert isinstance(self.reference_frame_id, str)
        assert isinstance(self.reference_frame_timestamp_us, int)
        assert isinstance(self.bbox3, types.BBox3)
        assert isinstance(self.timestamp_us, int)

        if not isinstance(self.source, types.LabelSource):
            self.source = types.LabelSource(self.source)
        assert self.source in types.LabelSource.__members__.values()

        assert isinstance(self.source_version, (type(None), str))
