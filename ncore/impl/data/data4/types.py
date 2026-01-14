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
from typing import Dict, List, Literal, Optional

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


@dataclass
class ComponentGroupAssignments:
    """Component group assignments for all default components in a V4 data output"""

    poses_component_group: Optional[str]
    intrinsics_component_group: Optional[str]
    masks_component_group: Optional[str]
    camera_component_groups: Dict[str, str]  # indexed by camera_id
    lidar_component_groups: Dict[str, str]  # indexed by lidar_id
    radar_component_groups: Dict[str, str]  # indexed by radar_id
    cuboid_track_observations_component_group: Optional[str]

    @staticmethod
    def create(
        camera_ids: List[str],
        lidar_ids: List[str],
        radar_ids: List[str],
        profile: Literal["default", "separate-sensors", "separate-all"],
        # Component-specific overrides
        poses_component_group: Optional[str] = None,
        intrinsics_component_group: Optional[str] = None,
        masks_component_group: Optional[str] = None,
        camera_component_groups: Optional[Dict[str, str]] = None,
        lidar_component_groups: Optional[Dict[str, str]] = None,
        radar_component_groups: Optional[Dict[str, str]] = None,
        cuboid_track_observations_component_group: Optional[str] = None,
    ) -> ComponentGroupAssignments:
        """Factory function to create ComponentGroups based on a profile.

        Args:
            source_data_loader: ShardDataLoader to determine available sensors
            profile: One of:
                - "default": Use provided overrites or fall back to default groups
                - "separate-sensors": Each sensor gets its own group named "<sensor_id>" unless overwritten, remaining components use default store
                - "separate-all": Each component type gets its own group named after the component type, e.g. "poses", "intrinsics", respecting overwrites if provided
            poses_component_group: Override for poses group
            intrinsics_component_group: Override for intrinsics group
            masks_component_group: Override for masks group
            camera_component_groups: Override for per-camera groups
            lidar_component_groups: Override for per-lidar groups
            radar_component_groups: Override for per-radar groups
            cuboid_track_observations_component_group: Override for cuboids group

        Returns:
            ComponentGroups with groups assigned according to profile
        """
        # Get all available sensor IDs and assign each sensor to its own group
        camera_groups = {camera_id: camera_id for camera_id in camera_ids}
        lidar_groups = {lidar_id: lidar_id for lidar_id in lidar_ids}
        radar_groups = {radar_id: radar_id for radar_id in radar_ids}

        # Apply optional overwrites
        if camera_component_groups is not None:
            camera_groups.update(camera_component_groups)
        if lidar_component_groups is not None:
            lidar_groups.update(lidar_component_groups)
        if radar_component_groups is not None:
            radar_groups.update(radar_component_groups)

        if profile == "default":
            return ComponentGroupAssignments(
                poses_component_group=poses_component_group,
                intrinsics_component_group=intrinsics_component_group,
                masks_component_group=masks_component_group,
                camera_component_groups=camera_component_groups if camera_component_groups else {},
                lidar_component_groups=lidar_component_groups if lidar_component_groups else {},
                radar_component_groups=radar_component_groups if radar_component_groups else {},
                cuboid_track_observations_component_group=cuboid_track_observations_component_group,
            )

        elif profile == "separate-sensors":
            return ComponentGroupAssignments(
                poses_component_group=poses_component_group,
                intrinsics_component_group=intrinsics_component_group,
                masks_component_group=masks_component_group,
                camera_component_groups=camera_groups,
                lidar_component_groups=lidar_groups,
                radar_component_groups=radar_groups,
                cuboid_track_observations_component_group=cuboid_track_observations_component_group,
            )

        elif profile == "separate-all":
            return ComponentGroupAssignments(
                poses_component_group="poses" if poses_component_group is None else poses_component_group,
                intrinsics_component_group="intrinsics"
                if intrinsics_component_group is None
                else intrinsics_component_group,
                masks_component_group="masks" if masks_component_group is None else masks_component_group,
                camera_component_groups=camera_groups,
                lidar_component_groups=lidar_groups,
                radar_component_groups=radar_groups,
                cuboid_track_observations_component_group="cuboids"
                if cuboid_track_observations_component_group is None
                else cuboid_track_observations_component_group,
            )
        else:
            raise ValueError(f"Unknown profile: {profile}. Must be one of 'default', 'separate-sensors', 'separate-all")
