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

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional


@dataclass
class ComponentGroupAssignments:
    """Component group assignments for all default components in a V4 data output"""

    camera_component_groups: Dict[str, str]  # indexed by camera_id
    lidar_component_groups: Dict[str, str]  # indexed by lidar_id
    radar_component_groups: Dict[str, str]  # indexed by radar_id
    point_clouds_component_groups: Dict[str, str]  # indexed by point_clouds_id
    camera_labels_component_groups: Dict[str, str]  # indexed by instance_name (e.g. "DEPTH@cam0")
    poses_component_group: Optional[str]
    intrinsics_component_group: Optional[str]
    masks_component_group: Optional[str]
    cuboid_track_observations_component_group: Optional[str]

    @staticmethod
    def create(
        camera_ids: List[str],
        lidar_ids: List[str],
        radar_ids: List[str],
        point_clouds_ids: List[str],
        camera_labels_ids: List[str],
        profile: Literal["default", "separate-sensors", "separate-all"],
        # Component-specific overrides
        poses_component_group: Optional[str] = None,
        intrinsics_component_group: Optional[str] = None,
        masks_component_group: Optional[str] = None,
        camera_component_groups: Optional[Dict[str, str]] = None,
        lidar_component_groups: Optional[Dict[str, str]] = None,
        radar_component_groups: Optional[Dict[str, str]] = None,
        point_clouds_component_groups: Optional[Dict[str, str]] = None,
        camera_labels_component_groups: Optional[Dict[str, str]] = None,
        cuboid_track_observations_component_group: Optional[str] = None,
    ) -> ComponentGroupAssignments:
        """Factory function to create ComponentGroups based on a profile.

        Args:
            camera_ids: IDs of camera sensors
            lidar_ids: IDs of lidar sensors
            radar_ids: IDs of radar sensors
            point_clouds_ids: IDs of native point cloud sources
            camera_labels_ids: IDs of camera label instances (e.g. "DEPTH@cam0")
            profile: One of:
                - "default": Use provided overrides or fall back to default groups
                - "separate-sensors": Each sensor gets its own group named "<sensor_id>" unless overwritten, remaining components use default store
                - "separate-all": Each component type gets its own group named after the component type, e.g. "poses", "intrinsics", respecting overwrites if provided
            poses_component_group: Override for poses group
            intrinsics_component_group: Override for intrinsics group
            masks_component_group: Override for masks group
            camera_component_groups: Override for per-camera groups
            lidar_component_groups: Override for per-lidar groups
            radar_component_groups: Override for per-radar groups
            point_clouds_component_groups: Override for per-point-cloud groups
            camera_labels_component_groups: Override for per-camera-label groups
            cuboid_track_observations_component_group: Override for cuboids group

        Returns:
            ComponentGroups with groups assigned according to profile
        """
        # Get all available sensor / source IDs and assign each to its own group
        camera_groups = {camera_id: camera_id for camera_id in camera_ids}
        lidar_groups = {lidar_id: lidar_id for lidar_id in lidar_ids}
        radar_groups = {radar_id: radar_id for radar_id in radar_ids}
        pc_groups = {pc_id: pc_id for pc_id in point_clouds_ids}
        cl_groups = {cl_id: cl_id for cl_id in camera_labels_ids}

        # Apply optional overwrites
        if camera_component_groups is not None:
            camera_groups.update(camera_component_groups)
        if lidar_component_groups is not None:
            lidar_groups.update(lidar_component_groups)
        if radar_component_groups is not None:
            radar_groups.update(radar_component_groups)
        if point_clouds_component_groups is not None:
            pc_groups.update(point_clouds_component_groups)
        if camera_labels_component_groups is not None:
            cl_groups.update(camera_labels_component_groups)

        if profile == "default":
            return ComponentGroupAssignments(
                poses_component_group=poses_component_group,
                intrinsics_component_group=intrinsics_component_group,
                masks_component_group=masks_component_group,
                camera_component_groups=camera_component_groups if camera_component_groups else {},
                lidar_component_groups=lidar_component_groups if lidar_component_groups else {},
                radar_component_groups=radar_component_groups if radar_component_groups else {},
                point_clouds_component_groups=point_clouds_component_groups if point_clouds_component_groups else {},
                camera_labels_component_groups=camera_labels_component_groups if camera_labels_component_groups else {},
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
                point_clouds_component_groups=pc_groups,
                camera_labels_component_groups=cl_groups,
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
                point_clouds_component_groups=pc_groups,
                camera_labels_component_groups=cl_groups,
                cuboid_track_observations_component_group="cuboids"
                if cuboid_track_observations_component_group is None
                else cuboid_track_observations_component_group,
            )
        else:
            raise ValueError(f"Unknown profile: {profile}. Must be one of 'default', 'separate-sensors', 'separate-all")
