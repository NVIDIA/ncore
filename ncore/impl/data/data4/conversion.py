# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

import logging

from dataclasses import dataclass
from typing import Dict, List, Optional, cast

import numpy as np

from typing_extensions import Literal
from upath import UPath

from ncore.impl.common.common import HalfClosedInterval, log_progress
from ncore.impl.common.transformations import MotionCompensator
from ncore.impl.data.data import JsonLike
from ncore.impl.data.data3 import Poses, ShardDataLoader
from ncore.impl.data.data4.components import (
    CameraSensorComponent,
    CuboidsComponent,
    IntrinsicsComponent,
    LidarSensorComponent,
    MasksComponent,
    PosesComponent,
    RadarSensorComponent,
    SequenceComponentGroupsWriter,
)
from ncore.impl.data.data4.types import CuboidTrackObservation
from ncore.impl.data.types import FrameTimepoint


_logger = logging.getLogger(__name__)


class NCore3To4:
    @dataclass
    class ComponentGroups:
        """Component group assignments for all components in the V4 data output."""

        poses_component_group: Optional[str]
        intrinsics_component_group: Optional[str]
        masks_component_group: Optional[str]
        camera_component_groups: Dict[str, str]  # indexed by camera_id
        lidar_component_groups: Dict[str, str]  # indexed by lidar_id
        radar_component_groups: Dict[str, str]  # indexed by radar_id
        cuboid_track_observations_component_group: Optional[str]

    @staticmethod
    def create_component_groups(
        source_data_loader: ShardDataLoader,
        profile: Literal["default", "separate-sensors", "separate-all"],
        # Component-specific overrides
        poses_component_group: Optional[str] = None,
        intrinsics_component_group: Optional[str] = None,
        masks_component_group: Optional[str] = None,
        camera_component_groups: Optional[Dict[str, str]] = None,
        lidar_component_groups: Optional[Dict[str, str]] = None,
        radar_component_groups: Optional[Dict[str, str]] = None,
        cuboid_track_observations_component_group: Optional[str] = None,
    ) -> ComponentGroups:
        """Factory function to create ComponentGroups based on a profile.

        Args:
            source_data_loader: ShardDataLoader to determine available sensors
            profile: One of:
                - "default": Use provided overrites or fall back to default groups
                - "separate-sensors": Each sensor gets its own group named "<sensor_id>", remaining components use default store
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
        camera_groups = {camera_id: camera_id for camera_id in source_data_loader.get_camera_ids()}
        lidar_groups = {lidar_id: lidar_id for lidar_id in source_data_loader.get_lidar_ids()}
        radar_groups = {radar_id: radar_id for radar_id in source_data_loader.get_radar_ids()}

        if profile == "default":
            return NCore3To4.ComponentGroups(
                poses_component_group=poses_component_group,
                intrinsics_component_group=intrinsics_component_group,
                masks_component_group=masks_component_group,
                camera_component_groups=camera_component_groups if camera_component_groups else {},
                lidar_component_groups=lidar_component_groups if lidar_component_groups else {},
                radar_component_groups=radar_component_groups if radar_component_groups else {},
                cuboid_track_observations_component_group=cuboid_track_observations_component_group,
            )

        elif profile == "separate-sensors":
            return NCore3To4.ComponentGroups(
                poses_component_group=poses_component_group,
                intrinsics_component_group=intrinsics_component_group,
                masks_component_group=masks_component_group,
                camera_component_groups=camera_groups,
                lidar_component_groups=lidar_groups,
                radar_component_groups=radar_groups,
                cuboid_track_observations_component_group=cuboid_track_observations_component_group,
            )

        elif profile == "separate-all":
            return NCore3To4.ComponentGroups(
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

    """Performs data conversion from NCore 3 to NCore 4 format"""

    @staticmethod
    def _convert_cameras(
        source_data_loader: ShardDataLoader,
        store_writer: SequenceComponentGroupsWriter,
        poses_writer: PosesComponent.Writer,
        intrinsics_writer: IntrinsicsComponent.Writer,
        masks_writer: MasksComponent.Writer,
        camera_ids: Optional[List[str]],
        camera_component_groups: Dict[str, str],
    ) -> None:
        """Convert and store camera sensor data from V3 to V4 format"""
        for camera_id in log_progress(
            camera_ids if camera_ids is not None else source_data_loader.get_camera_ids(),
            _logger,
            label="Converting cameras",
        ):
            camera_sensor = source_data_loader.get_camera_sensor(camera_id)

            # store intrinsics
            intrinsics_writer.store_camera_intrinsics(
                camera_id=camera_id,
                camera_model_parameters=camera_sensor.get_camera_model_parameters(),
            )

            # store mask to "default" name if available
            masks_writer.store_camera_masks(
                camera_id=camera_id,
                mask_images={"ego": camera_mask_image}
                if (camera_mask_image := camera_sensor.get_camera_mask_image())
                else {},
            )

            # store extrinsics
            poses_writer.store_static_pose(
                source_frame_id=camera_id,
                target_frame_id="rig",
                # we store sensor->rig poses as float32 in V4
                # (same as in V3, just be explicit about it)
                pose=camera_sensor.get_T_sensor_rig().astype(np.float32),
            )

            # store frames
            camera_writer = store_writer.register_component_writer(
                CameraSensorComponent.Writer,
                component_instance_name=camera_id,
                group_name=camera_component_groups.get(camera_id),
            )

            for source_frame_idx in log_progress(
                range(camera_sensor.get_frames_count()),
                _logger,
                label=f"Converting frames for camera {camera_id}",
                step_frequency=50,
            ):
                # skip frames outside of selected time range
                frame_timestamps_us = np.array(
                    [
                        camera_sensor.get_frame_timestamp_us(source_frame_idx, FrameTimepoint.START),
                        camera_sensor.get_frame_timestamp_us(source_frame_idx, FrameTimepoint.END),
                    ],
                    dtype=np.uint64,
                )

                if (
                    not HalfClosedInterval.from_start_end(*frame_timestamps_us.tolist())
                    in store_writer.sequence_timestamp_interval_us
                ):
                    continue

                frame_image_data = camera_sensor.get_frame_data(source_frame_idx)

                camera_writer.store_frame(
                    image_binary_data=frame_image_data.get_encoded_image_data(),
                    image_format=frame_image_data.get_encoded_image_format(),
                    frame_timestamps_us=frame_timestamps_us,
                    generic_data={
                        generic_data_name: camera_sensor.get_frame_generic_data(source_frame_idx, generic_data_name)
                        for generic_data_name in camera_sensor.get_frame_generic_data_names(source_frame_idx)
                    },
                    generic_meta_data=camera_sensor.get_frame_generic_meta_data(source_frame_idx),
                )

    @staticmethod
    def _convert_radars(
        source_data_loader: ShardDataLoader,
        store_writer: SequenceComponentGroupsWriter,
        poses_writer: PosesComponent.Writer,
        radar_ids: Optional[List[str]],
        radar_component_groups: Dict[str, str],
    ) -> None:
        """Convert and store radar sensor data from V3 to V4 format"""
        for radar_id in log_progress(
            radar_ids if radar_ids is not None else source_data_loader.get_radar_ids(),
            _logger,
            label="Converting radars",
        ):
            radar_sensor = source_data_loader.get_radar_sensor(radar_id)

            # store extrinsics
            poses_writer.store_static_pose(
                source_frame_id=radar_id,
                target_frame_id="rig",
                # we store sensor->rig poses as float32 in V4
                # (same as in V3, just be explicit about it)
                pose=radar_sensor.get_T_sensor_rig().astype(np.float32),
            )

            # store frames
            radar_writer = store_writer.register_component_writer(
                RadarSensorComponent.Writer,
                component_instance_name=radar_id,
                group_name=radar_component_groups.get(radar_id),
            )

            for source_frame_idx in log_progress(
                range(radar_sensor.get_frames_count()),
                _logger,
                label=f"Converting frames for radar {radar_id}",
                step_frequency=50,
            ):
                # skip frames outside of selected time range
                frame_timestamps_us = np.array(
                    [
                        radar_sensor.get_frame_timestamp_us(source_frame_idx, FrameTimepoint.START),
                        radar_sensor.get_frame_timestamp_us(source_frame_idx, FrameTimepoint.END),
                    ],
                    dtype=np.uint64,
                )

                if (
                    not HalfClosedInterval.from_start_end(*frame_timestamps_us.tolist())
                    in store_writer.sequence_timestamp_interval_us
                ):
                    continue

                # V3 data is stored at "instantaneous" end-of-frame times, so there is no need to undo motion-compensation
                xyz_m = radar_sensor.get_frame_data(source_frame_idx, "xyz_e")
                timestamp_us = np.array(
                    [frame_timestamps_us[1]] * len(xyz_m), dtype=np.uint64
                )  # all points have the same timestamp in V3 data

                # extract directions / distances
                distance_m = np.linalg.norm(xyz_m, axis=1)
                direction = xyz_m / distance_m[:, np.newaxis]

                # filter for non-negative distances
                valid_mask = distance_m > 0

                radar_writer.store_frame(
                    # non-motion-compensated per-point 3D directions in the sensor frame at measurement time (float32, [n, 3])
                    direction=direction[valid_mask],
                    # per-point point timestamp in microseconds (uint64, [n])
                    timestamp_us=timestamp_us[valid_mask],
                    # single per-point return (only single return supported in V3))
                    distance_m=distance_m[valid_mask][np.newaxis],
                    # frame start/end timestamps (uint64, [2])
                    frame_timestamps_us=frame_timestamps_us,
                    generic_data={
                        generic_data_name: radar_sensor.get_frame_generic_data(source_frame_idx, generic_data_name)
                        for generic_data_name in radar_sensor.get_frame_generic_data_names(source_frame_idx)
                    },
                    generic_meta_data=radar_sensor.get_frame_generic_meta_data(source_frame_idx),
                )

    @staticmethod
    def _convert_lidars(
        source_data_loader: ShardDataLoader,
        store_writer: SequenceComponentGroupsWriter,
        poses_writer: PosesComponent.Writer,
        intrinsics_writer: IntrinsicsComponent.Writer,
        source_poses: Poses,
        lidar_ids: Optional[List[str]],
        lidar_component_groups: Dict[str, str],
    ) -> List[CuboidTrackObservation]:
        """Convert and store lidar sensor data from V3 to V4 format. Returns collected cuboid track observations."""
        cuboid_track_observations: List[CuboidTrackObservation] = []

        for lidar_id in log_progress(
            lidar_ids if lidar_ids is not None else source_data_loader.get_lidar_ids(),
            _logger,
            label="Converting lidars",
        ):
            lidar_sensor = source_data_loader.get_lidar_sensor(lidar_id)

            # store intrinsics conditionally
            if (lidar_model_parameters := lidar_sensor.get_lidar_model_parameters()) is not None:
                intrinsics_writer.store_lidar_intrinsics(
                    lidar_id=lidar_id,
                    lidar_model_parameters=lidar_model_parameters,
                )

            # store extrinsics
            poses_writer.store_static_pose(
                source_frame_id=lidar_id,
                target_frame_id="rig",
                # we store sensor->rig poses as float32 in V4
                # (same as in V3, just be explicit about it)
                pose=lidar_sensor.get_T_sensor_rig().astype(np.float32),
            )

            # store frames
            lidar_writer = store_writer.register_component_writer(
                LidarSensorComponent.Writer,
                component_instance_name=lidar_id,
                group_name=lidar_component_groups.get(lidar_id),
            )

            motion_compensator = MotionCompensator.from_sensor_rig(
                lidar_sensor.get_sensor_id(),
                lidar_sensor.get_T_sensor_rig(),
                source_poses.T_rig_worlds,
                source_poses.T_rig_world_timestamps_us,
            )
            for source_frame_idx in log_progress(
                range(lidar_sensor.get_frames_count()),
                _logger,
                label=f"Converting frames for lidar {lidar_id}",
                step_frequency=50,
            ):
                # skip frames outside of selected time range
                frame_timestamps_us = np.array(
                    [
                        lidar_sensor.get_frame_timestamp_us(source_frame_idx, FrameTimepoint.START),
                        lidar_sensor.get_frame_timestamp_us(source_frame_idx, FrameTimepoint.END),
                    ],
                    dtype=np.uint64,
                )

                if not HalfClosedInterval(*frame_timestamps_us.tolist()) in store_writer.sequence_timestamp_interval_us:
                    continue

                # load relevant V3 data components
                xyz_e = lidar_sensor.get_frame_data(source_frame_idx, "xyz_e")
                timestamp_us = lidar_sensor.get_frame_data(source_frame_idx, "timestamp_us")
                intensity = lidar_sensor.get_frame_data(source_frame_idx, "intensity")

                # undo motion-compensation of V3 point clouds to non-motion-compensated "raw" V4 point cloud
                xyz_m = motion_compensator.motion_decompensate_points(
                    sensor_id=lidar_sensor.get_sensor_id(),
                    xyz_sensorend=xyz_e,
                    timestamp_us=timestamp_us,
                    frame_start_timestamp_us=frame_timestamps_us[0],
                    frame_end_timestamp_us=frame_timestamps_us[1],
                )

                # extract directions / distances
                distance_m = np.linalg.norm(xyz_m, axis=1)
                direction = xyz_m / distance_m[:, np.newaxis]

                # filter for non-negative distances
                valid_mask = distance_m > 0

                lidar_writer.store_frame(
                    # non-motion-compensated per-ray 3D directions in the sensor frame at measurement time (float32, [n, 3])
                    direction=direction[valid_mask],
                    # per-point point timestamp in microseconds (uint64, [n])
                    timestamp_us=timestamp_us[valid_mask],
                    # per-point model element indices, if applicable (uint16, [n, 2])
                    model_element=(
                        lidar_sensor.get_frame_data(source_frame_idx, "model_element")[valid_mask]
                        if lidar_model_parameters is not None
                        else None
                    ),
                    # per-point distance (only single return supported in V3) [n, r] with r=1)
                    distance_m=distance_m[valid_mask][np.newaxis],
                    # per-point intensity normalized to [0.0, 1.0] range (float32, [n, r] with r=1)
                    intensity=intensity[valid_mask][np.newaxis],
                    # frame start/end timestamps (uint64, [2])
                    frame_timestamps_us=frame_timestamps_us,
                    generic_data={
                        generic_data_name: lidar_sensor.get_frame_generic_data(source_frame_idx, generic_data_name)
                        for generic_data_name in lidar_sensor.get_frame_generic_data_names(source_frame_idx)
                    },
                    generic_meta_data=lidar_sensor.get_frame_generic_meta_data(source_frame_idx),
                )

                # collect frame labels
                for frame_label in lidar_sensor.get_frame_labels(source_frame_idx):
                    if frame_label.timestamp_us not in store_writer.sequence_timestamp_interval_us:
                        continue  # sanity check skip labels outside of selected time range (should not happen if input data is consistent)

                    cuboid_track_observations.append(
                        CuboidTrackObservation(
                            track_id=frame_label.track_id,
                            class_id=frame_label.label_class,
                            timestamp_us=frame_label.timestamp_us,
                            reference_frame_id=lidar_id,
                            reference_frame_timestamp_us=frame_timestamps_us[
                                1
                            ].item(),  # frame end timestamp is the reference time of the V3 observations
                            bbox3=frame_label.bbox3,
                            source=frame_label.source,
                            source_version=frame_label.source_version,
                        )
                    )

        return cuboid_track_observations

    @staticmethod
    def convert(
        ## V3 input
        source_data_loader: ShardDataLoader,
        ## V4 output
        output_dir_path: UPath,
        ## Time range selection (None means full range)
        start_timestamp_us: Optional[int] = None,
        end_timestamp_us: Optional[int] = None,
        ## Output store type
        store_type: Literal["itar", "directory"] = "itar",  # valid values: ['itar', 'directory']
        ## Sensor selection (None means all available sensors of that type)
        camera_ids: Optional[List[str]] = None,
        lidar_ids: Optional[List[str]] = None,
        radar_ids: Optional[List[str]] = None,
        ## Component target groups
        component_groups: Optional[NCore3To4.ComponentGroups] = None,
        ## Generic sequence meta-data (needs to be json-serializable) - will be merged with source generic meta data
        generic_meta_data: Dict[str, JsonLike] = {},
    ) -> List[UPath]:
        """Converts the given V3 sequence (unconditionally rig-trajectory-based) into V4 format,
        storing the result into the given output directory, potentially overwriting component group names as specified"""

        if component_groups is None:
            # Use default profile if not specified
            component_groups = NCore3To4.create_component_groups(
                source_data_loader=source_data_loader,
                profile="default",
            )

        ## infer time range from input poses if required and validate time range selection
        source_poses = source_data_loader.get_poses()
        if start_timestamp_us is None:
            start_timestamp_us = source_poses.T_rig_world_timestamps_us[0].item()
        if end_timestamp_us is None:
            end_timestamp_us = source_poses.T_rig_world_timestamps_us[-1].item()

        target_poses_range = HalfClosedInterval.from_start_end(
            cast(int, start_timestamp_us), cast(int, end_timestamp_us)
        ).cover_range(source_poses.T_rig_world_timestamps_us)

        assert len(target_poses_range) >= 2, "at least two poses required in selected time range"

        # define target sequence time interval to coincide with the available egomotion
        target_sequence_timestamp_interval_us = HalfClosedInterval.from_start_end(
            source_poses.T_rig_world_timestamps_us[target_poses_range.start].item(),
            source_poses.T_rig_world_timestamps_us[target_poses_range.stop - 1].item(),
        )

        ## create sequence writer

        # import sequence generic meta-data
        source_generic_meta_data = source_data_loader.get_generic_meta_data()
        source_generic_meta_data["calibration_type"] = source_data_loader.get_calibration_type()
        source_generic_meta_data["egomotion_type"] = source_data_loader.get_egomotion_type()

        store_writer = SequenceComponentGroupsWriter(
            output_dir_path=output_dir_path,
            store_base_name=source_data_loader.get_sequence_id(),
            sequence_id=source_data_loader.get_sequence_id(),
            sequence_timestamp_interval_us=target_sequence_timestamp_interval_us,
            store_type=store_type,
            generic_meta_data={**source_generic_meta_data, **generic_meta_data},
        )

        ## create poses component, store rig poses
        (
            poses_writer := store_writer.register_component_writer(
                PosesComponent.Writer,
                component_instance_name="default",
                group_name=component_groups.poses_component_group,
            )
        ).store_dynamic_pose(
            source_frame_id="rig",
            target_frame_id="world",
            # we store rig->world poses as float32 in V4 (sufficiently accurate as relative to local-world)
            poses=source_poses.T_rig_worlds[target_poses_range].astype(np.float32),
            timestamps_us=source_poses.T_rig_world_timestamps_us[target_poses_range],
        ).store_static_pose(
            source_frame_id="world",
            target_frame_id="world_global",
            # world->world_global potentially requires high-precision, use float64
            pose=source_poses.T_rig_world_base.astype(np.float64),
        )

        ## create intrinsics component
        intrinsics_writer = store_writer.register_component_writer(
            IntrinsicsComponent.Writer,
            component_instance_name="default",
            group_name=component_groups.intrinsics_component_group,
        )

        ## create masks component
        masks_writer = store_writer.register_component_writer(
            MasksComponent.Writer,
            component_instance_name="default",
            group_name=component_groups.masks_component_group,
        )

        ## iterate over all sensors, convert and store their data

        # cameras
        NCore3To4._convert_cameras(
            source_data_loader=source_data_loader,
            store_writer=store_writer,
            poses_writer=poses_writer,
            intrinsics_writer=intrinsics_writer,
            masks_writer=masks_writer,
            camera_ids=camera_ids,
            camera_component_groups=component_groups.camera_component_groups,
        )

        # radars
        NCore3To4._convert_radars(
            source_data_loader=source_data_loader,
            store_writer=store_writer,
            poses_writer=poses_writer,
            radar_ids=radar_ids,
            radar_component_groups=component_groups.radar_component_groups,
        )

        # lidars (collect cuboid track observations to be stored into separate component)
        cuboid_track_observations = NCore3To4._convert_lidars(
            source_data_loader=source_data_loader,
            store_writer=store_writer,
            poses_writer=poses_writer,
            intrinsics_writer=intrinsics_writer,
            source_poses=source_poses,
            lidar_ids=lidar_ids,
            lidar_component_groups=component_groups.lidar_component_groups,
        )

        # store cuboid track observations
        store_writer.register_component_writer(
            CuboidsComponent.Writer,
            "default",
            component_groups.cuboid_track_observations_component_group,
        ).store_observations(cuboid_track_observations)

        ## finalize
        return store_writer.finalize()
