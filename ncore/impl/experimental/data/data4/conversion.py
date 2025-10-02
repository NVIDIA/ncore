# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from typing_extensions import Literal

from ncore.impl.data.data import JsonLike
from ncore.impl.data.data3 import ShardDataLoader
from ncore.impl.data.types import FrameTimepoint
from ncore.impl.experimental.data.data4.components import (
    CameraSensorComponent,
    PosesSetComponent,
    SensorIntrinsicsComponent,
    SequenceComponentStoreWriter,
)


class Data3Converter:
    """Data3Converter converts Ncore V3 into V4"""

    @staticmethod
    def convert(
        source_data_loader: ShardDataLoader,
        target_output_dir_path: Path,
        # component target group overwrite
        component_group_targets: Dict[
            str, str
        ] = {},  # Map from component group name to target component store name (missing key for default). Valid names are "poses", "intrinsics", "<camera_id>", "<lidar_id>", "<radar_id>", "cuboid_tracks"
        generic_meta_data: Dict[
            str, JsonLike
        ] = {},  # generic sequence meta-data (needs to be json-serializable) - will be merged with source generic meta data
        store_type: Literal["itar", "directory"] = "itar",  # valid values: ['itar', 'directory']
    ):
        """Converts the given V3 sequence (unconditionally rig-trajectory-based) into V4 format, storing the result into the given output directory path, potentially overwriting component group names as specified"""

        ## create sequence writer

        # import sequence generic meta-data
        source_generic_meta_data = source_data_loader.get_generic_meta_data()
        source_generic_meta_data["calibration_type"] = source_data_loader.get_calibration_type()
        source_generic_meta_data["egomotion_type"] = source_data_loader.get_egomotion_type()

        store_writer = SequenceComponentStoreWriter(
            output_dir_path=target_output_dir_path,
            store_base_name=source_data_loader.get_sequence_id(),
            sequence_id=source_data_loader.get_sequence_id(),
            store_type=store_type,
            generic_meta_data={**source_generic_meta_data, **generic_meta_data},
        )

        ## create poses component, store rig poses
        poses_writer = store_writer.register_component_writer(
            PosesSetComponent.Writer,
            component_instance_name="poses",
            group_name=component_group_targets.get("poses", None),
        )

        source_poses = source_data_loader.get_poses()

        poses_writer.store_dynamic_poses(
            source_frame="rig",
            target_frame="world",
            poses=source_poses.T_rig_worlds,
            timestamps_us=source_poses.T_rig_world_timestamps_us,
        ).store_static_pose(
            source_frame="world",
            target_frame="world_global",
            pose=source_poses.T_rig_world_base,
        )

        ## create intrinsics component
        intrinsics_writer = store_writer.register_component_writer(
            SensorIntrinsicsComponent.Writer,
            component_instance_name="intrinsics",
            group_name=component_group_targets.get("intrinsics", None),
        )

        ## iterate over all sensors, convert and store their data

        # cameras
        for camera_id in source_data_loader.get_camera_ids():
            camera_sensor = source_data_loader.get_camera_sensor(camera_id)

            # store intrinsics
            intrinsics_writer.store_camera_intrinsics(
                camera_id=camera_id,
                camera_model_parameters=camera_sensor.get_camera_model_parameters(),
                mask_image=camera_sensor.get_camera_mask_image(),
            )

            # store extrinsics
            poses_writer.store_static_pose(
                source_frame=camera_id,
                target_frame="rig",
                pose=camera_sensor.get_T_sensor_rig(),
            )

            # store frames
            camera_writer = store_writer.register_component_writer(
                CameraSensorComponent.Writer,
                component_instance_name=camera_id,
                group_name=component_group_targets.get(camera_id, None),
            )

            for source_frame_idx in range(camera_sensor.get_frames_count()):
                frame_image_data = camera_sensor.get_frame_data(source_frame_idx)

                camera_writer.store_frame(
                    image_binary_data=frame_image_data.get_encoded_image_data(),
                    image_format=frame_image_data.get_encoded_image_format(),
                    timestamps_us=np.array(
                        [
                            camera_sensor.get_frame_timestamp_us(source_frame_idx, FrameTimepoint.START),
                            camera_sensor.get_frame_timestamp_us(source_frame_idx, FrameTimepoint.END),
                        ],
                        dtype=np.uint64,
                    ),
                    generic_data={
                        generic_data_name: camera_sensor.get_frame_generic_data(source_frame_idx, generic_data_name)
                        for generic_data_name in camera_sensor.get_frame_generic_data_names(source_frame_idx)
                    },
                    generic_meta_data=camera_sensor.get_frame_generic_meta_data(source_frame_idx),
                )

        ## finalize
        store_writer.finalize()
