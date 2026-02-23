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

import json
import logging

from dataclasses import asdict, dataclass
from typing import Literal

import click
import numpy as np
import tqdm

from upath import UPath

from ncore.impl.common.transformations import (
    HalfClosedInterval,
    PoseInterpolator,
    se3_inverse,
)
from ncore.impl.data.types import JsonLike, OpenCVPinholeCameraModelParameters
from ncore.impl.data.v4.components import (
    CameraSensorComponent,
    CuboidsComponent,
    IntrinsicsComponent,
    LidarSensorComponent,
    MasksComponent,
    PosesComponent,
    SequenceComponentGroupsWriter,
)
from ncore.impl.data.v4.types import ComponentGroupAssignments
from ncore.impl.data_converter.base import BaseDataConverter, BaseDataConverterConfig
from tools.data_converter.cli import cli
from tools.data_converter.colmap.scene_manager import ColmapSceneManager


@dataclass(kw_only=True, slots=True)
class ColmapConverter4Config(BaseDataConverterConfig):
    """Configuration for Waymo to NCore V4 conversion."""

    store_type: Literal["itar", "directory"] = "itar"
    component_group_profile: Literal["default", "separate-sensors", "separate-all"] = "separate-sensors"
    store_sequence_meta: bool = True
    start_time_sec: float = 0.0
    camera_prefix: str = "camera"
    include_downsampled_images: bool = True
    use_3d_points: bool = True


class ColmapDataConverter(BaseDataConverter):
    """
    NVIDIA-specific data conversion to ncore-v3 (based on Maglev ncore-pp workflows data extraction)
    """

    def __init__(self, config: ColmapConverter4Config):
        super().__init__(config)

        self.camera_prefix = config.camera_prefix
        self.start_time_sec = config.start_time_sec
        self.store_type: Literal["itar", "directory"] = config.store_type
        self.component_group_profile: Literal["default", "separate-sensors", "separate-all"] = (
            config.component_group_profile
        )

        # Downsampled images in folders images_2, images_4, etc will be included as additional cameras
        self.include_downsampled_images = config.include_downsampled_images
        self.use_3d_points = config.use_3d_points
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def get_sequence_paths(config) -> list[UPath]:
        return [p for p in UPath(config.root_dir).iterdir() if p.is_dir()]

    @staticmethod
    def from_config(config) -> ColmapDataConverter:
        return ColmapDataConverter(config)

    def convert_sequence(self, sequence_path: UPath) -> None:
        """
        Runs the conversion of a single sequence
        """
        self.logger.info(f"Processing sequence: {sequence_path}")

        self.sequence_path = sequence_path
        self.sequence_name = sequence_path.name

        bin_path = self.sequence_path / "sparse" / "0"
        self.scene_manager = ColmapSceneManager(str(bin_path))
        try:
            T_rig_worlds, T_rig_world_timestamps_us = self.scene_manager.process(
                parent_dir=self.sequence_path,
                camera_prefix=self.camera_prefix,
                start_time_sec=self.start_time_sec,
                downsample=self.include_downsampled_images,
            )
        except IOError as e:
            self.logger.error(f"Error loading data from {sequence_path}: {e}")
            return

        self.camera_ids = list(self.scene_manager.camera_info.keys())
        self.lidar_ids = ["dummy_lidar"] if len(self.scene_manager.points3D) > 0 and self.use_3d_points else []
        self.logger.info(f"Adding cameras: {self.camera_ids} and lidar: {self.lidar_ids}")

        self.pose_interpolator = PoseInterpolator(T_rig_worlds, T_rig_world_timestamps_us)
        self.T_world_world_global = np.eye(4, dtype="float64")

        # Create timestamp interval for sequence
        sequence_timestamp_interval_us = HalfClosedInterval.from_start_end(
            self.pose_interpolator.timestamps[0].item(),
            self.pose_interpolator.timestamps[-1].item(),
        )

        self.component_groups = ComponentGroupAssignments.create(
            camera_ids=self.camera_ids,
            lidar_ids=self.lidar_ids,
            radar_ids=[],  # No radars for now
            profile="default",
        )

        # Create main store writer
        self.store_writer = SequenceComponentGroupsWriter(
            output_dir_path=self.output_dir / self.sequence_name,
            store_base_name=self.sequence_name,
            sequence_id=self.sequence_name,
            sequence_timestamp_interval_us=sequence_timestamp_interval_us,
            store_type=self.store_type,
            generic_meta_data={},  # no generic sequence meta data
        )

        # Create poses component
        self.poses_writer = self.store_writer.register_component_writer(
            PosesComponent.Writer,
            component_instance_name="default",
            group_name=self.component_groups.poses_component_group,
            generic_meta_data={
                "calibration_type": "waymo:calib",
                "egomotion_type": "waymo:egomotion",
            },
        )

        # Create intrinsics component
        self.intrinsics_writer = self.store_writer.register_component_writer(
            IntrinsicsComponent.Writer,
            component_instance_name="default",
            group_name=self.component_groups.intrinsics_component_group,
        )

        # Create masks component
        self.masks_writer = self.store_writer.register_component_writer(
            MasksComponent.Writer,
            component_instance_name="default",
            group_name=self.component_groups.masks_component_group,
        )

        self.store_writer.register_component_writer(
            CuboidsComponent.Writer,
            "default",
            self.component_groups.cuboid_track_observations_component_group,
        )  # .store_observations([])

        ## Store poses
        self.store_poses(T_rig_worlds, T_rig_world_timestamps_us)

        ## Decode 3D Points as a single lidar frame
        if self.lidar_ids:
            self.decode_lidars(T_rig_worlds[0, :, :])

        ## Decode and store camera frames
        self.decode_cameras()

        # Store per-shard meta data / final success state / close file
        ncore_4_paths = self.store_writer.finalize()

        # Output sequence meta file if requested
        # if self.store_sequence_meta:
        #     sequence_component_reader = SequenceComponentGroupsReader(ncore_4_paths)
        #     sequence_meta_path = self.output_dir / self.sequence_name / f"{sequence_component_reader.sequence_id}.json"

        #     with sequence_meta_path.open("w") as f:
        #         json.dump(sequence_component_reader.get_sequence_meta().to_dict(), f, indent=2)

        #     self.logger.info(f"Wrote sequence meta data {str(sequence_meta_path)}")

    def store_poses(self, T_rig_worlds, T_rig_world_timestamps_us):
        """Stores the processed egomotion poses into the poses component."""
        # Store dynamic rig->world poses (float32 for relative poses)
        self.poses_writer.store_dynamic_pose(
            source_frame_id="rig",
            target_frame_id="world",
            poses=T_rig_worlds.astype(np.float32),
            timestamps_us=T_rig_world_timestamps_us,
        )

        # Store static world->world_global pose (float64 for high precision)
        self.poses_writer.store_static_pose(
            source_frame_id="world",
            target_frame_id="world_global",
            pose=self.T_world_world_global.astype(np.float64),
        )

    def decode_lidars(self, T_rig_world_0) -> None:
        # Extrinsics
        T_sensor_rig = se3_inverse(T_rig_world_0)
        # T_sensor_rig = np.identity(4, dtype=np.float32)
        lidar_ncore_id = self.lidar_ids[0]

        lidar_writer = self.store_writer.register_component_writer(
            LidarSensorComponent.Writer,
            component_instance_name=lidar_ncore_id,
            group_name=self.component_groups.lidar_component_groups.get(lidar_ncore_id),
            generic_meta_data={},
        )

        xyz = self.scene_manager.points3D.astype(np.float32)
        distance = np.linalg.norm(xyz, axis=1)  # N
        direction = xyz / distance[:, np.newaxis]

        # Serialize lidar frame
        lidar_writer.store_frame(
            # Non-motion-compensated per-ray 3D directions (float32, [N, 3])
            direction=direction,
            # Per-point timestamp in microseconds (uint64, [N])
            timestamp_us=np.zeros((self.scene_manager.points3D.shape[0]), dtype=np.uint64),
            # Per-point model element indices (uint16, [N, 2])
            model_element=None,
            # Per-point distance (two returns, [2, N])
            distance_m=distance[np.newaxis, :],
            # Per-point intensity normalized to [0.0, 1.0] (float32, [2, N])
            intensity=np.zeros((1, self.scene_manager.points3D.shape[0]), dtype=np.float32),
            # Frame start/end timestamps (uint64, [2])
            frame_timestamps_us=np.array([0, 0], dtype=np.uint64),
            generic_data={
                "rgb": self.scene_manager.point3D_colors,  # uint8
            },
            generic_meta_data={},
        )

        # Store intrinsics
        # self.intrinsics_writer.store_lidar_intrinsics(
        #     lidar_id=lidar_ncore_id,
        #     lidar_model_parameters=lidar_model_parameters,
        # )

        # Store extrinsics (sensor->rig transform)
        self.poses_writer.store_static_pose(
            source_frame_id=lidar_ncore_id,
            target_frame_id="rig",
            pose=T_sensor_rig,
        )

    def decode_cameras(self) -> None:
        # Extrinsics
        T_sensor_rig = np.identity(4, dtype=np.float32)

        for camera_ncore_id, camera_info in self.scene_manager.camera_info.items():
            # camera ids are just numbers
            image_root: str = camera_info["image_root"]
            image_names: list[str] = camera_info["image_names"]
            T_rig_worlds: np.ndarray = camera_info["T_rig_worlds"]
            timestamps_us: np.ndarray = camera_info["timestamps_us"]
            camera_model: OpenCVPinholeCameraModelParameters = camera_info["camera_model"]

            # Create camera sensor writer
            camera_writer = self.store_writer.register_component_writer(
                CameraSensorComponent.Writer,
                component_instance_name=camera_ncore_id,
                group_name=self.component_groups.camera_component_groups.get(camera_ncore_id),
                generic_meta_data={
                    # "label-class-string-id-map": {
                    #     label_string: label_id
                    #     for label_id, label_string in self.CAMERA_LABEL_CLASS_ID_STRING_MAP.items()
                    # }
                },
            )

            # Store intrinsics
            self.intrinsics_writer.store_camera_intrinsics(
                camera_id=camera_ncore_id,
                camera_model_parameters=camera_model,
            )

            # Store empty masks (as none available in dataset)
            self.masks_writer.store_camera_masks(
                camera_id=camera_ncore_id,
                mask_images={},
            )

            # Store extrinsics (sensor->rig transform)
            self.poses_writer.store_static_pose(
                source_frame_id=camera_ncore_id,
                target_frame_id="rig",
                pose=T_sensor_rig,
            )

            for continuous_frame_index in tqdm.tqdm(range(len(image_names)), desc=f"Decoding {camera_ncore_id}"):
                image_name = image_names[continuous_frame_index]
                image_path = self.sequence_path / image_root / image_name

                if (file_extension := image_name.split(".")[-1].lower()) == "jpg":
                    # PIL in ncore uses jpeg instead of jpg
                    file_extension = "jpeg"

                ## Load current camera's image
                with image_path.open("rb") as f:
                    image_bytes = f.read()

                # single pose and timestamp for global shutter
                # frame_T_rig_worlds = np.array(
                #     [T_rig_worlds[continuous_frame_index, :, :], T_rig_worlds[continuous_frame_index, :, :]]
                # ).astype(np.float32)
                frame_timestamps_us = np.array(
                    [timestamps_us[continuous_frame_index], timestamps_us[continuous_frame_index]]
                ).astype(np.uint64)

                generic_data: dict[str, np.ndarray] = {}
                generic_meta_data: dict[str, JsonLike] = {}

                if continuous_frame_index == 0:
                    import io

                    import PIL.Image as PILImage

                    img = PILImage.open(io.BytesIO(image_bytes))
                    self.logger.info(
                        f"{camera_ncore_id} resolution: {np.asarray(img).shape} vs {camera_model.resolution}"
                    )

                camera_writer.store_frame(
                    image_binary_data=image_bytes,
                    image_format=file_extension,
                    frame_timestamps_us=frame_timestamps_us,
                    generic_data=generic_data,
                    generic_meta_data=generic_meta_data,
                )


@cli.command()
@click.option(
    "--store-type",
    type=click.Choice(["itar", "directory"], case_sensitive=False),
    default="itar",
    show_default=True,
    help="Output store type",
)
@click.option(
    "component_group_profile",
    "--profile",
    type=click.Choice(["default", "separate-sensors", "separate-all"], case_sensitive=False),
    default="separate-sensors",
    show_default=True,
    help=""""Output profile, one of:
        - "default": All components defaults or overrides
        - "separate-sensors": Each sensor gets its own group named "<sensor_id>", remaining components use overrides
        - "separate-all": Each component type gets its own group named after the component type, e.g. "poses", "intrinsics", respecting overwrites if provided""",
)
@click.option(
    "store_sequence_meta", "--sequence-meta/--no-sequence-meta", default=True, help="Generate sequence meta-data?"
)
@click.option(
    "--start-time-sec",
    type=click.FloatRange(min=0.0, max_open=True),
    default=0.0,
    help="Start time (in seconds).",
)
@click.option(
    "--camera-prefix",
    type=str,
    default="camera",
    help="Camera name prefix (for integer camera_ids). Default is 'camera'.",
)
@click.option(
    "--include-downsampled-images",
    type=click.BOOL,
    default=True,
    help="Include downsampled images as additional cameras. Images should be in folders such as `images_2`, etc.",
)
@click.option(
    "--use-3d-points",
    type=click.BOOL,
    default=True,
    help="Use 3D Points as a lidar sensor",
)
@click.pass_context
def colmap_v4(ctx, *_, **kwargs):
    """Colmap-specific data conversion (V4 format)"""

    # Extend base config with command-specific options
    config = ColmapConverter4Config(**{**asdict(ctx.obj), **kwargs})

    ColmapDataConverter.convert(config)


if __name__ == "__main__":
    cli(show_default=True)
