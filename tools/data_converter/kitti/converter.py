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

"""KITTI Raw Dataset to NCore V4 converter."""

from __future__ import annotations

import json
import logging

from dataclasses import dataclass
from typing import Literal

import click
import numpy as np
import tqdm

from upath import UPath

from ncore.impl.common.transformations import HalfClosedInterval, se3_inverse
from ncore.impl.data.types import (
    BBox3,
    CuboidTrackObservation,
    IdealPinholeCameraModelParameters,
    LabelSource,
    ShutterType,
)
from ncore.impl.data.v4.components import (
    CameraSensorComponent,
    CuboidsComponent,
    IntrinsicsComponent,
    LidarSensorComponent,
    MasksComponent,
    PosesComponent,
    SequenceComponentGroupsReader,
    SequenceComponentGroupsWriter,
)
from ncore.impl.data.v4.types import ComponentGroupAssignments
from ncore.impl.data_converter.base import FileBasedDataConverter, FileBasedDataConverterConfig
from tools.data_converter.cli import cli
from tools.data_converter.kitti.utils import (
    OXTS_FIELD_NAMES,
    compute_velodyne_timestamps_us,
    load_oxts_packets,
    load_timestamps,
    load_velodyne_scan,
    parse_calib_cam_to_cam,
    parse_calib_rigid,
    parse_tracklets,
    poses_from_oxts,
)


# -----------------------------------------------------------------------------
# Sensor IDs
# -----------------------------------------------------------------------------

CAMERA_MAP: dict[str, str] = {
    "image_00": "camera_gray_left",
    "image_01": "camera_gray_right",
    "image_02": "camera_color_left",
    "image_03": "camera_color_right",
}

LIDAR_ID = "lidar_top"


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass(kw_only=True, slots=True)
class KittiConverter4Config(FileBasedDataConverterConfig):
    """Configuration for KITTI Raw to NCore V4 conversion."""

    store_type: Literal["itar", "directory"] = "itar"
    component_group_profile: Literal["default", "separate-sensors", "separate-all"] = "separate-sensors"
    store_sequence_meta: bool = True


# -----------------------------------------------------------------------------
# Converter
# -----------------------------------------------------------------------------


class KittiConverter4(FileBasedDataConverter):
    """Dataset preprocessing class for converting KITTI Raw data to NCore V4 format.

    KITTI raw data can be downloaded from https://www.cvlibs.net/datasets/kitti/raw_data.php
    in the form of synchronized+rectified data folders. Further details on the dataset
    are available in https://www.cvlibs.net/publications/Geiger2013IJRR.pdf
    """

    def __init__(self, config: KittiConverter4Config) -> None:
        super().__init__(config)

        self.component_group_profile: Literal["default", "separate-sensors", "separate-all"] = (
            config.component_group_profile
        )
        self.store_type: Literal["itar", "directory"] = config.store_type
        self.store_sequence_meta: bool = config.store_sequence_meta

        self.logger = logging.getLogger(__name__)

    @staticmethod
    def get_sequence_ids(config) -> list[str]:
        """Discover sequence directories matching *_drive_*_sync pattern."""
        return [str(p) for p in sorted(UPath(config.root_dir).glob("*_drive_*_sync"))]

    @staticmethod
    def from_config(config: KittiConverter4Config) -> KittiConverter4:
        return KittiConverter4(config)

    def convert_sequence(self, sequence_id: str) -> None:
        """Runs KITTI-specific conversion for a single sequence."""

        sequence_path = UPath(sequence_id)
        sequence_name = sequence_path.name  # e.g. "2011_09_26_drive_0048_sync"

        self.logger.info(f"Converting sequence: {sequence_name}")

        # Date directory is the parent (contains calibration files)
        date_dir = sequence_path.parent

        # --- Parse calibration ---------------------------------------------
        calib_cam = parse_calib_cam_to_cam(str(date_dir / "calib_cam_to_cam.txt"))
        # Convention: T_a_b transforms points FROM frame a TO frame b
        T_velo_cam0 = parse_calib_rigid(str(date_dir / "calib_velo_to_cam.txt"))  # velo -> cam0
        T_rig_velo = parse_calib_rigid(str(date_dir / "calib_imu_to_velo.txt"))  # rig(=IMU) -> velo

        # Derived: velo -> rig
        T_velo_rig = se3_inverse(T_rig_velo)

        # R_rect_00 extended to 4x4
        R_rect_00 = calib_cam["R_rect_00"]
        R_rect_00_4x4 = np.eye(4, dtype=np.float64)
        R_rect_00_4x4[:3, :3] = R_rect_00

        # --- Load OXTS (ego poses) ----------------------------------------
        oxts_data_dir = sequence_path / "oxts" / "data"
        packets, raw_oxts_array = load_oxts_packets(str(oxts_data_dir))
        T_rig_world, T_world_world_global = poses_from_oxts(packets)

        # Load OXTS timestamps
        oxts_timestamps = load_timestamps(str(sequence_path / "oxts" / "timestamps.txt"))

        # --- Load sensor timestamps ---------------------------------------
        lidar_timestamps_start = load_timestamps(str(sequence_path / "velodyne_points" / "timestamps_start.txt"))
        lidar_timestamps_end = load_timestamps(str(sequence_path / "velodyne_points" / "timestamps_end.txt"))
        lidar_timestamps = load_timestamps(str(sequence_path / "velodyne_points" / "timestamps.txt"))

        n_frames = len(oxts_timestamps)
        assert n_frames >= 2, f"Need at least 2 frames, got {n_frames}"
        assert len(lidar_timestamps_start) == n_frames
        assert len(lidar_timestamps_end) == n_frames

        # --- Determine active sensors -------------------------------------
        camera_ids = self.get_active_camera_ids(list(CAMERA_MAP.values()))
        lidar_ids = self.get_active_lidar_ids([LIDAR_ID])

        # --- Create timestamp interval ------------------------------------
        # Use the widest possible time range to encompass all sensor data.
        # The dynamic poses must cover the full interval, so we extend the poses
        # to match the full sensor time range.
        pose_timestamps_us = np.array(oxts_timestamps, dtype=np.uint64)
        all_timestamps = oxts_timestamps + lidar_timestamps_start + lidar_timestamps_end
        seq_start_us = min(all_timestamps)
        seq_end_us = max(all_timestamps)
        sequence_timestamp_interval_us = HalfClosedInterval.from_start_end(
            seq_start_us,
            seq_end_us,
        )

        # Extend pose timestamps to cover sequence interval boundaries.
        # Replicate boundary poses to cover the full time range.
        if seq_start_us < int(pose_timestamps_us[0]):
            pose_timestamps_us = np.concatenate([np.array([seq_start_us], dtype=np.uint64), pose_timestamps_us])
            T_rig_world = np.concatenate([T_rig_world[:1], T_rig_world], axis=0)

        if seq_end_us > int(pose_timestamps_us[-1]):
            pose_timestamps_us = np.concatenate([pose_timestamps_us, np.array([seq_end_us], dtype=np.uint64)])
            T_rig_world = np.concatenate([T_rig_world, T_rig_world[-1:]], axis=0)

        # --- Component group assignments ----------------------------------
        component_groups = ComponentGroupAssignments.create(
            camera_ids=camera_ids,
            lidar_ids=lidar_ids,
            radar_ids=[],
            point_clouds_ids=[],
            camera_labels_ids=[],
            profile=self.component_group_profile,
        )

        # --- Create main store writer -------------------------------------
        store_writer = SequenceComponentGroupsWriter(
            output_dir_path=self.output_dir / sequence_name,
            store_base_name=sequence_name,
            sequence_id=sequence_name,
            sequence_timestamp_interval_us=sequence_timestamp_interval_us,
            store_type=self.store_type,
            generic_meta_data={},
        )

        # --- Register component writers -----------------------------------
        poses_writer = store_writer.register_component_writer(
            PosesComponent.Writer,
            component_instance_name="default",
            group_name=component_groups.poses_component_group,
            generic_meta_data={
                "calibration_type": "kitti:calib",
                "egomotion_type": "kitti:oxts",
            },
        )

        intrinsics_writer = store_writer.register_component_writer(
            IntrinsicsComponent.Writer,
            component_instance_name="default",
            group_name=component_groups.intrinsics_component_group,
        )

        masks_writer = store_writer.register_component_writer(
            MasksComponent.Writer,
            component_instance_name="default",
            group_name=component_groups.masks_component_group,
        )

        # --- Store ego poses ----------------------------------------------
        poses_writer.store_dynamic_pose(
            source_frame_id="rig",
            target_frame_id="world",
            poses=T_rig_world.astype(np.float32),
            timestamps_us=pose_timestamps_us,
        )

        poses_writer.store_static_pose(
            source_frame_id="world",
            target_frame_id="world_global",
            pose=T_world_world_global,
        )

        # Store raw OXTS as generic data on poses component
        poses_writer.set_generic_data(
            data={
                "oxts_data": raw_oxts_array,
                "oxts_timestamps_us": np.array(oxts_timestamps, dtype=np.uint64),
            },
            meta_data={"oxts_field_names": OXTS_FIELD_NAMES},
        )

        # --- Decode lidars ------------------------------------------------
        if LIDAR_ID in lidar_ids:
            self._decode_lidar(
                sequence_path=sequence_path,
                store_writer=store_writer,
                poses_writer=poses_writer,
                component_groups=component_groups,
                T_velo_rig=T_velo_rig,
                lidar_timestamps_start=lidar_timestamps_start,
                lidar_timestamps_end=lidar_timestamps_end,
                n_frames=n_frames,
            )

        # --- Decode cameras -----------------------------------------------
        self._decode_cameras(
            sequence_path=sequence_path,
            store_writer=store_writer,
            poses_writer=poses_writer,
            intrinsics_writer=intrinsics_writer,
            masks_writer=masks_writer,
            component_groups=component_groups,
            camera_ids=camera_ids,
            calib_cam=calib_cam,
            T_velo_cam0=T_velo_cam0,
            T_rig_velo=T_rig_velo,
            R_rect_00_4x4=R_rect_00_4x4,
            n_frames=n_frames,
        )

        # --- Parse and store tracklets ------------------------------------
        tracklet_path = sequence_path / "tracklet_labels.xml"
        if tracklet_path.exists():
            self._decode_tracklets(
                tracklet_path=tracklet_path,
                store_writer=store_writer,
                component_groups=component_groups,
                lidar_timestamps=lidar_timestamps,
            )

        # --- Finalize -----------------------------------------------------
        ncore_4_paths = store_writer.finalize()

        if self.store_sequence_meta:
            sequence_component_reader = SequenceComponentGroupsReader(ncore_4_paths)
            sequence_meta_path = self.output_dir / sequence_name / f"{sequence_component_reader.sequence_id}.json"

            with sequence_meta_path.open("w") as f:
                json.dump(sequence_component_reader.get_sequence_meta().to_dict(), f, indent=2)

            self.logger.info(f"Wrote sequence meta data {str(sequence_meta_path)}")

    def _decode_lidar(
        self,
        sequence_path: UPath,
        store_writer: SequenceComponentGroupsWriter,
        poses_writer: PosesComponent.Writer,
        component_groups: ComponentGroupAssignments,
        T_velo_rig: np.ndarray,
        lidar_timestamps_start: list[int],
        lidar_timestamps_end: list[int],
        n_frames: int,
    ) -> None:
        """Decode and store all lidar frames."""
        # Create lidar sensor writer
        lidar_writer = store_writer.register_component_writer(
            LidarSensorComponent.Writer,
            component_instance_name=LIDAR_ID,
            group_name=component_groups.lidar_component_groups.get(LIDAR_ID),
            generic_meta_data={},
        )

        # Store lidar extrinsic (velo -> rig)
        poses_writer.store_static_pose(
            source_frame_id=LIDAR_ID,
            target_frame_id="rig",
            pose=T_velo_rig.astype(np.float32),
        )

        velodyne_data_dir = sequence_path / "velodyne_points" / "data"

        for i in tqdm.tqdm(range(n_frames), desc=f"Process {LIDAR_ID}"):
            # Load point cloud
            bin_path = velodyne_data_dir / f"{i:010d}.bin"
            if not bin_path.exists():
                self.logger.warning(f"Missing velodyne file: {bin_path}")
                continue

            points = load_velodyne_scan(str(bin_path))

            # Frame timestamps
            frame_start_us = lidar_timestamps_start[i]
            frame_end_us = lidar_timestamps_end[i]

            # Compute per-point timestamps from azimuth
            point_timestamps_us = compute_velodyne_timestamps_us(points, frame_start_us, frame_end_us)

            # Compute direction and distance from raw xyz
            xyz = points[:, :3]
            distance_m = np.linalg.norm(xyz, axis=1, keepdims=False).astype(np.float32)

            # Avoid division by zero
            valid_mask = distance_m > 0
            direction = np.zeros_like(xyz)
            direction[valid_mask] = xyz[valid_mask] / distance_m[valid_mask, np.newaxis]

            # Intensity (reflectance already in [0,1] range for KITTI)
            intensity = points[:, 3:4].T  # [1, N]

            # Store frame (model_element=None since KITTI velodyne has no structured model)
            lidar_writer.store_frame(
                direction=direction.astype(np.float32),
                timestamp_us=point_timestamps_us,
                model_element=None,
                distance_m=distance_m.reshape(1, -1),  # [1, N] single return
                intensity=intensity.astype(np.float32),
                frame_timestamps_us=np.array([frame_start_us, frame_end_us], dtype=np.uint64),
                generic_data={},
                generic_meta_data={},
            )

    def _decode_cameras(
        self,
        sequence_path: UPath,
        store_writer: SequenceComponentGroupsWriter,
        poses_writer: PosesComponent.Writer,
        intrinsics_writer: IntrinsicsComponent.Writer,
        masks_writer: MasksComponent.Writer,
        component_groups: ComponentGroupAssignments,
        camera_ids: list[str],
        calib_cam: dict[str, np.ndarray],
        T_velo_cam0: np.ndarray,
        T_rig_velo: np.ndarray,
        R_rect_00_4x4: np.ndarray,
        n_frames: int,
    ) -> None:
        """Decode and store all camera frames."""
        for kitti_cam_name, ncore_cam_id in CAMERA_MAP.items():
            if ncore_cam_id not in camera_ids:
                continue

            # Camera index (e.g., "image_02" -> "02")
            cam_idx = kitti_cam_name.split("_")[1]

            # Load camera timestamps
            cam_timestamps = load_timestamps(str(sequence_path / kitti_cam_name / "timestamps.txt"))

            # Get intrinsics from calibration
            P_rect = calib_cam[f"P_rect_{cam_idx}"]
            S_rect = calib_cam.get(f"S_rect_{cam_idx}")

            # Extract intrinsic parameters from P_rect (3x4)
            fu = P_rect[0, 0]
            fv = P_rect[1, 1]
            cu = P_rect[0, 2]
            cv = P_rect[1, 2]

            # Resolution from S_rect (rectified image size)
            if S_rect is not None:
                width = int(S_rect[0, 0])
                height = int(S_rect[0, 1])
            else:
                # Fallback: read from first image
                width, height = 1242, 375

            # Compute camera extrinsic: T_cam_rig (cam -> rig)
            # Chain: rig -> velo -> cam0 -> rectcam0
            T_rig_rectcam0 = R_rect_00_4x4 @ T_velo_cam0 @ T_rig_velo

            # For cameras with stereo baseline, add translation from P_rect column 4
            # P_rect_xx[0,3] = bx * fu, so bx = P_rect_xx[0,3] / fu
            # P_rect_xx[1,3] = by * fv (usually 0 for KITTI horizontal stereo)
            bx = P_rect[0, 3] / fu if fu != 0 else 0.0
            by = P_rect[1, 3] / fv if fv != 0 else 0.0
            bz = P_rect[2, 3] if P_rect.shape[1] > 3 else 0.0

            # Stereo baseline: rectcam0 -> cam_xx
            T_rectcam0_cam = np.eye(4, dtype=np.float64)
            T_rectcam0_cam[0, 3] = bx
            T_rectcam0_cam[1, 3] = by
            T_rectcam0_cam[2, 3] = bz

            # Full chain rig -> cam; invert to get cam -> rig for the static pose
            T_rig_cam = T_rectcam0_cam @ T_rig_rectcam0
            T_cam_rig = se3_inverse(T_rig_cam)

            # Create camera sensor writer
            camera_writer = store_writer.register_component_writer(
                CameraSensorComponent.Writer,
                component_instance_name=ncore_cam_id,
                group_name=component_groups.camera_component_groups.get(ncore_cam_id),
                generic_meta_data={},
            )

            # Store frames
            image_data_dir = sequence_path / kitti_cam_name / "data"
            for i in tqdm.tqdm(range(n_frames), desc=f"Process {ncore_cam_id}"):
                img_path = image_data_dir / f"{i:010d}.png"
                if not img_path.exists():
                    self.logger.warning(f"Missing image file: {img_path}")
                    continue

                # Read PNG as binary
                with img_path.open("rb") as f:
                    image_binary = f.read()

                # Global shutter: start == end timestamp
                frame_ts = cam_timestamps[i]

                camera_writer.store_frame(
                    image_binary_data=image_binary,
                    image_format="png",
                    frame_timestamps_us=np.array([frame_ts, frame_ts], dtype=np.uint64),
                    generic_data={},
                    generic_meta_data={},
                )

            # Store camera intrinsics (rectified = ideal pinhole, no distortion)
            intrinsics_writer.store_camera_intrinsics(
                camera_id=ncore_cam_id,
                camera_model_parameters=IdealPinholeCameraModelParameters(
                    resolution=np.array([width, height], dtype=np.uint64),
                    shutter_type=ShutterType.GLOBAL,
                    external_distortion_parameters=None,
                    principal_point=np.array([cu, cv], dtype=np.float32),
                    focal_length=np.array([fu, fv], dtype=np.float32),
                ),
            )

            # Store empty masks
            masks_writer.store_camera_masks(
                camera_id=ncore_cam_id,
                mask_images={},
            )

            # Store camera extrinsic (sensor -> rig)
            poses_writer.store_static_pose(
                source_frame_id=ncore_cam_id,
                target_frame_id="rig",
                pose=T_cam_rig.astype(np.float32),
            )

    def _decode_tracklets(
        self,
        tracklet_path: UPath,
        store_writer: SequenceComponentGroupsWriter,
        component_groups: ComponentGroupAssignments,
        lidar_timestamps: list[int],
    ) -> None:
        """Parse tracklets and store as cuboid track observations in lidar frame.

        KITTI tracklet poses are in the velodyne frame at the mid-scan reference
        time. The viewer transforms them to world via the pose graph at runtime
        using the lidar_top -> rig -> world chain.
        """
        tracklets = parse_tracklets(str(tracklet_path))

        if not tracklets:
            self.logger.info("No tracklets found")
            return

        cuboid_track_observations: list[CuboidTrackObservation] = []

        for idx, tracklet in enumerate(tracklets):
            track_id = f"{tracklet.object_type}_{idx}"

            for pose_idx, pose in enumerate(tracklet.poses):
                frame_idx = tracklet.first_frame + pose_idx

                if frame_idx >= len(lidar_timestamps):
                    break

                frame_timestamp_us = lidar_timestamps[frame_idx]

                # KITTI tracklet (tx,ty,tz) is at the bottom-center of the box.
                # BBox3 expects the centroid, so shift Z up by half the height.
                centroid_z = pose.tz + tracklet.h / 2.0

                cuboid_track_observations.append(
                    CuboidTrackObservation(
                        track_id=track_id,
                        class_id=tracklet.object_type,
                        timestamp_us=frame_timestamp_us,
                        reference_frame_id=LIDAR_ID,
                        reference_frame_timestamp_us=frame_timestamp_us,
                        bbox3=BBox3.from_array(
                            np.array(
                                [
                                    pose.tx,
                                    pose.ty,
                                    centroid_z,
                                    tracklet.l,
                                    tracklet.w,
                                    tracklet.h,
                                    pose.rx,
                                    pose.ry,
                                    pose.rz,
                                ],
                                dtype=np.float32,
                            )
                        ),
                        source=LabelSource.EXTERNAL,
                    )
                )

        store_writer.register_component_writer(
            CuboidsComponent.Writer,
            "default",
            component_groups.cuboid_track_observations_component_group,
        ).store_observations(cuboid_track_observations)

        self.logger.info(f"Stored {len(cuboid_track_observations)} cuboid observations from {len(tracklets)} tracklets")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


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
        - "separate-all": Each component type gets its own group named after the component type""",
)
@click.option(
    "store_sequence_meta", "--sequence-meta/--no-sequence-meta", default=True, help="Generate sequence meta-data?"
)
@click.pass_context
def kitti_v4(ctx, *_, **kwargs):
    """KITTI Raw data conversion (V4 format)"""

    config = KittiConverter4Config(**{**vars(ctx.obj), **kwargs})

    KittiConverter4.convert(config)
