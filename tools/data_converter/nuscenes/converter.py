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

"""nuScenes dataset to NCore V4 converter."""

from __future__ import annotations

import json
import logging

from dataclasses import dataclass, replace
from typing import Dict, List, Literal, Optional

import click
import numpy as np
import tqdm

from nuscenes.utils.data_classes import RadarPointCloud
from pyquaternion import Quaternion
from upath import UPath

from ncore.impl.common.transformations import HalfClosedInterval, MotionCompensator, se3_inverse
from ncore.impl.data.types import (
    BBox3,
    CuboidTrackObservation,
    IdealPinholeCameraModelParameters,
    LabelSource,
    RowOffsetStructuredSpinningLidarModelParameters,
    ShutterType,
)
from ncore.impl.data.v4.components import (
    CameraSensorComponent,
    CuboidsComponent,
    IntrinsicsComponent,
    LidarSensorComponent,
    MasksComponent,
    PosesComponent,
    RadarSensorComponent,
    SequenceComponentGroupsReader,
    SequenceComponentGroupsWriter,
)
from ncore.impl.data.v4.types import ComponentGroupAssignments
from ncore.impl.data_converter.base import FileBasedDataConverter, FileBasedDataConverterConfig
from tools.data_converter.cli import cli
from tools.data_converter.nuscenes.utils import (
    CAMERA_MAP,
    LIDAR_CHANNEL,
    LIDAR_ID,
    NUSCENES_CATEGORY_MAP,
    RADAR_MAP,
    get_boxes_for_sample_data,
    get_nuscenes,
    get_sweep_tokens,
    resolve_scene_token,
)
from tools.data_converter.structured_lidar_model import (
    HDL32E_FIRING_PAIR_INTERVAL_US,
    HDL32E_N_BEAMS,
    HDL32E_N_COLUMNS,
    HDL32E_SCAN_DURATION_US,
    AlignedFrameData,
    align_frame,
    compute_frame_timestamps,
    compute_model_consistency,
    derive_model_from_decompensated,
    derive_nominal_hdl32e,
    optimize_model,
    upsample_model,
)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass(kw_only=True, slots=True)
class NuScenesConverter4Config(FileBasedDataConverterConfig):
    """Configuration for nuScenes to NCore V4 conversion."""

    version: str = "v1.0-trainval"
    scene_token: Optional[str] = None
    scene_name: Optional[str] = None
    store_type: Literal["itar", "directory"] = "itar"
    component_group_profile: Literal["default", "separate-sensors", "separate-all"] = "separate-sensors"
    store_sequence_meta: bool = True
    # Defaults below match the CLI options (see nuscenes_v4) and the README so a
    # programmatically-constructed config behaves the same as the command line.
    lidar_model_optimization_passes: int = 1
    lidar_model_source: Literal["empirical", "nominal"] = "empirical"
    lidar_model_resolution: int = 4


# -----------------------------------------------------------------------------
# Converter
# -----------------------------------------------------------------------------


class NuScenesConverter4(FileBasedDataConverter):
    """Dataset preprocessing class for converting nuScenes data to NCore V4 format.

    Supports nuScenes versions: v1.0-mini, v1.0-trainval, v1.0-test.

    Sensor assumptions:
    - Cameras: Treated as global shutter (ShutterType.GLOBAL). nuScenes provides a
      single capture timestamp per image with no rolling-shutter metadata. Images are
      already undistorted.
    - Lidar: Velodyne HDL-32E spinning lidar (20 Hz, ~50ms scan duration). Per-point
      timestamps are derived from the 32-beam column structure in the .bin file.
      Source data is motion-compensated; we decompensate to raw measurements before
      storing. Structured lidar model parameters (elevation/azimuth per beam/column)
      are derived from the first frame's geometry.
    - Cuboid annotations: Stored in the world coordinate frame. For non-keyframe
      sweeps, box positions are linearly interpolated between bracketing keyframes.
    """

    def __init__(self, config: NuScenesConverter4Config) -> None:
        super().__init__(config)

        self.component_group_profile = config.component_group_profile
        self.store_type = config.store_type
        self.store_sequence_meta = config.store_sequence_meta
        self._lidar_model_optimization_passes = config.lidar_model_optimization_passes
        self._lidar_model_source = config.lidar_model_source
        self._lidar_model_resolution = config.lidar_model_resolution

        self._version = config.version
        self._scene_token = config.scene_token
        self._scene_name = config.scene_name

        self.logger = logging.getLogger(__name__)

    @staticmethod
    def get_sequence_ids(config: NuScenesConverter4Config) -> List[str]:
        """Discover scene tokens to convert."""
        nusc = get_nuscenes(version=config.version, dataroot=config.root_dir)

        resolved = resolve_scene_token(nusc, config.scene_token, config.scene_name)
        if resolved is not None:
            return [resolved]

        # All scenes
        return [s["token"] for s in nusc.scene]

    @staticmethod
    def from_config(config: NuScenesConverter4Config) -> NuScenesConverter4:
        return NuScenesConverter4(config)

    def convert_sequence(self, sequence_id: str) -> None:
        """Convert a single nuScenes scene to NCore V4 format."""
        scene_token = sequence_id
        nusc = get_nuscenes(version=self._version, dataroot=str(self.root_dir))
        scene_record = nusc.get("scene", scene_token)
        scene_name = scene_record["name"]

        self.logger.info(f"Converting scene {scene_name} ({scene_token})")

        # Use scene name as output directory (more readable than token)
        sequence_output_name = scene_name

        # --- Gather lidar sweep timeline (used as pose timeline) ---------------
        lidar_sweep_tokens = get_sweep_tokens(nusc, scene_record, LIDAR_CHANNEL)
        lidar_sweep_data = [nusc.get("sample_data", t) for t in lidar_sweep_tokens]
        lidar_timestamps_us = np.array([sd["timestamp"] for sd in lidar_sweep_data], dtype=np.uint64)

        n_lidar_frames = len(lidar_sweep_tokens)
        assert n_lidar_frames >= 2, f"Scene has fewer than 2 lidar sweeps: {n_lidar_frames}"

        # --- Decode ego poses from lidar sweep ego_pose records ----------------
        T_rig_world_list: List[np.ndarray] = []
        for sd in lidar_sweep_data:
            ego_pose = nusc.get("ego_pose", sd["ego_pose_token"])
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = Quaternion(ego_pose["rotation"]).rotation_matrix
            T[:3, 3] = ego_pose["translation"]
            T_rig_world_list.append(T)

        T_rig_world_all = np.stack(T_rig_world_list)  # [N, 4, 4] float64 (global coords)
        pose_timestamps_us = lidar_timestamps_us.copy()

        # Store first pose as the world_global anchor (high precision for global coordinates),
        # then make all poses relative to it (local coords -> float32 sufficient).
        T_world_world_global = T_rig_world_all[0].copy()  # float64 for global accuracy
        T_world_global_inv = se3_inverse(T_world_world_global)
        T_rig_world_relative = (T_world_global_inv @ T_rig_world_all).astype(np.float32)

        # --- Determine active sensors ------------------------------------------
        camera_ids = self.get_active_camera_ids(list(CAMERA_MAP.keys()))
        lidar_ids = self.get_active_lidar_ids([LIDAR_ID])
        radar_ids = self.get_active_radar_ids(list(RADAR_MAP.keys()))

        # --- Compute sequence time interval ------------------------------------
        # Per-point timestamps span [prev_sweep_ts, current_sweep_ts] for each frame.
        # For the first frame, extrapolate backward using the gap to the next frame.
        if len(lidar_timestamps_us) >= 2:
            first_gap = int(lidar_timestamps_us[1]) - int(lidar_timestamps_us[0])
        else:
            first_gap = HDL32E_SCAN_DURATION_US
        seq_start_us = int(lidar_timestamps_us[0]) - first_gap
        seq_end_us = int(lidar_timestamps_us[-1])

        # Also include camera and radar timestamps for full coverage
        for ncore_cam_id, nusc_channel in CAMERA_MAP.items():
            if ncore_cam_id not in camera_ids:
                continue
            cam_tokens = get_sweep_tokens(nusc, scene_record, nusc_channel)
            if cam_tokens:
                cam_data = [nusc.get("sample_data", t) for t in cam_tokens]
                cam_ts = [sd["timestamp"] for sd in cam_data]
                seq_start_us = min(seq_start_us, min(cam_ts))
                seq_end_us = max(seq_end_us, max(cam_ts))

        for ncore_radar_id, nusc_channel in RADAR_MAP.items():
            if ncore_radar_id not in radar_ids:
                continue
            radar_tokens = get_sweep_tokens(nusc, scene_record, nusc_channel)
            if radar_tokens:
                radar_data = [nusc.get("sample_data", t) for t in radar_tokens]
                radar_ts = [sd["timestamp"] for sd in radar_data]
                seq_start_us = min(seq_start_us, min(radar_ts))
                seq_end_us = max(seq_end_us, max(radar_ts))

        sequence_timestamp_interval_us = HalfClosedInterval.from_start_end(seq_start_us, seq_end_us)

        # Extend pose timestamps to cover sequence interval boundaries.
        # For the start boundary, extrapolate backward using the motion between
        # the first two poses (constant-velocity assumption). This is critical
        # for the first lidar frame's decompensation -- if we just replicate the
        # first pose, the decompensator sees zero motion and produces no correction.
        if seq_start_us < int(pose_timestamps_us[0]):
            if len(T_rig_world_relative) >= 2:
                # Extrapolate: apply the inverse of the motion from 0->1 to get the pose before 0.
                T_0 = T_rig_world_relative[0]
                T_1 = T_rig_world_relative[1]
                T_delta_inv = se3_inverse(T_1) @ T_0
                T_boundary = (T_0 @ T_delta_inv).astype(np.float32)
                T_rig_world_relative = np.concatenate([T_boundary[np.newaxis], T_rig_world_relative], axis=0)
            else:
                T_rig_world_relative = np.concatenate([T_rig_world_relative[:1], T_rig_world_relative], axis=0)
            pose_timestamps_us = np.concatenate([np.array([seq_start_us], dtype=np.uint64), pose_timestamps_us])

        if seq_end_us > int(pose_timestamps_us[-1]):
            if len(T_rig_world_relative) >= 2:
                # Extrapolate forward using constant-velocity from last two poses.
                T_n1 = T_rig_world_relative[-2]
                T_n = T_rig_world_relative[-1]
                T_delta = se3_inverse(T_n1) @ T_n
                T_boundary = (T_n @ T_delta).astype(np.float32)
                T_rig_world_relative = np.concatenate([T_rig_world_relative, T_boundary[np.newaxis]], axis=0)
            else:
                T_rig_world_relative = np.concatenate([T_rig_world_relative, T_rig_world_relative[-1:]], axis=0)
            pose_timestamps_us = np.concatenate([pose_timestamps_us, np.array([seq_end_us], dtype=np.uint64)])

        # --- Component group assignments --------------------------------------
        component_groups = ComponentGroupAssignments.create(
            camera_ids=camera_ids,
            lidar_ids=lidar_ids,
            radar_ids=radar_ids,
            point_clouds_ids=[],
            camera_labels_ids=[],
            profile=self.component_group_profile,
        )

        # --- Create writer ----------------------------------------------------
        store_writer = SequenceComponentGroupsWriter(
            output_dir_path=self.output_dir / sequence_output_name,
            store_base_name=sequence_output_name,
            sequence_id=sequence_output_name,
            sequence_timestamp_interval_us=sequence_timestamp_interval_us,
            store_type=self.store_type,
            generic_meta_data={
                "source_dataset": "nuscenes",
                "nuscenes_version": self._version,
                "nuscenes_scene_token": scene_token,
                "nuscenes_scene_name": scene_name,
            },
        )

        # --- Register component writers ---------------------------------------
        poses_writer = store_writer.register_component_writer(
            PosesComponent.Writer,
            component_instance_name="default",
            group_name=component_groups.poses_component_group,
            generic_meta_data={
                "calibration_type": "nuscenes:calibrated_sensor",
                "egomotion_type": "nuscenes:ego_pose",
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

        # --- Store ego poses --------------------------------------------------
        poses_writer.store_dynamic_pose(
            source_frame_id="rig",
            target_frame_id="world",
            poses=T_rig_world_relative,
            timestamps_us=pose_timestamps_us,
        )

        poses_writer.store_static_pose(
            source_frame_id="world",
            target_frame_id="world_global",
            pose=T_world_world_global,
        )

        # --- Decode lidar -----------------------------------------------------
        if LIDAR_ID in lidar_ids:
            self._decode_lidars(
                nusc=nusc,
                store_writer=store_writer,
                poses_writer=poses_writer,
                intrinsics_writer=intrinsics_writer,
                component_groups=component_groups,
                lidar_sweep_tokens=lidar_sweep_tokens,
                lidar_sweep_data=lidar_sweep_data,
                T_rig_world_relative=T_rig_world_relative,
                pose_timestamps_us=pose_timestamps_us,
            )

        # --- Decode cameras ---------------------------------------------------
        self._decode_cameras(
            nusc=nusc,
            scene_record=scene_record,
            store_writer=store_writer,
            poses_writer=poses_writer,
            intrinsics_writer=intrinsics_writer,
            masks_writer=masks_writer,
            component_groups=component_groups,
            camera_ids=camera_ids,
        )

        # --- Decode radars ----------------------------------------------------
        self._decode_radars(
            nusc=nusc,
            scene_record=scene_record,
            store_writer=store_writer,
            poses_writer=poses_writer,
            component_groups=component_groups,
            radar_ids=radar_ids,
        )

        # --- Decode cuboid annotations ----------------------------------------
        self._decode_cuboids(
            nusc=nusc,
            store_writer=store_writer,
            component_groups=component_groups,
            lidar_sweep_tokens=lidar_sweep_tokens,
            lidar_sweep_data=lidar_sweep_data,
        )

        # --- Finalize ---------------------------------------------------------
        ncore_4_paths = store_writer.finalize()

        if self.store_sequence_meta:
            sequence_component_reader = SequenceComponentGroupsReader(ncore_4_paths)
            sequence_meta_path = (
                self.output_dir / sequence_output_name / f"{sequence_component_reader.sequence_id}.json"
            )
            with sequence_meta_path.open("w") as f:
                json.dump(sequence_component_reader.get_sequence_meta().to_dict(), f, indent=2)
            self.logger.info(f"Wrote sequence meta data {str(sequence_meta_path)}")

    # -------------------------------------------------------------------------
    # Lidar
    # -------------------------------------------------------------------------

    def _decode_lidars(
        self,
        nusc,
        store_writer: SequenceComponentGroupsWriter,
        poses_writer: PosesComponent.Writer,
        intrinsics_writer: IntrinsicsComponent.Writer,
        component_groups: ComponentGroupAssignments,
        lidar_sweep_tokens: List[str],
        lidar_sweep_data: List[Dict],
        T_rig_world_relative: np.ndarray,
        pose_timestamps_us: np.ndarray,
    ) -> None:
        """Decode and store all lidar frames.

        nuScenes point clouds are motion-compensated to the sensor frame at the sweep's
        reference timestamp. We decompensate them back to per-point-time sensor frames
        before storing, since NCore V4 expects raw (non-motion-compensated) measurements.

        Per-point timestamps are derived from the model column index (constant angular
        velocity: one full rotation per frame). A structured lidar model with per-row
        azimuth offsets is extracted from a reference frame and used for column alignment.
        """
        # Get extrinsic from calibrated_sensor (lidar -> rig)
        calibrated_sensor = nusc.get("calibrated_sensor", lidar_sweep_data[0]["calibrated_sensor_token"])
        T_lidar_rig = np.eye(4, dtype=np.float32)
        T_lidar_rig[:3, :3] = Quaternion(calibrated_sensor["rotation"]).rotation_matrix
        T_lidar_rig[:3, 3] = calibrated_sensor["translation"]

        # Store static extrinsic pose (lidar -> rig)
        poses_writer.store_static_pose(source_frame_id=LIDAR_ID, target_frame_id="rig", pose=T_lidar_rig)

        # Initialize motion compensator for decompensation
        motion_compensator = MotionCompensator.from_sensor_rig(
            sensor_id=LIDAR_ID,
            T_sensor_rig=T_lidar_rig,
            T_rig_worlds=T_rig_world_relative,
            T_rig_worlds_timestamps_us=pose_timestamps_us,
        )

        # Register lidar component writer
        lidar_writer = store_writer.register_component_writer(
            LidarSensorComponent.Writer,
            component_instance_name=LIDAR_ID,
            group_name=component_groups.lidar_component_groups.get(LIDAR_ID),
            generic_meta_data={"sensor_model": "Velodyne HDL-32E"},
        )

        # Precompute frame start timestamps from consecutive sweep times.
        frame_start_timestamps = []
        for i in range(len(lidar_sweep_data)):
            if i == 0:
                if len(lidar_sweep_data) >= 2:
                    gap = int(lidar_sweep_data[1]["timestamp"]) - int(lidar_sweep_data[0]["timestamp"])
                else:
                    gap = HDL32E_SCAN_DURATION_US
                frame_start_timestamps.append(int(lidar_sweep_data[0]["timestamp"]) - gap)
            else:
                frame_start_timestamps.append(int(lidar_sweep_data[i - 1]["timestamp"]))

        # --- Derive lidar model from a "good" frame ---
        # Find the first frame with the target column count, decompensate it, and
        # extract the model from the decompensated azimuths.
        lidar_model_parameters: RowOffsetStructuredSpinningLidarModelParameters | None = None

        if self._lidar_model_source == "nominal":
            # Use nominal model from HDL-32E spec (no data dependency for model derivation).
            # Still need one frame to determine spinning frequency and starting azimuth.
            for j, sd_j in enumerate(lidar_sweep_data):
                scan_j = np.fromfile(str(UPath(str(self.root_dir)) / sd_j["filename"]), dtype=np.float32).reshape(-1, 5)
                xyz_j = scan_j[:, :3].astype(np.float32)
                if len(xyz_j) // HDL32E_N_BEAMS != HDL32E_N_COLUMNS:
                    continue
                frame_end_j = int(sd_j["timestamp"])
                frame_start_j = frame_start_timestamps[j]
                frame_duration_s = (frame_end_j - frame_start_j) / 1e6
                freq_hz = 1.0 / frame_duration_s if frame_duration_s > 0 else 20.0

                # Determine starting azimuth from far-range points in the MC'd data.
                # This is approximate (MC'd azimuths) but good enough for alignment
                # to lock onto the correct column phase within +/- 20 columns.
                n_pts_j = len(xyz_j)
                dist_j = np.linalg.norm(xyz_j, axis=1)
                az_j = np.arctan2(xyz_j[:, 1], xyz_j[:, 0])
                col_idx_j = np.arange(n_pts_j) // HDL32E_N_BEAMS
                # Median azimuth of far-range points in the first few columns
                far_mask = dist_j > 20.0
                first_cols = col_idx_j < 10
                start_az = float(np.median(az_j[far_mask & first_cols])) if (far_mask & first_cols).any() else 0.0

                lidar_model_parameters = derive_nominal_hdl32e(
                    spinning_frequency_hz=freq_hz,
                    start_azimuth_rad=start_az,
                )
                self.logger.info(
                    f"Using nominal HDL-32E model (freq={freq_hz:.2f} Hz, start_az={np.degrees(start_az):.1f} deg)"
                )
                break
        else:
            # Empirical: derive from decompensated reference frame
            for j, sd_j in enumerate(lidar_sweep_data):
                scan_j = np.fromfile(str(UPath(str(self.root_dir)) / sd_j["filename"]), dtype=np.float32).reshape(-1, 5)
                xyz_j = scan_j[:, :3].astype(np.float32)

                if len(xyz_j) // HDL32E_N_BEAMS != HDL32E_N_COLUMNS:
                    continue

                frame_end_j = int(sd_j["timestamp"])
                frame_start_j = frame_start_timestamps[j]
                frame_duration_s = (frame_end_j - frame_start_j) / 1e6
                freq_hz = 1.0 / frame_duration_s if frame_duration_s > 0 else 20.0

                # Decompensate using column-index timestamps
                n_pts_j = len(xyz_j)
                col_idx_j = np.arange(n_pts_j) // HDL32E_N_BEAMS
                ts_j = compute_frame_timestamps(col_idx_j, HDL32E_N_COLUMNS, frame_start_j, frame_end_j)

                xyz_decomp_j = motion_compensator.motion_decompensate_points(
                    sensor_id=LIDAR_ID,
                    xyz_reftime=xyz_j,
                    timestamp_us=ts_j,
                    frame_start_timestamp_us=frame_start_j,
                    frame_end_timestamp_us=frame_end_j,
                )

                lidar_model_parameters = derive_model_from_decompensated(
                    xyz_decompensated=xyz_decomp_j,
                    n_beams_per_column=HDL32E_N_BEAMS,
                    n_target_cols=HDL32E_N_COLUMNS,
                    spinning_direction="cw",
                    spinning_frequency_hz=freq_hz,
                    beam_pair_interval_us=HDL32E_FIRING_PAIR_INTERVAL_US,
                )
                if lidar_model_parameters is not None:
                    self.logger.info(f"Derived lidar model from frame {j} (n_cols={lidar_model_parameters.n_columns})")
                    break

        assert lidar_model_parameters is not None, (
            f"Failed to derive lidar model: no frame with {HDL32E_N_COLUMNS} columns found"
        )

        # Upsample model for sub-column alignment precision (applies to both paths)
        if self._lidar_model_resolution > 1:
            lidar_model_parameters = upsample_model(lidar_model_parameters, self._lidar_model_resolution)
            self.logger.info(
                f"Upsampled model to {self._lidar_model_resolution}x ({lidar_model_parameters.n_columns} columns)"
            )

        n_model_cols = lidar_model_parameters.n_columns

        # --- Process each frame ---
        optimization_data: List[AlignedFrameData] = []

        for i, (_, sd) in enumerate(
            tqdm.tqdm(
                zip(lidar_sweep_tokens, lidar_sweep_data),
                total=len(lidar_sweep_tokens),
                desc=f"Process {LIDAR_ID}",
            )
        ):
            source_pc_path = UPath(str(self.root_dir)) / sd["filename"]
            scan = np.fromfile(str(source_pc_path), dtype=np.float32).reshape(-1, 5)
            xyz_mc = scan[:, :3].astype(np.float32)
            raw_intensity = scan[:, 3]
            ring_index = scan[:, 4].astype(np.uint16)
            intensity = (raw_intensity / 255.0).astype(np.float32)

            frame_end_us = int(sd["timestamp"])
            frame_start_us = frame_start_timestamps[i]

            # Align and decompensate using modular pipeline
            frame_data = align_frame(
                xyz_mc=xyz_mc,
                ring_index=ring_index,
                intensity=intensity,
                n_beams_per_column=HDL32E_N_BEAMS,
                model_params=lidar_model_parameters,
                motion_compensator=motion_compensator,
                sensor_id=LIDAR_ID,
                frame_start_us=frame_start_us,
                frame_end_us=frame_end_us,
                model_resolution_factor=self._lidar_model_resolution,
            )

            if frame_data is None:
                continue

            # Collect data for optional multi-frame optimization
            if self._lidar_model_optimization_passes > 0:
                optimization_data.append(frame_data)

            # Compute direction and distance from decompensated points
            distance_m = np.linalg.norm(frame_data.xyz_decompensated, axis=1).astype(np.float32)
            direction = np.zeros_like(frame_data.xyz_decompensated)
            nonzero_mask = distance_m > 0
            direction[nonzero_mask] = frame_data.xyz_decompensated[nonzero_mask] / distance_m[nonzero_mask, np.newaxis]

            lidar_writer.store_frame(
                direction=direction,
                timestamp_us=frame_data.timestamps_us,
                model_element=frame_data.model_element,
                distance_m=distance_m.reshape(1, -1),
                intensity=frame_data.intensity.reshape(1, -1),
                frame_timestamps_us=np.array([frame_start_us, frame_end_us], dtype=np.uint64),
                generic_data={},
                generic_meta_data={},
            )

        # --- Optional: multi-frame model optimization ---
        if self._lidar_model_optimization_passes > 0 and optimization_data:
            frame_azimuths = []
            frame_model_cols = []
            frame_model_rows = []
            frame_distances = []

            for fd in optimization_data:
                az = np.arctan2(fd.xyz_decompensated[:, 1], fd.xyz_decompensated[:, 0]).astype(np.float64)
                frame_azimuths.append(az)
                frame_model_cols.append(fd.model_element[:, 1].astype(np.int64))
                frame_model_rows.append(fd.model_element[:, 0].astype(np.int64))
                frame_distances.append(np.linalg.norm(fd.xyz_decompensated, axis=1))

            lidar_model_parameters = optimize_model(
                model_params=lidar_model_parameters,
                frame_azimuths=frame_azimuths,
                frame_model_cols=frame_model_cols,
                frame_model_rows=frame_model_rows,
                frame_distances=frame_distances,
                min_range_m=10.0,
                n_iterations=self._lidar_model_optimization_passes,
            )
            self.logger.info(
                f"Optimized lidar model across {len(optimization_data)} frames "
                f"({self._lidar_model_optimization_passes} iterations)"
            )

            # Log model consistency metrics from a representative frame
            sample_fd = optimization_data[len(optimization_data) // 2]
            sample_xyz = sample_fd.xyz_decompensated
            sample_dist = np.linalg.norm(sample_xyz, axis=1)
            nonzero = sample_dist > 0
            directions = np.zeros_like(sample_xyz)
            directions[nonzero] = sample_xyz[nonzero] / sample_dist[nonzero, np.newaxis]
            err_all, err_far, az_shift = compute_model_consistency(
                directions, sample_fd.model_element, sample_dist, lidar_model_parameters
            )
            self.logger.info(
                f"Lidar model consistency (mid-frame): "
                f"{err_all:.3f} deg all, {err_far:.3f} deg far, "
                f"{az_shift:.4f} deg systematic az shift"
            )

        # Store lidar intrinsics (structured model, possibly optimized)
        intrinsics_writer.store_lidar_intrinsics(
            lidar_id=LIDAR_ID,
            lidar_model_parameters=lidar_model_parameters,
        )

    # -------------------------------------------------------------------------
    # Cameras
    # -------------------------------------------------------------------------

    def _decode_cameras(
        self,
        nusc,
        scene_record: Dict,
        store_writer: SequenceComponentGroupsWriter,
        poses_writer: PosesComponent.Writer,
        intrinsics_writer: IntrinsicsComponent.Writer,
        masks_writer: MasksComponent.Writer,
        component_groups: ComponentGroupAssignments,
        camera_ids: List[str],
    ) -> None:
        """Decode and store all camera frames."""
        for ncore_cam_id, nusc_channel in CAMERA_MAP.items():
            if ncore_cam_id not in camera_ids:
                continue

            self.logger.info(f"Processing camera {ncore_cam_id} ({nusc_channel})")

            cam_sweep_tokens = get_sweep_tokens(nusc, scene_record, nusc_channel)
            cam_sweep_data = [nusc.get("sample_data", t) for t in cam_sweep_tokens]

            if not cam_sweep_data:
                self.logger.warning(f"No data for camera {nusc_channel}")
                continue

            # Get calibration from first sweep
            calibrated_sensor = nusc.get("calibrated_sensor", cam_sweep_data[0]["calibrated_sensor_token"])

            # Camera extrinsic: sensor -> rig
            T_cam_rig = np.eye(4, dtype=np.float32)
            T_cam_rig[:3, :3] = Quaternion(calibrated_sensor["rotation"]).rotation_matrix
            T_cam_rig[:3, 3] = calibrated_sensor["translation"]

            # Store camera extrinsic
            poses_writer.store_static_pose(
                source_frame_id=ncore_cam_id,
                target_frame_id="rig",
                pose=T_cam_rig,
            )

            # Parse intrinsics
            I_cam = np.array(calibrated_sensor["camera_intrinsic"], dtype=np.float32)  # [3, 3]
            width = int(cam_sweep_data[0]["width"])
            height = int(cam_sweep_data[0]["height"])

            # Store camera intrinsics
            # nuScenes images are undistorted, so an ideal (distortion-free) pinhole is used.
            # ShutterType.GLOBAL: nuScenes provides a single capture timestamp per image
            # with no rolling-shutter metadata available.
            intrinsics_writer.store_camera_intrinsics(
                camera_id=ncore_cam_id,
                camera_model_parameters=IdealPinholeCameraModelParameters(
                    resolution=np.array([width, height], dtype=np.uint64),
                    shutter_type=ShutterType.GLOBAL,
                    external_distortion_parameters=None,
                    principal_point=np.array([I_cam[0, 2], I_cam[1, 2]], dtype=np.float32),
                    focal_length=np.array([I_cam[0, 0], I_cam[1, 1]], dtype=np.float32),
                ),
            )

            # Store empty masks
            masks_writer.store_camera_masks(
                camera_id=ncore_cam_id,
                mask_images={},
            )

            # Register camera component writer
            camera_writer = store_writer.register_component_writer(
                CameraSensorComponent.Writer,
                component_instance_name=ncore_cam_id,
                group_name=component_groups.camera_component_groups.get(ncore_cam_id),
                generic_meta_data={},
            )

            # Store frames
            for sd in tqdm.tqdm(cam_sweep_data, desc=f"Process {ncore_cam_id}"):
                image_path = UPath(str(self.root_dir)) / sd["filename"]

                with image_path.open("rb") as f:
                    image_binary = f.read()

                # Global shutter: frame start == frame end timestamp
                frame_ts = int(sd["timestamp"])

                camera_writer.store_frame(
                    image_binary_data=image_binary,
                    image_format="jpeg",
                    frame_timestamps_us=np.array([frame_ts, frame_ts], dtype=np.uint64),
                    generic_data={},
                    generic_meta_data={},
                )

        self.logger.info(f"Processed {len(camera_ids)} cameras")

    # -------------------------------------------------------------------------
    # Radars
    # -------------------------------------------------------------------------

    def _decode_radars(
        self,
        nusc,
        scene_record: Dict,
        store_writer: SequenceComponentGroupsWriter,
        poses_writer: PosesComponent.Writer,
        component_groups: ComponentGroupAssignments,
        radar_ids: List[str],
    ) -> None:
        """Decode and store all radar frames.

        nuScenes radars (Continental ARS 408) provide sparse detections with
        Cartesian position (x, y, z), ego-motion-compensated velocity (vx_comp,
        vy_comp), and radar cross section (rcs). Data is stored in .pcd files
        with 18 fields per detection.

        We compute radial velocity by projecting the compensated velocity vector
        onto the detection direction (positive = moving away from sensor).
        """

        for ncore_radar_id, nusc_channel in RADAR_MAP.items():
            if ncore_radar_id not in radar_ids:
                continue

            self.logger.info(f"Processing radar {ncore_radar_id} ({nusc_channel})")

            radar_sweep_tokens = get_sweep_tokens(nusc, scene_record, nusc_channel)
            radar_sweep_data = [nusc.get("sample_data", t) for t in radar_sweep_tokens]

            if not radar_sweep_data:
                self.logger.warning(f"No data for radar {nusc_channel}")
                continue

            # Get calibration (radar -> rig extrinsic)
            calibrated_sensor = nusc.get("calibrated_sensor", radar_sweep_data[0]["calibrated_sensor_token"])
            T_radar_rig = np.eye(4, dtype=np.float32)
            T_radar_rig[:3, :3] = Quaternion(calibrated_sensor["rotation"]).rotation_matrix
            T_radar_rig[:3, 3] = calibrated_sensor["translation"]

            # Store radar extrinsic
            poses_writer.store_static_pose(
                source_frame_id=ncore_radar_id,
                target_frame_id="rig",
                pose=T_radar_rig,
            )

            # Register radar component writer
            radar_writer = store_writer.register_component_writer(
                RadarSensorComponent.Writer,
                component_instance_name=ncore_radar_id,
                group_name=component_groups.radar_component_groups.get(ncore_radar_id),
                generic_meta_data={
                    "sensor_model": "Continental ARS 408",
                },
            )

            # Store frames
            for sd in tqdm.tqdm(radar_sweep_data, desc=f"Process {ncore_radar_id}"):
                radar_path = UPath(str(self.root_dir)) / sd["filename"]

                # Load radar point cloud (18 fields)
                pc = RadarPointCloud.from_file(str(radar_path))
                pts = pc.points.T  # [N, 18]

                if len(pts) == 0:
                    continue

                # Extract fields
                xyz = pts[:, :3].astype(np.float32)  # x, y, z in sensor frame
                rcs = pts[:, 5].astype(np.float32)  # radar cross section (dBsm)
                vx_comp = pts[:, 8].astype(np.float32)  # ego-motion-compensated velocity x
                vy_comp = pts[:, 9].astype(np.float32)  # ego-motion-compensated velocity y

                # Compute distance and direction
                distance = np.linalg.norm(xyz, axis=1).astype(np.float32)
                valid_mask = distance > 0.1  # filter degenerate detections

                if not valid_mask.any():
                    continue

                xyz = xyz[valid_mask]
                distance = distance[valid_mask]
                rcs = rcs[valid_mask]
                vx_comp = vx_comp[valid_mask]
                vy_comp = vy_comp[valid_mask]

                direction = (xyz / distance[:, np.newaxis]).astype(np.float32)

                # Compute radial velocity: project compensated velocity onto direction
                # Positive = moving away from sensor
                velocity_vec = np.stack([vx_comp, vy_comp, np.zeros_like(vx_comp)], axis=1)
                radial_velocity = np.sum(velocity_vec * direction, axis=1).astype(np.float32)

                # Radar is not a spinning sensor -- all detections share one timestamp
                frame_ts = int(sd["timestamp"])
                timestamp_us = np.full(len(xyz), frame_ts, dtype=np.uint64)

                radar_writer.store_frame(
                    direction=direction,
                    timestamp_us=timestamp_us,
                    distance_m=distance.reshape(1, -1),  # [1, N] single return
                    frame_timestamps_us=np.array([frame_ts, frame_ts], dtype=np.uint64),
                    generic_data={
                        "radial_velocity_m_s": radial_velocity,
                        "rcs_dBsm": rcs,
                    },
                    generic_meta_data={},
                )

        self.logger.info(f"Processed {len(radar_ids)} radars")

    # -------------------------------------------------------------------------
    # Cuboid annotations
    # -------------------------------------------------------------------------

    def _decode_cuboids(
        self,
        nusc,
        store_writer: SequenceComponentGroupsWriter,
        component_groups: ComponentGroupAssignments,
        lidar_sweep_tokens: List[str],
        lidar_sweep_data: List[Dict],
    ) -> None:
        """Decode nuScenes 3D annotations and store as cuboid track observations.

        Annotations are stored in the world coordinate frame. For non-keyframe sweeps,
        box positions are interpolated between bracketing keyframes.
        """
        cuboid_observations: List[CuboidTrackObservation] = []

        for token, sd in tqdm.tqdm(
            zip(lidar_sweep_tokens, lidar_sweep_data),
            total=len(lidar_sweep_tokens),
            desc="Process cuboids",
        ):
            # Only process keyframes (annotations are defined at keyframes)
            if not sd["is_key_frame"]:
                continue

            boxes = get_boxes_for_sample_data(nusc, token)
            timestamp_us = int(sd["timestamp"])

            for box in boxes:
                # Filter to mapped categories only
                if box.name not in NUSCENES_CATEGORY_MAP:
                    continue

                class_id = NUSCENES_CATEGORY_MAP[box.name]

                # nuScenes Box: center=[x,y,z] in global frame, wlh=[width, length, height]
                # BBox3 format: [cx, cy, cz, size_x, size_y, size_z, rx, ry, rz]
                # nuScenes wlh order is [width, length, height]
                # Heading: extract yaw from quaternion
                yaw = Quaternion(box.orientation).yaw_pitch_roll[0]

                bbox3 = BBox3.from_array(
                    np.array(
                        [
                            box.center[0],
                            box.center[1],
                            box.center[2],
                            box.wlh[1],  # length -> size_x
                            box.wlh[0],  # width -> size_y
                            box.wlh[2],  # height -> size_z
                            0.0,  # rx (pitch) -- only yaw used
                            0.0,  # ry (roll) -- only yaw used
                            yaw,  # rz (yaw)
                        ],
                        dtype=np.float32,
                    )
                )

                cuboid_observations.append(
                    CuboidTrackObservation(
                        track_id=box.token,  # instance_token as track ID
                        class_id=class_id,
                        timestamp_us=timestamp_us,
                        reference_frame_id="world_global",
                        reference_frame_timestamp_us=timestamp_us,
                        bbox3=bbox3,
                        source=LabelSource.EXTERNAL,
                    )
                )

        if cuboid_observations:
            store_writer.register_component_writer(
                CuboidsComponent.Writer,
                "default",
                component_groups.cuboid_track_observations_component_group,
            ).store_observations(cuboid_observations)

            self.logger.info(f"Stored {len(cuboid_observations)} cuboid observations")
        else:
            self.logger.info("No cuboid annotations found (test split or empty scenes)")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


@cli.command(name="nuscenes-v4")
@click.option(
    "--version",
    "nuscenes_version",
    type=str,
    default="v1.0-trainval",
    show_default=True,
    help="nuScenes dataset version (v1.0-mini, v1.0-trainval, v1.0-test)",
)
@click.option(
    "--scene-token",
    type=str,
    default=None,
    help="Convert only the scene with this token (mutually exclusive with --scene-name)",
)
@click.option(
    "--scene-name",
    type=str,
    default=None,
    help="Convert only the scene with this name, e.g. 'scene-0001' (mutually exclusive with --scene-token)",
)
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
    help="Output profile for component group assignment",
)
@click.option(
    "store_sequence_meta",
    "--sequence-meta/--no-sequence-meta",
    default=True,
    help="Generate sequence meta-data JSON?",
)
@click.option(
    "lidar_model_optimization_passes",
    "--lidar-model-optimization-passes",
    type=int,
    default=1,
    show_default=True,
    help="Number of multi-frame optimization passes for the lidar model (0 to disable).",
)
@click.option(
    "lidar_model_source",
    "--lidar-model-source",
    type=click.Choice(["empirical", "nominal"], case_sensitive=False),
    default="empirical",
    show_default=True,
    help="Model derivation source: 'empirical' derives from data, 'nominal' uses HDL-32E spec values.",
)
@click.option(
    "lidar_model_resolution",
    "--lidar-model-resolution",
    type=int,
    default=4,
    show_default=True,
    help="Model column resolution factor (1=native, 2=2x, 4=4x). Higher gives sub-column alignment precision.",
)
@click.pass_context
def nuscenes_v4(ctx, nuscenes_version, scene_token, scene_name, **kwargs):
    """nuScenes data conversion (V4 format)"""

    config = NuScenesConverter4Config(
        **{**vars(ctx.obj), "version": nuscenes_version, "scene_token": scene_token, "scene_name": scene_name, **kwargs}
    )

    NuScenesConverter4.convert(config)
