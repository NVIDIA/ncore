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

"""
Physical AI to NCore v4 Data Converter

Converts Physical AI autonomous driving dataset to NCore canonical format.

FEATURES:
  - Egomotion poses from sensor data
  - 7 cameras (120° FOV front/cross, 70° FOV rear, 30° FOV tele)
  - 1 top lidar (360° FOV, DRACO compressed)
  - Up to 19 radar sensors (spherical coordinates, per-detection Doppler/RCS/SNR)
  - Cuboid obstacle labels

LIMITATIONS:
  - Lidar uses DRACO compression (requires DracoPy library)
  - Cameras are in MP4 format (frames extracted to JPEG)
"""

from __future__ import annotations

import json
import logging
import tempfile

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Literal, Optional, cast

import click
import DracoPy
import imageio
import numpy as np
import tqdm

from scipy.spatial.transform import Rotation as R
from upath import UPath

from ncore.data.v4 import (
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
from ncore.data_converter import (
    BaseDataConverter,
    BaseDataConverterConfig,
    FileBasedDataConverter,
    FileBasedDataConverterConfig,
)
from ncore.impl.common.transformations import HalfClosedInterval, se3_inverse, time_bounds
from ncore.impl.data.types import (
    BBox3,
    ConcreteLidarModelParametersUnion,
    CuboidTrackObservation,
    FThetaCameraModelParameters,
    JsonLike,
    LabelSource,
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
    RowOffsetStructuredSpinningLidarModelParameters,
)
from ncore.impl.data.v4.types import ComponentGroupAssignments
from tools.data_converter.cli import cli
from tools.data_converter.pai.data_provider import ClipDataProvider, LocalClipDataProvider, StreamingClipDataProvider
from tools.data_converter.pai.pai_remote.config import DEFAULT_REVISION, Config
from tools.data_converter.pai.pai_remote.index import ClipIndex
from tools.data_converter.pai.pai_remote.remote import HFRemote
from tools.data_converter.pai.utils import (
    filter_ego_vehicle_points,
    parse_camera_intrinsics,
    parse_egomotion_parquet,
    parse_sensor_extrinsics,
    parse_vehicle_dimensions,
)


logger = logging.getLogger(__name__)

CONVERTER_VERSION = "1.0.0"


@dataclass(kw_only=True, slots=True)
class PaiLocalConverterConfig(FileBasedDataConverterConfig):
    """Configuration for local PAI conversion (``pai-v4``).

    Inherits :class:`FileBasedDataConverterConfig` so ``root_dir`` is required
    and validated at construction time.
    """

    seek_sec: float | None = None
    duration_sec: float | None = None
    clip_id: list[str] = field(default_factory=list)
    store_type: Literal["itar", "directory"] = "itar"
    component_group_profile: Literal["default", "separate-sensors", "separate-all"] = "separate-sensors"
    store_sequence_meta: bool = True


@dataclass(kw_only=True, slots=True)
class PaiStreamConverterConfig(BaseDataConverterConfig):
    """Configuration for streaming PAI conversion (``pai-stream-v4``).

    Does **not** require ``root_dir``.  The ``make_provider`` callable is used
    to create a :class:`ClipDataProvider` for each clip at conversion time.
    """

    seek_sec: float | None = None
    duration_sec: float | None = None
    clip_id: list[str] = field(default_factory=list)
    store_type: Literal["itar", "directory"] = "itar"
    component_group_profile: Literal["default", "separate-sensors", "separate-all"] = "separate-sensors"
    store_sequence_meta: bool = True
    make_provider: Callable[[str], ClipDataProvider]  # factory: clip_id -> provider


class _PaiConversionMixin:
    """Shared conversion logic for PAI data converters.

    This mixin contains all the sensor decoding, metadata loading, and data
    writing logic shared between :class:`PaiConverter` (local file-based) and
    :class:`PaiStreamConverter` (HuggingFace streaming).

    Subclasses must set ``self.provider`` (a :class:`ClipDataProvider`) before
    the body of :meth:`_convert_clip` runs.
    """

    # -- Type stubs for attributes provided by concrete subclasses / base classes --
    # These are not set here; they exist on the concrete classes via their
    # BaseDataConverter (or FileBasedDataConverter) parent.
    logger: logging.Logger
    output_dir: UPath
    provider: ClipDataProvider

    # Methods from BaseDataConverter that the mixin calls
    get_active_camera_ids: Callable[..., list[str]]
    get_active_lidar_ids: Callable[..., list[str]]
    get_active_radar_ids: Callable[..., list[str]]

    # Camera sensor mapping
    CAMERA_SENSORS = [
        "camera_cross_left_120fov",
        "camera_cross_right_120fov",
        "camera_front_tele_30fov",
        "camera_front_wide_120fov",
        "camera_rear_left_70fov",
        "camera_rear_right_70fov",
        "camera_rear_tele_30fov",
    ]

    # Lidar sensor mapping
    LIDAR_SENSORS = [
        "lidar_top_360fov",
    ]

    def _init_pai_fields(self, config) -> None:
        """Initialise PAI-specific fields from *config* (a namespace or dataclass)."""
        self.seek_sec: float | None = config.seek_sec
        self.duration_sec: float | None = config.duration_sec
        self.store_type: Literal["itar", "directory"] = config.store_type
        self.component_group_profile: Literal["default", "separate-sensors", "separate-all"] = (
            config.component_group_profile
        )
        self.store_sequence_meta = config.store_sequence_meta

    def _convert_clip(self, clip_id: str) -> None:
        """Run the full conversion pipeline for a single clip.

        ``self.provider`` must already be set to the appropriate
        :class:`ClipDataProvider` before calling this method.
        """
        self.clip_id = clip_id
        self.logger.info(f"Converting clip {clip_id}")

        # Verify offline features are available (non-offline variants are currently not supported)
        self.provider.check_offline_features()

        # Load calibration and metadata
        try:
            self.load_metadata()
        except Exception as e:
            raise RuntimeError(f"Failed to load metadata for clip {clip_id}: {e}") from e

        # Determine active sensors
        all_camera_ids = [s for s in self.camera_intrinsics.keys()]
        all_lidar_ids = [s for s in self.sensor_extrinsics.keys() if "lidar" in s]
        active_camera_ids = self.get_active_camera_ids(all_camera_ids)
        active_lidar_ids = self.get_active_lidar_ids(all_lidar_ids)
        all_radar_ids = [s for s in self.sensor_extrinsics.keys() if s.startswith("radar_")]
        active_radar_ids = self.get_active_radar_ids(all_radar_ids)

        # Load egomotion (timestamps are shifted to be positive)
        ego_df = self.provider.load_parquet("egomotion")
        T_rig_worlds, T_rig_world_timestamps_us = parse_egomotion_parquet(ego_df)

        # Evaluate time restriction
        start_timesbound_us, end_timesbound_us = time_bounds(
            T_rig_world_timestamps_us.tolist(), self.seek_sec, self.duration_sec
        )

        selected_range = np.logical_and(
            T_rig_world_timestamps_us >= start_timesbound_us,
            T_rig_world_timestamps_us <= end_timesbound_us,
        )

        T_rig_worlds = T_rig_worlds[selected_range, :, :]
        T_rig_world_timestamps_us = T_rig_world_timestamps_us[selected_range]

        assert len(T_rig_worlds) >= 2, "at least two poses required in selected time range"

        # Select local world base pose (first pose defines global frame)
        T_world_world_global = T_rig_worlds[0].copy()

        # Convert all poses relative to local world base pose
        T_rig_worlds = se3_inverse(T_world_world_global) @ T_rig_worlds

        # Define target sequence time interval to coincide with the available egomotion
        self.sequence_timestamp_interval_us = HalfClosedInterval.from_start_end(
            T_rig_world_timestamps_us[0].item(),
            T_rig_world_timestamps_us[-1].item(),
        )

        # Create data writer
        sequence_id = f"pai_{self.clip_id}"

        # Use default profile if not specified
        self.component_groups = ComponentGroupAssignments.create(
            camera_ids=active_camera_ids,
            lidar_ids=active_lidar_ids,
            radar_ids=active_radar_ids,
            point_clouds_ids=[],  # No native point cloud sources
            camera_labels_ids=[],  # No camera labels
            profile=self.component_group_profile,
        )

        # Collect sequence metadata
        ego_bbox = BBox3(
            centroid=(
                self.vehicle_dimensions["rear_axle_to_bbox_center"],
                0.0,
                self.vehicle_dimensions["height"] * 0.5,
            ),
            dim=(
                self.vehicle_dimensions["length"],
                self.vehicle_dimensions["width"],
                self.vehicle_dimensions["height"],
            ),
            rot=(0.0, 0.0, 0.0),
        )
        generic_meta_data: Dict[str, JsonLike] = {
            "vehicle-bbox": cast(Dict[str, JsonLike], ego_bbox.to_dict()),
            "platform_class": self.platform_class,
        }
        source_meta = self.provider.get_source_metadata()
        source_generic_meta_data: Dict[str, JsonLike] = {
            "calibration_type": "pai-calibration",
            "egomotion_type": "pai-egomotion",
            "converter_version": CONVERTER_VERSION,
            "source_clip_id": source_meta.get("clip_id", self.clip_id),
            "source_repo_id": source_meta.get("repo_id"),
            "source_revision": source_meta.get("revision"),
            "source_commit_sha": source_meta.get("commit_sha"),
        }

        # Create main component group writer for this sequence
        self.store_writer = SequenceComponentGroupsWriter(
            output_dir_path=self.output_dir / sequence_id,
            store_base_name=sequence_id,
            sequence_id=sequence_id,
            sequence_timestamp_interval_us=self.sequence_timestamp_interval_us,
            store_type=self.store_type,  # can also be "directory"
            generic_meta_data={**source_generic_meta_data, **generic_meta_data},
        )

        ## Create poses component, store rig poses
        self.poses_writer = self.store_writer.register_component_writer(
            PosesComponent.Writer,
            component_instance_name="default",
            group_name=self.component_groups.poses_component_group,
        )
        self.poses_writer.store_dynamic_pose(
            source_frame_id="rig",
            target_frame_id="world",
            # we store rig->world poses as float32 (sufficiently accurate as relative to local-world)
            poses=T_rig_worlds.astype(np.float32),
            timestamps_us=T_rig_world_timestamps_us,
        )
        self.poses_writer.store_static_pose(
            source_frame_id="world",
            target_frame_id="world_global",
            # world->world_global potentially requires higher-precision, use it's float64 source precision
            pose=T_world_world_global,
        )

        ## Create intrinsics component
        self.intrinsics_writer = self.store_writer.register_component_writer(
            IntrinsicsComponent.Writer,
            component_instance_name="default",
            group_name=self.component_groups.intrinsics_component_group,
        )

        ## Create masks component
        self.masks_writer = self.store_writer.register_component_writer(
            MasksComponent.Writer,
            component_instance_name="default",
            group_name=self.component_groups.masks_component_group,
        )

        # Decode all sensors and store data in components
        self.decode_cameras(active_camera_ids)
        self.decode_lidars(active_lidar_ids)
        self.decode_radars(active_radar_ids)

        # Create and store cuboids component
        self.store_writer.register_component_writer(
            CuboidsComponent.Writer,
            "default",
            self.component_groups.cuboid_track_observations_component_group,
        ).store_observations(self._load_cuboid_track_observations())

        # Finalize, output sequence meta file if requested
        ncore_4_paths = self.store_writer.finalize()

        if self.store_sequence_meta:
            sequence_component_reader = SequenceComponentGroupsReader(ncore_4_paths)
            sequence_meta_path = self.output_dir / sequence_id / f"{sequence_component_reader.sequence_id}.json"

            with sequence_meta_path.open("w") as f:
                json.dump(sequence_component_reader.get_sequence_meta().to_dict(), f, indent=2)

            self.logger.info(f"Wrote sequence meta data {str(sequence_meta_path)}")

        # Close data streaming provider
        self.provider.close()

    def _load_lidar_model_parameters(self) -> ConcreteLidarModelParametersUnion | None:
        """Load lidar model parameters from lidar_intrinsics parquet."""

        if self.provider.has_file("lidar_intrinsics"):
            li_df = self.provider.load_parquet("lidar_intrinsics")
            model_params_dict = json.loads(li_df["model_parameters"].iloc[0])
            return RowOffsetStructuredSpinningLidarModelParameters.from_dict(model_params_dict)

        return None

    def _load_cuboid_track_observations(self) -> list[CuboidTrackObservation]:
        """Load and parse cuboid obstacle labels into CuboidTrackObservation list.

        Returns:
            List of CuboidTrackObservation, empty if no obstacle data is available.
        """

        if not self.provider.has_file("obstacle"):
            self.logger.warning("No obstacle parquet found, skipping cuboid tracks")
            return []

        self.logger.info("Loading cuboid observations")

        obstacle_df = self.provider.load_parquet("obstacle")

        ## Pre-filter data range based on sequence time bounds
        obstacle_df = obstacle_df[obstacle_df["timestamp_us"].ge(self.sequence_timestamp_interval_us.start)]
        obstacle_df = obstacle_df[obstacle_df["timestamp_us"].le(self.sequence_timestamp_interval_us.stop - 1)]

        cuboid_track_observations: list[CuboidTrackObservation] = []

        for row in tqdm.tqdm(obstacle_df.itertuples(), total=len(obstacle_df), desc="Processing cuboid labels"):
            bbox = BBox3(
                centroid=(cast(float, row.center_x), cast(float, row.center_y), cast(float, row.center_z)),
                dim=(cast(float, row.size_x), cast(float, row.size_y), cast(float, row.size_z)),
                rot=tuple(
                    R.from_quat(
                        (
                            cast(float, row.orientation_x),
                            cast(float, row.orientation_y),
                            cast(float, row.orientation_z),
                            cast(float, row.orientation_w),
                        )
                    )
                    .as_euler("xyz", degrees=False)
                    .tolist()
                ),
            )

            cuboid_track_observations.append(
                CuboidTrackObservation(
                    track_id=str(row.track_id),
                    class_id=str(row.label_class),
                    timestamp_us=cast(int, row.timestamp_us),
                    reference_frame_id=str(row.reference_frame),
                    reference_frame_timestamp_us=cast(int, row.reference_frame_timestamp_us),
                    bbox3=bbox,
                    source=LabelSource.AUTOLABEL if "autolabel" in str(row.source) else LabelSource.EXTERNAL,
                    source_version=str(row.source),
                )
            )

        self.logger.info(f"Loaded {len(cuboid_track_observations)} cuboid observations")

        return cuboid_track_observations

    def load_metadata(self) -> None:
        """Load calibration and metadata via the data provider."""
        self.logger.info("Loading metadata...")

        # Available sensors
        self.sensor_presence = self.provider.get_sensor_presence()

        # Load calibration data
        intrinsics_df = self.provider.load_parquet("camera_intrinsics")
        self.camera_intrinsics = parse_camera_intrinsics(intrinsics_df, self.clip_id, self.sensor_presence)

        extrinsics_df = self.provider.load_parquet("sensor_extrinsics")
        self.sensor_extrinsics = parse_sensor_extrinsics(extrinsics_df, self.clip_id, self.sensor_presence)

        # The .offline extrinsics may only contain cameras + lidar.  If radar
        # sensors are present according to feature_presence but missing from the
        # loaded extrinsics, fall back to the non-offline extrinsics for those.
        radar_present = [s for s in self.sensor_presence.index if s.startswith("radar_") and self.sensor_presence[s]]
        radar_missing = [s for s in radar_present if s not in self.sensor_extrinsics]
        if radar_missing and self.provider.has_file("sensor_extrinsics_online"):
            self.logger.info(
                f"Loading online extrinsics for {len(radar_missing)} radar sensors missing from offline calibration"
            )
            online_extrinsics_df = self.provider.load_parquet("sensor_extrinsics_online")
            online_extrinsics = parse_sensor_extrinsics(online_extrinsics_df, self.clip_id, self.sensor_presence)
            for radar_id in radar_missing:
                if radar_id in online_extrinsics:
                    self.sensor_extrinsics[radar_id] = online_extrinsics[radar_id]

        dimensions_df = self.provider.load_parquet("vehicle_dimensions")
        self.vehicle_dimensions = parse_vehicle_dimensions(dimensions_df, self.clip_id)

        self.logger.info(
            f"Loaded metadata: {len(self.camera_intrinsics)} cameras, "
            f"{len([k for k in self.sensor_extrinsics if 'lidar' in k])} lidars, "
            f"{len([k for k in self.sensor_extrinsics if k.startswith('radar_')])} radars"
        )

        # Platform details
        self.platform_class = self.provider.get_platform_class()
        assert self.platform_class in [
            "hyperion_8",
            "hyperion_8.1",
        ]

    def decode_lidars(self, active_lidar_id):
        logger = self.logger.getChild("decode_lidar")

        for lidar_id in active_lidar_id:
            logger.info(f"Decoding lidar data from {lidar_id}")
            if not self.provider.has_file(lidar_id):
                self.logger.warning(f"Lidar file not found for {lidar_id}")
                return

            # Load lidar parquet
            lidar_df = self.provider.load_parquet(lidar_id)

            # Get sensor transform
            T_sensor_rig = self.sensor_extrinsics[lidar_id]

            # Load lidar model parameters
            lidar_model_params = self._load_lidar_model_parameters()

            if lidar_model_params is None:
                self.logger.warning("Lidar model parameters could not be loaded")

            # store intrinsics conditionally
            if lidar_model_params is not None:
                self.intrinsics_writer.store_lidar_intrinsics(
                    lidar_id=lidar_id,
                    lidar_model_parameters=lidar_model_params,
                )

            # store extrinsics
            self.poses_writer.store_static_pose(
                source_frame_id=lidar_id,
                target_frame_id="rig",
                # we store sensor->rig poses as float32 in V4
                # (same as in V3, just be explicit about it)
                pose=T_sensor_rig.astype(np.float32),
            )

            # store frames
            lidar_writer = self.store_writer.register_component_writer(
                LidarSensorComponent.Writer,
                component_instance_name=lidar_id,
                group_name=self.component_groups.lidar_component_groups.get(lidar_id),
            )

            for source_frame_idx in tqdm.tqdm(range(lidar_df.shape[0]), desc="Decoding lidar point clouds"):
                row = lidar_df.iloc[source_frame_idx]

                frame_end_timestamp_us = row["spin_end_timestamp"]
                frame_start_timestamp_us = row["spin_start_timestamp"]

                # Skip frames outside of sequence time interval
                if frame_start_timestamp_us < self.sequence_timestamp_interval_us.start:
                    continue
                if frame_end_timestamp_us >= self.sequence_timestamp_interval_us.stop:
                    break

                # Timestamps at frame start/end
                timestamps_us = np.array([frame_start_timestamp_us, frame_end_timestamp_us], dtype=np.uint64)

                # Decompress DRACO point cloud
                pc = DracoPy.decode(lidar_df["draco_encoded_pointcloud"].iloc[source_frame_idx])

                xyz_m = np.array(pc.points)
                attributes = {attr["name"]: attr["data"] for attr in pc.attributes}
                intensity = attributes["intensity"].squeeze().astype(np.float32) / 255.0
                point_timestamps_us = attributes["timestamp"].squeeze()
                model_element = attributes["model_element"]

                # Filter ego vehicle points
                valid_mask = filter_ego_vehicle_points(xyz_m, T_sensor_rig, self.vehicle_dimensions, padding=0.5)

                xyz_m = xyz_m[valid_mask]
                point_timestamps_us = point_timestamps_us[valid_mask]
                intensity = intensity[valid_mask]
                model_element = model_element[valid_mask]

                # After filtering, ensure point timestamps are still within bounds
                # (In case filtering changed the distribution)
                if len(point_timestamps_us) > 0:
                    valid_mask = np.logical_and(
                        point_timestamps_us >= frame_start_timestamp_us, point_timestamps_us <= frame_end_timestamp_us
                    )

                    xyz_m = xyz_m[valid_mask]
                    point_timestamps_us = point_timestamps_us[valid_mask].astype(np.uint64)
                    intensity = intensity[valid_mask]
                    model_element = model_element[valid_mask].astype(np.uint16)

                # extract directions / distances
                distance_m = np.linalg.norm(xyz_m, axis=1)
                direction = xyz_m / distance_m[:, np.newaxis]

                # filter for non-negative distances
                valid_mask = distance_m > 0

                lidar_writer.store_frame(
                    # non-motion-compensated per-ray 3D directions in the sensor frame at measurement time (float32, [n, 3])
                    direction=direction[valid_mask],
                    # per-point point timestamp in microseconds (uint64, [n])
                    timestamp_us=point_timestamps_us[valid_mask],
                    # per-point model element indices, if applicable (uint16, [n, 2])
                    model_element=model_element[valid_mask],
                    # per-point distance (only single return supported in V3) [n, r] with r=1)
                    distance_m=distance_m[valid_mask][np.newaxis],
                    # per-point intensity normalized to [0.0, 1.0] range (float32, [n, r] with r=1)
                    intensity=intensity[valid_mask][np.newaxis],
                    # frame start/end timestamps (uint64, [2])
                    frame_timestamps_us=timestamps_us,
                    generic_data={},
                    generic_meta_data={},
                )

    def decode_radars(self, active_radar_ids: list[str]) -> None:
        logger = self.logger.getChild("decode_radars")

        for radar_id in active_radar_ids:
            logger.info(f"Decoding radar data from {radar_id}")
            if not self.provider.has_file(radar_id):
                logger.warning(f"Radar file not found for {radar_id}")
                continue

            # Load radar parquet
            radar_df = self.provider.load_parquet(radar_id)

            # Get sensor transform
            T_sensor_rig = self.sensor_extrinsics[radar_id]

            # Store extrinsics
            self.poses_writer.store_static_pose(
                source_frame_id=radar_id,
                target_frame_id="rig",
                pose=T_sensor_rig.astype(np.float32),
            )

            # Create radar writer
            radar_writer = self.store_writer.register_component_writer(
                RadarSensorComponent.Writer,
                component_instance_name=radar_id,
                group_name=self.component_groups.radar_component_groups.get(radar_id),
            )

            # Group detections by scan_index (stored as the DataFrame index)
            for scan_index, scan_df in tqdm.tqdm(radar_df.groupby(level=0), desc=f"Radar {radar_id}"):
                # Infer frame timestamps from per-detection timestamps
                timestamp_us = cast(np.ndarray, scan_df["sensor_timestamp"].values.astype(np.int64))

                frame_start_us = int(timestamp_us.min())
                frame_end_us = int(timestamp_us.max())

                # Skip scans with non-positive timestamps (before time origin)
                if frame_end_us < 0:
                    continue

                # Skip frames outside sequence time interval
                if frame_start_us < self.sequence_timestamp_interval_us.start:
                    continue
                if frame_end_us >= self.sequence_timestamp_interval_us.stop:
                    break

                # Filter out negative timestamps
                valid = timestamp_us >= 0
                if not valid.all():
                    scan_df = scan_df[valid]
                    timestamp_us = timestamp_us[valid]

                # Convert spherical to unit direction vectors
                azimuth = scan_df["azimuth"].values.astype(np.float64)
                elevation = scan_df["elevation"].values.astype(np.float64)

                cos_el = np.cos(elevation)
                direction = np.stack(
                    [
                        cos_el * np.cos(azimuth),
                        cos_el * np.sin(azimuth),
                        np.sin(elevation),
                    ],
                    axis=-1,
                ).astype(np.float32)

                # Distance (metric, single return)
                distance_m = cast(np.ndarray, scan_df["distance"].values.astype(np.float32))

                # Filter out zero/negative distances
                valid_mask = distance_m > 0
                direction = direction[valid_mask]
                timestamp_us = timestamp_us[valid_mask]
                distance_m = distance_m[np.newaxis, valid_mask]  # [1, N]

                # Frame timestamps
                frame_timestamps_us = np.array([frame_start_us, frame_end_us], dtype=np.uint64)

                # Generic data: per-detection radar signals
                generic_data: dict[str, np.ndarray] = {}

                if "radial_velocity" in scan_df.columns:
                    # Units: m/s (meters per second), positive away from the sensor
                    generic_data["radial_velocity_m_s"] = cast(
                        np.ndarray, scan_df["radial_velocity"].values[valid_mask].astype(np.float32)
                    )
                if "rcs" in scan_df.columns:
                    # Units: dBsm (decibels relative to square meter)
                    generic_data["rcs_dBsm"] = cast(np.ndarray, scan_df["rcs"].values[valid_mask].astype(np.float32))
                if "snr" in scan_df.columns:
                    # Units: dB (decibels), higher ~ more reliable detection
                    generic_data["snr_dB"] = cast(np.ndarray, scan_df["snr"].values[valid_mask].astype(np.float32))
                if "exist_probb" in scan_df.columns:
                    # Normalize uint8 (0-100) to float32 (0-1)
                    generic_data["exist_probb"] = (
                        cast(np.ndarray, scan_df["exist_probb"].values[valid_mask].astype(np.float32)) / 100.0
                    )

                # Generic metadata: scan-level scalars
                generic_meta_data: dict[str, JsonLike] = {
                    "scan_index": cast(int, scan_index),
                }
                for col, meta_key, conv in [
                    ("doppler_ambiguity", "doppler_ambiguity", float),
                    ("radar_model", "radar_model", int),
                    ("num_returns", "num_returns", int),
                ]:
                    if col in scan_df.columns:
                        val = scan_df[col].iloc[0]
                        try:
                            if not np.isnan(float(val)):
                                generic_meta_data[meta_key] = conv(val)
                        except (ValueError, TypeError):
                            pass

                radar_writer.store_frame(
                    direction=direction,
                    timestamp_us=timestamp_us.astype(np.uint64),
                    distance_m=distance_m,
                    frame_timestamps_us=frame_timestamps_us,
                    generic_data=generic_data,
                    generic_meta_data=generic_meta_data,
                )

    def decode_cameras(self, active_camera_ids: list[str]):
        logger = self.logger.getChild("decode_cameras")

        for camera_id in active_camera_ids:
            logger.info(f"Processing camera {camera_id}")

            timestamps_key = f"{camera_id}_timestamps"
            blur_boxes_key = f"{camera_id}_blurred_boxes"

            # Store intrinsics
            intrinsics = self.camera_intrinsics[camera_id]

            shutter_delay_us: int
            camera_model_parameters: (
                FThetaCameraModelParameters | OpenCVFisheyeCameraModelParameters | OpenCVPinholeCameraModelParameters
            )
            if (model_parameters := intrinsics.get("model_parameters")) is not None:
                shutter_delay_us = model_parameters["shutter_delay_us"]
                model_type = intrinsics.get("model_type")
                if model_type == FThetaCameraModelParameters.type():
                    camera_model_parameters = FThetaCameraModelParameters.from_dict(model_parameters)
                elif model_type == OpenCVFisheyeCameraModelParameters.type():
                    camera_model_parameters = OpenCVFisheyeCameraModelParameters.from_dict(model_parameters)
                elif model_type == OpenCVPinholeCameraModelParameters.type():
                    camera_model_parameters = OpenCVPinholeCameraModelParameters.from_dict(model_parameters)
                else:
                    raise ValueError(f"Unsupported camera model type: {model_parameters['model_type']}")
            else:
                raise ValueError(f"Camera intrinsics for {camera_id} missing model_parameters")

            self.intrinsics_writer.store_camera_intrinsics(
                camera_id=camera_id,
                camera_model_parameters=camera_model_parameters,
            )

            # Store extrinsics
            T_sensor_rig = self.sensor_extrinsics[camera_id]
            self.poses_writer.store_static_pose(
                source_frame_id=camera_id,
                target_frame_id="rig",
                pose=T_sensor_rig.astype(np.float32),
            )

            # Store ego mask (if available) in masks component as 'ego' mask type
            mask_images = {}
            if (ego_mask_image := intrinsics.get("ego_mask_image")) is not None:
                mask_images["ego"] = ego_mask_image

            self.masks_writer.store_camera_masks(
                camera_id=camera_id,
                mask_images=mask_images,
            )

            # Create camera writer for this sensor
            camera_writer = self.store_writer.register_component_writer(
                CameraSensorComponent.Writer,
                component_instance_name=camera_id,
                group_name=self.component_groups.camera_component_groups.get(camera_id),
            )

            # Load frame timestamps for this sensor
            timestamps_df = self.provider.load_parquet(timestamps_key)

            # Load blur boxes for mask generation (optional)
            blur_boxes_df = None
            if self.provider.has_file(blur_boxes_key):
                try:
                    blur_boxes_df = self.provider.load_parquet(blur_boxes_key)
                    self.logger.info(f"Loaded {len(blur_boxes_df)} blur boxes for {camera_id}")
                except Exception as e:
                    self.logger.warning(f"Failed to load blur boxes for {camera_id}: {e}")

            # Get video path (local file or temp file from streaming)
            video_path = self.provider.get_video_path(camera_id)
            self.logger.info(f"Extracting {len(timestamps_df)} frames from {video_path.name}")

            reader = imageio.get_reader(video_path, "ffmpeg")  # type: ignore

            for frame_idx in tqdm.tqdm(range(len(timestamps_df)), desc=f"Camera {camera_id}"):
                # Keep timestamp as int because as these might be negative
                frame_end_timestamp_us = int(timestamps_df.iloc[frame_idx]["timestamp"])

                # Read the image / forward the reader
                image = reader.get_data(frame_idx)

                # Compute start of frame from rolling shutter delay
                frame_start_timestamp_us = frame_end_timestamp_us - shutter_delay_us

                # Skip frame if outside of sequence time
                if frame_start_timestamp_us < self.sequence_timestamp_interval_us.start:
                    continue
                if frame_end_timestamp_us >= self.sequence_timestamp_interval_us.stop:
                    break

                # Encode image to JPEG bytes
                jpeg_bytes = imageio.imwrite("<bytes>", image, format="jpeg", quality=93)  # type: ignore

                # Append blur box metadata, if available
                generic_meta_data = {}
                if blur_boxes_df is not None:
                    frame_blur_boxes = blur_boxes_df[blur_boxes_df["frame_index"] == frame_idx]
                    generic_meta_data["blur_boxes"] = frame_blur_boxes[["x1", "y1", "x2", "y2"]].to_dict(orient="list")

                camera_writer.store_frame(
                    image_binary_data=jpeg_bytes,
                    image_format="jpeg",
                    frame_timestamps_us=np.array([frame_start_timestamp_us, frame_end_timestamp_us], dtype=np.uint64),
                    generic_data={},
                    generic_meta_data=generic_meta_data,
                )


# ---------------------------------------------------------------------------
# Concrete converter classes
# ---------------------------------------------------------------------------


class PaiConverter(_PaiConversionMixin, FileBasedDataConverter):
    """PAI data converter for local files (``pai-v4``).

    Reads clip data from a local root directory previously downloaded with
    the ``pai-clip-dl`` tool.  ``--root-dir`` must point to the directory
    containing clip subdirectories.
    """

    def __init__(self, config: PaiLocalConverterConfig) -> None:
        super().__init__(config)
        self._init_pai_fields(config)

    @staticmethod
    def get_sequence_ids(config) -> list[str]:
        """Discover clips by listing subdirectories under ``root_dir``."""
        data_root = UPath(config.root_dir)
        clip_ids = sorted(d.name for d in data_root.iterdir() if d.is_dir())

        if not clip_ids:
            raise FileNotFoundError(f"No clip directories found under {data_root}")

        # Filter by specific clip IDs if provided
        if hasattr(config, "clip_id") and config.clip_id:
            requested_ids = set(config.clip_id)
            found_ids = set(clip_ids)
            missing_ids = requested_ids - found_ids
            if missing_ids:
                logger.warning(f"Clip IDs not found under {data_root}: {missing_ids}")
            clip_ids = [c for c in clip_ids if c in requested_ids]

        return clip_ids

    @staticmethod
    def from_config(config) -> PaiConverter:
        return PaiConverter(config)

    def convert_sequence(self, sequence_id: str) -> None:
        clip_id = sequence_id
        clip_dir = self.root_dir / clip_id
        self.provider: ClipDataProvider = LocalClipDataProvider(clip_dir, clip_id)
        self._convert_clip(clip_id)


class PaiStreamConverter(_PaiConversionMixin, BaseDataConverter):
    """PAI data converter for HuggingFace streaming (``pai-stream-v4``).

    Converts PAI clips directly from HuggingFace without prior download.
    Does **not** require ``--root-dir``.
    """

    def __init__(self, config: PaiStreamConverterConfig) -> None:
        super().__init__(config)
        self._init_pai_fields(config)
        self._make_provider = config.make_provider

    @staticmethod
    def get_sequence_ids(config) -> list[str]:
        """Return clip IDs provided via ``--clip-id``."""
        clip_ids: list[str] = list(config.clip_id)
        if not clip_ids:
            raise ValueError("--clip-id is required for streaming mode")
        return clip_ids

    @staticmethod
    def from_config(config) -> PaiStreamConverter:
        return PaiStreamConverter(config)

    def convert_sequence(self, sequence_id: str) -> None:
        clip_id = sequence_id
        self.provider: ClipDataProvider = self._make_provider(clip_id)
        self._convert_clip(clip_id)


# ---------------------------------------------------------------------------
# Shared PAI options (used by both pai-v4 and pai-stream-v4)
# ---------------------------------------------------------------------------

_pai_shared_options = [
    click.option(
        "--seek-sec",
        default=None,
        type=click.FloatRange(min=0.0, max_open=True),
        help="Time to skip for the dataset conversion (in seconds)",
    ),
    click.option(
        "--duration-sec",
        default=None,
        type=click.FloatRange(min=0.0, max_open=True),
        help="Restrict total duration of the dataset conversion (in seconds)",
    ),
    click.option(
        "--clip-id",
        multiple=True,
        type=str,
        default=[],
        help="Specific clip ID(s) to convert. If not specified, converts all clip directories found under root-dir.",
    ),
    click.option(
        "--store-type",
        type=click.Choice(["itar", "directory"], case_sensitive=False),
        default="itar",
        show_default=True,
        help="Output store type",
    ),
    click.option(
        "component_group_profile",
        "--profile",
        type=click.Choice(["default", "separate-sensors", "separate-all"], case_sensitive=False),
        default="separate-sensors",
        show_default=True,
        help="""Output profile, one of:
        - "default": All components defaults or overrides
        - "separate-sensors": Each sensor gets its own group named "<sensor_id>", remaining components use overrides
        - "separate-all": Each component type gets its own group named after the component type, e.g. "poses", "intrinsics", respecting overwrites if provided""",
    ),
    click.option(
        "store_sequence_meta", "--sequence-meta/--no-sequence-meta", default=True, help="Generate sequence meta-data?"
    ),
]


def _apply_options(options):
    """Decorator that applies a list of click options to a command."""

    def decorator(func):
        for option in reversed(options):
            func = option(func)
        return func

    return decorator


@cli.command("pai-v4")
@_apply_options(_pai_shared_options)
@click.pass_context
def pai_v4(ctx, *_, **kwargs):
    """Physical AI data conversion (local pai-clip-dl output)

    Converts PAI clips previously downloaded with the pai-clip-dl tool.
    --root-dir must point to the directory containing clip subdirectories.
    """

    config = vars(ctx.obj)
    config.update(kwargs)

    PaiConverter.convert(PaiLocalConverterConfig(**config))


@cli.command("pai-stream-v4")
@_apply_options(_pai_shared_options)
@click.option(
    "--hf-token",
    type=str,
    default=None,
    envvar="HF_TOKEN",
    help="HuggingFace API token. Reads from HF_TOKEN env var if not provided.",
)
@click.option(
    "--revision",
    type=str,
    default=DEFAULT_REVISION,
    show_default=True,
    help="HuggingFace dataset revision (branch/tag).",
)
@click.pass_context
def pai_stream_v4(ctx, *_, **kwargs):
    """Physical AI data conversion (streaming from HuggingFace)

    Converts PAI clips directly from HuggingFace without prior download.
    --clip-id is required.  --root-dir is not needed.
    """

    hf_token = kwargs.pop("hf_token")
    revision = kwargs.pop("revision")

    config = vars(ctx.obj)
    config.pop("root_dir", None)  # not a field on PaiStreamConverterConfig
    config.update(kwargs)

    clip_ids = config.get("clip_id", ())
    if not clip_ids:
        raise click.UsageError("--clip-id is required for pai-stream-v4")

    pai_config = Config.from_env(token=hf_token, revision=revision)
    remote = HFRemote(pai_config)
    index = ClipIndex(remote)

    # Shared chunk parquet cache across clips
    chunk_parquet_cache: dict = {}

    with tempfile.TemporaryDirectory(prefix="pai-stream-v4-") as temp_root:

        def make_provider(clip_id: str) -> StreamingClipDataProvider:
            clip_temp = Path(temp_root) / clip_id
            return StreamingClipDataProvider(
                clip_id=clip_id,
                remote=remote,
                index=index,
                temp_dir=clip_temp,
                chunk_parquet_cache=chunk_parquet_cache,
            )

        config["make_provider"] = make_provider

        PaiStreamConverter.convert(PaiStreamConverterConfig(**config))


if __name__ == "__main__":
    cli(show_default=True)
