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

"""Waymo Open Dataset to NCore V4 converter."""

from __future__ import annotations

import json
import logging

from dataclasses import asdict, dataclass
from typing import Dict, List, Literal, Optional

import click
import numpy as np
import tqdm

from upath import UPath

from ncore.impl.common.transformations import (
    HalfClosedInterval,
    MotionCompensator,
    PoseInterpolator,
    se3_inverse,
    transform_bbox,
    transform_point_cloud,
)
from ncore.impl.common.util import unpack_optional
from ncore.impl.data.types import (
    BBox3,
    CuboidTrackObservation,
    JsonLike,
    LabelSource,
    OpenCVPinholeCameraModelParameters,
    RowOffsetStructuredSpinningLidarModelParameters,
    ShutterType,
)
from ncore.impl.data.util import FOV, relative_angle
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
from ncore.impl.data_converter.base import BaseDataConverter, BaseDataConverterConfig
from ncore.impl.sensors.lidar import RowOffsetStructuredSpinningLidarModel
from tools.data_converter.cli import cli
from tools.data_converter.waymo.deps import camera_segmentation_pb2, dataset_pb2, label_pb2, tf
from tools.data_converter.waymo.utils import (
    convert_range_images_to_point_clouds,
    extrapolate_pose_based_on_velocity,
)
from tools.data_converter.waymo.waymo_range_image_utils import parse_range_image_and_segmentations


@dataclass(kw_only=True, slots=True)
class WaymoConverter4Config(BaseDataConverterConfig):
    """Configuration for Waymo to NCore V4 conversion."""

    store_type: Literal["itar", "directory"] = "itar"
    component_group_profile: Literal["default", "separate-sensors", "separate-all"] = "separate-sensors"
    store_sequence_meta: bool = True


class WaymoConverter4(BaseDataConverter):
    """
    Dataset preprocessing class for converting Waymo Open Dataset to NCore V4 format.

    Waymo-open data can be downloaded from https://waymo.com/intl/en_us/open/download/ in form of
    tfrecords files. Further details on the dataset are available in the original publication
    https://arxiv.org/abs/1912.04838 or the GitHub repository
    https://github.com/waymo-research/waymo-open-dataset
    """

    CAMERA_MAP = {
        dataset_pb2.CameraName.FRONT: "camera_front_50fov",
        dataset_pb2.CameraName.FRONT_LEFT: "camera_front_left_50fov",
        dataset_pb2.CameraName.FRONT_RIGHT: "camera_front_right_50fov",
        dataset_pb2.CameraName.SIDE_LEFT: "camera_side_left_50fov",
        dataset_pb2.CameraName.SIDE_RIGHT: "camera_side_right_50fov",
    }

    LIDAR_MAP = {
        dataset_pb2.LaserName.TOP: "lidar_top",
        # TODO: currently only support top lidar, as motion-compensation poses for
        # other lidars seems to be missing in the source data
    }

    FINE_LIDAR_TIMESTAMPS_MINIMUM_SENSOR_MOTION_THRESHOLD_M = (
        0.1  # threshold on the per spin linear motion of the lidar sensor to allow accurate timestamp inference
    )

    def __init__(self, config: WaymoConverter4Config) -> None:
        super().__init__(config)

        self.component_group_profile: Literal["default", "separate-sensors", "separate-all"] = (
            config.component_group_profile
        )
        self.store_type: Literal["itar", "directory"] = config.store_type
        self.store_sequence_meta: bool = config.store_sequence_meta

        self.logger = logging.getLogger(__name__)

    @staticmethod
    def get_sequence_paths(config) -> list[UPath]:
        return [p for p in sorted(UPath(config.root_dir).glob("*.tfrecord"))]

    @staticmethod
    def from_config(config) -> BaseDataConverter:
        return WaymoConverter4(config)

    def convert_sequence(self, sequence_path: UPath) -> None:
        """Runs dataset-specific conversion for a sequence."""
        self.logger.info(sequence_path)

        dataset = tf.data.TFRecordDataset(sequence_path, compression_type="")

        # Check that all frames in the dataset have the same sequence name (i.e. belong to the same sequence)
        # and deserialize into memory
        frames: list[dataset_pb2.Frame] = []
        sequence_name = ""
        for data in dataset:
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            if not frames:
                sequence_name = frame.context.name
            frames.append(frame)
            if frame.context.name != sequence_name:
                raise ValueError("NOT ALL FRAMES BELONG TO THE SAME SEQUENCE. ABORTING THE CONVERSION!")

        # Decode poses
        self.decode_poses(frames)

        ## Initialize V4 SequenceComponentGroupsWriter and component writers.

        # Create timestamp interval for sequence
        sequence_timestamp_interval_us = HalfClosedInterval.from_start_end(
            self.pose_interpolator.timestamps[0].item(),
            self.pose_interpolator.timestamps[-1].item(),
        )

        # Create component group assignments
        self.camera_ids = self.get_active_camera_ids([camera for camera in self.CAMERA_MAP.values()])
        self.lidar_ids = self.get_active_lidar_ids([lidar for lidar in self.LIDAR_MAP.values()])
        self.radar_ids = (self.get_active_radar_ids([]),)

        self.component_groups = ComponentGroupAssignments.create(
            camera_ids=self.camera_ids,
            lidar_ids=self.lidar_ids,
            radar_ids=[],  # No radars for now
            profile=self.component_group_profile,
        )

        # Create main store writer
        self.store_writer = SequenceComponentGroupsWriter(
            output_dir_path=self.output_dir / sequence_name,
            store_base_name=sequence_name,
            sequence_id=sequence_name,
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

        ## Store poses
        self.store_poses()

        ## Decode and store lidar frames
        self.decode_lidars(frames)

        ## Decode and store camera frames
        self.decode_cameras(frames)

        # Store per-shard meta data / final success state / close file
        ncore_4_paths = self.store_writer.finalize()

        # Output sequence meta file if requested
        if self.store_sequence_meta:
            sequence_component_reader = SequenceComponentGroupsReader(ncore_4_paths)
            sequence_meta_path = self.output_dir / sequence_name / f"{sequence_component_reader.sequence_id}.json"

            with sequence_meta_path.open("w") as f:
                json.dump(sequence_component_reader.get_sequence_meta().to_dict(), f, indent=2)

            self.logger.info(f"Wrote sequence meta data {str(sequence_meta_path)}")

    def decode_poses(self, frames: list[dataset_pb2.Frame]) -> None:
        """Decode and interpolate vehicle poses from frame data."""
        # Grab poses from images, as for images the pose/timestamps correspond to each other
        T_rig_worlds_array = []
        T_rig_world_timestamps_us_array: List[int] = []

        # Buffers holding extrapolated poses to cover all frame start/end timestamps fully
        T_rig_worlds_array_extrapolated = []
        T_rig_world_timestamps_us_array_extrapolated = []

        for i, frame in enumerate(frames):
            for image in frame.images:
                # Get the rig / SDC car pose
                # Confirmed in issue https://github.com/waymo-research/waymo-open-dataset/issues/464
                # That this pose and timestamp are corresponding
                T_rig_worlds_array.append(
                    np.array(tf.reshape(tf.constant(image.pose.transform, dtype=tf.float64), [4, 4]))
                )
                T_rig_world_timestamps_us_array.append(
                    int(image.pose_timestamp * 1e6)
                )  # Convert the poses to microseconds (rounding decimal)

                # Extrapolate pose points on the boundaries using velocity information
                # to allow interpolation at full lidar / camera timestamps
                dts_us = []
                if i == 0:
                    # extrapolate exactly to first lidar start-of-spin time
                    dts_us.append(int(frame.timestamp_micros - T_rig_world_timestamps_us_array[-1]))

                    # extrapolate to camera frame start time
                    camera_frame_start_timestamp_us = int((image.camera_trigger_time + image.shutter / 2) * 1e6)
                    if camera_frame_start_timestamp_us < T_rig_world_timestamps_us_array[-1]:
                        dts_us.append(camera_frame_start_timestamp_us - T_rig_world_timestamps_us_array[-1])

                if i == len(frames) - 1:
                    # make sure to overshoot a little over last lidar end-of-spin time
                    dts_us.append(int(1.25 * (frames[-1].timestamp_micros - frames[-2].timestamp_micros)))

                    # extrapolate to camera frame end time
                    camera_frame_end_timestamp_us = int((image.camera_readout_done_time - image.shutter / 2) * 1e6)
                    if camera_frame_end_timestamp_us > T_rig_world_timestamps_us_array[-1]:
                        dts_us.append(camera_frame_end_timestamp_us - T_rig_world_timestamps_us_array[-1])

                for dt_us in dts_us:
                    T_rig_world = T_rig_worlds_array[-1]
                    velocity_global = np.array(
                        [image.velocity.v_x, image.velocity.v_y, image.velocity.v_z],
                        dtype=np.float32,
                    ).reshape(3, 1)
                    omega_vehicle = np.array(
                        [image.velocity.w_x, image.velocity.w_y, image.velocity.w_z],
                        dtype=np.float32,
                    ).reshape(3, 1)
                    omega_world = np.matmul(T_rig_world[:3, :3], omega_vehicle)

                    T_rig_worlds_array_extrapolated.append(
                        extrapolate_pose_based_on_velocity(T_rig_world, velocity_global, omega_world, dt_us / 1e6)
                    )
                    T_rig_world_timestamps_us_array_extrapolated.append(T_rig_world_timestamps_us_array[-1] + dt_us)

        # concat + make unique + sort + stack all poses (common canonical format convention)
        T_rig_world_timestamps_us, unique_indices = np.unique(
            np.concatenate(
                (
                    np.array(T_rig_world_timestamps_us_array, dtype=np.uint64),
                    np.array(T_rig_world_timestamps_us_array_extrapolated, dtype=np.uint64),
                )
            ),
            return_index=True,
        )
        T_rig_worlds = np.concatenate((T_rig_worlds_array, T_rig_worlds_array_extrapolated))[unique_indices]

        # Use identity base pose as waymo data is already shifted
        self.T_world_world_global = np.eye(4, dtype="float64")

        self.pose_interpolator = PoseInterpolator(T_rig_worlds, T_rig_world_timestamps_us)

        # Log base pose to share it more easily with downstream teams (it is serialized also explicitly)
        with np.printoptions(floatmode="unique", linewidth=200):  # print in highest precision
            self.logger.info(f"> processed {len(T_rig_worlds)} poses, using base pose:\n{self.T_world_world_global}")

    def store_poses(self) -> None:
        """Stores the processed egomotion poses into the poses component."""
        # Store dynamic rig->world poses (float32 for relative poses)
        self.poses_writer.store_dynamic_pose(
            source_frame_id="rig",
            target_frame_id="world",
            poses=self.pose_interpolator.poses.astype(np.float32),
            timestamps_us=self.pose_interpolator.timestamps,
        )

        # Store static world->world_global pose (float64 for high precision)
        self.poses_writer.store_static_pose(
            source_frame_id="world",
            target_frame_id="world_global",
            pose=self.T_world_world_global.astype(np.float64),
        )

    # Label IDs to label type strings
    LIDAR_LABEL_CLASS_ID_STRING_MAP = {
        label_pb2.Label.Type.TYPE_UNKNOWN: "unknown",
        label_pb2.Label.Type.TYPE_VEHICLE: "vehicle",
        label_pb2.Label.Type.TYPE_PEDESTRIAN: "pedestrian",
        label_pb2.Label.Type.TYPE_SIGN: "sign",
        label_pb2.Label.Type.TYPE_CYCLIST: "cyclist",
    }

    # Unconditionally dynamic / static label types
    LABEL_STRINGS_UNCONDITIONALLY_DYNAMIC: set[str] = set(
        [
            "pedestrian",
            "cyclist",
        ]
    )
    LABEL_STRINGS_UNCONDITIONALLY_STATIC: set[str] = set(["sign"])

    # Velocity threshold to classify moving objects as dynamic
    GLOBAL_SPEED_DYNAMIC_THRESHOLD = 1.0 / 3.6

    # Dynamic flag from label bbox padding
    LIDAR_DYNAMIC_FLAG_BBOX_PADDING_METERS = 0.5

    def decode_lidars(self, frames: list[dataset_pb2.Frame]) -> None:
        """
        Converts the raw point cloud data into intrinsic 3D depth rays in space also compensating for the
        motion of the ego-car (lidar unwinding)
        """
        ## Collect calibrations
        calibrations = {c.name: c for c in frames[0].context.laser_calibrations}

        ## Collect frame start timestamps
        raw_frame_start_timestamps_us = [frame.timestamp_micros for frame in frames]

        ## Parse frame-associated labels in rig space (will be transformed to sensor frames below)

        # Type representing a parsed waymo 3D label
        @dataclass
        class RawFrameLabel3:
            track_id: str
            label_class: str
            bbox3: BBox3
            global_speed: float

        raw_frame_labels: dict[int, list[RawFrameLabel3]] = {}  # timestamp to label in vehicle frame
        for frame in tqdm.tqdm(frames, desc="Parse frame labels"):
            frame_label_list: list[RawFrameLabel3] = []

            for label in frame.laser_labels:
                box = label.box
                frame_label_list.append(
                    RawFrameLabel3(
                        track_id=label.id,
                        label_class=self.LIDAR_LABEL_CLASS_ID_STRING_MAP[label.type],
                        bbox3=BBox3.from_array(
                            np.array(
                                [
                                    box.center_x,
                                    box.center_y,
                                    box.center_z,
                                    box.length,
                                    box.width,
                                    box.height,
                                    0,
                                    0,
                                    box.heading,
                                ],
                                dtype=np.float32,
                            )
                        ),
                        # Velocity is given in the global frame -> map to frame-independent speed
                        global_speed=float(
                            np.linalg.norm(
                                np.array(
                                    [label.metadata.speed_x, label.metadata.speed_y],
                                    dtype=np.float32,
                                )
                            )
                        ),
                    )
                )

            raw_frame_labels[frame.timestamp_micros] = frame_label_list

        # Initialize labels struct that gets assembled while processing each frame label
        cuboid_track_observations: list[CuboidTrackObservation] = []  # list of all cuboid track observations

        for lidar_id, lidar_ncore_id in self.LIDAR_MAP.items():
            if lidar_ncore_id not in self.lidar_ids:
                continue  # skip sensor if not active

            # Determine sensor extrinsics
            T_sensor_rig = np.array(calibrations[lidar_id].extrinsic.transform, dtype=np.float32).reshape(4, 4)

            # Initialize motion compensator
            motion_compensator = MotionCompensator.from_sensor_rig(
                sensor_id=lidar_ncore_id,
                T_sensor_rig=T_sensor_rig,
                T_rig_worlds=self.pose_interpolator.poses,
                T_rig_worlds_timestamps_us=self.pose_interpolator.timestamps,
            )

            # Variables associated with intrinsics
            lidar_model: Optional[RowOffsetStructuredSpinningLidarModel] = None
            lidar_model_parameters: Optional[RowOffsetStructuredSpinningLidarModelParameters] = None
            lidar_horizontal_fov: Optional[FOV] = None

            # Create lidar sensor writer
            lidar_writer = self.store_writer.register_component_writer(
                LidarSensorComponent.Writer,
                component_instance_name=lidar_ncore_id,
                group_name=self.component_groups.lidar_component_groups.get(lidar_ncore_id),
                generic_meta_data={
                    "label-class-string-id-map": {
                        label_string: label_id
                        for label_id, label_string in self.LIDAR_LABEL_CLASS_ID_STRING_MAP.items()
                    },
                },
            )

            # Collect all lidar per-frame data
            assert len(frames) > 1  # require at least two frames to compute frame bound timestamps
            for i, frame in tqdm.tqdm(enumerate(frames), desc=f"Process {lidar_ncore_id}", total=len(frames)):
                # Get frame timestamps
                frame_start_timestamp_us = raw_frame_start_timestamps_us[i]
                if i < len(frames) - 1:
                    # take next start-of-spin time as current end-of-spin time
                    frame_end_timestamp_us = raw_frame_start_timestamps_us[i + 1]
                else:
                    # approximate last end-of-spin time
                    frame_end_timestamp_us = raw_frame_start_timestamps_us[i] + (
                        raw_frame_start_timestamps_us[i] - raw_frame_start_timestamps_us[i - 1]
                    )

                timestamps_us = np.array([frame_start_timestamp_us, frame_end_timestamp_us], dtype=np.uint64)

                # Extract the range image and corresponding poses for all rays
                range_image, segmentation_ri, range_image_top_pose = parse_range_image_and_segmentations(
                    frame, lidar_id, ri_index=0
                )

                range_image_second, _, _ = parse_range_image_and_segmentations(
                    frame, lidar_id, ri_index=1, range_image_top_pose=range_image_top_pose
                )

                # Convert both range image returns to point clouds in a single batched pass
                pc_first, pc_second = convert_range_images_to_point_clouds(
                    frame,
                    lidar_id,
                    [
                        unpack_optional(range_image, msg=f"Missing range image for laser {lidar_id}"),
                        unpack_optional(
                            range_image_second, msg=f"Missing second-return range image for laser {lidar_id}"
                        ),
                    ],
                    segmentation_ri,
                    unpack_optional(range_image_top_pose, msg=f"Missing top-lidar pose for laser {lidar_id}"),
                    timestamps_us,
                )
                points_world = pc_first.points
                segmentation = pc_first.segmentation
                point_timestamps_us = pc_first.timestamps
                range_image_indices = pc_first.range_image_indices
                inclinations_rad = pc_first.inclinations
                azimuths_rad = pc_first.azimuths

                points_world_second = pc_second.points
                range_image_indices_second = pc_second.range_image_indices

                del (range_image, range_image_second, range_image_top_pose)

                # perform primary <-> secondary ray matching via linear indices
                # (as every secondary ray has a parent primary ray)
                range_image_width = azimuths_rad.size
                linear_indices_primary = range_image_indices[:, 0] + range_image_indices[:, 1] * range_image_width
                linear_indices_second = (
                    range_image_indices_second[:, 0] + range_image_indices_second[:, 1] * range_image_width
                )

                # Build a hash map from primary linear index -> position for O(N+S) matching
                primary_index_map = np.empty(linear_indices_primary.max() + 1, dtype=np.intp)
                primary_index_map[linear_indices_primary] = np.arange(len(linear_indices_primary))
                primary_indices = primary_index_map[linear_indices_second]  # S

                # Pick semantic_class if available in current frame
                semantic_class = segmentation[:, 1].astype(np.int8) if (segmentation is not None) else None  # N

                # Interpolate poses
                T_rig_worlds = self.pose_interpolator.interpolate_to_timestamps(timestamps_us)

                ## Determine more accurate timestamps if possible (i.e., if there is sufficient motion)
                #  by undoing the data's pose interpolation
                #  (see https://github.com/waymo-research/waymo-open-dataset/issues/619#issuecomment-2830136376)
                P_sensor_startend_world = (T_rig_worlds @ T_sensor_rig[None])[
                    :, :3, 3
                ]  # start/end positions of the sensor in world frame
                if (
                    # only perform timestamp inference if there is sufficient motion
                    np.linalg.norm(
                        # vector between start/end positions of the sensor in world frame
                        V := P_sensor_startend_world[1] - P_sensor_startend_world[0]
                    )
                    > self.FINE_LIDAR_TIMESTAMPS_MINIMUM_SENSOR_MOTION_THRESHOLD_M
                ):
                    # compute the relative time parameter for each lidar point (given by it's
                    # time-interpolated ray start points) relative to the start/end points of the sensor
                    W = points_world[:, :3] - P_sensor_startend_world[0]

                    # compute the relative time parameter of the orthogonal projection of the ray's start
                    # point onto the line between start/end points of the sensor, and clip to make sure
                    # to stay within frame time bounds
                    t = np.dot(W, V) / np.dot(V, V)
                    t = np.clip(t, 0, 1)

                    # use per-point relative time parameter to interpolate more accurate per point absolute
                    # timestamps
                    point_timestamps_us = (timestamps_us[0] + t * (timestamps_us[1] - timestamps_us[0])).astype(
                        np.uint64
                    )

                # Bring point-cloud data into the right format
                T_world_sensor_start = se3_inverse(T_sensor_rig) @ se3_inverse(T_rig_worlds[0])
                T_world_sensor_end = se3_inverse(T_sensor_rig) @ se3_inverse(T_rig_worlds[1])

                xyz_e = transform_point_cloud(points_world[:, 3:6], T_world_sensor_end).astype(np.float32)  # N x 3

                # Undo motion-compensation for ray bundle direction and distance computation
                xyz_m = motion_compensator.motion_decompensate_points(
                    sensor_id=lidar_ncore_id,
                    xyz_sensorend=xyz_e,
                    timestamp_us=point_timestamps_us,
                    frame_start_timestamp_us=frame_start_timestamp_us,
                    frame_end_timestamp_us=frame_end_timestamp_us,
                )

                # Combine first and second return data
                distance_m = np.full(
                    (2, xyz_m.shape[0]), np.nan, dtype=np.float32
                )  # [R = n-returnes = 2, N = n-points]
                distance_m[0, :] = np.linalg.norm(xyz_m, axis=1)  # N
                distance_m[1, primary_indices] = np.linalg.norm(
                    points_world_second[:, 3:6] - points_world_second[:, :3], axis=1
                )  # S
                direction = xyz_m / distance_m[0, :, np.newaxis]

                # normalize intensity (https://github.com/ouster-lidar/ouster_example/issues/488) to [0,1]
                intensity = np.full(
                    (2, points_world.shape[0]), np.nan, dtype=np.float32
                )  # [R = n-returnes = 2, N = n-points]
                intensity[0, :] = np.tanh(points_world[:, 6])  # N
                intensity[1, primary_indices] = np.tanh(points_world_second[:, 6])  # S

                elongation = np.full(
                    (2, points_world.shape[0]), np.nan, dtype=np.float32
                )  # [R = n-returnes = 2, N = n-points]
                elongation[0, :] = points_world[:, 7]  # N
                elongation[1, primary_indices] = points_world_second[:, 7]  # S

                # Process frame labels (defined in frame-associated rig frame)
                T_rig_labelstime_world = (
                    np.array(frame.pose.transform, dtype=np.float64).reshape(4, 4).astype(np.float32)
                )
                T_rig_labelstime_sensor_end = T_world_sensor_end @ T_rig_labelstime_world

                # Initialize intrinsics and associated lookup tables
                if lidar_model_parameters is None:
                    # Normalize azimuth angle ranges to (-pi, pi]
                    azimuths_rad[azimuths_rad > np.pi] -= 2 * np.pi
                    azimuths_rad[azimuths_rad <= -np.pi] += 2 * np.pi

                    lidar_model_parameters = RowOffsetStructuredSpinningLidarModelParameters(
                        spinning_frequency_hz=10.0,
                        spinning_direction="cw",
                        n_rows=len(inclinations_rad),
                        n_columns=len(azimuths_rad),
                        row_elevations_rad=inclinations_rad,
                        column_azimuths_rad=azimuths_rad,
                        row_azimuth_offsets_rad=np.zeros_like(inclinations_rad, dtype=np.float32),  # no row offsets
                    )

                    # Initialize the lidar model
                    lidar_model = RowOffsetStructuredSpinningLidarModel(
                        lidar_model_parameters,
                        angles_to_columns_map_init=False,
                        angles_to_columns_map_resolution_factor=3,
                        device="cpu",  # we only ever project a single cuboid centroid in each call,
                        # which is faster via CPU
                    )

                if lidar_horizontal_fov is None:
                    lidar_horizontal_fov = lidar_model_parameters.get_horizontal_fov()

                assert lidar_model is not None
                del (azimuths_rad, inclinations_rad)

                for raw_frame_label in raw_frame_labels[frame.timestamp_micros]:
                    # Map label in rig space to frame label in sensor space
                    bbox3_sensor = BBox3.from_array(
                        transform_bbox(
                            raw_frame_label.bbox3.to_array(),
                            T_rig_labelstime_sensor_end,
                        )
                    )

                    # Try to determine timestamp from projecting into the rolling-shutter-aware lidar model
                    bbox_centroid_sensor_angle = lidar_model.world_points_to_sensor_angles_shutter_pose(
                        world_points=transform_point_cloud(
                            np.asarray([bbox3_sensor.centroid]),
                            se3_inverse(T_world_sensor_end),
                        ),
                        T_world_sensor_start=T_world_sensor_start,
                        T_world_sensor_end=T_world_sensor_end,
                        start_timestamp_us=timestamps_us[0],
                        end_timestamp_us=timestamps_us[1],
                        return_timestamps=True,
                    )

                    # If projection succeeded use it's timestamp, otherwise fall back to angle-based approximation
                    if bbox_centroid_sensor_angle.timestamps_us is not None and len(
                        bbox_centroid_sensor_angle.timestamps_us
                    ):
                        frame_label_timestamp = int(bbox_centroid_sensor_angle.timestamps_us.item())
                    else:
                        # Approximate measurement time by azimuth angle of centroid in sensor's x/y plane
                        # and performing linear interpolation between start / end times
                        azimuth_rad = np.arctan2(bbox3_sensor.centroid[1], bbox3_sensor.centroid[0])

                        relative_azimuth = relative_angle(
                            lidar_horizontal_fov.start_rad,
                            azimuth_rad,
                            lidar_horizontal_fov.direction,
                        )
                        bbox_relative_time = relative_azimuth.relative_angle_rad.item() / lidar_horizontal_fov.span_rad

                        # It can happen that the bbox centroid is slightly outside the FOV (e.g if only one
                        # side of the object is measured), so we clip it to [0, 1]
                        bbox_relative_time = np.clip(bbox_relative_time, 0, 1)

                        frame_label_timestamp = int(timestamps_us[0]) + int(
                            bbox_relative_time * (timestamps_us[1] - timestamps_us[0])
                        )

                    cuboid_track_observation = CuboidTrackObservation(
                        track_id=raw_frame_label.track_id,
                        class_id=raw_frame_label.label_class,
                        timestamp_us=frame_label_timestamp,
                        reference_frame_id=lidar_ncore_id,
                        reference_frame_timestamp_us=frame_end_timestamp_us,
                        bbox3=bbox3_sensor,
                        source=LabelSource.EXTERNAL,
                    )

                    cuboid_track_observations.append(cuboid_track_observation)

                ## Determine model indices from range-image indices, applying azimuth reordering
                model_element = range_image_indices.reshape((-1, 2)).astype(np.uint16)

                # Serialize lidar frame
                lidar_writer.store_frame(
                    # Non-motion-compensated per-ray 3D directions (float32, [N, 3])
                    direction=direction,
                    # Per-point timestamp in microseconds (uint64, [N])
                    timestamp_us=point_timestamps_us,
                    # Per-point model element indices (uint16, [N, 2])
                    model_element=model_element,
                    # Per-point distance (two returns, [2, N])
                    distance_m=distance_m,
                    # Per-point intensity normalized to [0.0, 1.0] (float32, [2, N])
                    intensity=intensity,
                    # Frame start/end timestamps (uint64, [2])
                    frame_timestamps_us=np.array(
                        [frame_start_timestamp_us, frame_end_timestamp_us],
                        dtype=np.uint64,
                    ),
                    generic_data={"elongation": elongation}  # [2, N]
                    | ({"semantic_class": semantic_class} if semantic_class is not None else {}),  # [N]
                    generic_meta_data={},
                )

            # Store intrinsics
            self.intrinsics_writer.store_lidar_intrinsics(
                lidar_id=lidar_ncore_id,
                lidar_model_parameters=unpack_optional(lidar_model_parameters),
            )

            # Store extrinsics (sensor->rig transform)
            self.poses_writer.store_static_pose(
                source_frame_id=lidar_ncore_id,
                target_frame_id="rig",
                pose=T_sensor_rig,
            )

        # Store cuboid track observations

        self.store_writer.register_component_writer(
            CuboidsComponent.Writer,
            "default",
            self.component_groups.cuboid_track_observations_component_group,
        ).store_observations(cuboid_track_observations)

        self.logger.info(f"Stored {len(cuboid_track_observations)} cuboid observations")

    # Semantic classes labels for camera segmentation
    CAMERA_LABEL_CLASS_ID_STRING_MAP: Dict[int, str] = {
        camera_segmentation_pb2.CameraSegmentation.Type.TYPE_UNDEFINED: "UNDEFINED",
        # The Waymo vehicle.
        camera_segmentation_pb2.CameraSegmentation.Type.TYPE_EGO_VEHICLE: "EGO_VEHICLE",
        # Small vehicle such as a sedan, SUV, pickup truck, minivan or golf cart.
        camera_segmentation_pb2.CameraSegmentation.Type.TYPE_CAR: "CAR",
        # Large vehicle that carries cargo.
        camera_segmentation_pb2.CameraSegmentation.Type.TYPE_TRUCK: "TRUCK",
        # Large vehicle that carries more than 8 passengers.
        camera_segmentation_pb2.CameraSegmentation.Type.TYPE_BUS: "BUS",
        # Large vehicle that is not a truck or a bus.
        camera_segmentation_pb2.CameraSegmentation.Type.TYPE_OTHER_LARGE_VEHICLE: "OTHER_LARGE_VEHICLE",
        # Bicycle with no rider.
        camera_segmentation_pb2.CameraSegmentation.Type.TYPE_BICYCLE: "BICYCLE",
        # Motorcycle with no rider.
        camera_segmentation_pb2.CameraSegmentation.Type.TYPE_MOTORCYCLE: "MOTORCYCLE",
        # Trailer attached to another vehicle or horse.
        camera_segmentation_pb2.CameraSegmentation.Type.TYPE_TRAILER: "TRAILER",
        # Pedestrian. Does not include objects associated with the pedestrian, such as suitcases, strollers or cars.
        camera_segmentation_pb2.CameraSegmentation.Type.TYPE_PEDESTRIAN: "PEDESTRIAN",
        # Bicycle with rider.
        camera_segmentation_pb2.CameraSegmentation.Type.TYPE_CYCLIST: "CYCLIST",
        # Motorcycle with rider.
        camera_segmentation_pb2.CameraSegmentation.Type.TYPE_MOTORCYCLIST: "MOTORCYCLIST",
        # Birds, including ones on the ground.
        camera_segmentation_pb2.CameraSegmentation.Type.TYPE_BIRD: "BIRD",
        # Animal on the ground such as a dog, cat, cow, etc.
        camera_segmentation_pb2.CameraSegmentation.Type.TYPE_GROUND_ANIMAL: "GROUND_ANIMAL",
        # Cone or short pole related to construction.
        camera_segmentation_pb2.CameraSegmentation.Type.TYPE_CONSTRUCTION_CONE_POLE: "CONSTRUCTION_CONE_POLE",
        # Permanent horizontal and vertical lamp pole, traffic sign pole, etc.
        camera_segmentation_pb2.CameraSegmentation.Type.TYPE_POLE: "POLE",
        # Large object carried/pushed/dragged by a pedestrian.
        camera_segmentation_pb2.CameraSegmentation.Type.TYPE_PEDESTRIAN_OBJECT: "PEDESTRIAN_OBJECT",
        # Sign related to traffic, including front and back facing signs.
        camera_segmentation_pb2.CameraSegmentation.Type.TYPE_SIGN: "SIGN",
        # The box that contains traffic lights regardless of front or back facing.
        camera_segmentation_pb2.CameraSegmentation.Type.TYPE_TRAFFIC_LIGHT: "TRAFFIC_LIGHT",
        # Permanent building and walls, including solid fences.
        camera_segmentation_pb2.CameraSegmentation.Type.TYPE_BUILDING: "BUILDING",
        # Drivable road with proper markings, including parking lots and gas stations.
        camera_segmentation_pb2.CameraSegmentation.Type.TYPE_ROAD: "ROAD",
        # Marking on the road that is parallel to the ego vehicle and defines lanes.
        camera_segmentation_pb2.CameraSegmentation.Type.TYPE_LANE_MARKER: "LANE_MARKER",
        # All markings on the road other than lane markers.
        camera_segmentation_pb2.CameraSegmentation.Type.TYPE_ROAD_MARKER: "ROAD_MARKER",
        # Paved walkable surface for pedestrians, including curbs.
        camera_segmentation_pb2.CameraSegmentation.Type.TYPE_SIDEWALK: "SIDEWALK",
        # Vegetation including tree trunks, tree branches, bushes, tall grasses, flowers and so on.
        camera_segmentation_pb2.CameraSegmentation.Type.TYPE_VEGETATION: "VEGETATION",
        # The sky, including clouds.
        camera_segmentation_pb2.CameraSegmentation.Type.TYPE_SKY: "SKY",
        # Other horizontal surfaces that are drivable or walkable.
        camera_segmentation_pb2.CameraSegmentation.Type.TYPE_GROUND: "GROUND",
        # Object that is not permanent in its current position and does not belong to any of the above classes.
        camera_segmentation_pb2.CameraSegmentation.Type.TYPE_DYNAMIC: "DYNAMIC",
        # Object that is permanent in its current position and does not belong to any of the above classes.
        camera_segmentation_pb2.CameraSegmentation.Type.TYPE_STATIC: "STATIC",
    }

    def decode_cameras(self, frames: list[dataset_pb2.Frame]) -> None:
        """
        Extracts the images and camera metadata for all cameras within a single frame.
        Camera metadata must hold the information used to compensate for rolling shutter
        effect and to convert RGB images to 3D RGB rays in space.
        """
        calibrations = {c.name: c for c in frames[0].context.camera_calibrations}

        for camera_id, camera_ncore_id in self.CAMERA_MAP.items():
            if camera_ncore_id not in self.camera_ids:
                continue  # skip sensor if not active

            ## Get the calibration data
            calibration = calibrations[camera_id]

            T_sensor_rig = np.array(
                tf.reshape(
                    tf.constant(calibration.extrinsic.transform, dtype=tf.float32),
                    [4, 4],
                ),
                dtype=np.float32,
            )

            ## Fix camera frame convention from
            # - waymo camera: principal axis along +x axis, y-axis points left, z-axis points up
            # to
            # - NCore camera: principal axis along +z axis, x-axis points right, y-axis points down
            T_sensor_rig[:3, :3] = T_sensor_rig[:3, :3] @ np.array(
                [[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float32
            )

            # Create camera sensor writer
            camera_writer = self.store_writer.register_component_writer(
                CameraSensorComponent.Writer,
                component_instance_name=camera_ncore_id,
                group_name=self.component_groups.camera_component_groups.get(camera_ncore_id),
                generic_meta_data={
                    "label-class-string-id-map": {
                        label_string: label_id
                        for label_id, label_string in self.CAMERA_LABEL_CLASS_ID_STRING_MAP.items()
                    }
                },
            )

            for frame in tqdm.tqdm(frames, desc=f"Process {camera_ncore_id}"):
                ## Load current camera's image
                image = {image.name: image for image in frame.images}[camera_id]

                ## Get frame timestamps
                frame_start_timestamp_us = int((image.camera_trigger_time + image.shutter / 2) * 1e6)
                frame_end_timestamp_us = int((image.camera_readout_done_time - image.shutter / 2) * 1e6)

                # Assert we can interpolate poses for these timestamps
                self.pose_interpolator.interpolate_to_timestamps(
                    np.array([frame_start_timestamp_us, frame_end_timestamp_us])
                )

                # Parse panoptic segmentation, if available
                generic_data: Dict[str, np.ndarray] = {}
                generic_meta_data: Dict[str, JsonLike] = {}

                if (
                    hasattr(image, "camera_segmentation_label")
                    and hasattr(
                        camera_segmentation_label := image.camera_segmentation_label,
                        "panoptic_label_divisor",
                    )
                    and (panoptic_label_divisor := camera_segmentation_label.panoptic_label_divisor) > 0
                    and hasattr(camera_segmentation_label, "panoptic_label")
                ):
                    # Store the original waymo png segmentation data
                    generic_data["panoptic_label_png"] = np.frombuffer(
                        camera_segmentation_label.panoptic_label, dtype=np.uint8
                    )
                    generic_meta_data["panoptic_label_divisor"] = panoptic_label_divisor

                # Store the image and its metadata
                camera_writer.store_frame(
                    image_binary_data=image.image,
                    image_format="jpeg",
                    frame_timestamps_us=np.array(
                        [frame_start_timestamp_us, frame_end_timestamp_us],
                        dtype=np.uint64,
                    ),
                    generic_data=generic_data,
                    generic_meta_data=generic_meta_data,
                )

            # Extract intrinsic data
            width = calibration.width
            height = calibration.height
            f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3 = calibration.intrinsic[:]
            match calibration.rolling_shutter_direction:
                case dataset_pb2.CameraCalibration.TOP_TO_BOTTOM:
                    rolling_shutter_direction = ShutterType.ROLLING_TOP_TO_BOTTOM
                case dataset_pb2.CameraCalibration.LEFT_TO_RIGHT:
                    rolling_shutter_direction = ShutterType.ROLLING_LEFT_TO_RIGHT
                case dataset_pb2.CameraCalibration.BOTTOM_TO_TOP:
                    rolling_shutter_direction = ShutterType.ROLLING_BOTTOM_TO_TOP
                case dataset_pb2.CameraCalibration.RIGHT_TO_LEFT:
                    rolling_shutter_direction = ShutterType.ROLLING_RIGHT_TO_LEFT
                case dataset_pb2.CameraCalibration.GLOBAL_SHUTTER:
                    rolling_shutter_direction = ShutterType.GLOBAL
                case _:
                    raise TypeError(f"unsupported shutter direction {calibration.rolling_shutter_direction}")

            # Store intrinsics
            self.intrinsics_writer.store_camera_intrinsics(
                camera_id=camera_ncore_id,
                camera_model_parameters=OpenCVPinholeCameraModelParameters(
                    resolution=np.array([width, height], dtype=np.uint64),
                    shutter_type=rolling_shutter_direction,
                    external_distortion_parameters=None,
                    principal_point=np.array([c_u, c_v], dtype=np.float32),
                    focal_length=np.array([f_u, f_v], dtype=np.float32),
                    radial_coeffs=np.array([k1, k2, k3, 0, 0, 0], dtype=np.float32),
                    tangential_coeffs=np.array([p1, p2], dtype=np.float32),
                    thin_prism_coeffs=np.array([0, 0, 0, 0], dtype=np.float32),
                ),
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
@click.pass_context
def waymo_v4(ctx, *_, **kwargs):
    """Waymo-specific data conversion (V4 format)"""

    # Extend base config with command-specific options
    config = WaymoConverter4Config(**{**asdict(ctx.obj), **kwargs})

    WaymoConverter4.convert(config)


if __name__ == "__main__":
    cli(show_default=True)
