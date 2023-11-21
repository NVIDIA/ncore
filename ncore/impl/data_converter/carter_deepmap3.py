# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field

import logging
import os

from pathlib import Path
from typing import Optional

import tqdm
import numpy as np

from google.protobuf import text_format
from protobuf_to_dict import protobuf_to_dict

from ncore.impl.data_converter.protos.deepmap import track_data_pb2, pointcloud_pb2, camera_calibration_pb2
from ncore.impl.data_converter.protos.deepmap.util import extract_sensor_2_sdc
from ncore.impl.data_converter.data_converter import DataConverter
from ncore.impl.data.data3 import ContainerDataWriter
from ncore.impl.data.types import (
    FrameLabel3,
    Poses,
    OpenCVPinholeCameraModelParameters,
    ShutterType,
    TrackLabel,
    Tracks,
)
from ncore.impl.common.common import PoseInterpolator
from ncore.impl.common.nvidia_utils import LabelProcessor, extract_pose
from ncore.impl.av_utils import isWithin3DBBox


class CarterDeepmapConverter(DataConverter):
    """
    NVIDIA-specific data converter (based on DeepMap tracks)
    """

    # Sensor specifications
    CAMERA_SENSOR_IDS = ["camera00", "camera01"]
    LIDAR_SENSOR_IDS = ["lidar00"]
    LIDAR_FILTER_MAX_DISTANCE_METERS = 150.0

    @dataclass
    class SensorData:
        frame_timestamps_us: list[int] = field(default_factory=list)
        frame_file_paths: list[str] = field(default_factory=list)
        extrinsics: np.ndarray = np.eye(4)
        intrinsics: Optional[camera_calibration_pb2.MonoCalibrationParameters] = None

    def __init__(self, config):
        super().__init__(config)

        self.logger = logging.getLogger(__name__)

        self.start_timestamp_us = config.start_timestamp_us
        self.end_timestamp_us = config.end_timestamp_us

    @staticmethod
    def get_sequence_paths(config) -> list[Path]:
        return [Path(config.root_dir)]

    @staticmethod
    def from_config(config) -> CarterDeepmapConverter:
        return CarterDeepmapConverter(config)

    def convert_sequence(self, sequence_path: Path) -> None:
        """
        Runs the conversion of a single sequence
        """

        self.sequence_path = sequence_path
        self.sequence_name = sequence_path.name

        for track_dir in [subdir for subdir in sequence_path.iterdir() if subdir.is_dir() and subdir.name.isdigit()]:
            self.track_dir = track_dir
            self.track_name = track_dir.name

            self.logger.info(f"Processing track {self.track_name}")

            # ContainerDataWriter for all outputs (always single-shard)
            self.data_writer = ContainerDataWriter(
                self.output_dir / f"{self.sequence_name}-{self.track_name}",
                f"{self.sequence_name}-{self.track_name}",
                self.get_active_camera_ids(self.CAMERA_SENSOR_IDS),
                self.get_active_lidar_ids(self.LIDAR_SENSOR_IDS),
                self.get_active_radar_ids([]),  # no radars yet
                # TODO: parse these from the data
                "carter",
                "deepmap",
                f"{self.sequence_name}-{self.track_name}",
                # always single-shard
                0,
                1,
                False,
            )

            # Initialize the track aligned track record structure
            self.track_data = track_data_pb2.AlignedTrackRecords()

            # Read in the track record data from a proto file
            # This includes camera_records and lidar_records (see track_record proto for more detail)
            with open(
                os.path.join(track_dir, "aligned_track_records_segment_from0_to18446744073709551615.pb.txt"), "r"
            ) as f:
                text_format.Parse(f.read(), self.track_data)

            # Extract all the lidar paths, timestamps and poses from the track record
            self.track_data = protobuf_to_dict(self.track_data)

            self.decode_track()

            self.decode_labels()

            self.decode_lidar()

            self.decode_cameras()

            # Store per-shard meta data / final success state / close file
            self.data_writer.finalize()

    def decode_track(self):
        # Poses are represented as deepmap SDC-to-world

        # Extract all relevant track data, which are converted to the nvidia convention
        # extract_pose() extracts the T_sdc_world - we interpret these as T_rig_world
        self.sensor_datas: dict[str, CarterDeepmapConverter.SensorData] = {}

        self.poses = []
        self.poses_timestamps_us = []

        def decode_record(sensor_type, sensor_record):
            sensor_data = CarterDeepmapConverter.SensorData()

            for frame in sensor_record["records"]:
                sensor_data.frame_timestamps_us.append(frame["timestamp_microseconds"])
                sensor_data.frame_file_paths.append(frame["file_path"])

                if "pose" in frame:
                    self.poses_timestamps_us.append(frame["timestamp_microseconds"])
                    self.poses.append(extract_pose(frame["pose"]))

            sensor_data.extrinsics = extract_sensor_2_sdc(
                self.sequence_path / sensor_record[f"{sensor_type}_to_vehicle_transform_path"]
            ).astype(np.float32)

            if sensor_type == "camera":
                sensor_data.intrinsics = camera_calibration_pb2.MonoCalibrationParameters()
                with open(self.sequence_path / sensor_record["mono_calibration_parameters_path"], "r") as f:
                    text_format.Parse(f.read(), sensor_data.intrinsics)

            return sensor_data

        # Decode poses and sensor-specific data
        for sensor_id, sensor_record in zip(
            CarterDeepmapConverter.LIDAR_SENSOR_IDS,
            self.track_data["lidar_records"],
        ):
            if sensor_id in self.data_writer.lidar_ids:
                self.sensor_datas[sensor_id] = decode_record("lidar", sensor_record)
        for sensor_id, sensor_record in zip(
            CarterDeepmapConverter.CAMERA_SENSOR_IDS,
            self.track_data["camera_records"],
        ):
            if sensor_id in self.data_writer.camera_ids:
                self.sensor_datas[sensor_id] = decode_record("camera", sensor_record)

        # Stack, sort the poses and make them unique
        self.poses = np.stack(self.poses)
        self.poses_timestamps_us = np.stack(self.poses_timestamps_us).astype(np.uint64)
        self.poses_timestamps_us, unique_idx = np.unique(self.poses_timestamps_us, return_index=True)
        self.poses = self.poses[unique_idx]

        # Select base-pose
        base_pose = self.poses[0]

        # Convert the poses to the sequence coordinate frame
        self.poses = np.linalg.inv(base_pose) @ self.poses

        # Subselect poses in case timestamp ranges were provided (only subselect serialized poses, keep all poses for pose interpolation of frame data)
        start_timestamp_us = self.start_timestamp_us if self.start_timestamp_us else self.poses_timestamps_us[0]
        end_timestamp_us = self.end_timestamp_us if self.end_timestamp_us else self.poses_timestamps_us[-1]

        local_pose_range = np.logical_and(
            start_timestamp_us <= self.poses_timestamps_us, self.poses_timestamps_us <= end_timestamp_us
        )

        # Save the poses
        self.data_writer.store_poses(
            Poses(
                T_rig_world_base=base_pose,
                T_rig_worlds=self.poses[local_pose_range],
                T_rig_world_timestamps_us=self.poses_timestamps_us[local_pose_range],
            )
        )

    def decode_labels(self):
        # No labels / tracks to load currently
        self.track_labels: dict[str, TrackLabel] = {}
        self.frame_labels: dict[str, dict[int, list[FrameLabel3]]] = {}
        self.track_global_dynamic_flag: dict[str, bool] = {}

        self.data_writer.store_tracks(Tracks(self.track_labels))

    def decode_lidar(self):
        # Initialize the pose interpolator object
        pose_interpolator = PoseInterpolator(self.poses, self.poses_timestamps_us)

        for lidar_id in self.data_writer.lidar_ids:
            sensor_data = self.sensor_datas[lidar_id]

            # Extrinsics
            T_lidar_rig = sensor_data.extrinsics

            # Hardcoded vehicle bounding box
            bbox_centroid = T_lidar_rig[:3, 3]  # center box around lidar sensor
            bbox_dimensions = np.array([0.75, 0.75, 0.75], dtype=np.float32)
            bbox_orientation = np.zeros(
                3, dtype=np.float32
            )  # vehicle bbox is aligned with the rig, i.e., is an axis-aligned bbox

            vehicle_bbox_rig = np.hstack([bbox_centroid, bbox_dimensions, bbox_orientation])

            lidar_end_timestamps = []
            frame_idx = 0
            for frame_file_path in tqdm.tqdm(sensor_data.frame_file_paths):
                # Load the point cloud data
                data = pointcloud_pb2.PointCloud()
                with open(self.sequence_path / frame_file_path, "rb") as f:
                    data.ParseFromString(f.read())

                frame_end_timestamp = data.meta_data.end_timestamp_microseconds

                if self.start_timestamp_us and frame_end_timestamp < self.start_timestamp_us:
                    continue  # not there yet
                if self.end_timestamp_us and frame_end_timestamp > self.end_timestamp_us:
                    break  # already passed end - no need to keep on processing

                raw_pc = np.concatenate(
                    [
                        np.array(data.data.points_x, dtype=np.float32)[:, None],
                        np.array(data.data.points_y, dtype=np.float32)[:, None],
                        np.array(data.data.points_z, dtype=np.float32)[:, None],
                    ],
                    axis=1,
                )

                # [0 .. 255] -> [0.0 .. 1.0]
                intensities = np.frombuffer(data.data.intensities, dtype=np.uint8).astype(np.float32) / 255.0

                # Determine per-column rig-to-world pose and compute per-column lidar-to-world transformations
                column_timestamps = np.array(data.data.column_timestamps_microseconds, dtype=np.uint64)
                try:
                    # First / last frame might not work as the pose is available only at the middle of the frame
                    column_poses = pose_interpolator.interpolate_to_timestamps(column_timestamps)
                except Exception as e:  # work on python 3.x
                    self.logger.warn(
                        f"Lidar frame conversion failed for lidar frame {frame_idx} due to out-of-egomotion pose: {e}"
                    )
                    continue
                T_column_lidar_worlds = column_poses @ T_lidar_rig[None, :, :]
                # Pose of the rig at the end of the lidar spin, can be used to transform points into a local coordinate frame
                T_rig_world = column_poses[-1]
                T_world_rig = np.linalg.inv(T_rig_world)
                T_world_lidar = np.linalg.inv(T_lidar_rig) @ T_world_rig

                # Perform per-column unwinding, transforming from lidar to world coordinates
                transformed_pc = np.empty((len(raw_pc), 6), dtype=np.float32)
                transformed_pc[:, :3] = T_column_lidar_worlds[
                    data.data.column_indices, :3, -1
                ]  # N X 3 - ray start points in world space
                transformed_pc[:, 3:] = (
                    T_column_lidar_worlds[data.data.column_indices, :3, :3] @ raw_pc[:, :, None]
                ).squeeze(-1) + transformed_pc[
                    :, :3
                ]  # N x 3 - ray end points in world space

                pc_world_homogeneous = np.row_stack(
                    [transformed_pc[:, 3:6].transpose(), np.ones(transformed_pc.shape[0], dtype=np.float32)]
                )  # 4 x N
                pc_rig = (T_world_rig @ pc_world_homogeneous)[:-1, :].transpose()  # N x 3
                pc_lidar = (T_world_lidar @ pc_world_homogeneous)[:-1, :].transpose()  # N x 3

                pc_start_world_homogeneous = np.row_stack(
                    [transformed_pc[:, 0:3].transpose(), np.ones(transformed_pc.shape[0], dtype=np.float32)]
                )  # 4 x N
                pc_start_lidar = (T_world_lidar @ pc_start_world_homogeneous)[:-1, :].transpose()  # N x 3

                timestamps = column_timestamps[data.data.column_indices]

                # Filter outs points that are inside the vehicles bounding-box
                valid_idxs_vehicle_bbox = np.logical_not(
                    isWithin3DBBox(pc_rig.astype(np.float32), vehicle_bbox_rig.reshape(1, -1))
                )

                # Filter points based on distances
                dist = np.linalg.norm(transformed_pc[:, :3] - transformed_pc[:, 3:6], axis=1)

                # Filter points on the distances LIDAR_FILTER_MAX_DISTANCE (remove points that are very far away)
                valid_idx_dist = np.less_equal(dist, self.LIDAR_FILTER_MAX_DISTANCE_METERS)
                valid_idx = np.logical_and(valid_idxs_vehicle_bbox, valid_idx_dist)

                # Subselect to valid points
                xyz_e = pc_lidar[valid_idx, :].astype(np.float32)
                xyz_s = pc_start_lidar[valid_idx, :].astype(np.float32)
                intensity = intensities[valid_idx]
                timestamp = timestamps[valid_idx]

                # Interpolate start / end pose
                timestamps_us = np.array([np.min(timestamp), frame_end_timestamp], dtype=np.uint64)
                T_rig_worlds = pose_interpolator.interpolate_to_timestamps(timestamps_us)

                # Use the bounding boxes to remove dynamic objects
                dynamic_flag, frame_labels = LabelProcessor.lidar_dynamic_flag(
                    lidar_id, xyz_e, -1, self.frame_labels, self.track_global_dynamic_flag
                )

                # Serialize lidar frame
                self.data_writer.store_lidar_frame(
                    lidar_id,
                    frame_idx,
                    xyz_s,
                    xyz_e,
                    intensity,
                    timestamp,
                    dynamic_flag,
                    frame_labels,
                    T_rig_worlds,
                    timestamps_us,
                    {},
                    {},
                )

                # Save the end time stamp of the lidar spin
                lidar_end_timestamps.append(frame_end_timestamp)

                frame_idx += 1

            # Save sensor meta data
            self.data_writer.store_lidar_meta(
                lidar_id=lidar_id,
                frame_timestamps_us=np.array(lidar_end_timestamps, dtype=np.uint64),
                T_sensor_rig=T_lidar_rig,
                generic_meta_data={},
            )

    def decode_cameras(self):
        # Pose interpolator to obtain start / end egomotion poses
        pose_interpolator = PoseInterpolator(self.poses, self.poses_timestamps_us)

        for camera_id in self.data_writer.camera_ids:
            sensor_data = self.sensor_datas[camera_id]

            # Extrinsics
            T_camera_rig = sensor_data.extrinsics

            # Intrinsics
            camera_matrix = np.array(sensor_data.intrinsics.camera_matrix.data, dtype=np.float32).reshape((3, 3))
            camera_model = OpenCVPinholeCameraModelParameters(
                resolution=np.array(
                    [sensor_data.intrinsics.image_width, sensor_data.intrinsics.image_height], dtype=np.uint64
                ),
                shutter_type=ShutterType.GLOBAL,
                principal_point=camera_matrix[:2, 2],
                focal_length=camera_matrix[np.diag_indices(2)],
                # Images are rectified already -> identity distortion
                radial_coeffs=np.zeros((6,), dtype=np.float32),
                tangential_coeffs=np.zeros((2,), dtype=np.float32),
                thin_prism_coeffs=np.zeros((4,), dtype=np.float32),
            )

            frame_end_timestamps_us = []
            continuous_frame_idx = 0
            for frame_file_path, frame_timestamp_us in tqdm.tqdm(
                zip(sensor_data.frame_file_paths, sensor_data.frame_timestamps_us),
                total=len(sensor_data.frame_file_paths),
            ):

                if frame_timestamp_us < self.poses_timestamps_us[0]:
                    continue  # no pose for this frame yet
                if frame_timestamp_us > self.poses_timestamps_us[-1]:
                    break  # no pose for this frame anymore

                if self.start_timestamp_us and frame_timestamp_us < self.start_timestamp_us:
                    continue  # not there yet
                if self.end_timestamp_us and frame_timestamp_us > self.end_timestamp_us:
                    break  # already passed end - no need to keep on processing

                # Load the image frame data
                with open(frame_file_path := self.sequence_path / frame_file_path, "rb") as f:
                    image_file_binary_data = f.read()

                # Interpolate the common start/end pose to the frame's single timestamp (global-shutter)
                timestamps_us = np.array([frame_timestamp_us, frame_timestamp_us], dtype=np.uint64)
                T_rig_worlds = pose_interpolator.interpolate_to_timestamps(timestamps_us)

                self.data_writer.store_camera_frame(
                    camera_id,
                    continuous_frame_idx,
                    image_file_binary_data,
                    "jpeg",
                    T_rig_worlds,
                    timestamps_us,
                    {},
                    {},
                )

                frame_end_timestamps_us.append(frame_timestamp_us)  # single timestamp is end-of-frame timestamp

                continuous_frame_idx += 1

            self.data_writer.store_camera_meta(
                camera_id,
                frame_timestamps_us=np.array(frame_end_timestamps_us, dtype=np.uint64),
                T_sensor_rig=T_camera_rig,
                camera_model_parameters=camera_model,
                mask_image=None,
                generic_meta_data={},
            )
