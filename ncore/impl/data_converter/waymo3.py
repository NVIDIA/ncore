# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

import logging

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import tqdm

import tensorflow.compat.v1 as tf

from waymo_open_dataset import dataset_pb2, label_pb2

from ncore.impl.av_utils import isWithin3DBBox
from ncore.impl.data.data3 import ContainerDataWriter
from ncore.impl.data.types import (
    Poses,
    OpenCVPinholeCameraModelParameters,
    ShutterType,
    TrackLabel,
    FrameLabel3,
    BBox3,
    LabelSource,
    DynamicFlagState,
    Tracks,
)
from ncore.impl.common.common import PoseInterpolator
from ncore.impl.common.transformations import transform_point_cloud, se3_inverse, transform_bbox
from ncore.impl.data_converter.data_converter import DataConverter
from ncore.impl.data_converter.waymo_utils import (
    parse_range_image_and_segmentations,
    convert_range_image_to_point_cloud,
    extrapolate_pose_based_on_velocity,
)


class WaymoConverter(DataConverter):
    """
    Dataset preprossing class, which preprocess waymo-open dataset to a canonical data representation as used within the Nvidia NRECore-SDK project.
    Waymo-open data can be downloaded from https://waymo.com/intl/en_us/open/download/ in form of tfrecords files. Further details on the dataset are
    available in the original publication https://arxiv.org/abs/1912.04838 or the githbub repository https://github.com/waymo-research/waymo-open-dataset

    DISCLAIMER: THIS SOURCE CODE IS NVIDIA INTERNAL/CONFIDENTIAL. DO NOT SHARE EXTERNALLY.
    IF YOU PLAN TO USE THIS CODEBASE FOR YOUR RESEARCH, PLEASE CONTACT ZAN GOJCIC zgojcic@nvidia.com / JANICK MARTINEZ ESTURO <janickm@nvidia.com>.
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

    def __init__(self, config):
        super().__init__(config)

        self.logger = logging.getLogger(__name__)

    @staticmethod
    def get_sequence_paths(config) -> list[Path]:
        return [p for p in sorted(Path(config.root_dir).glob("*.tfrecord"))]

    @staticmethod
    def from_config(config) -> DataConverter:
        return WaymoConverter(config)

    def convert_sequence(self, sequence_path: Path) -> None:
        """
        Runs dataset-specific conversion for a sequence
        """
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

        # DataWriter for all outputs
        self.data_writer = ContainerDataWriter(
            self.output_dir / sequence_name,
            sequence_name,
            [camera for camera in self.CAMERA_MAP.values()],
            [lidar for lidar in self.LIDAR_MAP.values()],
            [],
            "waymo-calibration",
            "waymo-egomotion",
            sequence_name,
            # single shard
            0,
            1,
            False,
        )

        # Decode poses
        self.decode_poses(frames)

        # Decode lidar frames
        self.decode_lidars(frames)

        # Decode camera frames
        self.decode_cameras(frames)

        # Store per-shard meta data / final success state / close file
        self.data_writer.finalize()

    def decode_poses(self, frames):
        # Grab poses from images, as for images the pose/timestamps correspond to each other
        T_rig_worlds_array = []
        T_rig_world_timestamps_us_array = []

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

                # Extrapolate pose points on the boundaries using velocity information to allow interpolation at lidar timestamps
                dt_us = 0.0
                if i == 0:
                    # extrapolate exactly to first lidar start-of-spin time
                    dt_us = frame.timestamp_micros - T_rig_world_timestamps_us_array[-1]
                if i == len(frames) - 1:
                    # make sure to overshoot a little over last lidar end-of-spin time
                    dt_us = int(1.25 * (frames[-1].timestamp_micros - frames[-2].timestamp_micros))

                if dt_us:
                    T_rig_world = T_rig_worlds_array[-1]
                    velocity_global = np.array(
                        [image.velocity.v_x, image.velocity.v_y, image.velocity.v_z], dtype=np.float32
                    ).reshape(3, 1)
                    omega_vehicle = np.array(
                        [image.velocity.w_x, image.velocity.w_y, image.velocity.w_z], dtype=np.float32
                    ).reshape(3, 1)
                    omega_world = np.matmul(T_rig_world[:3, :3], omega_vehicle)

                    T_rig_worlds_array.append(
                        extrapolate_pose_based_on_velocity(T_rig_world, velocity_global, omega_world, dt_us / 1e6)
                    )
                    T_rig_world_timestamps_us_array.append(T_rig_world_timestamps_us_array[-1] + dt_us)

        # make unique + sort + stack all poses (common canonical format convention)
        T_rig_world_timestamps_us, unique_indices = np.unique(
            np.array(T_rig_world_timestamps_us_array, dtype=np.uint64), return_index=True
        )
        T_rig_worlds = np.stack(T_rig_worlds_array)[unique_indices]

        # Use identity base pose as waymo data is already shifted
        T_rig_world_base = np.eye(4, dtype="float64")

        self.poses = Poses(
            T_rig_world_base=T_rig_world_base,
            T_rig_worlds=T_rig_worlds,
            T_rig_world_timestamps_us=T_rig_world_timestamps_us,
        )

        self.pose_interpolator = PoseInterpolator(self.poses.T_rig_worlds, self.poses.T_rig_world_timestamps_us)

        # Save the poses
        self.data_writer.store_poses(self.poses)

        # Log base pose to share it more easily with downstream teams (it is serialized also explicitly)
        with np.printoptions(floatmode="unique", linewidth=200):  # print in highest precision
            self.logger.info(f"> processed {len(T_rig_worlds)} poses, using base pose:\n{T_rig_world_base}")

    # Label IDs to label type strings
    LABEL_TYPE_STRING_MAP = {
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

    def decode_lidars(self, frames):
        """
        Converts the raw point cloud data into 3D depth rays in space also compensating for the
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
            label_id: str
            track_id: str
            label_class: str
            bbox3: BBox3
            global_speed: float

        label_id = 0
        raw_frame_labels: dict[int, list[RawFrameLabel3]] = {}  # timestamp to label in vehicle frame
        for frame in tqdm.tqdm(frames, desc="Parse frame labels"):
            frame_label_list: list[RawFrameLabel3] = []

            for label in frame.laser_labels:
                box = label.box
                frame_label_list.append(
                    RawFrameLabel3(
                        label_id=str(label_id),
                        track_id=label.id,
                        label_class=self.LABEL_TYPE_STRING_MAP[label.type],
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
                            np.linalg.norm(np.array([label.metadata.speed_x, label.metadata.speed_y], dtype=np.float32))
                        ),
                    )
                )

                label_id += 1

            raw_frame_labels[frame.timestamp_micros] = frame_label_list
        del label_id

        # Initialize labels struct that gets assembled while processing each frame label
        track_labels: dict[str, TrackLabel] = {}  # {TrackLabel} in track_labels[track_id]
        track_global_dynamic_flag: dict[str, bool] = {}  # bool in dynamic_tracks[track_id]

        for lidar_id, lidar_name in self.LIDAR_MAP.items():
            # Determine sensor extrinsics
            T_sensor_rig = np.array(calibrations[lidar_id].extrinsic.transform, dtype=np.float32).reshape(4, 4)

            # Range image properties / intrinsics
            inclinations_rad = np.empty(0)
            azimuths_rad = np.empty(0)

            # Collect all lidar per-frame data
            assert len(frames) > 1  # require at least two frames to compute frame bound timestamps
            frame_end_timestamps_us = []
            continuous_frame_index = 0
            for i, frame in tqdm.tqdm(enumerate(frames), desc=f"Process {lidar_name}", total=len(frames)):
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
                range_image, segmentation, range_image_top_pose = parse_range_image_and_segmentations(
                    frame, lidar_id, ri_index=0
                )

                range_image_second, _, _ = parse_range_image_and_segmentations(frame, lidar_id, ri_index=1)

                # Convert the range image to a ego-motion compensated 3D rays in sequence world coordinate frame
                # (motion-compensated to start frame time)
                (
                    points_world,
                    segmentation,
                    point_timestamps_us,
                    range_image_indices,  # N x 2
                    inclinations_rad,
                    azimuths_rad,
                ) = convert_range_image_to_point_cloud(
                    frame, lidar_id, range_image, segmentation, range_image_top_pose, timestamps_us
                )

                (points_world_second, _, _, range_image_indices_second, _, _,) = convert_range_image_to_point_cloud(
                    frame, lidar_id, range_image_second, None, range_image_top_pose, timestamps_us
                )

                # perform primary <-> secondary ray matching via linear indices (as every secondary ray has a parent primary ray)
                range_image_width = azimuths_rad.size
                linear_indices_primary = range_image_indices[:, 0] + range_image_indices[:, 1] * range_image_width
                linear_indices_second = (
                    range_image_indices_second[:, 0] + range_image_indices_second[:, 1] * range_image_width
                )

                primary_indices = np.where(linear_indices_second[:, None] == linear_indices_primary[None, :])[1]  # S

                # Pick semantic_class if available in current frame
                semantic_class = segmentation[:, 1].astype(np.int8) if (segmentation is not None) else None  # N

                frame_end_timestamps_us.append(frame_end_timestamp_us)

                # Interpolate poses
                T_rig_worlds = self.pose_interpolator.interpolate_to_timestamps(timestamps_us)

                # Bring point-cloud data into the right format
                T_world_sensor_end = se3_inverse(T_sensor_rig) @ se3_inverse(T_rig_worlds[1])

                xyz_s = transform_point_cloud(points_world[:, :3], T_world_sensor_end).astype(np.float32)  # N x 3
                xyz_e = transform_point_cloud(points_world[:, 3:6], T_world_sensor_end).astype(np.float32)  # N x 3
                xyz_e_second = transform_point_cloud(points_world_second[:, 3:6], T_world_sensor_end).astype(
                    np.float32
                )  # S x 3
                # normalize intensity (https://github.com/ouster-lidar/ouster_example/issues/488)
                intensity = np.tanh(points_world[:, 6])  # N
                intensity_second = np.tanh(points_world_second[:, 6])  # S
                elongation = points_world[:, 7]  # N
                elongation_second = points_world_second[:, 7]  # S

                # Process frame labels (defined in frame-associated rig frame)
                frame_labels: list[FrameLabel3] = []
                T_rig_labelstime_world = np.array(
                    tf.reshape(tf.constant(frame.pose.transform, dtype=tf.float64), [4, 4])
                ).astype(np.float32)
                T_rig_labelstime_sensor_end = T_world_sensor_end @ T_rig_labelstime_world

                for raw_frame_label in raw_frame_labels[frame.timestamp_micros]:
                    # Map label in rig space to frame label in sensor space
                    bbox3_sensor = BBox3.from_array(
                        transform_bbox(raw_frame_label.bbox3.to_array(), T_rig_labelstime_sensor_end)
                    )

                    # Approximate measurement time by azimuth angle of centroid in sensor's x/y plane
                    # and performing linear interpolation between start / end times
                    azimuth_rad = np.arctan2(bbox3_sensor.centroid[0], bbox3_sensor.centroid[1])
                    t = 1 - (azimuth_rad + np.pi) / (
                        2 * np.pi
                    )  # clockwise spinning (largest azimuth are measured first)
                    frame_label_timestamp = int(timestamps_us[0]) + int(t * (timestamps_us[1] - timestamps_us[0]))

                    frame_label = FrameLabel3(
                        label_id=raw_frame_label.label_id,
                        track_id=raw_frame_label.track_id,
                        label_class=raw_frame_label.label_class,
                        bbox3=bbox3_sensor,
                        global_speed=raw_frame_label.global_speed,
                        confidence=None,
                        timestamp_us=frame_label_timestamp,
                        source=LabelSource.EXTERNAL,
                    )

                    frame_labels.append(frame_label)

                    # store track label data
                    if frame_label.track_id not in track_labels:
                        track_labels[frame_label.track_id] = TrackLabel(sensors={})
                        track_global_dynamic_flag[frame_label.track_id] = (
                            True if frame_label.label_class in self.LABEL_STRINGS_UNCONDITIONALLY_DYNAMIC else False
                        )

                    if lidar_name not in track_labels[frame_label.track_id].sensors:
                        track_labels[frame_label.track_id].sensors[lidar_name] = []

                    # append frame timestamp into *sorted* list (frames are processed sorted by timestamp)
                    track_labels[frame_label.track_id].sensors[lidar_name].append(frame_end_timestamp_us)

                    if (
                        frame_label.label_class not in self.LABEL_STRINGS_UNCONDITIONALLY_STATIC
                        and frame_label.global_speed >= self.GLOBAL_SPEED_DYNAMIC_THRESHOLD
                    ):
                        track_global_dynamic_flag[frame_label.track_id] = True

                ## Compute point dynamic flag
                dynamic_flag: np.ndarray = np.full(len(xyz_e), DynamicFlagState.STATIC.value, dtype=np.int8)  # N x 1

                # Use the bounding boxes to remove dynamic objects / set dynamic flag
                for frame_label in frame_labels:
                    # If the object is classified to be globally dynamic, update the points that fall in that bounding box
                    if track_global_dynamic_flag[frame_label.track_id]:
                        bbox = frame_label.bbox3.to_array()
                        # enlarge the bounding box for the check *only*
                        bbox[3:6] += self.LIDAR_DYNAMIC_FLAG_BBOX_PADDING_METERS
                        dynamic_flag[isWithin3DBBox(xyz_e, bbox.reshape(1, -1))] = DynamicFlagState.DYNAMIC.value

                # Serialize lidar frame
                self.data_writer.store_lidar_frame(
                    lidar_name,
                    continuous_frame_index,
                    xyz_s,
                    xyz_e,
                    intensity,
                    point_timestamps_us,
                    dynamic_flag,
                    semantic_class,
                    frame_labels,
                    T_rig_worlds,
                    timestamps_us,
                    {
                        # primary ray data
                        "elongation": elongation.reshape(-1).astype(np.float32),  # N
                        "range_image_indices": range_image_indices.reshape((-1, 2)).astype(
                            np.uint32
                        ),  # N x 2 (indices into HxW source range image)
                        # secondary ray data
                        "primary_indices": primary_indices.reshape(-1).astype(
                            np.uint32
                        ),  # S (indices of the primary parent ray)
                        "xyz_e_second": xyz_e_second.reshape((-1, 3)).astype(np.float32),  # S x 3
                        "intensity_second": intensity_second.reshape(-1).astype(np.float32),  # S
                        "elongation_second": elongation_second.reshape(-1).astype(np.float32),  # S
                    },
                )

                continuous_frame_index += 1

            # Store all static sensor data
            self.data_writer.store_lidar_meta(
                lidar_name,
                np.array(frame_end_timestamps_us, dtype=np.uint64),
                T_sensor_rig,
                {
                    # angles associated with range-image "pixels"
                    "inclinations_rad": inclinations_rad.reshape(-1).astype(np.float32),  # H (one per range-image row)
                    "azimuths_rad": azimuths_rad.reshape(-1).astype(np.float32),  # W (one per range-image column)
                },
            )

        # Save the accumulated tracks in global time
        self.data_writer.store_tracks(Tracks(track_labels))

    def decode_cameras(self, frames):
        """
        Extracts the images and camera metadata for all cameras within a single frame. Camera metadata must hold
        the information used to compensate for rolling shutter effect and to convert RGB images to 3D RGB rays in space
        """

        calibrations = {c.name: c for c in frames[0].context.camera_calibrations}

        for camera_id, camera_name in self.CAMERA_MAP.items():
            ## Get the calibration data
            calibration = calibrations[camera_id]

            T_sensor_rig = np.array(tf.reshape(tf.constant(calibration.extrinsic.transform, dtype=tf.float32), [4, 4]))

            ## Fix camera frame convention from
            # - waymo camera: principal axis along the +x axis, the y-axis points to the left, and the z-axis points up
            # to
            # - NCORE camera: principal axis along the +z axis, the x-axis points to the right, and the y-axis points down
            T_sensor_rig[:3, :3] = T_sensor_rig[:3, :3] @ np.array(
                [[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float32
            )

            frame_end_timestamps_us = []
            continuous_frame_index = 0
            for frame in tqdm.tqdm(frames, desc=f"Process {camera_name}"):
                ## Load current camera's image
                image = {image.name: image for image in frame.images}[camera_id]

                ## Get frame timestamps
                frame_start_timestamp_us = int((image.camera_trigger_time + image.shutter / 2) * 1e6)
                frame_end_timestamp_us = int((image.camera_readout_done_time - image.shutter / 2) * 1e6)

                ## Collect timestamps poses
                timestamps_us = np.array([frame_start_timestamp_us, frame_end_timestamp_us], dtype=np.uint64)

                ## Determine start/end poses
                # Velocity and angular velocity of the SDC / rig at camera pose timestamp.
                # Velocity is provided in the world reference frame and the ang. velocity is in SDC / rig frame
                # https://github.com/waymo-research/waymo-open-dataset/issues/462
                T_rig_world = np.array(tf.reshape(tf.constant(image.pose.transform, dtype=tf.float32), [4, 4]))
                velocity_global = np.array(
                    [image.velocity.v_x, image.velocity.v_y, image.velocity.v_z], dtype=np.float32
                ).reshape(3, 1)
                omega_vehicle = np.array(
                    [image.velocity.w_x, image.velocity.w_y, image.velocity.w_z], dtype=np.float32
                ).reshape(3, 1)
                omega_world = np.matmul(T_rig_world[:3, :3], omega_vehicle)

                # Extrapolate the pose to the start and end timestamp of the image frame considering the (angular) velocity at the time of the acquisition
                T_rig_worlds = np.stack(
                    [
                        extrapolate_pose_based_on_velocity(
                            T_rig_world,
                            velocity_global,
                            omega_world,
                            (image.camera_trigger_time + image.shutter / 2) - image.pose_timestamp,
                        ),
                        extrapolate_pose_based_on_velocity(
                            T_rig_world,
                            velocity_global,
                            omega_world,
                            (image.camera_readout_done_time - image.shutter / 2) - image.pose_timestamp,
                        ),
                    ]
                )

                frame_end_timestamps_us.append(frame_end_timestamp_us)

                # Store the image and its metadata
                self.data_writer.store_camera_frame(
                    camera_name,
                    continuous_frame_index,
                    image.image,
                    "jpeg",
                    T_rig_worlds,
                    timestamps_us,
                    {},
                )
                continuous_frame_index += 1

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

            self.data_writer.store_camera_meta(
                camera_name,
                np.array(frame_end_timestamps_us, dtype=np.uint64),
                T_sensor_rig,
                OpenCVPinholeCameraModelParameters(
                    np.array([width, height], dtype=np.uint64),
                    rolling_shutter_direction,
                    np.array([c_u, c_v], dtype=np.float32),
                    np.array([f_u, f_v], dtype=np.float32),
                    np.array([k1, k2, k3, 0, 0, 0], dtype=np.float32),
                    np.array([p1, p2], dtype=np.float32),
                    np.array([0, 0, 0, 0], dtype=np.float32),
                ),
                None,
                {},
            )
