# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import logging
import glob
import os
import json
import io

from pathlib import Path

import tqdm
import numpy as np
import cv2
import PIL.Image as PILImage

from google.protobuf import text_format
from protobuf_to_dict import protobuf_to_dict

from ncore.impl.data_converter.protos.deepmap import track_data_pb2, pointcloud_pb2
from ncore.impl.data_converter.protos.deepmap.util import extract_sensor_2_sdc
from ncore.impl.data_converter.data_converter import BaseNvidiaDataConverter
from ncore.impl.data.data3 import ContainerDataWriter
from ncore.impl.data.types import Poses, FThetaCameraModelParameters, LabelSource, ShutterType, Tracks
from ncore.impl.common.common import PoseInterpolator
from ncore.impl.common.nvidia_utils import (LabelProcessor, parse_rig_sensors_from_dict,
                                            load_maglev_lidar_indexer_frame_meta, sensor_to_rig, extract_pose,
                                            vehicle_bbox, camera_intrinsic_parameters, compute_fw_polynomial,
                                            compute_ftheta_fov, camera_car_mask)
from ncore.impl.av_utils import isWithin3DBBox


class NvidiaDeepmapConverter(BaseNvidiaDataConverter):
    """
    NVIDIA-specific data converter (based on DeepMap tracks)
    """
    def __init__(self, config):
        super().__init__(config)

        self.logger = logging.getLogger(__name__)

        self.start_timestamp_us = config.start_timestamp_us
        self.end_timestamp_us = config.end_timestamp_us

    @staticmethod
    def get_sequence_paths(config) -> list[Path]:
        return [Path(p) for p in sorted(glob.glob(os.path.join(config.root_dir, '*/')))]

    @staticmethod
    def from_config(config) -> NvidiaDeepmapConverter:
        return NvidiaDeepmapConverter(config)

    def convert_sequence(self, sequence_path: Path) -> None:
        """
        Runs the conversion of a single sequence
        """

        self.sequence_name = sequence_path.name

        sequence_tracks = sorted(glob.glob(os.path.join(sequence_path, 'tracks', '*/')))

        for track in sequence_tracks:
            self.track_name = track.split(os.sep)[-2]

            # Read rig json file
            with open(os.path.join(sequence_path, 'rig.json'), 'r') as fp:
                self.rig = json.load(fp)

            self.calibration_data = parse_rig_sensors_from_dict(self.rig)

            self.constants = self.get_constants(self.rig['rig']['properties'], list(self.calibration_data.keys()))

            # *Single* reference lidar sensor
            match self.constants:
                case self.Hyperion8Constants():
                    self.LIDAR_SENSOR_ID = 'lidar_gt_top_p128_v4p5'
                case self.Hyperion81Constants():
                    self.LIDAR_SENSOR_ID = 'lidar_gt_top_p128'
                case _:
                    raise ValueError(f'code-update required to select main-sensor for platform {self.constants}')

            # ContainerDataWriter for all outputs (always single-shard)
            self.data_writer = ContainerDataWriter(
                self.output_dir / f'{self.sequence_name}-{self.track_name}',
                f'{self.sequence_name}-{self.track_name}',
                list(self.constants.CAMERAID_TO_RIGNAME.keys()),
                list(self.constants.LIDARID_TO_RIGNAME.keys()),
                # no radars yet
                [],
                # TODO: parse these from the data
                'scene-calib',
                'deepmap',
                f'{self.sequence_name}-{self.track_name}',
                # always single-shard
                0,
                1,
                False)

            

            # Initialize the track aligned track record structure
            self.track_data = track_data_pb2.AlignedTrackRecords()

            # Read in the track record data from a proto file
            # This includes camera_records and lidar_records (see track_record proto for more detail)
            with open(os.path.join(track, 'aligned_track_records.pb.txt'), 'r') as f:
                text_format.Parse(f.read(), self.track_data)

            # Extract all the lidar paths, timestamps and poses from the track record
            self.track_data = protobuf_to_dict(self.track_data)

            self.decode_poses_timestamps(sequence_path)

            self.decode_labels(sequence_path)

            self.decode_lidar(sequence_path)

            self.decode_cameras(sequence_path)

            # Store per-shard meta data / final success state / close file
            self.data_writer.finalize()

    def decode_poses_timestamps(self, sequence_path):
        # Compute the transformation from the SDC (deepmap rig) to the NV rig definition
        T_lidar_rig = sensor_to_rig(self.calibration_data[self.constants.LIDARID_TO_RIGNAME[self.LIDAR_SENSOR_ID]])
        T_lidar_sdc = extract_sensor_2_sdc(os.path.join(sequence_path, 'to_vehicle_transform_lidar00.pb.txt'))
        T_rig_sdc = T_lidar_sdc @ np.linalg.inv(T_lidar_rig)

        # Extract poses and timestamps, which are converted to the nvidia convention
        # extract_pose() extracts the T_sdc_world transformation and we want to get T_rig_world
        # the poses hence have to be right-multiplied with T_rig_sdc
        self.poses = []
        self.poses_timestamps = []
        self.lidar_timestamps = []
        self.lidar_data_paths = []
        if 'lidar_records' in self.track_data:
            for frame in self.track_data['lidar_records'][0]['records']:
                self.lidar_timestamps.append(frame['timestamp_microseconds'])
                self.lidar_data_paths.append(frame['file_path'])

                if 'pose' in frame:
                    self.poses_timestamps.append(frame['timestamp_microseconds'])
                    self.poses.append(extract_pose(frame['pose']) @ T_rig_sdc)

        if 'camera_records' in self.track_data:
            for frame in self.track_data['camera_records'][0]['records']:
                if 'pose' in frame:
                    self.poses_timestamps.append(frame['timestamp_microseconds'])
                    self.poses.append(extract_pose(frame['pose']) @ T_rig_sdc)

        # Stack and sort the poses
        self.poses = np.stack(self.poses)
        self.poses_timestamps = np.stack(self.poses_timestamps).astype(np.uint64)
        sort_idx = np.argsort(self.poses_timestamps)

        # All the available poses
        self.poses = self.poses[sort_idx]
        self.poses_timestamps = self.poses_timestamps[sort_idx]

        # Select base-pose
        base_pose = self.poses[0]

        # Convert the poses to the sequence coordinate frame
        self.poses = np.linalg.inv(base_pose) @ self.poses

        # Subselect poses in case timestamp ranges were provided (only subselect serialized poses, keep all poses for pose interpolation of frame data)
        start_timestamp_us = self.start_timestamp_us if self.start_timestamp_us else self.poses_timestamps[0]
        end_timestamp_us = self.end_timestamp_us if self.end_timestamp_us else self.poses_timestamps[-1]

        local_pose_range = np.logical_and(start_timestamp_us <= self.poses_timestamps, self.poses_timestamps
                                          <= end_timestamp_us)

        # Save the poses
        self.data_writer.store_poses(
            Poses(T_rig_world_base=base_pose,
                  T_rig_worlds=self.poses[local_pose_range],
                  T_rig_world_timestamps_us=self.poses_timestamps[local_pose_range]))

    def decode_labels(self, sequence_path):
        # Perform label parsing
        self.track_labels, self.frame_labels, self.track_global_dynamic_flag = LabelProcessor.parse(
            os.path.join(sequence_path, 'labels', 'autolabels.parquet'), {
                self.LIDAR_SENSOR_ID:
                load_maglev_lidar_indexer_frame_meta(
                    Path(sequence_path) / 'labels' / f'{self.LIDAR_SENSOR_ID}_meta.json')
            }, {
                self.LIDAR_SENSOR_ID:
                sensor_to_rig(self.calibration_data[self.constants.LIDARID_TO_RIGNAME[self.LIDAR_SENSOR_ID]])
            }, self.poses_timestamps, self.poses, LabelSource.AUTOLABEL)

        # Save the accumulated track
        self.data_writer.store_tracks(Tracks(self.track_labels))

    def decode_lidar(self, sequence_path):
        # Load lidar extrinsics to compute poses of the rig frame if egomotion is represented in lidar frame
        T_lidar_rig = sensor_to_rig(self.calibration_data[self.constants.LIDARID_TO_RIGNAME[self.LIDAR_SENSOR_ID]])

        # Load vehicle bounding box (defined in rig frame)
        vehicle_bbox_rig = vehicle_bbox(self.rig)
        vehicle_bbox_rig[
            3:6] += self.constants.LIDAR_FILTER_VEHICLE_BBOX_PADDING_METERS  # pad the bounding box slightly

        # Initialize the pose interpolator object
        pose_interpolator = PoseInterpolator(self.poses, self.poses_timestamps)

        # Frame-annotation timestamps
        fa_timestamps = np.array(sorted(list(self.frame_labels[self.LIDAR_SENSOR_ID].keys())))

        lidar_end_timestmap = []
        frame_idx = 0
        for frame_path in tqdm.tqdm(self.lidar_data_paths):

            # Load the point cloud data
            data = pointcloud_pb2.PointCloud()
            with open(os.path.join(sequence_path, 'tracks', frame_path), 'rb') as f:
                data.ParseFromString(f.read())

            frame_end_timestamp = data.meta_data.end_timestamp_microseconds

            if self.start_timestamp_us and frame_end_timestamp < self.start_timestamp_us:
                continue  # not there yet
            if self.end_timestamp_us and frame_end_timestamp > self.end_timestamp_us:
                break  # already passed end - no need to keep on processing

            # Find the closest frame in the annotations
            time_diff = np.abs(fa_timestamps - frame_end_timestamp)
            annotation_frame_idx = np.argmin(time_diff)

            if time_diff[annotation_frame_idx] <= 10000:
                fa_timestamp = fa_timestamps[annotation_frame_idx]
            else:
                self.logger.debug(f'no annotated frame found for lidar frame {frame_idx}')
                fa_timestamp = None

            raw_pc = np.concatenate([
                np.array(data.data.points_x, dtype=np.float32)[:, None],
                np.array(data.data.points_y, dtype=np.float32)[:, None],
                np.array(data.data.points_z, dtype=np.float32)[:, None]
            ],
                                    axis=1)

            intensities = np.frombuffer(data.data.intensities, dtype=np.uint8).astype(
                np.float32) / 255.0  # [0 .. 255] -> [0.0 .. 1.0]

            # Determine per-column rig-to-world pose and compute per-column lidar-to-world transformations
            column_timestamps = np.array(data.data.column_timestamps_microseconds, dtype=np.uint64)
            try:
                # First / last frame might not work as the pose is available only at the middle of the frame
                column_poses = pose_interpolator.interpolate_to_timestamps(column_timestamps)
            except Exception as e:  # work on python 3.x
                self.logger.warn(
                    f'Lidar frame conversion failed for lidar frame {frame_idx} due to out-of-egomotion pose: {e}')
                continue
            T_column_lidar_worlds = column_poses @ T_lidar_rig[None, :, :]
            # Pose of the rig at the end of the lidar spin, can be used to transform points into a local coordinate frame
            T_rig_world = column_poses[-1]
            T_world_rig = np.linalg.inv(T_rig_world)
            T_world_lidar = np.linalg.inv(T_lidar_rig) @ T_world_rig

            # Perform per-column unwinding, transforming from lidar to world coordinates
            transformed_pc = np.empty((len(raw_pc), 6), dtype=np.float32)
            transformed_pc[:, :3] = T_column_lidar_worlds[data.data.column_indices, :3,
                                                          -1]  # N X 3 - ray start points in world space
            transformed_pc[:, 3:] = (T_column_lidar_worlds[data.data.column_indices, :3, :3] @ raw_pc[:, :, None]
                                     ).squeeze(-1) + transformed_pc[:, :3]  # N x 3 - ray end points in world space

            pc_world_homogeneous = np.row_stack(
                [transformed_pc[:, 3:6].transpose(),
                 np.ones(transformed_pc.shape[0], dtype=np.float32)])  # 4 x N
            pc_rig = (T_world_rig @ pc_world_homogeneous)[:-1, :].transpose()  # N x 3
            pc_lidar = (T_world_lidar @ pc_world_homogeneous)[:-1, :].transpose()  # N x 3

            pc_start_world_homogeneous = np.row_stack(
                [transformed_pc[:, 0:3].transpose(),
                 np.ones(transformed_pc.shape[0], dtype=np.float32)])  # 4 x N
            pc_start_lidar = (T_world_lidar @ pc_start_world_homogeneous)[:-1, :].transpose()  # N x 3

            timestamps = column_timestamps[data.data.column_indices]

            # Filter outs points that are inside the vehicles bounding-box
            valid_idxs_vehicle_bbox = np.logical_not(
                isWithin3DBBox(pc_rig.astype(np.float32), vehicle_bbox_rig.reshape(1, -1)))

            # Filter points based on distances
            dist = np.linalg.norm(transformed_pc[:, :3] - transformed_pc[:, 3:6], axis=1)

            # Filter points on the distances LIDAR_FILTER_MAX_DISTANCE (remove points that are very far away)
            valid_idx_dist = np.less_equal(dist,
                                           self.constants.LIDARID_TO_FILTER_MAX_DISTANCE_METERS[self.LIDAR_SENSOR_ID])
            valid_idx = np.logical_and(valid_idxs_vehicle_bbox, valid_idx_dist)

            # Subselect to valid points
            xyz_e = pc_lidar[valid_idx, :].astype(np.float32)
            xyz_s = pc_start_lidar[valid_idx, :].astype(np.float32)
            intensity = intensities[valid_idx, None].flatten()
            timestamp = timestamps[valid_idx, None].flatten()

            # Interpolate start / end pose
            timestamps_us = np.array([np.min(timestamp), frame_end_timestamp], dtype=np.uint64)
            T_rig_worlds = pose_interpolator.interpolate_to_timestamps(timestamps_us)

            # Use the bounding boxes to remove dynamic objects
            dynamic_flag, frame_labels = LabelProcessor.lidar_dynamic_flag(self.LIDAR_SENSOR_ID,
                                                                           xyz_e,
                                                                           fa_timestamp if fa_timestamp else -1,
                                                                           self.frame_labels,
                                                                           self.track_global_dynamic_flag)

            # Serialize lidar frame
            self.data_writer.store_lidar_frame(self.LIDAR_SENSOR_ID, frame_idx, xyz_s, xyz_e, intensity, timestamp,
                                               dynamic_flag, None, frame_labels, T_rig_worlds, timestamps_us)

            # Save the end time stamp of the lidar spin
            lidar_end_timestmap.append(frame_end_timestamp)

            frame_idx += 1

        # Save sensor meta data
        self.data_writer.store_lidar_meta(lidar_id=self.LIDAR_SENSOR_ID,
                                          frame_timestamps_us=np.array(lidar_end_timestmap, dtype=np.uint64),
                                          T_sensor_rig=T_lidar_rig)

    def decode_cameras(self, sequence_path):
        # Pose interpolator to obtain start / end egomotion poses
        pose_interpolator = PoseInterpolator(self.poses, self.poses_timestamps)

        # Filter the images based on the pose timestamps
        for camera_id, camera_rig_name in self.constants.CAMERAID_TO_RIGNAME.items():
            sensor_type = self.constants.CAMERAID_TO_SENSORTYPE[camera_id]

            # Get the camera timestamps
            frame_timestamps = np.genfromtxt(os.path.join(sequence_path, 'camera_data/', camera_id + '.mp4.timestamps'),
                                             delimiter='\t',
                                             dtype=np.uint64)

            # Get the frame index of the first and last frame
            start_timestamp_us = self.start_timestamp_us if self.start_timestamp_us else self.poses_timestamps[0]
            end_timestamp_us = self.end_timestamp_us if self.end_timestamp_us else self.poses_timestamps[-1]

            start_idx = np.where(frame_timestamps[:, 1] > start_timestamp_us +
                                 self.constants.SENSORTYPE_TO_ROLLINGSHUTTERDELAY_US[sensor_type] +
                                 2 * self.constants.SENSORTYPE_TO_EXPOSURETIME_HALF_US[sensor_type])[0][0]
            end_idx = np.where(frame_timestamps[:, 1] >= end_timestamp_us)[0]
            end_idx = end_idx[0] if len(end_idx) else len(frame_timestamps[:, 1])

            frame_timestamps = frame_timestamps[start_idx:end_idx, :]

            # Extract all relevant images
            path = os.path.join(sequence_path, 'camera_data/', camera_id + '.mp4')
            self.logger.info(f'Loading video {camera_id} @ {path}')
            vidcap = cv2.VideoCapture(
                path,
                cv2.CAP_FFMPEG,
                # set high timeout values to open video / read frames in case IO bandwith is low
                [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 25 * 60000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 25 * 60000])

            if not vidcap.isOpened():
                self.logger.warn(f'skipping {camera_id} - can\'t open video')
                continue

            # seek video to first valid frame
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_timestamps[0, 0])

            success, image_bgr = vidcap.read()

            if not success:
                self.logger.warn(f'skipping {camera_id} - can\'t read frame')
                continue

            raw_frame_index = frame_timestamps[0, 0]
            continous_frame_index = 0
            while success:
                if raw_frame_index > frame_timestamps[-1, 0]:
                    break

                assert vidcap.get(
                    cv2.CAP_PROP_POS_FRAMES  # represents the *next* frame to be read
                ) - 1 == raw_frame_index, f'invalid frame decoded / frames were skipped for {camera_id}'

                frame_timestamp_us = frame_timestamps[continous_frame_index, 1]

                # Interpolate the start and end pose to the timestamps of the first and last row
                timestamps_us = np.array([
                    # sof-timestamp
                    frame_timestamp_us - self.constants.SENSORTYPE_TO_ROLLINGSHUTTERDELAY_US[sensor_type] -
                    self.constants.SENSORTYPE_TO_EXPOSURETIME_HALF_US[sensor_type],
                    # eof-timestamp
                    frame_timestamp_us - self.constants.SENSORTYPE_TO_EXPOSURETIME_HALF_US[sensor_type]
                ])

                T_rig_worlds = pose_interpolator.interpolate_to_timestamps(timestamps_us)

                with io.BytesIO() as buffer:
                    FORMAT = 'jpeg'
                    image_rgb = image_bgr[..., ::-1]  # invert last dimension from BGR -> RGB (reverse BGR)
                    PILImage.fromarray(image_rgb).save(buffer, format=FORMAT, optimize=True,
                                                       quality=91)  # encode image as jpeg

                    self.data_writer.store_camera_frame(camera_id, continous_frame_index, buffer.getvalue(), FORMAT,
                                                        T_rig_worlds, timestamps_us)

                success, image_bgr = vidcap.read()
                raw_frame_index += 1
                continous_frame_index += 1

            assert continous_frame_index == frame_timestamps.shape[0], \
                f'not all camera frames serialized for camera {camera_id}'

            # Extract the calibration metadata
            camera_calibration_data = self.calibration_data[camera_rig_name]
            T_sensor_rig = sensor_to_rig(camera_calibration_data)

            # Estimate the forward polynomial
            intrinsic = camera_intrinsic_parameters(camera_calibration_data, self.logger)

            bw_poly = intrinsic[4:]
            fw_poly = compute_fw_polynomial(intrinsic)
            max_angle = min(compute_ftheta_fov(intrinsic)[2].item(), np.deg2rad(self.constants.MAX_CAMERA_FOV_DEG / 2))

            # Constant mask image, which currently only contains the ego car mask
            # TODO: extend this with dynamic object masks
            mask_image = camera_car_mask(camera_calibration_data)

            # all end-of-frame timestamps
            eof_camera_timestamps_us = (frame_timestamps[:, 1] -
                                        self.constants.SENSORTYPE_TO_EXPOSURETIME_HALF_US[sensor_type]).flatten()

            self.data_writer.store_camera_meta(
                camera_id, eof_camera_timestamps_us, T_sensor_rig,
                FThetaCameraModelParameters(intrinsic[2:4].astype(np.uint64), ShutterType.ROLLING_BOTTOM_TO_TOP,
                                            intrinsic[0:2],
                                            FThetaCameraModelParameters.PolynomialType.PIXELDIST_TO_ANGLE, bw_poly,
                                            fw_poly, max_angle), mask_image.get_image())
