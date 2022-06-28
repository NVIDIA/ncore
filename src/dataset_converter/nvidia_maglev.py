# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import json
import os
import logging
import shutil

from collections import defaultdict

import numpy as np

from plyfile import PlyData

from src.dataset_converter import BaseNvidiaDataConverter

from src.nvidia_utils import (sensor_to_rig, parse_rig_sensors_from_dict,
                              camera_intrinsic_parameters,
                              compute_fw_polynomial, compute_ftheta_parameters,
                              camera_car_mask)

from src.common import (load_jsonl, save_pkl, save_pc_dat, PoseInterpolator)


class NvidiaMaglevConverter(BaseNvidiaDataConverter):
    """
    NVIDIA-specific data conversion (based on Maglev dsai-pp workflows data extraction)
    """

    def __init__(self, config):
        self.logger = logging.getLogger(__name__)

        super().__init__(config)

        self.sequence_pathnames = [self.root_dir]

        self.seek_sec = config.seek_sec
        self.duration_sec = config.duration_sec

    @staticmethod
    def time_bounds(timestamps_us, seek_sec, duration_sec):
        """
        Determine start and end timestamps given optional seek and duration times

        Args:
            timestamps_us (List[numeric]): list of all available timestamps (in microseconds)
            seek_sec (float): Optional: if non-None, the time (in seconds)  to skip starting from the first timestamp
            duration_sec (float): Optional: if non-None, the total time (in seconds) between the start and end time bounds

        Return:
            start_timestamp_us (float): first valid timestamp in restricted bounds (in microseconds)
            end_timestamp_us (float): last valid timestamp in restricted bounds (in microseconds)
        """

        start_timestamp_us = timestamps_us[0]
        end_timestamp_us = timestamps_us[-1]

        if seek_sec:
            start_timestamp_us += seek_sec * 1e6

        if duration_sec:
            end_timestamp_us = start_timestamp_us + duration_sec * 1e6

        assert start_timestamp_us < end_timestamp_us, "Arguments lead to invalid time bounds"

        return start_timestamp_us, end_timestamp_us

    def convert_one(self, sequence_path):
        """
        Runs the conversion of a single session (single job output of Maglev dsai-pp workflow)

        Args:
            sequence_path (string): path to Maglev dsai-pp workflow job output
        
        Return:
            sub_sequence_names List[string]: names of the processed sub-sequences
        """

        self.sequence_path = sequence_path

        # Read rig json file and sensor information
        with open(os.path.join(sequence_path, 'rig.json'), 'r') as fp:
            self.rig = json.load(fp)

        self.sensors_calibration_data = parse_rig_sensors_from_dict(self.rig)

        # Sessions to be processed
        self.session_id = self.rig['rig']['properties']['session_id']

        self.logger.info(f'Converting session {self.session_id}')

        # Create all output folders
        self.create_folders(self.session_id)

        # Decode data from maglev
        self.decode_poses_timestamps()

        self.decode_images()

        self.decode_lidar()

        return [self.session_id]

    def decode_poses_timestamps(self):
        logger = self.logger.getChild('decode_poses_timestamps')
        logger.info(f'Loading poses')

        # Initialize pose / timestamp variables
        self.poses = []
        self.poses_timestamps = []

        # Load lidar extrinsics to compute poses of the rig frame if egomotion is represented in lidar frame
        T_lidar_rig = sensor_to_rig(
            self.sensors_calibration_data[self.LIDAR_SENSORNAME])
        T_rig_lidar = np.linalg.inv(T_lidar_rig)

        # Load egomotion trajectory
        for egomotion_pose_entry in load_jsonl(
                os.path.join(self.sequence_path, 'egomotion/egomotion.json')):
            # Skip invalid poses
            if not egomotion_pose_entry['valid']:
                continue

            # Note: there is additional data like lat/long and sensor-related information
            #       which could be used in the future
            egomotion_pose_timestamp = egomotion_pose_entry['timestamp']
            egomotion_pose = np.asfarray(
                egomotion_pose_entry['pose'].split(' ')).reshape(
                    (4, 4)).transpose()

            # Make sure poses represent rigToWorld transformations
            # Note: there currently seems to be an inconsistency in the egomotion indexer output - keep
            #       this verified workaround logic for now (might need to be adapted if egomotion indexer
            #       is fixed)
            if egomotion_pose_entry['sensor_name'] == 'dgps':
                pass
            elif egomotion_pose_entry['sensor_name'] == self.LIDAR_SENSORNAME:
                # Convert pose in lidar frame to pose in rig frame
                egomotion_pose = egomotion_pose @ T_rig_lidar
            else:
                raise ValueError(
                    f"Unsupported source ego frame {egomotion_pose_entry['in_sensor_name_frame']}"
                )

            self.poses_timestamps.append(egomotion_pose_timestamp)
            self.poses.append(egomotion_pose)

        # Stack all poses (common canonical format convention)
        self.poses = np.stack(self.poses)

        # Save the poses
        poses_save_path = os.path.join(self.output_dir, self.session_id,
                                       self.poses_save_dir, 'poses.npz')
        np.savez(poses_save_path,
                 ego_poses=self.poses,
                 timestamps=self.poses_timestamps)

        logger.info(f'> processed {len(self.poses_timestamps)} poses')

    def decode_images(self):
        logger = self.logger.getChild('decode_images')
        logger.info(f'Loading camera data')

        # Collect timestamps of individual cameras
        camera_timestamps = defaultdict(list)

        # Determine time bounds from available egomotion poses
        start_timestamp_us, end_timestamp_us = self.time_bounds(
            self.poses_timestamps, self.seek_sec, self.duration_sec)

        # Process all camera images based on the pose timestamps
        for camera in self.CAMERA_2_IDTYPERIG.keys():
            cam_id, cam_type, cam_id_rig = self.CAMERA_2_IDTYPERIG[camera]
            camera_calibration_data = self.sensors_calibration_data[cam_id_rig]

            logger.info(f'Processing camera {cam_id_rig}')

            # Target folder for all camera-specific outputs
            camera_base_save_path = os.path.join(self.output_dir,
                                                 self.session_id,
                                                 self.image_save_dir,
                                                 'image_' + cam_id)

            # Load frame numbers and timestamps
            frames_metadata = load_jsonl(
                os.path.join(self.sequence_path, 'cameras', cam_id_rig,
                             'meta.json'))
            frame_numbers = np.array(
                [frame_data['frame_number'] for frame_data in frames_metadata])
            frame_timestamps = np.array(
                [frame_data['timestamp'] for frame_data in frames_metadata])
            del (frames_metadata)

            # Get the frame range of the first and last frame relative to available egomotion poses and respecting exposure timings
            start_idx = np.argmax(
                frame_timestamps - self.CAM2ROLLINGSHUTTERDELAY[cam_type] -
                self.CAM2EXPOSURETIME[cam_type] / 2 >= start_timestamp_us)
            end_idx = np.argmax(
                frame_timestamps -
                self.CAM2EXPOSURETIME[cam_type] / 2 > end_timestamp_us
            ) if frame_timestamps[-1] - self.CAM2EXPOSURETIME[
                cam_type] / 2 > end_timestamp_us else -1  # take all frames if all are within egomotion range, or determine last valid frame

            # Subsample frames to valid range
            frame_numbers = frame_numbers[start_idx:end_idx]
            frame_timestamps = frame_timestamps[start_idx:end_idx]

            # Copy all valid images
            for continuos_frame_index, frame_number in enumerate(
                    frame_numbers):
                source_image_path = os.path.join(self.sequence_path, 'cameras',
                                                 cam_id_rig,
                                                 str(frame_number) + '.jpeg')
                target_image_path = os.path.join(
                    camera_base_save_path,
                    str(continuos_frame_index).zfill(self.INDEX_DIGITS) +
                    '.jpeg')  # store as *increasing* canonical frame IDs

                shutil.copy(source_image_path, target_image_path)

            # Extract the calibration metadata
            T_cam_rig = sensor_to_rig(camera_calibration_data)
            intrinsic = camera_intrinsic_parameters(camera_calibration_data)

            # Estimate the forward polynomial and other F-theta parameters
            fw_poly_coeff = compute_fw_polynomial(intrinsic)
            max_ray_distortion, max_angle = compute_ftheta_parameters(
                np.concatenate((intrinsic, fw_poly_coeff)))
            intrinsic = np.concatenate(
                (intrinsic, fw_poly_coeff, max_ray_distortion, max_angle))

            # Pose interpolator to obtain start / end egomotion poses
            pose_interpolator = PoseInterpolator(self.poses,
                                                 self.poses_timestamps)

            # Constant mask image, which currently only contains the ego car mask
            # TODO: extend this with dynamic object masks
            mask_image = camera_car_mask(camera_calibration_data)

            for frame_idx, frame_timestamp in enumerate(frame_timestamps):
                mask_image.get_image().save(os.path.join(
                    camera_base_save_path,
                    f'mask_{str(frame_idx).zfill(self.INDEX_DIGITS)}.png'),
                                            optimize=True)

                metadata = {}
                metadata['img_width'] = intrinsic[2]
                metadata['img_height'] = intrinsic[3]
                metadata[
                    'rolling_shutter_direction'] = 1  # 1 = TOP_TO_BOTTOM, 2 = LEFT_TO_RIGHT, 3 = BOTTOM_TO_TOP, 4 = RIGHT_TO_LEFT

                metadata['camera_model'] = 'f_theta' if cam_type in [
                    'wide', 'fisheye'
                ] else 'pinhole'
                metadata['exposure_time'] = self.CAM2EXPOSURETIME[cam_type]
                metadata['intrinsic'] = intrinsic
                metadata['T_cam_rig'] = T_cam_rig

                # Interpolate the start and end pose to the timestamps of the first and last row
                sofTimestamp = frame_timestamp - self.CAM2ROLLINGSHUTTERDELAY[
                    cam_type] - self.CAM2EXPOSURETIME[cam_type] / 2
                eofTimestamp = frame_timestamp - self.CAM2EXPOSURETIME[
                    cam_type] / 2
                metadata['ego_pose_timestamps'] = np.array(
                    [sofTimestamp, eofTimestamp])
                metadata[
                    'ego_pose_s'] = pose_interpolator.interpolate_to_timestamps(
                        sofTimestamp)[0]
                metadata[
                    'ego_pose_e'] = pose_interpolator.interpolate_to_timestamps(
                        eofTimestamp)[0]

                metadata_save_path = os.path.join(
                    camera_base_save_path,
                    str(frame_idx).zfill(self.INDEX_DIGITS) + '.pkl')
                save_pkl(metadata, metadata_save_path)

                # Save the camera pose timestamps, corresponds approximately to the timestamp of the principle point pixel
                camera_timestamps[cam_id].append(
                    (eofTimestamp + sofTimestamp) / 2)

            logger.info(f'> processed {len(camera_timestamps[cam_id])} frames')

        # Save timestamps of all cameras
        for cam in camera_timestamps.keys():
            # check that at least a single frame was processed
            if len(camera_timestamps[cam]):
                camera_timestamps[cam] = np.stack(camera_timestamps[cam])

        image_timestamps_save_path = os.path.join(self.output_dir,
                                                  self.session_id,
                                                  self.image_save_dir,
                                                  'timestamps.pkl')
        save_pkl(camera_timestamps, image_timestamps_save_path)

        logger.info(f'> processed {len(camera_timestamps)} cameras')

    def decode_lidar(self):
        logger = self.logger.getChild('decode_lidar')
        logger.info(f'Loading lidar data')

        # Target folder for all lidar-specific outputs
        lidar_base_save_path = os.path.join(self.output_dir, self.session_id,
                                            self.point_cloud_save_dir)

        # Load extrinsics
        lidar_calibration_data = self.sensors_calibration_data[
            self.LIDAR_SENSORNAME]
        T_lidar_rig = sensor_to_rig(lidar_calibration_data)

        # Initialize the pose interpolator object
        pose_interpolator = PoseInterpolator(self.poses, self.poses_timestamps)

        # Load frame numbers and timestamps
        frames_metadata = load_jsonl(
            os.path.join(self.sequence_path, 'lidars',
                         self.LIDAR_SENSORNAME, 'meta.json'))
        frame_numbers = np.array(
            [frame_data['frame_number'] for frame_data in frames_metadata])
        frame_timestamps = np.array(
            [frame_data['timestamp'] for frame_data in frames_metadata])
        del (frames_metadata)

        # Determine time bounds from available egomotion poses
        start_timestamp_us, end_timestamp_us = self.time_bounds(
            self.poses_timestamps, self.seek_sec, self.duration_sec)

        # Get the frame range of the first and last frame relative to available egomotion poses
        start_idx = np.argmax(frame_timestamps >= start_timestamp_us)
        end_idx = np.argmax(
            frame_timestamps > end_timestamp_us
        ) if frame_timestamps[
            -1] > end_timestamp_us else -1  # take all frames if all are within egomotion range, or determine last valid frame

        # Subsample frames to valid range
        frame_numbers = frame_numbers[start_idx:end_idx]
        frame_timestamps = frame_timestamps[start_idx:end_idx]

        # Copy all valid point clouds
        for continuos_frame_index, (frame_number,
                                    frame_timestamp) in enumerate(
                                        zip(frame_numbers, frame_timestamps)):
            source_pc_path = os.path.join(self.sequence_path,
                                          'lidars',
                                          self.LIDAR_SENSORNAME,
                                          str(frame_number) + '.ply')
            target_pc_path = os.path.join(
                lidar_base_save_path,
                str(continuos_frame_index).zfill(self.INDEX_DIGITS) +
                '.dat')  # store as *increasing* canonical frame IDs

            # Interpolate egomotion at frame timestamp to obtain lidar start point
            T_rig_world = pose_interpolator.interpolate_to_timestamps(
                frame_timestamp)[0]
            T_lidar_world = T_rig_world @ T_lidar_rig

            # Load point cloud (already motion-compensated)
            plydata = PlyData.read(source_pc_path)
            point_count = plydata.elements[0].count

            # Create 3D ray structure of 3D rays in space with accompanying metadata.
            # Format; x_s, y_s, z_s, x_e, y_e, z_e, dist, intensity, dynamic_flag
            # Dynamic flag is set to -1 if the information is not available, 0 static, 1 = dynamic

            # Constant time-compensated lidar origin in world frame, N x 3
            xyz_s = np.full((point_count, 3), T_lidar_world[:3, -1])

            # Homogeneous points in lidar frame (4 x N)
            xyz_e = np.row_stack([
                plydata.elements[0].data['x'], plydata.elements[0].data['y'],
                plydata.elements[0].data['z'],
                np.ones(point_count)
            ])

            # Transform points from lidar to rig frame and remember minimum height filter condition
            xyz_e = T_lidar_rig @ xyz_e
            valid_idxs = xyz_e[2, :] > self.LIDAR_FILTER_MIN_RIG_HEIGHT

            # Transform points from rig to world frame + drop homogenous dimension and transpose to match output dimension
            xyz_e = T_rig_world @ xyz_e
            xyz_e = xyz_e[:-1, :].transpose()  # N x 3

            # Compute distances
            dist = np.linalg.norm(xyz_s - xyz_e, axis=1)  # N x 1

            # Load intensities
            intensity = plydata.elements[0].data['intensity']  # N x 1

            # Dynamic flag
            # TODO: properly set dynamic flag of point cloud based on labels
            dynamic_flag = np.full(point_count, -1.)  # N x 1

            # Assemble full point-cloud ray structure
            point_cloud = np.column_stack(
                (xyz_s, xyz_e, dist, intensity, dynamic_flag))

            # Filter points based on minimal / max distance and minimal height
            valid_idxs &= point_cloud[:, 6] > self.LIDAR_FILTER_MIN_DISTANCE
            valid_idxs &= point_cloud[:, 6] < self.LIDAR_FILTER_MAX_DISTANCE

            point_cloud = point_cloud[valid_idxs, :]

            logger.debug(f'> filtered {valid_idxs.sum()} invalid points')

            # Serialize point cloud
            save_pc_dat(target_pc_path, point_cloud)

        # Save all lidar timestamps
        lidar_timestamp_save_path = os.path.join(lidar_base_save_path,
                                                 'timestamps.npz')
        np.savez(lidar_timestamp_save_path,
                 timestamps=frame_timestamps.tolist())

        logger.info(f'> processed {len(frame_timestamps)} point clouds')
