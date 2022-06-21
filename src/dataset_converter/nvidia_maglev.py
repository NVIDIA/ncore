# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import json
import os
import logging
import shutil

import numpy as np

import point_cloud_utils as pcu

from plyfile import PlyData

from collections import defaultdict

from src.dataset_converter import DataConverter

from src.nvidia_utils import (sensor_to_rig, parse_rig_sensors_from_dict,
                              camera_intrinsic_parameters, compute_fw_polynomial, compute_ftheta_parameters,
                              camera_car_mask)

from src.common import (load_jsonl, save_pkl, save_pc_dat, PoseInterpolator)


class NvidiaMaglevConverter(DataConverter):
    """
    DataConverter consuming the preprocessed output of Maglev dsai-pp workflows
    """

    ## Constants defined for *Hyperion8* sensor-set

    # TODO: the value for the 70FoV wide camera seems to be different, we need to clarify
    CAM2EXPOSURETIME = {'wide': 1641.58, 'fisheye': 10987.00}

    CAM2ROLLINGSHUTTERDELAY = {'wide': 31611.55, 'fisheye': 32561.63}

    CAMERA_2_IDTYPERIG = {
        'camera_front_wide_120fov': ['00', 'wide', 'camera:front:wide:120fov'],
        'camera_cross_left_120fov': ['01', 'wide', 'camera:cross:left:120fov'],
        'camera_cross_right_120fov': ['02', 'wide', 'camera:cross:right:120fov'],
        'camera_rear_left_70fov': ['03', 'wide', 'camera:rear:left:70fov'],
        'camera_rear_right_70fov': ['04', 'wide', 'camera:rear:right:70fov'],
        'camera_rear_tele_30fov': ['05', 'wide', 'camera:rear:tele:30fov'],
        'camera_front_fisheye_200fov': ['10', 'fisheye', 'camera:front:fisheye:200fov'],
        'camera_left_fisheye_200fov': ['11', 'fisheye', 'camera:left:fisheye:200fov'],
        'camera_right_fisheye_200fov': ['12', 'fisheye', 'camera:right:fisheye:200fov'],
        'camera_rear_fisheye_200fov': ['13', 'fisheye', 'camera:rear:fisheye:200fov']
    }

    ID_TO_CAMERA = {'00': 'camera_front_wide_120fov',
                    '01': 'camera_cross_left_120fov',
                    '02': 'camera_cross_right_120fov',
                    '03': 'camera_rear_left_70fov',
                    '04': 'camera_rear_right_70fov',
                    '05': 'camera_rear_tele_30fov',
                    '10': 'camera_front_fisheye_200fov',
                    '11': 'camera_left_fisheye_200fov',
                    '12': 'camera_right_fisheye_200fov',
                    '13': 'camera_rear_fisheye_200fov'
                    }

    LIDAR_SENSORNAME = 'lidar:gt:top:p128:v4p5'

    # Minimum / maximum distances (in meters) for point cloud measurements (to filter out invalid points, points on the ego-car),
    # as well as minimum height (there might be some spurious measurements bellow ground)
    LIDAR_FILTER_MIN_DISTANCE = 3.5
    LIDAR_FILTER_MAX_DISTANCE = 100.0
    LIDAR_FILTER_MIN_RIG_HEIGHT = -1.0

    DBG_MAX_FRAMES_ = 10  # REMOVE, for testing only

    def __init__(self, config):
        self.logger = logging.getLogger(__name__)

        super().__init__(config)

        self.sequence_pathnames = [self.root_dir]

    def convert_one(self, sequence_path):
        """
        Runs the conversion of a single session (single job output of Maglev dsai-pp workflow)

        Args:
            sequence_path (string): path to Maglev dsai-pp workflow job output
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
        self.create_folders(os.path.join(self.session_id))

        # Decode data from maglev
        self.decode_poses_timestamps()

        self.decode_images()

        self.decode_lidar()

        # Perform instance and semantic segmentation of all the images
        if self.sem_seg_flag:
            self.run_semantic_segmentation(os.path.join(self.session_id))   

        # Tracks are far to big to do this for the whole track
        # TODO: talk about the strategy here, do we want to maybe chunk this?
        if self.surf_rec_flag:
            self.run_surface_extraction(os.path.join(self.session_id)) 

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
        for egomotion_pose_entry in load_jsonl(os.path.join(self.sequence_path, 'egomotion/egomotion.json')):
            # Skip invalid poses
            if not egomotion_pose_entry['valid']:
                continue

            # Note: there is additional data like lat/long and sensor-related information
            #       which could be used in the future
            egomotion_pose_timestamp = egomotion_pose_entry['timestamp']
            egomotion_pose = np.asfarray(
                egomotion_pose_entry['pose'].split(' ')).reshape((4, 4)).transpose()

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
                    f"Unsupported source ego frame {egomotion_pose_entry['in_sensor_name_frame']}")

            self.poses_timestamps.append(egomotion_pose_timestamp)
            self.poses.append(egomotion_pose)

        # Stack all poses (common canonical format convention)
        self.poses = np.stack(self.poses)

        # Save the poses
        poses_save_path = os.path.join(self.output_dir, self.session_id,
                                       self.poses_save_dir, 'poses.npz')
        np.savez(poses_save_path, ego_poses=self.poses,
                 timestamps=self.poses_timestamps)

        logger.info(f'> processed {len(self.poses_timestamps)} poses')

    def decode_images(self):
        logger = self.logger.getChild('decode_images')
        logger.info(f'Loading camera data')

        # Collect timestamps of individual cameras
        camera_timestamps = defaultdict(list)

        # Process all camera images based on the pose timestamps
        for camera in self.CAMERA_2_IDTYPERIG.keys():
            cam_id, cam_type, cam_id_rig = self.CAMERA_2_IDTYPERIG[camera]
            camera_calibration_data = self.sensors_calibration_data[cam_id_rig]

            logger.info(f'Processing camera {cam_id_rig}')

            # Target folder for all camera-specific outputs
            camera_base_save_path = os.path.join(
                self.output_dir, self.session_id, self.image_save_dir, 'image_' + cam_id)

            # Load frame numbers and timestamps
            frames_metadata = load_jsonl(os.path.join(
                self.sequence_path, 'cameras', cam_id_rig, 'meta.json'))
            frame_numbers = np.array(
                [frame_data['frame_number'] for frame_data in frames_metadata])
            frame_timestamps = np.array(
                [frame_data['timestamp'] for frame_data in frames_metadata])
            del(frames_metadata)

            # Get the frame range of the first and last frame relative to available egomotion poses and respecting exposure timings
            start_idx = np.argmax(
                frame_timestamps - self.CAM2ROLLINGSHUTTERDELAY[cam_type] - self.CAM2EXPOSURETIME[cam_type] / 2 > self.poses_timestamps[0])
            # take all frames if all are within egomotion range, or determine last valid frame
            end_idx = np.argmax(frame_timestamps - self.CAM2EXPOSURETIME[cam_type] / 2 >= self.poses_timestamps[-1]) \
                if frame_timestamps[-1] - self.CAM2EXPOSURETIME[cam_type] / 2 > self.poses_timestamps[-1] else -1

            # Subsample frames to valid range
            frame_numbers = frame_numbers[start_idx:end_idx]
            frame_timestamps = frame_timestamps[start_idx:end_idx]

            # REMOVE, for testing only
            frame_numbers = frame_numbers[:self.DBG_MAX_FRAMES_]
            frame_timestamps = frame_timestamps[:self.DBG_MAX_FRAMES_]

            # Copy all valid images
            for continuos_frame_index, frame_number in enumerate(frame_numbers):
                source_image_path = os.path.join(
                    self.sequence_path, 'cameras', cam_id_rig, str(frame_number) + '.jpeg')
                # store as *increasing* canonical frame IDs
                target_image_path = os.path.join(camera_base_save_path, str(
                    continuos_frame_index).zfill(self.INDEX_DIGITS) + '.jpeg')
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
            pose_interpolator = PoseInterpolator(
                self.poses, self.poses_timestamps)

            # Constant mask image, which currently only contains the ego car mask
            # TODO: extend this with dynamic object masks
            mask_image = camera_car_mask(camera_calibration_data)

            for frame_idx, frame_timestamp in enumerate(frame_timestamps):
                mask_image.get_image().save(os.path.join(camera_base_save_path,
                                                         f'mask_{str(frame_idx).zfill(self.INDEX_DIGITS)}.png'), optimize=True)

                metadata = {}
                metadata['img_width'] = intrinsic[2]
                metadata['img_height'] = intrinsic[3]
                # 1 = TOP_TO_BOTTOM, 2 = LEFT_TO_RIGHT, 3 = BOTTOM_TO_TOP, 4 = RIGHT_TO_LEFT
                metadata['rolling_shutter_direction'] = 1
                metadata['camera_model'] = 'f_theta' if cam_type in [
                    'wide', 'fisheye'] else 'pinhole'
                metadata['exposure_time'] = self.CAM2EXPOSURETIME[cam_type]
                metadata['intrinsic'] = intrinsic
                metadata['T_cam_rig'] = T_cam_rig

                # Interpolate the start and end pose to the timestamps of the first and last row
                sofTimestamp = frame_timestamp - \
                    self.CAM2ROLLINGSHUTTERDELAY[cam_type] - \
                    self.CAM2EXPOSURETIME[cam_type] / 2
                eofTimestamp = frame_timestamp - \
                    self.CAM2EXPOSURETIME[cam_type] / 2
                metadata['ego_pose_timestamps'] = np.array(
                    [sofTimestamp, eofTimestamp])
                metadata['ego_pose_s'] = pose_interpolator.interpolate_to_timestamps(sofTimestamp)[
                    0]
                metadata['ego_pose_e'] = pose_interpolator.interpolate_to_timestamps(eofTimestamp)[
                    0]

                metadata_save_path = os.path.join(camera_base_save_path, str(
                    frame_idx).zfill(self.INDEX_DIGITS) + '.pkl')
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

        image_timestamps_save_path = os.path.join(
            self.output_dir, self.session_id, self.image_save_dir, 'timestamps.pkl')
        save_pkl(camera_timestamps, image_timestamps_save_path)

        logger.info(f'> processed {len(camera_timestamps)} cameras')

    def decode_lidar(self):
        logger = self.logger.getChild('decode_lidar')
        logger.info(f'Loading lidar data')

        # Target folder for all lidar-specific outputs
        lidar_base_save_path = os.path.join(
            self.output_dir, self.session_id, self.point_cloud_save_dir)

        # Load extrinsics
        lidar_calibration_data = self.sensors_calibration_data[self.LIDAR_SENSORNAME]
        T_lidar_rig = sensor_to_rig(lidar_calibration_data)

        # Initialize the pose interpolator object
        pose_interpolator = PoseInterpolator(self.poses, self.poses_timestamps)

        # Load frame numbers and timestamps
        frames_metadata = load_jsonl(os.path.join(
            self.sequence_path, 'lidars', self.LIDAR_SENSORNAME, 'meta.json'))
        frame_numbers = np.array(
            [frame_data['frame_number'] for frame_data in frames_metadata])
        frame_timestamps = np.array(
            [frame_data['timestamp'] for frame_data in frames_metadata])
        del(frames_metadata)

        # Get the frame range of the first and last frame relative to available egomotion poses and respecting exposure timings
        start_idx = np.argmax(frame_timestamps > self.poses_timestamps[0])
        # take all frames if all are within egomotion range, or determine last valid frame
        end_idx = np.argmax(frame_timestamps >= self.poses_timestamps[-1]) \
            if frame_timestamps[-1] > self.poses_timestamps[-1] else -1

        # Subsample frames to valid range
        frame_numbers = frame_numbers[start_idx:end_idx]
        frame_timestamps = frame_timestamps[start_idx:end_idx]

        # REMOVE, for testing only
        frame_numbers = frame_numbers[:self.DBG_MAX_FRAMES_]
        frame_timestamps = frame_timestamps[:self.DBG_MAX_FRAMES_]

        # Copy all valid point clouds
        for continuos_frame_index, (frame_number, frame_timestamp) in enumerate(zip(frame_numbers, frame_timestamps)):
            source_pc_path = os.path.join(
                self.sequence_path, 'lidars', self.LIDAR_SENSORNAME, str(frame_number) + '.ply')
            # store as *increasing* canonical frame IDs
            target_pc_path = os.path.join(lidar_base_save_path, str(
                continuos_frame_index).zfill(self.INDEX_DIGITS) + '.dat')

            # Interpolate egomotion at frame timestamp to obtain lidar start point
            T_rig_world = pose_interpolator.interpolate_to_timestamps(frame_timestamp)[
                0]
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
            xyz_e = np.row_stack([plydata.elements[0].data['x'],
                                  plydata.elements[0].data['y'],
                                  plydata.elements[0].data['z'],
                                  np.ones(point_count)])

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
        lidar_timestamp_save_path = os.path.join(lidar_base_save_path, 'timestamps.npz')
        np.savez(lidar_timestamp_save_path, timestamps=frame_timestamps.tolist())

        logger.info(f'> processed {len(frame_timestamps)} point clouds')

    def decode_labels(self):
        raise NotImplementedError("WIP")
