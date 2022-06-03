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

from collections import defaultdict

from src.dataset_converter import DataConverter

from src.nvidia_utils import (sensor_to_rig, parse_rig_sensors_from_dict,
                              camera_intrinsic_parameters, compute_fw_polynomial, compute_ftheta_parameters,
                              camera_car_mask)

from src.common import (load_jsonl, save_pkl, PoseInterpolator)


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

        # Read rig json file
        with open(os.path.join(sequence_path, 'rig.json'), 'r') as fp:
            self.rig = json.load(fp)

        # Sessions to be processed
        self.session_id = self.rig['rig']['properties']['session_id']

        self.logger.info(f'Converting session {self.session_id}')

        # Create all output folders
        self.create_folders(os.path.join(self.session_id))

        # Decode data from maglev
        self.decode_poses_timestamps()

        self.decode_images()

    def decode_poses_timestamps(self):
        # Initialize pose / timestamp variables
        self.poses = []
        self.poses_timestamps = []

        # Load egomotion trajectory
        for egomotion_pose in load_jsonl(os.path.join(self.sequence_path, 'egomotion/egomotion.json')):
            # Skip invalid poses
            if not egomotion_pose['valid']:
                continue

            # Note: there is additional data like lat/long and sensor-related information
            #       which could be used in the future
            egomotion_pose_timestamp = egomotion_pose['timestamp']
            egomotion_pose = np.asfarray(
                egomotion_pose['pose'].split(' ')).reshape((4, 4)).transpose()

            self.poses_timestamps.append(egomotion_pose_timestamp)
            self.poses.append(egomotion_pose)

        # Stack all poses (common canonical format convention)
        self.poses = np.stack(self.poses)

        # Save the poses
        poses_save_path = os.path.join(self.output_dir, self.session_id,
                                       self.poses_save_dir, 'poses.npz')
        np.savez(poses_save_path, ego_poses=self.poses,
                 timestamps=self.poses_timestamps)

    def decode_images(self):
        # Parse the rig calibration file
        calibration_data = parse_rig_sensors_from_dict(self.rig)
        camera_timestamps = defaultdict(list)

        DBG_MAX_IMAGES = 10  # REMOVE, for testing only

        # Filter the images based on the pose timestamps
        for camera in self.CAMERA_2_IDTYPERIG.keys():
            cam_id, cam_type, cam_id_rig = self.CAMERA_2_IDTYPERIG[camera]
            camera_calibration_data = calibration_data[cam_id_rig]

            # Target folder for all camera-specific outputs
            camera_base_save_path = os.path.join(
                self.output_dir, self.session_id, self.image_save_dir, 'image_' + cam_id)

            # Load frame numbers and timestamps
            frame_metadata = load_jsonl(os.path.join(
                self.sequence_path, 'cameras', cam_id_rig, 'meta.json'))
            frame_numbers = np.array(
                [frame_data['frame_number'] for frame_data in frame_metadata])
            frame_timestamps = np.array(
                [frame_data['timestamp'] for frame_data in frame_metadata])
            del(frame_metadata)

            # Get the frame range of the first and last frame relative to available egomotion poses and respecting exposure timings
            start_idx = np.argmax(
                frame_timestamps - self.CAM2ROLLINGSHUTTERDELAY[cam_type] - self.CAM2EXPOSURETIME[cam_type] / 2 > self.poses_timestamps[0])
            # take all frames if all are within egomotion range, or determine last valid frame
            end_idx = np.argmax(frame_timestamps - self.CAM2EXPOSURETIME[cam_type] / 2 >= self.poses_timestamps[-1]) \
                if frame_timestamps[-1] - self.CAM2EXPOSURETIME[cam_type] / 2 > self.poses_timestamps[-1] else -1

            frame_numbers = frame_numbers[start_idx:end_idx]
            frame_timestamps = frame_timestamps[start_idx:end_idx]

            # REMOVE, for testing only
            frame_numbers = frame_numbers[:DBG_MAX_IMAGES]
            frame_timestamps = frame_timestamps[:DBG_MAX_IMAGES]

            # Copy all valid images
            continuos_frame_index = 0  # store as *increasing* canonical frame IDs
            for frame_number in frame_numbers:
                source_image_path = os.path.join(
                    self.sequence_path, 'cameras', cam_id_rig, str(frame_number) + '.jpeg')
                target_image_path = os.path.join(camera_base_save_path, str(
                    continuos_frame_index).zfill(self.INDEX_DIGITS) + '.jpeg')
                shutil.copy(source_image_path, target_image_path)

                continuos_frame_index += 1

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
            cam_pose_interpolator = PoseInterpolator(
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
                metadata['ego_pose_s'] = cam_pose_interpolator.interpolate_to_timestamps(sofTimestamp)[
                    0]
                metadata['ego_pose_e'] = cam_pose_interpolator.interpolate_to_timestamps(eofTimestamp)[
                    0]

                metadata_save_path = os.path.join(camera_base_save_path, str(
                    frame_idx).zfill(self.INDEX_DIGITS) + '.pkl')
                save_pkl(metadata, metadata_save_path)

                # Save the camera pose timestamps, corresponds approximately to the timestamp of the principle point pixel
                camera_timestamps[cam_id].append(
                    (eofTimestamp + sofTimestamp) / 2)

        # Save timestamps of all cameras
        for cam in camera_timestamps.keys():
            # check that at least a single frame was processed
            if len(camera_timestamps[cam]):
                camera_timestamps[cam] = np.stack(camera_timestamps[cam])

        image_timestamps_save_path = os.path.join(
            self.output_dir, self.session_id, self.image_save_dir, 'timestamps.pkl')
        save_pkl(camera_timestamps, image_timestamps_save_path)

    def decode_lidar(self):
        raise NotImplementedError("WIP")

    def decode_labels(self):
        raise NotImplementedError("WIP")
