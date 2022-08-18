# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import json
import os
import logging
import shutil
import re
import gc
import multiprocessing
import tqdm

import numpy as np
import point_cloud_utils as pcu

from typing import Optional
from collections import defaultdict
from functools import partial

from src.py.dataset_converter import BaseNvidiaDataConverter
from src.py.common.nvidia_utils import (sensor_to_rig, parse_rig_sensors_from_dict, camera_intrinsic_parameters,
                                        compute_fw_polynomial, compute_ftheta_parameters, camera_car_mask, vehicle_bbox, LabelProcessor)
from src.py.common.common import (load_jsonl, save_pkl, save_pc_dat, PoseInterpolator)
from src.cpp.av_utils import isWithin3DBBox

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

        self.multiprocessing_camera = config.multiprocessing_camera
        self.multiprocessing_lidar = config.multiprocessing_lidar

        self.egomotion_file = config.egomotion_file

        self.skip_dynamic_flag = config.skip_dynamic_flag

    @staticmethod
    def time_bounds(timestamps_us: list[int], seek_sec: Optional[float], duration_sec: Optional[float]) -> tuple[int, int]:
        """
        Determine start and end timestamps given optional seek and duration times

        Args:
            timestamps_us : list of all available timestamps (in microseconds)
            seek_sec: Optional: if non-None, the time (in seconds)  to skip starting from the first timestamp
            duration_sec: Optional: if non-None, the total time (in seconds) between the start and end time bounds

        Return:
            start_timestamp_us: first valid timestamp in restricted bounds (in microseconds)
            end_timestamp_us: last valid timestamp in restricted bounds (in microseconds)
        """

        start_timestamp_us = timestamps_us[0]
        end_timestamp_us = timestamps_us[-1]

        if seek_sec:
            assert seek_sec >= 0.0, "Require positive seek time"
            start_timestamp_us += int(seek_sec * 1e6)

        if duration_sec:
            assert duration_sec >= 0.0, "Require positive duration time"
            end_timestamp_us = start_timestamp_us + int(duration_sec * 1e6)

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

        # Determine session-id to be processed
        # Note: session_id in loaded rig meta might not reflect the actual current session due to bugs
        #       in the rig generation, prefer loading correct ID from rig used for egomotion for now
        with open(os.path.join(sequence_path, 'egomotion', 'rig_egomotion_indexer.json'), 'r') as fp:
            match = re.search(r'session_data/(\w{8}-\w{4}-\w{4}-\w{4}-\w{12})/', fp.read())
            if match:
                self.session_id = match[1]
            else:
                raise ValueError("Unable to determine trustable session_id")

        self.logger.info(f'Converting session {self.session_id}')

        # Create all output folders
        self.create_folders(self.session_id)

        # Decode data from maglev
        self.decode_poses()

        self.decode_labels()

        self.decode_images()

        self.decode_lidar()

        return [self.session_id]

    def decode_poses(self):
        logger = self.logger.getChild('decode_poses')
        logger.info(f'Loading poses')

        # Initialize pose / timestamp variables
        self.poses = []
        self.poses_timestamps = []

        # Load lidar extrinsics to compute poses of the rig frame if egomotion is represented in lidar frame
        T_lidar_rig = sensor_to_rig(
            self.sensors_calibration_data[self.LIDAR_SENSORNAME])
        T_rig_lidar = np.linalg.inv(T_lidar_rig)

        # Load egomotion trajectory
        if not self.egomotion_file:
            # Use default egomotion jsonl location
            egomotion_file = os.path.join(self.sequence_path, 'egomotion/egomotion.json')
        else:
            # Use overwrite file
            egomotion_file = self.egomotion_file 

        for egomotion_pose_entry in load_jsonl(egomotion_file):
            # Skip invalid poses
            if not egomotion_pose_entry['valid']:
                continue

            # Note: there is additional data like lat/long and sensor-related information
            #       which could be used in the future
            egomotion_pose_timestamp = int(egomotion_pose_entry['timestamp'])
            egomotion_pose = np.asfarray(
                egomotion_pose_entry['pose'].split(' '), dtype=np.float64).reshape(
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

    def decode_labels(self):
        logger = self.logger.getChild('decode_labels')
        logger.info(f'Loading labels')

        # Initialize annotation structs (defaults in case no labels are available loaded)
        self.labels = {'3d_labels': {}}
        self.frame_labels = {}

        # Process autolabels, if available
        labels_path = os.path.join(self.sequence_path, 'cuboids_tracked', 'labels_lidar.parquet')
        if not os.path.exists(labels_path):
            logger.warn(f'> no {labels_path} file available, skipping label generation')
            return

        # Determine time bounds from available egomotion poses and user-provided restrictions
        start_timestamp_us, end_timestamp_us = self.time_bounds(self.poses_timestamps, self.seek_sec, self.duration_sec)

        # Perform label parsing
        self.labels, self.frame_labels = LabelProcessor.parse(labels_path, start_timestamp_us, end_timestamp_us, logger)

        # Save the accumulated data
        save_pkl(self.labels, os.path.join(self.output_dir, self.session_id, 'labels.pkl'))

        # Save the per frame data
        save_pkl(self.frame_labels, os.path.join(self.output_dir, self.session_id, 'frame_labels.pkl'))

    def decode_images(self):
        logger = self.logger.getChild('decode_images')
        logger.info(f'Loading camera data')

        # Collect timestamps of individual cameras
        camera_timestamps = defaultdict(list)

        # Determine time bounds from available egomotion poses
        start_timestamp_us, end_timestamp_us = self.time_bounds(
            self.poses_timestamps, self.seek_sec, self.duration_sec)

        # Process all valid camera images
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
            start_idx = np.argmax(frame_timestamps - self.CAM2ROLLINGSHUTTERDELAY[cam_type] - self.CAM2EXPOSURETIME[cam_type] / 2 >= start_timestamp_us)
            end_idx = np.argmax(frame_timestamps - self.CAM2EXPOSURETIME[cam_type] / 2 > end_timestamp_us
            ) if frame_timestamps[-1] - self.CAM2EXPOSURETIME[
                cam_type] / 2 > end_timestamp_us else -1  # take all frames if all are within egomotion range, or determine last valid frame

            # Subsample frames to valid range
            frame_numbers = frame_numbers[start_idx:end_idx]
            frame_timestamps = frame_timestamps[start_idx:end_idx]

            # Copy all valid images
            process_function = partial(self._copy_image_process,
                                       camera_base_save_path=camera_base_save_path,
                                       cam_id_rig=cam_id_rig)
            process_iterable = enumerate(frame_numbers)
            if self.multiprocessing_camera:
                # Use multiprocessing to speed up IO
                with multiprocessing.Pool(
                        # limit the number of processes to not allocate too many resources concurrently
                        processes=min(12, os.cpu_count())) as pool:
                    logger.info(f'> copying {len(frame_timestamps)} images using {pool._processes} worker processes')
                    pool.map(func=process_function, iterable=process_iterable)
            else:
                # Use single process
                for arg in tqdm.tqdm(process_iterable, total=len(frame_numbers)):
                    process_function(arg)
            logger.info(f'> finished copying {len(frame_timestamps)} images')

            # Extract the calibration metadata
            T_cam_rig = sensor_to_rig(camera_calibration_data)
            intrinsic = camera_intrinsic_parameters(camera_calibration_data, logger)

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
                sofTimestamp = frame_timestamp - self.CAM2ROLLINGSHUTTERDELAY[cam_type] - self.CAM2EXPOSURETIME[cam_type] / 2
                eofTimestamp = frame_timestamp - self.CAM2EXPOSURETIME[cam_type] / 2
                metadata['ego_pose_timestamps'] = np.array([sofTimestamp, eofTimestamp])
                metadata['ego_pose_s'] = pose_interpolator.interpolate_to_timestamps(sofTimestamp)[0]
                metadata['ego_pose_e'] = pose_interpolator.interpolate_to_timestamps(eofTimestamp)[0]

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

    def _copy_image_process(self, args, camera_base_save_path, cam_id_rig):
        """ Copies a single image """

        # Decode current frame data to process
        continuos_frame_index, frame_number = args[0], args[1]

        # Copy image from source to target
        source_image_path = os.path.join(self.sequence_path, 'cameras', cam_id_rig, str(frame_number) + '.jpeg')
        target_image_path = os.path.join(camera_base_save_path,
                                         str(continuos_frame_index).zfill(self.INDEX_DIGITS) +
                                         '.jpeg')  # store as *increasing* canonical frame IDs

        shutil.copy(source_image_path, target_image_path)

    def decode_lidar(self):
        logger = self.logger.getChild('decode_lidar')
        logger.info(f'Loading lidar data')

        # Target folder for all lidar-specific outputs
        lidar_base_save_path = os.path.join(self.output_dir, self.session_id, self.point_cloud_save_dir)

        # Load extrinsics
        T_lidar_rig = sensor_to_rig(self.sensors_calibration_data[self.LIDAR_SENSORNAME])

        # Load vehicle bounding box (defined in rig frame)
        vehicle_bbox_rig = vehicle_bbox(self.rig)
        vehicle_bbox_rig[3:6] += self.LIDAR_FILTER_VEHICLE_BBOX_PADDING  # pad the bounding box slightly

        # Initialize the pose interpolator object
        pose_interpolator = PoseInterpolator(self.poses, self.poses_timestamps)

        # Load frame numbers and timestamps
        frames_metadata = load_jsonl(
            os.path.join(self.sequence_path, 'lidars', self.LIDAR_SENSORNAME, 'meta.json'))
        frame_numbers = np.array([frame_data['frame_number'] for frame_data in frames_metadata])
        frame_timestamps = np.array([frame_data['timestamp'] for frame_data in frames_metadata])
        del (frames_metadata)

        # Determine time bounds from available egomotion poses
        start_timestamp_us, end_timestamp_us = self.time_bounds(self.poses_timestamps, self.seek_sec,
                                                                self.duration_sec)

        # Get the frame range of the first and last frame relative to available egomotion poses
        start_idx = np.argmax(frame_timestamps >= start_timestamp_us)
        end_idx = np.argmax(frame_timestamps > end_timestamp_us) if frame_timestamps[
            -1] > end_timestamp_us else -1  # take all frames if all are within egomotion range, or determine last valid frame

        # Subsample frames to valid range
        frame_numbers = frame_numbers[start_idx:end_idx]
        frame_timestamps = frame_timestamps[start_idx:end_idx]

        # Process all valid point clouds using multi-processing
        process_function = partial(
            self._decode_lidar_process,
            lidar_base_save_path=lidar_base_save_path,
            T_lidar_rig=T_lidar_rig,
            pose_interpolator=pose_interpolator,
            vehicle_bbox_rig=vehicle_bbox_rig,
            num_frames=len(frame_numbers),
            logger=logger,
        )
        process_iterable = zip(range(len(frame_numbers)), frame_numbers, frame_timestamps)
        if self.multiprocessing_lidar:
            # Use multiprocessing to speed up IO
            with multiprocessing.Pool(
                    # limit the number of processes to not allocate too many resources concurrently
                    processes=min(12, os.cpu_count())) as pool:
                logger.info(
                    f'> processing {len(frame_timestamps)} point clouds using {pool._processes} worker processes')
                pool.map(func=process_function, iterable=process_iterable)
        else:
            # Use single process
            for arg in tqdm.tqdm(process_iterable, total=len(frame_numbers)):
                process_function(arg)

        # Save all lidar timestamps
        lidar_timestamp_save_path = os.path.join(lidar_base_save_path, 'timestamps.npz')
        np.savez(lidar_timestamp_save_path, timestamps=frame_timestamps.tolist())

        logger.info(f'> processed {len(frame_timestamps)} point clouds')

    def _decode_lidar_process(
        self,
        args,
        lidar_base_save_path,
        T_lidar_rig,
        pose_interpolator,
        vehicle_bbox_rig,
        num_frames,
        logger,
    ):
        """ Process a single lidar frame executed by a dedicated process """
        # Add PID to logger
        logger = logger.getChild(f'PID={os.getpid()}')

        # Decode current frame data to process
        continuos_frame_index = args[0]
        frame_number = args[1]
        frame_timestamp = args[2]

        source_pc_path = os.path.join(self.sequence_path, 'lidars', self.LIDAR_SENSORNAME,
                                      str(frame_number) + '.ply')
        target_pc_path = os.path.join(lidar_base_save_path,
                                      str(continuos_frame_index).zfill(self.INDEX_DIGITS) +
                                      '.dat.xz')  # store as *increasing* canonical frame IDs

        # Interpolate egomotion at frame timestamp to obtain vehicle pose at lidar end-time
        T_rig_world = pose_interpolator.interpolate_to_timestamps(frame_timestamp)[0]

        # Load point cloud (already motion-compensated)
        mesh = pcu.load_triangle_mesh(source_pc_path)

        # Remove all points with *duplicate* coordinates (these seem to be present in the input already)
        # and remember the indices of the valid points to load other attributes
        xyz, unique_input_idxs = np.unique(mesh.vertex_data.positions, axis=0, return_index=True)
        intensity = mesh.vertex_data.custom_attributes['intensity'][unique_input_idxs].flatten()
        point_count = xyz.shape[0]

        # Create 3D ray structure of 3D rays in space with accompanying metadata.
        # Format; x_s, y_s, z_s, x_e, y_e, z_e, dist, intensity, dynamic_flag
        # Dynamic flag is set to -1 if the information is not available, 0 static, 1 = dynamic

        # Determine ray start-point by interpolating poses at per-point timestamps
        if all(key in mesh.vertex_data.custom_attributes for key in ('timestamp_lo', 'timestamp_hi')):
            # Perform time-dependent per sample start-point interpolation,
            # stitching together uin64 timestamps from lo/hi parts
            ts_lo = np.array(mesh.vertex_data.custom_attributes["timestamp_lo"])[unique_input_idxs].astype(np.uint32)
            ts_hi = np.array(mesh.vertex_data.custom_attributes["timestamp_hi"])[unique_input_idxs].astype(np.uint32)
            timestamps = (np.left_shift(ts_hi, 32, dtype=np.uint64) + ts_lo).flatten()

            # Special case: allow snapping to end-of-frame timestamp for *initial* frames as
            # valid egomotion (in particular lidar-based egomotion) might not have been
            # evaluated in the past before the processed lidar frame's end-of-frame timestamp
            # (usually at start of sequence)
            if frame_timestamp - self.LIDAR_APPROX_SPIN_TIME < self.poses_timestamps[0]:
                past_idxs = timestamps < self.poses_timestamps[0]

                if np.any(past_idxs):
                    logger.info("> snapping point timestamps of *initial* spins to start of egomotion")

                timestamps[past_idxs] = frame_timestamp

            # Lidar to world poses for each point (will throw in case invalid timestamps are loaded)
            xyz_s = pose_interpolator.interpolate_to_timestamps(timestamps) @ T_lidar_rig

            # Pick lidar to world positions for each point
            xyz_s = xyz_s[:, 0:3, -1]  # N x 3
        else:
            if continuos_frame_index == 0:  # Warn once in first iteration only
                logger.warn('> no lidar point timestamps available (missing \'timestamp_lo\' / '
                            ' \'timestamp_hi\' attributes), falling back to *constant* lidar start points')

            # No per-point timestamps available, fallback to using *constant*
            # lidar origin in world frame as start point for all rays
            T_lidar_world = T_rig_world @ T_lidar_rig
            xyz_s = np.full((point_count, 3), T_lidar_world[:3, -1])  # N x 3

        # Homogeneous ray end points in lidar frame
        xyz_e = np.row_stack([xyz.transpose(), np.ones(point_count, dtype=np.float32)])  # 4 x N

        # Transform points from lidar to rig frame
        xyz_e = T_lidar_rig @ xyz_e

        # Filter points inside the vehicles bounding-box
        valid_idxs = np.logical_not(isWithin3DBBox(xyz_e[0:3, :].transpose(), vehicle_bbox_rig.reshape(1,-1)))

        # Transform points from rig to world frame + drop homogenous dimension and transpose to match output dimension
        xyz_e = T_rig_world @ xyz_e
        xyz_e = xyz_e[:-1, :].transpose()  # N x 3

        # Compute distances
        dist = np.linalg.norm(xyz_s - xyz_e, axis=1)  # N x 1

        # Compute dynamic flag / load current frame labels
        dynamic_flag, current_frame_labels = LabelProcessor.lidar_dynamic_flag(xyz, frame_timestamp, self.labels, self.frame_labels, skip_dynamic_flag=self.skip_dynamic_flag)

        # Assemble full point-cloud ray structure
        point_cloud = np.column_stack((xyz_s, xyz_e, dist, intensity, dynamic_flag))

        # Filter points based on max distance
        valid_idxs &= point_cloud[:, 6] < self.LIDAR_FILTER_MAX_DISTANCE

        point_cloud = point_cloud[valid_idxs, :]

        logger.debug(f'> filtered {point_count - valid_idxs.sum()} invalid points for '
                     f'lidar spin {continuos_frame_index}/{num_frames-1}')

        # Serialize point cloud
        save_pc_dat(target_pc_path, point_cloud)

        # Serialize per-frame labels
        # Remark: it's currently simpler to serialize per lidar-frame labels for this timestamp here, as we perform frame subsampling as part of lidar processing.
        # However, in the future we might also incorporate camera data also, and this serialization might need to be relocated.
        save_pkl(
            current_frame_labels,
            os.path.join(self.output_dir, self.session_id, self.label_save_dir,
                         str(continuos_frame_index).zfill(self.INDEX_DIGITS) + '.pkl'))

        # Store metadata of the lidar frame
        metadata = {}
        metadata['T_lidar_rig'] = T_lidar_rig  # Lidar extrinsic parameters (note: this can be assumed to be constant and could be stored only once)
        metadata['T_rig_world'] = T_rig_world  # Pose of the rig at the end of the lidar spin, can be used to transform points into a local coordinate frame
        metadata['elevation_angles'] = None  # [TODO: currently missing for NV sensors] Lidar elevation angles, can be used to simulate the lidar or recover points that did not return
        save_pkl(metadata, target_pc_path.replace('.dat.xz', '.pkl'))

        # Explicitly collect garbage to free up resources
        gc.collect()
