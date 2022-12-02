# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import logging
import re
import json
import os
import multiprocessing
import shutil

import numpy as np
import tqdm
import point_cloud_utils as pcu

from pathlib import Path
from typing import Optional, Tuple
from functools import partial

from src.dsai_internal.data_converter.data_converter import BaseNvidiaDataConverter
from src.dsai_internal.data.data2 import DataWriter
from src.dsai_internal.data.types import FThetaCameraModelParameters, LabelSource, Poses, ShutterType

from src.dsai_internal.common.nvidia_utils import (parse_rig_sensors_from_dict, sensor_to_rig, LabelProcessor,
                                        camera_intrinsic_parameters, compute_fw_polynomial, compute_ftheta_parameters,
                                        camera_car_mask, vehicle_bbox)
from src.dsai_internal.common.common import load_jsonl, PoseInterpolator, uniform_subdivide_range, platform_cpu_count, SimpleTimer
from src.dsai_internal.av_utils import isWithin3DBBox

class NvidiaMaglevConverter(BaseNvidiaDataConverter):
    """
    NVIDIA-specific data conversion (based on Maglev dsai-pp workflows data extraction)
    """

    def __init__(self, config):
        super().__init__(config)

        self.logger = logging.getLogger(__name__)

        self.seek_sec: float = config.seek_sec
        self.duration_sec: float = config.duration_sec

        self.multiprocessing_camera: bool = config.multiprocessing_camera
        self.multiprocessing_lidar: bool = config.multiprocessing_lidar
        self.max_processes: Optional[int] = config.max_processes

        self.shard_id: int = config.shard_id
        self.shard_count: int = config.shard_count
        self.shard_meta: bool = config.shard_meta

        self.symlink_camera_frames: bool = config.symlink_camera_frames

        self.egomotion_file : str = config.egomotion_file

        self.skip_dynamic_flag: bool = config.skip_dynamic_flag

    @staticmethod
    def get_sequence_dirs(config) -> list[Path]:
        return [Path(config.root_dir)]

    @staticmethod
    def from_config(config) -> NvidiaMaglevConverter:
        return NvidiaMaglevConverter(config)

    @staticmethod
    def time_bounds(timestamps_us: list[int], seek_sec: Optional[float],
                    duration_sec: Optional[float]) -> tuple[int, int]:
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

        start_timestamp_us = int(timestamps_us[0])
        end_timestamp_us = int(timestamps_us[-1])

        if seek_sec:
            assert seek_sec >= 0.0, "Require positive seek time"
            start_timestamp_us += int(seek_sec * 1e6)

        if duration_sec:
            assert duration_sec >= 0.0, "Require positive duration time"
            end_timestamp_us = start_timestamp_us + int(duration_sec * 1e6)

        assert start_timestamp_us < end_timestamp_us, "Arguments lead to invalid time bounds"

        return start_timestamp_us, end_timestamp_us

    def convert_sequence(self, sequence_path: Path) -> None:
        """
        Runs the conversion of a single session (single job output of Maglev dsai-pp workflow)
        """

        self.sequence_path = sequence_path

        # Read rig json file and sensor information
        with open(sequence_path / 'rig.json', 'r') as fp:
            self.rig = json.load(fp)

        self.sensors_calibration_data = parse_rig_sensors_from_dict(self.rig)

        # Determine session-id to be processed
        # Note: session_id in loaded rig meta might not reflect the actual current session due to bugs
        #       in the rig generation, prefer loading correct ID from rig used for egomotion for now
        with open(self.sequence_path / 'egomotion' / 'rig_egomotion_indexer.json', 'r') as fp:
            match = re.search(r'session_data/(\w{8}-\w{4}-\w{4}-\w{4}-\w{12})/', fp.read())
            if match:
                session_id = match[1]
            else:
                raise ValueError("Unable to determine trustable session_id")

        self.logger.info(f'Converting session {session_id} [shard {self.shard_id + 1}/{self.shard_count}]')

        # DataWriter for all outputs
        self.data_writer = DataWriter(
            self.output_dir / session_id,
            list(self.CAMERAID_TO_RIGNAME.keys()),
            list(self.LIDARID_TO_RIGNAME.keys()),
            []  # no radars yet
        )

        if self.shard_meta:
            # Store per-shard meta data / initial success state
            self.data_writer.store_shard_meta(self.shard_id, self.shard_count, False)

        # Decode data from maglev
        self.decode_poses()

        self.decode_labels()

        self.decode_cameras()

        self.decode_lidars()

        if self.shard_id == 0:
            self.data_writer.store_meta(
                # TODO: parse these from the data
                'scene-calib',
                'lidar-egomotion')

        if self.shard_meta:
            # Store per-shard meta data / final success state
            self.data_writer.store_shard_meta(self.shard_id, self.shard_count, True)

    def decode_poses(self):
        logger = self.logger.getChild('decode_poses')
        logger.info(f'Loading poses')

        # Initialize pose / timestamp variables
        self.T_rig_worlds = []
        self.T_rig_world_timestamps_us = []

        # Load sensor extrinsics to compute poses of the rig frame if egomotion is represented in a sensor frame
        T_rig_sensors = {
            lidar_sensor_name: np.linalg.inv(sensor_to_rig(self.sensors_calibration_data[lidar_sensor_name]))
            for lidar_sensor_name in list(self.CAMERAID_TO_RIGNAME.values()) + list(self.LIDARID_TO_RIGNAME.values())
        }

        # Load egomotion trajectory
        if not self.egomotion_file:
            # Use default egomotion jsonl location
            egomotion_file = self.sequence_path / 'egomotion/egomotion.json'
        else:
            # Use overwrite file
            egomotion_file = self.egomotion_file

        for egomotion_pose_entry in load_jsonl(egomotion_file):
            # Skip invalid poses
            if not egomotion_pose_entry['valid']:
                continue

            # Note: there is additional data like lat/long and sensor-related information
            #       which could be used in the future
            # Note: make sure all poses information is represented as f64 to have sufficient
            #       precision in case poses are representing global / map-associated coordinates
            T_rig_world_timestamp_us = int(egomotion_pose_entry['timestamp'])
            T_rig_world = np.asfarray(egomotion_pose_entry['pose'].split(' '), dtype=np.float64).reshape(
                (4, 4)).transpose()

            # Make sure poses represent *rigToWorld* transformations
            # (actually *rigToGlobal* as they include the base pose also - this is the case for non-identity initial poses)
            # Note: there currently seems to be an inconsistency in the egomotion indexer output - keep
            #       this verified workaround logic for now (might need to be adapted if egomotion indexer
            #       is fixed)
            if egomotion_pose_entry['sensor_name'] == 'dgps':
                pass
            elif egomotion_pose_entry['sensor_name'] in T_rig_sensors:
                # Convert pose in lidar frame to pose in rig frame
                T_rig_world = T_rig_world @ T_rig_sensors[egomotion_pose_entry['sensor_name']]
            else:
                raise ValueError(f"Unsupported source ego frame {egomotion_pose_entry['in_sensor_name_frame']}")

            # Sanity check on data-type
            assert T_rig_world.dtype is np.dtype('float64'), \
                "Require pose to be double-precision (to suppoglobally aligned / map-associated)"

            self.T_rig_worlds.append(T_rig_world)
            self.T_rig_world_timestamps_us.append(T_rig_world_timestamp_us)

        assert len(self.T_rig_worlds), "No valid egomotion poses loaded"

        # Stack all poses (common canonical format convention)
        self.T_rig_worlds = np.stack(self.T_rig_worlds)
        self.T_rig_world_timestamps_us = np.array(self.T_rig_world_timestamps_us, dtype=np.uint64)

        # Select refence base pose and convert all poses relative to this reference.
        # The base pose represents a worldToGlobal transformation and the first pose
        # of the trajectory defines the global frame of reference
        # (all other world poses are enconded relative to this global frame from here one,
        # allowing to represent, e.g., point world-coordinates in single f32 precision)
        self.T_rig_world_base = self.T_rig_worlds[0]
        self.T_rig_worlds = np.linalg.inv(self.T_rig_world_base) @ self.T_rig_worlds

        # Save the poses [only by main shard]
        if self.shard_id == 0:
            self.data_writer.store_poses(Poses(self.T_rig_world_base, self.T_rig_worlds, self.T_rig_world_timestamps_us))

        # Log base pose to share it more easily with downstream teams (it's serialized also explicitly)
        with np.printoptions(floatmode='unique', linewidth=200):  # print in highest precision
            logger.info(
                f'> processed {len(self.T_rig_world_timestamps_us)} poses, using base pose:\n{self.T_rig_world_base}')


    def decode_labels(self):
        logger = self.logger.getChild('decode_labels')
        logger.info(f'Loading labels')

        # Initialize annotation structs (defaults in case no labels are available loaded)
        self.track_labels: dict[int, dict] = {}
        self.frame_labels: dict[str, dict] = {}

        # Process autolabels, if available
        labels_path = self.sequence_path / 'cuboids_tracked' / 'labels_lidar.parquet'
        if not labels_path.exists():
            logger.warn(f'> file {labels_path} doesn\'t exist, skipping label generation')
            return

        # Determine time bounds from available egomotion poses and user-provided restrictions
        start_timestamp_us, end_timestamp_us = self.time_bounds(self.T_rig_world_timestamps_us, self.seek_sec,
                                                                self.duration_sec)

        # Perform label parsing
        self.track_labels, self.frame_labels = LabelProcessor.parse(labels_path, start_timestamp_us, end_timestamp_us, LabelSource.AUTOLABEL, logger)

        # Save the accumulated track [only by main shard]
        if self.shard_id == 0:
            self.data_writer.store_labels(self.track_labels)


    def decode_cameras(self):
        logger = self.logger.getChild('decode_cameras')
        logger.info(f'Loading camera data [shard {self.shard_id + 1}/{self.shard_count}]')

        # Determine time bounds from available egomotion poses
        start_timestamp_us, end_timestamp_us = self.time_bounds(self.T_rig_world_timestamps_us, self.seek_sec,
                                                                self.duration_sec)

        # Pose interpolator to obtain start / end egomotion poses
        pose_interpolator = PoseInterpolator(self.T_rig_worlds, self.T_rig_world_timestamps_us)

        # Process all camera sensors
        for camera_id, camera_rig_name in self.CAMERAID_TO_RIGNAME.items():
            logger.info(f'Processing camera {camera_rig_name}')

            camera_type = self.CAMERAID_TO_CAMERATYPE[camera_id]

            # Load frame numbers and timestamps
            frames_metadata = load_jsonl(self.sequence_path / 'cameras' / camera_rig_name / 'meta.json')
            raw_frame_numbers = np.array([frame_data['frame_number'] for frame_data in frames_metadata],
                                         dtype=np.uint64)
            raw_frame_timestamps_us = np.array([frame_data['timestamp'] for frame_data in frames_metadata],
                                               dtype=np.uint64)
            del (frames_metadata)

            assert len(raw_frame_numbers) == len(raw_frame_timestamps_us)

            # Get the frame range of the first and last frame relative to available egomotion poses and respecting exposure timings
            global_range_start = np.argmax(raw_frame_timestamps_us - self.CAMERATYPE_TO_ROLLINGSHUTTERDELAY_US[camera_type] - \
                self.CAMERATYPE_TO_EXPOSURETIME_HALF_US[camera_type] >= start_timestamp_us)
            global_range_end = np.argmax(raw_frame_timestamps_us - self.CAMERATYPE_TO_EXPOSURETIME_HALF_US[camera_type] > end_timestamp_us) \
                if raw_frame_timestamps_us[-1] - self.CAMERATYPE_TO_EXPOSURETIME_HALF_US[camera_type] > end_timestamp_us else len(raw_frame_timestamps_us
            )  # take all frames if all are within egomotion range, or determine last valid frame
            num_global_frames = global_range_end - global_range_start

            # Compute sensor-specific data (timestamps *all* frames in all shards, meta data) [only by main shard]
            if self.shard_id == 0:
                global_frame_timestamps_us = raw_frame_timestamps_us[global_range_start:global_range_end]
                global_eof_timestamps_us = global_frame_timestamps_us - self.CAMERATYPE_TO_EXPOSURETIME_HALF_US[camera_type] \

                # global camera pose timestamps, corresponding to end-of-frame timestamps
                global_camera_timestamps_us = np.stack(global_eof_timestamps_us) \
                    if global_frame_timestamps_us.size != 0 else np.empty_like(global_frame_timestamps_us) # check that at least a single frame was processed

                # Extract the calibration metadata
                camera_calibration_data = self.sensors_calibration_data[camera_rig_name]
                T_sensor_rig = sensor_to_rig(camera_calibration_data)

                # Estimate the forward polynomial
                intrinsic = camera_intrinsic_parameters(
                    camera_calibration_data, logger
                )  # TODO: make sure we return 6th-order polynomial unconditionally. Ideally also cleanup clumpsy single-array representation for intrinsics

                bw_poly = intrinsic[4:]
                fw_poly = compute_fw_polynomial(intrinsic)
                _, max_angle = compute_ftheta_parameters(np.concatenate((intrinsic, fw_poly)))

                # Constant mask image, which currently only contains the ego car mask
                # TODO: extend this with dynamic object masks
                mask_image = camera_car_mask(camera_calibration_data)

                self.data_writer.store_camera_meta(
                    camera_id, global_camera_timestamps_us, T_sensor_rig,
                    FThetaCameraModelParameters(intrinsic[2:4].astype(np.uint64), ShutterType.ROLLING_TOP_TO_BOTTOM,
                                                self.CAMERATYPE_TO_EXPOSURETIME_US[camera_type].item(), intrinsic[0:2],
                                                bw_poly, fw_poly, float(max_angle)), mask_image.get_image())

            # Apply uniform subdivision of current shard to get local data range
            local_range, local_offset = uniform_subdivide_range(self.shard_id, self.shard_count, global_range_start,
                                                                global_range_end)

            # Subsample frames to valid local ranges
            local_frame_numbers = raw_frame_numbers[local_range]
            local_frame_timestamps_us = raw_frame_timestamps_us[local_range]

            # Process all valid images
            process_function = partial(self._decode_camera_process,
                                       camera_id=camera_id,
                                       camera_rig_name=camera_rig_name,
                                       camera_type=camera_type,
                                       pose_interpolator=pose_interpolator)
            process_iterable = zip(range(local_offset,
                                         len(local_frame_numbers) + local_offset), local_frame_numbers,
                                   local_frame_timestamps_us)
            if self.multiprocessing_camera:
                # Use multiprocessing to speed up IO
                with multiprocessing.Pool(
                        # limit the number of processes to what is available in the current system / MagLev workflow
                        processes=platform_cpu_count(upper_limit=self.max_processes)) as pool:
                    logger.info(
                        f'> processing {len(local_frame_numbers)} images using {pool._processes} worker processes')
                    for _ in tqdm.tqdm(pool.imap_unordered(func=process_function, iterable=process_iterable),
                                       total=len(local_frame_numbers)):
                        pass
            else:
                # Use single process
                for arg in tqdm.tqdm(process_iterable, total=len(local_frame_numbers)):
                    process_function(arg)
            logger.info(f'> processed {len(local_frame_timestamps_us)}/{num_global_frames} local/global'
                        f' images [shard {self.shard_id + 1}/{self.shard_count}]')

        logger.info(f'> processed {len(self.CAMERAID_TO_RIGNAME)} cameras')


    def _decode_camera_process(
        self,
        args: Tuple[int, int, int],
        camera_id: str,
        camera_rig_name: str,
        camera_type: str,
        pose_interpolator: PoseInterpolator,
    ) -> None:
        """ Process a single image executed by a dedicated process """

        # Decode current frame data to process
        continous_frame_index = args[0]
        frame_number = args[1]
        frame_timestamp_us = args[2]

        # Copy / symlink image from source to target
        source_image_path = self.sequence_path / 'cameras' / camera_rig_name / (str(frame_number) + '.jpeg')

        # Interpolate the start and end pose to the timestamps of the first and last row
        timestamps_us = np.array([
            frame_timestamp_us - self.CAMERATYPE_TO_ROLLINGSHUTTERDELAY_US[camera_type] - self.CAMERATYPE_TO_EXPOSURETIME_HALF_US[camera_type], \
            frame_timestamp_us - self.CAMERATYPE_TO_EXPOSURETIME_HALF_US[camera_type]
        ])
        T_rig_worlds = pose_interpolator.interpolate_to_timestamps(timestamps_us)

        def store_image(target_image_path: Path) -> None:
            ''' Callback to store the image at the target path '''
            # Copy / symlink image from source to target
            if self.symlink_camera_frames:
                # Create symlink target -> source
                Path(target_image_path).symlink_to(source_image_path)
            else:
                # Perform explicit frame file copy
                shutil.copy(source_image_path, target_image_path)

        self.data_writer.store_camera_frame(camera_id, continous_frame_index, store_image, T_rig_worlds, timestamps_us)


    def decode_lidars(self):
        logger = self.logger.getChild('decode_lidars')
        logger.info(f'Loading lidar data [shard {self.shard_id + 1}/{self.shard_count}]')

        # Determine time bounds from available egomotion poses
        start_timestamp_us, end_timestamp_us = self.time_bounds(self.T_rig_world_timestamps_us, self.seek_sec,
                                                                self.duration_sec)

        # Pose interpolator to obtain start / end egomotion poses
        pose_interpolator = PoseInterpolator(self.T_rig_worlds, self.T_rig_world_timestamps_us)

        # Load vehicle bounding box (defined in rig frame)
        vehicle_bbox_rig = vehicle_bbox(self.rig)
        vehicle_bbox_rig[3:6] += self.LIDAR_FILTER_VEHICLE_BBOX_PADDING_METERS  # pad the bounding box slightly

        # Process all lidar sensors
        for lidar_id, lidar_rig_name in self.LIDARID_TO_RIGNAME.items():
            logger.info(f'Processing lidar {lidar_rig_name}')

            # Load extrinsics
            T_sensor_rig = sensor_to_rig(self.sensors_calibration_data[lidar_rig_name])

            # Load frame numbers and timestamps
            frames_metadata = load_jsonl(self.sequence_path / 'lidars' / lidar_rig_name / 'meta.json')
            raw_frame_numbers = np.array([frame_data['frame_number'] for frame_data in frames_metadata],
                                         dtype=np.uint64)
            raw_frame_timestamps_us = np.array([frame_data['timestamp'] for frame_data in frames_metadata],
                                               dtype=np.uint64)
            del (frames_metadata)

            assert len(raw_frame_numbers) == len(raw_frame_timestamps_us)

            # Get the frame range of the first and last frame relative to available egomotion poses
            global_range_start = np.argmax(raw_frame_timestamps_us >= start_timestamp_us)
            global_range_end = np.argmax(raw_frame_timestamps_us > end_timestamp_us) \
            if raw_frame_timestamps_us[-1] > end_timestamp_us else len(raw_frame_timestamps_us
            )  # take all frames if all are within egomotion range, or determine last valid frame
            num_global_frames = global_range_end - global_range_start

            # Store all static sensor data [only by main shard]
            if self.shard_id == 0:
                global_frame_timestamps_us = raw_frame_timestamps_us[global_range_start:global_range_end]

                self.data_writer.store_lidar_meta(lidar_id, global_frame_timestamps_us, T_sensor_rig)

            # Apply uniform subdivision of current shard to get local data range
            local_range, local_offset = uniform_subdivide_range(self.shard_id, self.shard_count, global_range_start, global_range_end)

            # Subsample frames to valid local ranges
            local_frame_numbers = raw_frame_numbers[local_range]
            local_frame_timestamps = raw_frame_timestamps_us[local_range]

            # Process all valid point clouds using multi-processing
            process_function = partial(
                self._decode_lidar_process,
                lidar_id=lidar_id,
                lidar_rig_name=lidar_rig_name,
                T_sensor_rig=T_sensor_rig,
                pose_interpolator=pose_interpolator,
                vehicle_bbox_rig=vehicle_bbox_rig,
                num_global_frames=num_global_frames,
                logger=logger,
            )
            process_iterable = zip(range(local_offset, len(local_frame_numbers) + local_offset), local_frame_numbers, local_frame_timestamps)
            if self.multiprocessing_lidar:
                # Use multiprocessing to speed up IO
                with multiprocessing.Pool(
                        # limit the number of processes to what is available in the current system / MagLev workflow
                        processes=platform_cpu_count(upper_limit=self.max_processes)) as pool:
                    logger.info(
                        f'> processing {len(local_frame_numbers)} point clouds using {pool._processes} worker processes')
                    for _ in tqdm.tqdm(pool.imap_unordered(func=process_function, iterable=process_iterable), total=len(local_frame_numbers)):
                        pass
            else:
                # Use single process
                for arg in tqdm.tqdm(process_iterable, total=len(local_frame_numbers)):
                    process_function(arg)

            logger.info(f'> processed {len(local_frame_timestamps)}/{num_global_frames} local/global'
                        f' point clouds [shard {self.shard_id + 1}/{self.shard_count}]')

        logger.info(f'> processed {len(self.LIDARID_TO_RIGNAME)} lidars')


    def _decode_lidar_process(
        self,
        args,
        lidar_id,
        lidar_rig_name,
        T_sensor_rig,
        pose_interpolator,
        vehicle_bbox_rig,
        num_global_frames,
        logger,
    ):
        """ Process a single lidar frame executed by a dedicated process """
        # Add PID to logger
        logger = logger.getChild(f'PID={os.getpid()}')

        # Decode current frame data to process
        continuos_frame_index = args[0]
        frame_number = args[1]
        frame_end_timestamp = args[2]

        source_pc_path = os.path.join(self.sequence_path, 'lidars', lidar_rig_name, str(frame_number) + '.ply')

        # Interpolate egomotion at frame end timestamp for sensor reference pose at end-of-spin time
        T_sensor_world = pose_interpolator.interpolate_to_timestamps(frame_end_timestamp)[0] @ T_sensor_rig
        T_world_sensor = np.linalg.inv(T_sensor_world)

        # Start timer
        timer = SimpleTimer()

        # Load point cloud (already motion-compensated)
        mesh = pcu.load_triangle_mesh(source_pc_path)
        time_load = timer.elapsed_sec(restart = True)

        # Remove all points with *duplicate* coordinates (these seem to be present in the input already)
        # and remember the indices of the valid points to load other attributes
        xyz, unique_input_idxs = np.unique(mesh.vertex_data.positions, axis=0, return_index=True) # Motion-compensated end-points in end-of-spin frame
        intensity = mesh.vertex_data.custom_attributes['intensity'][unique_input_idxs].flatten() / 2.55 # Intensities are oddly represented as [0.0 .. 2.55], normalize to [0.0 .. 1.0]
        point_count = xyz.shape[0]

        # Create 3D ray structure of 3D rays in sensor space with accompanying metadata.
        # Colums; xyz_s, xyz_e, intensity, dynamic_flag, timestamp
        # Dynamic flag is set to -1 if the information is not available, 0 static, 1 = dynamic

        # Determine ray start-point by interpolating poses at per-point timestamps
        if all(key in mesh.vertex_data.custom_attributes for key in ('timestamp_lo', 'timestamp_hi')):
            # Perform time-dependent per sample start-point interpolation,
            # stitching together uin64 timestamps from lo/hi parts
            ts_lo = np.array(mesh.vertex_data.custom_attributes["timestamp_lo"])[unique_input_idxs].astype(np.uint32)
            ts_hi = np.array(mesh.vertex_data.custom_attributes["timestamp_hi"])[unique_input_idxs].astype(np.uint32)
            timestamp = (np.left_shift(ts_hi, 32, dtype=np.uint64) + ts_lo).flatten()
            del(ts_lo, ts_hi)

            # Special case: allow snapping to end-of-frame timestamp for *initial* frames as
            # valid egomotion (in particular lidar-based egomotion) might not have been
            # evaluated in the past before the processed lidar frame's end-of-frame timestamp
            # (usually at start of sequence)
            if continuos_frame_index == 0:
                past_idxs = timestamp < self.T_rig_world_timestamps_us[0]

                if np.any(past_idxs):
                    logger.info("> snapping out-of-range point timestamps of *initial* spin to start of egomotion")
                    timestamp[past_idxs] = frame_end_timestamp

            # Special case: snap too far in the future point timestamps to last valid end-of-frame timestamp
            elif continuos_frame_index == num_global_frames - 1:
                future_idxs = timestamp > self.T_rig_world_timestamps_us[-1]

                if np.any(future_idxs):
                    logger.info("> snapping out-of-range point timestamps of *last* spins to end of egomotion")
                    timestamp[future_idxs] = frame_end_timestamp

            # Determine unique timestamps to only perform actually required pose interpolations (a lot of points share the same timestamp)
            timestamp_unique, unique_timestamp_reverse_idxs = np.unique(timestamp, return_inverse=True)

            # Lidar frame poses for each point (will throw in case invalid timestamps are loaded) expressed in the reference sensor's frame
            # sensor_sensorRef = world_sensorRef * sensor_world = world_sensorRef * (rig_world * sensor_rig)
            xyz_s_unique = T_world_sensor @ pose_interpolator.interpolate_to_timestamps(timestamp_unique) @ T_sensor_rig
            del(timestamp_unique)

            # Pick sensor positions (in end-of-spin pose) for each start point (blow up to original potentially non-unique timestamp range)
            xyz_s = xyz_s_unique[unique_timestamp_reverse_idxs, 0:3, -1]  # N x 3
            del(unique_timestamp_reverse_idxs)

        else:
            if continuos_frame_index == 0:  # Warn once in first iteration only
                logger.warn('> no lidar point timestamps available (missing \'timestamp_lo\' / '
                            ' \'timestamp_hi\' attributes), falling back to *constant* lidar start points')

            # No per-point timestamps available, fallback to using *constant*
            # lidar origin as start point for all rays
            timestamp = np.full((point_count), frame_end_timestamp, dtype=np.uint64)
            xyz_s = np.full((point_count, 3), [0, 0, 0])  # N x 3

        del(mesh)

        # Homogeneous ray end points in sensor frame
        xyz_e = np.row_stack([xyz.transpose(), np.ones(point_count, dtype=np.float32)])  # 4 x N

        # Transform points from sensor to rig frame
        xyz_e_rig = T_sensor_rig @ xyz_e

        # Filter points inside the vehicles bounding-box
        valid_idxs = np.logical_not(isWithin3DBBox(xyz_e_rig[0:3, :].transpose(), vehicle_bbox_rig.reshape(1,-1)))

        # Drop homogenous dimension and transpose to match output dimension
        xyz_e = xyz_e[:-1, :].transpose()  # N x 3

        # Compute distances and filter points based on max distance
        valid_idxs &= np.linalg.norm(xyz_s - xyz_e, axis=1) < self.LIDARID_TO_FILTER_MAX_DISTANCE_METERS[lidar_id]

        time_process = timer.elapsed_sec(restart = True)

        # Compute dynamic flag / load current frame labels
        dynamic_flag, frame_labels = LabelProcessor.lidar_dynamic_flag(lidar_id,
                                                                       xyz,
                                                                       frame_end_timestamp,
                                                                       self.track_labels,
                                                                       self.frame_labels,
                                                                       skip_dynamic_flag=self.skip_dynamic_flag)

        time_dynflag = timer.elapsed_sec(restart = True)

        # Subselect to valid points
        xyz_s = xyz_s[valid_idxs, :]
        xyz_e = xyz_e[valid_idxs, :]
        intensity = intensity[valid_idxs]
        dynamic_flag = dynamic_flag[valid_idxs]
        timestamp = timestamp[valid_idxs]

        # Interpolate start / end pose
        timestamps_us = np.array([np.min(timestamp), frame_end_timestamp])
        T_rig_worlds = pose_interpolator.interpolate_to_timestamps(timestamps_us)

        time_process += timer.elapsed_sec(restart = True)

        # Serialize lidar frame
        self.data_writer.store_lidar_frame(lidar_id,
                                           continuos_frame_index,
                                           xyz_s,
                                           xyz_e,
                                           intensity,
                                           timestamp,
                                           dynamic_flag,
                                           frame_labels,
                                           T_rig_worlds,
                                           timestamps_us)

        time_store = timer.elapsed_sec(restart = True)

        logger.debug(f'> spin {continuos_frame_index+1}/{num_global_frames} | load/process/dynflag/store {time_load:.2f}/{time_process:.2f}/{time_dynflag:.2f}/{time_store:.2f}sec')
