# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import logging
import re
import json
import os
import multiprocessing

import numpy as np
import tqdm

from pathlib import Path
from typing import Optional, Tuple
from functools import partial

from src.py.data_converter.v2.data_converter import BaseNvidiaDataConverter
from src.py.data_converter.v2.data_writer import DataWriter, FThetaCameraModel

from src.py.common.nvidia_utils import (parse_rig_sensors_from_dict, sensor_to_rig, LabelProcessor,
                                        camera_intrinsic_parameters, compute_fw_polynomial,
                                        camera_car_mask)
from src.py.common.common import load_jsonl, PoseInterpolator, uniform_subdivide_range, platform_cpu_count


class NvidiaMaglevConverter(BaseNvidiaDataConverter):
    """
    NVIDIA-specific data conversion (based on Maglev dsai-pp workflows data extraction)
    """

    VERSION = '2.0.0'

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

        self.symlink_camera_frames: bool = config.symlink_camera_frames
        self.compress_lidar: bool = config.compress_lidar

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

        # Decode data from maglev
        self.decode_poses()

        self.decode_labels()

        self.decode_cameras()

        if self.shard_id == 0:
            self.data_writer.store_meta(
                self.VERSION,
                # TODO: parse these from the data
                'scene-calib',
                'lidar-egomotion')

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
            self.data_writer.store_poses(self.T_rig_world_base, self.T_rig_worlds, self.T_rig_world_timestamps_us)

        # Log base pose to share it more easily with downstream teams (it's serialized also explicitly)
        with np.printoptions(floatmode='unique', linewidth=200):  # print in highest precision
            logger.info(
                f'> processed {len(self.T_rig_world_timestamps_us)} poses, using base pose:\n{self.T_rig_world_base}')

    def decode_labels(self):
        logger = self.logger.getChild('decode_labels')
        logger.info(f'Loading labels')

        # Initialize annotation structs (defaults in case no labels are available loaded)
        self.labels = {'3d_labels': {}}
        self.frame_labels = {}

        # Process autolabels, if available
        labels_path = self.sequence_path / 'cuboids_tracked' / 'labels_lidar.parquet'
        if not labels_path.exists():
            logger.warn(f'> file {labels_path} doesn\'t exist, skipping label generation')
            return

        # Determine time bounds from available egomotion poses and user-provided restrictions
        start_timestamp_us, end_timestamp_us = self.time_bounds(self.T_rig_world_timestamps_us, self.seek_sec,
                                                                self.duration_sec)

        # Perform label parsing
        self.labels, self.frame_labels = LabelProcessor.parse(labels_path, start_timestamp_us, end_timestamp_us, logger)

        # Save the accumulated data / per frame data [only by main shard]
        if self.shard_id == 0:
            self.data_writer.store_labels(self.labels, self.frame_labels)

    def decode_cameras(self):
        logger = self.logger.getChild('decode_cameras')
        logger.info(f'Loading camera data [shard {self.shard_id + 1}/{self.shard_count}]')

        # Determine time bounds from available egomotion poses
        start_timestamp_us, end_timestamp_us = self.time_bounds(self.T_rig_world_timestamps_us, self.seek_sec,
                                                                self.duration_sec)

        # Process all valid camera images
        for camera_id, camera_rig_name in self.CAMERAID_TO_RIGNAME.items():
            camera_type = self.CAMERAID_TO_CAMERATYPE[camera_id]

            logger.info(f'Processing camera {camera_rig_name}')

            # Pose interpolator to obtain start / end egomotion poses
            pose_interpolator = PoseInterpolator(self.T_rig_worlds, self.T_rig_world_timestamps_us)

            # Load frame numbers and timestamps
            frames_metadata = load_jsonl(self.sequence_path / 'cameras' / camera_rig_name / 'meta.json')
            raw_frame_numbers = np.array([frame_data['frame_number'] for frame_data in frames_metadata])
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
                    camera_calibration_data, logger)  # TODO: make sure we return 6th-order polynomial unconditionally.
                #       Ideally also cleanup clumpsy single-array representation for intrinsics

                bw_poly = intrinsic[4:]
                fw_poly = compute_fw_polynomial(intrinsic)

                # Constant mask image, which currently only contains the ego car mask
                # TODO: extend this with dynamic object masks
                mask_image = camera_car_mask(camera_calibration_data)

                self.data_writer.store_camera_meta(
                    camera_id, global_camera_timestamps_us, T_sensor_rig,
                    FThetaCameraModel(intrinsic[2:4].astype(np.uint64).tolist(), 'TOP_TO_BOTTOM',
                                      self.CAMERATYPE_TO_EXPOSURETIME_US[camera_type].item(), intrinsic[0:2].tolist(),
                                      bw_poly.tolist(), fw_poly.tolist()), mask_image.get_image())

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

        self.data_writer.store_camera_frame(camera_id, continous_frame_index, source_image_path,
                                            self.symlink_camera_frames, T_rig_worlds, timestamps_us)
