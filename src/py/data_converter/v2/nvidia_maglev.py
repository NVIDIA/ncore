# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import logging
import re
import json

import numpy as np

from pathlib import Path
from typing import Optional

from src.py.common.common import Config
from src.py.data_converter.v2.data_converter import BaseNvidiaDataConverter
from src.py.data_converter.v2.data_writer import DataWriter

from src.py.common.nvidia_utils import (parse_rig_sensors_from_dict, sensor_to_rig, LabelProcessor)
from src.py.common.common import load_jsonl


class NvidiaMaglevConverter(BaseNvidiaDataConverter):
    """
    NVIDIA-specific data conversion (based on Maglev dsai-pp workflows data extraction)
    """

    VERSION = '2.0.0'

    def __init__(self, config):
        super().__init__(config)

        self.logger = logging.getLogger(__name__)

        self.seek_sec = config.seek_sec
        self.duration_sec = config.duration_sec

        self.multiprocessing_camera = config.multiprocessing_camera
        self.multiprocessing_lidar = config.multiprocessing_lidar
        self.max_processes : Optional[int] = config.max_processes

        self.shard_id : int = config.shard_id
        self.shard_count : int = config.shard_count

        self.symlink_camera_frames : bool = config.symlink_camera_frames
        self.compress_lidar : bool = config.compress_lidar

        self.egomotion_file = config.egomotion_file

        self.skip_dynamic_flag = config.skip_dynamic_flag


    @staticmethod
    def get_sequence_dirs(config) -> list[Path]:
        return [Path(config.root_dir)]


    @staticmethod
    def from_config(config) -> NvidiaMaglevConverter:
        return NvidiaMaglevConverter(config)


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
        self.data_writer = DataWriter(self.output_dir / session_id,
                                      list(self.CAMERAID_TO_RIGNAME.keys()),
                                      list(self.LIDARID_TO_RIGNAME.keys()),
                                      [] # no radars yet
                                      )

        # Decode data from maglev
        self.decode_poses()

        self.decode_labels()

        if self.shard_id == 0:
            self.data_writer.store_meta(self.VERSION,
                                        # TODO: parse these from the data
                                        'scene-calib', 'lidar-egomotion')

    def decode_poses(self):
        logger = self.logger.getChild('decode_poses')
        logger.info(f'Loading poses')

        # Initialize pose / timestamp variables
        self.T_rig_worlds = []
        self.T_rig_world_timestamps_ms = []

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
            T_rig_world_timestamp_ms = int(egomotion_pose_entry['timestamp'])
            T_rig_world = np.asfarray(
                egomotion_pose_entry['pose'].split(' '), dtype=np.float64).reshape(
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
                raise ValueError(
                    f"Unsupported source ego frame {egomotion_pose_entry['in_sensor_name_frame']}"
                )

            # Sanity check on data-type
            assert T_rig_world.dtype is np.dtype('float64'), \
                "Require pose to be double-precision (to suppoglobally aligned / map-associated)"

            self.T_rig_world_timestamps_ms.append(T_rig_world_timestamp_ms)
            self.T_rig_worlds.append(T_rig_world)

        assert len(self.T_rig_worlds), "No valid egomotion poses loaded"

        # Stack all poses (common canonical format convention)
        self.T_rig_worlds = np.stack(self.T_rig_worlds)

        # Select refence base pose and convert all poses relative to this reference.
        # The base pose represents a worldToGlobal transformation and the first pose
        # of the trajectory defines the global frame of reference
        # (all other world poses are enconded relative to this global frame from here one,
        # allowing to represent, e.g., point world-coordinates in single f32 precision)
        self.T_rig_world_base = self.T_rig_worlds[0]
        self.T_rig_worlds = np.linalg.inv(self.T_rig_world_base) @ self.T_rig_worlds

        # Save the poses [only by main shard]
        if self.shard_id == 0:
            self.data_writer.store_poses(self.T_rig_world_base, self.T_rig_worlds, self.T_rig_world_timestamps_ms)

        # Log base pose to share it more easily with downstream teams (it's serialized also explicitly)
        with np.printoptions(floatmode='unique', linewidth=200): # print in highest precision
            logger.info(f'> processed {len(self.T_rig_world_timestamps_ms)} poses, using base pose:\n{self.T_rig_world_base}')


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
        start_timestamp_us, end_timestamp_us = self.time_bounds(self.T_rig_world_timestamps_ms, self.seek_sec, self.duration_sec)

        # Perform label parsing
        self.labels, self.frame_labels = LabelProcessor.parse(labels_path, start_timestamp_us, end_timestamp_us, logger)

        # Save the accumulated data / per frame data [only by main shard]
        if self.shard_id == 0:
            self.data_writer.store_labels(self.labels, self.frame_labels)
