# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import logging
import re
import json

import numpy as np

from pathlib import Path
from typing import Optional, Type

from src.py.common.common import Config
from src.py.data_converter.v2.data_converter import BaseNvidiaDataConverter
from src.py.data_converter.v2.data_writer import DataWriter

from src.py.common.nvidia_utils import (parse_rig_sensors_from_dict, sensor_to_rig)
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
    def from_config(config):
        return NvidiaMaglevConverter(config)


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

        # self.decode_labels()

        if self.shard_id == 0:
            self.data_writer.store_meta(self.VERSION, 
                                        # TODO: parse these from the data
                                        'scene-calib', 'lidar-egomotion')

    def decode_poses(self):
        logger = self.logger.getChild('decode_poses')
        logger.info(f'Loading poses')

        # Initialize pose / timestamp variables
        self.poses = []
        self.poses_timestamps = []

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
            egomotion_pose_timestamp = int(egomotion_pose_entry['timestamp'])
            egomotion_pose = np.asfarray(
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
                egomotion_pose = egomotion_pose @ T_rig_sensors[egomotion_pose_entry['sensor_name']]
            else:
                raise ValueError(
                    f"Unsupported source ego frame {egomotion_pose_entry['in_sensor_name_frame']}"
                )

            # Sanity check on data-type
            assert egomotion_pose.dtype is np.dtype('float64'), \
                "Require pose to be double-precision (to suppoglobally aligned / map-associated)"

            self.poses_timestamps.append(egomotion_pose_timestamp)
            self.poses.append(egomotion_pose)

        assert len(self.poses), "No valid egomotion poses loaded"

        # Stack all poses (common canonical format convention)
        self.poses = np.stack(self.poses)

        # Select refence base pose and convert all poses relative to this reference.
        # The base pose represents a worldToGlobal transformation and the first pose
        # of the trajectory defines the global frame of reference
        # (all other world poses are enconded relative to this global frame from here one,
        # allowing to represent, e.g., point world-coordinates in single f32 precision)
        self.base_pose = self.poses[0]
        self.poses = np.linalg.inv(self.base_pose) @ self.poses

        # Save the poses [only by main shard]
        if self.shard_id == 0:
            self.data_writer.store_poses(self.base_pose, self.poses, self.poses_timestamps)

        # Log base pose to share it more easily with downstream teams (it's serialized also explicitly)
        with np.printoptions(floatmode='unique', linewidth=200): # print in highest precision
            logger.info(f'> processed {len(self.poses_timestamps)} poses, using base pose:\n{self.base_pose}')
