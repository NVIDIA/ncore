# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import logging

from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional

import numpy as np


class DataConverter(ABC):
    '''
    Base preprocessing class used to preprocess AV datasets in a canonical representation as used in the Nvidia NRECore-SDK project.
    
    For adding a new dataset, please inherit this class and implement the required functions. 
    
    The output data should follow the conventions defined in 
    https://toronto_dl_lab.gitlab-master-pages.nvidia.com/ncore/notes/conventions.html

    Please also use the facilities of the 'data_writer' module, which simplifies adding new datasets.

    DISCLAIMER: THIS SOURCE CODE IS NVIDIA INTERNAL/CONFIDENTIAL. DO NOT SHARE EXTERNALLY.
    IF YOU PLAN TO USE THIS CODEBASE FOR YOUR RESEARCH, PLEASE CONTACT ZAN GOJCIC zgojcic@nvidia.com / JANICK MARTINEZ ESTURO janickm@nvidia.com. 
    '''
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)

        self.root_dir = Path(config.root_dir)
        self.output_dir = Path(config.output_dir)

    @classmethod
    def convert(cls, config) -> None:
        '''
        Main entry-point to perform conversion of all sequences
        '''

        logger = logging.getLogger(__name__)

        sequence_dirs = cls.get_sequence_paths(config)

        logger.info(f'Start converting {sequence_dirs} ...')

        # create new instance of converter for each task and execute synchonously
        for sequence_dir in sequence_dirs:
            converter = cls.from_config(config)
            converter.convert_sequence(sequence_dir)

        logger.info(f'Finished converting {sequence_dirs} in {config.output_dir} ...')

    @staticmethod
    @abstractmethod
    def get_sequence_paths(config) -> list[Path]:
        '''
        Return sequence pathnames to process
        '''
        pass

    @staticmethod
    @abstractmethod
    def from_config(config) -> DataConverter:
        '''
        Return an instance of the data converter
        '''
        pass

    @abstractmethod
    def convert_sequence(self, sequence_path: Path) -> None:
        '''
        Runs dataset-specific conversion for a sequence referenced by a directory/file path
        '''
        pass


class BaseNvidiaDataConverter(DataConverter):
    '''
    Base class for all Nvidia-specific data converters, maintaining common definitions and logic
    '''

    # Common constants
    @dataclass
    class Constants:
        # Vehicle BBOX padding distances (for each axis) and maximum distances (in meters) for point cloud measurements (to filter points on the ego-car / out invalid points)
        LIDAR_FILTER_VEHICLE_BBOX_PADDING_METERS = np.array([1.0, 0.2, 1.0], dtype=np.float32)

    # Constants for *Hyperion8* sensor-set
    @dataclass
    class Hyperion8Constants(Constants):
        CAMERAID_TO_RIGNAME = {
            'camera_front_wide_120fov': 'camera:front:wide:120fov',
            'camera_cross_left_120fov': 'camera:cross:left:120fov',
            'camera_cross_right_120fov': 'camera:cross:right:120fov',
            'camera_rear_left_70fov': 'camera:rear:left:70fov',
            'camera_rear_right_70fov': 'camera:rear:right:70fov',
            'camera_rear_tele_30fov': 'camera:rear:tele:30fov',
            'camera_front_fisheye_200fov': 'camera:front:fisheye:200fov',
            'camera_left_fisheye_200fov': 'camera:left:fisheye:200fov',
            'camera_right_fisheye_200fov': 'camera:right:fisheye:200fov',
            'camera_rear_fisheye_200fov': 'camera:rear:fisheye:200fov'
        }

        # Upper field-of-view limit across all cameras (in particular for fisheye cameras)
        MAX_CAMERA_FOV_DEG = 200.0

        # Per-camera sensor types
        CAMERAID_TO_SENSORTYPE = {
            'camera_front_wide_120fov': 'AR0820',
            'camera_cross_left_120fov': 'AR0820',
            'camera_cross_right_120fov': 'AR0820',
            'camera_rear_left_70fov': 'AR0820',
            'camera_rear_right_70fov': 'AR0820',
            'camera_rear_tele_30fov': 'AR0820',
            'camera_front_fisheye_200fov': 'IMX390',
            'camera_left_fisheye_200fov': 'IMX390',
            'camera_right_fisheye_200fov': 'IMX390',
            'camera_rear_fisheye_200fov': 'IMX390',
        }

        # Sensor-specific exposure times (rounded to integer US)
        SENSORTYPE_TO_EXPOSURETIME_HALF_US = {
            'AR0820': np.uint64(1641.58 / 2),
            'IMX390': np.uint64(10987.00 / 2)
        }  # rounded to integer US

        SENSORTYPE_TO_ROLLINGSHUTTERDELAY_US = {
            'AR0820': np.uint64(31611.55),
            'IMX390': np.uint64(32561.63)
        }  # rounded to integer US

        LIDARID_TO_RIGNAME = {
            'lidar_gt_top_p128_v4p5': 'lidar:gt:top:p128:v4p5',
        }

        LIDARID_TO_FILTER_MAX_DISTANCE_METERS = {'lidar_gt_top_p128_v4p5': 100.0}

    # Constants for *Hyperion8.1* sensor-set
    @dataclass
    class Hyperion81Constants(Constants):
        CAMERAID_TO_RIGNAME = {
            'camera_front_wide_120fov': 'camera:front:wide:120fov',
            'camera_cross_left_120fov': 'camera:cross:left:120fov',
            'camera_cross_right_120fov': 'camera:cross:right:120fov',
            'camera_rear_left_70fov': 'camera:rear:left:70fov',
            'camera_rear_right_70fov': 'camera:rear:right:70fov',
            'camera_rear_tele_30fov': 'camera:rear:tele:30fov',
            'camera_front_fisheye_200fov': 'camera:front:fisheye:200fov',
            'camera_left_fisheye_200fov': 'camera:left:fisheye:200fov',
            'camera_right_fisheye_200fov': 'camera:right:fisheye:200fov',
            'camera_rear_fisheye_200fov': 'camera:rear:fisheye:200fov'
        }

        # Upper field-of-view limit across all cameras (in particular for fisheye cameras)
        MAX_CAMERA_FOV_DEG = 200.0

        # Per-camera types
        CAMERAID_TO_SENSORTYPE = {
            'camera_front_wide_120fov': 'IMX728',
            'camera_cross_left_120fov': 'IMX728',
            'camera_cross_right_120fov': 'IMX728',
            'camera_rear_left_70fov': 'IMX728',
            'camera_rear_right_70fov': 'IMX728',
            'camera_rear_tele_30fov': 'IMX728',
            'camera_front_fisheye_200fov': 'IMX623',
            'camera_left_fisheye_200fov': 'IMX623',
            'camera_right_fisheye_200fov': 'IMX623',
            'camera_rear_fisheye_200fov': 'IMX623',
        }

        # Sensor-specific exposure times (rounded to integer US)
        # (see timing measurements in https://docs.google.com/spreadsheets/d/1khZRA0J2KBQrVTBmP5Z1cqCc4lIXXQUniNv9977fDUg/edit#gid=787900965)
        SENSORTYPE_TO_EXPOSURETIME_HALF_US = {
            'IMX728': np.uint64(10000 / 2),
            'IMX623': np.uint64(10000 / 2)
        }  # rounded to integer US

        SENSORTYPE_TO_ROLLINGSHUTTERDELAY_US = {
            'IMX728': np.uint64(30559.06582),
            'IMX623': np.uint64(30061.61757)
        }  # rounded to integer US

        LIDARID_TO_RIGNAME = {
            'lidar_gt_top_p128': 'lidar:gt:top:p128',
        }

        LIDARID_TO_FILTER_MAX_DISTANCE_METERS = {'lidar_gt_top_p128': 100.0}

    @classmethod
    def get_constants(cls, rig_properties: dict[str, str],
                      rig_sensor_ids: Sequence[str]) -> Hyperion8Constants | Hyperion81Constants:
        ''' Parse rig properties into constants for a given platform '''

        constants: Optional[BaseNvidiaDataConverter.Hyperion8Constants
                            | BaseNvidiaDataConverter.Hyperion81Constants] = None

        # Determine major platform version from 'platform_name' property
        if platform_name := rig_properties.get('platform_name', None):
            if platform_name.startswith('hy8.1_') or platform_name.startswith('hy8.1p_'):
                constants = cls.Hyperion81Constants()
            elif platform_name.startswith('hy8_'):
                constants = cls.Hyperion8Constants()

        # Older rigs use 'layout' property
        if layout := rig_properties.get('layout', None):
            if layout.startswith('hyperion_8_'):
                constants = cls.Hyperion8Constants()

        assert isinstance(constants,
                          (BaseNvidiaDataConverter.Hyperion8Constants, BaseNvidiaDataConverter.Hyperion81Constants
                           )), f'Unknown / unsupported platform in rig-properties {rig_properties}'

        # Remove sensors that are not in 'rig_sensor_ids'
        camera_ids_to_remove: list[str] = []
        for camera_id, rig_sensor_id in constants.CAMERAID_TO_RIGNAME.items():
            if rig_sensor_id not in rig_sensor_ids:
                camera_ids_to_remove.append(camera_id)
        for key in camera_ids_to_remove:
            constants.CAMERAID_TO_RIGNAME.pop(key)
            constants.CAMERAID_TO_SENSORTYPE.pop(key)

        lidar_ids_to_remove: list[str] = []
        for lidar_id, rig_sensor_id in constants.LIDARID_TO_RIGNAME.items():
            if rig_sensor_id not in rig_sensor_ids:
                lidar_ids_to_remove.append(lidar_id)
        for key in lidar_ids_to_remove:
            constants.LIDARID_TO_RIGNAME.pop(key)

        return constants
