# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import logging

import numpy as np

from pathlib import Path
from abc import ABC, abstractmethod

# Initialize basic top-level logger configuration
logging.basicConfig(level=logging.DEBUG,
                    format='<%(asctime)s|%(levelname)s|%(filename)s:%(lineno)d|%(name)s> %(message)s')


class DataConverter(ABC):
    '''
    Base preprocessing class used to preprocess AV datasets in a canonical representation (V2) as used in the Nvidia DriveSim-AI project.
    
    For adding a new dataset, please inherit this class and implement the required functions. 
    
    The output data should follow the conventions defined in 
    https://toronto_dl_lab.gitlab-master-pages.nvidia.com/dsai/notes/conventions.html

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

        sequence_dirs = cls.get_sequence_dirs(config)

        logger.info(f'Start converting {sequence_dirs} ...')

        # create new instance of converter for each task and execute synchonously
        for sequence_dir in sequence_dirs:
            converter = cls.from_config(config)
            converter.convert_sequence(sequence_dir)
        
        logger.info(f'Finished converting {sequence_dirs} in {config.output_dir} ...')


    @staticmethod
    @abstractmethod
    def get_sequence_dirs(config) -> list[Path]:
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
    def convert_sequence(self, sequence_dir: Path) -> None:
        '''
        Runs dataset-specific conversion for a sequence
        '''
        pass


class BaseNvidiaDataConverter(DataConverter):
    '''
    Base class for all Nvidia-specific data converters, maintaining common definitions and logic
    '''

    ## Constants defined for *Hyperion8* sensor-set

    # Camera exposure times (rounded to integer US)
    CAMERATYPE_TO_EXPOSURETIME_US = {'wide': np.uint64(1641.58), 'fisheye': np.uint64(10987.00)} # rounded to integer US
    CAMERATYPE_TO_EXPOSURETIME_HALF_US = {'wide': np.uint64(1641.58 / 2), 'fisheye': np.uint64(10987.00 / 2)} # rounded to integer US
    CAMERATYPE_TO_ROLLINGSHUTTERDELAY_US = {'wide': np.uint64(31611.55), 'fisheye': np.uint64(32561.63)} # rounded to integer US

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

    LIDARID_TO_RIGNAME = {
        'lidar_gt_top_p128_v4p5': 'lidar:gt:top:p128:v4p5',
    }

    # Per-camera types
    CAMERAID_TO_CAMERATYPE = {
        'camera_front_wide_120fov': 'wide',
        'camera_cross_left_120fov': 'wide',
        'camera_cross_right_120fov': 'wide',
        'camera_rear_left_70fov': 'wide',
        'camera_rear_right_70fov': 'wide',
        'camera_rear_tele_30fov': 'wide',
        'camera_front_fisheye_200fov': 'fisheye',
        'camera_left_fisheye_200fov': 'fisheye',
        'camera_right_fisheye_200fov': 'fisheye',
        'camera_rear_fisheye_200fov': 'fisheye',
    }

    # Approximate spin time in microseconds
    LIDARID_TO_APPROX_SPIN_TIME_US = {
        'lidar_gt_top_p128_v4p5': 1e6 / 10  # based on 10Hz frequency
    }

    # Vehicle BBOX padding distances (for each axis) and maximum distances (in meters) for point cloud measurements (to filter points on the ego-car / out invalid points)
    LIDAR_FILTER_VEHICLE_BBOX_PADDING_METERS = np.array([0.4, 0.2, 1.0], dtype=np.float32)
    LIDARID_TO_FILTER_MAX_DISTANCE_METERS = {
        'lidar_gt_top_p128_v4p5': 100.0 
    }
