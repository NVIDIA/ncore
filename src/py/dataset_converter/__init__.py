# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.


import os
import logging

import numpy as np

from abc import ABC, abstractmethod

# Initialize basic top-level logger configuration
logging.basicConfig(level=logging.DEBUG,
                    format='<%(asctime)s|%(levelname)s|%(filename)s:%(lineno)d|%(name)s> %(message)s')

class DataConverter(ABC):
    '''
    Base preprocessing class used to preprocess AV datasets in a canonical representation as used in the Nvidia DriveSim-AI project. For adding a new dataset,
    please inherit this class and implement the required functions. The output data should follow the conventions defined in 
    https://gitlab-master.nvidia.com/zgojcic/drivesim-ai/-/blob/main/docs/data.md

    DISCLAIMER: THIS SOURCE CODE IS NVIDIA INTERNAL/CONFIDENTIAL. DO NOT SHARE EXTERNALLY.
    IF YOU PLAN TO USE THIS CODEBASE FOR YOUR RESEARCH, PLEASE CONTACT ZAN GOJCIC zgojcic@nvidia.com. 
    '''

    INDEX_DIGITS = 6 # the number of integer digits to pad counters in output filenames to

    class Config(object):
        """ Simple dictionary holding all options as key/value pairs """

        def __init__(self, kwargs):
            self.__dict__ = kwargs

        def __iadd__(self, other):
            """ Extend with more key/value options """
            for key, value in other.items():
                self.__dict__[key] = value

            return self


    def __init__(self, config):
        self.logger = logging.getLogger(__name__)

        self.label_save_dir = 'labels'
        self.image_save_dir = 'images'
        self.point_cloud_save_dir = 'lidar'
        self.poses_save_dir = 'poses'

        self.root_dir = config.root_dir
        self.output_dir = config.output_dir

    def create_folders(self, sequence_name):
        ''' 
        Creates the default folder structure for a given sequence

        Args: 
            sequence_name (string): unique identifier of the sequence
        '''

        seq_path = os.path.join(self.output_dir, sequence_name)

        if not os.path.isdir(seq_path):
            os.makedirs(seq_path)

        for d in [self.label_save_dir,self.image_save_dir, self.poses_save_dir, self.point_cloud_save_dir]:
            if not os.path.isdir(os.path.join(seq_path, d)):
                os.makedirs(os.path.join(seq_path, d))

        for cam in self.CAMERA_2_IDTYPERIG.keys():
            cam_id = self.CAMERA_2_IDTYPERIG[cam][0]
            if not os.path.isdir(os.path.join(seq_path, self.image_save_dir, 'image_' + cam_id)):
                os.makedirs(os.path.join(seq_path, self.image_save_dir, 'image_' + cam_id))

    def convert(self):
        self.logger.info(f"Start converting {self.sequence_pathnames} ...")

        # Perform single-threaded conversion in main thread
        for sequence_pathname in self.sequence_pathnames:
            self.convert_one(sequence_pathname)

        self.logger.info("Finished conversion ...")

    @abstractmethod
    def convert_one(self, sequence_path):
        """
        Runs dataset-specific conversion

        Args:
            sequence_path (string): path to dataset-specific raw sequence data
        
        Return:
            sub_sequence_names List[string]: names of all processed sub-sequences
        """
        pass


class BaseNvidiaDataConverter(DataConverter):
    """
    Base class for all Nvidia-specific data converters, maintaining common definitions and logic
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

    # Reference lidar sensor name
    LIDAR_SENSORNAME = 'lidar:gt:top:p128:v4p5'

    # Vehicle BBOX padding distances (for each axis) and maximum distances (in meters) for point cloud measurements (to filter points on the ego-car / out invalid points)
    LIDAR_FILTER_VEHICLE_BBOX_PADDING = np.array([0.4, 0.2, 1.0], dtype=np.float32)
    LIDAR_FILTER_MAX_DISTANCE = 100.0
