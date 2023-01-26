# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

import numpy as np

from google.protobuf import text_format

from dsai.impl.data_converter.protos.deepmap import transform_pb2
from dsai.impl.common.transformations import axis_angle_trans_2_se3


def extract_sensor_2_sdc(file_path):
    ''' Extract the sensor to self driving car (SDC) rig transformation parameters 

    Args:
        file_path (string): path to the calibration file
    Out:
        (np.array): transformation from the sensor to SDC in se3 representation [m,4,4]
    '''

    # Initialize the Rigid Transform data structure

    data = transform_pb2.RigidTransform3d()

    with open(file_path, 'r') as f:
        text_format.Parse(f.read(), data)

    translation = np.array([data.translation.x, data.translation.y, data.translation.z]).reshape(-1, 3)

    rot_axis = np.array([data.axis_angle.x, data.axis_angle.y, data.axis_angle.z]).reshape(-1, 3)

    rot_angle = np.array(data.axis_angle.angle_degrees).reshape(-1, 1)

    return axis_angle_trans_2_se3(rot_axis, rot_angle, translation, degrees=True)[0]
