# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import numpy as np

from scipy.spatial.transform import Rotation as R

def so3_trans_2_se3(so3, trans):
    """Create a 4x4 rigid transformation matrix given so3 rotation and translation.

    Args:
        so3: rotation matrix [n,3,3]
        trans: x, y, z translation [n, 3]

    Returns:
        np.ndarray: the constructed transformation matrix [n,4,4]
    """

    if so3.ndim > 2:
        T = np.eye(4)
        T = np.tile(T,(so3.shape[0], 1, 1))
        T[:, 0:3, 0:3] = so3
        T[:, 0:3, 3] = trans.reshape(-1,3,)

    else:
        T = np.eye(4)
        T[0:3, 0:3] = so3
        T[0:3, 3] = trans.reshape(3,)

    return T

def axis_angle_trans_2_se3(rot_axis, rot_angle, trans, degrees=True):
    ''' Converts the axis/angle rotation and translation to a se3 transformation matrix
    Args:
        translation (np.array): translation vectors (x,y,z) [n,3]
        axis (np.array): the rotation axes [n,3]
        angle float/double: rotation angles either in degrees or radians [n,1]
        degrees bool: True if angle is given in degrees else False

    Out:
        (np array): transformations in a se3 matrix representation [n,4,4]
    '''

    return so3_trans_2_se3(axis_angle_2_so3(rot_axis, rot_angle, degrees), trans)




def euler_trans_2_se3(euler_angles, trans, degrees=True, seq='xyz'):
    """Create a 4x4 rigid transformation matrix given euler angles and translation.

    Args:
        euler_angles (np.array): euler angles [n,3]
        trans (Sequence[float]): x, y, z translation.
        seq string: sequence in which the euler angles are given

    Returns:
        np.ndarray: the constructed transformation matrix.
    """

    return so3_trans_2_se3(euler_2_so3(euler_angles, degrees), trans)



def axis_angle_2_so3(axis, angle, degrees=True):
    ''' Converts the axis angle representation of the so3 rotation matrix
    Args:
        axis (np.array): the rotation axes [n,3]
        angle float/double: rotation angles either in degrees or radians [n]
        degrees bool: True if angle is given in degrees else False

    Out:
        (np array): rotations given so3 matrix representation [n,3,3]
    '''
    # Treat angle (radians) below this as 0.
    cutoff_angle = 1e-9 if not degrees else np.rad2deg(1e-9)
    angle[angle < cutoff_angle] = 0.0

    # Scale the axis to have the norm representing the angle
    axis_angle = (angle/np.linalg.norm(axis, axis=1, keepdims=True)) * axis

    return R.from_rotvec(axis_angle, degrees=degrees).as_matrix()


def euler_2_so3(euler_angles, degrees=True, seq='xyz'):
    ''' Converts the euler angles representation to the so3 rotation matrix
    Args:
        euler_angles (np.array): euler angles [n,3]
        degrees bool: True if angle is given in degrees else False
        seq string: sequence in which the euler angles are given

    Out:
        (np array): rotations given so3 matrix representation [n,3,3]
    '''

    return R.from_euler(seq=seq, angles=euler_angles, degrees=degrees).as_matrix().astype(np.float32)


def axis_angle_2_quaternion(axis, angle, degrees=True):
    ''' Converts the axis angle representation of the rotation to a unit quaternion
    Args:
        axis (np.array): the rotation axis [n,3]
        angle float/double: rotation angle either in degrees or radians [n,1]
        degrees bool: True if angle is given in degrees else False

    Out:
        (np array): rotation given in unit quaternion [n,4]
    '''
    return axis_angle_2_so3(axis, angle, degrees).as_quat()


def so3_2_axis_angle(so3, degrees=True):
    ''' Converts the so3 representation to axis_angle
    Args:
        so3 (np.array): the rotation matrices [n,3,3]
        degrees bool: True if angle should be given in degrees

    Out:
        axis (np array): the rotation axis [n,3]
        angle (np array): the rotation angles, either in degrees (if degrees=True) or radians [n,]
    '''
    rot_vec = R.from_matrix(so3).as_rotvec(degrees=degrees)

    angle = np.linalg.norm(rot_vec, axis=-1, keepdims=True)
    axis = rot_vec / angle

    return axis, angle


def euclidean_2_spherical_coords(coords):

    r = np.linalg.norm(coords, axis=-1, keepdims=True)
    el = np.arctan(coords[:,2]/np.linalg.norm(coords[:,:2], axis=-1)).reshape(-1,1)
    az = np.arctan2(coords[:,1],coords[:,0]).reshape(-1,1)

    return np.concatenate((r,az,el),axis=-1)

def spherical_2_direction(spherical_coords):

    dx = np.cos(spherical_coords[:,2]) * np.cos(spherical_coords[:,1])
    dy = np.cos(spherical_coords[:,2]) * np.sin(spherical_coords[:,1])
    dz = np.sin(spherical_coords[:,2])

    return np.concatenate((dx[:,None],dy[:,None],dz[:,None]),axis=-1)

def transform_point_cloud(pc, T):
    ''' Transform the point cloud with the provided transformation matrix, 
        support torch.Tensor and np.ndarry. 
    Args:
        pc (np.array): point cloud coordinates (x,y,z) [num_pts, 3] or [bs, num_pts, 3]
        T (np.array): se3 transformation matrix  [4, 4] or [bs, 4, 4]

    Out:
        (np array): transformed point cloud coordinated [num_pts, 3] or [bs, num_pts, 3]
    '''
    if len(pc.shape) == 3:
        if isinstance(pc, np.ndarray):
            trans_pts = T[:, :3, :3] @ pc.transpose(0, 2, 1) + T[:, :3, 3:4]
            return trans_pts.transpose(0, 2, 1)
        else:
            trans_pts = T[:, :3, :3] @ pc.permute(0, 2, 1) + T[:, :3, 3:4]
            return trans_pts.permute(0, 2, 1)

    else:
        trans_pts = T[:3, :3] @ pc.transpose() + T[:3, 3:4]
        return trans_pts.transpose()


def transform_bbox(bbox_source: np.ndarray, T_source_target: np.ndarray) -> np.ndarray:
    ''' Applies a rigid-transformation to a bounding box 
     Args:
        bbox (np.ndarray): bounding-box in source-frame parameterized by [x, y, z, length, width, height, eulerX, eulerY, eulerZ]
        T (np.array): se3 source->target transformation matrix to apply [4,4]
     Out:
        (np array): transformed bounding-box [num_pts, 3] or [bs, num_pts, 3]
     '''

    # Convert bbox to corresponding pose
    T_bbox_source = np.block(
        [[R.from_euler('xyz', bbox_source[6:9], degrees=False).as_matrix(),
          np.array(bbox_source[:3]).reshape((3, 1))], [np.array([0, 0, 0, 1])]])

    # Apply transformation
    T_bbox_target = T_source_target @ T_bbox_source

    # Convert back to bbox parametrization
    bbox_target = np.empty_like(bbox_source)
    bbox_target[:3] = T_bbox_target[:3, 3]  # centroid
    bbox_target[3:6] = bbox_source[3:6]  # dimensions stay unchanged
    bbox_target[6:9] = R.from_matrix(T_bbox_target[:3, :3]).as_euler('xyz', degrees=False)  # orientation
    return bbox_target


def se3_inverse(T: np.ndarray) -> np.ndarray:
    ''' Computed the inverse of a rigid transformation 
    Args:
        T (np.array): se3 transformation matrix to invert [4,4]

    Out:
        (np array): inverse transformation [4,4]
    '''
    Rt = T[:3, :3].transpose()
    return np.block([[Rt, - Rt @ T[:3,3:]], [np.zeros((1, 3)), 1]])


def local_ENU_2_ECEF_orientation(theta, phi):
    ''' Computes the rotation matrix between the world_pose and ECEF coordinate system
    Args:
        theta (np.array): theta coordinates in radians [n,1]
        phi (np.array): phi coordinates in radians [n,1]
    Out:
        (np.array): rotation from world pose to ECEF in so3 representation [n,3,3]
    '''
    z_dir = np.concatenate([(np.sin(theta)*np.cos(phi))[:,None],
                            (np.sin(theta)*np.sin(phi))[:,None],
                            (np.cos(theta))[:,None] ],axis=1)
    z_dir = z_dir/np.linalg.norm(z_dir, axis=-1, keepdims=True)

    y_dir = np.concatenate([-(np.cos(theta)*np.cos(phi))[:,None],
                            -(np.cos(theta)*np.sin(phi))[:,None],
                            (np.sin(theta))[:,None] ],axis=1)
    y_dir = y_dir/np.linalg.norm(y_dir, axis=-1, keepdims=True)

    x_dir = np.cross(y_dir, z_dir)

    return np.concatenate([x_dir[:,:,None], y_dir[:,:,None], z_dir[:,:,None]], axis = -1)


def lat_lng_alt_2_ECEF_elipsoidal(lat_lng_alt, a, b):
    ''' Converts the GPS (lat,long, alt) coordinates to the ECEF ones based on the ellipsoidal earth model
    Args:
        lat_lng_alt (np.array): latitude, longitude and altitude coordinate (in degrees and meters) [n,3]
        a (float/double): Semi-major axis of the ellipsoid
        b (float/double): Semi-minor axis of the ellipsoid
    Out:
        (np.array): ECEF coordinates[n,3]
    '''

    phi =  np.deg2rad(lat_lng_alt[:, 0])
    gamma =  np.deg2rad(lat_lng_alt[:, 1])

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)
    e_square = (a * a - b * b) / (a * a)

    N = a / np.sqrt(1 - e_square * sin_phi * sin_phi)


    x = (N + lat_lng_alt[:, 2]) * cos_phi * cos_gamma
    y = (N + lat_lng_alt[:, 2]) * cos_phi * sin_gamma
    z = (N *  (b*b)/(a*a) + lat_lng_alt[:, 2]) * sin_phi

    return np.concatenate([x[:,None] ,y[:,None], z[:,None]], axis=1 )

def translation_2_lat_lng_alt_spherical(translation, earth_radius):
    ''' Computes the translation in the ECEF to latitude, longitude, altitude based on the spherical earth model
    Args:
        translation (np.array): translation in the ECEF coordinate frame (in meters) [n,3]
        earth_radius (float/double): earth radius
    Out:
        (np.array): latitude, longitude and altitude [n,3]
    '''
    altitude = np.linalg.norm(translation, axis=-1) - earth_radius
    latitude = np.rad2deg(90 - np.arccos(translation[:,2] / np.linalg.norm(translation, axis=-1, keepdims=True)))
    longitude =  np.rad2deg(np.arctan2(translation[:,1],translation[:,0]))

    return np.concatenate([latitude[:,None], longitude[:,None], altitude[:,None]], axis=1)

def translation_2_lat_lng_alt_ellipsoidal(translation, a, f):
    ''' Computes the translation in the ECEF to latitude, longitude, altitude based on the ellipsoidal earth model
    Args:
        translation (np.array): translation in the ECEF coordinate frame (in meters) [n,3]
        a (float/double): Semi-major axis of the ellipsoid
        f (float/double): flattening factor of the earth
 radius
    Out:
        (np.array): latitude, longitude and altitude [n,3]
    '''

    # Compute support parameters
    f0 = (1 - f) * (1 - f)
    f1 = 1 - f0
    f2 = 1 / f0 - 1

    z_div_1_f =  translation[:,2] / (1 - f)
    x2y2 = np.square(translation[:,0]) + np.square(translation[:,1])

    x2y2z2 = x2y2 + z_div_1_f*z_div_1_f
    x2y2z2_pow_3_2 = x2y2z2 * np.sqrt(x2y2z2)

    gamma = (x2y2z2_pow_3_2 + a * f2 * z_div_1_f * z_div_1_f) / (x2y2z2_pow_3_2 - a * f1 * x2y2) *  translation[:,2] / np.sqrt(x2y2)

    longitude = np.rad2deg(np.arctan2(translation[:,1], translation[:,0]))
    latitude = np.rad2deg(np.arctan(gamma))
    altitude = np.sqrt(1 + np.square(gamma)) * (np.sqrt(x2y2) - a / np.sqrt(1 + f0 * np.square(gamma)))

    return np.concatenate([latitude[:,None], longitude[:,None], altitude[:,None]], axis=1)

def lat_lng_alt_2_ecef(lat_lng_alt, orientation_axis, orientation_angle, earth_model='WGS84'):
    ''' Computes the transformation from the world pose coordiante system to the earth centered earth fixed (ECEF) one
    Args:
        lat_lng_alt (np.array): latitude, longitude and altitude coordinate (in degrees and meters) [n,3]
        orientation_axis (np.array): orientation in the local ENU coordinate system [n,3]
        orientation_angle (np.array): orientation angle of the local ENU coordinate system in degrees [n,1]
        earth_model (string): earth model used for conversion (spheric will be unaccurate when maps are large)
    Out:
        trans (np.array): transformation parameters from world pose to ECEF coordinate system in se3 form (n, 4, 4)
    '''
    n = lat_lng_alt.shape[0]
    trans = np.tile(np.eye(4).reshape(1,4,4),[n,1,1])

    theta = np.deg2rad(90. - lat_lng_alt[:, 0])
    phi = np.deg2rad(lat_lng_alt[:, 1])

    R_enu_ecef = local_ENU_2_ECEF_orientation(theta, phi)


    if earth_model == 'WGS84':
        a = 6378137.0
        flattening = 1.0 / 298.257223563
        b = a * (1.0 - flattening)
        translation = lat_lng_alt_2_ECEF_elipsoidal(lat_lng_alt, a, b)

    elif earth_model == 'sphere':
        earth_radius = 6378137.0 # Earth radius in meters
        z_dir =  np.concatenate([(np.sin(theta)*np.cos(phi))[:,None],
                            (np.sin(theta)*np.sin(phi))[:,None],
                            (np.cos(theta))[:,None] ],axis=1)

        translation = (earth_radius + lat_lng_alt[:, -1])[:,None] * z_dir

    else:
        raise ValueError ("Selected ellipsoid not implemented!")

    world_pose_orientation = axis_angle_2_so3(orientation_axis, orientation_angle)

    trans[:,:3,:3] =  R_enu_ecef @ world_pose_orientation
    trans[:,:3,3] =  translation

    return trans


def ecef_2_lat_lng_alt(trans, earth_model='WGS84'):
    ''' Converts the transformation from the earth centered earth fixed (ECEF) coordinate frame to the world pose
    Args:
        trans (np.array): transformation parameters in ECEF [n,4,4]
        earth_model (string): earth model used for conversion (spheric will be unaccurate when maps are large)
    Out:
        lat_lng_alt (np.array): latitude, longitude and altitude coordinate (in degrees and meters) [n,3]
        orientation_axis (np.array): orientation in the local ENU coordinate system [n,3]
        orientation_angle (np.array): orientation angle of the local ENU coordinate system in degrees [n,1]
    '''

    translation = trans[:,:3,3]
    rotation = trans[:,:3,:3]

    if earth_model == 'WGS84':
        a = 6378137.0
        flattening = 1.0 / 298.257223563
        lat_lng_alt = translation_2_lat_lng_alt_ellipsoidal(translation, a, flattening)

    elif earth_model == 'sphere':
        earth_radius = 6378137.0 # Earth radius in meters
        lat_lng_alt = translation_2_lat_lng_alt_spherical(translation, earth_radius)

    else:
        raise ValueError ("Selected ellipsoid not implemented!")


    # Compute the orientation axis and angle
    theta = np.deg2rad((90. - lat_lng_alt[:, 0]))
    phi = np.deg2rad(lat_lng_alt[:, 1])

    R_ecef_enu = local_ENU_2_ECEF_orientation(theta, phi).transpose(0,2,1)

    orientation = R_ecef_enu @ rotation
    orientation_axis, orientation_angle = so3_2_axis_angle(orientation)


    return lat_lng_alt, orientation_axis, orientation_angle

def ecef_2_ENU(loc_ref_point: np.ndarray, earth_model: str ='WGS84'):
    ''' 
    Compute the transformation matrix that transforms points from the ECEF to a local ENU coordinate frame
    Args:
        loc_ref_point: GPS coordinates of the local reference point of the map [1,3]
        earth_model: earth model used for conversion (spheric will be unaccurate when maps are large)
    Out:
        T_ecef_enu: transformation matrix from ECEF to ENU [4,4]
    '''

    # initialize the transformation to identity
    T_ecef_enu = np.eye(4)

    if earth_model == 'WGS84':
        a = 6378137.0
        flattening = 1.0 / 298.257223563
        b = a * (1.0 - flattening)
        translation = lat_lng_alt_2_ECEF_elipsoidal(loc_ref_point, a, b).reshape(3,1)

    elif earth_model == 'sphere':
        earth_radius = 6378137.0 # Earth radius in meters
        z_dir =  np.concatenate([(np.sin(loc_ref_point[1])*np.cos(loc_ref_point[0]))[:,None],
                            (np.sin(loc_ref_point[1])*np.sin(loc_ref_point[0]))[:,None],
                            (np.cos(loc_ref_point[0]))[:,None] ],axis=1)

        translation = ((earth_radius + loc_ref_point[:, -1])[:,None] * z_dir).reshape(3,1)

    else:
        raise ValueError ("Selected ellipsoid not implemented!")

    rad_lat = np.deg2rad(loc_ref_point[:, 0])
    rad_lon = np.deg2rad(loc_ref_point[:, 1])
    T_ecef_enu[:3,:3] = np.array([[-np.sin(rad_lon), np.cos(rad_lon), 0],
                  [ -np.sin(rad_lat)*np.cos(rad_lon), -np.sin(rad_lat)*np.sin(rad_lon),  np.cos(rad_lat)],
                  [  np.cos(rad_lat)*np.cos(rad_lon), np.cos(rad_lat)*np.sin(rad_lon),  np.sin(rad_lat)]])

    T_ecef_enu[:3,3:4] = -T_ecef_enu[:3,:3] @ translation

    return T_ecef_enu
