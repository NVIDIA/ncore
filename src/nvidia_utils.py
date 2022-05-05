import json
import numpy as np
from protos import transform_pb2, camera_calibration_pb2
from google.protobuf import text_format
from scipy.spatial.transform import Rotation as R
from scipy.optimize import curve_fit
from protos import transform_pb2, camera_calibration_pb2
from google.protobuf import text_format
from numpy.polynomial.polynomial import Polynomial
from src.common import so3_2_axis_angle, axis_angle_2_so3, PoseInterpolator

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


    translation = np.array([data.translation.x, 
                            data.translation.y, 
                            data.translation.z]).reshape(-1,3)

    rot_axis = np.array([data.axis_angle.x, 
                         data.axis_angle.y, 
                         data.axis_angle.z]).reshape(-1,3)

    rot_angle = np.array(data.axis_angle.angle_degrees).reshape(-1,1)


    return extract_se3_transformations(translation, rot_axis, rot_angle)[0]


def extract_camera_calibration(file_path):
    ''' Extract the camera calibration parameters

    Args:
        file_path (string): path to the calibration file
    Out:
        intrinsic (np.array): camera intrinsic parameters [1,9]
        img_width (float): image width in pixels
        img_height (float): image height in pixels
        roll_shutter_delay (float): rolling shutter offset between the first and last row
    '''  

    # Initialize the Rigid Transform data structure
    data = camera_calibration_pb2.MonoCalibrationParameters()

    with open(file_path, 'r') as f:
        text_format.Parse(f.read(), data)

    intrinsic = np.array(data.camera_matrix.data)
    img_width = float(data.image_width)
    img_height = float(data.image_height)
    roll_shutter_delay = float(data.rolling_shutter_delay_microseconds)

    return intrinsic, img_width, img_height, roll_shutter_delay

def extract_pose(data, earth_model='WGS84'):
    ''' Extract the pose of the SDC  

    Args:
        data (dict): pose data
    Out:
        (np.array): transformation from lidar to SDC in se3 representation [m,4,4]
    '''  


    lat_lng_alt = np.array([data['lat_lng_alt']['latitude_degrees'], 
                            data['lat_lng_alt']['longitude_degrees'], 
                            data['lat_lng_alt']['altitude_meters']]).reshape(-1,3)

    rot_axis = np.array([data['axis_angle']['x'], 
                         data['axis_angle']['y'], 
                         data['axis_angle']['z']]).reshape(-1,3)

    rot_angle = np.array(data['axis_angle']['angle_degrees']).reshape(-1,1)

    return lat_lng_alt_2_ecef(lat_lng_alt, rot_axis, rot_angle, earth_model)[0]


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


def lat_lng_alt_2_translation_ellipsoidal(lat_lng_alt, a, b):
    ''' Computes the translation based on the ellipsoidal earth model
    Args:
        lat_lng_alt (np.array): latitude, longitude and altitude coordinate (in degrees and meters) [n,3]
        a (float/double): Semi-major axis of the ellipsoid
        b (float/double): Semi-minor axis of the ellipsoid
    Out:
        (np.array): translation from world pose to ECEF [n,3]
    '''

    phi =  lat_lng_alt[:, 0] * np.pi/180
    gamma =  lat_lng_alt[:, 1] * np.pi/180

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
    latitude = 90 - np.arccos(translation[:,2] / np.linalg.norm(translation, axis=-1, keepdims=True)) * 180/np.pi
    longitude =  np.arctan2(translation[:,1],translation[:,0]) * 180/np.pi

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

    longitude = np.arctan2(translation[:,1], translation[:,0]) * 180/np.pi
    latitude = np.arctan(gamma) * 180/np.pi
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

    theta = (90. - lat_lng_alt[:, 0]) * np.pi/180
    phi = lat_lng_alt[:, 1] * np.pi/180

    R_enu_ecef = local_ENU_2_ECEF_orientation(theta, phi)


    if earth_model == 'WGS84':
        a = 6378137.0
        flattening = 1.0 / 298.257223563
        b = a * (1.0 - flattening)
        translation = lat_lng_alt_2_translation_ellipsoidal(lat_lng_alt, a, b)

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
    theta = (90. - lat_lng_alt[:, 0]) * np.pi/180
    phi = lat_lng_alt[:, 1] * np.pi/180

    R_ecef_enu = local_ENU_2_ECEF_orientation(theta, phi).transpose(0,2,1)

    orientation = R_ecef_enu @ rotation
    orientation_axis, orientation_angle = so3_2_axis_angle(orientation)


    return lat_lng_alt, orientation_axis, orientation_angle


def axis_angle_2_quaternion(axis, angle, degrees=True):
    ''' Converts the axis angle representation of the rotation to a unit quaternion
    Args:
        axis (np.array): the rotation axis [n,3]
        angle float/double: rotation angle either in degrees or radians [n,1]
        degrees bool: True if angle is given in degrees else False

    Out:
        (np array): rotation given in unit quaternion [n,4]
    '''
    # Treat angle (radians) below this as 0.
    cutoff_angle = 1e-9 if not degrees else 1e-9*180/np.pi
    angle[angle < cutoff_angle] = 0.0

    # Scale the axis to have the norm representing the angle
    axis_angle = (angle/np.linalg.norm(axis, axis=-1, keepdims=True)) * axis

    return R.from_rotvec(axis_angle, degrees=degrees).as_quat()

def extract_se3_transformations(translation, rot_axis, rot_angle, degrees=True):
    ''' Converts the axis/angle rotation and translation to a se3 transformation matrix
    Args:
        translation (np.array): translation vectors (x,y,z) [n,3]
        axis (np.array): the rotation axes [n,3]
        angle float/double: rotation angles either in degrees or radians [n,1]
        degrees bool: True if angle is given in degrees else False

    Out:
        (np array): transformations in a se3 matrix representation [n,4,4]
    '''

    n = translation.shape[0]

    # Initialize the matrices
    trans = np.tile(np.eye(4).reshape(1,4,4),[n,1,1])

    # Fill in with the so3 rotations an translation vectors
    trans[:,:3,:3] = axis_angle_2_so3(rot_axis, rot_angle, degrees)
    trans[:,:3,3] = translation

    return trans


def get_sensor_to_sensor_flu(sensor):
    """Compute a rotation transformation matrix that rotates sensor to Front-Left-Up format.

    Args:
        sensor (str): sensor name.

    Returns:
        np.ndarray: the resulting rotation matrix.
    """
    if "cam" in sensor:
        rot = [
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    else:
        rot = np.eye(4, dtype=np.float32)

    return np.asarray(rot, dtype=np.float32)


def transform_from_eulers(rpy_deg, translation):
    """Create a 4x4 rigid transformation matrix given euler angles and translation.

    Args:
        rpy_deg (Sequence[float]): Euler angles as roll, pitch, yaw in degrees.
        translation (Sequence[float]): x, y, z translation.

    Returns:
        np.ndarray: the constructed transformation matrix.
    """
    transform = np.eye(4)
    transform[:3, :3] = R.from_euler(
        seq="xyz", angles=rpy_deg, degrees=True
    ).as_matrix()
    transform[:3, 3] = translation

    return transform.astype(np.float32)


def parse_rig_sensors_from_dict(rig):
    """Parses the provided rig dictionary into a dictionary indexed by sensor name.

    Args:
        rig (Dict): Complete rig file as a dictionary.

    Returns:
        (Dict): Dictionary of sensor rigs indexed by sensor name.
    """
    # Parse the properties from the rig file
    sensors = rig["rig"]["sensors"]

    sensors_dict = {sensor["name"]: sensor for sensor in sensors}
    return sensors_dict


def parse_rig_sensors_from_file(rig_fp):
    """Parses the provided rig file into a dictionary indexed by sensor name.

    Args:
        rig_fp (str): Filepath to rig file.

    Returns:
        (Dict): Dictionary of sensor rigs indexed by sensor name.
    """
    # Read the json file
    with open(rig_fp, "r") as fp:
        rig = json.load(fp)

    return parse_rig_sensors_from_dict(rig)


# def sensor_to_rig(sensor, ignore_correction_T):
#     """Parses the provided rig-style transform dictionary into sensor to rig matrices.

#     Args:
#         sensor (Dict): the dictionary of the transformation values read from
#             the rig file.
#         ignore_correction_T (bool): if `True`, the correction translation values in the rig will
#             be ignored and set to zero.

#     Returns:
#         sensor_to_rig (np.ndarray): the 4x4 sensor to rig transformation matrix with the correction
#             factors applied.
#         nominal_sensor_to_rig (np.ndarray) the nominal 4x4 sensor to rig transformation matrix
#             without the correction factors.
#     """
#     sensor_name = sensor["name"]
#     sensor_to_FLU = get_sensor_to_sensor_flu(sensor_name)

#     nominal_FLU_to_rig = transform_from_eulers(
#         sensor["nominalSensor2Rig_FLU"]["roll-pitch-yaw"],
#         sensor["nominalSensor2Rig_FLU"]["t"],
#     )

#     # Use or ignore the correction for translation
#     if "correction_rig_T" in sensor.keys() and not ignore_correction_T:
#         correction_T = sensor["correction_rig_T"]
#     else:
#         correction_T = np.zeros(3, dtype=np.float32)

#     if "correction_sensor_R_FLU" in sensor.keys():
#         correction_R_FLU = sensor["correction_sensor_R_FLU"]["roll-pitch-yaw"]
#     else:
#         correction_R_FLU = [0.0, 0.0, 0.0]


#     # Apply the correction 

#     sensor_to_rig = nominal_FLU_to_rig @ correction_transform @ sensor_to_FLU
#     nominal_sensor_to_rig = nominal_FLU_to_rig @ sensor_to_FLU

#     return sensor_to_rig, nominal_sensor_to_rig


    @staticmethod
    def rotation_from_eulers(rpy_deg: typing.Sequence[float]):
        """Create a 3x3 rigid transformation matrix given euler angles.

        Args:
            rpy_deg (typing.Sequence[float]): Euler angles as roll, pitch, yaw in degrees.

        Returns:
            np.array: the constructed transformation matrix.
        """
        transform = R.from_euler(
            seq="xyz", angles=rpy_deg, degrees=True
        ).as_matrix()
        return transform.astype(np.float32)


def rotation_from_eulers(rpy_deg):
    """Create a 3x3 rigid transformation matrix given euler angles.

    Args:
        rpy_deg (typing.Sequence[float]): Euler angles as roll, pitch, yaw in degrees.

    Returns:
        np.array: the constructed transformation matrix.
    """
    transform = R.from_euler(
        seq="xyz", angles=rpy_deg, degrees=True
    ).as_matrix()
    return transform.astype(np.float32)

def sensor_to_rig(sensor):

    sensor_name = sensor["name"]
    sensor_to_FLU = get_sensor_to_sensor_flu(sensor_name)


    nominal_T = sensor["nominalSensor2Rig_FLU"]["t"]
    nominal_R = sensor["nominalSensor2Rig_FLU"]["roll-pitch-yaw"]

    correction_T = np.zeros(3, dtype=np.float32)
    correction_R = np.zeros(3, dtype=np.float32)

    if ("correction_rig_T" in sensor.keys()):
        correction_T = sensor["correction_rig_T"]

    if ("correction_sensor_R_FLU" in sensor.keys()):
        assert "roll-pitch-yaw" in sensor["correction_sensor_R_FLU"].keys(), str(sensor["correction_sensor_R_FLU"])
        correction_R = sensor["correction_sensor_R_FLU"]["roll-pitch-yaw"]

    nominal_R = rotation_from_eulers(nominal_R)
    correction_R = rotation_from_eulers(correction_R)

    R = nominal_R @ correction_R
    T =  np.array(nominal_T) + np.array(correction_T)

    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = T

    sensor_to_rig = transform @ sensor_to_FLU

    return sensor_to_rig


def camera_intrinsic_parameters(sensor):
    """Parses the provided rig-style transform dictionary into camera intrinsic parameters.

    Args:
        sensor (Dict): the dictionary of the transformation values read from
            the rig file.
    Returns:
        intrinsic (np.ndarray): the 3x4 camera intrinsic matrix (last column contains width, height)
    """
    sensor_name = sensor['name']
    cx = float(sensor['properties']['cx'])
    cy = float(sensor['properties']['cy'])
    width = float(sensor['properties']['width'])
    height = float(sensor['properties']['height'])

    poly = 'polynomial' if 'polynomial' in sensor['properties'] else 'bw-poly'
    bwpoly = [np.float32(val) for val in sensor['properties'][poly].split()]
    
    intrinsic = [cx, cy, width, height] + bwpoly

    return np.array(intrinsic, dtype=float)


def get_rig_info(json_path, camera_folder_list = None):
    """
    Get data information from rig json file

    Args:
        json_path (string): json file path
        camera_folder_list (list): list of camera folders
    Return:
        rig_info (struct): data information from rig json
    """
    if not os.path.isfile(json_path):
        print('File path {} does not exist'.format(json_path))
    
    print("Read json file {}". format(json_path))
    with open(json_path) as json_file:
        json_load = json.load(json_file)

    rig_info = {}
    sensor_data = json_load['rig']['sensors']
    for data in sensor_data:
        sensor_name = data['name']
        cx = float(data['properties']['cx'])
        cy = float(data['properties']['cy'])
        bwpoly = data['properties']['bw-poly']
        bwpoly_data = bwpoly.split('\"')[0].split(' ')
        bwpoly_vec = np.array([float(bwpoly_data[0]), float(bwpoly_data[1]), float(bwpoly_data[2]),
                               float(bwpoly_data[3]), float(bwpoly_data[4])], dtype=float)

        rpy = data['nominalSensor2Rig_FLU']['roll-pitch-yaw']
        rpy_vec = np.array([float(rpy[0]), float(rpy[1]), float(rpy[2])], dtype=float)
        t = data['nominalSensor2Rig_FLU']['t']
        t_vec = np.array([float(t[0]), float(t[1]), float(t[2])], dtype=float)

        sensor = Sensor_rig(name=sensor_name, rpy=rpy_vec, t=t_vec, cx=cx, cy=cy, bwpoly=bwpoly_vec)
        rig_info[sensor_name] = sensor

    return rig_info


# Functions realted to the F-THeta camera model
def numericallyStable2Norm2D(x, y):
    absX = abs(x)
    absY = abs(y)
    minimum = min(absX, absY)
    maximum = max(absX, absY)

    if maximum <= np.float32(0.0):
        return np.float32(0.0)

    oneOverMaximum = np.float32(1.0) / maximum
    minMaxRatio = np.float32(minimum) * oneOverMaximum
    return maximum * np.sqrt(np.float32(1.0) + minMaxRatio * minMaxRatio)


def backwards_polynomial(pixel_norms, intrinsic):
    ret = 0
    for k, coeff in enumerate(intrinsic):

        ret += coeff * pixel_norms**k

    return ret

def pixel_2_camera_ray(pixel_coords, intrinsic, camera_model):
    ''' Convert the pixel coordinates to a 3D ray in the camera coordinate system.

    Args:
        pixel_coords (np.array): pixel coordinates of the selected points [n,2]
        intrinsic (np.array): camera intrinsic parameters (size depends on the camera model)
        camera_model (string): camera model used for projection. Must be one of ['pinhole', 'f_theta']

    Out:
        camera_rays (np.array): rays in the camera coordinate system [n,3]
    ''' 

    camera_rays = np.ones((pixel_coords.shape[0],3))

    if camera_model == 'pinhole':
        camera_rays[:,0] = (pixel_coords[:,0] + 0.5 - intrinsic[2]) / intrinsic[0]
        camera_rays[:,1] = (pixel_coords[:,1] + 0.5 - intrinsic[5]) / intrinsic[4]

    elif camera_model == "f_theta":
        centered_pix_coords = np.ones((pixel_coords.shape[0],2))
        centered_pix_coords[:,0] = pixel_coords[:,0] + 0.5 - intrinsic[0]
        centered_pix_coords[:,1] = pixel_coords[:,1] + 0.5 - intrinsic[1]         

        pixel_norms = np.linalg.norm(centered_pix_coords, axis=1, keepdims=True)

        alphas = backwards_polynomial(pixel_norms, intrinsic[4:9])
        camera_rays[:,0:1] = (np.sin(alphas) * centered_pix_coords[:,0:1]) / pixel_norms 
        camera_rays[:,1:2] = (np.sin(alphas) * centered_pix_coords[:,1:2]) / pixel_norms 
        camera_rays[:,2:3] = np.cos(alphas)

        # special case: ray is perpendicular to image plane normal
        valid = (pixel_norms > np.finfo(np.float32).eps).squeeze()
        camera_rays[~valid, :] = (0, 0, 1)  # This is what DW sets these rays to

    return camera_rays

def compute_fw_polynomial(intrinsic):

    img_width = intrinsic[2]
    img_height = intrinsic[3]
    cxcy = np.array(intrinsic[0:2])

    max_value = 0.0
    value =  np.linalg.norm(np.asarray([0.0, 0.0], dtype=cxcy.dtype) - cxcy)
    max_value = max(max_value, value)
    value = np.linalg.norm(np.asarray([0.0, img_height], dtype=cxcy.dtype) - cxcy)
    max_value = max(max_value, value)
    value = np.linalg.norm(np.asarray([img_width, 0.0], dtype=cxcy.dtype) - cxcy)
    max_value = max(max_value, value)
    value = np.linalg.norm(np.asarray([img_width, img_height], dtype=cxcy.dtype) - cxcy)
    max_value = max(max_value, value)

    SAMPLE_COUNT = 500
    samples_x = []
    samples_b = []
    step = max_value / SAMPLE_COUNT
    x = step

    for _ in range(0, SAMPLE_COUNT):
        p = np.asarray([cxcy[0] + x - 0.5, cxcy[1] - 0.5], dtype=np.float64).reshape(-1,2)
        ray = pixel_2_camera_ray(p, intrinsic, 'f_theta')
        xy_norm = np.linalg.norm(ray[0, :2])
        theta = np.arctan2(float(xy_norm), float(ray[0, 2]))
        samples_x.append(theta)
        samples_b.append(float(x))
        x += step

    x = np.asarray(samples_x, dtype=np.float64)
    y = np.asarray(samples_b, dtype=np.float64)
    # Fit a 4th degree polynomial. The polynomial function is as follows:

    def f(x, b, x1, x2, x3, x4):
        """4th degree polynomial."""
        return b + x1 * x + x2 * (x ** 2) + x3 * (x ** 3) + x4 * (x ** 4)

    # The constant in the polynomial should be zero, so add the `bounds` condition.
    coeffs, _ = curve_fit(
        f,
        x,
        y,
        bounds=(
            [0, -np.inf, -np.inf, -np.inf, -np.inf],
            [np.finfo(np.float64).eps, np.inf, np.inf, np.inf, np.inf],
        ),
    )

    return np.array([np.float32(val) if i > 0 else 0 for i, val in enumerate(coeffs)])


def compute_ftheta_fov(intrinsic):
    """Computes the FOV of this camera model."""
    max_x = intrinsic[2] - 1
    max_y = intrinsic[3] - 1

    point_left = np.asarray([0.0, intrinsic[1]]).reshape(-1,2) - 0.5
    point_right = np.asarray([max_x, intrinsic[1]]).reshape(-1,2) - 0.5
    point_top = np.asarray([intrinsic[0], 0.0]).reshape(-1,2) - 0.5
    point_bottom = np.asarray([intrinsic[0], max_y]).reshape(-1,2) - 0.5

    fov_left = _get_pixel_fov(point_left, intrinsic)
    fov_right = _get_pixel_fov(point_right, intrinsic)
    fov_top = _get_pixel_fov(point_top, intrinsic)
    fov_bottom = _get_pixel_fov(point_bottom, intrinsic)

    v_fov = fov_top + fov_bottom
    hz_fov = fov_left + fov_right
    max_angle = _compute_max_angle(intrinsic)

    return v_fov, hz_fov, max_angle


def _get_pixel_fov(pt, intrinsic):
    """Gets the FOV for a given point. Used internally for FOV computation of the F-theta camera.

    Args:
        pt (np.ndarray): 2D pixel.

    Returns:
        fov (float): the FOV of the pixel.
    """
    ray = pixel_2_camera_ray(pt, intrinsic, 'f_theta')
    fov = np.arctan2(np.linalg.norm(ray[:, :2], axis=1), ray[:, 2])
    return fov


def _compute_max_angle(intrinsic):

        p = np.asarray(
            [[0, 0], [intrinsic[2] - 1, 0], [0, intrinsic[3] - 1], [intrinsic[2] - 1, intrinsic[3] - 1]], dtype=np.float32
        ) - 0.5

        return max(
            max(_get_pixel_fov(p[0:1, ...], intrinsic), _get_pixel_fov(p[1:2, ...], intrinsic)),
            max(_get_pixel_fov(p[2:3, ...], intrinsic), _get_pixel_fov(p[3:4, ...], intrinsic)),
        )


def compute_ftheta_parameters(intrinsic):

    # Initialize the forward polynomial
    fw_poly = Polynomial(intrinsic[9:14])

    v_fov, hz_fov, max_angle = compute_ftheta_fov(intrinsic)

    is_fw_poly_slope_negative_in_domain = False
    ray_angle = max_angle.copy()
    deg2rad = np.pi / 180.0
    while ray_angle >= np.float32(0.0):
        temp_dval = fw_poly.deriv()(max_angle).item()
        if temp_dval < 0:
            is_fw_poly_slope_negative_in_domain = True
        ray_angle -= deg2rad

    if is_fw_poly_slope_negative_in_domain:
        ray_angle = max_angle.copy()
        while ray_angle >= 0.0:
            ray_angle -= deg2rad
        raise ArithmeticError(
            "FThetaCamera: derivative of distortion within image interior is negative"
        )

    # Evaluate the forward polynomial at point (self._max_ray_angle, 0)
    # Also evaluate its derivative at the same point
    val = fw_poly(max_angle).item()
    dval = fw_poly.deriv()(max_angle).item()

    if dval < 0:
        raise ArithmeticError(
            "FThetaCamera: derivative of distortion at edge of image is negative"
        )

    max_ray_distortion = np.asarray([val, dval], dtype=np.float32)

    return max_ray_distortion, max_angle


##### FUNCTIONS FOR OBJECT LABEL PARSING ##########

def dict2numpy_3d(dict_):
    """Converts a dictionary with x,y,z values to a numpy array

    Args:
        dict_ (Dict): the dictionary containing the x, y, z values (typically originating from the auto/manual labels)
    Returns:
        (np.ndarray): values represented as a numpy array [3,]
    """
    return np.array([dict_['x'], dict_['y'], dict_['z']])

def extract_cuboid(cuboid, enlarge_dim=0):
    """ Extracts the center, dimensions and orientation of the 3D cuboid

    Args:
        cuboid (Dict): the dictionary containing the x, y, z values (typically originating from the auto/manual labels)
        enlarge_dim (float): percentage for which the dimensions of the bounding box will be enlarged
    Returns:
        (np.ndarray): values represented as a numpy array [3,]
    """
    center = dict2numpy_3d(cuboid['center'])
    orientation = dict2numpy_3d(cuboid['orientation'])
    dimensions = dict2numpy_3d(cuboid['dimensions']) * (1 + enlarge_dim)
    
    return np.concatenate((center, dimensions, orientation))


def cuboid_2_verts(cuboid):
    """
    Converts an ILF cuboid3d object into x,y,z vertex coordiantes.
    """
    rotation = R.from_euler('xyz', cuboid[-3:], degrees=False)

    dimensions = cuboid[3:6]
    bv = np.array([[-dimensions[0] / 2, -dimensions[1] / 2, -dimensions[2] / 2],
                   [-dimensions[0] / 2, -dimensions[1] / 2, dimensions[2] / 2],
                   [-dimensions[0] / 2, dimensions[1] / 2, dimensions[2] / 2],
                   [-dimensions[0] / 2, dimensions[1] / 2, -dimensions[2] / 2],
                   [dimensions[0] / 2, -dimensions[1] / 2, -dimensions[2] / 2],
                   [dimensions[0] / 2, -dimensions[1] / 2, dimensions[2] / 2],
                   [dimensions[0] / 2, dimensions[1] / 2, dimensions[2] / 2],
                   [dimensions[0] / 2, dimensions[1] / 2, -dimensions[2] / 2]])

    cuboid = rotation.apply(bv) + cuboid[:3]

    return cuboid




def pixel_2_camera_ray(pixel_coords, intrinsic, camera_model):
    ''' Convert the pixel coordinates to a 3D ray in the camera coordinate system.

    Args:
        pixel_coords (np.array): pixel coordinates of the selected points [n,2]
        intrinsic (np.array): camera intrinsic parameters (size depends on the camera model)
        camera_model (string): camera model used for projection. Must be one of ['pinhole', 'f_theta']

    Out:
        camera_rays (np.array): rays in the camera coordinate system [n,3]
    ''' 

    camera_rays = np.ones((pixel_coords.shape[0],3))

    if camera_model == 'pinhole':
        camera_rays[:,0] = (pixel_coords[:,0] + 0.5 - intrinsic[2]) / intrinsic[0]
        camera_rays[:,1] = (pixel_coords[:,1] + 0.5 - intrinsic[5]) / intrinsic[4]

    elif camera_model == "f_theta":
        centered_pix_coords = np.ones((pixel_coords.shape[0],2))
        centered_pix_coords[:,0] = pixel_coords[:,0] + 0.5 - intrinsic[0]
        centered_pix_coords[:,1] = pixel_coords[:,1] + 0.5 - intrinsic[1]         

        pixel_norms = np.linalg.norm(centered_pix_coords, axis=1, keepdims=True)

        alphas = backwards_polynomial(pixel_norms, intrinsic[4:9])
        camera_rays[:,0:1] = (np.sin(alphas) * centered_pix_coords[:,0:1]) / pixel_norms 
        camera_rays[:,1:2] = (np.sin(alphas) * centered_pix_coords[:,1:2]) / pixel_norms 
        camera_rays[:,2:3] = np.cos(alphas)

        # special case: ray is perpendicular to image plane normal
        valid = (pixel_norms > np.finfo(np.float32).eps).squeeze()
        camera_rays[~valid, :] = (0, 0, 1)  # This is what DW sets these rays to

    return camera_rays


def pixel_2_world_ray_py(pixel_coords, intrinsic, camera_model, img_height, 
                         roll_shutter_delay, pix_exp_t, eof_t, poses, pose_t):

    ''' Convert the pixel coordinates to 3D rays in the global coordinate system by compensating for the
    the rolling shutter.

    Args:
        pixel_coords (np.array): pixel coordinates of the selected points [n,2]
        intrinsic (np.array): camera intrinsic parameters (size depends on the camera model)
        camera_model (string): camera model used for projection. Must be one of ['pinhole', 'f_theta']
        img_height (float): image hight in pixels
        roll_shutter_delay (float): time offset between the first and the last row, due to the rolling shutter
        pix_exp_t (float): exposure time of each row
        eof_t (float): end of frame timestamp
        poses (np.array): global to camera transformation matrices used for interpolation (one before and one after) [2,4,4]
        pose_t (np.array): timestamp of the transformation matrixes [2,4,4]
    
    Out:
        world_rays (np.array): 3d rays in the global coordinate system [n,3]
    ''' 

    camera_rays = pixel_2_camera_ray(pixel_coords, intrinsic, camera_model)

    pose_interpolator = PoseInterpolator(poses, pose_t.flatten())

    sof_t = eof_t - roll_shutter_delay
    first_row_t = sof_t - pix_exp_t
    last_row_t = eof_t - pix_exp_t
    d_first_last_row  = last_row_t - first_row_t


    world_rays = np.zeros((pixel_coords.shape[0], 6))
    for i in range(camera_rays.shape[0]):

        pix_t = first_row_t + pixel_coords[i,1] * d_first_last_row / (img_height - 1)

        pix_pose = pose_interpolator.interpolate_to_timestamps(pix_t)

        world_rays[i,:3] = pix_pose[0, :3,3, None].transpose()
        world_rays[i,3:] = (pix_pose[0,:3,:3] @ camera_rays[i:i+1,:].transpose()).transpose()

    world_rays[:,3:] /=  np.linalg.norm(world_rays[:,3:],axis=1, keepdims=True)
    return world_rays


def transform_point_cloud(pc, T):
    ''' Transform the point cloud with the provided transformation matrix
    Args:
        pc (np.array): point cloud coordinates (x,y,z) [n,3]
        T (np.array): se3 transformation matrix [4,4]

    Out:
        (np array): transformed point cloud coordinated [n,3]
    '''
    return (T[:3,:3] @ pc[:,:3].transpose() + T[:3,3:4]).transpose()


def world_points_2_pixel(points, cam_metadata, iterate=False):

    ''' Projects the points in the global coordinate system to the image plane by compensating for the rollign shutter effect 
        See the https://docs.google.com/document/u/2/d/1hdzTpDlONoltAtvcUh7HFH7qWeyWaRut62H8-jdQ6o0/edit for more information
        on the rolling shutter times, and its effect on point projection.

    Args:
        points (np.array): point coordinates in the global coordinate system [n,3]
        camera_metadata (dict): camera metadata
    
    Out:
        points_img (np.array): pixel coordinates of the image projections [m,2]
        valid (np.array): array of boolean flags. True for point that project to the image plane
    '''  


    intrinsic = cam_metadata['intrinsic']
    img_width = cam_metadata['img_width']
    img_height = cam_metadata['img_height']
    camera_model = cam_metadata['camera_model']
    exposure_time = cam_metadata['exposure_time']

    t_sof, t_eof = cam_metadata['ego_pose_timestamps'] 
    T_global_cam_sof = np.linalg.inv(cam_metadata['T_cam_rig']) @ np.linalg.inv(cam_metadata['ego_pose_s'])
    T_global_cam_eof = np.linalg.inv(cam_metadata['T_cam_rig']) @ np.linalg.inv(cam_metadata['ego_pose_e'])    
    pose_interpolator = PoseInterpolator(np.stack([T_global_cam_sof,T_global_cam_eof]), np.array([t_sof, t_eof]))

    # Transform the point cloud to the cam coordinate system based on the last pose
    points_cam = transform_point_cloud(points, T_global_cam_eof)

    # Preform an initial projection
    initial_proj, valid_flag = project_camera_rays_2_img(points_cam, cam_metadata)
    
    initial_proj = initial_proj[valid_flag,:]
    valid_pts = points[valid_flag,:]
    initial_valid_idx = np.where(valid_flag == True)[0]

    # Get the time of the acquisition of the first and last row
    first_row_t = t_sof + exposure_time/2
    last_row_t = t_eof - exposure_time/2
    d_first_last_row  = last_row_t - first_row_t

    optimized_proj = []
    valid_idx = []
    trans_matrices = []
    
    for pt_idx, point in enumerate(initial_proj):
        # TODO: ADAPT THIS FOR ALL ROLLING SHUTTER DIRECTIONS
        t_h = first_row_t + point[1] * d_first_last_row / (img_height - 1)

        k_maxIterNum = 10 # Max 10 iterations
        k_threshold = 300 # Threshold of 300 ms

        iter_num = 0
        residual = 5*k_threshold
        valid = True

        if iterate:
            while abs(residual) > k_threshold and iter_num < k_maxIterNum and valid:
                pix_pose = pose_interpolator.interpolate_to_timestamps(t_h)[0]

                tmp_point = transform_point_cloud(valid_pts[pt_idx].reshape(1,-1), pix_pose)

                new_proj, valid_pts = project_camera_rays_2_img(tmp_point, intrinsic, img_width, img_height, camera_model)
                
                if new_proj.shape[0] > 0:
                    residual = t_h - (first_row_t + new_proj[0,1] * d_first_last_row / (img_height - 1))
                    t_h -= residual/2
                else:
                    valid = False

        else:
            pix_pose = pose_interpolator.interpolate_to_timestamps(t_h)[0]
            trans_matrices.append(pix_pose)
            tmp_point = transform_point_cloud(valid_pts[pt_idx].reshape(1,-1), pix_pose)

            # new_proj,_ = project_points_2_img(valid_pts[pt_idx].reshape(1,-1), pix_pose, intrinsic, img_width, img_height)
            new_proj, _ = project_camera_rays_2_img(tmp_point, cam_metadata)

        if new_proj.shape[0] > 0:
            optimized_proj.append(new_proj[0])
            valid_idx.append(initial_valid_idx[pt_idx])

    return np.stack(optimized_proj), np.stack(trans_matrices), np.stack(valid_idx)



def project_camera_rays_2_img(points, cam_metadata):
    ''' Projects the points in the camera coordinate system to the image plane

    Args:
        points (np.array): point coordinates in the camera coordinate system [n,3]
        intrinsic (np.array): camera intrinsic parameters (size depends on the camera model)
        img_width (float): image width in pixels
        img_height (float): image hight in pixels
        camera_model (string): camera model used for projection. Must be one of ['pinhole', 'f_theta']
    Out:
        points_img (np.array): pixel coordinates of the image projections [m,2]
        valid (np.array): array of boolean flags. True for point that project to the image plane
    '''  

    intrinsic = cam_metadata['intrinsic']
    camera_model = cam_metadata['camera_model']
    img_width = cam_metadata['img_width']
    img_height = cam_metadata['img_height']

    if camera_model == "pinhole":

        intrinsic = intrinsic.reshape(3,3)
        points_img = np.matmul(intrinsic, points.transpose()).transpose()

        points_img /= points_img[:,2:]

        x_ok = np.logical_and(0 <= points_img[:, 0], points_img[:, 0] < img_width)
        y_ok = np.logical_and(0 <= points_img[:, 1], points_img[:, 1] < img_height)
        valid = np.logical_and(x_ok, y_ok)

        return points_img[valid,:2], valid

    elif camera_model == "f_theta":

        # Initialize the forward polynomial
        fw_poly = Polynomial(intrinsic[9:14])

        xy_norm = np.zeros((points.shape[0], 1))

        for i, point in enumerate(points):
            xy_norm[i] = numericallyStable2Norm2D(point[0], point[1])

        cos_alpha = points[:, 2:] / np.linalg.norm(points, axis=1, keepdims=True)
        alpha = np.arccos(np.clip(cos_alpha, -1 + 1e-7, 1 - 1e-7))
        delta = np.zeros_like(cos_alpha)
        valid = alpha <= intrinsic[16]


        delta[valid] = fw_poly(alpha[valid])

        # For outside the model (which need to do linear extrapolation)
        delta[~valid] = (intrinsic[14] + (alpha[~valid] - intrinsic[16]) * intrinsic[15])

        # Determine the bad points with a norm of zero, and avoid division by zero
        bad_norm = xy_norm <= 0
        xy_norm[bad_norm] = 1
        delta[bad_norm] = 0

        # compute pixel relative to center
        scale = delta / xy_norm
        pixel = scale * points

        # Handle the edge cases (ray along image plane normal)
        edge_case_cond = (xy_norm <= 0.0).squeeze()
        pixel[edge_case_cond, :] = points[edge_case_cond, :]
        points_img = pixel
        points_img[:, :2] += intrinsic[0:2]

        x_ok = np.logical_and(0 <= points_img[:, 0], points_img[:, 0] < img_width)
        y_ok = np.logical_and(0 <= points_img[:, 1], points_img[:, 1] < img_height)
        valid = np.logical_and(x_ok, y_ok)


        return points_img, valid