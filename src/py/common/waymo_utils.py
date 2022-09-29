# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

import numpy as np
import tensorflow as tf

from waymo_open_dataset import dataset_pb2
from collections import defaultdict

## This code is adapted from the official waymo open github page
def global_vel_to_ref(vel, global_from_ref_rotation):
  # inverse means ref_from_global, rotation_matrix for normalization
  vel = [vel[0], vel[1], 0]
  ref = np.dot(global_from_ref_rotation.transpose(), vel) 
  ref = [ref[0], ref[1], 0.0]

  return ref

def extract_lidar_labels(frame):
  pose = np.reshape(np.array(frame.pose.transform), [4, 4])


  objects = []
  for object_id, label in enumerate(frame.laser_labels):
    category_label = label.type
    box = label.box

    # Speed and acceleration are given in the global coordinate frame
    speed = [label.metadata.speed_x, label.metadata.speed_y]
    accel = [label.metadata.accel_x, label.metadata.accel_y]
    num_lidar_points_in_box = label.num_lidar_points_in_box

    # Difficulty level is 0 if labeler did not say this was LEVEL_2.
    # Set difficulty level of "999" for boxes with no points in box.
    if num_lidar_points_in_box <= 0:
      combined_difficulty_level = 999
    if label.detection_difficulty_level == 0:
      # Use points in box to compute difficulty level.
      if num_lidar_points_in_box >= 5:
        combined_difficulty_level = 1
      else:
        combined_difficulty_level = 2
    else:
      combined_difficulty_level = label.detection_difficulty_level

    objects.append({
        'id': object_id,
        'track_id': label.id,
        'label': category_label,
        '3D_bbox': np.array([box.center_x, box.center_y, box.center_z,
                         box.length, box.width, box.height, 0, 0, box.heading], dtype=np.float32),
        'num_points':
            num_lidar_points_in_box,
        'detection_difficulty_level':
            label.detection_difficulty_level,
        'combined_difficulty_level':
            combined_difficulty_level,
        'global_speed':
            np.array(speed, dtype=np.float32),
        'global_accel':
            np.array(accel, dtype=np.float32),
    })

  return objects


def extract_camera_labels(frame):

    cam_labels = defaultdict(list)

    for camera in sorted(frame.camera_labels, key=lambda i:i.name):

        for label in camera.labels:
            cam_labels['cam_{}'.format(camera.name)].append({
                'name': label.id,
                'label': label.type,
                '2D_bbox': np.array([label.box.center_x, label.box.center_y, 
                                 label.box.length, label.box.width], dtype=np.float32)
                                  })

    return cam_labels

def extract_projected_labels(frame):
    name_map = {1: 'FRONT', 2:'FRONT_LEFT', 3:'FRONT_RIGHT', 4:'SIDE_LEFT', 5:'SIDE_RIGHT'}

    proj_camera_labels = defaultdict(list)

    for camera in sorted(frame.projected_lidar_labels, key=lambda i:i.name):
        for label in camera.labels:
            proj_camera_labels['cam_{}'.format(camera.name)].append({
                'name': label.id[0:label.id.find(name_map[camera.name])-1],
                'label': label.type,
                '3D_bbox_proj': np.array([label.box.center_x, label.box.center_y, 
                                 label.box.length, label.box.width], dtype=np.float32)
                                  })

    return proj_camera_labels


def extrapolate_pose_based_on_velocity(T_SDC_global,v_global, w_global, dt):
    T_extrapolated = np.eye(4)
    T_extrapolated[:3,:3] = (np.eye(3) + get_skew_symmetric(w_global)*dt) @ T_SDC_global[:3,:3]
    T_extrapolated[:3,3:4] = T_SDC_global[:3,3:4] + v_global * dt

    return T_extrapolated


def get_skew_symmetric(vec):

    skew_sym = np.zeros((3,3))

    skew_sym[0,1] = -vec[2]
    skew_sym[0,2] = vec[1]
    skew_sym[1,0] = vec[2]
    skew_sym[1,2] = -vec[0]
    skew_sym[2,0] = -vec[1]
    skew_sym[2,1] = vec[0]

    return skew_sym



def direction_to_image(normalized_coordinates, camera_metadata):
    f_u = camera_metadata['intrinsic'][0]
    f_v = camera_metadata['intrinsic'][1]
    c_u = camera_metadata['intrinsic'][2]
    c_v = camera_metadata['intrinsic'][3]
    k1 = camera_metadata['intrinsic'][4]
    k2 = camera_metadata['intrinsic'][5]
    k3 = camera_metadata['intrinsic'][6]  # same as p1 in OpenCV.
    k4 = camera_metadata['intrinsic'][7]  # same as p2 in OpenCV
    k5 = camera_metadata['intrinsic'][8]  # same as k3 in OpenCV.

    u_n = normalized_coordinates[0]
    v_n = normalized_coordinates[1]

    r2 = u_n**2 + v_n**2
    r4 = r2 * r2
    r6 = r4 * r2

    r_d = 1.0 + k1 * r2 + k2 * r4 + k5 * r6

    kMinRadialDistortion = 0.8
    kMaxRadialDistortion = 1.2

    if (r_d < kMinRadialDistortion or r_d > kMaxRadialDistortion):
        roi_clipping_radius = np.sqrt(camera_metadata['image_width']**2 +  camera_metadata['image_height']**2)
        r2_sqrt_rcp = 1.0 / np.sqrt(r2)
        u_d = u_n * r2_sqrt_rcp * roi_clipping_radius + c_u
        v_d = v_n * r2_sqrt_rcp * roi_clipping_radius + c_v

        return False, u_d, v_d

    u_nd = u_n * r_d + 2.0 * k3 * u_n * v_n + k4 * (r2 + 2.0 * u_n * u_n)
    v_nd = v_n * r_d + k3 * (r2 + 2.0 * v_n * v_n) + 2.0 * k4 * u_n * v_n

    u_d = u_nd * f_u + c_u
    v_d = v_nd * f_v + c_v

    return True, u_d, v_d


def get_pixel_timestamp(rs_redout_direction, shutter, cam_trigger_time, cam_readout_time, img_w, img_h, x, y):
    """
    Cameras have a rolling shutter, so each *sensor* row is exposed at a
    slightly different time, starting with the top row and ending with the
    bottom row. Because the sensor itself may be rotated, this means that the
    *image* is captured row-by-row or column-by-column, depending on
    `readout_direction`.
    Final time for this pixel is the initial trigger time + the column and row
    offset (exactly one of these will be non-zero) + half the shutter time to
    get the middle of the exposure.
    """
    readout_duration = cam_readout_time - cam_trigger_time - shutter

    base_ts = cam_trigger_time + 0.5 * shutter

    if rs_redout_direction == 1: # TOP_TO_BOTTOM
        return base_ts + readout_duration / img_h * y
    elif rs_redout_direction == 3: # BOTTOM_TO_TOP
        return base_ts + readout_duration / img_h * (img_h - y)
    elif rs_redout_direction == 2: # LEFT_TO_RIGHT
        return base_ts + readout_duration / img_w * x
    elif rs_redout_direction == 4: # RIGHT_TO_LEFT
        return base_ts + readout_duration / img_w * (img_w - x)
    else:
        raise ValueError('Wrong rollign shutter direction.')


def image_to_direction(u_d, v_d, intrinsic):

    f_u = intrinsic[0]
    f_v = intrinsic[1]
    c_u = intrinsic[2]
    c_v = intrinsic[3]

    u_nd = (u_d - c_u) / f_u
    v_nd = (v_d - c_v) / f_v

    return iterative_undistortion(u_nd, v_nd, intrinsic)

# In normalized camera, undistorts point coordinates via iteration.
def iterative_undistortion(u_nd, v_nd, intrinsic):

    f_u = intrinsic[0]
    f_v = intrinsic[1]
    k1 =  intrinsic[4]
    k2 =  intrinsic[5]
    k3 =  intrinsic[6] # same as p1 in OpenCV.
    k4 =  intrinsic[7] # same as p2 in OpenCV.
    k5 =  intrinsic[8] # same as k3 in OpenCV.

    # Initial guess
    u = u_nd
    v = v_nd

    # Check that the focal length is a positive number
    assert f_u > 0.0, "Focal length is negative"
    assert f_v > 0.0, "Focal length is negative"


    # Minimum required squared delta before terminating. Note that it is set in
    # normalized camera coordinates at a fraction of a pixel^2. The threshold
    # should satisfy unittest accuracy threshold kEpsilon = 1e-6 even for very
    # slow convergence.
    min_delta2 = 1e-12 / (f_u * f_u + f_v * f_v)

    # Iteratively apply the distortion model to undistort the image coordinates.
    # Maximum number of iterations when estimating undistorted point.
    max_iter = 20

    for i in range(0, max_iter):
        r2 = u * u + v * v
        r4 = r2 * r2
        r6 = r4 * r2

        rd = 1.0 + r2 * k1 + r4 * k2 + r6 * k5
        u_prev = u
        v_prev = v

        u_tangential = 2.0 * k3 * u * v + k4 * (r2 + 2.0 * u * u)
        v_tangential = 2.0 * k4 * u * v + k3 * (r2 + 2.0 * v * v)
        u = (u_nd - u_tangential) / rd
        v = (v_nd - v_tangential) / rd

        du = u - u_prev
        dv = v - v_prev

        # Early exit
        if (du * du + dv * dv) < min_delta2:
            return u, v
        
    return u, v



def compute_residual_and_jacobian(world_point, t_h, camera_metadata):

    normalized_coord = np.zeros((2,1))
    R_global_cam = camera_metadata['rolling_shutter_state']['T_cam_global'][:3,:3].transpose()
    t_cam_global = camera_metadata['rolling_shutter_state']['T_cam_global'][:3,3:4]

    cam_dcm_n = R_global_cam + t_h * camera_metadata['rolling_shutter_state']['skew_omega_dcm']
    n_pos_cam = t_cam_global + t_h * camera_metadata['rolling_shutter_state']['n_vel_cam0'].reshape(-1,1)
    cam_pos_f = np.matmul(cam_dcm_n, (world_point - n_pos_cam))

    if cam_pos_f[0] <= 0:
        return False, 0,0, normalized_coord

    normalized_coord[0] = -cam_pos_f[1] / cam_pos_f[0]
    normalized_coord[1] = -cam_pos_f[2] / cam_pos_f[0]

    normalized_spacing = normalized_coord[0] if camera_metadata['rolling_shutter_state']['readout_hz_dir'] else normalized_coord[1]
    residual = t_h - normalized_spacing * camera_metadata['rolling_shutter_state']['readout_time_factor'] + \
                camera_metadata['rolling_shutter_state']['t_pose_offset']

    jacobian_landmark_to_index = -np.matmul(R_global_cam, camera_metadata['rolling_shutter_state']['n_vel_cam0']) -  \
                        np.matmul(camera_metadata['rolling_shutter_state']['skew_omega'], cam_pos_f)

    if camera_metadata['rolling_shutter_state']['readout_hz_dir']:

        jacobian_combined = camera_metadata['rolling_shutter_state']['readout_time_factor'] / cam_pos_f[0] * \
                    (normalized_coord[0] * jacobian_landmark_to_index[0] - jacobian_landmark_to_index[1])
    else:
        jacobian_combined = camera_metadata['rolling_shutter_state']['readout_time_factor'] / cam_pos_f[0] * \
            (normalized_coord[1] * jacobian_landmark_to_index[0] - jacobian_landmark_to_index[2])

    jacobian = 1. - jacobian_combined

    return True, residual, jacobian, normalized_coord

def extract_image_metadata(extrinsic, intrinsic, metadata, image):

    image_metadata = {}
    image_metadata['intrinsic'] = np.array(intrinsic)
    image_metadata['T_cam_sdc'] = np.array(extrinsic).reshape(4,4) # vehicle_tfm_cam
    image_metadata['T_sdc_global'] = np.reshape(np.array(image.pose.transform), [4, 4]) # n_tfm_vehicle0
    image_metadata['image_width'] = metadata[0]
    image_metadata['image_height'] = metadata[1]
    image_metadata['rolling_shutter_dir'] = metadata[2]
    image_metadata['rolling_shutter_state'] = extract_rolling_shutter_state(image, image_metadata)

    return image_metadata

def extract_rolling_shutter_state(img, img_metadata):

    rolling_shutter_state = {}

    readout_time = img.camera_readout_done_time - img.camera_trigger_time - img.shutter
    # Check if rolling shutter works in reverse direction
    redout_reverse_dir = True if img_metadata['rolling_shutter_dir'] in [3,4] else False
    
    # Check whether rolling shutter direction is horizontal.
    rolling_shutter_state['readout_hz_dir'] = True if img_metadata['rolling_shutter_dir'] in [2,4] else False

    # Compute the time stamp of the principal point 
    t_principal_point = get_pixel_timestamp(img_metadata['rolling_shutter_dir'], img.shutter, 
                                            img.camera_trigger_time, img.camera_readout_done_time,
                                            img_metadata['image_width'], img_metadata['image_height'], 
                                            img_metadata['intrinsic'][2], img_metadata['intrinsic'][3])
    
    # Compute readout time factor.
    if rolling_shutter_state['readout_hz_dir'] == True:
        u_n_first, _ = image_to_direction(0, 0.5 * img_metadata['image_height'], img_metadata['intrinsic'])
        u_n_end, _   = image_to_direction(img_metadata['image_width'], 0.5 * img_metadata['image_height'], img_metadata['intrinsic'])
        normalized_coord_range = u_n_end - u_n_first
        range_in_pixel_space = img_metadata['image_width']
    else:
        _, v_n_first = image_to_direction(0.5 * img_metadata['image_width'],  0, img_metadata['intrinsic'])
        _, v_n_end   = image_to_direction(0.5 * img_metadata['image_width'], img_metadata['image_height'], img_metadata['intrinsic'])
        normalized_coord_range = v_n_end - v_n_first
        range_in_pixel_space = img_metadata['image_height']

    # t_pose_offset_ = t_pose - t_principal_point. In seconds.
    rolling_shutter_state['t_pose_offset'] = img.pose_timestamp - t_principal_point

    # sign * readout time / normalized_coordinate_range.
    # The sign depends on readout direction.
    rolling_shutter_state['readout_time_factor'] = -readout_time / normalized_coord_range if redout_reverse_dir == True \
                                                    else readout_time / normalized_coord_range

    # sign * readout time / range_in_pixel_space.
    # The sign depends on readout direction.
    rolling_shutter_state['readout_time_factor_pixel'] = -readout_time / range_in_pixel_space if redout_reverse_dir == True \
                                                    else readout_time / range_in_pixel_space

    # The principal point image coordinate, in pixels
    rolling_shutter_state['principal_point'] = np.array([img_metadata['intrinsic'][2], img_metadata['intrinsic'][3]])

    # Transformation from camera to ENU at pose timestamp.
    rolling_shutter_state['T_cam_global'] = img_metadata['T_sdc_global'] @ img_metadata['T_cam_sdc'] # CHECK THIS
    
    # Velocity of camera at ENU frame at pose timestamp.
    n_vel_vehicle = np.array([img.velocity.v_x, img.velocity.v_y, img.velocity.v_z]).reshape(3,1)
    vehicle_omega_vehicle = np.array([img.velocity.w_x, img.velocity.w_y, img.velocity.w_z]).reshape(3,1)
    n_omega_vehicle = np.matmul(img_metadata['T_sdc_global'][:3,:3], vehicle_omega_vehicle)
    cam_omega_cam0 = np.matmul(img_metadata['T_cam_sdc'][:3,:3].transpose(), vehicle_omega_vehicle)
    rolling_shutter_state['skew_omega'] = get_skew_symmetric(cam_omega_cam0)

    n_pos_cam0 = np.matmul(img_metadata['T_sdc_global'][:3,:3], img_metadata['T_cam_sdc'][:3,3:4])


    # Define: skew_omega = SkewSymmetric(cam_omega_cam0).
    rolling_shutter_state['n_vel_cam0'] = n_vel_vehicle + np.matmul(get_skew_symmetric(n_omega_vehicle), n_pos_cam0)
    
    # skew_omega_dcm = skew_omega * cam0_dcm_n.
    rolling_shutter_state['skew_omega_dcm'] = -np.matmul(rolling_shutter_state['skew_omega'], rolling_shutter_state['T_cam_global'][:3,:3].transpose())

    return rolling_shutter_state


def image_to_world_ray_python(image_points, camera_metadata):

    rolling_shutter_state = camera_metadata['rolling_shutter_state']
    rays = []
    # for i in range(image_points.shape[0]):
    for i in range(1000):

        if rolling_shutter_state['readout_hz_dir']:
            pixel_spacing = image_points[i,0] - rolling_shutter_state['principal_point'][0]
        else:
            pixel_spacing = image_points[i,1] - rolling_shutter_state['principal_point'][1]


        t_h = rolling_shutter_state['readout_time_factor_pixel'] * pixel_spacing - rolling_shutter_state['t_pose_offset']

        R_global_cam = rolling_shutter_state['T_cam_global'][:3,:3].transpose() + t_h * rolling_shutter_state['skew_omega_dcm']
        n_pos_cam = rolling_shutter_state['T_cam_global'][:3,3:4] + t_h * rolling_shutter_state['n_vel_cam0']

        u_n, v_n = image_to_direction(image_points[i,0],  image_points[i,1], camera_metadata['intrinsic'])

        r_dir = np.matmul(R_global_cam.transpose(), np.array([image_points[i,2], -u_n*image_points[i,2], -v_n *image_points[i,2]]).reshape(3,1))

        r_dir /= np.linalg.norm(r_dir)

        rays.append(np.concatenate([n_pos_cam,r_dir]).reshape(1,-1))

    return np.vstack(rays)




def world_to_image_python(world_points, camera_metadata):

    projected_points = []

    for i in range(world_points.shape[0]):

        t_h = 0.0
        world_point = world_points[i,:].reshape(-1,1)
        iter_num = 0
        kThreshold = 1e-5
        kMaxIterNum = 4
        residual = 2 * kThreshold
        jacobian = 0.0
        valid = True
        while (np.abs(residual) > kThreshold and iter_num < kMaxIterNum):
            flag, residual, jacobian, normalized_coord = compute_residual_and_jacobian(world_point, t_h, camera_metadata)
            if not flag:
                valid = False
                break
            else:
                delta_t = -residual / jacobian
                t_h += delta_t
                iter_num +=1


        valid,_ , _, normalized_coord = compute_residual_and_jacobian(world_point, t_h, camera_metadata) 

        valid, u_d, v_d = direction_to_image(normalized_coord, camera_metadata)

        if not (u_d.item() >= 0.0 and u_d.item() < camera_metadata['image_width'] and \
                 v_d.item() >= 0.0 and v_d.item() < camera_metadata['image_height']):
            valid = False

        projected_points.append(np.array([u_d,v_d, float(valid)]).reshape(1,-1))

    return np.vstack(projected_points)


### From here on the function are adopted from the waymo utils 

def parse_range_image_and_camera_projection(frame):
  """Parse range images and camera projections given a frame.

  Args:
     frame: open dataset frame proto

  Returns:
     range_images: A dict of {laser_name,
       [range_image_first_return, range_image_second_return]}.
     camera_projections: A dict of {laser_name,
       [camera_projection_from_first_return,
        camera_projection_from_second_return]}.
    range_image_top_pose: range image pixel pose for top lidar.
  """
  range_images = {}
  camera_projections = {}
  range_image_top_pose = None
  for laser in frame.lasers:
    if len(laser.ri_return1.range_image_compressed) > 0:  # pylint: disable=g-explicit-length-test
        range_image_str_tensor = tf.io.decode_compressed(
            laser.ri_return1.range_image_compressed, 'ZLIB')
        ri = dataset_pb2.MatrixFloat()
        ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
        range_images[laser.name] = [ri]

    if laser.name == dataset_pb2.LaserName.TOP:
        range_image_top_pose_str_tensor = tf.io.decode_compressed(
            laser.ri_return1.range_image_pose_compressed, 'ZLIB')
        range_image_top_pose = dataset_pb2.MatrixFloat()
        range_image_top_pose.ParseFromString(
            bytearray(range_image_top_pose_str_tensor.numpy()))


    camera_projection_str_tensor = tf.io.decode_compressed(
        laser.ri_return1.camera_projection_compressed, 'ZLIB')
    cp = dataset_pb2.MatrixInt32()
    cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
    camera_projections[laser.name] = [cp]

    # if len(laser.ri_return2.range_image_compressed) > 0:  # pylint: disable=g-explicit-length-test
    #     range_image_str_tensor = tf.io.decode_compressed(
    #         laser.ri_return2.range_image_compressed, 'ZLIB')
    #     ri = dataset_pb2.MatrixFloat()
    #     ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
    #     range_images[laser.name].append(ri)

    #     camera_projection_str_tensor = tf.io.decode_compressed(
    #         laser.ri_return2.camera_projection_compressed, 'ZLIB')
    #     cp = dataset_pb2.MatrixInt32()
    #     cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
    #     camera_projections[laser.name].append(cp)

  return range_images, camera_projections, range_image_top_pose


def convert_range_image_to_point_cloud(frame,
                                       range_images,
                                       camera_projections,
                                       range_image_top_pose,
                                       ri_index=0,
                                       keep_polar_features=False,
                                       return_rays=True):
  """Convert range images to point cloud.

  Args:
    frame: open dataset frame
    range_images: A dict of {laser_name, [range_image_first_return,
      range_image_second_return]}.
    camera_projections: A dict of {laser_name,
      [camera_projection_from_first_return,
      camera_projection_from_second_return]}.
    range_image_top_pose: range image pixel pose for top lidar.
    ri_index: 0 for the first return, 1 for the second return.
    keep_polar_features: If true, keep the features from the polar range image
      (i.e. range, intensity, and elongation) as the first features in the
      output range image.

  Returns:
    points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
      (NOTE: Will be {[N, 6]} if keep_polar_features is true.
    cp_points: {[N, 6]} list of camera projections of length 5
      (number of lidars).
  """
  calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
  points = []

  cp_points = []

  cartesian_range_images = convert_range_image_to_cartesian(
      frame, range_images, range_image_top_pose, ri_index, keep_polar_features, return_rays)

  for c in calibrations:
    range_image = range_images[c.name][ri_index]
    range_image_tensor = tf.reshape(
        tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
    range_image_mask = range_image_tensor[..., 0] > 0

    range_image_cartesian = cartesian_range_images[c.name]
    points_tensor = tf.gather_nd(range_image_cartesian,
                                 tf.compat.v1.where(range_image_mask))

    cp = camera_projections[c.name][ri_index]
    cp_tensor = tf.reshape(tf.convert_to_tensor(value=cp.data), cp.shape.dims)
    cp_points_tensor = tf.gather_nd(cp_tensor,
                                    tf.compat.v1.where(range_image_mask))
    points.append(points_tensor.numpy())
    cp_points.append(cp_points_tensor.numpy())

  return points, cp_points


def convert_range_image_to_cartesian(frame,
                                     range_images,
                                     range_image_top_pose,
                                     ri_index=0,
                                     keep_polar_features=False,
                                     keep_rays=True):
  """Convert range images from polar coordinates to Cartesian coordinates.

  Args:
    frame: open dataset frame
    range_images: A dict of {laser_name, [range_image_first_return,
       range_image_second_return]}.
    range_image_top_pose: range image pixel pose for top lidar.
    ri_index: 0 for the first return, 1 for the second return.
    keep_polar_features: If true, keep the features from the polar range image
      (i.e. range, intensity, and elongation) as the first features in the
      output range image.

  Returns:
    dict of {laser_name, (H, W, D)} range images in Cartesian coordinates. D
      will be 3 if keep_polar_features is False (x, y, z) and 6 if
      keep_polar_features is True (range, intensity, elongation, x, y, z).
  """
  cartesian_range_images = {}
  frame_pose = tf.convert_to_tensor(
      value=np.reshape(np.array(frame.pose.transform), [4, 4]))

  # [H, W, 6]
  range_image_top_pose_tensor = tf.reshape(
      tf.convert_to_tensor(value=range_image_top_pose.data),
      range_image_top_pose.shape.dims)
  # [H, W, 3, 3]
  range_image_top_pose_tensor_rotation = get_rotation_matrix(
      range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
      range_image_top_pose_tensor[..., 2])
  range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
  range_image_top_pose_tensor = get_transform(
      range_image_top_pose_tensor_rotation,
      range_image_top_pose_tensor_translation)

  for c in frame.context.laser_calibrations:
    range_image = range_images[c.name][ri_index]
    if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
      beam_inclinations = compute_inclination(
          tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
          height=range_image.shape.dims[0])
    else:
      beam_inclinations = tf.constant(c.beam_inclinations)

    beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
    extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

    range_image_tensor = tf.reshape(
        tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
    pixel_pose_local = None
    frame_pose_local = None
    if c.name == dataset_pb2.LaserName.TOP:
      pixel_pose_local = range_image_top_pose_tensor
      pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
      frame_pose_local = tf.expand_dims(frame_pose, axis=0)

    range_image_cartesian = extract_point_cloud_from_range_image(
        tf.expand_dims(range_image_tensor[..., 0], axis=0),
        tf.expand_dims(extrinsic, axis=0),
        tf.expand_dims(tf.convert_to_tensor(value=beam_inclinations), axis=0),
        pixel_pose=pixel_pose_local,
        frame_pose=frame_pose_local,
        keep_rays=True)


    intensity = range_image_tensor[...,1]
    elongation = range_image_tensor[...,2]

    if keep_rays:
        end_point_cartesian = tf.squeeze(range_image_cartesian[0], axis=0)
        source_point_cartesian = tf.squeeze(range_image_cartesian[1], axis=0)

        range_image_cartesian = tf.concat(
          [source_point_cartesian, end_point_cartesian, intensity[:,:,None], elongation[:,:,None]], axis=-1)  
    else:
        range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
    
    cartesian_range_images[c.name] = range_image_cartesian

  return cartesian_range_images


def extract_point_cloud_from_range_image(range_image,
                                         extrinsic,
                                         inclination,
                                         pixel_pose=None,
                                         frame_pose=None,
                                         dtype=tf.float32,
                                         scope=None,
                                         keep_rays=True):
  """Extracts point cloud from range image.

  Args:
    range_image: [B, H, W] tensor. Lidar range images.
    extrinsic: [B, 4, 4] tensor. Lidar extrinsic.
    inclination: [B, H] tensor. Inclination for each row of the range image.
      0-th entry corresponds to the 0-th row of the range image.
    pixel_pose: [B, H, W, 4, 4] tensor. If not None, it sets pose for each range
      image pixel.
    frame_pose: [B, 4, 4] tensor. This must be set when pixel_pose is set. It
      decides the vehicle frame at which the cartesian points are computed.
    dtype: float type to use internally. This is needed as extrinsic and
      inclination sometimes have higher resolution than range_image.
    scope: the name scope.

  Returns:
    range_image_cartesian: [B, H, W, 3] with {x, y, z} as inner dims in vehicle
    frame.
  """
  with tf.compat.v1.name_scope(
      scope, 'ExtractPointCloudFromRangeImage',
      [range_image, extrinsic, inclination, pixel_pose, frame_pose]):
    range_image_polar = compute_range_image_polar(
        range_image, extrinsic, inclination, dtype=dtype)
    range_image_cartesian = compute_range_image_cartesian(
        range_image_polar,
        extrinsic,
        pixel_pose=pixel_pose,
        frame_pose=frame_pose,
        dtype=dtype,
        keep_rays=keep_rays)
    
    if keep_rays:
        return range_image_cartesian[0], range_image_cartesian[1]
    else:
        return range_image_cartesian


def compute_inclination(inclination_range, height, scope=None):
  """Computes uniform inclination range based the given range and height.

  Args:
    inclination_range: [..., 2] tensor. Inner dims are [min inclination, max
      inclination].
    height: an integer indicates height of the range image.
    scope: the name scope.

  Returns:
    inclination: [..., height] tensor. Inclinations computed.
  """
  with tf.compat.v1.name_scope(scope, 'ComputeInclination',
                               [inclination_range]):
    diff = inclination_range[..., 1] - inclination_range[..., 0]
    inclination = (
        (.5 + tf.cast(tf.range(0, height), dtype=inclination_range.dtype)) /
        tf.cast(height, dtype=inclination_range.dtype) *
        tf.expand_dims(diff, axis=-1) + inclination_range[..., 0:1])
    return inclination



def compute_range_image_polar(range_image,
                              extrinsic,
                              inclination,
                              dtype=tf.float32,
                              scope=None):
  """Computes range image polar coordinates.

  Args:
    range_image: [B, H, W] tensor. Lidar range images.
    extrinsic: [B, 4, 4] tensor. Lidar extrinsic.
    inclination: [B, H] tensor. Inclination for each row of the range image.
      0-th entry corresponds to the 0-th row of the range image.
    dtype: float type to use internally. This is needed as extrinsic and
      inclination sometimes have higher resolution than range_image.
    scope: the name scope.

  Returns:
    range_image_polar: [B, H, W, 3] polar coordinates.
  """
  # pylint: disable=unbalanced-tuple-unpacking
  _, height, width = _combined_static_and_dynamic_shape(range_image)
  range_image_dtype = range_image.dtype
  range_image = tf.cast(range_image, dtype=dtype)
  extrinsic = tf.cast(extrinsic, dtype=dtype)
  inclination = tf.cast(inclination, dtype=dtype)

  with tf.compat.v1.name_scope(scope, 'ComputeRangeImagePolar',
                               [range_image, extrinsic, inclination]):
    with tf.compat.v1.name_scope('Azimuth'):
      # [B].
      az_correction = tf.atan2(extrinsic[..., 1, 0], extrinsic[..., 0, 0])
      # [W].
      ratios = (tf.cast(tf.range(width, 0, -1), dtype=dtype) - .5) / tf.cast(
          width, dtype=dtype)
      # [B, W].
      azimuth = (ratios * 2. - 1.) * np.pi - tf.expand_dims(az_correction, -1)

    # [B, H, W]
    azimuth_tile = tf.tile(azimuth[:, tf.newaxis, :], [1, height, 1])
    # [B, H, W]
    inclination_tile = tf.tile(inclination[:, :, tf.newaxis], [1, 1, width])
    range_image_polar = tf.stack([azimuth_tile, inclination_tile, range_image],
                                 axis=-1)
    return tf.cast(range_image_polar, dtype=range_image_dtype)


def compute_range_image_cartesian(range_image_polar,
                                  extrinsic,
                                  pixel_pose=None,
                                  frame_pose=None,
                                  dtype=tf.float32,
                                  scope=None,
                                  keep_rays=True,
                                  return_local_coordinates=False):
  """Computes range image cartesian coordinates from polar ones.

  Args:
    range_image_polar: [B, H, W, 3] float tensor. Lidar range image in polar
      coordinate in sensor frame.
    extrinsic: [B, 4, 4] float tensor. Lidar extrinsic.
    pixel_pose: [B, H, W, 4, 4] float tensor. If not None, it sets pose for each
      range image pixel.
    frame_pose: [B, 4, 4] float tensor. This must be set when pixel_pose is set.
      It decides the vehicle frame at which the cartesian points are computed.
    dtype: float type to use internally. This is needed as extrinsic and
      inclination sometimes have higher resolution than range_image.
    scope: the name scope.

  Returns:
    range_image_cartesian: [B, H, W, 3] cartesian coordinates.
  """
  range_image_polar_dtype = range_image_polar.dtype
  range_image_polar = tf.cast(range_image_polar, dtype=dtype)
  extrinsic = tf.cast(extrinsic, dtype=dtype)
  if pixel_pose is not None:
    pixel_pose = tf.cast(pixel_pose, dtype=dtype)
  if frame_pose is not None:
    frame_pose = tf.cast(frame_pose, dtype=dtype)

  with tf.compat.v1.name_scope(
      scope, 'ComputeRangeImageCartesian',
      [range_image_polar, extrinsic, pixel_pose, frame_pose]):
    azimuth, inclination, range_image_range = tf.unstack(
        range_image_polar, axis=-1)

    cos_azimuth = tf.cos(azimuth)
    sin_azimuth = tf.sin(azimuth)
    cos_incl = tf.cos(inclination)
    sin_incl = tf.sin(inclination)

    # [B, H, W].
    x = cos_azimuth * cos_incl * range_image_range
    y = sin_azimuth * cos_incl * range_image_range
    z = sin_incl * range_image_range

    # [B, H, W, 3]
    range_image_points = tf.stack([x, y, z], -1)
    range_image_center = tf.zeros_like(range_image_points)

    # [B, 3, 3]
    rotation = extrinsic[..., 0:3, 0:3]
    # translation [B, 1, 3]
    translation = tf.expand_dims(tf.expand_dims(extrinsic[..., 0:3, 3], 1), 1)

    # To vehicle frame.
    # [B, H, W, 3]
    range_image_points = tf.einsum('bkr,bijr->bijk', rotation,
                                   range_image_points) + translation

    range_image_center = tf.einsum('bkr,bijr->bijk', rotation,
                                range_image_center) + translation
    if pixel_pose is not None:
      # To global frame.
      # [B, H, W, 3, 3]
      pixel_pose_rotation = pixel_pose[..., 0:3, 0:3]
      # [B, H, W, 3]
      pixel_pose_translation = pixel_pose[..., 0:3, 3]
      # [B, H, W, 3]
      range_image_points = tf.einsum(
          'bhwij,bhwj->bhwi', pixel_pose_rotation,
          range_image_points) + pixel_pose_translation

      range_image_center = tf.einsum(
          'bhwij,bhwj->bhwi', pixel_pose_rotation,
          range_image_center) + pixel_pose_translation
          
      if frame_pose is None:
        raise ValueError('frame_pose must be set when pixel_pose is set.')
      
      if return_local_coordinates:
        # To vehicle frame corresponding to the given frame_pose
        # [B, 4, 4]
        world_to_vehicle = tf.linalg.inv(frame_pose)
        world_to_vehicle_rotation = world_to_vehicle[:, 0:3, 0:3]
        world_to_vehicle_translation = world_to_vehicle[:, 0:3, 3]
        # [B, H, W, 3]
        range_image_points = tf.einsum(
            'bij,bhwj->bhwi', world_to_vehicle_rotation,
            range_image_points) + world_to_vehicle_translation[:, tf.newaxis,
                                                                tf.newaxis, :]
        range_image_center = tf.einsum(
            'bij,bhwj->bhwi', world_to_vehicle_rotation,
            range_image_center) + world_to_vehicle_translation[:, tf.newaxis,
                                                                tf.newaxis, :]

    range_image_points = tf.cast(
        range_image_points, dtype=range_image_polar_dtype)
    range_image_center = tf.cast(
        range_image_center, dtype=range_image_polar_dtype)

    if keep_rays:
        return range_image_points, range_image_center
    else: 
        return range_image_points


def get_rotation_matrix(roll, pitch, yaw, name=None):
  """Gets a rotation matrix given roll, pitch, yaw.

  roll-pitch-yaw is z-y'-x'' intrinsic rotation which means we need to apply
  x(roll) rotation first, then y(pitch) rotation, then z(yaw) rotation.

  https://en.wikipedia.org/wiki/Euler_angles
  http://planning.cs.uiuc.edu/node102.html

  Args:
    roll : x-rotation in radians.
    pitch: y-rotation in radians. The shape must be the same as roll.
    yaw: z-rotation in radians. The shape must be the same as roll.
    name: the op name.

  Returns:
    A rotation tensor with the same data type of the input. Its shape is
      [input_shape_of_yaw, 3 ,3].
  """
  with tf.compat.v1.name_scope(name, 'GetRotationMatrix', [yaw, pitch, roll]):
    cos_roll = tf.cos(roll)
    sin_roll = tf.sin(roll)
    cos_yaw = tf.cos(yaw)
    sin_yaw = tf.sin(yaw)
    cos_pitch = tf.cos(pitch)
    sin_pitch = tf.sin(pitch)

    ones = tf.ones_like(yaw)
    zeros = tf.zeros_like(yaw)

    r_roll = tf.stack([
        tf.stack([ones, zeros, zeros], axis=-1),
        tf.stack([zeros, cos_roll, -1.0 * sin_roll], axis=-1),
        tf.stack([zeros, sin_roll, cos_roll], axis=-1),
    ],
                      axis=-2)
    r_pitch = tf.stack([
        tf.stack([cos_pitch, zeros, sin_pitch], axis=-1),
        tf.stack([zeros, ones, zeros], axis=-1),
        tf.stack([-1.0 * sin_pitch, zeros, cos_pitch], axis=-1),
    ],
                       axis=-2)
    r_yaw = tf.stack([
        tf.stack([cos_yaw, -1.0 * sin_yaw, zeros], axis=-1),
        tf.stack([sin_yaw, cos_yaw, zeros], axis=-1),
        tf.stack([zeros, zeros, ones], axis=-1),
    ],
                     axis=-2)

    return tf.matmul(r_yaw, tf.matmul(r_pitch, r_roll))


def get_transform(rotation, translation):
  """Combines NxN rotation and Nx1 translation to (N+1)x(N+1) transform.

  Args:
    rotation: [..., N, N] rotation tensor.
    translation: [..., N] translation tensor. This must have the same type as
      rotation.

  Returns:
    transform: [..., (N+1), (N+1)] transform tensor. This has the same type as
      rotation.
  """
  with tf.name_scope('GetTransform'):
    # [..., N, 1]
    translation_n_1 = translation[..., tf.newaxis]
    # [..., N, N+1]
    transform = tf.concat([rotation, translation_n_1], axis=-1)
    # [..., N]
    last_row = tf.zeros_like(translation)
    # [..., N+1]
    last_row = tf.concat([last_row, tf.ones_like(last_row[..., 0:1])], axis=-1)
    # [..., N+1, N+1]
    transform = tf.concat([transform, last_row[..., tf.newaxis, :]], axis=-2)
    return transform


def _combined_static_and_dynamic_shape(tensor):
  """Returns a list containing static and dynamic values for the dimensions.

  Returns a list of static and dynamic values for shape dimensions. This is
  useful to preserve static shapes when available in reshape operation.

  Args:
    tensor: A tensor of any type.

  Returns:
    A list of size tensor.shape.ndims containing integers or a scalar tensor.
  """
  static_tensor_shape = tensor.shape.as_list()
  dynamic_tensor_shape = tf.shape(input=tensor)
  combined_shape = []
  for index, dim in enumerate(static_tensor_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(dynamic_tensor_shape[index])
  return combined_shape


def project_point_to_image_plane(world_points, camera_metadata, T_global_cam):

  # Transform points in the camera coordinate system 
  cam_points = (np.matmul(world_points[:,None,:], T_global_cam[:,:3,:3].transpose(0,2,1)) + T_global_cam[:,:3,3:4].transpose(0,2,1)).squeeze(1)
  normalized_points = -cam_points[:,1:3] / cam_points[:,0:1]

  f_u = camera_metadata['intrinsic'][0]
  f_v = camera_metadata['intrinsic'][1]
  c_u = camera_metadata['intrinsic'][2]
  c_v = camera_metadata['intrinsic'][3]
  k1 = camera_metadata['intrinsic'][4]
  k2 = camera_metadata['intrinsic'][5]
  k3 = camera_metadata['intrinsic'][6]  # same as p1 in OpenCV.
  k4 = camera_metadata['intrinsic'][7]  # same as p2 in OpenCV
  k5 = camera_metadata['intrinsic'][8]  # same as k3 in OpenCV.
  img_width = camera_metadata['image_width']
  img_height = camera_metadata['image_height']

  u_n = normalized_points[:,0]
  v_n = normalized_points[:,1]

  r2 = np.square(u_n) + np.square(v_n)
  r4 = r2 * r2
  r6 = r4 * r2

  r_d = 1.0 + k1 * r2 + k2 * r4 + k5 * r6


  # If the radial distortion is too large, the computed coordinates will be unreasonable
  kMinRadialDistortion = 0.8
  kMaxRadialDistortion = 1.2

  invalid_idx = np.where(np.logical_or(np.less_equal(r_d,kMinRadialDistortion),np.greater_equal(r_d,kMaxRadialDistortion)))[0]

  u_nd = u_n * r_d + 2.0 * k3 * u_n * v_n + k4 * (r2 + 2.0 * u_n * u_n)
  v_nd = v_n * r_d + k3 * (r2 + 2.0 * v_n * v_n) + 2.0 * k4 * u_n * v_n

  u_d = u_nd * f_u + c_u
  v_d = v_nd * f_v + c_v

  valid_flag = np.ones_like(u_d)
  # Replace the invalid ones
  r2_sqrt_rcp = 1.0 / np.sqrt(r2)
  clipping_radius = np.sqrt(img_width**2 + img_height**2)
  u_d[invalid_idx] = u_n[invalid_idx] * r2_sqrt_rcp[invalid_idx] * clipping_radius + c_u
  v_d[invalid_idx] = v_n[invalid_idx] * r2_sqrt_rcp[invalid_idx] * clipping_radius + c_v
  valid_flag[invalid_idx] = 0

  # Change the flags of the pixels that project outside of an image
  valid_flag[u_d < 0 ] = 0
  valid_flag[v_d < 0 ] = 0
  valid_flag[u_d > img_width] = 0
  valid_flag[v_d > img_height] = 0

  return np.concatenate((u_d[:,None], v_d[:,None], valid_flag[:,None]), axis=1)
