__version__ = "1.0.0"

import av_utils
import numpy as np

def unwind_lidar(pc, transformation_matrices, column_idx):
    return av_utils._lidarUnwinding(pc, transformation_matrices, column_idx)


def image_to_world_ray(image_points, camera_metadata):
    # Initialize the parameters
    intrinsic   = np.array(camera_metadata['intrinsic']).reshape(1,-1)
    camera_model   = camera_metadata['camera_model'] 
    img_height  = camera_metadata['img_height']
    img_width  = camera_metadata['img_width']
    rs_direction   = camera_metadata['rolling_shutter_direction']
    ego_pose_timestamps =  np.array(camera_metadata['ego_pose_timestamps']).reshape(1,-1).astype(np.float64)

    # Extract the poses 
    ego_pose_s = camera_metadata['ego_pose_s'] # These are ego poses in the rig coordinate system
    ego_pose_e = camera_metadata['ego_pose_e']
    T_cam_rig = camera_metadata['T_cam_rig'] # Transforms the points/rays from camera coordinate system to the rig

    T_cam_world = np.concatenate(
        [ego_pose_s @ T_cam_rig, ego_pose_e @ T_cam_rig], axis=0)

    if image_points.shape[0] == 1:
        image_points = np.tile(image_points, [2, 1])
        return av_utils._pixel2WorldRay(image_points, intrinsic, camera_model, img_height, img_width,
                                        T_cam_world, ego_pose_timestamps, rs_direction)[0, :]
    else:
        return av_utils._pixel2WorldRay(image_points, intrinsic, camera_model, img_height, img_width, 
                                        T_cam_world, ego_pose_timestamps, rs_direction)


def cameraRay2Pixel(cameraPoints, camera_metadata):
    # Initialize the parameters
    intrinsic   = np.array(camera_metadata['intrinsic']).reshape(1,-1)
    camera_model   = camera_metadata['camera_model'] 
    img_height   = camera_metadata['img_height']
    img_width   = camera_metadata['img_width']

    return av_utils._cameraRay2Pixel(cameraPoints, intrinsic, img_width, img_height, camera_model)


def rollingShutterProjection(points, camera_metadata, iter=1):
    # Initialize the parameters
    intrinsic   = np.array(camera_metadata['intrinsic']).reshape(1,-1)
    camera_model   = camera_metadata['camera_model'] 
    img_height  = camera_metadata['img_height']
    img_width  = camera_metadata['img_width']
    rs_direction   = camera_metadata['rolling_shutter_direction']
    ego_pose_timestamps =  np.array(camera_metadata['ego_pose_timestamps']).reshape(1,-1).astype(np.float64)

    # Extract the poses 
    ego_pose_s = camera_metadata['ego_pose_s'] # These are ego poses in the rig coordinate system
    ego_pose_e = camera_metadata['ego_pose_e']
    T_cam_rig = camera_metadata['T_cam_rig'] # Transforms the points/rays from camera coordinate system to the rig

    T_world_cam = np.concatenate([np.linalg.inv(T_cam_rig) @ np.linalg.inv(ego_pose_s),
                        np.linalg.inv(T_cam_rig) @ np.linalg.inv(ego_pose_e)], axis=0)

    pixel_coords, trans_matrices, valid_proj, initial_valid_idx = av_utils._rollingShutterProjection(points, intrinsic, img_height, img_width, 
                        rs_direction, T_world_cam, ego_pose_timestamps, camera_model, iter)

    valid_idx = initial_valid_idx[valid_proj]
    pixel_coords = pixel_coords[valid_proj,:]
    trans_matrices = trans_matrices.reshape(-1,4,4)[valid_proj,:,:]

    return pixel_coords, trans_matrices, valid_idx