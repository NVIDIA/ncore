__version__ = "1.0.0"

import nvidia_utils
import numpy as np

def unwind_lidar(pc, transformation_matrices, column_idx):
    return nvidia_utils._lidarUnwinding(pc, transformation_matrices, column_idx)

def image_to_world_ray(image_points, camera_metadata):

    # Initialize the parameters
    intrinsic   = np.array(camera_metadata['intrinsic']).reshape(1,-1)
    camera_model   = camera_metadata['camera_model'] 
    img_height   = camera_metadata['img_height']
    exposure_time   = camera_metadata['exposure_time']
    rolling_shutter_delay   = camera_metadata['rolling_shutter_delay']
    pose_timestamps =  np.array(camera_metadata['ego_pose_timestamps']).reshape(1,-1)
    t_eof = camera_metadata['t_eof']

    # Extract the poses 
    ego_pose_s = camera_metadata['ego_pose_s'] # These are ego poses in the rig coordinate system
    ego_pose_e = camera_metadata['ego_pose_e']
    T_cam_rig = camera_metadata['T_cam_rig'] # Transforms the points/rays from camera coordinate system to the rig

    poses = np.concatenate([ego_pose_s @ T_cam_rig, ego_pose_e @ T_cam_rig], axis=0)

    if image_points.shape[0] == 1:
        image_points = np.tile(image_points, [2,1])
        return nvidia_utils._pixel2WorldRay(image_points, intrinsic, camera_model, img_height, rolling_shutter_delay, exposure_time, T_eof, poses, pose_timestamps)[0,:]
    else:
        return nvidia_utils._pixel2WorldRay(image_points, intrinsic, camera_model, img_height, rolling_shutter_delay, exposure_time, t_eof, poses, pose_timestamps)


def cameraRay2Pixel(cameraPoints, camera_metadata):
    # Initialize the parameters
    intrinsic   = np.array(camera_metadata['intrinsic']).reshape(1,-1)
    camera_model   = camera_metadata['camera_model'] 
    img_height   = camera_metadata['img_height']
    img_width   = camera_metadata['img_width']

    return nvidia_utils._cameraRay2Pixel(cameraPoints, intrinsic, img_width, img_height, camera_model)


def rollingShutterProjection(points, camera_metadata, iterate=False):

    # Initialize the parameters
    intrinsic   = np.array(camera_metadata['intrinsic']).reshape(1,-1)
    camera_model   = camera_metadata['camera_model'] 
    img_height  = camera_metadata['img_height']
    img_width  = camera_metadata['img_width']
    exposure_time   = camera_metadata['exposure_time']
    rolling_shutter_delay   = camera_metadata['rolling_shutter_delay']
    pose_timestamps =  np.array(camera_metadata['ego_pose_timestamps']).reshape(1,-1)
    t_eof = camera_metadata['t_eof']

    # Extract the poses 
    ego_pose_s = camera_metadata['ego_pose_s'] # These are ego poses in the rig coordinate system
    ego_pose_e = camera_metadata['ego_pose_e']
    T_cam_rig = camera_metadata['T_cam_rig'] # Transforms the points/rays from camera coordinate system to the rig

    poses = np.concatenate([np.linalg.inv(T_cam_rig) @ np.linalg.inv(ego_pose_s),
                        np.linalg.inv(T_cam_rig) @ np.linalg.inv(ego_pose_e)], axis=0)


    pixel_coords, trans_matrices, valid_idx = nvidia_utils._rollingShutterProjection(points, intrinsic, img_height, img_width, 
                        rolling_shutter_delay, exposure_time, t_eof, poses, pose_timestamps, camera_model, iterate)

    # Filter out points behind the camera
    frontIdx = np.where(pixel_coords[:,2] > 0.0)[0]

    # Image coordinate check
    pixelcoordIdx = np.where(np.logical_and(np.logical_and(0.0 < pixel_coords[:,0], pixel_coords[:,0] < img_width),
                                            np.logical_and(0.0 < pixel_coords[:,1], pixel_coords[:,1] < img_height)))[0]

    finalIdx = np.intersect1d(frontIdx,pixelcoordIdx)

    return pixel_coords[finalIdx,:2], trans_matrices.reshape(-1,4,4)[finalIdx], valid_idx[finalIdx]