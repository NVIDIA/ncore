# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

__version__ = "1.0.0"

import libav_utils_cc  # type: ignore
import numpy as np
from src.dsai_internal.data import types

def unwind_lidar(pc, transformation_matrices, column_idx):
    return libav_utils_cc._lidarUnwinding(pc, transformation_matrices, column_idx)


def image_to_world_ray(image_points, camera_metadata, T_sensor_world):
    assert len(image_points.shape) == 2, "image points need to be a 2D numpy array"
    assert image_points.shape[1] == 2, "image points need to be a N x 2 array"

    if isinstance(camera_metadata, types.FThetaCameraModelParameters):
        img_width, img_height  = camera_metadata.resolution
        cam_rays = libav_utils_cc._pixel2WorldRayFTheta(
                        image_points.astype(np.float64), camera_metadata.fw_poly.reshape(1,-1).astype(np.float64),
                        camera_metadata.bw_poly.reshape(1,-1).astype(np.float64),
                        camera_metadata.principal_point.reshape(1,-1).astype(np.float64), img_width,
                        img_height, camera_metadata.max_angle, camera_metadata.shutter_type.name,
                        T_sensor_world.astype(np.float64))

    elif isinstance(camera_metadata, types.PinholeCameraModelParameters):
        img_width, img_height  = camera_metadata.resolution
        cam_rays = libav_utils_cc._pixel2WorldRayPinhole(
                    image_points.astype(np.float64), 
                    camera_metadata.principal_point.reshape(1,-1).astype(np.float64),
                    camera_metadata.focal_length.reshape(1,-1).astype(np.float64),
                    camera_metadata.radial_poly[:3].reshape(1,-1).astype(np.float64),
                    camera_metadata.tangential_poly.reshape(1,-1).astype(np.float64),
                    img_width, img_height, camera_metadata.shutter_type.name,
                    T_sensor_world.astype(np.float64))

    return cam_rays

def pixel2CameraRay(image_points, camera_metadata):
    assert len(image_points.shape) == 2, "image points need to be a 2D numpy array"
    assert image_points.shape[1] == 2, "image points need to be a N x 2 array"

    if isinstance(camera_metadata,types.FThetaCameraModelParameters):
        img_width, img_height  = camera_metadata.resolution
        cam_rays = libav_utils_cc._pixel2CameraRayFTheta(
                        image_points.astype(np.float64), camera_metadata.fw_poly.reshape(1,-1).astype(np.float64),
                        camera_metadata.bw_poly.reshape(1,-1).astype(np.float64),
                        camera_metadata.principal_point.reshape(1,-1).astype(np.float64), img_width,
                        img_height, camera_metadata.max_angle, camera_metadata.shutter_type.name)

    elif isinstance(camera_metadata, types.PinholeCameraModelParameters):
        img_width, img_height  = camera_metadata.resolution
        cam_rays = libav_utils_cc._pixel2CameraRayPinhole(
                    image_points.astype(np.float64), 
                    camera_metadata.principal_point.reshape(1,-1).astype(np.float64),
                    camera_metadata.focal_length.reshape(1,-1).astype(np.float64),
                    camera_metadata.radial_poly[:3].reshape(1,-1).astype(np.float64),
                    camera_metadata.tangential_poly.reshape(1,-1).astype(np.float64),
                    img_width, img_height, camera_metadata.shutter_type.name)

    return cam_rays


def cameraRay2Pixel(cameraPoints, camera_metadata):

    if isinstance(camera_metadata,types.FThetaCameraModelParameters):
        img_width, img_height  = camera_metadata.resolution
        pixel_coords, valid_flag = libav_utils_cc._cameraRay2PixelFTheta(
                        cameraPoints.astype(np.float64), camera_metadata.fw_poly.reshape(1,-1).astype(np.float64),
                        camera_metadata.bw_poly.reshape(1,-1).astype(np.float64),
                        camera_metadata.principal_point.reshape(1,-1).astype(np.float64), img_width,
                        img_height, camera_metadata.max_angle,camera_metadata.shutter_type.name)

    elif isinstance(camera_metadata, types.PinholeCameraModelParameters):
        img_width, img_height  = camera_metadata.resolution
        pixel_coords, valid_flag = libav_utils_cc._cameraRay2PixelPinhole(
                        cameraPoints.astype(np.float64), 
                        camera_metadata.principal_point.reshape(1,-1).astype(np.float64),
                        camera_metadata.focal_length.reshape(1,-1).astype(np.float64),
                        camera_metadata.radial_poly[:3].reshape(1,-1).astype(np.float64),
                        camera_metadata.tangential_poly.reshape(1,-1).astype(np.float64),
                        img_width, img_height, camera_metadata.shutter_type.name)

    return pixel_coords, valid_flag


def rollingShutterProjection(points, camera_metadata, T_world_cam,  iter=1):

    if isinstance(camera_metadata, types.FThetaCameraModelParameters):

        img_width, img_height  = camera_metadata.resolution

        pixel_coords, trans_matrices, valid_proj, initial_valid_idx = libav_utils_cc._rollingShutterProjectionFTheta(
                                points.astype(np.float64), camera_metadata.fw_poly.reshape(1,-1).astype(np.float64),
                                camera_metadata.bw_poly.reshape(1,-1).astype(np.float64),
                                camera_metadata.principal_point.reshape(1,-1).astype(np.float64), img_width,
                                img_height, camera_metadata.max_angle,camera_metadata.shutter_type.name, 
                                T_world_cam.astype(np.float64), iter)

    elif isinstance(camera_metadata, types.PinholeCameraModelParameters):

        img_width, img_height  = camera_metadata.resolution

        pixel_coords, trans_matrices, valid_proj, initial_valid_idx = libav_utils_cc._rollingShutterProjectionFTheta(
                                points.astype(np.float64), camera_metadata.principal_point.reshape(1,-1).astype(np.float64),
                                camera_metadata.focal_length.reshape(1,-1).astype(np.float64),
                                camera_metadata.radial_poly[:3].reshape(1,-1).astype(np.float64),
                                camera_metadata.tangential_poly.reshape(1,-1).astype(np.float64),
                                img_width, img_height, camera_metadata.shutter_type.name, 
                                T_world_cam.astype(np.float64), iter)
    
    else:
        raise TypeError("Camera model must be one of [types.FThetaCameraModelParameters, types.PinholeCameraModelParameters]")

    valid_idx = initial_valid_idx[valid_proj]
    pixel_coords = pixel_coords[valid_proj,:]
    trans_matrices = trans_matrices.reshape(-1,4,4)[valid_proj,:,:]

    return pixel_coords, trans_matrices, valid_idx


def isWithin3DBBox(pc, bboxes):
    # Chech the validity of the input
    assert pc.shape[1] == 3, "Wrong PC input size"
    assert len(bboxes.shape) == 2, "bboxes need to be a 2D numpy array"
    assert bboxes.shape[1] == 9, "bboxes need to be a 2D numpy array"

    return libav_utils_cc._isWithin3DBoundingBox(pc, bboxes)
