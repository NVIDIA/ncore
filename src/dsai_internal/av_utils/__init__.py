# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

__version__ = "1.0.0"

from typing import Tuple, Union

import numpy as np

import libav_utils_cc  # type: ignore
from src.dsai_internal.data import types

def unwind_lidar(pc, transformation_matrices, column_idx):
    return libav_utils_cc._lidarUnwinding(pc, transformation_matrices, column_idx)


def image_to_world_ray(image_points: np.ndarray, camera_metadata: Union[types.FThetaCameraModelParameters, types.PinholeCameraModelParameters], T_sensor_world: np.ndarray) -> np.ndarray:
    assert len(image_points.shape) == 2, "image points need to be a 2D numpy array"
    assert image_points.shape[1] == 2, "image points need to be a N x 2 array"

    match camera_metadata:
        case types.FThetaCameraModelParameters() as ftheta_parameters:
            img_width, img_height = ftheta_parameters.resolution
            cam_rays = libav_utils_cc._pixel2WorldRayFTheta(
                            image_points.astype(np.float64),
                            ftheta_parameters.fw_poly.reshape(1,-1).astype(np.float64),
                            ftheta_parameters.bw_poly.reshape(1,-1).astype(np.float64),
                            ftheta_parameters.principal_point.reshape(1,-1).astype(np.float64),
                            img_width,
                            img_height,
                            ftheta_parameters.max_angle,
                            ftheta_parameters.shutter_type.name,
                            T_sensor_world.astype(np.float64))
        case types.PinholeCameraModelParameters() as pinhole_parameters:
            img_width, img_height = pinhole_parameters.resolution
            cam_rays = libav_utils_cc._pixel2WorldRayPinhole(
                        image_points.astype(np.float64), 
                        pinhole_parameters.principal_point.reshape(1,-1).astype(np.float64),
                        pinhole_parameters.focal_length.reshape(1,-1).astype(np.float64),
                        pinhole_parameters.radial_poly[:3].reshape(1,-1).astype(np.float64),
                        pinhole_parameters.tangential_poly.reshape(1,-1).astype(np.float64),
                        img_width, img_height, pinhole_parameters.shutter_type.name,
                        T_sensor_world.astype(np.float64))
        case _:
                raise TypeError(
                        f"unsupported camera model parameters type {type(camera_metadata)}, currently supporting Ftheta/Pinhole only"
                    )

    return cam_rays

def pixel2CameraRay(image_points: np.ndarray, camera_metadata: Union[types.FThetaCameraModelParameters, types.PinholeCameraModelParameters]) -> np.ndarray:
    assert len(image_points.shape) == 2, "image points need to be a 2D numpy array"
    assert image_points.shape[1] == 2, "image points need to be a N x 2 array"

    match camera_metadata:
        case types.FThetaCameraModelParameters() as ftheta_parameters:
            img_width, img_height  = ftheta_parameters.resolution
            cam_rays = libav_utils_cc._pixel2CameraRayFTheta(
                            image_points.astype(np.float64), ftheta_parameters.fw_poly.reshape(1,-1).astype(np.float64),
                            ftheta_parameters.bw_poly.reshape(1,-1).astype(np.float64),
                            ftheta_parameters.principal_point.reshape(1,-1).astype(np.float64), img_width,
                            img_height, ftheta_parameters.max_angle, ftheta_parameters.shutter_type.name)
        case types.PinholeCameraModelParameters() as pinhole_parameters:
            img_width, img_height  = pinhole_parameters.resolution
            cam_rays = libav_utils_cc._pixel2CameraRayPinhole(
                        image_points.astype(np.float64), 
                        pinhole_parameters.principal_point.reshape(1,-1).astype(np.float64),
                        pinhole_parameters.focal_length.reshape(1,-1).astype(np.float64),
                        pinhole_parameters.radial_poly[:3].reshape(1,-1).astype(np.float64),
                        pinhole_parameters.tangential_poly.reshape(1,-1).astype(np.float64),
                        img_width, img_height, pinhole_parameters.shutter_type.name)
        case _:
                raise TypeError(
                        f"unsupported camera model parameters type {type(camera_metadata)}, currently supporting Ftheta/Pinhole only"
                    )

    return cam_rays


def cameraRay2Pixel(cameraPoints: np.ndarray, camera_metadata: Union[types.FThetaCameraModelParameters, types.PinholeCameraModelParameters]) -> Tuple[np.ndarray, np.ndarray]:

    match camera_metadata:
        case types.FThetaCameraModelParameters() as ftheta_parameters:
            img_width, img_height  = ftheta_parameters.resolution
            pixel_coords, valid_flag = libav_utils_cc._cameraRay2PixelFTheta(
                            cameraPoints.astype(np.float64), ftheta_parameters.fw_poly.reshape(1,-1).astype(np.float64),
                            ftheta_parameters.bw_poly.reshape(1,-1).astype(np.float64),
                            ftheta_parameters.principal_point.reshape(1,-1).astype(np.float64), img_width,
                            img_height, ftheta_parameters.max_angle, ftheta_parameters.shutter_type.name)
        case types.PinholeCameraModelParameters() as pinhole_parameters:
            img_width, img_height  = pinhole_parameters.resolution
            pixel_coords, valid_flag = libav_utils_cc._cameraRay2PixelPinhole(
                            cameraPoints.astype(np.float64), 
                            pinhole_parameters.principal_point.reshape(1,-1).astype(np.float64),
                            pinhole_parameters.focal_length.reshape(1,-1).astype(np.float64),
                            pinhole_parameters.radial_poly[:3].reshape(1,-1).astype(np.float64),
                            pinhole_parameters.tangential_poly.reshape(1,-1).astype(np.float64),
                            img_width, img_height, pinhole_parameters.shutter_type.name)
        case _:
                raise TypeError(
                        f"unsupported camera model parameters type {type(camera_metadata)}, currently supporting Ftheta/Pinhole only"
                    )
    
    return pixel_coords, valid_flag


def rollingShutterProjection(points: np.ndarray,
                             camera_metadata: Union[types.FThetaCameraModelParameters, types.PinholeCameraModelParameters],
                             T_world_cam: np.ndarray,
                             iter : int=1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    match camera_metadata:
        case types.FThetaCameraModelParameters() as ftheta_parameters:
            img_width, img_height  = camera_metadata.resolution

            pixel_coords, trans_matrices, valid_proj, initial_valid_idx = libav_utils_cc._rollingShutterProjectionFTheta(
                                    points.astype(np.float64), ftheta_parameters.fw_poly.reshape(1,-1).astype(np.float64),
                                    ftheta_parameters.bw_poly.reshape(1,-1).astype(np.float64),
                                    ftheta_parameters.principal_point.reshape(1,-1).astype(np.float64), img_width,
                                    img_height, ftheta_parameters.max_angle,ftheta_parameters.shutter_type.name, 
                                    T_world_cam.astype(np.float64), iter)

        case types.PinholeCameraModelParameters() as pinhole_parameters:
            img_width, img_height  = pinhole_parameters.resolution

            pixel_coords, trans_matrices, valid_proj, initial_valid_idx = libav_utils_cc._rollingShutterProjectionFTheta(
                                    points.astype(np.float64), pinhole_parameters.principal_point.reshape(1,-1).astype(np.float64),
                                    pinhole_parameters.focal_length.reshape(1,-1).astype(np.float64),
                                    pinhole_parameters.radial_poly[:3].reshape(1,-1).astype(np.float64),
                                    pinhole_parameters.tangential_poly.reshape(1,-1).astype(np.float64),
                                    img_width, img_height, pinhole_parameters.shutter_type.name, 
                                    T_world_cam.astype(np.float64), iter)
        case _:
                raise TypeError(
                        f"unsupported camera model parameters type {type(camera_metadata)}, currently supporting Ftheta/Pinhole only"
                    )

    valid_idx = initial_valid_idx[valid_proj]
    pixel_coords = pixel_coords[valid_proj,:]
    trans_matrices = trans_matrices.reshape(-1,4,4)[valid_proj,:,:]

    return pixel_coords, trans_matrices, valid_idx


def isWithin3DBBox(pc: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
    # Chech the validity of the input
    assert pc.shape[1] == 3, "Wrong PC input size"
    assert len(bboxes.shape) == 2, "bboxes need to be a 2D numpy array"
    assert bboxes.shape[1] == 9, "bboxes need to be a 2D numpy array"

    return libav_utils_cc._isWithin3DBoundingBox(pc, bboxes)
