# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import logging

from abc import ABC, abstractmethod
from typing import Tuple, Union

import torch 
import numpy as np

from src.dsai_internal.data import types


class CameraModel(ABC):
    ''' Base camera model class '''
    def __init__(self):
        self.resolution: torch.Tensor
        self.shutter_type: str
        self.device: str
        self.dtype: torch.dtype


    @abstractmethod 
    def pixel_to_camera_ray(self, image_points: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        '''
        Computes camera rays for each image point
        '''
        pass
    
    @abstractmethod
    def camera_ray_to_pixel(self, cam_rays: Union[torch.Tensor, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        For each camera ray computes the corresponding pixel coordinates
        '''
        pass

    @staticmethod
    def from_parameters(cam_model_parameters: Union[types.FThetaCameraModelParameters, types.PinholeCameraModelParameters]) -> CameraModel:
        '''
        Initialize a camera model class
        '''
        match cam_model_parameters:
            case types.FThetaCameraModelParameters():
                return FThetaCameraModel(cam_model_parameters)
            case types.PinholeCameraModelParameters():
                return PinholeCameraModel(cam_model_parameters)
            case _:
                raise TypeError(
                        f"unsupported camera model type {type(cam_model_parameters)}, currently supporting Ftheta/Pinhole only"
                    )

    def to_torch(self, var: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:

        if isinstance(var, np.ndarray):
            var = torch.from_numpy(var)
  
        return var.to(self.device)

    def rolling_shutter_projection(self, points: Union[torch.Tensor, np.ndarray], 
                                   T_world_sensor: Union[torch.Tensor, np.ndarray], 
                                   max_iter: int = 10,
                                   min_error: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Check if the variables are numpy, convert them to torch and send them to correct device
        points = self.to_torch(points).to(self.dtype)
        T_world_sensor = self.to_torch(T_world_sensor).to(self.dtype)

        assert T_world_sensor.shape == (2,4,4)
        assert len(points.shape) == 2
        assert points.shape[1] == 3
        assert points.dtype == self.dtype
        assert T_world_sensor.dtype == self.dtype

        T_world_sensor_s = T_world_sensor[0,:,:]
        T_world_sensor_e = T_world_sensor[1,:,:]

        # Convert the start and end rotation matrix to quaternions
        ego_pose_s_quat = self.__rotmat_to_unitquat(T_world_sensor_s[None, :3, :3]) # [1, 4]
        ego_pose_e_quat = self.__rotmat_to_unitquat(T_world_sensor_e[None, :3, :3]) # [1, 4]

        mof_rot = self.__unitquat_to_rotmat(self.__unitquat_slerp(ego_pose_s_quat, 
                                            ego_pose_e_quat,
                                            torch.Tensor([0.5]).to(T_world_sensor_s))).squeeze() # [3, 3]

        mof_trans = 0.5 * T_world_sensor_s[:3, 3] + 0.5 * T_world_sensor_e[:3, 3] # [3]

        # Do the initial transformation
        cam_rays = (mof_rot @ points.transpose(0,1) + mof_trans[..., None]).transpose(0,1)
        init_pixel, valid = self.camera_ray_to_pixel(cam_rays)
        
        # For valid pixels, compute the new timestamp and project again 
        if valid.any():
            pixel_rs_prev = init_pixel[valid,:].clone()
            current_int = 0
            error = 1e12

            while current_int < max_iter and error > min_error:
                t = self.__get_interpolation_timestamp(pixel_rs_prev)

                rot_rs = self.__unitquat_to_rotmat(self.__unitquat_slerp(ego_pose_s_quat.repeat(t.shape[0], 1), 
                                                                         ego_pose_e_quat.repeat(t.shape[0], 1), t)).squeeze() #[n_valid, 3, 3]

                trans_rs = (1-t)[..., None] * T_world_sensor_s[:3, 3:4].transpose(0,1).repeat(t.shape[0],1) + t[..., None] * T_world_sensor_e[:3, 3:4].transpose(0,1).repeat(t.shape[0],1)

                cam_points_rs = (torch.bmm(rot_rs, points[valid, :, None]) + trans_rs[...,None]).squeeze(-1)
                pixel_rs, valid_rs = self.camera_ray_to_pixel(cam_points_rs.squeeze())

                error = torch.linalg.norm(pixel_rs - pixel_rs_prev, dim=1).mean()
                pixel_rs_prev = pixel_rs.clone()
                current_int += 1

        # Combine valid flags
        valid_idx = valid.clone()
        valid[valid_idx] = valid[valid_idx] & valid_rs

        # Generate the output matrix
        trans_matrices = torch.empty((valid.sum().int().item(), 4, 4)).to(rot_rs)   # type: ignore
        trans_matrices[:,:3, 3] = trans_rs[valid_rs, :]
        trans_matrices[:,:3,:3] = rot_rs[valid_rs, ...]

        return pixel_rs[valid_rs,:], trans_matrices, torch.where(valid)[0]


    def project_without_rolling_shutter(self, points: Union[torch.Tensor, np.ndarray], 
                                        T_world_sensor: Union[torch.Tensor, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Check if the variables are numpy, convert them to torch and send them to correct device
        points = self.to_torch(points).to(self.dtype)
        T_world_sensor = self.to_torch(T_world_sensor).to(self.dtype)

        assert T_world_sensor.shape == (8,4)
        assert len(points.shape) == 2
        assert points.shape[1] == 3
        assert points.dtype == self.dtype
        assert T_world_sensor.dtype == self.dtype

        T_world_sensor_s = T_world_sensor[:4,:4]
        T_world_sensor_e = T_world_sensor[4:,:4]

        # Convert the start and end rotation matrix to quaternions
        ego_pose_s_quat = self.__rotmat_to_unitquat(T_world_sensor_s[None, :3, :3]) # [1, 4]
        ego_pose_e_quat = self.__rotmat_to_unitquat(T_world_sensor_e[None, :3, :3]) # [1, 4]

        mof_rot = self.__unitquat_to_rotmat(self.__unitquat_slerp(ego_pose_s_quat, 
                                            ego_pose_e_quat, torch.Tensor([0.5]).to(T_world_sensor_s))).squeeze() # [3, 3]

        mof_trans = 0.5 * T_world_sensor_s[:3, 3] + 0.5 * T_world_sensor_e[:3, 3] # [3]

        # Do the initial transformation
        cam_rays = (mof_rot @ points.transpose(0,1) + mof_trans[..., None]).transpose(0,1)
        init_pixel, valid = self.camera_ray_to_pixel(cam_rays)

        trans_matrix = torch.empty((1, 4, 4)).to(self.device, self.dtype)
        trans_matrix[0,:3, 3] = mof_trans
        trans_matrix[0,:3,:3] = mof_rot

        return init_pixel[valid,:], trans_matrix, torch.where(valid)[0]


    def camera_to_world_ray(self, image_points: Union[torch.Tensor, np.ndarray], 
                            T_sensor_world: Union[torch.Tensor, np.ndarray]) -> torch.Tensor: 

        # Check if the variables are numpy, convert them to torch and send them to correct device
        image_points = self.to_torch(image_points).to(self.dtype)
        T_sensor_world = self.to_torch(T_sensor_world).to(self.dtype)

        assert T_sensor_world.shape == (8,4)
        assert len(image_points.shape) == 2
        assert image_points.shape[1] == 2
        assert image_points.dtype == self.dtype
        assert T_sensor_world.dtype == self.dtype

        # Initialize the output variable 
        world_rays = torch.empty((image_points.shape[0], 6), dtype=self.dtype).to(self.device)

        # Unproject the pixels to camera rays
        camera_rays = self.pixel_to_camera_ray(image_points)

        # Extract the start and end pose 
        T_sensor_world_s = T_sensor_world[:4,:4]
        T_sensor_world_e = T_sensor_world[4:,:4]

        # Convert the start and end rotation matrix to quaternions
        T_sensor_world_s_quat = self.__rotmat_to_unitquat(T_sensor_world_s[None, :3, :3]) # [1, 4]
        T_sensor_world_e_quat = self.__rotmat_to_unitquat(T_sensor_world_e[None, :3, :3]) # [1, 4]

        t = self.__get_interpolation_timestamp(image_points)
                   
        rot_rs = self.__unitquat_to_rotmat(self.__unitquat_slerp(T_sensor_world_s_quat.repeat(t.shape[0], 1), 
                                           T_sensor_world_e_quat.repeat(t.shape[0], 1), t)).squeeze() #[n_image_points, 3, 3]

        trans_rs = (1-t)[..., None] * T_sensor_world_s_quat[:3, 3:4].transpose(0,1).repeat(t.shape[0],1) + \
                                                 t[..., None] * T_sensor_world_e_quat[:3, 3:4].transpose(0,1).repeat(t.shape[0],1)

        cam_points_rs = (torch.bmm(rot_rs, camera_rays[:, :, None]) + trans_rs[...,None]).squeeze(-1)

        # Copy the values in the output variable
        world_rays[:,0:3] = trans_rs
        world_rays[:,3:] = cam_points_rs

        return world_rays

    def __rotmat_to_unitquat(self, R: torch.Tensor) -> torch.Tensor:
        """
        Converts a batch of rotation matrices to unit quaternion representation.

        Args:
            R: batch of rotation matrices [bs, 3, 3]

        Returns:
            batch of unit quaternions (XYZW convention)  [bs, 4]
        """

        num_rotations, D1, D2 = R.shape
        assert((D1, D2) == (3,3)), "Input has to be a Bx3x3 tensor."

        decision_matrix = torch.empty((num_rotations, 4), dtype=R.dtype).to(self.device)
        quat = torch.empty((num_rotations, 4), dtype=R.dtype).to(self.device)

        decision_matrix[:, :3] = R.diagonal(dim1=1, dim2=2)
        decision_matrix[:, -1] = decision_matrix[:, :3].sum(dim=1)
        choices = decision_matrix.argmax(dim=1)

        ind = torch.nonzero(choices != 3, as_tuple=True)[0]
        i = choices[ind]
        j = (i + 1) % 3
        k = (j + 1) % 3

        quat[ind, i] = 1 - decision_matrix[ind, -1] + 2 * R[ind, i, i]
        quat[ind, j] = R[ind, j, i] + R[ind, i, j]
        quat[ind, k] = R[ind, k, i] + R[ind, i, k]
        quat[ind, 3] = R[ind, k, j] - R[ind, j, k]

        ind = torch.nonzero(choices == 3, as_tuple=True)[0]
        quat[ind, 0] = R[ind, 2, 1] - R[ind, 1, 2]
        quat[ind, 1] = R[ind, 0, 2] - R[ind, 2, 0]
        quat[ind, 2] = R[ind, 1, 0] - R[ind, 0, 1]
        quat[ind, 3] = 1 + decision_matrix[ind, -1]

        quat = quat / torch.norm(quat, dim=1)[:, None]

        return quat
        
    def __unitquat_to_rotmat(self, quat: torch.Tensor) -> torch.Tensor:
        """
        Converts a batch of unit quaternions into a SO3 representation.
        Args:
            quat: batch of unit quaternions (XYZW convention) [bs, 4]

        Returns:
            batch of SO3 rotation matrices [bs, 3, 3]
        """

        x = quat[..., 0]
        y = quat[..., 1]
        z = quat[..., 2]
        w = quat[..., 3]

        R = torch.empty(quat.shape[:-1] + (3, 3), dtype=quat.dtype).to(self.device)

        R[..., 0, 0] = torch.pow(x,2) - torch.pow(y,2) - torch.pow(z,2) + torch.pow(w,2)
        R[..., 1, 0] = 2 * (x * y + z * w)
        R[..., 2, 0] = 2 * (x * z - y * w)

        R[..., 0, 1] = 2 * (x * y - z * w)
        R[..., 1, 1] = - torch.pow(x,2) + torch.pow(y,2) - torch.pow(z,2) + torch.pow(w,2)
        R[..., 2, 1] = 2 * (y * z + x * w)

        R[..., 0, 2] = 2 * (x * z + y * w)
        R[..., 1, 2] = 2 * (y * z - x * w)
        R[..., 2, 2] = - torch.pow(x,2) - torch.pow(y,2) + torch.pow(z,2) + torch.pow(w,2)

        return R

    def __unitquat_slerp(self, quat_s: torch.Tensor, quat_e: torch.Tensor, 
                         t: torch.Tensor, shortest_arc=True) -> torch.Tensor:
        """
        Batch-wise implementation of SLERP (spherical linear interpolation)

        Args: 
            quat_s: batch of unit quaternions denoting the start rotation [bs, 4]
            quat_e: batch of unit quaternions denoting the end rotation  [bs, 4]
            t: interpolation steps within 0.0 and 1.0, 0.0 corresponding to q0 and 1.0 to q1 [bs, 1]
            shortest_arc: if True, interpolation will be performed along the shortest arc on SO(3)
        Returns: 
            batch of interpolated quaternions [bs, 4]
        """

        assert quat_s.shape == quat_e.shape, "Input quaternions must be of the same shape."

        if len(quat_s.shape) == 1:
            quat_s = torch.unsqueeze(quat_s, 0)
            quat_e = torch.unsqueeze(quat_e, 0)
        
        # omega is the 'angle' between both quaternions
        cos_omega = torch.sum(quat_s * quat_e, dim=-1)

        if shortest_arc:
            # Flip quaternions with negative angle to perform shortest arc interpolation.
            quat_e = quat_e.clone()
            quat_e[cos_omega < 0,:] *= -1
            cos_omega = torch.abs(cos_omega)

        # True when q0 and q1 are close.
        nearby_quaternions = cos_omega > (1.0 - 1e-3)

        # General approach    
        omega = torch.acos(cos_omega)
        alpha = torch.sin((1-t)*omega)
        
        beta = torch.sin(t*omega)
        # Use linear interpolation for nearby quaternions
        alpha[nearby_quaternions] = (1 - t)[nearby_quaternions]
        beta[nearby_quaternions] = t[nearby_quaternions]

        # Interpolation
        quat = (alpha.reshape(-1,1) * quat_s + beta.reshape(-1,1) * quat_e)
        quat /= torch.norm(quat, dim=-1, keepdim=True)

        return quat

    def __get_interpolation_timestamp(self, points: torch.Tensor) -> torch.Tensor:
        ''' Get interpolation timestamp based on the pixel coordinates and rolling shutter type '''

        if self.shutter_type == "ROLLING_TOP_TO_BOTTOM":
            t = torch.floor(points[:,1]) / (self.resolution[1] - 1)
        elif self.shutter_type == "ROLLING_LEFT_TO_RIGHT":
            t = torch.floor(points[:,0]) / (self.resolution[0] - 1)
        elif self.shutter_type == "ROLLING_BOTTOM_TO_TOP":
            t = (self.resolution[1] - torch.ceil(points[:,1])) / (self.resolution[1] - 1)
        elif self.shutter_type == "ROLLING_RIGHT_TO_LEFT":
            t = (self.resolution[0] - torch.ceil(points[:,0])) / (self.resolution[0] - 1)

        return t

class FThetaCameraModel(CameraModel):
    def __init__(self, camera_model_parameters: types.FThetaCameraModelParameters, 
                 device: str = 'cuda', dtype: torch.dtype = torch.float32):
        
        # Check if cuda device is actually available
        if device == 'cuda' and not torch.cuda.is_available():
            logging.warning("Cuda device selected but not available, reverting to CPU!")
            device = 'cpu'

        self.device = device
        self.dtype = dtype

        self.principal_point = self.to_torch(camera_model_parameters.principal_point).to(self.dtype)
        self.fw_poly = self.to_torch(camera_model_parameters.fw_poly).to(self.dtype)
        self.bw_poly = self.to_torch(camera_model_parameters.bw_poly).to(self.dtype)
        self.resolution = self.to_torch(camera_model_parameters.resolution.astype(np.int32))
        self.shutter_type = camera_model_parameters.shutter_type.name
        self.max_angle = float(camera_model_parameters.max_angle)

        assert self.principal_point.shape == (2,)
        assert self.principal_point.dtype == self.dtype
        assert self.fw_poly.shape == (6,)
        assert self.fw_poly.dtype == self.dtype
        assert self.bw_poly.shape == (6,)
        assert self.bw_poly.dtype == self.dtype
        assert self.resolution.shape == (2,)
        assert self.resolution.dtype == torch.int32

    def pixel_to_camera_ray(self, image_points: Union[torch.Tensor, np.ndarray]) ->  torch.Tensor:
        '''
        Computes the camera ray for each image point
        '''
        
        image_points = self.to_torch(image_points).to(self.dtype)

        pixels_dist = image_points - self.principal_point[:, None]
        rdist = torch.linalg.norm(pixels_dist, axis=1, keepdims=True)

        alphas = self.__eval_bw_poly(rdist)
        min_norm = torch.Tensor(1e-6).to(pixels_dist)

        # Compute the camera rays and set the ones at the image center to [0,0,1]
        cam_rays = torch.hstack((torch.sin(alphas) * pixels_dist / torch.maximum(rdist, min_norm), torch.cos(alphas)))
        cam_rays[rdist < min_norm] = torch.Tensor([[0,0,1]]).to(pixels_dist)

        return cam_rays
    
    def camera_ray_to_pixel(self, cam_rays: Union[torch.Tensor, np.ndarray]) ->  Tuple[torch.Tensor, torch.Tensor]:
        '''
        For each camera ray it computes the corresponding pixel coordinates
        '''
        
        # If the input is a numpy array first convert it to torch otherwise just send to correct device
        cam_rays = self.to_torch(cam_rays).to(self.dtype)

        ray_norm = self.__nummerically_stable_norm(cam_rays)
        alphas = torch.atan2(torch.linalg.norm(cam_rays[:, :2], axis=1, keepdims=True), cam_rays[:, 2:])
        delta = self.__eval_fw_poly(alphas)

        # Replace the invalid angles and prevent division by 0
        delta[ray_norm <= 0.0] = 0.0
        ray_norm[ray_norm <= 0.0] = 1.0

        scale = delta / ray_norm
        image_points = scale * cam_rays[:,:2] + self.principal_point[None, :]

        # Extract valid points
        valid_x = torch.logical_and(0.0 <= image_points[:,0], image_points[:,0] < self.resolution[0])
        valid_y = torch.logical_and(0.0 <= image_points[:,1], image_points[:,1] < self.resolution[1])
        valid_delta = alphas[:,0] < self.max_angle
        valid = valid_x & valid_y & valid_delta

        # If the input was numpy, return numpy arrays as well
        return image_points, valid

    def __eval_bw_poly(self, pixel_norms: torch.Tensor) -> torch.Tensor:
        ''' Evaluate the backward polynomial using Horner scheme '''
        val = torch.zeros_like(pixel_norms)
        for it in torch.flip(self.bw_poly, dims=(0,)):
            val = val * pixel_norms + it

        return val

    def __eval_fw_poly(self, theta: torch.Tensor) -> torch.Tensor:
        ''' Evaluate the forward polynomial using Horner scheme '''
        val = torch.zeros_like(theta)
        for it in torch.flip(self.fw_poly, dims=(0,)):
            val = val * theta + it

        return val

    def __nummerically_stable_norm(self, cam_rays: torch.Tensor) -> torch.Tensor:
        ''' Evaluate the norm in a numarically stable manner '''
        xy_norms = torch.zeros_like(cam_rays[:,0]).unsqueeze(1)

        abs_pts = torch.abs(cam_rays[:,:2])
        min_pts = torch.min(abs_pts, dim = 1, keepdim=True).values
        max_pts = torch.max(abs_pts, dim = 1, keepdim=True).values

        # Set the norm of zero points to zero
        xy_norms[max_pts <= 0.0, None] = 0.0

        min_max_ratio = min_pts / max_pts
        xy_norms[max_pts > 0.0, None] = max_pts * torch.sqrt(1.0 + torch.pow(min_max_ratio, 2))

        return xy_norms


class PinholeCameraModel(CameraModel):
    def __init__(self, camera_model_parameters: types.PinholeCameraModelParameters, 
                 device: str = 'cuda', dtype: torch.dtype = torch.float32):
        
        # Check if cuda device is actually available
        if device == 'cuda' and not torch.cuda.is_available():
            logging.warning("Cuda device selected but not available, reverting to CPU!")
            device = 'cpu'

        self.device = device
        self.dtype = dtype
        self.principal_point = self.to_torch(camera_model_parameters.principal_point).to(self.dtype)
        self.focal_length = self.to_torch(camera_model_parameters.focal_length).to(self.dtype)
        self.radial_poly = self.to_torch(camera_model_parameters.radial_poly[:3]).to(self.dtype)
        self.tangential_poly = self.to_torch(camera_model_parameters.tangential_poly).to(self.dtype)
        self.resolution = self.to_torch(camera_model_parameters.resolution.astype(np.int32))
        self.shutter_type = camera_model_parameters.shutter_type.name

        assert self.principal_point.shape == (2,)
        assert self.principal_point.dtype == self.dtype
        assert self.focal_length.shape == (2,)
        assert self.focal_length.dtype == self.dtype
        assert self.radial_poly.shape == (3,)
        assert self.radial_poly.dtype == self.dtype
        assert self.tangential_poly.shape == (2,)
        assert self.tangential_poly.dtype == self.dtype
        assert self.resolution.shape == (2,)
        assert self.resolution.dtype == torch.int32

    def pixel_to_camera_ray(self, image_points: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        '''
        Computes the camera ray for each image point
        '''
        image_points = self.to_torch(image_points).to(self.dtype)
        camera_rays = self.__iterative_undistort(image_points)
        cam_rays = torch.cat([camera_rays, torch.ones_like(camera_rays[:,0:1])], dim=1)

        return cam_rays
        
    def camera_ray_to_pixel(self, cam_rays: Union[torch.Tensor, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        For each camera ray it computes the corresponding pixel coordinates
        '''

        cam_rays = self.to_torch(cam_rays).to(self.dtype)

        # Initialize the valid flag and set all the points behind the camera plane to invalid
        valid = torch.ones_like(cam_rays[:,0], dtype=torch.bool)
        image_points = torch.zeros_like(cam_rays[:,:2])

        valid[cam_rays[:,2] < 0.0] = False 
        valid_idx = torch.where(valid)[0]

        uv_normalized = cam_rays[valid, :2] / cam_rays[valid, 2:3] # [n,2]
        icD, delta_x, delta_y = self.__compute_distortion(uv_normalized)

        k_min_radial_dist = 0.8
        k_max_radial_dist = 1.2

        valid_radial = torch.logical_and(icD > k_min_radial_dist, icD < k_max_radial_dist)

        # Apply tangential distortion
        uND = uv_normalized[:,0] * icD + delta_x # [n]
        vND = uv_normalized[:,1] * icD + delta_y # [n]

        image_points[valid_idx,0] = uND * self.focal_length[0] + self.principal_point[0]
        image_points[valid_idx,1] = vND * self.focal_length[1] + self.principal_point[1]

        # Check if the point falls within the image
        valid_x = torch.logical_and(0.0 <= image_points[valid_idx,0], image_points[valid_idx,0] < self.resolution[0])
        valid_y = torch.logical_and(0.0 <= image_points[valid_idx,1], image_points[valid_idx,1] < self.resolution[1])

        # Set the points that have too large distortion or fall outside the image sensor to invalid
        valid_pts = valid_x & valid_y & valid_radial
        valid[valid_idx[~valid_pts]] = False

        return image_points, valid

    def __compute_distortion(self, xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ''' Computes the radial and tangential distortion given the camera rays '''
 
        # Compute the helper variables
        xy_squared = torch.pow(xy, 2)
        xy_prod = xy[:,0] * xy[:,1]
        r_2 = torch.sum(xy_squared, dim=1)
        r_4 = r_2 * r_2
        r_6 = r_4 * r_2
        a1 = 2*xy_prod
        a2 = r_2 + 2*xy_squared[:,0]
        a3 = r_2 + 2*xy_squared[:,1]

        icD_numerator = (1.0 + self.radial_poly[0] * r_2 + self.radial_poly[1] * r_4 + self.radial_poly[2] * r_6)
        icD_denominator = (1.0 + self.radial_poly[3] * r_2 + self.radial_poly[4] * r_4 + self.radial_poly[5] * r_6)
        icD = icD_numerator/icD_denominator

        delta_x = self.tangential_poly[0] * a1 + self.tangential_poly[1] * a2 + self.tangential_poly[2] * r_2 + self.tangential_poly[3] * r_4
        delta_y = self.tangential_poly[0] * a3 + self.tangential_poly[1] * a1 + self.tangential_poly[4] * r_2 + self.tangential_poly[5] * r_4
 
        return icD, delta_x, delta_y

    def __iterative_undistort(self, image_points: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        cam_rays = (image_points - self.principal_point[:,None]) / self.focal_length[:,None]
        cam_rays_0 = torch.clone(cam_rays)

        c_iter = 0
        max_iter = 20
        error = 1e12
        while (error > eps and c_iter < max_iter):
            icD, delta_x, delta_y = self.__compute_distortion(cam_rays)
            
            # Get the previous values to compute the residual
            cam_rays_prev = torch.clone(cam_rays)
            cam_rays = (cam_rays_0 - torch.cat([delta_x[:, None], delta_y[:, None]], dim=1)) * icD
    
            error = torch.mean(torch.square(cam_rays - cam_rays_prev)).item()

        return cam_rays
