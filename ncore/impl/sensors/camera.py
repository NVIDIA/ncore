# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import logging
import math

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, cast
from dataclasses import dataclass

import torch
import numpy as np

from ncore.impl.data import types
from ncore.impl.sensors.common import (
    to_torch,
    rotmat_to_unitquat,
    unitquat_to_rotmat,
    unitquat_slerp,
    eval_poly_horner,
    eval_poly_inverse_horner_newton,
)


class ExternalDistortion(ABC):
    """Base class for distortion effects from external causes to the camera"""

    device: torch.device  #: Torch device to perform computations on
    dtype: torch.dtype  #: Torch floating-point datatype to perform computations in

    def __init__(
        self,
        device: Union[str, torch.device],
        dtype: torch.dtype,
    ):
        # Make sure device is a torch device
        if isinstance(device, str):
            device = torch.device(device)

        # Check if cuda device is actually available
        if device.type == "cuda" and not torch.cuda.is_available():
            logging.warning("Cuda device selected but not available, reverting to CPU!")
            device = torch.device("cpu")

        self.device = device
        self.dtype = dtype

    @staticmethod
    def maybe_from_parameters(
        external_distortion_parameters: Optional[types.ConcreteExternalDistortionParametersUnion],
        device: Union[str, torch.device] = torch.device("cuda"),
        dtype: torch.dtype = torch.float32,
    ) -> Optional[ExternalDistortion]:
        """
        Initialize a generic external distortion model from parameters, if available
        """
        if external_distortion_parameters is None:
            return None
        if isinstance(external_distortion_parameters, types.BivariateWindshieldModelParameters):
            return BivariateWindshieldModel(external_distortion_parameters, device, dtype)
        raise TypeError(
            f"Unsupported external distortion type {type(external_distortion_parameters)}, currently only supporting 'BivariateWindshieldModel' type."
        )

    @abstractmethod
    def distort_camera_rays(self, camera_rays: torch.Tensor) -> torch.Tensor:
        """
        Applies distortion to camera rays in forward direction, from external to internal
        """
        pass

    @abstractmethod
    def undistort_camera_rays(self, camera_rays: torch.Tensor) -> torch.Tensor:
        """
        Applies distortion to camera rays in backward direction, from internal to external
        """
        pass


class BivariateWindshieldModel(ExternalDistortion):
    """Implements an external distortion caused by a vehicle's windshield. The model is only applicable for cameras where the whole area of interest is projected through the windshield.

    The distortion is computed on spherical phi/theta angle-based representations of a sensor ray with direction=[x,y,z] such that phi = asin(x/(x^2+y+2+z^2)) and theta = asin(y/(x^2+y^2+z^2)).

    Phi and theta are then deflected via distortion polynomials before the transformed ray is re-constructed with direction [sin(phi'), sin(theta'), 1-(x^2+y^2)]
    The distortion on phi and theta are computed via separate polynomials of order N in both phi and theta, e.g. phi' = c0 + c1*phi + c2*phi^2 + (c3 + c4*phi)*theta + c5*theta^2"""

    # Forward correction coefficients (transforms external to internal)
    horizontal_poly: torch.Tensor  #: Polynomial used for horizontal component of distortion in forward direction
    vertical_poly: torch.Tensor  #: Polynomial used for vertical component of distortion in forward direction

    # Backward correction coefficient (transforms internal to external)
    horizontal_poly_inverse: (
        torch.Tensor
    )  #: Polynomial used for horizontal component of distortion in backward direction
    vertical_poly_inverse: torch.Tensor  #: Polynomial used for vertical component of distortion in backward direction

    order_phi: int  #:  Order of the distortion polynomial on phi
    order_theta: int  #:  Order of the distortion polynomial on theta

    def __init__(
        self,
        windshield_distortion_parameters: types.BivariateWindshieldModelParameters,
        device: Union[str, torch.device] = torch.device("cuda"),
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(device, dtype)

        self.horizontal_poly = to_torch(
            windshield_distortion_parameters.horizontal_poly, device=self.device, dtype=dtype
        )
        self.vertical_poly = to_torch(windshield_distortion_parameters.vertical_poly, device=self.device, dtype=dtype)
        self.horizontal_poly_inverse = to_torch(
            windshield_distortion_parameters.horizontal_poly_inverse, device=self.device, dtype=dtype
        )
        self.vertical_poly_inverse = to_torch(
            windshield_distortion_parameters.vertical_poly_inverse, device=self.device, dtype=dtype
        )

        # Compute the order of the polynomial
        self.order_phi = self.compute_poly_order(self.horizontal_poly)
        self.order_theta = self.compute_poly_order(self.vertical_poly)

    @staticmethod
    def compute_poly_order(poly_coeffs: torch.Tensor):
        """Computes the order of a bivariate polynomial give it's array of coefficients"""
        order = 0
        num_terms = 0
        for order_candidate in range(torch.numel(poly_coeffs)):
            num_terms += order_candidate + 1
            if num_terms == torch.numel(poly_coeffs):
                order = order_candidate
                break
            elif num_terms > torch.numel(poly_coeffs):
                raise ValueError(
                    "The input length of the windshield distortion coefficients is not consistent with the assumed polynomial form."
                )
        return order

    @staticmethod
    def poly_eval_2d(coefficients: torch.Tensor, x: torch.Tensor, y: torch.Tensor, order: int) -> torch.Tensor:
        """
        The bivariate polynomial, provided as a single-dimension tensor [c0, c1, c2...cn] is evaluated as:
        c0*x^0 +c1*x^1 + c2*x^2 + (c3*x^0 + c4*x^1)y^1 + (c5*x^0)y^2
        In essence, each coefficient to y is a polynomial evaluation of increasing degree.
        """
        if x.shape != y.shape:
            raise ValueError("Expected x and y tensors to be of the same size, but got {x.shape} and {y.shape}")
        x_flat = x.view(-1)
        y_flat = y.view(-1)
        y_coeffs = torch.zeros(order + 1, x_flat.shape[0], dtype=x.dtype, device=x.device)
        start_idx = 0
        for inner_order in reversed(range(order + 1)):
            x_coeffs = coefficients[start_idx : start_idx + inner_order + 1]
            y_coeffs[order - inner_order, :] = eval_poly_horner(x_coeffs, x_flat)
            start_idx += inner_order + 1
        z = eval_poly_horner(y_coeffs, y_flat)
        return z.reshape(x.shape)

    @staticmethod
    def distort_rays(
        camera_rays: torch.Tensor,
        phi_poly: torch.Tensor,
        theta_poly: torch.Tensor,
        order_phi: int,
        order_theta: int,
        poly2d_eval_func,
    ) -> torch.Tensor:
        """
        Applies distortion to rays in forward direction, from external to internal
        """
        normalized_rays = torch.nn.functional.normalize(camera_rays, dim=-1)
        phi = torch.asin(normalized_rays[..., 0])
        theta = torch.asin(normalized_rays[..., 1])
        adj_phi = poly2d_eval_func(phi_poly, phi, theta, order_phi)
        adj_theta = poly2d_eval_func(theta_poly, phi, theta, order_theta)
        x = torch.sin(adj_phi)
        y = torch.sin(adj_theta)
        xy_norm = x * x + y * y
        z = torch.sqrt(torch.clamp(torch.ones_like(x) - xy_norm, 0.0, 1.0)) * torch.sign(normalized_rays[..., 2])
        return torch.stack((x, y, z), -1)

    def distort_camera_rays(self, camera_rays: torch.Tensor) -> torch.Tensor:
        """
        Applies distortion to camera rays in forward direction, from external to internal
        """
        return self.distort_rays(
            camera_rays,
            self.horizontal_poly,
            self.vertical_poly,
            self.order_phi,
            self.order_theta,
            self.poly_eval_2d,
        )

    def undistort_camera_rays(self, camera_rays: torch.Tensor) -> torch.Tensor:
        """
        Applies distortion to camera rays in backward direction, from external to internal
        """
        return self.distort_rays(
            camera_rays,
            self.horizontal_poly_inverse,
            self.vertical_poly_inverse,
            self.order_phi,
            self.order_theta,
            self.poly_eval_2d,
        )


class CameraModel(ABC):
    """Base camera model class"""

    resolution: torch.Tensor  #: Width and height of the image in pixels (uint32, [2,])
    shutter_type: types.ShutterType  #: Shutter type of the camera's imaging sensor
    device: torch.device  #: Torch device to perform computations on
    dtype: torch.dtype  #: Torch floating-point datatype to perform computations in
    external_distortion: Optional[
        ExternalDistortion
    ]  #: Source of distortion external to the camera (e.g. windshield). Can be left empty (None) if no such source exists. If a source exits, rays will be distorted prior to reaching the camera and it's associated lens distortion if applicable

    def __init__(
        self, camera_model_parameters: types.CameraModelParameters, device: Union[str, torch.device], dtype: torch.dtype
    ):
        # Make sure device is a torch device
        if isinstance(device, str):
            device = torch.device(device)

        # Check if cuda device is actually available
        if device.type == "cuda" and not torch.cuda.is_available():
            logging.warning("Cuda device selected but not available, reverting to CPU!")
            device = torch.device("cpu")

        self.device = device
        self.dtype = dtype

        self.resolution = to_torch(camera_model_parameters.resolution.astype(np.int32), device=self.device)
        self.shutter_type = camera_model_parameters.shutter_type

        # Initialize external distortion if available
        self.external_distortion = ExternalDistortion.maybe_from_parameters(
            camera_model_parameters.external_distortion_parameters, self.device, self.dtype
        )

    @abstractmethod
    def _image_points_to_camera_rays_impl(self, image_points: torch.Tensor) -> torch.Tensor:
        """
        Camera model-specific implementation of image_points_to_camera_rays
        """
        pass

    @abstractmethod
    def _camera_rays_to_image_points_impl(
        self, cam_rays: torch.Tensor, return_jacobians: bool
    ) -> CameraModel.ImagePointsReturn:
        """
        Camera model-specific implementation of camera_rays_to_image_points
        """
        pass

    def image_points_to_camera_rays(self, image_points: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Computes camera rays for each image point
        """

        # If the input is a numpy array first convert it to torch otherwise just send to correct device
        image_points = to_torch(image_points, device=self.device)

        # Make sure users don't accidentally pass pixel coordinates (integer indices)
        assert image_points.is_floating_point(), "[CameraModel]: image_points must be floating point values"

        image_points = image_points.to(self.dtype)

        # Evaluate regular lens model
        cam_rays = self._image_points_to_camera_rays_impl(image_points)

        # Apply external distortion if available
        if self.external_distortion is not None:
            cam_rays = self.external_distortion.undistort_camera_rays(cam_rays)

        return cam_rays

    def camera_rays_to_image_points(
        self, cam_rays: Union[torch.Tensor, np.ndarray], return_jacobians: bool = False
    ) -> CameraModel.ImagePointsReturn:
        """
        For each camera ray, computes the corresponding image point coordinates and a valid flag.
        Optionally, the Jacobians of the per-ray transformations can be computed as well
        """

        # If the input is a numpy array first convert it to torch otherwise just send to correct device
        cam_rays = to_torch(cam_rays, dtype=self.dtype, device=self.device)

        # Apply external distortion if available
        if self.external_distortion is not None:
            cam_rays = self.external_distortion.distort_camera_rays(cam_rays)

        # Evaluate regular lens model
        return self._camera_rays_to_image_points_impl(cam_rays, return_jacobians)

    def pixels_to_camera_rays(self, pixel_idxs: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        For each pixel index computes its corresponding camera ray
        """

        return self.image_points_to_camera_rays(self.pixels_to_image_points(pixel_idxs))

    def camera_rays_to_pixels(self, cam_rays: Union[torch.Tensor, np.ndarray]) -> CameraModel.PixelsReturn:
        """
        For each camera ray, computes the corresponding pixel index and a valid flag
        """
        image_points = self.camera_rays_to_image_points(cam_rays)

        return CameraModel.PixelsReturn(
            pixels=self.image_points_to_pixels(image_points.image_points), valid_flag=image_points.valid_flag
        )

    @staticmethod
    def from_parameters(
        cam_model_parameters: types.ConcreteCameraModelParametersUnion,
        device: Union[str, torch.device] = torch.device("cuda"),
        dtype: torch.dtype = torch.float32,
    ) -> CameraModel:
        """
        Initialize a generic camera model class from camera model parameters
        """
        if isinstance(cam_model_parameters, types.FThetaCameraModelParameters):
            return FThetaCameraModel(cam_model_parameters, device, dtype)
        elif isinstance(cam_model_parameters, types.OpenCVPinholeCameraModelParameters):
            return OpenCVPinholeCameraModel(cam_model_parameters, device, dtype)
        elif isinstance(cam_model_parameters, types.OpenCVFisheyeCameraModelParameters):
            return OpenCVFisheyeCameraModel(cam_model_parameters, device, dtype)
        else:
            raise TypeError(
                f"unsupported camera model type {type(cam_model_parameters)}, currently supporting Ftheta/OpenCV-Pinhole/OpenCV-Fisheye only"
            )

    @dataclass
    class WorldPointsToPixelsReturn:
        """
        Contains
            - pixel indices of the valid projections [int] (n,2)
            - [optional] world-to-sensor poses of valid projections [float] (n,4,4)
            - [optional] indices of the valid projections relative to the input points [int] (n,)
            - [optional] timestamps of the valid projections [int] (n,)
        """

        pixels: torch.Tensor
        T_world_sensors: Optional[torch.Tensor] = None
        valid_indices: Optional[torch.Tensor] = None
        timestamps_us: Optional[torch.Tensor] = None

    @dataclass
    class WorldPointsToImagePointsReturn:
        """
        Contains
            - image point coordinates of the valid projections [float] (n,2)
            - [optional] world-to-sensor poses of valid projections [float] (n,4,4)
            - [optional] indices of the valid projections relative to the input points [int] (n,)
            - [optional] timestamps of the valid projections [int] (n,)
        """

        image_points: torch.Tensor
        T_world_sensors: Optional[torch.Tensor] = None
        valid_indices: Optional[torch.Tensor] = None
        timestamps_us: Optional[torch.Tensor] = None

    @dataclass
    class WorldRaysReturn:
        """
        Contains
            - rays [point, direction] in the world coordinate frame, represented by 3d start of ray points and 3d ray directions [float] (n,6)
            - [optional] sensor-to-worlds poses of the returned rays [float] (n,4,4)
            - [optional] timestamps of the returned rays [int] (n,)
        """

        world_rays: torch.Tensor
        T_sensor_worlds: Optional[torch.Tensor] = None
        timestamps_us: Optional[torch.Tensor] = None

    @dataclass
    class ImagePointsReturn:
        """
        Contains
            - image point coordinates [float] (n,2)
            - valid_flag [bool] (n,)
            - [optional] Jacobians of the projection [float] (n,2,3)
        """

        image_points: torch.Tensor
        valid_flag: torch.Tensor
        jacobians: Optional[torch.Tensor] = None

    @dataclass
    class PixelsReturn:
        """
        Contains
            - pixel indices [int] (n,2)
            - valid_flag [bool] (n,)
        """

        pixels: torch.Tensor
        valid_flag: torch.Tensor

    def world_points_to_pixels_shutter_pose(
        self,
        world_points: Union[torch.Tensor, np.ndarray],
        T_world_sensor_start: Union[torch.Tensor, np.ndarray],
        T_world_sensor_end: Union[torch.Tensor, np.ndarray],
        start_timestamp_us: Optional[int] = None,
        end_timestamp_us: Optional[int] = None,
        max_iterations: int = 10,
        stop_mean_error_px: float = 1e-3,
        stop_delta_mean_error_px: float = 1e-5,
        return_T_world_sensors: bool = False,
        return_valid_indices: bool = False,
        return_timestamps: bool = False,
        return_all_projections: bool = False,
    ) -> CameraModel.WorldPointsToPixelsReturn:
        """Projects world points to corresponding pixel indices using *rolling-shutter compensation* of sensor motion"""

        if return_timestamps:
            assert start_timestamp_us is not None
            assert end_timestamp_us is not None
            assert end_timestamp_us >= start_timestamp_us, (
                "[CameraModel]: End timestamp must be larger or equal to the start timestamp"
            )

        tmp = self.world_points_to_image_points_shutter_pose(
            world_points,
            T_world_sensor_start,
            T_world_sensor_end,
            start_timestamp_us,
            end_timestamp_us,
            max_iterations,
            stop_mean_error_px,
            stop_delta_mean_error_px,
            return_T_world_sensors,
            return_valid_indices,
            return_timestamps,
            return_all_projections,
        )

        return self.WorldPointsToPixelsReturn(
            pixels=self.image_points_to_pixels(tmp.image_points),
            T_world_sensors=tmp.T_world_sensors,
            valid_indices=tmp.valid_indices,
            timestamps_us=tmp.timestamps_us,
        )

    def world_points_to_pixels_static_pose(
        self,
        world_points: Union[torch.Tensor, np.ndarray],
        T_world_sensor: Union[torch.Tensor, np.ndarray],
        timestamp_us: Optional[int] = None,
        return_T_world_sensors: bool = False,
        return_valid_indices: bool = False,
        return_timestamps: bool = False,
        return_all_projections: bool = False,
    ) -> CameraModel.WorldPointsToPixelsReturn:
        """Projects world points to corresponding pixel indices using a *fixed* sensor pose (not compensating for potential sensor-motion)."""

        if return_timestamps:
            assert timestamp_us is not None

        tmp = self.world_points_to_image_points_static_pose(
            world_points,
            T_world_sensor,
            timestamp_us,
            return_T_world_sensors,
            return_valid_indices,
            return_timestamps,
            return_all_projections,
        )

        return self.WorldPointsToPixelsReturn(
            pixels=self.image_points_to_pixels(tmp.image_points),
            T_world_sensors=tmp.T_world_sensors,
            valid_indices=tmp.valid_indices,
            timestamps_us=tmp.timestamps_us,
        )

    def world_points_to_pixels_mean_pose(
        self,
        world_points: Union[torch.Tensor, np.ndarray],
        T_world_sensor_start: Union[torch.Tensor, np.ndarray],
        T_world_sensor_end: Union[torch.Tensor, np.ndarray],
        start_timestamp_us: Optional[int] = None,
        end_timestamp_us: Optional[int] = None,
        return_T_world_sensors: bool = False,
        return_valid_indices: bool = False,
        return_timestamps: bool = False,
        return_all_projections: bool = False,
    ) -> CameraModel.WorldPointsToPixelsReturn:
        """Projects world points to corresponding pixel indices using the *mean pose* of the sensor between
        the start and end poses (not compensating for potential sensor-motion).
        """

        if return_timestamps:
            assert start_timestamp_us is not None
            assert end_timestamp_us is not None
            assert end_timestamp_us >= start_timestamp_us, (
                "[CameraModel]: End timestamp must be larger or equal to the start timestamp"
            )

            timestamp_us = (end_timestamp_us + start_timestamp_us) // 2

        else:
            timestamp_us = None

        return self.world_points_to_pixels_static_pose(
            world_points,
            self.__interpolate_poses(
                to_torch(T_world_sensor_start, device=self.device, dtype=self.dtype),
                to_torch(T_world_sensor_end, device=self.device, dtype=self.dtype),
                0.5,
            ),
            timestamp_us,
            return_T_world_sensors,
            return_valid_indices,
            return_timestamps,
            return_all_projections,
        )

    def world_points_to_image_points_shutter_pose(
        self,
        world_points: Union[torch.Tensor, np.ndarray],
        T_world_sensor_start: Union[torch.Tensor, np.ndarray],
        T_world_sensor_end: Union[torch.Tensor, np.ndarray],
        start_timestamp_us: Optional[int] = None,
        end_timestamp_us: Optional[int] = None,
        max_iterations: int = 10,
        stop_mean_error_px: float = 1e-3,
        stop_delta_mean_error_px: float = 1e-5,
        return_T_world_sensors: bool = False,
        return_valid_indices: bool = False,
        return_timestamps: bool = False,
        return_all_projections: bool = False,
    ) -> CameraModel.WorldPointsToImagePointsReturn:
        """Projects world points to corresponding image point coordinates using *rolling-shutter compensation* of sensor motion"""

        # Check if the variables are numpy, convert them to torch and send them to correct device
        world_points = to_torch(world_points, device=self.device, dtype=self.dtype)
        T_world_sensor_start = to_torch(T_world_sensor_start, device=self.device, dtype=self.dtype)
        T_world_sensor_end = to_torch(T_world_sensor_end, device=self.device, dtype=self.dtype)

        assert T_world_sensor_start.shape == (4, 4)
        assert T_world_sensor_end.shape == (4, 4)
        assert len(world_points.shape) == 2
        assert world_points.shape[1] == 3
        assert world_points.dtype == self.dtype
        assert T_world_sensor_start.dtype == self.dtype
        assert T_world_sensor_end.dtype == self.dtype
        assert isinstance(max_iterations, int)
        assert max_iterations > 0

        if return_timestamps:
            assert start_timestamp_us is not None
            assert end_timestamp_us is not None
            assert end_timestamp_us >= start_timestamp_us, (
                "[CameraModel]: End timestamp must be larger or equal to the start timestamp"
            )

            # Make sure timestamps have correct type (might be, e.g., np.uint64, which torch doesn't like)
            start_timestamp_us = int(start_timestamp_us)
            end_timestamp_us = int(end_timestamp_us)

        # Always perform transformation using start pose
        image_points_start = self.camera_rays_to_image_points(
            (T_world_sensor_start[:3, :3] @ world_points.transpose(0, 1) + T_world_sensor_start[:3, 3, None]).transpose(
                0, 1
            )
        )

        # Global-shutter special case - no need for rolling-shutter compensation, use projections from start-pose as single available pose
        if self.shutter_type == types.ShutterType.GLOBAL:
            return_var = self.WorldPointsToImagePointsReturn(
                image_points=(
                    image_points_start.image_points[image_points_start.valid_flag]
                    if not return_all_projections
                    else image_points_start.image_points
                )
            )
            if return_T_world_sensors:
                return_var.T_world_sensors = torch.tile(
                    T_world_sensor_start, dims=(int(image_points_start.valid_flag.sum().item()), 1, 1)
                )
            if return_valid_indices:
                return_var.valid_indices = torch.where(image_points_start.valid_flag)[0].squeeze()
            if return_timestamps:
                return_var.timestamps_us = torch.tile(
                    torch.tensor(start_timestamp_us, device=self.device),
                    dims=(int(image_points_start.valid_flag.sum().item()),),
                )
            return return_var

        # Do initial transformations using both start and mean pose to determine all candidate points and take union of valid projections as iteration starting points
        image_points_end = self.camera_rays_to_image_points(
            (T_world_sensor_end[:3, :3] @ world_points.transpose(0, 1) + T_world_sensor_end[:3, 3, None]).transpose(
                0, 1
            )
        )

        valid = image_points_start.valid_flag | image_points_end.valid_flag  # union of valid image points
        init_image_points = image_points_end.image_points
        init_image_points[image_points_start.valid_flag] = image_points_start.image_points[
            image_points_start.valid_flag
        ]  # this prefers points at the start-of-frame pose over end-of-frame points
        # - the optimization will determine the final timestamp for each point

        # Exit early if no point projected to a valid image point
        if not valid.any():
            return_var = self.WorldPointsToImagePointsReturn(
                image_points=(
                    torch.empty((0, 2), dtype=self.dtype, device=self.device)
                    if not return_all_projections
                    else init_image_points
                )
            )
            if return_T_world_sensors:
                return_var.T_world_sensors = torch.empty((0, 4, 4), dtype=self.dtype, device=self.device)
            if return_valid_indices:
                return_var.valid_indices = torch.empty((0,), dtype=torch.int64, device=self.device)
            if return_timestamps:
                return_var.timestamps_us = torch.empty((0,), dtype=torch.int64, device=self.device)
            return return_var

        # Convert the start and end rotation matrix to quaternions for subsequent interpolations
        world_sensor_s_quat = rotmat_to_unitquat(T_world_sensor_start[None, :3, :3])  # [1, 4]
        world_sensor_e_quat = rotmat_to_unitquat(T_world_sensor_end[None, :3, :3])  # [1, 4]

        # For valid image points, compute the new timestamp and project again
        image_points_rs_prev = init_image_points[valid, :]
        mean_error_px = 1e12
        for _ in range(max_iterations):
            t = self.image_points_relative_frame_times(image_points_rs_prev)

            rot_rs = unitquat_to_rotmat(
                unitquat_slerp(world_sensor_s_quat.repeat(t.shape[0], 1), world_sensor_e_quat.repeat(t.shape[0], 1), t)
            )  # [n_valid, 3, 3]

            trans_rs = (1 - t)[..., None] * T_world_sensor_start[:3, 3:4].transpose(0, 1).repeat(t.shape[0], 1) + t[
                ..., None
            ] * T_world_sensor_end[:3, 3:4].transpose(0, 1).repeat(t.shape[0], 1)

            cam_rays_rs = (torch.bmm(rot_rs, world_points[valid, :, None]) + trans_rs[..., None]).squeeze(-1)
            image_points_rs = self.camera_rays_to_image_points(cam_rays_rs)

            # Compute mean error of projections that are still valid now and check if we are still
            # making progress relative to previous iteration
            if (
                abs(
                    mean_error_px
                    - (
                        mean_error_px := torch.linalg.norm(
                            image_points_rs.image_points[image_points_rs.valid_flag]
                            - image_points_rs_prev[image_points_rs.valid_flag],
                            dim=1,
                        ).mean()
                    )
                )
                <= stop_delta_mean_error_px
            ):
                break

            # Check if error bound was reached
            if mean_error_px <= stop_mean_error_px:
                break

            image_points_rs_prev = image_points_rs.image_points

        # We always return image points
        return_var = self.WorldPointsToImagePointsReturn(
            image_points=image_points_rs.image_points[image_points_rs.valid_flag]
        )

        if return_T_world_sensors:
            # Generate the output matrix
            trans_matrices = torch.empty(
                (int(image_points_rs.valid_flag.sum().item()), 4, 4), dtype=self.dtype, device=self.device
            )
            trans_matrices[:, :3, 3] = trans_rs[image_points_rs.valid_flag]
            trans_matrices[:, :3, :3] = rot_rs[image_points_rs.valid_flag, ...]
            trans_matrices[:, 3] = torch.tensor([0, 0, 0, 1], device=self.device, dtype=self.dtype)

            return_var.T_world_sensors = trans_matrices

        if return_valid_indices:
            # Combine validity flags
            # (valid_rs represents a strict logical subset of full valid flags, so no logical operation required)
            valid[torch.argwhere(valid).squeeze()] = image_points_rs.valid_flag
            return_var.valid_indices = torch.argwhere(valid).squeeze(1)

        if return_timestamps:
            return_var.timestamps_us = (
                start_timestamp_us
                + (
                    t[image_points_rs.valid_flag, None] * (cast(int, end_timestamp_us) - cast(int, start_timestamp_us))
                ).to(torch.int64)
            ).squeeze(1)

        if return_all_projections:
            if not return_valid_indices:
                valid[torch.argwhere(valid).squeeze()] = image_points_rs.valid_flag
                valid_indices = torch.argwhere(valid).squeeze(1)
            else:
                valid_indices = return_var.valid_indices  # type: ignore

            return_var.image_points = init_image_points
            return_var.image_points[valid_indices] = image_points_rs.image_points[image_points_rs.valid_flag]

        return return_var

    def world_points_to_image_points_static_pose(
        self,
        world_points: Union[torch.Tensor, np.ndarray],
        T_world_sensor: Union[torch.Tensor, np.ndarray],
        timestamp_us: Optional[int] = None,
        return_T_world_sensors: bool = False,
        return_valid_indices: bool = False,
        return_timestamps: bool = False,
        return_all_projections: bool = False,
    ) -> CameraModel.WorldPointsToImagePointsReturn:
        """Projects world points to corresponding image point coordinates using a *fixed* sensor pose (not compensating for potential sensor-motion)."""

        # Check if the variables are numpy, convert them to torch and send them to correct device
        world_points = to_torch(world_points, device=self.device, dtype=self.dtype)
        T_world_sensor = to_torch(T_world_sensor, device=self.device, dtype=self.dtype)

        assert T_world_sensor.shape == (4, 4)
        assert len(world_points.shape) == 2
        assert world_points.shape[1] == 3
        assert world_points.dtype == self.dtype
        assert T_world_sensor.dtype == self.dtype

        if return_timestamps:
            assert timestamp_us is not None

        R_world_sensor = T_world_sensor[:3, :3]  # [3, 3]
        t_world_sensor = T_world_sensor[:3, 3]  # [3, 1]

        # Do the transformation
        cam_rays = torch.matmul(R_world_sensor, world_points[:, :, None]).squeeze(-1) + t_world_sensor
        image_points = self.camera_rays_to_image_points(cam_rays)

        # We always return image points
        return_var = self.WorldPointsToImagePointsReturn(
            image_points=(
                image_points.image_points[image_points.valid_flag]
                if not return_all_projections
                else image_points.image_points
            )
        )

        if return_T_world_sensors:
            # Repeat static pose n-valid times
            return_var.T_world_sensors = T_world_sensor.unsqueeze(0).repeat(
                int(image_points.valid_flag.sum().item()), 1, 1
            )

        if return_valid_indices:
            return_var.valid_indices = torch.where(image_points.valid_flag)[0].squeeze()

        if return_timestamps:
            return_var.timestamps_us = torch.tile(
                torch.tensor(timestamp_us, device=self.device), dims=(len(torch.where(image_points.valid_flag)[0]),)
            )

        return return_var

    def world_points_to_image_points_mean_pose(
        self,
        world_points: Union[torch.Tensor, np.ndarray],
        T_world_sensor_start: Union[torch.Tensor, np.ndarray],
        T_world_sensor_end: Union[torch.Tensor, np.ndarray],
        start_timestamp_us: Optional[int] = None,
        end_timestamp_us: Optional[int] = None,
        return_T_world_sensors: bool = False,
        return_valid_indices: bool = False,
        return_timestamps: bool = False,
        return_all_projections: bool = False,
    ) -> CameraModel.WorldPointsToImagePointsReturn:
        """Projects world points to corresponding image point coordinates using the *mean pose* of the sensor between
        the start and end poses (not compensating for potential sensor-motion).
        """

        if return_timestamps:
            assert start_timestamp_us is not None
            assert end_timestamp_us is not None
            assert end_timestamp_us >= start_timestamp_us, (
                "[CameraModel]: End timestamp must be larger or equal to the start timestamp"
            )

            timestamp_us = (end_timestamp_us + start_timestamp_us) // 2

        else:
            timestamp_us = None

        return self.world_points_to_image_points_static_pose(
            world_points,
            self.__interpolate_poses(
                to_torch(T_world_sensor_start, device=self.device, dtype=self.dtype),
                to_torch(T_world_sensor_end, device=self.device, dtype=self.dtype),
                0.5,
            ),
            timestamp_us,
            return_T_world_sensors,
            return_valid_indices,
            return_timestamps,
            return_all_projections,
        )

    def pixels_to_world_rays_static_pose(
        self,
        pixel_idxs: Union[torch.Tensor, np.ndarray],
        T_sensor_world: Union[torch.Tensor, np.ndarray],
        timestamp_us: Optional[int] = None,
        camera_rays: Optional[Union[torch.Tensor, np.ndarray]] = None,
        return_T_sensor_worlds: bool = False,
        return_timestamps: bool = False,
    ) -> CameraModel.WorldRaysReturn:
        return self.image_points_to_world_rays_static_pose(
            self.pixels_to_image_points(pixel_idxs),
            T_sensor_world,
            timestamp_us,
            camera_rays,
            return_T_sensor_worlds,
            return_timestamps,
        )

    def pixels_to_world_rays_shutter_pose(
        self,
        pixel_idxs: Union[torch.Tensor, np.ndarray],
        T_sensor_world_start: Union[torch.Tensor, np.ndarray],
        T_sensor_world_end: Union[torch.Tensor, np.ndarray],
        start_timestamp_us: Optional[int] = None,
        end_timestamp_us: Optional[int] = None,
        camera_rays: Optional[Union[torch.Tensor, np.ndarray]] = None,
        return_T_sensor_worlds: bool = False,
        return_timestamps: bool = False,
    ) -> CameraModel.WorldRaysReturn:
        return self.image_points_to_world_rays_shutter_pose(
            self.pixels_to_image_points(pixel_idxs),
            T_sensor_world_start,
            T_sensor_world_end,
            start_timestamp_us,
            end_timestamp_us,
            camera_rays,
            return_T_sensor_worlds,
            return_timestamps,
        )

    def pixels_to_world_rays_mean_pose(
        self,
        pixel_idxs: Union[torch.Tensor, np.ndarray],
        T_sensor_world_start: Union[torch.Tensor, np.ndarray],
        T_sensor_world_end: Union[torch.Tensor, np.ndarray],
        start_timestamp_us: Optional[int] = None,
        end_timestamp_us: Optional[int] = None,
        camera_rays: Optional[Union[torch.Tensor, np.ndarray]] = None,
        return_T_sensor_worlds: bool = False,
        return_timestamps: bool = False,
    ) -> CameraModel.WorldRaysReturn:
        if return_timestamps:
            assert start_timestamp_us is not None
            assert end_timestamp_us is not None
            assert end_timestamp_us >= start_timestamp_us, (
                "[CameraModel]: End timestamp must be larger or equal to the start timestamp"
            )

        return self.image_points_to_world_rays_mean_pose(
            self.pixels_to_image_points(pixel_idxs),
            T_sensor_world_start,
            T_sensor_world_end,
            start_timestamp_us,
            end_timestamp_us,
            camera_rays,
            return_T_sensor_worlds,
            return_timestamps,
        )

    def image_points_to_world_rays_static_pose(
        self,
        image_points: Union[torch.Tensor, np.ndarray],
        T_sensor_world: Union[torch.Tensor, np.ndarray],
        timestamp_us: Optional[int] = None,
        camera_rays: Optional[Union[torch.Tensor, np.ndarray]] = None,
        return_T_sensor_worlds: bool = False,
        return_timestamps: bool = False,
    ) -> CameraModel.WorldRaysReturn:
        """Unprojects image points to world rays using a using a *fixed* sensor pose (not compensating for potential sensor-motion).

        Can optionally re-use known camera rays associated with image points.

        For each image point returns 3d world rays [point, direction], represented by 3d start of ray points and 3d ray directions in the world frame
        """
        # Check if the variables are numpy, convert them to torch and send them to correct device
        image_points = to_torch(image_points, device=self.device, dtype=self.dtype)
        T_sensor_world = to_torch(T_sensor_world, device=self.device, dtype=self.dtype)

        assert T_sensor_world.shape == (4, 4)
        assert len(image_points.shape) == 2
        assert image_points.shape[1] == 2
        assert image_points.dtype == self.dtype
        assert T_sensor_world.dtype == self.dtype

        # Unproject the image points to camera rays
        if camera_rays is not None:
            # Reuse provided camera rays
            camera_rays = to_torch(camera_rays, device=self.device, dtype=self.dtype)
            assert len(camera_rays.shape) == 2
            assert len(camera_rays) == len(image_points)
            assert camera_rays.shape[1] == 3
            assert camera_rays.dtype == self.dtype
        else:
            camera_rays = self.image_points_to_camera_rays(image_points)

        world_positions = T_sensor_world[:3, 3:4].transpose(0, 1).repeat(len(camera_rays), 1)  # [n_image_points, 3]

        R_sensor_world = T_sensor_world[:3, :3]  # [3, 3]

        world_ray_directions = torch.matmul(R_sensor_world, camera_rays[:, :, None]).squeeze(-1)  # [n_image_points, 3]

        # Copy the values in the output variable
        world_rays = torch.empty((len(camera_rays), 6), dtype=self.dtype, device=self.device)
        world_rays[:, :3] = world_positions
        world_rays[:, 3:] = world_ray_directions

        return_var = self.WorldRaysReturn(world_rays=world_rays)

        if return_T_sensor_worlds:
            # Repeat constant transformation for all rays
            return_var.T_sensor_worlds = torch.repeat_interleave(T_sensor_world.unsqueeze(0), len(world_rays), dim=0)

        if return_timestamps:
            assert timestamp_us is not None
            # Repeat constant timestamp for all rays
            return_var.timestamps_us = torch.tile(
                torch.tensor(timestamp_us, device=self.device), dims=(len(world_rays),)
            )

        return return_var

    def image_points_to_world_rays_mean_pose(
        self,
        image_points: Union[torch.Tensor, np.ndarray],
        T_sensor_world_start: Union[torch.Tensor, np.ndarray],
        T_sensor_world_end: Union[torch.Tensor, np.ndarray],
        start_timestamp_us: Optional[int] = None,
        end_timestamp_us: Optional[int] = None,
        camera_rays: Optional[Union[torch.Tensor, np.ndarray]] = None,
        return_T_sensor_worlds: bool = False,
        return_timestamps: bool = False,
    ) -> CameraModel.WorldRaysReturn:
        """Unprojects image points to world rays using the *mean pose* of the sensor between
        the start and end poses (not compensating for potential sensor-motion).

        Can optionally re-use known camera rays associated with image points.

        For each image point returns 3d world rays [point, direction], represented by 3d start of ray points and 3d ray directions in the world frame
        """
        if return_timestamps:
            assert start_timestamp_us is not None
            assert end_timestamp_us is not None
            assert end_timestamp_us >= start_timestamp_us, (
                "[CameraModel]: End timestamp must be larger or equal to the start timestamp"
            )

            timestamp_us = (end_timestamp_us + start_timestamp_us) // 2

        else:
            timestamp_us = None

        return self.image_points_to_world_rays_static_pose(
            image_points,
            self.__interpolate_poses(
                to_torch(T_sensor_world_start, device=self.device, dtype=self.dtype),
                to_torch(T_sensor_world_end, device=self.device, dtype=self.dtype),
                0.5,
            ),
            timestamp_us,
            camera_rays,
            return_T_sensor_worlds,
            return_timestamps,
        )

    def image_points_to_world_rays_shutter_pose(
        self,
        image_points: Union[torch.Tensor, np.ndarray],
        T_sensor_world_start: Union[torch.Tensor, np.ndarray],
        T_sensor_world_end: Union[torch.Tensor, np.ndarray],
        start_timestamp_us: Optional[int] = None,
        end_timestamp_us: Optional[int] = None,
        camera_rays: Optional[Union[torch.Tensor, np.ndarray]] = None,
        return_T_sensor_worlds: bool = False,
        return_timestamps: bool = False,
    ) -> CameraModel.WorldRaysReturn:
        """Unprojects image points to world rays using *rolling-shutter compensation* of sensor motion.

        Can optionally re-use known camera rays associated with image points.

        For each image point returns 3d world rays [point, direction], represented by 3d start of ray points and 3d ray directions in the world frame
        """
        # Global-shutter special case - no need for rolling-shutter compensation, use unprojections from start-pose as single available pose
        if self.shutter_type == types.ShutterType.GLOBAL:
            return self.image_points_to_world_rays_static_pose(
                image_points,
                T_sensor_world_start,
                start_timestamp_us,
                camera_rays,
                return_T_sensor_worlds,
                return_timestamps,
            )

        # Check if the variables are numpy, convert them to torch and send them to correct device
        image_points = to_torch(image_points, device=self.device, dtype=self.dtype)
        T_sensor_world_start = to_torch(T_sensor_world_start, device=self.device, dtype=self.dtype)
        T_sensor_world_end = to_torch(T_sensor_world_end, device=self.device, dtype=self.dtype)

        assert T_sensor_world_start.shape == (4, 4)
        assert T_sensor_world_end.shape == (4, 4)
        assert len(image_points.shape) == 2
        assert image_points.shape[1] == 2
        assert image_points.dtype == self.dtype
        assert T_sensor_world_start.dtype == self.dtype
        assert T_sensor_world_end.dtype == self.dtype

        # Unproject the image points to camera rays
        if camera_rays is not None:
            # Reuse provided camera rays
            camera_rays = to_torch(camera_rays, device=self.device, dtype=self.dtype)
            assert len(camera_rays.shape) == 2
            assert camera_rays.shape[0] == image_points.shape[0]
            assert camera_rays.shape[1] == 3
            assert camera_rays.dtype == self.dtype
        else:
            camera_rays = self.image_points_to_camera_rays(image_points)

        # Convert the start and end rotation matrix to quaternions
        R_sensor_world_s_quat = rotmat_to_unitquat(T_sensor_world_start[None, :3, :3])  # [1, 4]
        R_sensor_world_e_quat = rotmat_to_unitquat(T_sensor_world_end[None, :3, :3])  # [1, 4]

        t = self.image_points_relative_frame_times(image_points)

        world_position_rs = (1 - t)[..., None] * T_sensor_world_start[:3, 3:4].transpose(0, 1).repeat(
            t.shape[0], 1
        ) + t[..., None] * T_sensor_world_end[:3, 3:4].transpose(0, 1).repeat(t.shape[0], 1)  # [n_image_points, 3]

        R_sensor_world_rs = unitquat_to_rotmat(
            unitquat_slerp(R_sensor_world_s_quat.repeat(t.shape[0], 1), R_sensor_world_e_quat.repeat(t.shape[0], 1), t)
        )  # [n_image_points, 3, 3]

        world_ray_directions_rs = torch.bmm(R_sensor_world_rs, camera_rays[:, :, None]).squeeze(
            -1
        )  # [n_image_points, 3]

        # Copy the values in the output variable
        world_rays = torch.empty((len(image_points), 6), dtype=self.dtype, device=self.device)
        world_rays[:, :3] = world_position_rs
        world_rays[:, 3:] = world_ray_directions_rs

        return_var = self.WorldRaysReturn(world_rays=world_rays)

        if return_T_sensor_worlds:
            return_var.T_sensor_worlds = torch.zeros((len(image_points), 4, 4), dtype=self.dtype, device=self.device)
            return_var.T_sensor_worlds[:, :3, :3] = R_sensor_world_rs
            return_var.T_sensor_worlds[:, :3, 3] = world_position_rs
            return_var.T_sensor_worlds[:, 3, 3] = 1

        if return_timestamps:
            assert start_timestamp_us is not None
            assert end_timestamp_us is not None
            assert end_timestamp_us >= start_timestamp_us, (
                "[CameraModel]: End timestamp must be larger or equal to the start timestamp"
            )
            return_var.timestamps_us = (
                start_timestamp_us + (t[..., None] * (end_timestamp_us - start_timestamp_us)).to(torch.int64)
            ).squeeze(-1)  # [n_image_points]

        return return_var

    def pixels_to_image_points(self, pixel_idxs: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Given integer-based pixels indices, computes corresponding continuous image point coordinates representing the *center* of each pixel."""

        # Convert to torch
        pixel_idxs = to_torch(pixel_idxs, device=self.device)

        assert not pixel_idxs.is_floating_point(), "[CameraModel]: Pixel indices must be integers"

        # Compute the image point coordinates representing the center of each pixel (shift from top left corner to the center)
        return pixel_idxs.to(self.dtype) + 0.5

    def image_points_to_pixels(self, image_points: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Given continuous image point coordinates, computes the corresponding pixel indices."""

        # Convert to torch
        image_points = to_torch(image_points, device=self.device)

        assert image_points.is_floating_point(), "[CameraModel]: Image points must be floating point values"

        # Compute the pixel indices for given image points (round to top left corner integer coordinate)
        return torch.floor(image_points).to(torch.int32)

    def image_points_relative_frame_times(self, image_points: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Get relative frame-times based on the image point coordinates and rolling shutter type"""

        image_points = to_torch(image_points, device=self.device, dtype=self.dtype)

        # Floor/Ceil the continuous image points to the row / column index following the image coordinate
        # convention that index defines the top left corner of each pixel, e.g., the first pixels
        # u/v-range is [0.0, 1.0]
        if self.shutter_type == types.ShutterType.ROLLING_TOP_TO_BOTTOM:
            t = torch.floor(image_points[:, 1]) / (self.resolution[1] - 1)
        elif self.shutter_type == types.ShutterType.ROLLING_LEFT_TO_RIGHT:
            t = torch.floor(image_points[:, 0]) / (self.resolution[0] - 1)
        elif self.shutter_type == types.ShutterType.ROLLING_BOTTOM_TO_TOP:
            t = (self.resolution[1] - torch.ceil(image_points[:, 1])) / (self.resolution[1] - 1)
        elif self.shutter_type == types.ShutterType.ROLLING_RIGHT_TO_LEFT:
            t = (self.resolution[0] - torch.ceil(image_points[:, 0])) / (self.resolution[0] - 1)
        elif self.shutter_type == types.ShutterType.GLOBAL:
            t = torch.zeros_like(image_points[:, 0])
        else:
            raise TypeError(f"unsupported shutter-type {self.shutter_type.name} for timestamp interpolation")

        return t

    def __interpolate_poses(self, pose_s: torch.Tensor, pose_e: torch.Tensor, t: float) -> torch.Tensor:
        """Interpolate/extrapolate pose components linearly between two poses using
        linear interpolation for positions / SLERP interpolation for orientations
        given an interpolation point t in [0,1]"""
        pose_s = pose_s.to(self.device)
        pose_e = pose_e.to(self.device)

        assert pose_s.shape == (4, 4)
        assert pose_e.shape == (4, 4)

        # Convert the start and end rotation matrix to quaternions
        pose_s_quat = rotmat_to_unitquat(pose_s[None, :3, :3])  # [1, 4]
        pose_e_quat = rotmat_to_unitquat(pose_e[None, :3, :3])  # [1, 4]

        # Evaluate orientation interpolation at t
        interp_rot = unitquat_to_rotmat(
            unitquat_slerp(pose_s_quat, pose_e_quat, torch.tensor([t], device=self.device, dtype=self.dtype))
        ).squeeze()  # [3, 3]

        # Evaluate translation interpolation at t
        interp_transl = (1 - t) * pose_s[:3, 3] + t * pose_e[:3, 3]  # [3]

        interp_pose = torch.eye(4, 4, device=self.device, dtype=self.dtype)
        interp_pose[:3, :3] = interp_rot
        interp_pose[:3, 3] = interp_transl

        return interp_pose

    @staticmethod
    def _numerically_stable_xy_norm(cam_rays: torch.Tensor) -> torch.Tensor:
        """Evaluate the norm in a numerically stable manner"""

        xy_norms = torch.zeros_like(cam_rays[:, 0]).unsqueeze(1)  # Zero rays stay with zero norm

        abs_pts = torch.abs(cam_rays[:, :2])
        min_pts = torch.min(abs_pts, dim=1, keepdim=True).values
        max_pts = torch.max(abs_pts, dim=1, keepdim=True).values

        # Output the norm of non-zero rays only
        non_zero_norms = max_pts > 0
        min_max_ratio = min_pts[non_zero_norms] / max_pts[non_zero_norms]
        xy_norms[non_zero_norms, None] = max_pts[non_zero_norms, None] * torch.sqrt(
            1 + torch.pow(min_max_ratio[:, None], 2)
        )

        return xy_norms


class FThetaCameraModel(CameraModel):
    def __init__(
        self,
        camera_model_parameters: types.FThetaCameraModelParameters,
        device: Union[str, torch.device] = torch.device("cuda"),
        dtype: torch.dtype = torch.float32,
        newton_iterations: int = 3,
        min_2d_norm: float = 1e-6,
    ):
        """Initializes a FThetaCameraModel to operate on a specific device and floating-point type.

        newton_iterations: the number of Newton iterations to perform polynomial inversion (zero to disable)
        min_2d_norm: Threshold for 2d image_points-distances (relative to principal point) below which the principal ray
                     is returned in ray generation (for points close to the principal point). Needs to be positive
        """
        super().__init__(camera_model_parameters, device, dtype)

        self.reference_poly = camera_model_parameters.reference_poly

        # FThetaCameraModelParameters are defined such that the image coordinate origin corresponds to
        # the center of the first pixel. To conform to the CameraModel specification (having the image
        # coordinate origin aligned with the top-left corner of the first pixel) we therefore need to
        # offset the principal point by half a pixel.
        # Please see documentation for more information.
        self.principal_point = (
            to_torch(camera_model_parameters.principal_point, device=self.device, dtype=self.dtype) + 0.5
        )

        self.fw_poly = to_torch(camera_model_parameters.fw_poly, device=self.device, dtype=self.dtype)
        self.bw_poly = to_torch(camera_model_parameters.bw_poly, device=self.device, dtype=self.dtype)

        # Linear term A = [c,d;e;1], A^-1 = 1/(c-e*d)*[1,-d;-e,c]
        c, d, e = camera_model_parameters.linear_cde
        self.A = torch.tensor(
            [
                [c, d],
                [e, 1],
            ],
            dtype=self.dtype,
            device=self.device,
        )
        self.Ainv = torch.tensor(
            [
                [1, -d],
                [-e, c],
            ],
            dtype=self.dtype,
            device=self.device,
        ) / (c - e * d)

        # Initialize first derivative of polynomials for Newton iteration-based inversions
        self.dfw_poly = torch.tensor(
            # coefficient of first derivative of the forward polynomial
            [i * c for i, c in enumerate(camera_model_parameters.fw_poly[1:], start=1)],
            dtype=self.dtype,
            device=self.device,
        )
        self.dbw_poly = torch.tensor(
            # coefficient of first derivative of the backwards polynomial
            [i * c for i, c in enumerate(camera_model_parameters.bw_poly[1:], start=1)],
            dtype=self.dtype,
            device=self.device,
        )

        self.max_angle = float(camera_model_parameters.max_angle)
        self.newton_iterations = newton_iterations

        # 2D pixel-distance threshold
        assert min_2d_norm > 0, "require positive minimum norm threshold"
        self.min_2d_norm = torch.tensor(min_2d_norm, dtype=self.dtype, device=self.device)

        assert self.principal_point.shape == (2,)
        assert self.principal_point.dtype == self.dtype
        assert self.fw_poly.shape == (6,)
        assert self.fw_poly.dtype == self.dtype
        assert self.dfw_poly.shape == (5,)
        assert self.dfw_poly.dtype == self.dtype
        assert self.bw_poly.shape == (6,)
        assert self.bw_poly.dtype == self.dtype
        assert self.dbw_poly.shape == (5,)
        assert self.dbw_poly.dtype == self.dtype
        assert self.A.shape == (2, 2)
        assert self.A.dtype == self.dtype
        assert self.Ainv.shape == (2, 2)
        assert self.Ainv.dtype == self.dtype
        assert self.resolution.shape == (2,)
        assert self.resolution.dtype == torch.int32

    def _image_points_to_camera_rays_impl(self, image_points: torch.Tensor) -> torch.Tensor:
        """
        Computes the camera ray for each image point
        """

        image_points = to_torch(image_points, device=self.device)
        assert image_points.is_floating_point(), "[CameraModel]: image_points must be floating point values"
        image_points = image_points.to(self.dtype)

        # Get f(theta)-weighted normalized 2d vectors (undoing linear term)
        image_points_dist = torch.einsum("ij,nj->ni", self.Ainv, image_points - self.principal_point)
        rdist = torch.linalg.norm(image_points_dist, axis=1, keepdims=True)

        # Evaluate backward polynomial to get theta = f^-1(rdist) factors
        if self.reference_poly == types.FThetaCameraModelParameters.PolynomialType.PIXELDIST_TO_ANGLE:
            thetas = eval_poly_horner(self.bw_poly, rdist)  # bw is reference, evaluate it directly
        else:
            # fw is reference, evaluate its inverse via newton-based inversion
            thetas = eval_poly_inverse_horner_newton(
                self.fw_poly, self.dfw_poly, self.bw_poly, self.newton_iterations, rdist
            )

        # Compute the camera rays and set the ones at the image center to [0,0,1]
        cam_rays = torch.hstack(
            (torch.sin(thetas) * image_points_dist / torch.maximum(rdist, self.min_2d_norm), torch.cos(thetas))
        )
        cam_rays[rdist.flatten() < self.min_2d_norm, :] = torch.tensor(
            [[0, 0, 1]], device=self.device, dtype=self.dtype
        )

        return cam_rays

    def _camera_rays_to_image_points_impl(
        self, cam_rays: torch.Tensor, return_jacobians
    ) -> CameraModel.ImagePointsReturn:
        """
        For each camera ray it computes the corresponding image point coordinates
        """

        cam_rays = to_torch(cam_rays, device=self.device, dtype=self.dtype)

        initial_requires_grad = cam_rays.requires_grad
        if return_jacobians:
            cam_rays.requires_grad = True

        ray_xy_norms = self._numerically_stable_xy_norm(cam_rays)

        # Make sure norm is non-vanishing (norm vanishes for points along the principal-axis)
        ray_xy_norms[ray_xy_norms[:, 0] <= 0.0] = torch.finfo(self.dtype).eps

        thetas_full = torch.atan2(ray_xy_norms[:], cam_rays[:, 2:])

        # Limit angles to max_angle to prevent projected points to leave valid cone around max_angle.
        # In particular for omnidirectional cameras, this prevents points outside the FOV to be
        # wrongly projected to in-image-domain points because of badly constrained polynomials outside
        # the effective FOV (which is different to the image boundaries).
        #
        # These FOV-clamped projections will be marked as *invalid*
        thetas = torch.clamp(thetas_full, max=self.max_angle)

        # Evaluate forward polynomial, giving delta = f(theta) factors
        if self.reference_poly == types.FThetaCameraModelParameters.PolynomialType.PIXELDIST_TO_ANGLE:
            # bw is reference, evaluate its inverse via newton-based inversion
            deltas = eval_poly_inverse_horner_newton(
                self.bw_poly, self.dbw_poly, self.fw_poly, self.newton_iterations, thetas
            )
        else:
            # fw is reference, evaluate it directly
            deltas = eval_poly_horner(self.fw_poly, thetas)

        # Apply linear term [c,d;e,1] to f(theta)-weighted normalized 2d vectors, relative to principal point
        image_points = (
            torch.einsum("ij,nj->ni", self.A, deltas / ray_xy_norms * cam_rays[:, :2]) + self.principal_point[None, :]
        )

        # Extract valid image points
        valid_x = torch.logical_and(0.0 <= image_points[:, 0], image_points[:, 0] < self.resolution[0])
        valid_y = torch.logical_and(0.0 <= image_points[:, 1], image_points[:, 1] < self.resolution[1])
        valid_thetas = (
            thetas[:, 0] < self.max_angle
        )  # explicitly check for strictly smaller angles to classify FOV-clamped points as invalid
        valid = valid_x & valid_y & valid_thetas

        if return_jacobians:
            # Evaluate Jacobians of valid points by gradients of both output dimensions
            jacobians = torch.empty((len(cam_rays), 2, 3), dtype=self.dtype, device=self.device)

            initial_gradient = torch.ones((len(cam_rays),), dtype=self.dtype, device=self.device)
            image_points[:, 0].backward(gradient=initial_gradient, retain_graph=True)

            assert cam_rays.grad is not None
            jacobians[:, 0] = cam_rays.grad

            cam_rays.grad.zero_()

            image_points[:, 1].backward(gradient=initial_gradient)
            jacobians[:, 1] = cam_rays.grad

            # Cleanup for other backprop users
            cam_rays.grad.zero_()
            cam_rays.requires_grad = initial_requires_grad
        else:
            jacobians = None

        # If the input was numpy, return numpy arrays as well
        return CameraModel.ImagePointsReturn(image_points=image_points, valid_flag=valid, jacobians=jacobians)


class OpenCVPinholeCameraModel(CameraModel):
    def __init__(
        self,
        camera_model_parameters: types.OpenCVPinholeCameraModelParameters,
        device: Union[str, torch.device] = torch.device("cuda"),
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(camera_model_parameters, device, dtype)

        self.principal_point = to_torch(camera_model_parameters.principal_point, device=self.device, dtype=self.dtype)
        self.focal_length = to_torch(camera_model_parameters.focal_length, device=self.device, dtype=self.dtype)
        self.radial_coeffs = to_torch(camera_model_parameters.radial_coeffs, device=self.device, dtype=self.dtype)
        self.tangential_coeffs = to_torch(
            camera_model_parameters.tangential_coeffs, device=self.device, dtype=self.dtype
        )
        self.thin_prism_coeffs = to_torch(
            camera_model_parameters.thin_prism_coeffs, device=self.device, dtype=self.dtype
        )

        assert self.principal_point.shape == (2,)
        assert self.principal_point.dtype == self.dtype
        assert self.focal_length.shape == (2,)
        assert self.focal_length.dtype == self.dtype
        assert self.radial_coeffs.shape == (6,)
        assert self.radial_coeffs.dtype == self.dtype
        assert self.tangential_coeffs.shape == (2,)
        assert self.tangential_coeffs.dtype == self.dtype
        assert self.thin_prism_coeffs.shape == (4,)
        assert self.thin_prism_coeffs.dtype == self.dtype
        assert self.resolution.shape == (2,)
        assert self.resolution.dtype == torch.int32

    def _image_points_to_camera_rays_impl(self, image_points: torch.Tensor) -> torch.Tensor:
        """
        Computes the camera ray for each image point, performing an iterative undistortion of the nonlinear distortion model
        """

        image_points = to_torch(
            image_points,
            device=self.device,
        )
        assert image_points.is_floating_point(), "[CameraModel]: image_points must be floating point values"
        image_points = image_points.to(self.dtype)

        camera_rays2 = self.__iterative_undistort(image_points)
        camera_rays3 = torch.cat([camera_rays2, torch.ones_like(camera_rays2[:, :1])], dim=1)

        # make sure rays are normalized
        return camera_rays3 / torch.linalg.norm(camera_rays3, axis=1, keepdims=True)

    def _camera_rays_to_image_points_impl(
        self, cam_rays: torch.Tensor, return_jacobians
    ) -> CameraModel.ImagePointsReturn:
        """
        For each camera ray compute the corresponding image point coordinates
        """

        cam_rays = to_torch(cam_rays, device=self.device, dtype=self.dtype)

        # Initialize the valid flag and set all the points behind the camera plane to invalid
        image_points = torch.zeros_like(cam_rays[:, :2])

        valid = cam_rays[:, 2] > 0.0
        valid_idx = torch.where(valid)[0]

        cam_rays_valid = torch.index_select(cam_rays, 0, valid_idx)

        if return_jacobians:
            cam_rays_valid.requires_grad = True

        uv_normalized = cam_rays_valid[:, :2] / cam_rays_valid[:, 2:3]  # [n,2]
        icD, delta_x, delta_y, r_2 = self.__compute_distortion(uv_normalized)

        k_min_radial_dist = 0.8
        k_max_radial_dist = 1.2

        valid_radial = torch.logical_and(icD > k_min_radial_dist, icD < k_max_radial_dist)

        # Project using ideal pinhole model (apply radial / tangential / thin-prism distortions)
        # in case radial distortion is within limits
        uvND = uv_normalized[valid_radial] * icD[valid_radial, None] + torch.cat(
            [delta_x[valid_radial, None], delta_y[valid_radial, None]], dim=1
        )
        image_points[valid_idx[valid_radial]] = uvND * self.focal_length + self.principal_point

        # If the radial distortion is out-of-limits, the computed coordinates will be unreasonable
        # (might even flip signs) - check on which side of the image we overshoot, and set the coordinates
        # out of the image bounds accordingly. The coordinates will be clipped to
        # viable range and direction but the exact values cannot be trusted / are still invalid
        roi_clipping_radius = math.hypot(self.resolution[0], self.resolution[1])
        image_points[valid_idx[~valid_radial]] = (
            uv_normalized[~valid_radial] * 1 / torch.sqrt(r_2[~valid_radial, None]) * roi_clipping_radius
            + self.principal_point
        )

        # Check if the image points fall within the image
        valid_x = torch.logical_and(0.0 <= image_points[valid_idx, 0], image_points[valid_idx, 0] < self.resolution[0])
        valid_y = torch.logical_and(0.0 <= image_points[valid_idx, 1], image_points[valid_idx, 1] < self.resolution[1])

        # Set the points that have too large distortion or fall outside the image sensor to invalid
        valid_pts = valid_x & valid_y & valid_radial
        valid[valid_idx[~valid_pts]] = False

        if return_jacobians:
            # Evaluate Jacobians of valid points by gradients of both output dimensions
            jacobians = torch.zeros((len(cam_rays), 2, 3), dtype=self.dtype, device=self.device)

            initial_gradient = torch.ones((len(valid_idx),), dtype=self.dtype, device=self.device)
            image_points[valid_idx, 0].backward(gradient=initial_gradient, retain_graph=True, inputs=cam_rays_valid)

            assert cam_rays_valid.grad is not None
            jacobians[valid_idx, 0] = cam_rays_valid.grad

            cam_rays_valid.grad.zero_()

            image_points[valid_idx, 1].backward(gradient=initial_gradient)
            jacobians[valid_idx, 1] = cam_rays_valid.grad
        else:
            jacobians = None

        return CameraModel.ImagePointsReturn(image_points=image_points, valid_flag=valid, jacobians=jacobians)

    def __compute_distortion(self, xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the radial, tangential, and thin-prism distortion given the camera rays"""

        # Compute the helper variables
        xy_squared = torch.square(xy)
        r_2 = torch.sum(xy_squared, dim=1)
        xy_prod = xy[:, 0] * xy[:, 1]
        a1 = 2 * xy_prod
        a2 = r_2 + 2 * xy_squared[:, 0]
        a3 = r_2 + 2 * xy_squared[:, 1]

        icD_numerator = 1.0 + r_2 * (
            self.radial_coeffs[0] + r_2 * (self.radial_coeffs[1] + r_2 * self.radial_coeffs[2])
        )
        icD_denominator = 1.0 + r_2 * (
            self.radial_coeffs[3] + r_2 * (self.radial_coeffs[4] + r_2 * self.radial_coeffs[5])
        )
        icD = icD_numerator / icD_denominator

        delta_x = (
            self.tangential_coeffs[0] * a1
            + self.tangential_coeffs[1] * a2
            + r_2 * (self.thin_prism_coeffs[0] + r_2 * self.thin_prism_coeffs[1])
        )
        delta_y = (
            self.tangential_coeffs[0] * a3
            + self.tangential_coeffs[1] * a1
            + r_2 * (self.thin_prism_coeffs[2] + r_2 * self.thin_prism_coeffs[3])
        )

        return icD, delta_x, delta_y, r_2

    def __iterative_undistort(
        self, image_points: torch.Tensor, stop_mean_of_squares_error_px2: float = 1e-12, max_iterations: int = 10
    ) -> torch.Tensor:
        # start by unprojecting points to rays using distortion-less ideal pinhole model only
        cam_rays_0 = (image_points - self.principal_point) / self.focal_length

        cam_rays = cam_rays_0
        for _ in range(max_iterations):
            # apply *inverse* of distortion to camera rays to iteratively find the rays that correspond to the *distorted* source points
            icD, delta_x, delta_y, _ = self.__compute_distortion(cam_rays)

            residual = cam_rays - (
                cam_rays := (cam_rays_0 - torch.cat([delta_x[:, None], delta_y[:, None]], dim=1)) / icD[:, None]
            )

            if torch.mean(torch.square(residual)).item() <= stop_mean_of_squares_error_px2:
                break

        return cam_rays


class OpenCVFisheyeCameraModel(CameraModel):
    def __init__(
        self,
        camera_model_parameters: types.OpenCVFisheyeCameraModelParameters,
        device: Union[str, torch.device] = torch.device("cuda"),
        dtype: torch.dtype = torch.float32,
        newton_iterations: int = 3,
        min_2d_norm: float = 1e-6,
    ):
        super().__init__(camera_model_parameters, device, dtype)

        self.principal_point = to_torch(camera_model_parameters.principal_point, device=self.device, dtype=self.dtype)
        self.focal_length = to_torch(camera_model_parameters.focal_length, device=self.device, dtype=self.dtype)

        self.max_angle = float(camera_model_parameters.max_angle)
        self.newton_iterations = newton_iterations

        # 2D pixel-distance threshold
        assert min_2d_norm > 0, "require positive minimum norm threshold"
        self.min_2d_norm = torch.tensor(min_2d_norm, dtype=self.dtype, device=self.device)

        assert self.principal_point.shape == (2,)
        assert self.principal_point.dtype == self.dtype
        assert self.focal_length.shape == (2,)
        assert self.focal_length.dtype == self.dtype
        assert self.resolution.shape == (2,)
        assert self.resolution.dtype == torch.int32

        k1, k2, k3, k4 = camera_model_parameters.radial_coeffs[:]
        # ninth-degree forward polynomial (mapping angles to normalized distances) theta + k1*theta^3 + k2*theta^5 + k3*theta^7 + k4*theta^9
        self.forward_poly = torch.tensor([0, 1, 0, k1, 0, k2, 0, k3, 0, k4], dtype=self.dtype, device=self.device)
        # eighth-degree differential of forward polynomial 1 + 3*k1*theta^2 + 5*k2*theta^4 + 7*k3*theta^8 + 9*k4*theta^8
        self.dforward_poly = torch.tensor(
            [1, 0, 3 * k1, 0, 5 * k2, 0, 7 * k3, 0, 9 * k4], dtype=self.dtype, device=self.device
        )

        # approximate backward poly (mapping normalized distances to angles) *very crudely* by linear interpolation / equidistant angle model (also assuming image-centered principal point)
        max_normalized_dist = np.max(camera_model_parameters.resolution / 2 / camera_model_parameters.focal_length)
        self.approx_backward_poly = torch.tensor(
            [0, self.max_angle / max_normalized_dist], dtype=self.dtype, device=self.device
        )

    def _image_points_to_camera_rays_impl(self, image_points: torch.Tensor) -> torch.Tensor:
        """
        Computes the camera ray for each image point, performing an iterative undistortion of the nonlinear distortion model
        """

        image_points = to_torch(
            image_points,
            device=self.device,
        )
        assert image_points.is_floating_point(), "[CameraModel]: image_points must be floating point values"
        image_points = image_points.to(self.dtype)

        normalized_image_points = (image_points - self.principal_point) / self.focal_length
        deltas = torch.linalg.norm(normalized_image_points, axis=1, keepdims=True)

        # Evaluate backward polynomial as the inverse of the forward one
        thetas = eval_poly_inverse_horner_newton(
            self.forward_poly, self.dforward_poly, self.approx_backward_poly, self.newton_iterations, deltas
        )

        # Compute the camera rays and set the ones at the image center to [0,0,1]
        cam_rays = torch.hstack(
            (torch.sin(thetas) * normalized_image_points / torch.maximum(deltas, self.min_2d_norm), torch.cos(thetas))
        )
        cam_rays[deltas.flatten() < self.min_2d_norm, :] = torch.tensor(
            [[0, 0, 1]], device=self.device, dtype=self.dtype
        )

        return cam_rays

    def _camera_rays_to_image_points_impl(
        self, cam_rays: torch.Tensor, return_jacobians
    ) -> CameraModel.ImagePointsReturn:
        """
        For each camera ray compute the corresponding image point coordinates
        """

        cam_rays = to_torch(cam_rays, device=self.device, dtype=self.dtype)

        initial_requires_grad = cam_rays.requires_grad
        if return_jacobians:
            cam_rays.requires_grad = True

        ray_xy_norms = self._numerically_stable_xy_norm(cam_rays)

        # Make sure norm is non-vanishing (norm vanishes for points along the principal-axis)
        ray_xy_norms[ray_xy_norms[:, 0] <= 0.0] = torch.finfo(self.dtype).eps

        thetas_full = torch.atan2(ray_xy_norms[:], cam_rays[:, 2:])

        # Limit angles to max_angle to prevent projected points to leave valid cone around max_angle.
        # In particular for omnidirectional cameras, this prevents points outside the FOV to be
        # wrongly projected to in-image-domain points because of badly constrained polynomials outside
        # the effective FOV (which is different to the image boundaries).
        #
        # These FOV-clamped projections will be marked as *invalid*
        thetas = torch.clamp(thetas_full, max=self.max_angle)

        # Evaluate forward polynomial
        deltas = eval_poly_horner(
            self.forward_poly, thetas
        )  # these correspond to the radial distances to the principal point in the normalized image domain (up to focal length scales)

        image_points = self.focal_length * (deltas / ray_xy_norms * cam_rays[:, :2]) + self.principal_point[None, :]

        # Extract valid image points (projections into image domain and within max angle range)
        valid_x = torch.logical_and(0.0 <= image_points[:, 0], image_points[:, 0] < self.resolution[0])
        valid_y = torch.logical_and(0.0 <= image_points[:, 1], image_points[:, 1] < self.resolution[1])
        valid_alphas = (
            thetas[:, 0] < self.max_angle
        )  # explicitly check for strictly smaller angles to classify FOV-clamped points as invalid
        valid = valid_x & valid_y & valid_alphas

        if return_jacobians:
            # Evaluate Jacobians of valid points by gradients of both output dimensions
            jacobians = torch.empty((len(cam_rays), 2, 3), dtype=self.dtype, device=self.device)

            initial_gradient = torch.ones((len(cam_rays),), dtype=self.dtype, device=self.device)
            image_points[:, 0].backward(gradient=initial_gradient, retain_graph=True)

            assert cam_rays.grad is not None
            jacobians[:, 0] = cam_rays.grad

            cam_rays.grad.zero_()

            image_points[:, 1].backward(gradient=initial_gradient)
            jacobians[:, 1] = cam_rays.grad

            # Cleanup for other backprop users
            cam_rays.grad.zero_()
            cam_rays.requires_grad = initial_requires_grad
        else:
            jacobians = None

        return CameraModel.ImagePointsReturn(image_points=image_points, valid_flag=valid, jacobians=jacobians)
