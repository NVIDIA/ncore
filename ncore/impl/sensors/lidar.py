# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import annotations

from abc import abstractmethod, ABC

from typing import Literal, Optional, Union, cast
from dataclasses import dataclass

import torch
import numpy as np

from scipy import spatial as scipy_spatial

from ncore.impl.data import types
from ncore.impl.sensors.common import BaseModel, to_torch, rotmat_to_unitquat, unitquat_to_rotmat, unitquat_slerp


class LidarModel(BaseModel, ABC):
    """Base class for all lidar models"""

    @dataclass
    class SensorAnglesReturn:
        """
        Contains
            - sensor angles [float] (n,2)
            - valid_flag [bool] (n,)
        """

        sensor_angles: torch.Tensor
        valid_flag: torch.Tensor

    @dataclass
    class SensorRayReturn:
        """
        Contains
            - sensor rays [float] (n,3)
            - valid_flag [bool] (n,)
        """

        sensor_rays: torch.Tensor
        valid_flag: torch.Tensor

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
    class WorldPointsToSensorAnglesReturn:
        """
        Contains
            - sensor angles of the valid projections [float] (n,2)
            - [optional] world-to-sensor poses of valid projections [float] (n,4,4)
            - [optional] indices of the valid projections relative to the input points [int] (n,)
            - [optional] timestamps of the valid projections [int] (n,)
        """

        sensor_angles: torch.Tensor
        T_world_sensors: Optional[torch.Tensor] = None
        valid_indices: Optional[torch.Tensor] = None
        timestamps_us: Optional[torch.Tensor] = None

    def __init__(
        self,
        device: Union[str, torch.device],
        dtype: torch.dtype,
    ):
        super().__init__(device, dtype)
        del (device, dtype)

    @staticmethod
    def maybe_from_parameters(
        lidar_model_parameters: Optional[types.ConcreteLidarModelParametersUnion],
        device: Union[str, torch.device] = torch.device("cuda"),
        dtype: torch.dtype = torch.float32,
    ) -> Optional[LidarModel]:
        """
        Initialize a generic lidar model from parameters, if available
        """
        if lidar_model_parameters is None:
            return None
        if isinstance(lidar_model_parameters, types.RowOffsetStructuredSpinningLidarModelParameters):
            return RowOffsetStructuredSpinningLidarModel(lidar_model_parameters, device=device, dtype=dtype)
        raise TypeError(
            f"Unsupported lidar model type {type(lidar_model_parameters)}, currently only supporting 'RowOffsetStructuredSpinningLidarModel'."
        )

    @abstractmethod
    def sensor_rays_to_sensor_angles(
        self, sensor_rays: Union[torch.Tensor, np.ndarray], normalized: bool = True
    ) -> SensorAnglesReturn:
        """
        Lidar model-specific implementation of sensor_rays_to_sensor_angles
        """
        ...

    @abstractmethod
    def sensor_angles_to_sensor_rays(self, sensor_angles: Union[torch.Tensor, np.ndarray]) -> SensorRayReturn:
        """Lidar model-specific implementation of elevation/azimuth angles to sensor rays"""
        ...


class StructuredLidarModel(LidarModel, ABC):
    @staticmethod
    def maybe_from_parameters(
        lidar_model_parameters: Optional[types.ConcreteLidarModelParametersUnion],
        device: Union[str, torch.device] = torch.device("cuda"),
        dtype: torch.dtype = torch.float32,
    ) -> Optional[StructuredLidarModel]:
        """
        Initialize a generic lidar model from parameters, if available
        """
        if lidar_model_parameters is None:
            return None
        if isinstance(lidar_model_parameters, types.RowOffsetStructuredSpinningLidarModelParameters):
            return RowOffsetStructuredSpinningLidarModel(lidar_model_parameters, device=device, dtype=dtype)
        raise TypeError(
            f"Unsupported structured lidar model type {type(lidar_model_parameters)}, currently only supporting 'RowOffsetStructuredSpinningLidarModel'."
        )

    def elements_to_sensor_points(
        self, elements: Union[torch.Tensor, np.ndarray], element_distances: Union[torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        """Computes 3d sensor points for elements in the structured lidar model. Elements are given as (row, column) indices."""

        # elements: N x 2 array of (row, column) indices
        assert elements.ndim == 2
        assert element_distances.ndim == 1

        elements = to_torch(elements, device=self.device, dtype=torch.long)
        element_distances = to_torch(element_distances, device=self.device, dtype=self.dtype)

        sensor_rays = self.elements_to_sensor_rays(elements)

        return sensor_rays * element_distances[:, None]

    def elements_to_sensor_rays(self, elements: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Computes normalized 3d sensor ray directions for elements in the structured lidar model. Elements are given as (row, column) indices."""

        elements = to_torch(elements, device=self.device, dtype=torch.long)

        sensor_angles = self.elements_to_sensor_angles(elements)
        return self.sensor_angles_to_sensor_rays(sensor_angles).sensor_rays

    @abstractmethod
    def elements_to_sensor_angles(self, elements: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Lidar model-specific implementation of elements_to_sensor_angles
        """
        ...


class RowOffsetStructuredSpinningLidarModel(StructuredLidarModel):
    """Represents a structured spinning lidar model that is using a per-row azimuth-offset (compatible with, e.g., Hesai P128 sensors)"""

    row_elevations_rad: torch.Tensor
    column_azimuths_rad: torch.Tensor
    row_azimuth_offsets_rad: torch.Tensor
    angles_to_columns_map_resolution_factor: int
    angles_to_columns_map_dtype: torch.dtype
    angles_to_columns_map: Optional[torch.Tensor]
    spinning_frequency_hz: float
    spinning_direction: Literal["cw", "ccw"]
    fov_vert_start_rad: float
    fov_vert_end_rad: float
    fov_horiz_start_rad: float
    fov_horiz_end_rad: float
    n_rows: int
    n_columns: int

    def __init__(
        self,
        parameters: types.RowOffsetStructuredSpinningLidarModelParameters,
        angles_to_columns_map_resolution_factor: int = 3,
        angles_to_columns_map_dtype=torch.int16,
        angles_to_columns_map_init=False,
        device: Union[str, torch.device] = torch.device("cuda"),
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(device=device, dtype=dtype)

        self.register_buffer(
            "row_elevations_rad", to_torch(parameters.row_elevations_rad, device=self.device, dtype=self.dtype)
        )
        self.register_buffer(
            "column_azimuths_rad", to_torch(parameters.column_azimuths_rad, device=self.device, dtype=self.dtype)
        )
        self.register_buffer(
            "row_azimuth_offsets_rad",
            to_torch(parameters.row_azimuth_offsets_rad, device=self.device, dtype=self.dtype),
        )

        self.angles_to_columns_map_resolution_factor = angles_to_columns_map_resolution_factor
        self.angles_to_columns_map_dtype = angles_to_columns_map_dtype
        self.angles_to_columns_map: Optional[torch.Tensor] = None

        self.spinning_frequency_hz = parameters.spinning_frequency_hz
        self.spinning_direction = parameters.spinning_direction
        self.n_rows = parameters.n_rows
        self.n_columns = parameters.n_columns
        self.fov_horiz_start_rad = parameters.fov_horiz_start_rad
        self.fov_horiz_end_rad = parameters.fov_horiz_end_rad
        self.fov_vert_start_rad = parameters.fov_vert_start_rad
        self.fov_vert_end_rad = parameters.fov_vert_end_rad

        if angles_to_columns_map_init:
            self._init_angles_to_columns_map()

    def get_parameters(self) -> types.RowOffsetStructuredSpinningLidarModelParameters:
        """Returns the lidar model parameters specific to the current lidar model instance"""

        return types.RowOffsetStructuredSpinningLidarModelParameters(
            spinning_frequency_hz=self.spinning_frequency_hz,
            spinning_direction=self.spinning_direction,
            n_rows=self.n_rows,
            n_columns=self.n_columns,
            fov_horiz_start_rad=self.fov_horiz_start_rad,
            fov_horiz_end_rad=self.fov_horiz_end_rad,
            fov_vert_start_rad=self.fov_vert_start_rad,
            fov_vert_end_rad=self.fov_vert_end_rad,
            row_elevations_rad=self.row_elevations_rad.cpu().numpy().astype(np.float32),
            column_azimuths_rad=self.column_azimuths_rad.cpu().numpy().astype(np.float32),
            row_azimuth_offsets_rad=self.row_azimuth_offsets_rad.cpu().numpy().astype(np.float32),
        )

    def elements_to_sensor_angles(self, elements: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Retrieves the elevation and azimuth angles for elements in the structured lidar model. Elements are given as (row, column) indices."""

        # elements: N x 2 array of (row, column) indices
        assert elements.ndim == 2

        elements = to_torch(elements, device=self.device, dtype=torch.long)

        # reconstruct angles from model parameterization
        elevations_rad = self.row_elevations_rad[elements[:, 0]]
        azimuths_rad = self.__warp_azimuth(
            self.column_azimuths_rad[elements[:, 1]] + self.row_azimuth_offsets_rad[elements[:, 0]]
        )

        sensor_angles = torch.stack([elevations_rad, azimuths_rad], dim=-1)

        return sensor_angles

    def sensor_rays_to_sensor_angles(
        self, sensor_rays: Union[torch.Tensor, np.ndarray], normalized: bool = True
    ) -> LidarModel.SensorAnglesReturn:
        """Computes the elevation and azimuth angles for normalized 3d sensor rays."""

        # sensor_rays: N x 3 array of sensor rays
        assert sensor_rays.ndim == 2

        sensor_rays = to_torch(sensor_rays, device=self.device, dtype=self.dtype)

        if not normalized:
            sensor_rays /= torch.norm(sensor_rays, dim=-1, keepdim=True)

        elevations_rad = torch.arcsin(sensor_rays[:, 2])
        azimuths_rad = torch.arctan2(sensor_rays[:, 1], sensor_rays[:, 0])
        sensor_angles = torch.stack([elevations_rad, azimuths_rad], dim=-1)

        return LidarModel.SensorAnglesReturn(
            sensor_angles=sensor_angles, valid_flag=self.__valid_sensor_angles(sensor_angles)
        )

    def sensor_angles_to_sensor_rays(
        self, sensor_angles: Union[torch.Tensor, np.ndarray]
    ) -> LidarModel.SensorRayReturn:
        """Computes the sensor rays for elevation/azimuth angles."""

        # sensor_angles: N x 2 array of elevation and azimuth angles
        assert len(sensor_angles.shape) == 2

        sensor_angles = to_torch(sensor_angles, device=self.device, dtype=self.dtype)

        elevations_rad = sensor_angles[:, 0]
        azimuths_rad = sensor_angles[:, 1]

        x = torch.cos(azimuths_rad) * (cos_elevations := torch.cos(elevations_rad))
        y = torch.sin(azimuths_rad) * cos_elevations
        z = torch.sin(elevations_rad)

        return LidarModel.SensorRayReturn(
            sensor_rays=torch.stack([x, y, z], dim=-1), valid_flag=self.__valid_sensor_angles(sensor_angles)
        )

    def _init_angles_to_columns_map(self):
        # angles to column map is a 2D array of shape resolution_factor * (n_rows, n_columns)

        assert torch.iinfo(self.angles_to_columns_map_dtype).max >= self.n_columns - 1, (
            "The dtype for the angles to columns map must be able to store the maximum column index, consider increasing angles_to_columns_map_dtype"
        )

        assert (
            not self.angles_to_columns_map_dtype.is_floating_point and not self.angles_to_columns_map_dtype.is_complex
        ), "The dtype for the angles to columns map must be an integer type"

        # create all element indices [relative to the static model]
        elements = torch.stack(
            torch.meshgrid(
                torch.arange(self.n_rows, dtype=torch.long),
                torch.arange(self.n_columns, dtype=torch.long),
                indexing="ij",
            ),
            dim=-1,
        )

        # reconstruct angles from model parameterization
        element_azimuths_rad = self.__warp_azimuth(
            self.column_azimuths_rad[elements[:, :, 1]] + self.row_azimuth_offsets_rad[elements[:, :, 0]]
        )

        # reconstruct angles from model parameterization

        # regular angles of the map
        grid_elevations_rad, grid_azimuths_rad = torch.meshgrid(
            torch.linspace(
                self.fov_vert_start_rad,
                self.fov_vert_end_rad,
                self.angles_to_columns_map_resolution_factor * self.n_rows,
                device=self.device,
            ),
            torch.linspace(
                self.fov_horiz_start_rad,
                self.fov_horiz_end_rad,
                self.angles_to_columns_map_resolution_factor * self.n_columns,
                device=self.device,
            ),
            indexing="ij",
        )

        self.map_resolution_horiz_rad = (self.fov_horiz_end_rad - self.fov_horiz_start_rad) / (
            self.angles_to_columns_map_resolution_factor * self.n_columns - 1
        )
        self.map_resolution_vert_rad = (self.fov_vert_start_rad - self.fov_vert_end_rad) / (
            self.angles_to_columns_map_resolution_factor * self.n_rows - 1
        )

        # Convert grid and sensor angles to unit norm rays
        grid_angles = torch.cat([grid_elevations_rad.reshape(-1, 1), grid_azimuths_rad.reshape(-1, 1)], dim=1)
        grid_rays = self.sensor_angles_to_sensor_rays(grid_angles)

        sensor_angles = torch.cat(
            [
                torch.tile(self.row_elevations_rad, [self.n_columns, 1]).reshape(-1, 1),
                element_azimuths_rad.transpose(0, 1).reshape(-1, 1),
            ],
            dim=1,
        )
        sensor_rays = self.sensor_angles_to_sensor_rays(sensor_angles)

        # Compute the NN of each grid ray in the sensor rays
        # TODO: can we move this to torch/CUDA? do we assume that we always have access?
        kdtree = scipy_spatial.cKDTree(sensor_rays.sensor_rays.cpu().numpy())
        _, idxs = kdtree.query(grid_rays.sensor_rays.cpu().numpy())
        idxs = to_torch(idxs, device=self.device, dtype=torch.int32)

        # Map the indices to the columns by dividing with the total number of rows and (implicit) flooring
        self.angles_to_columns_map = (
            (idxs / self.n_rows).to(self.angles_to_columns_map_dtype).reshape(grid_elevations_rad.shape)
        )

    def sensor_angles_relative_frame_times(self, sensor_angles: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Get relative frame-times of sensor angle coordinates using internal angle to column mapping.

        All sensor angles need to be in the FOV of the sensor.
        """

        # sensor_angles: N x 2 array of elevation and azimuth angles in radians

        if self.angles_to_columns_map is None:
            self._init_angles_to_columns_map()

        assert self.angles_to_columns_map is not None

        sensor_angles = to_torch(sensor_angles, device=self.device, dtype=self.dtype)

        # Normalize azimuths
        sensor_angles[:, 1] = self.__warp_azimuth(sensor_angles[:, 1])

        # Check that all angles are in the fov
        # TODO: this might be slow, need to check if we can do this outside or somehow guarantee that the angles are in the fov
        sensor_angles_ranges = torch.aminmax(
            sensor_angles
            # FOV ranges are only exact up to f32
            .to(torch.float32),
            dim=0,
        )
        assert self.fov_vert_end_rad <= sensor_angles_ranges.min[0].item()
        assert sensor_angles_ranges.max[0].item() <= self.fov_vert_start_rad

        assert self.fov_horiz_start_rad <= sensor_angles_ranges.min[1].item()
        assert sensor_angles_ranges.max[1].item() <= self.fov_horiz_end_rad

        elevations_rad = sensor_angles[:, 0]
        azimuths_rad = sensor_angles[:, 1]

        # Determine the location of the angle in the map (nearest neighbor lookup)
        horizontal_idxs = (
            (azimuths_rad - self.fov_horiz_start_rad + self.map_resolution_horiz_rad / 2)
            / self.map_resolution_horiz_rad
        ).to(torch.long)
        vertical_idxs = (
            (self.fov_vert_start_rad - elevations_rad + self.map_resolution_vert_rad / 2) / self.map_resolution_vert_rad
        ).to(torch.long)

        # Grab the corresponding column index from the map
        column_indices = self.angles_to_columns_map[vertical_idxs, horizontal_idxs]

        # Compute the relative frame time using the column-associated relative time
        return column_indices.to(self.dtype) / (self.n_columns - 1)

    def world_points_to_sensor_angles_shutter_pose(
        self,
        world_points: Union[torch.Tensor, np.ndarray],
        T_world_sensor_start: Union[torch.Tensor, np.ndarray],
        T_world_sensor_end: Union[torch.Tensor, np.ndarray],
        start_timestamp_us: Optional[int] = None,
        end_timestamp_us: Optional[int] = None,
        max_iterations: int = 10,
        stop_mean_relative_time_error: float = 1e-4,
        stop_delta_mean_relative_time_error: float = 1e-6,
        return_T_world_sensors: bool = False,
        return_valid_indices: bool = False,
        return_timestamps: bool = False,
    ) -> RowOffsetStructuredSpinningLidarModel.WorldPointsToSensorAnglesReturn:
        """Projects world points to corresponding sensor angle coordinates using *rolling-shutter compensation* of sensor motion"""

        if return_timestamps:
            assert start_timestamp_us is not None
            assert end_timestamp_us is not None
            assert end_timestamp_us >= start_timestamp_us, (
                "[LidarModel]: End timestamp must be larger or equal to the start timestamp"
            )

            # Make sure timestamps have correct type (might be, e.g., np.uint64, which torch doesn't like)
            start_timestamp_us = int(start_timestamp_us)
            end_timestamp_us = int(end_timestamp_us)

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

        # Do initial transformations using both start and end pose to determine all candidate points and take union of valid projections as iteration starting points
        sensor_angles_start = self.sensor_rays_to_sensor_angles(
            (T_world_sensor_start[:3, :3] @ world_points.transpose(0, 1) + T_world_sensor_start[:3, 3, None]).transpose(
                0, 1
            ),
            normalized=False,
        )

        sensor_angles_end = self.sensor_rays_to_sensor_angles(
            (T_world_sensor_end[:3, :3] @ world_points.transpose(0, 1) + T_world_sensor_end[:3, 3, None]).transpose(
                0, 1
            ),
            normalized=False,
        )

        valid = sensor_angles_start.valid_flag | sensor_angles_end.valid_flag  # union of valid image points
        initial_angles = sensor_angles_end.sensor_angles
        relative_time = torch.ones_like(initial_angles[:, 0])
        initial_angles[sensor_angles_start.valid_flag] = sensor_angles_start.sensor_angles[
            sensor_angles_start.valid_flag
        ]
        relative_time[sensor_angles_start.valid_flag] = 0.0
        # this prefers points at the start-of-frame pose over end-of-frame points
        # - the optimization will determine the final timestamp for each point

        # Exit early if no point projected to a valid sensor angle
        if not valid.any():
            return_var = self.WorldPointsToSensorAnglesReturn(
                sensor_angles=torch.empty((0, 2), dtype=self.dtype, device=self.device)
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
        sensor_angles_rs_prev = initial_angles[valid, :]
        relative_time_prev = relative_time[valid].clone()
        mean_relative_time_error = 0.5  # initialize the value to a expected value of random association [0,1]

        for _ in range(max_iterations):
            relative_time = self.sensor_angles_relative_frame_times(sensor_angles_rs_prev)  # [n_valid]

            rot_rs = unitquat_to_rotmat(
                unitquat_slerp(
                    world_sensor_s_quat.repeat(relative_time.shape[0], 1),
                    world_sensor_e_quat.repeat(relative_time.shape[0], 1),
                    relative_time,
                )
            )  # [n_valid, 3, 3]

            trans_rs = (1 - relative_time)[..., None] * T_world_sensor_start[:3, 3:4].transpose(0, 1).repeat(
                relative_time.shape[0], 1
            ) + relative_time[..., None] * T_world_sensor_end[:3, 3:4].transpose(0, 1).repeat(relative_time.shape[0], 1)

            sensor_angles = self.sensor_rays_to_sensor_angles(
                (torch.bmm(rot_rs, world_points[valid, :, None]) + trans_rs[..., None]).squeeze(-1), normalized=False
            )

            # Compute mean error of projections that are still valid now and check if we are still
            # making progress relative to previous iteration
            if (
                abs(
                    mean_relative_time_error
                    - (
                        mean_relative_time_error := (
                            relative_time[sensor_angles.valid_flag] - relative_time_prev[sensor_angles.valid_flag]
                        )
                        .abs()
                        .mean()
                        .item()
                    )
                )
                <= stop_delta_mean_relative_time_error
            ):
                break

            # Check if error bound was reached
            if mean_relative_time_error <= stop_mean_relative_time_error:
                break

            sensor_angles_rs_prev[sensor_angles.valid_flag] = sensor_angles.sensor_angles[sensor_angles.valid_flag]
            relative_time_prev[sensor_angles.valid_flag] = relative_time[sensor_angles.valid_flag].clone()

        # We always return sensor angles points
        return_var = self.WorldPointsToSensorAnglesReturn(
            sensor_angles=sensor_angles.sensor_angles[sensor_angles.valid_flag]
        )

        if return_T_world_sensors:
            # Generate the output matrix
            trans_matrices = torch.empty(
                (int(sensor_angles.valid_flag.sum().item()), 4, 4), dtype=self.dtype, device=self.device
            )
            trans_matrices[:, :3, 3] = trans_rs[sensor_angles.valid_flag]
            trans_matrices[:, :3, :3] = rot_rs[sensor_angles.valid_flag, ...]
            trans_matrices[:, 3] = torch.tensor([0, 0, 0, 1], device=self.device, dtype=self.dtype)

            return_var.T_world_sensors = trans_matrices

        if return_valid_indices:
            # Combine validity flags
            # (valid_rs represents a strict logical subset of full valid flags, so no logical operation required)
            valid[torch.argwhere(valid).squeeze()] = sensor_angles.valid_flag
            return_var.valid_indices = torch.argwhere(valid).squeeze(1)

        if return_timestamps:
            # MYPY is stupid and can't see that we are doing the same above already
            assert start_timestamp_us is not None
            assert end_timestamp_us is not None
            return_var.timestamps_us = (
                start_timestamp_us
                + (relative_time[sensor_angles.valid_flag, None] * (end_timestamp_us - start_timestamp_us)).to(
                    torch.int64
                )
            ).squeeze(1)

        return return_var

    def elements_to_world_rays_shutter_pose(
        self,
        elements: Union[torch.Tensor, np.ndarray],
        T_sensor_world_start: Union[torch.Tensor, np.ndarray],
        T_sensor_world_end: Union[torch.Tensor, np.ndarray],
        start_timestamp_us: Optional[int] = None,
        end_timestamp_us: Optional[int] = None,
        return_T_sensor_worlds: bool = False,
        return_timestamps: bool = False,
    ) -> RowOffsetStructuredSpinningLidarModel.WorldRaysReturn:
        """Unprojects elements to world rays using *rolling-shutter compensation* of sensor motion."""

        # Check if the variables are numpy, convert them to torch and send them to correct device
        elements = to_torch(elements, device=self.device, dtype=torch.long)

        T_sensor_world_start = to_torch(T_sensor_world_start, device=self.device, dtype=self.dtype)
        T_sensor_world_end = to_torch(T_sensor_world_end, device=self.device, dtype=self.dtype)

        assert T_sensor_world_start.shape == (4, 4)
        assert T_sensor_world_end.shape == (4, 4)
        assert len(elements.shape) == 2
        assert elements.shape[1] == 2
        assert elements.dtype == torch.long
        assert T_sensor_world_start.dtype == self.dtype
        assert T_sensor_world_end.dtype == self.dtype

        if return_timestamps:
            assert start_timestamp_us is not None
            assert end_timestamp_us is not None
            assert end_timestamp_us >= start_timestamp_us, (
                "[LidarModel]: End timestamp must be larger or equal to the start timestamp"
            )

            # Make sure timestamps have correct type (might be, e.g., np.uint64, which torch doesn't like)
            start_timestamp_us = int(start_timestamp_us)
            end_timestamp_us = int(end_timestamp_us)

        # Convert the start and end rotation matrix to quaternions
        R_sensor_world_s_quat = rotmat_to_unitquat(T_sensor_world_start[None, :3, :3])  # [1, 4]
        R_sensor_world_e_quat = rotmat_to_unitquat(T_sensor_world_end[None, :3, :3])  # [1, 4]

        # Compute the sensor rays for the elements
        sensor_rays = self.elements_to_sensor_rays(elements)

        # Get relative frame-times based on the elements column index relative to the total number of columns
        # (columns are measured in increasing time order irrespective of spin-direction)
        t = elements[:, 1].to(self.dtype) / (self.n_columns - 1)

        world_position_rs = (1 - t)[..., None] * T_sensor_world_start[:3, 3:4].transpose(0, 1).repeat(
            t.shape[0], 1
        ) + t[..., None] * T_sensor_world_end[:3, 3:4].transpose(0, 1).repeat(t.shape[0], 1)  # [n_elements, 3]

        R_sensor_world_rs = unitquat_to_rotmat(
            unitquat_slerp(R_sensor_world_s_quat.repeat(t.shape[0], 1), R_sensor_world_e_quat.repeat(t.shape[0], 1), t)
        )  # [n_elements, 3, 3]

        world_ray_directions_rs = torch.bmm(R_sensor_world_rs, sensor_rays[:, :, None]).squeeze(-1)  # [n_elements, 3]

        # Copy the values in the output variable
        return_var = self.WorldRaysReturn(
            world_rays=torch.cat(
                tensors=(cast(torch.Tensor, world_position_rs), cast(torch.Tensor, world_ray_directions_rs)), dim=1
            )  # [n_elements, 6])
        )

        if return_T_sensor_worlds:
            return_var.T_sensor_worlds = torch.zeros((len(sensor_rays), 4, 4), dtype=self.dtype, device=self.device)
            return_var.T_sensor_worlds[:, :3, :3] = R_sensor_world_rs
            return_var.T_sensor_worlds[:, :3, 3] = world_position_rs
            return_var.T_sensor_worlds[:, 3, 3] = 1

        if return_timestamps:
            assert start_timestamp_us is not None
            assert end_timestamp_us is not None
            return_var.timestamps_us = (
                start_timestamp_us + (t[..., None] * (end_timestamp_us - start_timestamp_us)).to(torch.int64)
            ).squeeze(-1)  # [n_elements]

        return return_var

    def __warp_azimuth(self, azimuths_rad: torch.Tensor) -> torch.Tensor:
        """Wraps the azimuth angle to the interval (-pi, pi]"""

        azimuths_rad[azimuths_rad > np.pi] -= 2 * np.pi
        azimuths_rad[azimuths_rad <= -np.pi] += 2 * np.pi

        return azimuths_rad

    def __valid_sensor_angles(self, sensor_angles: torch.Tensor) -> torch.Tensor:
        """Checks if a sensor angles are valid / within the sensor's field of view"""
        return (
            (self.fov_vert_end_rad <= sensor_angles[:, 0])
            & (sensor_angles[:, 0] <= self.fov_vert_start_rad)
            & (self.fov_horiz_start_rad <= sensor_angles[:, 1])
            & (sensor_angles[:, 1] <= self.fov_horiz_end_rad)
        )
