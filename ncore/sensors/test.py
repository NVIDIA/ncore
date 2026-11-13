# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def test_import_package():
    """Make sure package can be imported"""

    import ncore.sensors  # noqa: F401


def test_camera_model_keyword_only_parameters():
    """The tuning parameters of camera models (after the device and dtype) are keyword-only"""
    from typing import Any, cast

    import numpy as np
    import pytest
    import torch

    from ncore.data import (
        FThetaCameraModelParameters,
        OpenCVFisheyeCameraModelParameters,
        OpenCVPinholeCameraModelParameters,
        ShutterType,
    )
    from ncore.sensors import FThetaCameraModel, OpenCVFisheyeCameraModel, OpenCVPinholeCameraModel

    pinhole = OpenCVPinholeCameraModelParameters(
        resolution=np.array([640, 480], dtype=np.uint64),
        shutter_type=ShutterType.GLOBAL,
        principal_point=np.array([320.0, 240.0], dtype=np.float32),
        focal_length=np.array([500.0, 500.0], dtype=np.float32),
        radial_coeffs=np.zeros(6, dtype=np.float32),
        tangential_coeffs=np.zeros(2, dtype=np.float32),
        thin_prism_coeffs=np.zeros(4, dtype=np.float32),
    )
    fisheye = OpenCVFisheyeCameraModelParameters(
        resolution=np.array([640, 480], dtype=np.uint64),
        shutter_type=ShutterType.GLOBAL,
        principal_point=np.array([320.0, 240.0], dtype=np.float32),
        focal_length=np.array([500.0, 500.0], dtype=np.float32),
        radial_coeffs=np.zeros(4, dtype=np.float32),
        max_angle=1.0,
    )
    ftheta = FThetaCameraModelParameters(
        resolution=np.array([640, 480], dtype=np.uint64),
        shutter_type=ShutterType.GLOBAL,
        principal_point=np.array([320.0, 240.0], dtype=np.float32),
        reference_poly=FThetaCameraModelParameters.PolynomialType.PIXELDIST_TO_ANGLE,
        pixeldist_to_angle_poly=np.array([0.0, 1e-3, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        angle_to_pixeldist_poly=np.array([0.0, 1e3, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        max_angle=0.6,
    )
    cpu, dtype = torch.device("cpu"), torch.float32

    # (untyped: the positional calls don't type check)
    for model, parameters, tuning in (
        (OpenCVPinholeCameraModel, pinhole, (1.0,)),
        (OpenCVFisheyeCameraModel, fisheye, (5, 1e-5)),
        (FThetaCameraModel, ftheta, (5, 1e-5)),
    ):
        with pytest.raises(TypeError, match="positional argument"):
            cast(Any, model)(parameters, cpu, dtype, *tuning)

    # (keyword arguments are accepted)
    assert (
        OpenCVPinholeCameraModel(pinhole, cpu, dtype, max_normalized_radius_squared=1.0).max_normalized_radius_squared
        == 1.0
    )
    for model in (
        OpenCVFisheyeCameraModel(fisheye, cpu, dtype, newton_iterations=5, min_2d_norm=1e-5),
        FThetaCameraModel(ftheta, cpu, dtype, newton_iterations=5, min_2d_norm=1e-5),
    ):
        assert model.newton_iterations == 5
        assert float(model.min_2d_norm) == pytest.approx(1e-5)


def test_lidar_model_keyword_only_parameters():
    """The tuning parameters of lidar models (after the device and dtype) are keyword-only"""
    from typing import Any, cast

    import numpy as np
    import pytest
    import torch

    from ncore.data import RowOffsetStructuredSpinningLidarModelParameters
    from ncore.sensors import RowOffsetStructuredSpinningLidarModel

    elevations = np.linspace(0.1, -0.3, 8, dtype=np.float32)
    parameters = RowOffsetStructuredSpinningLidarModelParameters(
        spinning_frequency_hz=10.0,
        spinning_direction="cw",
        n_rows=len(elevations),
        n_columns=16,
        row_elevations_rad=elevations,
        column_azimuths_rad=np.linspace(np.pi, -np.pi, 16, endpoint=False, dtype=np.float32),
        row_azimuth_offsets_rad=np.zeros_like(elevations),
    )
    cpu = torch.device("cpu")

    with pytest.raises(TypeError, match="positional argument"):
        # (untyped: the call doesn't type check)
        cast(Any, RowOffsetStructuredSpinningLidarModel)(parameters, cpu, torch.float32, 2)

    # (device and dtype are positional-or-keyword, keyword tuning arguments are accepted)
    model = RowOffsetStructuredSpinningLidarModel(
        parameters,
        cpu,
        torch.float64,
        angles_to_columns_map_resolution_factor=2,
        angles_to_columns_map_dtype=torch.int32,
        angles_to_columns_map_init=False,
        fov_eps_factor=2.0,
    )
    assert model.device == cpu
    assert model.dtype == torch.float64


def test_world_rays_keyword_only_parameters():
    """The optional ray and return arguments of the world-ray unprojections of camera and lidar models are
    keyword-only"""
    import inspect

    from ncore.sensors import CameraModel, RowOffsetStructuredSpinningLidarModel

    for function, expected in (
        (CameraModel.pixels_to_world_rays_static_pose, ["camera_rays", "return_T_sensor_worlds", "return_timestamps"]),
        (CameraModel.pixels_to_world_rays_shutter_pose, ["camera_rays", "return_T_sensor_worlds", "return_timestamps"]),
        (CameraModel.pixels_to_world_rays_mean_pose, ["camera_rays", "return_T_sensor_worlds", "return_timestamps"]),
        (
            CameraModel.image_points_to_world_rays_static_pose,
            ["camera_rays", "return_T_sensor_worlds", "return_timestamps"],
        ),
        (
            CameraModel.image_points_to_world_rays_shutter_pose,
            ["camera_rays", "return_T_sensor_worlds", "return_timestamps"],
        ),
        (
            CameraModel.image_points_to_world_rays_mean_pose,
            ["camera_rays", "return_T_sensor_worlds", "return_timestamps"],
        ),
        (
            RowOffsetStructuredSpinningLidarModel.elements_to_world_rays_shutter_pose,
            ["sensor_rays", "return_T_sensor_worlds", "return_timestamps"],
        ),
    ):
        parameters = inspect.signature(function).parameters.values()
        assert [p.name for p in parameters if p.kind is p.KEYWORD_ONLY] == expected, function.__qualname__
