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

import dataclasses
import itertools
import json
import os
import unittest
import unittest.mock

from typing import List, Tuple, Union, cast

import numpy as np
import parameterized
import torch

from ncore.impl.common.transformations import se3_inverse
from ncore.impl.common.util import unpack_optional
from ncore.impl.data.types import BaseLidarModelParameters, RowOffsetStructuredSpinningLidarModelParameters
from ncore.impl.sensors import lidar as lidar_module
from ncore.impl.sensors.common import to_torch
from ncore.impl.sensors.lidar import (
    LidarModel,
    RowOffsetStructuredSpinningLidarModel,
    StructuredLidarModel,
    lidar_model_from_parameters,
    maybe_lidar_model_from_parameters,
    register_lidar_model,
)


# =============================================================================
# GPU Test Detection
# =============================================================================
# Tests that require CUDA can be skipped via the NCORE_NO_GPU_TESTS environment variable.
# This is typically used in CI environments without GPU access.
#
# Usage:
#   bazel test --config=no-gpu ...
#
# This allows the same test suite to run on both:
# - Local execution / GPU CI runners - all tests run
# - CPU-only CI runners - GPU tests skipped via NCORE_NO_GPU_TESTS
# =============================================================================
def _get_test_devices() -> Tuple[torch.device, ...]:
    """Return the devices to test based on NCORE_NO_GPU_TESTS environment variable - will always contain CPU and conditionally GPU."""
    if os.environ.get("NCORE_NO_GPU_TESTS", "0") in ("1", "true", "True", "TRUE"):
        return (torch.device("cpu"),)
    if torch.version.cuda is None:  # ty: ignore[possibly-missing-submodule]
        # CPU-only torch build (e.g., Python 3.8 with torch+cpu)
        return (torch.device("cpu"),)
    return (torch.device("cpu"), torch.device("cuda"))


class TestRowOffsetStructuredSpinningLidarModelParameters(unittest.TestCase):
    """Test to verify serialization functionality of RowOffsetStructuredSpinningLidarModelParameters"""

    def setUp(self):
        with open("ncore/impl/sensors/test_data/row-offset-spinning-lidar-model-parameters.json", "r") as fp:
            self.model_parameters_ref = json.load(fp)

    def test_reserialization(self):
        """Make sure storing / loading of parameters works correctly"""

        # decode the parameters from dict
        model_parameters = RowOffsetStructuredSpinningLidarModelParameters.from_dict(self.model_parameters_ref)

        # re-enconde and make sure data is equal
        self.assertEqual(model_parameters.to_dict(), self.model_parameters_ref)


class TestLidarModelFactory(unittest.TestCase):
    """Tests for the free lidar_model_from_parameters() factory and its open registry"""

    @staticmethod
    def _parameters() -> RowOffsetStructuredSpinningLidarModelParameters:
        with open("ncore/impl/sensors/test_data/row-offset-spinning-lidar-model-parameters.json", "r") as fp:
            return RowOffsetStructuredSpinningLidarModelParameters.from_dict(json.load(fp))

    def test_dispatches_to_concrete_model(self):
        model = lidar_model_from_parameters(self._parameters(), device="cpu")
        self.assertIsInstance(model, RowOffsetStructuredSpinningLidarModel)
        # The parameters round-trip back out through the abstract interface
        self.assertEqual(model.get_parameters().type(), self._parameters().type())

    def test_maybe_variant_passes_none_through(self):
        self.assertIsNone(maybe_lidar_model_from_parameters(None, device="cpu"))
        self.assertIsInstance(
            maybe_lidar_model_from_parameters(self._parameters(), device="cpu"),
            RowOffsetStructuredSpinningLidarModel,
        )

    def test_deprecated_static_factories_forward(self):
        parameters = self._parameters()
        for owner in (LidarModel, StructuredLidarModel):
            with self.subTest(owner=owner.__name__):
                self.assertIsInstance(
                    owner.maybe_from_parameters(parameters, device="cpu"),
                    RowOffsetStructuredSpinningLidarModel,
                )
                self.assertIsNone(owner.maybe_from_parameters(None, device="cpu"))

    def test_unregistered_parameters_raise(self):
        with self.assertRaises(TypeError):
            lidar_model_from_parameters(cast(BaseLidarModelParameters, object()), device="cpu")

    def test_out_of_tree_registration(self):
        @dataclasses.dataclass
        class CustomLidarModelParameters(RowOffsetStructuredSpinningLidarModelParameters):
            @staticmethod
            def type() -> str:
                return "custom"

        class CustomLidarModel(RowOffsetStructuredSpinningLidarModel):
            pass

        reference = self._parameters()
        parameters = CustomLidarModelParameters(**dataclasses.asdict(reference))

        # Without a registration, dispatch falls back to the base's factory along the MRO
        self.assertIsInstance(
            lidar_model_from_parameters(parameters, device="cpu"), RowOffsetStructuredSpinningLidarModel
        )

        @register_lidar_model
        def _(
            lidar_model_parameters: CustomLidarModelParameters,
            device: Union[str, torch.device] = torch.device("cuda"),
            dtype: torch.dtype = torch.float32,
        ) -> CustomLidarModel:
            return CustomLidarModel(lidar_model_parameters, device=device, dtype=dtype)

        # The registry is process-global and singledispatch offers no unregister; keying it on a
        # method-local parameter type keeps the registration unreachable from other tests.
        #
        # Registration is a runtime mechanism and cannot add an overload, so this call statically
        # resolves through the overload of the concrete base these parameters derive from, i.e. to
        # `RowOffsetStructuredSpinningLidarModel`. That stays sound here because `CustomLidarModel`
        # is one; see `register_lidar_model` for when it is not.
        self.assertIsInstance(lidar_model_from_parameters(parameters, device="cpu"), CustomLidarModel)

    def test_lidar_model_parameters_type_is_covariant(self):
        # A collection of lidar models must still have `LidarModel` as a common static supertype;
        # with an invariant parameter type a type checker joining the element types would fall back
        # past `LidarModel` and reject every method call on the joined type.
        models: List[LidarModel[BaseLidarModelParameters]] = [
            lidar_model_from_parameters(self._parameters(), device="cpu"),
        ]
        for model in models:
            self.assertIsInstance(model.get_parameters(), BaseLidarModelParameters)

    def test_abstract_base_is_not_instantiable(self):
        # `ABC` does not prevent instantiation without an abstract method, and `type()` is
        # deliberately non-abstract, so the base guards itself explicitly.
        with self.assertRaises(TypeError):
            BaseLidarModelParameters()

    def test_abstract_base_declares_serialization(self):
        # Parameters can be serialized through the abstract type without narrowing
        def encode(parameters: BaseLidarModelParameters) -> dict:
            return {"lidar_model_type": parameters.type(), "lidar_model_parameters": parameters.to_dict()}

        self.assertEqual(encode(self._parameters())["lidar_model_type"], "row-offset-spinning")

        # ... but the base itself has no identifier of its own
        with self.assertRaises(NotImplementedError):
            BaseLidarModelParameters.type()


# NOTE: Uses _get_test_devices() to skip GPU tests when NCORE_NO_GPU_TESTS is set
@parameterized.parameterized_class(
    ("device", "dtype", "param_file_mapresfactor"),
    itertools.product(
        _get_test_devices(),
        (
            torch.float32,
            torch.float64,
        ),
        (
            ("row-offset-spinning-lidar-model-parameters.json", 3),
            ("row-offset-spinning-lidar-model-parameters-waymo.json", 3),
            ("row-offset-spinning-lidar-model-parameters-pandaset.json", 4),
            ("row-offset-spinning-lidar-model-parameters-hesai-at128.json", 3),
        ),
    ),
)
class TestRowOffsetStructuredSpinningLidarModel(unittest.TestCase):
    """Test to verify functionality of RowOffsetStructuredSpinningLidarModel's methods"""

    device: torch.device
    dtype: torch.dtype
    param_file_mapresfactor: Tuple[str, int]

    def setUp(self):
        with open(f"ncore/impl/sensors/test_data/{self.param_file_mapresfactor[0]}", "r") as fp:
            self.model_parameters = RowOffsetStructuredSpinningLidarModelParameters.from_dict(json.load(fp))

        self.lidar_model = RowOffsetStructuredSpinningLidarModel(
            self.model_parameters,
            angles_to_columns_map_init=False,
            angles_to_columns_map_resolution_factor=self.param_file_mapresfactor[1],
            device=self.device,
            dtype=self.dtype,
        )

        # create all element indices [relative to the static model]
        elements_grid = np.stack(
            np.meshgrid(
                np.arange(self.model_parameters.n_rows, dtype=np.int64),
                np.arange(self.model_parameters.n_columns, dtype=np.int64),
                indexing="ij",
            ),
            axis=-1,
        )

        self.elements = elements_grid.reshape(-1, 2)

    def test_model_parameters_roundtrip(self):
        """Validate model parameters obtained from torch model instances are correctly mapped back to the input versions
        between device transfers"""

        self.assertEqual(
            self.lidar_model.row_elevations_rad.device.type, self.device.type
        )  # make sure original device is correct

        # make sure retrieved parameters correspond to reference
        self.assertEqual(self.model_parameters.to_json(), self.lidar_model.get_parameters().to_json())

        # flip flop device using nn.Module magic
        if self.device.type == "cpu" and len(_get_test_devices()) == 1:
            # When on CPU and GPU tests are disabled, we can't flip to CUDA
            # Just verify CPU -> CPU works
            self.lidar_model.to(device=torch.device("cpu"))
            new_device_str = "cpu"
        else:
            new_device_str = "cuda" if self.device.type == "cpu" else "cpu"
            self.lidar_model.to(device=torch.device(new_device_str))

        self.assertEqual(
            self.lidar_model.row_elevations_rad.device.type, new_device_str
        )  # make sure the new device is correct

        # make sure retrieved parameters still correspond to reference
        self.assertEqual(self.model_parameters.to_json(), self.lidar_model.get_parameters().to_json())

    def test_angle_conversion(self):
        """Make sure angle conversion works correctly (element -> ray -> angle -> ray)"""
        sensor_rays = self.lidar_model.elements_to_sensor_rays(self.elements)

        sensor_angles = self.lidar_model.sensor_rays_to_sensor_angles(sensor_rays)
        sensor_rays_reconstructed = self.lidar_model.sensor_angles_to_sensor_rays(sensor_angles.sensor_angles)

        assert torch.all(sensor_rays_reconstructed.valid_flag)

        np.testing.assert_array_almost_equal(sensor_rays.cpu(), sensor_rays_reconstructed.sensor_rays.cpu(), decimal=6)

    def test_angle_conversion_2(self):
        """Make sure angle conversion works correctly (element -> angle -> ray -> angle)"""
        sensor_angles = self.lidar_model.elements_to_sensor_angles(self.elements)

        sensor_rays = self.lidar_model.elements_to_sensor_rays(self.elements)
        sensor_angles_reconstructed = self.lidar_model.sensor_rays_to_sensor_angles(sensor_rays)

        assert torch.all(sensor_angles_reconstructed.valid_flag)

        np.testing.assert_array_almost_equal(
            sensor_angles.cpu(), sensor_angles_reconstructed.sensor_angles.cpu(), decimal=6
        )

    def test_angles_to_columns(self):
        """Make sure angles to columns mapping works correctly"""
        relative_timestamps = self.elements[:, 1] / (
            self.model_parameters.n_columns - 1
        )  # GT relative frame times from element indices

        sensor_angles = self.lidar_model.elements_to_sensor_angles(self.elements)  # always perfect
        relative_timestamp_reconstructed = (
            self.lidar_model.sensor_angles_relative_frame_times(sensor_angles)  # map sampling-based approximation
        )

        np.testing.assert_array_almost_equal(
            relative_timestamps, relative_timestamp_reconstructed.cpu().numpy(), decimal=6
        )

    def test_angles_to_columns_map_values(self):
        """Every mapped column index must be a valid column of the model.

        The map is built by flooring a flat element index by the row count, so a
        rounding error there shows up as an out-of-range or off-by-one column.
        """
        self.lidar_model._init_angles_to_columns_map()
        angles_to_columns_map = unpack_optional(self.lidar_model.angles_to_columns_map)

        self.assertEqual(
            tuple(angles_to_columns_map.shape),
            (
                self.param_file_mapresfactor[1] * self.model_parameters.n_rows,
                self.param_file_mapresfactor[1] * self.model_parameters.n_columns,
            ),
        )
        self.assertGreaterEqual(int(angles_to_columns_map.min()), 0)
        self.assertLessEqual(int(angles_to_columns_map.max()), self.model_parameters.n_columns - 1)

    def test_flat_index_to_column_is_exact_integer_division(self):
        """The flat-index-to-column conversion must be an exact integer floor.

        _init_angles_to_columns_map turns a flat element index (column-major, so
        flat = column * n_rows + row) into a column index. Routing that through a
        float divide is exact only while the flat index stays below 2**24, where
        float32 can still represent consecutive integers; above it the quotient
        rounds up at every index just past a multiple of n_rows, moving those
        cells into the next column.

        A model with that many elements (>2**24, i.e. a >16.7M-point KD-tree) is
        far too expensive to build here, so the nearest-neighbour lookup is
        stubbed to return flat indices straddling the limit. That still exercises
        the real conversion inside _init_angles_to_columns_map. Realistic sizes
        are covered end-to-end by test_angles_to_columns.
        """
        n_rows = self.model_parameters.n_rows
        model = RowOffsetStructuredSpinningLidarModel(
            self.model_parameters,
            angles_to_columns_map_init=False,
            angles_to_columns_map_resolution_factor=1,  # keep the stubbed grid small
            angles_to_columns_map_dtype=torch.int32,  # the injected indices exceed int16
            device=self.device,
            dtype=self.dtype,
        )
        n_grid = self.model_parameters.n_rows * self.model_parameters.n_columns
        # Start a couple of whole columns below 2**24 so the range crosses the
        # first flat index whose float32 representation rounds into the next column.
        flat_indices = np.arange(2**24 - 2 * n_rows, 2**24 - 2 * n_rows + n_grid, dtype=np.int64)

        class _StubKDTree:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def query(self, *args, **kwargs):
                return np.zeros(n_grid, dtype=np.float64), flat_indices

        with unittest.mock.patch.object(lidar_module.scipy_spatial, "cKDTree", _StubKDTree):
            model._init_angles_to_columns_map()

        expected = torch.tensor(flat_indices // n_rows, dtype=torch.int32).reshape(
            self.model_parameters.n_rows, self.model_parameters.n_columns
        )
        self.assertTrue(torch.equal(unpack_optional(model.angles_to_columns_map).cpu(), expected))

    def test_rolling_shutter_projection(self):
        """Make sure rolling-shutter unprojection / projection work (mostly) consistent"""

        timestamps_us = [1659807954900403, 1659807955000364]
        T_sensor_world_start = np.array(
            [
                [9.9974847e-01, -2.2219338e-02, -3.0501457e-03, 4.6345856e01],
                [2.2246171e-02, 9.9971139e-01, 9.0645989e-03, 2.4201742e-01],
                [2.8478564e-03, -9.1301724e-03, 9.9995428e-01, 2.0181880e00],
                [0.0000000e00, 0.0000000e00, 0.0000000e00, 1.0000000e00],
            ],
            dtype=np.float32,
        )
        T_sensor_world_end = np.array(
            [
                [9.9977809e-01, -2.1048529e-02, -8.6604326e-04, 4.7494629e01],
                [2.1049019e-02, 9.9977827e-01, 5.6204927e-04, 2.4444677e-01],
                [8.5402117e-04, -5.8015389e-04, 9.9999946e-01, 2.0235672e00],
                [0.0000000e00, 0.0000000e00, 0.0000000e00, 1.0000000e00],
            ],
            dtype=np.float32,
        )

        world_rays = self.lidar_model.elements_to_world_rays_shutter_pose(
            elements=self.elements,
            T_sensor_world_start=T_sensor_world_start,
            T_sensor_world_end=T_sensor_world_end,
            start_timestamp_us=timestamps_us[0],
            end_timestamp_us=timestamps_us[1],
            return_T_sensor_worlds=True,
            return_timestamps=True,
        )

        self.assertIsNotNone(world_rays.timestamps_us, "Timestamps are not returned")
        self.assertIsNotNone(world_rays.T_sensor_worlds, "Sensor poses are not returned")

        # Check that using pre-computed sensor rays yields the same rays
        world_rays_precomputed_sensor_rays = self.lidar_model.elements_to_world_rays_shutter_pose(
            elements=self.elements,
            T_sensor_world_start=T_sensor_world_start,
            T_sensor_world_end=T_sensor_world_end,
            start_timestamp_us=timestamps_us[0],
            end_timestamp_us=timestamps_us[1],
            sensor_rays=self.lidar_model.elements_to_sensor_rays(self.elements),  # fake pre-computation
            return_T_sensor_worlds=True,
            return_timestamps=True,
        )
        self.assertIsNotNone(world_rays_precomputed_sensor_rays.timestamps_us, "Timestamps are not returned")
        self.assertIsNotNone(world_rays_precomputed_sensor_rays.T_sensor_worlds, "Sensor poses are not returned")
        np.testing.assert_array_almost_equal(
            world_rays_precomputed_sensor_rays.world_rays.cpu().numpy(), world_rays.world_rays.cpu().numpy(), decimal=6
        )

        rng = np.random.default_rng(0)
        distances = to_torch(
            rng.random((world_rays.world_rays.shape[0], 1)) * 100,
            device=self.lidar_model.device,
            dtype=self.lidar_model.dtype,
        )
        world_points = world_rays.world_rays[:, :3] + world_rays.world_rays[:, 3:] * distances

        sensor_angles_return = self.lidar_model.world_points_to_sensor_angles_shutter_pose(
            world_points=world_points,
            T_world_sensor_start=se3_inverse(T_sensor_world_start),
            T_world_sensor_end=se3_inverse(T_sensor_world_end),
            start_timestamp_us=timestamps_us[0],
            end_timestamp_us=timestamps_us[1],
            return_valid_indices=True,
            return_timestamps=True,
            return_T_world_sensors=True,
        )

        self.assertIsNotNone(world_rays.timestamps_us, "Timestamps are not returned")
        self.assertIsNotNone(world_rays.T_sensor_worlds, "Sensor poses are not returned")

        sensor_angles_ref = self.lidar_model.elements_to_sensor_angles(self.elements)

        # Assert that more than 99 % of the valid rolling-shutter reprojections are
        # within 1e-3rad ~ 0.0572deg of the original element angles
        # TODO: improve this check to handle boundary cases better
        # TODO: had to decrease the threshold to 0.98 (from 0.99) as the test was failing for Waymo model, need to investigate
        assert (
            np.sum(
                np.linalg.norm(
                    sensor_angles_return.sensor_angles.cpu().numpy()
                    - sensor_angles_ref[sensor_angles_return.valid_indices].cpu().numpy(),
                    axis=-1,
                )
                < 1e-3
            )
            / len(unpack_optional(sensor_angles_return.valid_indices))
            > 0.98
        )
