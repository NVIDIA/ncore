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

from __future__ import annotations

import logging
import sys

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

from upath import UPath


@dataclass(**({"slots": True, "kw_only": True} if sys.version_info >= (3, 10) else {}))
class BaseDataConverterConfig:
    """Generic data converter parameters"""

    ## IO
    output_dir: str  # Path where the converted data will be saved

    ## Sensor selection
    no_cameras: bool  # Disable exporting any camera sensor
    camera_ids: Optional[Tuple[str, ...]]  # Cameras to be exported (multiple value option, all if not specified)
    no_lidars: bool  # Disable exporting any lidar sensor
    lidar_ids: Optional[Tuple[str, ...]]  # Lidars to be exported (multiple value option, all if not specified)
    no_radars: bool  # Disable exporting any radar sensor
    radar_ids: Optional[Tuple[str, ...]]  # Radars to be exported (multiple value option, all if not specified)

    ## Runtime
    verbose: bool  # Enables debug logging outputs
    debug: bool
    debug_port: int


class BaseDataConverter(ABC):
    """
    Base preprocessing class used to preprocess AV datasets in a canonical representation as used in the Nvidia NCore project.

    For adding a new dataset, please inherit this class, implement the required functions, and register a new CLI command.

    The output data should follow the conventions defined in the conventions documentation.

    Please use the facilities of the data writer modules, which simplifies adding new datasets.
    """

    def __init__(self, config: BaseDataConverterConfig) -> None:
        self.logger = logging.getLogger(__name__)

        self.output_dir = UPath(config.output_dir)

        # External sensor selection overwrites
        # Store `None`` for `_active_<sensor>_ids` in case all sensors should be used, as the
        # actual full list of sensor ids will be passed via `get_active_<sensor>_ids()`
        # at conversion time (as for some data-converters the set of sensors
        # is only available after dataset introspection)
        self._active_camera_ids: list[str] | None = (
            list(config.camera_ids) if config.camera_ids is not None and len(config.camera_ids) else None
        )
        if config.no_cameras:
            self._active_camera_ids = []

        self._active_lidar_ids: list[str] | None = (
            list(config.lidar_ids) if config.lidar_ids is not None and len(config.lidar_ids) else None
        )
        if config.no_lidars:
            self._active_lidar_ids = []

        self._active_radar_ids: list[str] | None = (
            list(config.radar_ids) if config.radar_ids is not None and len(config.radar_ids) else None
        )
        if config.no_radars:
            self._active_radar_ids = []

    @staticmethod
    def _get_active_sensor_ids(
        sensor_type: str, active_sensor_ids: list[str] | None, all_sensor_ids: list[str]
    ) -> list[str]:
        """Performs generic sensor subselection and asserts active-sensors are a subset of all sensors"""
        if active_sensor_ids is None:
            return all_sensor_ids

        # Make sure active sensors are a subset of all sensors
        assert set(active_sensor_ids).issubset(all_sensor_ids), (
            f"Selected active {sensor_type} sensors {active_sensor_ids} not a subset of all available sensors {all_sensor_ids}"
        )

        return active_sensor_ids

    def get_active_camera_ids(self, all_camera_ids: list[str]) -> list[str]:
        """Returns config-specified subselection of active camera ids or all camera ids if no subselection was performed"""
        return self._get_active_sensor_ids("camera", self._active_camera_ids, all_camera_ids)

    def get_active_lidar_ids(self, all_lidar_ids: list[str]) -> list[str]:
        """Returns config-specified subselection of active lidar ids or all lidar ids if no subselection was performed"""
        return self._get_active_sensor_ids("lidar", self._active_lidar_ids, all_lidar_ids)

    def get_active_radar_ids(self, all_radar_ids: list[str]) -> list[str]:
        """Returns config-specified subselection of active radar ids or all radar ids if no subselection was performed"""
        return self._get_active_sensor_ids("radar", self._active_radar_ids, all_radar_ids)

    @classmethod
    def convert(cls, config: BaseDataConverterConfig) -> None:
        """
        Main entry-point to perform conversion of all sequences
        """

        logger = logging.getLogger(__name__)

        sequence_dirs = cls.get_sequence_paths(config)

        logger.info(f"Start converting {sequence_dirs} ...")

        # create new instance of converter for each task and execute synchronously
        for sequence_dir in sequence_dirs:
            converter = cls.from_config(config)
            converter.convert_sequence(sequence_dir)

        logger.info(f"Finished converting {sequence_dirs} in {config.output_dir} ...")

    @staticmethod
    @abstractmethod
    def get_sequence_paths(config) -> list[UPath]:
        """
        Return sequence pathnames to process
        """
        pass

    @staticmethod
    @abstractmethod
    def from_config(config) -> BaseDataConverter:
        """
        Return an instance of the data converter
        """
        pass

    @abstractmethod
    def convert_sequence(self, sequence_path: UPath) -> None:
        """
        Runs dataset-specific conversion for a sequence referenced by a directory/file path
        """
        pass


@dataclass(**({"slots": True, "kw_only": True} if sys.version_info >= (3, 10) else {}))
class FileBasedDataConverterConfig(BaseDataConverterConfig):
    """Config for converters that read from a local root directory.

    Subclass this instead of :class:`BaseDataConverterConfig` when the
    converter discovers its input sequences from ``--root-dir``.
    """

    ## IO
    root_dir: str  # Path to the raw data sequences

    def __post_init__(self) -> None:
        if not self.root_dir:
            raise ValueError(
                "--root-dir is required for this converter but was not provided. "
                "Please supply a path to the raw data sequences."
            )


class FileBasedDataConverter(BaseDataConverter):
    """Base class for converters that read source data from a local root directory.

    Subclass this instead of :class:`BaseDataConverter` when the converter needs
    ``self.root_dir``.  Pair with :class:`FileBasedDataConverterConfig` (or a
    subclass of it) to get early validation of ``root_dir``.
    """

    def __init__(self, config: FileBasedDataConverterConfig) -> None:
        super().__init__(config)
        self.root_dir = UPath(config.root_dir)
