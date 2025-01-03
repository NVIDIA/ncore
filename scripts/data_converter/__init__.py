# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import logging

from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional

try:
    from scripts.util import breakpoint
except ImportError:
    # if included externally as 'ncore_repo' use fully-evaluated path
    from ncore_repo.scripts.util import breakpoint  # type: ignore
except ImportError:
    # fall back to 'external'-prefixed 'ncore_repo' reference if required
    from external.ncore_repo.scripts.util import breakpoint  # type: ignore

import click


class Config(object):
    """Simple dictionary holding all options as key/value pairs"""

    def __init__(self, kwargs):
        self.__dict__ = kwargs

    def __iadd__(self, other):
        """Extend with more key/value options"""
        for key, value in other.items():
            self.__dict__[key] = value

        return self


class DataConverter(ABC):
    """
    Base preprocessing class used to preprocess AV datasets in a canonical representation as used in the Nvidia NCore-SDK project.

    For adding a new dataset, please inherit this class, implement the required functions, and register a new CLI command.

    The output data should follow the conventions defined in
    https://nrs.gitlab-master-pages.nvidia.com/ncore/notes/conventions.html

    Please also use the facilities of the 'data_writer' module, which simplifies adding new datasets.
    """

    def __init__(self, config):
        self.logger = logging.getLogger(__name__)

        self.root_dir = Path(config.root_dir)
        self.output_dir = Path(config.output_dir)

        # External sensor selection overwrites
        # Store `None`` for `_active_<sensor>_ids` in case all sensors should be used, as the
        # actual full list of sensor ids will be passed via `get_active_<sensor>_ids()`
        # at conversion time (as for some data-converters the set of sensors
        # is only available after dataset introspection)
        self._active_camera_ids = list(config.camera_ids) if len(config.camera_ids) else None
        if config.no_cameras:
            self._active_camera_ids = []

        self._active_lidar_ids = list(config.lidar_ids) if len(config.lidar_ids) else None
        if config.no_lidars:
            self._active_lidar_ids = []

        self._active_radar_ids = list(config.radar_ids) if len(config.radar_ids) else None
        if config.no_radars:
            self._active_radar_ids = []

    @staticmethod
    def _get_active_sensor_ids(
        sensor_type: str, active_sensor_ids: Optional[list[str]], all_sensor_ids: list[str]
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
    def convert(cls, config) -> None:
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
    def get_sequence_paths(config) -> list[Path]:
        """
        Return sequence pathnames to process
        """
        pass

    @staticmethod
    @abstractmethod
    def from_config(config) -> DataConverter:
        """
        Return an instance of the data converter
        """
        pass

    @abstractmethod
    def convert_sequence(self, sequence_path: Path) -> None:
        """
        Runs dataset-specific conversion for a sequence referenced by a directory/file path
        """
        pass


@click.group()
@click.option("--root-dir", type=str, help="Path to the raw data sequences", required=True)
@click.option("--output-dir", type=str, help="Path where the converted data will be saved", required=True)
@click.option("--verbose", is_flag=True, default=False, help="Enables debug logging outputs")
@click.option("--debug", is_flag=True, default=False, help="Start a debugpy remote debugging sessions to listen on")
@click.option("--no-cameras", is_flag=True, default=False, help="Disable exporting any camera sensor")
@click.option(
    "--camera-id",
    "camera_ids",
    multiple=True,
    type=str,
    help="Cameras to be exported (multiple value option, all if not specified)",
    default=None,
)
@click.option("--no-lidars", is_flag=True, default=False, help="Disable exporting any lidar sensor")
@click.option(
    "--lidar-id",
    "lidar_ids",
    multiple=True,
    type=str,
    help="Lidars to be exported (multiple value option, all if not specified)",
    default=None,
)
@click.option("--no-radars", is_flag=True, default=False, help="Disable exporting any radar sensor")
@click.option(
    "--radar-id",
    "radar_ids",
    multiple=True,
    type=str,
    help="Radars to be exported (multiple value option, all if not specified)",
    default=None,
)
@click.pass_context
def cli(ctx, *_, **kwargs):
    """Data Preprocessing Pipeline

    Source data format is selected via subcommands, for which dedicated options can be specified.

    Example invocation for 'NV maglev' data

    \b
    ./convert_raw_data.py
      --root-dir <FOLDER WITH SOURCE DATASETS>
      --output-dir <FOLDER DATA WILL BE PRODUCED>
      <your-data-variant>
    """
    # Create a config object and remember it as the context object. From
    # this point onwards other commands can refer to it by using the
    # @click.pass_context decorator.
    ctx.obj = Config(kwargs)

    # Initialize basic top-level logger configuration
    logging.basicConfig(
        level=logging.DEBUG if ctx.obj.verbose else logging.INFO,
        format="<%(asctime)s|%(levelname)s|%(filename)s:%(lineno)d|%(name)s> %(message)s",
    )

    # If the debug flag is set, add a breakpoint and wait for remote debugger
    if ctx.obj.debug:
        breakpoint()
