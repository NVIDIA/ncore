# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import annotations

import io
import logging
import sys

from dataclasses import dataclass
from enum import IntEnum, auto, unique
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, TypeVar, Union, cast

import dataclasses_json

if sys.version_info >= (3, 11):
    # Older python versions have issues with type-hints for nested types in
    # combination with typing.get_type_hints() (used by, e.g., 'dataclasses_json')
    # - alias these globally as a workaround
    from typing import Self

import concurrent
import zarr
import numpy as np
import PIL.Image as PILImage

from . import stores, util, types, data

VERSION = 4


class SequenceStoreWriter:
    """SequenceStoreWriter manages store groups for writing for NCore V4 / zarr data for a single NCore sequence"""

    def __init__(
        self,
        output_dir_path: Path,
        store_base_name: str,
        sequence_id: str,
        generic_meta_data: Dict[
            str, data.JsonLike
        ],  # generic sequence meta-data (needs to be json-serializable) - will be stored into each component store group
        # Zarr store type: either serialize as .itar archive store (default / production) or plain "directory" store (simpler for introspection / asynchronous / external setup)
        store_type: str = "itar",  # valid values: ['itar', 'directory']
        validate_on_finalize: bool = True,
    ):
        """
        Instantiate sequence store writer and initialize the default data groups and file stores for a given sequence and sensor IDs
        """

        self._output_dir_path = output_dir_path
        self._store_base_name = store_base_name

        self._sequence_id = sequence_id
        self._generic_meta_data = generic_meta_data

        # Individual stores for each group are initialized lazily on-demand (indexed tar file or zarr directories)
        self._stores_basegroups: dict[
            str, Tuple[zarr.storage.Store, Path, zarr.Group]
        ] = {}  # maps component group names to stores, store path, and base groups
        self._store_type = store_type
        self._validate_on_finalize = validate_on_finalize

        # registered component writers
        self._component_writers: dict[
            str, ComponentWriter
        ] = {}  # maps from component name to associated component writer

    def get_base_group(self, component_group_name: Optional[str]) -> zarr.Group:
        """Lazily initializes ncore base-groups and underlying stores on demand"""

        if component_group_name is None:
            # empty group name represents the default group
            component_group_name = ""

        if (store_basegroup := self._stores_basegroups.get(component_group_name)) is not None:
            # Store already initialized, return it's base group
            return store_basegroup[2]

        # Store doesn't exist yet, create it
        self._output_dir_path.mkdir(parents=True, exist_ok=True)

        # always use 'ncore4' as store name prefix
        store_name = "ncore4"
        if len(component_group_name):
            # append group name as suffix to store name if given
            store_name += f"-{component_group_name}"

        store: zarr._storage.store.Store
        if self._store_type == "itar":
            # container-based zarr stores <base-name>.<store_name>.zarr.itar
            store_path = self._output_dir_path / f"{self._store_base_name}.{store_name}.zarr.itar"
            store = stores.IndexedTarStore(store_path, mode="w")
            component_root_group_name = "ncore"
        elif self._store_type == "directory":
            # directory-based zarr stores <base-name>.<store_name>.zarr.zarr
            store_path = self._output_dir_path / f"{self._store_base_name}.{store_name}.zarr"
            store = zarr.storage.DirectoryStore(store_path)
            component_root_group_name = "ncore"
        else:
            raise ValueError(f"Unknown store type {self._store_type}")

        # Create root group in store
        root_group = zarr.group(store=store)

        # Store dataset associated meta-data to root
        root_group.attrs.put(
            {
                "sequence_id": self._sequence_id,
                "generic_meta_data": self._generic_meta_data,
                "version": VERSION,
                "component_root_group_name": component_root_group_name,  # the group name in the store's root group containing all data components
                "component_group_name": component_group_name,
            }
        )

        # Load root group based on meta-data (creating it on-demand)
        component_root_group = root_group.require_group(root_group.attrs["component_root_group_name"])

        # Create store / base-group mapping
        self._stores_basegroups[component_group_name] = store, store_path, component_root_group

        return component_root_group

    # To be called after all data was written
    def finalize(self) -> List[Path]:
        """Validates all writers and closes all stores after consolidating their meta data.

        Returns a list of the store paths
        """
        # Finalize all writers
        for component_writer in self._component_writers.values():
            component_writer.finalize()

        # Validate all writers
        if self._validate_on_finalize:
            for component_name, component_writer in self._component_writers.items():
                if not component_writer.validate():
                    raise RuntimeError(f"Validation of component writer {component_name} failed")

        # Make sure the stores are consolidated and closed
        ret = []
        for store, store_path, _ in self._stores_basegroups.values():
            stores.consolidate_compressed_metadata(store)

            # Finish writing all files
            store.close()

            ret.append(store_path)

        return ret

    def register_component_writer(
        self,
        component_writer_type: Type[CW],
        component_instance_name: str,
        group_name: Optional[str] = None,
        # generic sensor meta-data (needs to be json-serializable)
        generic_meta_data: Dict[str, data.JsonLike] = {},
    ) -> CW:
        """Instantiates a component writer instance for the given component type, component instance name, and group name.
        Additionally stores associated generic meta data"""

        assert len(component_instance_name) > 0, "Component instance name must not be empty"

        # Create component name from component base name and component instance name
        component_base_name = component_writer_type.get_component_base_name()
        component_name = f"{component_base_name}:{component_instance_name}"

        assert component_name not in self._component_writers, (
            f"Component writer for {component_name} already registered"
        )

        # Create the component in the requested group, separated by component base name
        component_group = (
            self.get_base_group(group_name).require_group(component_base_name).require_group(component_instance_name)
        )

        # Prepare meta-data
        meta_data = {
            "component_version": component_writer_type.get_component_version(),
            "generic_meta_data": generic_meta_data,
        }

        # Store meta-data
        component_group.attrs.put(meta_data)

        self._component_writers[component_name] = (component_writer := component_writer_type(component_group))

        return component_writer


class SequenceStoreReader:
    """SequenceStoreReader manages component store groups for reading for NCore V4 / zarr data for a single NCore sequence"""

    def __init__(
        self, component_store_paths: List[Path], open_consolidated: bool = True, max_threads: int | None = None
    ):
        """Initialize a SequenceStoreReader for a virtual sequence represented by a list of components.

        Args:
            component_store_paths: Paths / URLs to component stores to load, which need to represent a *single* sequence
            open_consolidated: If 'True', pre-load per-component meta-data when opening the components.
                               This is advisable if component data is accessed from *non-local*
                               storage to prevent latencies introduced when accessing the data.
                               If the component data is available on fast *local* storage, disabling
                               this option can speed up initial load times.
            max_threads:       The maximum number of threads used to load the different components (if None,
                               use interpreter-default number of threads for a ThreadPoolExecutor)
        """

        assert len(component_store_paths), "No component inputs provided"

        # Load component stores concurrently (to hide latency) and check for sequence consistency
        self._component_stores: Dict[str, zarr.Group] = {}  # use str as the generic path / URL type

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:

            def thread_load_component_store(component_store_path: Path) -> zarr.Group:
                """Thread-executed shard opening"""

                # Make sure paths are absolute at this point - in the future we might have fully-resolved URLs instead here, too
                component_store_path = component_store_path.absolute()

                logging.info(f"SequenceStoreReader: Loading component store {component_store_path}")

                component_store: zarr._storage.store.Store
                if component_store_path.is_file():
                    if not component_store_path.name.endswith(".zarr.itar"):
                        # not a supported file-based store format
                        raise RuntimeError(
                            f"Unsupported file-based store format {component_store_path}, expected .zarr.itar"
                        )

                    component_store = stores.IndexedTarStore(component_store_path, mode="r")
                else:
                    component_store = zarr.storage.DirectoryStore(component_store_path)

                component_root = (
                    stores.open_compressed_consolidated(store=component_store, mode="r")
                    if open_consolidated
                    else zarr.open(store=component_store, mode="r")
                )

                return cast(zarr.Group, component_root)

            for future in concurrent.futures.as_completed(
                [
                    executor.submit(thread_load_component_store, component_store_path)
                    for component_store_path in component_store_paths
                ]
            ):
                # Note: thread completion order is not relevant here
                component_root = future.result()

                component_root_attrs = dict(component_root.attrs.items())

                component_store_sequence_id = component_root_attrs["sequence_id"]
                component_store_generic_meta_data = component_root_attrs["generic_meta_data"]
                component_root_version = component_root_attrs["version"]

                if not self._component_stores:
                    self._sequence_id: str = component_store_sequence_id
                    self._generic_meta_data: Dict[str, data.JsonLike] = component_store_generic_meta_data
                    self._version: str = component_root_version

                if not self._sequence_id == component_store_sequence_id:
                    raise RuntimeError("Can't load component store from different sequences")
                if not self._generic_meta_data == component_store_generic_meta_data:
                    raise RuntimeError("Can't load component store with different generic meta-data")
                if not self._version == component_root_version:
                    raise RuntimeError("Can't load shards from different data versions")

                component_group_name = component_root_attrs["component_group_name"]
                if component_group_name in self._component_stores:
                    raise RuntimeError(f"Component group {component_group_name} loaded multiple times")

                self._component_stores[component_group_name] = component_root

        # Check version-compatibility
        if self._version != VERSION:
            raise ValueError(
                f"Loading incompatible version {self._version}, supporting {VERSION} only"
            )  # TODO: this check can still be refined

    @property
    def sequence_id(self) -> str:
        return self._sequence_id

    @property
    def generic_meta_data(self) -> Dict[str, data.JsonLike]:
        return self._generic_meta_data

    def open_component_readers(
        self,
        component_reader_type: Type[CR],
    ) -> Dict[str, CR]:
        """Instantiates all component readers for the given component of all associated stores, identified by the component instance names"""

        ret = {}

        for component_store in self._component_stores.values():
            component_root_group = component_store[component_store.attrs["component_root_group_name"]]

            if (component_group := component_root_group.get(component_reader_type.get_component_base_name())) is None:
                continue

            # instantiate a reader for each of the components
            for component_instance_name, component_group in component_group.items():
                assert component_instance_name not in ret, (
                    f"Component instance {component_instance_name} encountered multiple times"
                )

                # check if the reader supports the component version
                if not component_reader_type.supports_component_version(component_group.attrs["component_version"]):
                    continue

                ret[component_instance_name] = component_reader_type(component_instance_name, component_group)

        return ret


class ComponentWriter(ABC):
    @staticmethod
    @abstractmethod
    def get_component_base_name() -> str:
        """Returns the base name of the component writer"""
        ...

    @staticmethod
    @abstractmethod
    def get_component_version() -> str:
        """Returns the version of the current component writer"""
        ...

    def __init__(self, component_group: zarr.Group) -> None:
        """Initializes a component writer targeting the given component group"""
        self._group = component_group

    def finalize(self) -> None:
        """Overwrite to perform final operations after all user-data was written"""
        pass

    def validate(self) -> bool:
        """Overwrite to validates the writen data"""
        return True


class ComponentReader(ABC):
    @staticmethod
    @abstractmethod
    def get_component_base_name() -> str:
        """Returns the base name of the current component"""

    @staticmethod
    @abstractmethod
    def supports_component_version(version: str) -> bool:
        """Returns true if the component version is supported by the reader"""

    def __init__(self, component_instance_name: str, component_group: zarr.Group) -> None:
        """Initializes a component reader for a given component instance name and group"""
        self._instance_name = component_instance_name
        self._group = component_group

    @property
    def instance_name(self) -> str:
        return self._instance_name

    @property
    def component_version(self) -> str:
        """Returns the component version of the loaded component"""
        return self._group.attrs["component_version"]

    @property
    def generic_meta_data(self) -> Dict[str, data.JsonLike]:
        """Returns the generic meta data of the loaded component"""
        return self._group.attrs["generic_meta_data"]


CW = TypeVar("CW", bound=ComponentWriter)
CR = TypeVar("CR", bound=ComponentReader)


def validate_frame_name(name: str) -> str:
    """Checks if the given name is a valid frame name (non-empty, no whitespace), returns it if valid, raises AssertionError otherwise"""
    assert len(name) and not name.isspace(), f"Frame '{name}' is invalid, must not be empty or contain whitespace"

    return name


class PosesSetComponent:
    """Represents a generic set of static / dynamic poses (rigid transformations) between named coordinate frames"""

    COMPONENT_BASE_NAME: str = "poses_set"

    class Writer(ComponentWriter):
        """Poses set data component writer"""

        @staticmethod
        def get_component_base_name() -> str:
            """Returns the base name of the current component'"""
            return PosesSetComponent.COMPONENT_BASE_NAME

        @staticmethod
        def get_component_version() -> str:
            """Returns the version of the current component writer"""
            return "0.1"

        def __init__(self, component_group: zarr.Group):
            super().__init__(component_group)

            self.data: Dict = {"static_poses": {}, "dynamic_poses": {}}

        def finalize(self):
            """Actually store the json-encoded pose data"""

            self._group.create_group("static_poses").attrs.put(self.data["static_poses"])
            self._group.create_group("dynamic_poses").attrs.put(self.data["dynamic_poses"])

        def store_static_pose(
            self,
            source_frame: str,
            target_frame: str,
            pose: np.ndarray,  #: Source-to-target SE3 transformation (float64, [4,4])
        ) -> "Self":
            """Store a static pose (rigid transformation) between two named coordinate frames.

            Makes sure the inverse transformation is not already stored."""

            # Sanity checks
            assert pose.shape == (4, 4)
            assert pose.dtype == np.dtype("float64")
            assert np.all(pose[3, :] == [0.0, 0.0, 0.0, 1.0]), "Invalid SE3 transformation"

            key = f"T_{validate_frame_name(source_frame)}_{validate_frame_name(target_frame)}"
            inv_key = f"T_{target_frame}_{source_frame}"

            assert key not in self.data["static_poses"], f"Static pose {key} already exists"
            assert inv_key not in self.data["static_poses"], f"Inverse static pose {inv_key} already exists"

            self.data["static_poses"][key] = {
                "pose": pose.tolist(),
            }

            return self

        def store_dynamic_poses(
            self,
            source_frame: str,
            target_frame: str,
            poses: np.ndarray,  #: Source-to-target SE3 transformation trajectory (float64, [N,4,4])
            timestamps_us: np.ndarray,  #: All source-to-target transformation timestamps of the trajectory (uint64, [N,])
        ) -> "Self":
            """Store a trajectory of dynamic poses (time-dependent rigid transformations) between two named coordinate frames.

            Makes sure the inverse transformation is not already stored."""

            # Sanity checks
            assert poses.shape[1:] == (4, 4)
            assert poses.dtype == np.dtype("float64")
            assert np.all(poses[:, 3, :] == [0.0, 0.0, 0.0, 1.0]), "Invalid SE3 transformations"

            assert timestamps_us.ndim == 1
            assert timestamps_us.dtype == np.dtype("uint64")

            assert len(poses) == len(timestamps_us)

            key = f"T_{validate_frame_name(source_frame)}_{validate_frame_name(target_frame)}"
            inv_key = f"T_{target_frame}_{source_frame}"

            assert key not in self.data["dynamic_poses"], f"Dynamic poses {key} already exists"
            assert inv_key not in self.data["dynamic_poses"], f"Inverse dynamic poses {inv_key} already exists"

            self.data["dynamic_poses"][key] = {"poses": poses.tolist(), "timestamps_us": timestamps_us.tolist()}

            return self

    class Reader(ComponentReader):
        @staticmethod
        def get_component_base_name() -> str:
            """Returns the base name of the current component"""
            return PosesSetComponent.COMPONENT_BASE_NAME

        @staticmethod
        def supports_component_version(version: str) -> bool:
            """Returns true if the component version is supported by the reader"""
            return version == "0.1"

        def get_static_pose(self, source_frame: str, target_frame: str) -> np.ndarray:
            """Returns static pose (rigid transformation) between two named coordinate frames, if available"""

            if (
                static_pose := self._group["static_poses"].attrs.get(
                    key := f"T_{validate_frame_name(source_frame)}_{validate_frame_name(target_frame)}"
                )
            ) is None:
                raise KeyError(f"Static pose {key} not found")

            return np.array(static_pose["pose"], dtype=np.float64)

        def get_dynamic_poses(self, source_frame: str, target_frame: str) -> Tuple[np.ndarray, np.ndarray]:
            """Returns dynamic poses (time-dependent rigid transformations) between two named coordinate frames, if available"""

            if (
                dynamic_poses := self._group["dynamic_poses"].attrs.get(
                    key := f"T_{validate_frame_name(source_frame)}_{validate_frame_name(target_frame)}"
                )
            ) is None:
                raise KeyError(f"Dynamic poses {key} not found")

            return np.array(dynamic_poses["poses"], dtype=np.float64), np.array(
                dynamic_poses["timestamps_us"], dtype=np.uint64
            )


class SensorIntrinsicsComponent:
    """Sensor intrinsic calibration data component"""

    COMPONENT_BASE_NAME: str = "sensor_intrinsics"

    class Writer(ComponentWriter):
        """Sensor intrinsics data component writer"""

        @staticmethod
        def get_component_base_name() -> str:
            """Returns the base name of the current intrinsic calibration component"""
            return SensorIntrinsicsComponent.COMPONENT_BASE_NAME

        @staticmethod
        def get_component_version() -> str:
            """Returns the version of the current intrinsic calibration component"""
            return "0.1"

        def __init__(self, component_group: zarr.Group):
            super().__init__(component_group)

            self._group.create_group("cameras")
            self._group.create_group("lidars")

        def store_camera_intrinsics(
            self,
            camera_id: str,
            # intrinsics
            camera_model_parameters: types.ConcreteCameraModelParametersUnion,
            # sensor constants
            mask_image: Optional[PILImage.Image],
        ) -> "Self":
            """Store camera-associated intrinsics"""

            # Prepare meta-data containing the serialization of the mandatory camera model / optional external distortion parameters

            meta_data = data.encode_camera_model_parameters(camera_model_parameters)

            (camera_grp := self._group["cameras"].create_group(camera_id)).attrs.put(meta_data)

            # Store mask if available
            if mask_image:
                with io.BytesIO() as buffer:
                    FORMAT = "png"
                    mask_image.save(buffer, format=FORMAT, optimize=True)  # encodes as png
                    camera_grp.create_dataset("mask", data=np.asarray(buffer.getvalue()), compressor=None).attrs[
                        "format"
                    ] = FORMAT

            return self

        def store_lidar_intrinsics(
            self,
            lidar_id: str,
            # intrinsics
            lidar_model_parameters: types.ConcreteLidarModelParametersUnion,
        ) -> "Self":
            """Store lidar-associated intrinsics"""

            # Prepare meta-data containing the serialization of the mandatory lidar model
            meta_data = data.encode_lidar_model_parameters(lidar_model_parameters)

            self._group["lidars"].create_group(lidar_id).attrs.put(meta_data)

            return self

    class Reader(ComponentReader):
        """Sensor intrinsics data component writer"""

        @staticmethod
        def get_component_base_name() -> str:
            """Returns the base name of the current component"""
            return SensorIntrinsicsComponent.COMPONENT_BASE_NAME

        @staticmethod
        def supports_component_version(version: str) -> bool:
            """Returns true if the component version is supported by the reader"""
            return version == "0.1"

        def get_camera_model_parameters(self, camera_id: str) -> types.ConcreteCameraModelParametersUnion:
            """Returns the camera model associated with the requested camera sensor"""
            return data.decode_camera_model_parameters(self._group["cameras"][camera_id].attrs)

        def get_camera_mask_image(self, camera_id: str) -> Optional[PILImage.Image]:
            """Returns constant camera mask image, if available"""

            if (mask_dataset := self._group["cameras"][camera_id].get("mask", default=None)) is None:
                return None

            return PILImage.open(io.BytesIO(mask_dataset[()]), formats=[mask_dataset.attrs["format"]])

        def get_lidar_model_parameters(self, lidar_id: str) -> types.ConcreteLidarModelParametersUnion:
            """Returns the lidar model associated with the requested lidar sensor"""
            return data.decode_lidar_model_parameters(self._group["lidars"][lidar_id].attrs)


class BaseSensorComponentWriter(ComponentWriter):
    """Base class for all sensor component writers"""

    def __init__(self, component_group: zarr.Group):
        super().__init__(component_group)

        self._group.create_group("frames")

    def finalize(self):
        """Perform final operations after all user-data was written to the sensor component"""

        # Collect all frame timestamps to be stored as global property (supporting no frames at all and out-of-order frames)
        frames_timestamps_us = [
            self._group["frames"][frame_grp]["timestamps_us"][...] for frame_grp in sorted(self._group["frames"].keys())
        ]
        frames_timestamps_us = np.array(frames_timestamps_us, dtype=np.uint64).reshape((-1, 2))

        # Validate all start/end-of-frame timestamps to be monotonically increasing
        assert np.all(frames_timestamps_us[:-1, 0] < frames_timestamps_us[1:, 0]), (
            "Start of frame timestamps are not monotonically increasing"
        )
        assert np.all(frames_timestamps_us[:-1, 1] < frames_timestamps_us[1:, 1]), (
            "End of frame timestamps are not monotonically increasing"
        )

        # Store as global property
        self._group.create_dataset("frames_timestamps_us", data=frames_timestamps_us)

    def _get_frame_group(
        self,
        # end-of-frame timestamp, or start-of-frame / end-of-frame timestamps
        timestamps_us: Union[int, np.ndarray],
    ) -> zarr.Group:
        """Returns the group of a frame, initializing it if required"""

        if isinstance(timestamps_us, np.ndarray):
            frame_id = timestamps_us[1].item()  # end-of-frame timestamp is frame ID
        else:
            frame_id = timestamps_us

        return self._group["frames"].require_group(util.padded_index_string(frame_id, index_digits=20))

    def _store_base_frame(
        self,
        # start-of-frame / end-of-frame timestamps
        timestamps_us: np.ndarray,
        # generic per-frame data (key-value pairs, *not* dimension / dtype validated) and meta-data
        generic_data: Dict[str, np.ndarray],
        generic_meta_data: Dict[str, data.JsonLike],
    ) -> zarr.Group:
        # Sanity / consistency checks
        assert np.shape(timestamps_us) == (2,)
        assert timestamps_us.dtype == np.dtype("uint64")
        assert timestamps_us[1] >= timestamps_us[0]

        # Initialize frame group
        frame_group = self._get_frame_group(timestamps_us)

        # Store pose data
        frame_group.create_dataset("timestamps_us", data=timestamps_us)

        # Store additional generic frame data and meta-data (not dimension / dtype checked)
        (frame_generic_data_group := frame_group.create_group("generic_data")).attrs.put(generic_meta_data)
        for name, value in generic_data.items():
            frame_generic_data_group.create_dataset(name, data=value)

        return frame_group


class BaseSensorComponentReader(ComponentReader):
    """Base class for all sensor component readers"""

    def _get_frame_group(
        self,
        # end-of-frame timestamp, or start-of-frame / end-of-frame timestamps
        timestamps_us: Union[int, np.ndarray],
    ) -> zarr.Group:
        """Returns the group of a frame"""

        if isinstance(timestamps_us, np.ndarray):
            frame_id = timestamps_us[1].item()  # end-of-frame timestamp is frame ID
        else:
            frame_id = timestamps_us

        return self._group["frames"][util.padded_index_string(frame_id, index_digits=20)]

    @property
    def frames_timestamps_us(self) -> np.ndarray:
        return np.array(self._group["frames_timestamps_us"])

    def get_frame_timestamps_us(self, timestamp_us: int) -> np.ndarray:
        return np.array(self._get_frame_group(timestamp_us)["timestamps_us"])

    # Generic per-frame data
    def get_frame_generic_data_names(self, timestamp_us: int) -> List[str]:
        """List of all generic frame-data names"""

        return list(self._get_frame_group(timestamp_us)["generic_data"].keys())

    def has_frame_generic_data(self, timestamp_us: int, name: str) -> bool:
        """Signals if named generic frame-data exists"""

        return name in self.get_frame_generic_data_names(timestamp_us)

    def get_frame_generic_data(self, timestamp_us: int, name: str) -> np.ndarray:
        """Returns generic frame-data for a specific frame and name"""

        return np.array(self._get_frame_group(timestamp_us)["generic_data"][name])

    def get_frame_generic_meta_data(self, timestamp_us: int) -> Dict[str, data.JsonLike]:
        generic_data_group = self._get_frame_group(timestamp_us)["generic_data"]
        return dict(generic_data_group.attrs)


class BasePointCloudSensorComponentWriter(BaseSensorComponentWriter):
    """Base class for all point cloud sensor component writers"""

    def _store_frame_point_cloud(
        self,
        # start-of-frame / end-of-frame timestamps
        timestamps_us: np.ndarray,
        # point-cloud components - need to have same lenght consistent with point count dimension
        point_count: int,
        point_cloud_data: Dict[str, np.ndarray],
    ) -> zarr.Group:
        # Initialize point cloud group
        (point_cloud_group := self._get_frame_group(timestamps_us).create_group("point_cloud")).attrs.put(
            {"point_count": point_count}
        )

        # Store point cloud components
        for name, data in point_cloud_data.items():
            assert len(data) == point_count, f"{name} doesn't have required point count"

            point_cloud_group.create_dataset(name, data=data)

        return point_cloud_group


class BasePointCloudSensorComponentReader(BaseSensorComponentReader):
    """Base class for all point cloud sensor component readers"""

    def get_frame_point_cloud_size(self, timestamp_us: int) -> int:
        """Returns the number of point cloud points for a specific frame"""

        return self._get_frame_group(timestamp_us)["point_cloud"].attrs["point_count"]

    def get_frame_point_cloud_data_names(self, timestamp_us: int) -> List[str]:
        """List of all point cloud data names for a frame"""

        return list(self._get_frame_group(timestamp_us)["point_cloud"].keys())

    def has_frame_point_cloud_data(self, timestamp_us: int, name: str) -> bool:
        """Signals if named point cloud data exists for a frame"""

        return name in self._get_frame_group(timestamp_us)["point_cloud"].keys()

    def get_frame_point_cloud_data(self, timestamp_us: int, name: str) -> np.ndarray:
        """Returns named point cloud data for a frame"""

        return np.array(self._get_frame_group(timestamp_us)["point_cloud"][name])


class CameraSensorComponent:
    """Camera sensor data component"""

    COMPONENT_BASE_NAME: str = "camera_sensor"

    class Writer(BaseSensorComponentWriter):
        """Camera sensor data component writer"""

        @staticmethod
        def get_component_base_name() -> str:
            """Returns the base name of the current camera sensor component"""
            return CameraSensorComponent.COMPONENT_BASE_NAME

        @staticmethod
        def get_component_version() -> str:
            """Returns the version of the current camera sensor component"""
            return "0.1"

        def store_frame(
            self,
            # image data
            image_binary_data: bytes,
            image_format: str,
            # start-of-frame / end-of-frame timestamps
            timestamps_us: np.ndarray,
            # generic per-frame data (key-value pairs, *not* dimension / dtype validated) and meta-data
            generic_data: Dict[str, np.ndarray],
            generic_meta_data: Dict[str, data.JsonLike],
        ) -> "Self":
            # Initialize frame
            frame_group = self._store_base_frame(timestamps_us, generic_data, generic_meta_data)

            # Store image data
            frame_group.create_dataset("image", data=np.asarray(image_binary_data), compressor=None).attrs["format"] = (
                image_format
            )

            return self

    class Reader(BaseSensorComponentReader):
        """Camera sensor data component reader"""

        @staticmethod
        def get_component_base_name() -> str:
            """Returns the base name of the current component"""
            return CameraSensorComponent.COMPONENT_BASE_NAME

        @staticmethod
        def supports_component_version(version: str) -> bool:
            """Returns true if the component version is supported by the reader"""
            return version == "0.1"
        class EncodedImageDataHandle:
            """References encoded image data without loading it"""

            def __init__(self, image_dataset: zarr.Array):
                self._image_dataset = image_dataset

            def get_data(self) -> types.EncodedImageData:
                """Loads the referenced encoded image data to memory"""
                return types.EncodedImageData(self._image_dataset[()], self._image_dataset.attrs["format"])

        def get_frame_handle(self, timestamp_us: int) -> EncodedImageDataHandle:
            """Returns the frame's encoded image data"""
            return self.EncodedImageDataHandle(self._get_frame_group(timestamp_us)["image"])

        def get_frame_data(self, timestamp_us: int) -> types.EncodedImageData:
            """Returns the frame's encoded image data"""
            return self.get_frame_handle(timestamp_us).get_data()

        def get_frame_image(self, timestamp_us: int) -> PILImage.Image:
            """Returns the frame's decoded image data"""
            return self.get_frame_data(timestamp_us).get_decoded_image()


class LidarSensorComponent:
    """Lidar sensor data component"""

    COMPONENT_BASE_NAME: str = "lidar_sensor"

    class Writer(BasePointCloudSensorComponentWriter):
        """Lidar sensor data component writer"""

        @staticmethod
        def get_component_base_name() -> str:
            """Returns the base name of the current lidar sensor component"""
            return LidarSensorComponent.COMPONENT_BASE_NAME

        @staticmethod
        def get_component_version() -> str:
            """Returns the version of the current lidar sensor component"""
            return "0.1"

        def store_frame(
            self,
            # mandatory point-cloud data
            xyz_m: np.ndarray,  # per-point metric coordinates relative to the sensor frame at measure time (raw / not motion-compensated, needs to be non-zero) (float32, [n, 3])
            intensity: np.ndarray,  # per-point intensity normalized to [0.0, 1.0] range (float32, [n])
            timestamp_us: np.ndarray,  # per-point point timestamp in microseconds (uint64, [n])
            model_element: Optional[np.ndarray],  # per-point model element indices, if applicable (uint16, [n, 2])
            # start-of-frame / end-of-frame timestamps
            timestamps_us: np.ndarray,
            # generic per-frame data (key-value pairs, *not* dimension / dtype validated) and meta-data
            generic_data: Dict[str, np.ndarray],
            generic_meta_data: Dict[str, data.JsonLike],
        ) -> "Self":
            # Sanity / consistency checks
            assert xyz_m.ndim == 2
            assert np.shape(xyz_m)[1] == 3
            assert xyz_m.dtype == np.dtype("float32")
            point_count = len(xyz_m)

            # Initialize frame
            self._store_base_frame(timestamps_us, generic_data, generic_meta_data)

            point_cloud_data = {
                "xyz_m": xyz_m,
            }

            assert intensity.dtype == np.dtype("float32")
            assert 0.0 <= intensity.min() and intensity.max() <= 1.0, "Intensity not normalized"
            point_cloud_data["intensity"] = intensity

            assert timestamp_us.dtype == np.dtype("uint64")
            if point_count:
                assert (timestamps_us[0] <= timestamp_us.min()) and (timestamp_us.max() <= timestamps_us[1]), (
                    "Point timestamps outside frame time bounds"
                )
            point_cloud_data["timestamp_us"] = timestamp_us

            if model_element is not None:
                assert model_element.shape == (point_count, 2)
                assert model_element.dtype == np.dtype("uint16")
                point_cloud_data["model_element"] = model_element

            # Store point-cloud data
            self._store_frame_point_cloud(timestamps_us, point_count, point_cloud_data)

            return self

    class Reader(BasePointCloudSensorComponentReader):
        """Lidar sensor data component reader"""

        @staticmethod
        def get_component_base_name() -> str:
            """Returns the base name of the current component"""
            return LidarSensorComponent.COMPONENT_BASE_NAME

        @staticmethod
        def supports_component_version(version: str) -> bool:
            """Returns true if the component version is supported by the reader"""
            return version == "0.1"


# TODO: move to types once stable


@dataclass
class CuboidTrack(dataclasses_json.DataClassJsonMixin):
    """Cuboid track instance"""

    @dataclass
    class Observation(dataclasses_json.DataClassJsonMixin):
        """Individual cuboid track observation relative to the reference frame"""

        observation_id: str  #: Identifier of the current observation (unique among all observations)
        timestamp_us: (
            int  #: The timestamp associated with the centroid of the observation (possibly an accurate in-frame time)
        )
        reference_frame_timestamp_us: int  #: The timestamp of the reference frame
        bbox3: (
            types.BBox3
        )  #: Bounding-box coordinates of the object relative to the reference frame's coordinate system

        def __post_init__(self):
            # Sanity checks
            assert isinstance(self.observation_id, str)
            assert isinstance(self.reference_frame_timestamp_us, int)
            assert isinstance(self.bbox3, types.BBox3)
            assert isinstance(self.timestamp_us, int)

    track_id: str  #: Unique identifier of the object's track this observation is associated with
    label_class: str  #: String-representation of the labeled class associated with this observation
    reference_frame_name: str  #: String-identifier of the reference frame (e.g., sensor name)
    observations: List[Observation]  #: All observations associated with this track
    source: types.LabelSource = util.enum_field(types.LabelSource)  #: The source for the current label
    source_version: Optional[str] = (
        None  #: If provided, the unique version ID of the source for the current label (to distinguish between different versions of the same source)
    )

    def __post_init__(self):
        # Sanity checks
        assert isinstance(self.track_id, str)
        assert isinstance(self.label_class, str)
        assert isinstance(self.reference_frame_name, str)

        if not isinstance(self.source, types.LabelSource):
            self.source = types.LabelSource(self.source)
        assert self.source in types.LabelSource.__members__.values()

        assert isinstance(self.source_version, (type(None), str))

        assert isinstance(self.observations, List)


class CuboidTracksComponent:
    """Data component representing cuboid tracks"""

    COMPONENT_BASE_NAME: str = "cuboid_tracks"

    class Writer(ComponentWriter):
        """Cuboid tracks component writer"""

        @staticmethod
        def get_component_base_name() -> str:
            """Returns the base name of the current lidar sensor component"""
            return CuboidTracksComponent.COMPONENT_BASE_NAME

        @staticmethod
        def get_component_version() -> str:
            """Returns the version of the current lidar sensor component"""
            return "0.1"

        def store_tracks(
            self,
            cuboid_tracks: List[CuboidTrack],  # individual track instances
        ) -> "Self":
            self._group.create_group("cuboid_tracks").attrs.put(
                {"cuboid_tracks": [track.to_dict() for track in cuboid_tracks]}
            )

            return self

    class Reader(ComponentReader):
        """Cuboid tracks component reader"""

        @staticmethod
        def get_component_base_name() -> str:
            """Returns the base name of the current component"""
            return CuboidTracksComponent.COMPONENT_BASE_NAME

        @staticmethod
        def supports_component_version(version: str) -> bool:
            """Returns true if the component version is supported by the reader"""
            return version == "0.1"

        def get_tracks(self) -> List[CuboidTrack]:
            """Returns all stored cuboid tracks"""

            return [CuboidTrack.from_dict(track) for track in self._group["cuboid_tracks"].attrs["cuboid_tracks"]]
