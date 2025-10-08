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

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Generator, List, Literal, Optional, Tuple, Type, TypeVar, Union, cast

from numcodecs import Blosc

from ncore.impl.common.common import HalfClosedInterval


if sys.version_info >= (3, 11):
    # Older python versions have issues with type-hints for nested types in
    # combination with typing.get_type_hints() (used by, e.g., 'dataclasses_json')
    # - alias these globally as a workaround
    from typing import Self

import concurrent

import numpy as np
import PIL.Image as PILImage
import zarr

from ncore.impl.data import data, stores, types

from .types import CuboidTrackObservation


VERSION = "4.0"


class SequenceComponentStoreWriter:
    """SequenceComponentWriter manages store groups for writing for NCore V4 / zarr data components for a single NCore sequence"""

    def __init__(
        self,
        output_dir_path: Path,
        store_base_name: str,
        # Identifier of the sequence
        sequence_id: str,
        # The time range for the sequence
        sequence_timestamp_interval_us: HalfClosedInterval,
        # Generic sequence meta-data (needs to be json-serializable) - will be stored into each component store group
        generic_meta_data: Dict[str, data.JsonLike],
        # Zarr store type: either serialize as .itar archive store (default / production) or plain "directory" store (simpler for introspection / asynchronous / external setup)
        store_type: Literal["itar", "directory"] = "itar",  # valid values: ['itar', 'directory']
    ):
        """
        Instantiate sequence store writer and initialize the default data groups and file stores for a given sequence and sensor IDs
        """

        self._output_dir_path = output_dir_path
        self._store_base_name = store_base_name

        self._sequence_id = sequence_id
        self._sequence_timestamp_interval_us = sequence_timestamp_interval_us
        self._generic_meta_data = generic_meta_data

        # Individual stores for each group are initialized lazily on-demand (indexed tar file or zarr directories)
        self._stores_rootgroups: dict[
            str, Tuple[zarr.Group, Path]
        ] = {}  # maps component group names to stores, store path, and base groups
        self._store_type = store_type

        # registered component writers
        self._component_writers: dict[
            str, ComponentWriter
        ] = {}  # maps from component id to associated component writer

    def get_base_group(self, component_group_name: Optional[str]) -> zarr.Group:
        """Lazily initializes ncore base-groups and underlying stores on demand"""

        if component_group_name is None:
            # empty group name represents the default group
            component_group_name = ""

        if (store_rootgroup := self._stores_rootgroups.get(component_group_name)) is not None:
            # Store already initialized, return it's root group
            return store_rootgroup[0]

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
        elif self._store_type == "directory":
            # directory-based zarr stores <base-name>.<store_name>.zarr.zarr
            store_path = self._output_dir_path / f"{self._store_base_name}.{store_name}.zarr"
            store = zarr.storage.DirectoryStore(store_path)
        else:
            raise ValueError(f"Unknown store type {self._store_type}")

        # Create root group in store
        root_group = zarr.group(store=store)

        # Store dataset associated meta-data to root
        root_group.attrs.put(
            {
                "sequence_id": self._sequence_id,
                "sequence_timestamp_interval_us": {
                    "start": self._sequence_timestamp_interval_us.start,
                    "stop": self._sequence_timestamp_interval_us.stop,
                },
                "generic_meta_data": self._generic_meta_data,
                "version": VERSION,
                "component_group_name": component_group_name,
            }
        )

        # Create store / base-group mapping
        self._stores_rootgroups[component_group_name] = root_group, store_path

        return root_group

    # To be called after all data was written
    def finalize(self) -> List[Path]:
        """Validates all writers and closes all stores after consolidating their meta data.

        Returns a list of the store paths
        """
        # Finalize all writers
        for component_writer in self._component_writers.values():
            component_writer.finalize()

        # Make sure the stores are consolidated and closed
        ret = []
        for root_group, store_path in self._stores_rootgroups.values():
            store = root_group.store

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
        component_base_name = component_writer_type.get_component_name()
        component_id = f"{component_base_name}:{component_instance_name}"

        assert component_id not in self._component_writers, f"Component writer for {component_id} already registered"

        # Create the component in the requested group, separated by component base name
        component_group = (
            self.get_base_group(group_name).require_group(component_base_name).require_group(component_instance_name)
        )

        # Prepare meta-data
        meta_data = {
            "component_name": component_base_name,
            "component_instance_name": component_instance_name,
            "component_version": component_writer_type.get_component_version(),
            "generic_meta_data": generic_meta_data,
        }

        # Store meta-data
        component_group.attrs.put(meta_data)

        self._component_writers[component_id] = (
            component_writer := component_writer_type(component_group, self._sequence_timestamp_interval_us)
        )

        return component_writer


class SequenceComponentStoreReader:
    """SequenceComponentReader manages data component store groups for reading for NCore V4 / zarr data for a single NCore sequence"""

    def __init__(
        self, component_store_paths: List[Path], open_consolidated: bool = True, max_threads: int | None = None
    ):
        """Initialize a SequenceComponentReader for a virtual sequence represented by a list of components.

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
                component_store_sequence_timestamp_interval_us = HalfClosedInterval(
                    component_root_attrs["sequence_timestamp_interval_us"]["start"],
                    component_root_attrs["sequence_timestamp_interval_us"]["stop"],
                )
                component_store_generic_meta_data = component_root_attrs["generic_meta_data"]
                component_root_version = component_root_attrs["version"]

                if not self._component_stores:
                    self._sequence_id: str = component_store_sequence_id
                    self._sequence_timestamp_interval_us = component_store_sequence_timestamp_interval_us
                    self._generic_meta_data: Dict[str, data.JsonLike] = component_store_generic_meta_data
                    self._version: str = component_root_version

                if not self._sequence_id == component_store_sequence_id:
                    raise RuntimeError("Can't load component store from different sequences")
                if not self._sequence_timestamp_interval_us == component_store_sequence_timestamp_interval_us:
                    raise RuntimeError("Can't load component store with different sequence timestamp intervals")
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
            raise ValueError(f"Loading incompatible version {self._version}, supporting {VERSION} only")

    @property
    def sequence_id(self) -> str:
        return self._sequence_id

    @property
    def sequence_timestamp_interval_us(self) -> HalfClosedInterval:
        return self._sequence_timestamp_interval_us

    @property
    def generic_meta_data(self) -> Dict[str, data.JsonLike]:
        return self._generic_meta_data

    def open_component_readers(
        self,
        component_reader_type: Type[CR],
    ) -> Dict[str, CR]:
        """Instantiates all component readers for the given component of all associated stores, identified by the component instance names"""

        ret = {}

        for component_root_group in self._component_stores.values():
            if (component_group := component_root_group.get(component_reader_type.get_component_name())) is None:
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
    def get_component_name() -> str:
        """Returns the base name of the component writer"""
        ...

    @staticmethod
    @abstractmethod
    def get_component_version() -> str:
        """Returns the version of the current component writer"""
        ...

    def __init__(self, component_group: zarr.Group, sequence_timestamp_interval_us: HalfClosedInterval) -> None:
        """Initializes a component writer targeting the given component group and sequence time interval"""
        self._group = component_group
        self._sequence_timestamp_interval_us = sequence_timestamp_interval_us

    def finalize(self) -> None:
        """Overwrite to perform final operations after all user-data was written"""
        pass


class ComponentReader(ABC):
    @staticmethod
    @abstractmethod
    def get_component_name() -> str:
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


class PosesComponent:
    """Represents a generic set of static / dynamic poses (rigid transformations) between named coordinate frames"""

    COMPONENT_NAME: str = "poses"

    class Writer(ComponentWriter):
        """Poses data component writer"""

        @staticmethod
        def get_component_name() -> str:
            """Returns the base name of the current component'"""
            return PosesComponent.COMPONENT_NAME

        @staticmethod
        def get_component_version() -> str:
            """Returns the version of the current component writer"""
            return "0.1"

        def __init__(self, component_group: zarr.Group, sequence_timestamp_interval_us: HalfClosedInterval) -> None:
            """Initializes the current component writer targeting the given component group and sequence time interval"""
            super().__init__(component_group, sequence_timestamp_interval_us)

            self.data: Dict = {"static_poses": {}, "dynamic_poses": {}}

        def finalize(self):
            """Actually store the json-encoded pose data"""

            self._group.create_group("static_poses").attrs.put(self.data["static_poses"])
            self._group.create_group("dynamic_poses").attrs.put(self.data["dynamic_poses"])

        def store_static_pose(
            self,
            source_frame_id: str,
            target_frame_id: str,
            pose: np.ndarray,  #: Source-to-target SE3 transformation (float32/64, [4,4])
        ) -> "Self":
            """Store a static pose (rigid transformation) between two named coordinate frames.

            Makes sure the inverse transformation is not already stored."""

            # Sanity checks
            assert pose.shape == (4, 4)
            assert np.issubdtype(pose.dtype, np.floating), "Poses must be of float type"
            assert np.all(pose[3, :] == [0.0, 0.0, 0.0, 1.0]), "Invalid SE3 transformation"

            key = (validate_frame_name(source_frame_id), validate_frame_name(target_frame_id))
            inv_key = key[::-1]

            assert key not in self.data["static_poses"], f"Static pose {key} already exists"
            assert inv_key not in self.data["static_poses"], f"Inverse static pose {inv_key} already exists"

            self.data["static_poses"][str(key)] = {"pose": pose.tolist(), "dtype": str(pose.dtype)}

            return self

        def store_dynamic_pose(
            self,
            source_frame_id: str,
            target_frame_id: str,
            poses: np.ndarray,  #: Source-to-target SE3 transformation trajectory (float32/64, [N,4,4])
            timestamps_us: np.ndarray,  #: All source-to-target transformation timestamps of the trajectory (uint64, [N,])
        ) -> "Self":
            """Store a trajectory of dynamic poses (time-dependent rigid transformations) between two named coordinate frames.

            Makes sure the inverse transformation is not already stored."""

            # Sanity / timestamp consistency checks
            assert poses.shape[1:] == (4, 4)
            assert np.issubdtype(poses.dtype, np.floating), "Poses must be of float type"
            assert np.all(poses[:, 3, :] == [0.0, 0.0, 0.0, 1.0]), "Invalid SE3 transformations"

            assert timestamps_us.ndim == 1
            assert timestamps_us.dtype == np.dtype("uint64")

            assert len(poses) == len(timestamps_us)

            assert len(poses) > 1, "At least two poses required for a dynamic pose trajectory to support interpolation"
            assert self._sequence_timestamp_interval_us in HalfClosedInterval(
                timestamps_us[0].item(), timestamps_us[-1].item() + 1
            ), "Dynamic poses samples must be fully contained in the sequence time range"

            key = (validate_frame_name(source_frame_id), validate_frame_name(target_frame_id))
            inv_key = key[::-1]

            assert key not in self.data["dynamic_poses"], f"Dynamic poses {key} already exists"
            assert inv_key not in self.data["dynamic_poses"], f"Inverse dynamic poses {inv_key} already exists"

            self.data["dynamic_poses"][str(key)] = {
                "poses": poses.tolist(),
                "timestamps_us": timestamps_us.tolist(),
                "dtype": str(poses.dtype),
            }

            return self

    class Reader(ComponentReader):
        @staticmethod
        def get_component_name() -> str:
            """Returns the base name of the current component"""
            return PosesComponent.COMPONENT_NAME

        @staticmethod
        def supports_component_version(version: str) -> bool:
            """Returns true if the component version is supported by the reader"""
            return version == "0.1"

        def get_static_poses(self) -> Generator[Tuple[Tuple[str, str], np.ndarray]]:
            """Returns all static poses (rigid transformations) between named coordinate frames, if available"""

            for key, static_pose in self._group["static_poses"].attrs.items():
                yield eval(key), np.array(static_pose["pose"], dtype=static_pose["dtype"])

        def get_dynamic_poses(self) -> Generator[Tuple[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]]:
            """Returns all dynamic poses (time-dependent rigid transformations) between named coordinate frames, if available"""

            for key, dynamic_poses in self._group["dynamic_poses"].attrs.items():
                yield (
                    eval(key),
                    (
                        np.array(dynamic_poses["poses"], dtype=dynamic_poses["dtype"]),
                        np.array(dynamic_poses["timestamps_us"], dtype=np.uint64),
                    ),
                )

        def get_static_pose(self, source_frame_id: str, target_frame_id: str) -> np.ndarray:
            """Returns static pose (rigid transformation) between two named coordinate frames, if available"""

            if (
                static_pose := self._group["static_poses"].attrs.get(
                    key := str((validate_frame_name(source_frame_id), validate_frame_name(target_frame_id)))
                )
            ) is None:
                raise KeyError(f"Static pose {key} not found")

            return np.array(static_pose["pose"], dtype=np.float64)

        def get_dynamic_pose(self, source_frame_id: str, target_frame_id: str) -> Tuple[np.ndarray, np.ndarray]:
            """Returns dynamic poses (time-dependent rigid transformations) between two named coordinate frames, if available"""

            if (
                dynamic_poses := self._group["dynamic_poses"].attrs.get(
                    key := str((validate_frame_name(source_frame_id), validate_frame_name(target_frame_id)))
                )
            ) is None:
                raise KeyError(f"Dynamic poses {key} not found")

            return np.array(dynamic_poses["poses"], dtype=np.float64), np.array(
                dynamic_poses["timestamps_us"], dtype=np.uint64
            )


class IntrinsicsComponent:
    """Sensor intrinsic calibration data component"""

    COMPONENT_NAME: str = "intrinsics"

    class Writer(ComponentWriter):
        """Sensor intrinsics data component writer"""

        @staticmethod
        def get_component_name() -> str:
            """Returns the base name of the current intrinsic calibration component"""
            return IntrinsicsComponent.COMPONENT_NAME

        @staticmethod
        def get_component_version() -> str:
            """Returns the version of the current intrinsic calibration component"""
            return "0.1"

        def __init__(self, component_group: zarr.Group, sequence_timestamp_interval_us: HalfClosedInterval) -> None:
            """Initializes the current component writer targeting the given component group and sequence time interval"""
            super().__init__(component_group, sequence_timestamp_interval_us)

            self._group.create_group("cameras")
            self._group.create_group("lidars")

        def store_camera_intrinsics(
            self,
            camera_id: str,
            # intrinsics
            camera_model_parameters: types.ConcreteCameraModelParametersUnion,
        ) -> "Self":
            """Store camera-associated intrinsics"""

            # Prepare meta-data containing the serialization of the mandatory camera model / optional external distortion parameters

            meta_data = data.encode_camera_model_parameters(camera_model_parameters)

            self._group["cameras"].create_group(camera_id).attrs.put(meta_data)

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
        """Sensor intrinsics data component reader"""

        @staticmethod
        def get_component_name() -> str:
            """Returns the base name of the current component"""
            return IntrinsicsComponent.COMPONENT_NAME

        @staticmethod
        def supports_component_version(version: str) -> bool:
            """Returns true if the component version is supported by the reader"""
            return version == "0.1"

        def get_camera_model_parameters(self, camera_id: str) -> types.ConcreteCameraModelParametersUnion:
            """Returns the camera model associated with the requested camera sensor"""
            return data.decode_camera_model_parameters(self._group["cameras"][camera_id].attrs)

        def get_lidar_model_parameters(self, lidar_id: str) -> types.ConcreteLidarModelParametersUnion:
            """Returns the lidar model associated with the requested lidar sensor"""
            return data.decode_lidar_model_parameters(self._group["lidars"][lidar_id].attrs)


class MasksComponent:
    """Sensor masks data component"""

    COMPONENT_NAME: str = "masks"

    class Writer(ComponentWriter):
        """Sensor masks data component writer"""

        @staticmethod
        def get_component_name() -> str:
            """Returns the base name of the current sensor masks component"""
            return MasksComponent.COMPONENT_NAME

        @staticmethod
        def get_component_version() -> str:
            """Returns the version of the current sensor masks component"""
            return "0.1"

        def __init__(self, component_group: zarr.Group, sequence_timestamp_interval_us: HalfClosedInterval) -> None:
            """Initializes the current component writer targeting the given component group and sequence time interval"""
            super().__init__(component_group, sequence_timestamp_interval_us)

            self._group.create_group("cameras")

        def store_camera_masks(
            self,
            camera_id: str,
            # named camera sensor masks
            mask_images: Dict[str, PILImage.Image],
        ) -> "Self":
            """Store camera-associated masks"""

            # Store mask names
            (camera_grp := self._group["cameras"].create_group(camera_id)).attrs.put(
                {"mask_names": list(mask_images.keys())}
            )

            # Store mask images
            for mask_name, mask_image in mask_images.items():
                with io.BytesIO() as buffer:
                    FORMAT = "png"
                    mask_image.save(buffer, format=FORMAT, optimize=True)  # encodes as png
                    # store mask data (uncompressed, as already encoded)
                    camera_grp.create_dataset(mask_name, data=np.asarray(buffer.getvalue()), compressor=None).attrs[
                        "format"
                    ] = FORMAT

            return self

    class Reader(ComponentReader):
        """Sensor masks data component reader"""

        @staticmethod
        def get_component_name() -> str:
            """Returns the base name of the current component"""
            return MasksComponent.COMPONENT_NAME

        @staticmethod
        def supports_component_version(version: str) -> bool:
            """Returns true if the component version is supported by the reader"""
            return version == "0.1"

        def get_camera_mask_names(self, camera_id: str) -> List[str]:
            """Returns all constant camera mask names"""

            return list(self._group["cameras"][camera_id].attrs.get("mask_names", []))

        def get_camera_mask_image(self, camera_id: str, mask_name: str) -> PILImage.Image:
            """Returns constant named camera mask image"""

            mask_dataset = self._group["cameras"][camera_id][mask_name]

            return PILImage.open(io.BytesIO(mask_dataset[()]), formats=[mask_dataset.attrs["format"]])

        def get_camera_mask_images(self, camera_id: str) -> Generator[Tuple[str, PILImage.Image]]:
            """Returns all constant named camera mask images"""

            for mask_name in self.get_camera_mask_names(camera_id):
                yield mask_name, self.get_camera_mask_image(camera_id, mask_name)


class BaseSensorComponentWriter(ComponentWriter):
    """Base class for all sensor component writers"""

    def __init__(self, component_group: zarr.Group, sequence_timestamp_interval_us: HalfClosedInterval) -> None:
        """Initializes the current component writer targeting the given component group and sequence time interval"""
        super().__init__(component_group, sequence_timestamp_interval_us)

        self._group.create_group("frames")

        self._frames_timestamps_us: Dict[
            int, int
        ] = {}  # collect end-of-frame timestamps mapping to start of frame timestamps

    def finalize(self):
        """Perform final operations after all user-data was written to the sensor component"""

        # Collect all frame timestamps to be stored as global property (supporting no frames at all and out-of-order frames)
        frames_timestamps_us = np.array(
            [(self._frames_timestamps_us[end], end) for end in sorted(self._frames_timestamps_us.keys())],
            dtype=np.uint64,
        ).reshape((-1, 2))

        # Validate all start/end-of-frame timestamps to be monotonically increasing
        assert np.all(frames_timestamps_us[:-1, 0] < frames_timestamps_us[1:, 0]), (
            "Start of frame timestamps are not monotonically increasing"
        )
        assert np.all(frames_timestamps_us[:-1, 1] < frames_timestamps_us[1:, 1]), (
            "End of frame timestamps are not monotonically increasing"
        )

        # Store as meta-data of frames group
        self._group["frames"].attrs.put({"frames_timestamps_us": frames_timestamps_us.tolist()})

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

        return self._group["frames"].require_group(str(frame_id))

    def _store_base_frame(
        self,
        # start-of-frame / end-of-frame timestamps
        frame_timestamps_us: np.ndarray,
        # generic per-frame data (key-value pairs, *not* dimension / dtype validated) and meta-data
        generic_data: Dict[str, np.ndarray],
        generic_meta_data: Dict[str, data.JsonLike],
    ) -> zarr.Group:
        # Sanity / timestamp consistency checks
        assert np.shape(frame_timestamps_us) == (2,)
        assert frame_timestamps_us.dtype == np.dtype("uint64")
        assert frame_timestamps_us[1] >= frame_timestamps_us[0]

        assert frame_timestamps_us[0].item() in self._sequence_timestamp_interval_us, (
            "Frame start timestamp must be contained in the sequence time range"
        )
        assert frame_timestamps_us[1].item() in self._sequence_timestamp_interval_us, (
            "Frame end timestamp must be contained in the sequence time range"
        )

        # Initialize frame group
        frame_group = self._get_frame_group(frame_timestamps_us)

        # Store timestamp data
        assert frame_timestamps_us[1].item() not in self._frames_timestamps_us, (
            "Frame with the same end-of-frame timestamp already exists"
        )
        self._frames_timestamps_us[frame_timestamps_us[1].item()] = frame_timestamps_us[0].item()

        # Store additional generic frame data and meta-data (not dimension / dtype checked)
        (frame_generic_data_group := frame_group.create_group("generic_data")).attrs.put(generic_meta_data)
        for name, value in generic_data.items():
            frame_generic_data_group.create_dataset(
                name,
                data=value,
                # we are not accessing sub-ranges, so disable chunking
                chunks=value.shape,
                # use compression that is fast to decode on modern hardware
                compressor=Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE),
            )

        return frame_group


class BaseSensorComponentReader(ComponentReader):
    """Base class for all sensor component readers"""

    def __init__(self, component_instance_name: str, component_group: zarr.Group) -> None:
        """Initializes a component reader for a given component instance name and group"""
        super().__init__(component_instance_name, component_group)

        if "frames" not in self._group:
            raise RuntimeError("Sensor component doesn't contain any frames")

        # preload frame timestamps and create map
        self._frames_timestamps_us = np.array(self._group["frames"].attrs["frames_timestamps_us"], dtype=np.uint64)
        self._frame_end_to_frame_timestamps_us = {
            end: np.array([self._frames_timestamps_us[i, 0], end], dtype=np.uint64)
            for i, end in enumerate(self._frames_timestamps_us[:, 1])
        }

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

        return self._group["frames"][str(frame_id)]

    @property
    def frames_timestamps_us(self) -> np.ndarray:
        return np.array(self._group["frames"].attrs["frames_timestamps_us"], dtype=np.uint64)

    def get_frame_timestamps_us(self, timestamp_us: int) -> np.ndarray:
        return self._frame_end_to_frame_timestamps_us[timestamp_us]

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
        """Returns generic frame meta-data for a specific frame"""

        return dict(self._get_frame_group(timestamp_us)["generic_data"].attrs)


class BasePointCloudSensorComponentWriter(BaseSensorComponentWriter):
    """Base class for all point cloud sensor component writers"""

    def _store_frame_point_cloud(
        self,
        # start-of-frame / end-of-frame timestamps
        frame_timestamps_us: np.ndarray,
        # point-cloud components - need to have same lenght consistent with point count dimension
        point_count: int,
        point_cloud_data: Dict[str, np.ndarray],
    ) -> zarr.Group:
        # Initialize point cloud group
        (point_cloud_group := self._get_frame_group(frame_timestamps_us).create_group("point_cloud")).attrs.put(
            {"point_count": point_count}
        )

        # Store point cloud components
        for name, data in point_cloud_data.items():
            assert len(data) == point_count, f"{name} doesn't have required point count"

            point_cloud_group.create_dataset(
                name,
                data=data,
                # we are not accessing sub-ranges, so disable chunking
                chunks=data.shape,
                # use compression that is fast to decode on modern hardware
                compressor=Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE),
            )

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

    COMPONENT_NAME: str = "cameras"

    class Writer(BaseSensorComponentWriter):
        """Camera sensor data component writer"""

        @staticmethod
        def get_component_name() -> str:
            """Returns the base name of the current camera sensor component"""
            return CameraSensorComponent.COMPONENT_NAME

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
            frame_timestamps_us: np.ndarray,
            # generic per-frame data (key-value pairs, *not* dimension / dtype validated) and meta-data
            generic_data: Dict[str, np.ndarray],
            generic_meta_data: Dict[str, data.JsonLike],
        ) -> "Self":
            # Initialize frame
            frame_group = self._store_base_frame(frame_timestamps_us, generic_data, generic_meta_data)

            # Store image data (uncompressed, as already encoded)
            frame_group.create_dataset("image", data=np.asarray(image_binary_data), compressor=None).attrs["format"] = (
                image_format
            )

            return self

    class Reader(BaseSensorComponentReader):
        """Camera sensor data component reader"""

        @staticmethod
        def get_component_name() -> str:
            """Returns the base name of the current component"""
            return CameraSensorComponent.COMPONENT_NAME

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

    COMPONENT_NAME: str = "lidars"

    class Writer(BasePointCloudSensorComponentWriter):
        """Lidar sensor data component writer"""

        @staticmethod
        def get_component_name() -> str:
            """Returns the base name of the current lidar sensor component"""
            return LidarSensorComponent.COMPONENT_NAME

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
            model_element: np.ndarray,  # per-point model element indices, if applicable (uint16, [n, 2])
            # start-of-frame / end-of-frame timestamps
            frame_timestamps_us: np.ndarray,
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
            self._store_base_frame(frame_timestamps_us, generic_data, generic_meta_data)

            point_cloud_data = {
                "xyz_m": xyz_m,
            }

            assert intensity.dtype == np.dtype("float32")
            assert 0.0 <= intensity.min() and intensity.max() <= 1.0, "Intensity not normalized"
            point_cloud_data["intensity"] = intensity

            assert timestamp_us.dtype == np.dtype("uint64")
            if point_count:
                assert (frame_timestamps_us[0] <= timestamp_us.min()) and (
                    timestamp_us.max() <= frame_timestamps_us[1]
                ), "Point timestamps outside frame time bounds"
            point_cloud_data["timestamp_us"] = timestamp_us

            assert model_element.shape == (point_count, 2)
            assert model_element.dtype == np.dtype("uint16")
            point_cloud_data["model_element"] = model_element

            # Store point-cloud data
            self._store_frame_point_cloud(frame_timestamps_us, point_count, point_cloud_data)

            return self

    class Reader(BasePointCloudSensorComponentReader):
        """Lidar sensor data component reader"""

        @staticmethod
        def get_component_name() -> str:
            """Returns the base name of the current component"""
            return LidarSensorComponent.COMPONENT_NAME

        @staticmethod
        def supports_component_version(version: str) -> bool:
            """Returns true if the component version is supported by the reader"""
            return version == "0.1"


class RadarSensorComponent:
    """Radar sensor data component"""

    COMPONENT_NAME: str = "radars"

    class Writer(BasePointCloudSensorComponentWriter):
        """Radar sensor data component writer"""

        @staticmethod
        def get_component_name() -> str:
            """Returns the base name of the current radar sensor component"""
            return RadarSensorComponent.COMPONENT_NAME

        @staticmethod
        def get_component_version() -> str:
            """Returns the version of the current radar sensor component"""
            return "0.1"

        def store_frame(
            self,
            # mandatory point-cloud data
            xyz_m: np.ndarray,  # per-point metric coordinates relative to the sensor frame at measure time (raw / not motion-compensated, needs to be non-zero) (float32, [n, 3])
            timestamp_us: np.ndarray,  # per-point point timestamp in microseconds (uint64, [n])
            # start-of-frame / end-of-frame timestamps
            frame_timestamps_us: np.ndarray,
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
            self._store_base_frame(frame_timestamps_us, generic_data, generic_meta_data)

            # Store point-cloud data
            point_cloud_data = {
                "xyz_m": xyz_m,
            }

            assert timestamp_us.dtype == np.dtype("uint64")
            if point_count:
                assert (frame_timestamps_us[0] <= timestamp_us.min()) and (
                    timestamp_us.max() <= frame_timestamps_us[1]
                ), "Point timestamps outside frame time bounds"
            point_cloud_data["timestamp_us"] = timestamp_us

            self._store_frame_point_cloud(
                frame_timestamps_us,
                point_count,
                point_cloud_data,
            )

            return self

    class Reader(BasePointCloudSensorComponentReader):
        """Radar sensor data component reader"""

        @staticmethod
        def get_component_name() -> str:
            """Returns the base name of the current component"""
            return RadarSensorComponent.COMPONENT_NAME

        @staticmethod
        def supports_component_version(version: str) -> bool:
            """Returns true if the component version is supported by the reader"""
            return version == "0.1"


class CuboidsComponent:
    """Data component representing cuboid track observations"""

    COMPONENT_NAME: str = "cuboids"

    class Writer(ComponentWriter):
        """Cuboid track observations component writer"""

        @staticmethod
        def get_component_name() -> str:
            """Returns the base name of the current lidar sensor component"""
            return CuboidsComponent.COMPONENT_NAME

        @staticmethod
        def get_component_version() -> str:
            """Returns the version of the current lidar sensor component"""
            return "0.1"

        def store_observations(
            self,
            cuboid_observations: List[CuboidTrackObservation],  # individual observation
        ) -> "Self":
            self._group.create_group("cuboids").attrs.put(
                {"cuboid_track_observations": [obs.to_dict() for obs in cuboid_observations]}
            )

            return self

    class Reader(ComponentReader):
        """Cuboid tracks component reader"""

        @staticmethod
        def get_component_name() -> str:
            """Returns the base name of the current component"""
            return CuboidsComponent.COMPONENT_NAME

        @staticmethod
        def supports_component_version(version: str) -> bool:
            """Returns true if the component version is supported by the reader"""
            return version == "0.1"

        def get_observations(self) -> Generator[CuboidTrackObservation]:
            """Returns all stored cuboid track observations"""

            for obs in self._group["cuboids"].attrs["cuboid_track_observations"]:
                yield CuboidTrackObservation.from_dict(obs)
