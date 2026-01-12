# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Compatibility layer for unified access to NCore V3 and V4 sequence data.

This module provides protocol-based interfaces and adapter implementations that enable
unified access to both V3 (shard-based) and V4 (component-based) data formats through
a common API.

Key Components:
    - SequenceLoaderProtocol: Unified interface for sequence-level data access
    - SensorProtocol: Common interface for sensor data (cameras, lidars, radars)
    - CameraSensorProtocol: Camera-specific extensions
    - RayBundleSensorProtocol: Ray bundle sensor interface (lidar/radar)
    - LidarSensorProtocol: Lidar-specific extensions
    - SequenceLoaderV3: Adapter for V3 shard-based data
    - SequenceLoaderV4: Adapter for V4 component-based data

The compatibility layer handles differences between V3 and V4 formats including:
    - Different storage mechanisms (shards vs. components)
    - Motion compensation conventions (V3 stores compensated, V4 stores uncompensated)
    - Pose graph construction and transformation APIs
    - Frame indexing and timestamp access patterns
    - Metadata retrieval

Example:
    # Load V4 data
    reader = SequenceComponentGroupsReader([Path("file.zarr.itar"), Path("some/folder.zarr")])
    loader = SequenceLoaderV4(reader)

    # Load V3 data
    shard_loader = ShardDataLoader(["data_shard_*.zarr.itar"])
    loader = SequenceLoaderV3(shard_loader)

    # Use unified API for either format
    camera = loader.get_camera_sensor("camera_front")
    image = camera.get_frame_image(0)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Protocol, Tuple, Union, cast

import numpy as np
import PIL.Image as PILImage

from typing_extensions import override
from upath import UPath

from ncore.impl.common.common import HalfClosedInterval, unpack_optional
from ncore.impl.common.transformations import MotionCompensator, PoseGraphInterpolator
from ncore.impl.data import data3, types, util
from ncore.impl.data.data import JsonLike
from ncore.impl.data.data3 import LidarSensor, ShardDataLoader
from ncore.impl.data.data4.components import (
    BaseRayBundleSensorComponentReader,
    CameraSensorComponent,
    CuboidsComponent,
    IntrinsicsComponent,
    LidarSensorComponent,
    MasksComponent,
    PosesComponent,
    RadarSensorComponent,
    SequenceComponentGroupsReader,
)
from ncore.impl.data.data4.types import CuboidTrackObservation
from ncore.impl.data.types import FrameTimepoint


if TYPE_CHECKING:
    import numpy.typing as npt  # type: ignore[import-not-found]


class SequenceLoaderProtocol(Protocol):
    """SequenceLoaderProtocol provides unified access to a relevant subset of common NCore V3 and default V4 sequence data APIs"""

    @property
    def sequence_id(self) -> str:
        """The unique identifier of the sequence"""
        ...

    @property
    def generic_meta_data(self) -> Dict[str, JsonLike]:
        """Generic meta-data associated with the sequence"""
        ...

    @property
    def sequence_timestamp_interval_us(self) -> HalfClosedInterval:
        """The time range of the sequence in microseconds"""
        ...

    @property
    def sequence_paths(self) -> List[UPath]:
        """List of all dataset paths comprising this sequence (shards / components)"""
        ...

    def reload_resources(self) -> None:
        """Reloads any resources used by the internal sequence loader (potentially required for multi-process data loading)"""
        ...

    def get_sequence_meta(self) -> Dict[str, JsonLike]:
        """Returns sequence-wide meta-data summary (format is instance-dependent)"""
        ...

    @property
    def camera_ids(self) -> List[str]:
        """All camera sensor IDs in the sequence"""
        ...

    @property
    def lidar_ids(self) -> List[str]:
        """All lidar sensor IDs in the sequence"""
        ...

    @property
    def radar_ids(self) -> List[str]:
        """All radar sensor IDs in the sequence"""
        ...

    @property
    def pose_graph(self) -> PoseGraphInterpolator:
        """The pose graph representing all static and dynamic transformations in the sequence"""
        ...

    def get_camera_sensor(self, sensor_id: str) -> CameraSensorProtocol:
        """Returns a camera sensor instance for a given sensor id"""
        ...

    def get_lidar_sensor(self, sensor_id: str) -> LidarSensorProtocol:
        """Returns a lidar sensor instance for a given sensor id"""
        ...

    def get_radar_sensor(self, sensor_id: str) -> RadarSensorProtocol:
        """Returns a radar sensor instance for a given sensor id"""
        ...

    def get_cuboid_track_observations(self) -> Generator[CuboidTrackObservation]:
        """Returns all cuboid track observations in the sequence"""
        ...


class SensorProtocol(Protocol):
    """SensorProtocol provides unified access to a relevant subset of common NCore V3 and default V4 sensor data APIs"""

    _pose_graph: PoseGraphInterpolator

    @property
    def sensor_id(self) -> str:
        """The ID of the sensor"""
        ...

    @property
    def frames_count(self) -> int:
        """The number of frames associated with the sensor"""
        ...

    @property
    def frames_timestamps_us(self) -> npt.NDArray[np.uint64]:
        """The start/end timestamps of the frames associated with the sensor [(N,2) array]"""
        ...

    def get_frames_timestamps_us(self, frame_timepoint: FrameTimepoint = FrameTimepoint.END) -> npt.NDArray[np.uint64]:
        """Returns the timestamps of all frames at the specified frame-relative timepoint (start or end) [shape (N,)]"""

        return self.frames_timestamps_us[:, frame_timepoint.value]

    ## Poses API (use pose-graph exclusively)
    @property
    def pose_graph(self) -> PoseGraphInterpolator:
        """Access the sensor-associated pose graph (usually the sequence-wide one, unless overwritten)"""
        return self._pose_graph

    def set_pose_graph(self, pose_graph: PoseGraphInterpolator) -> None:
        """Assigns a new pose graph to the sensor instance (e.g., to overwrite / use a sensor-exclusive pose graph)"""
        self._pose_graph = pose_graph

    @property
    def T_sensor_rig(self) -> Optional[npt.NDArray[np.floating]]:
        """Return static extrinsic transformation from sensor to rig coordinate frame.

        Returns the 4x4 homogeneous transformation matrix T_sensor_rig that transforms
        points from the sensor coordinate frame to the rig coordinate frame.

        Returns:
            4x4 transformation matrix if the static transformation exists, None otherwise
        """
        try:
            return self._pose_graph.evaluate_poses(
                source_node=self.sensor_id,
                target_node="rig",
                timestamps_us=np.empty((), dtype=np.uint64),
            )
        except KeyError:
            return None

    def get_frame_T_sensor_world(
        self, frame_index: int, frame_timepoint: FrameTimepoint = FrameTimepoint.END
    ) -> npt.NDArray[np.floating]:
        """Evaluates the sensor-to-world pose for a specific frame and frame timepoint (start or end) [(4,4) array]"""

        # Rely on a special case of the generic frame pose evaluation
        return self.get_frames_T_source_target(
            source_node=self.sensor_id,
            target_node="world",
            frame_indices=np.array(frame_index, dtype=np.int64),
            frame_timepoint=frame_timepoint,
        )

    # Generic relative pose evaluation
    def get_frames_T_source_target(
        self,
        source_node: str,
        target_node: str,
        frame_indices: npt.NDArray[np.integer],
        frame_timepoint: Optional[FrameTimepoint] = None,
    ) -> npt.NDArray[np.floating]:
        """Evaluates relative poses at timestamps inferred from frame indices.

        Computes transformation matrices T_source_target that transform points from the
        source coordinate frame to the target coordinate frame at specified frame times.

        Args:
            source_node: Name of the source coordinate frame
            target_node: Name of the target coordinate frame
            frame_indices: Array of frame indices at which to evaluate poses
            frame_timepoint: Frame-relative timepoint (START or END). If None, returns both

        Returns:
            Transformation matrices with shape [frame_indices-shape,2,4,4] if frame_timepoint
            is None (both start and end poses), else [frame_indices-shape,4,4] (single timepoint)
        """

        if frame_timepoint is None:
            # Return start and end poses at given frame indices
            timestamps_us = self.frames_timestamps_us[frame_indices, :]
        else:
            timestamps_us = self.frames_timestamps_us[frame_indices, frame_timepoint.value]

        return self.pose_graph.evaluate_poses(
            source_node,
            target_node,
            timestamps_us=timestamps_us,
        )

    ## Generic per-frame data
    def get_frame_generic_data_names(self, frame_index: int) -> List[str]:
        """List of all generic frame-data names"""
        ...

    def has_frame_generic_data(self, frame_index: int, name: str) -> bool:
        """Signals if named generic frame-data exists"""
        ...

    def get_frame_generic_data(self, frame_index: int, name: str) -> npt.NDArray[Any]:
        """Returns generic frame-data for a specific frame and name"""
        ...

    def get_frame_generic_meta_data(self, frame_index: int) -> Dict[str, JsonLike]:
        """Returns generic frame meta-data for a specific frame"""
        ...

    ## Helper
    def get_frame_index_range(
        self,
        start_frame_index: Optional[int] = None,
        stop_frame_index: Optional[int] = None,
        step_frame_index: Optional[int] = None,
    ) -> range:
        """Returns a (potentially empty) range of frame indices following start:stop:step slice conventions,
        defaulting to full frame index range for absent range bound specifiers
        """

        return range(*slice(start_frame_index, stop_frame_index, step_frame_index).indices(self.frames_count))

    def get_frame_timestamp_us(self, frame_index: int, frame_timepoint: FrameTimepoint = FrameTimepoint.END) -> int:
        """Returns the timestamp of a specific frame at the specified relative frame timepoint (start or end)"""

        return int(self.frames_timestamps_us[frame_index, frame_timepoint.value])

    def get_closest_frame_index(self, timestamp_us: int, relative_frame_time: float = 1.0) -> int:
        """Given a timestamp, returns the frame index of the closest frame based on the specified relative frame time-point (0.0 ~= start-of-frames / 1.0 ~= end-of-frames)"""

        # Special cases: avoid computation for boundary values
        if relative_frame_time == 0.0:
            target_timestamps_us = self.frames_timestamps_us[:, 0]
        elif relative_frame_time == 1.0:
            target_timestamps_us = self.frames_timestamps_us[:, 1]
        else:
            assert 0.0 <= relative_frame_time <= 1.0, (
                f"relative_frame_time must be in [0, 1], got {relative_frame_time}"
            )
            target_timestamps_us = (
                self.frames_timestamps_us[:, 0]
                + relative_frame_time * (self.frames_timestamps_us[:, 1] - self.frames_timestamps_us[:, 0])
            ).astype(np.uint64)

        return util.closest_index_sorted(target_timestamps_us, timestamp_us)


class CameraSensorProtocol(SensorProtocol, Protocol):
    """CameraSensorProtocol provides unified access to a relevant subset of common NCore V3 and default V4 camera sensor APIs"""

    @property
    def model_parameters(self) -> types.ConcreteCameraModelParametersUnion:
        """Returns parameters specific to the camera's intrinsic model"""
        ...

    def get_mask_images(self) -> Dict[str, PILImage.Image]:
        """Returns all named camera mask images"""
        ...

    # Image Frame Data
    class EncodedImageDataHandleProtocol(Protocol):
        """References encoded image data without loading it"""

        def get_data(self) -> types.EncodedImageData:
            """Loads the referenced encoded image data to memory"""
            ...

    def get_frame_handle(self, frame_index: int) -> EncodedImageDataHandleProtocol:
        """Returns the frame's encoded image data"""
        ...

    def get_frame_data(self, frame_index: int) -> types.EncodedImageData:
        """Returns the frame's encoded image data"""
        return self.get_frame_handle(frame_index).get_data()

    def get_frame_image(self, frame_index: int) -> PILImage.Image:
        """Returns the frame's decoded image data"""
        return self.get_frame_data(frame_index).get_decoded_image()

    def get_frame_image_array(self, frame_index: int) -> npt.NDArray[np.uint8]:
        """Returns decoded image data as array [W,H,C]"""
        return np.asarray(self.get_frame_image(frame_index))


class RayBundleSensorProtocol(SensorProtocol, Protocol):
    """RayBundleSensorProtocol provides unified access to a relevant subset of common NCore V3 and default V4 ray-bundle sensor APIs"""

    def get_frame_ray_bundle_count(self, frame_index: int) -> int:
        """Returns the number of rays for a specific frame without decoding it.

        Args:
            frame_index: Index of the frame

        Returns:
            Number of rays for a specific frame
        """
        ...

    def get_frame_ray_bundle_direction(self, frame_index: int) -> npt.NDArray[np.float32]:
        """Returns the per-ray directions for the ray-bundle for a specific frame.

        Args:
            frame_index: Index of the frame
        Returns:
            Array of per-ray directions [N,3]
        """
        ...

    def get_frame_ray_bundle_timestamp_us(self, frame_index: int) -> npt.NDArray[np.uint64]:
        """Returns the per-ray timestamps for the ray-bundle for a specific frame.

        Args:
            frame_index: Index of the frame
        Returns:
            Array of per-ray timestamps [N,]
        """
        ...

    def get_frame_ray_bundle_return_count(self, frame_index: int) -> int:
        """Returns the number of different ray returns for a specific frame without decoding it.

        Args:
            frame_index: Index of the frame

        Returns:
            Number of ray returns for a specific frame
        """
        ...

    def get_frame_ray_bundle_return_distance(self, frame_index: int, return_index: int = 0) -> npt.NDArray[np.float32]:
        """Returns the per-ray measured distances for the ray bundle returns of a specific frame.

        Args:
            frame_index: Index of the frame
            return_index: Index of the ray bundle return to retrieve (for multi-return sensors)

        Returns:
            Array of per-ray distances [N,]
        """
        ...

    @dataclass
    class FramePointCloud:
        """Container for point cloud data with optional motion compensation.

        Attributes:
            motion_compensation: Whether coordinates are relative to sensor frame at
                end-of-frame time (True) or sensor frame at point-time (False)
            xyz_m_start: Motion-compensated ray segment start points [N,3], or None if not requested
            xyz_m_end: Motion-compensated ray segment end points [N,3]
        """

        motion_compensation: bool
        xyz_m_start: Optional[npt.NDArray[np.floating]]
        xyz_m_end: npt.NDArray[np.floating]

    def get_frame_point_cloud(
        self, frame_index: int, motion_compensation: bool, with_start_points: bool, return_index: int = 0
    ) -> FramePointCloud:
        """Returns a frame-point cloud motion-compensated or non-motion-compensated for a specific frame.

        Args:
            frame_index: Index of the frame to retrieve
            motion_compensation: If True, returns points in sensor frame at end-of-frame time.
                If False, returns points in sensor frame at point-time
            with_start_points: If True, include ray segment start points
            return_index: Index of the point cloud return to retrieve (for multi-return sensors)

        Returns:
            FramePointCloud containing the point cloud data with requested motion compensation
        """
        ...


class LidarSensorProtocol(RayBundleSensorProtocol, Protocol):
    """LidarSensorProtocol provides unified access to a relevant subset of common NCore V3 and default V4 lidar sensor APIs"""

    @property
    def model_parameters(self) -> Optional[types.ConcreteLidarModelParametersUnion]:
        """Returns parameters specific to the lidar's intrinsic model (optional as not mandatory in V3)"""
        ...

    def get_frame_ray_bundle_model_element(self, frame_index: int) -> Optional[npt.NDArray[np.uint16]]:
        """Returns the per-ray model elements for a ray bundle for a specific frame, if available.

        Args:
            frame_index: Index of the frame
        Returns:
            Array of per-ray model elements [N,] or None if not available
        """
        ...

    def get_frame_ray_bundle_return_intensity(self, frame_index: int, return_index: int = 0) -> npt.NDArray[np.float32]:
        """Returns the per-ray measured intensities for a ray bundle return for a specific frame.

        Args:
            frame_index: Index of the frame
            return_index: Index of the ray bundle return to retrieve (for multi-return sensors)

        Returns:
            Array of per-ray intensities [N,]
        """
        ...


class RadarSensorProtocol(RayBundleSensorProtocol, Protocol):
    """RadarSensorProtocol provides unified access to a relevant subset of common NCore V3 and default V4 radar sensor APIs"""

    ...


class SequenceLoaderV4(SequenceLoaderProtocol):
    """SequenceLoader implementation for NCore V4 data.

    Provides a unified interface to access V4 format sequence data including sensors,
    poses, intrinsics, masks, and cuboid annotations.

    Args:
        reader: Component store reader for V4 data
        poses_component_group_name: Name of the poses component group to load
        intrinsics_component_group_name: Name of the intrinsics component group to load
        masks_component_group_name: Name of the masks component group to load
        cuboids_component_group_name: Name of the cuboids component group to load
    """

    def __init__(
        self,
        reader: SequenceComponentGroupsReader,
        # Component group names to load
        poses_component_group_name: str = "default",
        intrinsics_component_group_name: str = "default",
        masks_component_group_name: str = "default",
        cuboids_component_group_name: str = "default",
    ):
        self._reader: SequenceComponentGroupsReader = reader

        # open all default component readers
        assert poses_component_group_name in (
            poses_readers := self._reader.open_component_readers(PosesComponent.Reader)
        ), f"PosesComponent group '{poses_component_group_name}' not found"
        self._poses_reader: PosesComponent.Reader = poses_readers[poses_component_group_name]

        assert intrinsics_component_group_name in (
            intrinsics_readers := self._reader.open_component_readers(IntrinsicsComponent.Reader)
        ), f"IntrinsicsComponent group '{intrinsics_component_group_name}' not found"
        self._intrinsics_reader: IntrinsicsComponent.Reader = intrinsics_readers[intrinsics_component_group_name]

        assert masks_component_group_name in (
            masks_readers := self._reader.open_component_readers(MasksComponent.Reader)
        ), f"MasksComponent group '{masks_component_group_name}' not found"
        self._masks_reader: MasksComponent.Reader = masks_readers[masks_component_group_name]

        assert cuboids_component_group_name in (
            cuboids_readers := self._reader.open_component_readers(CuboidsComponent.Reader)
        ), f"CuboidsComponent group '{cuboids_component_group_name}' not found"
        self._cuboids_reader: CuboidsComponent.Reader = cuboids_readers[cuboids_component_group_name]

        self._cameras_readers: Dict[str, CameraSensorComponent.Reader] = self._reader.open_component_readers(
            CameraSensorComponent.Reader
        )
        self._lidars_readers: Dict[str, LidarSensorComponent.Reader] = self._reader.open_component_readers(
            LidarSensorComponent.Reader
        )
        self._radars_readers: Dict[str, RadarSensorComponent.Reader] = self._reader.open_component_readers(
            RadarSensorComponent.Reader
        )

        # init pose graph
        self._pose_graph: PoseGraphInterpolator = PoseGraphInterpolator(
            # static edges
            [
                PoseGraphInterpolator.Edge(source, target, pose, None)
                for (source, target), pose in self._poses_reader.get_static_poses()
            ]
            +
            # dynamic edges
            [
                PoseGraphInterpolator.Edge(source, target, poses, timestamps_us)
                for (source, target), (
                    poses,
                    timestamps_us,
                ) in self._poses_reader.get_dynamic_poses()
            ]
        )

    @property
    @override
    def sequence_id(self) -> str:
        return self._reader.sequence_id

    @property
    @override
    def generic_meta_data(self) -> Dict[str, JsonLike]:
        return self._reader.generic_meta_data

    @property
    @override
    def sequence_timestamp_interval_us(self) -> HalfClosedInterval:
        return self._reader.sequence_timestamp_interval_us

    @property
    @override
    def sequence_paths(self) -> List[UPath]:
        return self._reader.component_store_paths

    @override
    def reload_resources(self) -> None:
        self._reader.reload_resources()

    @override
    def get_sequence_meta(self) -> Dict[str, JsonLike]:
        return cast(Dict[str, JsonLike], self._reader.get_sequence_meta().to_dict())

    @property
    @override
    def camera_ids(self) -> List[str]:
        return list(self._cameras_readers.keys())

    @property
    @override
    def lidar_ids(self) -> List[str]:
        return list(self._lidars_readers.keys())

    @property
    @override
    def radar_ids(self) -> List[str]:
        return list(self._radars_readers.keys())

    @property
    @override
    def pose_graph(self) -> PoseGraphInterpolator:
        return self._pose_graph

    class Sensor(SensorProtocol):
        """Base sensor implementation for V4 data providing common sensor functionality.

        Args:
            sensor_reader: Component reader for the sensor data
            pose_graph: Pose graph interpolator for coordinate transformations
        """

        def __init__(
            self,
            sensor_reader: Union[
                CameraSensorComponent.Reader, LidarSensorComponent.Reader, RadarSensorComponent.Reader
            ],
            pose_graph: PoseGraphInterpolator,
        ):
            self.set_pose_graph(pose_graph)

            self._reader: Union[
                CameraSensorComponent.Reader, LidarSensorComponent.Reader, RadarSensorComponent.Reader
            ] = sensor_reader

            # preload frame timestamps once
            self._frames_timestamps_us = self._reader.frames_timestamps_us

        @property
        @override
        def sensor_id(self) -> str:
            return self._reader.instance_name

        @property
        @override
        def frames_count(self) -> int:
            return self._reader.frames_count

        @property
        @override
        def frames_timestamps_us(self) -> npt.NDArray[np.uint64]:
            return self._frames_timestamps_us

        # Generic per-frame data
        @override
        def get_frame_generic_data_names(self, frame_index: int) -> List[str]:
            """List of all generic frame-data names"""

            return self._reader.get_frame_generic_data_names(
                self.frames_timestamps_us[frame_index, FrameTimepoint.END.value].item()
            )

        @override
        def has_frame_generic_data(self, frame_index: int, name: str) -> bool:
            """Signals if named generic frame-data exists"""

            return self._reader.has_frame_generic_data(
                self.frames_timestamps_us[frame_index, FrameTimepoint.END.value].item(), name
            )

        @override
        def get_frame_generic_data(self, frame_index: int, name: str) -> npt.NDArray[Any]:
            """Returns generic frame-data for a specific frame and name"""

            return self._reader.get_frame_generic_data(
                self.frames_timestamps_us[frame_index, FrameTimepoint.END.value].item(), name
            )

        @override
        def get_frame_generic_meta_data(self, frame_index: int) -> Dict[str, JsonLike]:
            """Returns generic frame meta-data for a specific frame"""

            return self._reader.get_frame_generic_meta_data(
                self.frames_timestamps_us[frame_index, FrameTimepoint.END.value].item()
            )

    class CameraSensor(Sensor, CameraSensorProtocol):
        """Camera sensor implementation for V4 data.

        Args:
            reader: Camera component reader
            mask_reader: Masks component reader for camera masks
            model_parameters: Camera intrinsic model parameters
            pose_graph: Pose graph interpolator for coordinate transformations
        """

        def __init__(
            self,
            reader: CameraSensorComponent.Reader,
            mask_reader: MasksComponent.Reader,
            model_parameters: types.ConcreteCameraModelParametersUnion,
            pose_graph: PoseGraphInterpolator,
        ):
            super().__init__(reader, pose_graph)

            self._mask_reader: MasksComponent.Reader = mask_reader
            self._model_parameters: types.ConcreteCameraModelParametersUnion = model_parameters

        @property
        def camera_reader(self) -> CameraSensorComponent.Reader:
            return cast(CameraSensorComponent.Reader, self._reader)

        @property
        @override
        def model_parameters(self) -> types.ConcreteCameraModelParametersUnion:
            """Returns parameters specific to the camera's intrinsic model"""
            return self._model_parameters

        @override
        def get_mask_images(self) -> Dict[str, PILImage.Image]:
            """Returns all named camera mask images"""
            return dict(self._mask_reader.get_camera_mask_images(self.sensor_id))

        @override
        def get_frame_handle(self, frame_index: int) -> CameraSensorProtocol.EncodedImageDataHandleProtocol:
            """Returns the frame's encoded image data"""
            return self.camera_reader.get_frame_handle(
                self.frames_timestamps_us[frame_index, FrameTimepoint.END.value].item()
            )

    @override
    def get_camera_sensor(self, sensor_id: str) -> CameraSensorProtocol:
        return self.CameraSensor(
            reader=unpack_optional(
                self._cameras_readers.get(sensor_id),
                msg=f"Camera sensor '{sensor_id}' not available in loaded camera sensor groups",
            ),
            mask_reader=self._masks_reader,
            pose_graph=self._pose_graph,
            model_parameters=self._intrinsics_reader.get_camera_model_parameters(sensor_id),
        )

    class RayBundleSensor(Sensor, RayBundleSensorProtocol):
        """Base ray bundle sensor implementation for V4 data (lidar/radar).

        Args:
            reader: Ray bundle sensor component reader (lidar or radar)
            pose_graph: Pose graph interpolator for coordinate transformations and motion compensation
        """

        def __init__(
            self,
            reader: Union[LidarSensorComponent.Reader, RadarSensorComponent.Reader],
            pose_graph: PoseGraphInterpolator,
        ):
            super().__init__(reader, pose_graph)

            self._motion_compensator: MotionCompensator = MotionCompensator(pose_graph)

        @property
        def ray_bundle_reader(self) -> BaseRayBundleSensorComponentReader:
            return cast(BaseRayBundleSensorComponentReader, self._reader)

        @override
        def get_frame_ray_bundle_count(self, frame_index: int) -> int:
            """Returns the number of rays for a specific frame without decoding it"""
            return self.ray_bundle_reader.get_frame_ray_bundle_count(
                self.frames_timestamps_us[frame_index, FrameTimepoint.END.value].item()
            )

        @override
        def get_frame_ray_bundle_return_count(self, frame_index: int) -> int:
            """Returns the number of different ray returns for a specific frame without decoding it"""
            return self.ray_bundle_reader.get_frame_ray_bundle_return_count(
                self.frames_timestamps_us[frame_index, FrameTimepoint.END.value].item()
            )

        @override
        def get_frame_ray_bundle_direction(self, frame_index):
            """Returns the per-ray directions for the ray-bundle for a specific frame"""
            return self.ray_bundle_reader.get_frame_ray_bundle_data(
                self.frames_timestamps_us[frame_index, FrameTimepoint.END.value].item(), "direction"
            )

        @override
        def get_frame_ray_bundle_timestamp_us(self, frame_index: int) -> npt.NDArray[np.uint64]:
            """Returns the per-ray timestamps for the ray-bundle for a specific frame"""
            return self.ray_bundle_reader.get_frame_ray_bundle_data(
                self.frames_timestamps_us[frame_index, FrameTimepoint.END.value].item(), "timestamp_us"
            )

        @override
        def get_frame_point_cloud(
            self, frame_index: int, motion_compensation: bool, with_start_points: bool, return_index: int = 0
        ) -> RayBundleSensorProtocol.FramePointCloud:
            """Returns motion-compensated or non-motion-compensated point-cloud for a specific frame"""
            frame_timestamps_us = self.frames_timestamps_us[frame_index, :]

            # V4 stores non-motion-compensated ray directions in 'direction' field and return-specific distances
            xyz_m = (
                self.ray_bundle_reader.get_frame_ray_bundle_data(
                    int(frame_timestamps_us[FrameTimepoint.END.value]), "direction"
                )
                * self.ray_bundle_reader.get_frame_ray_bundle_return_data(
                    int(frame_timestamps_us[FrameTimepoint.END.value]), "distance_m", return_index=return_index
                )[:, np.newaxis]
            )

            if not motion_compensation:
                return RayBundleSensorProtocol.FramePointCloud(
                    motion_compensation=False,
                    xyz_m_start=np.zeros_like(xyz_m) if with_start_points else None,
                    xyz_m_end=xyz_m,
                )

            # Apply motion compensation
            motion_compensation_result = self._motion_compensator.motion_compensate_points(
                sensor_id=self.sensor_id,
                xyz_pointtime=xyz_m,
                timestamp_us=self.ray_bundle_reader.get_frame_ray_bundle_data(
                    int(frame_timestamps_us[FrameTimepoint.END.value]), "timestamp_us"
                ),
                frame_start_timestamp_us=int(frame_timestamps_us[FrameTimepoint.START.value]),
                frame_end_timestamp_us=int(frame_timestamps_us[FrameTimepoint.END.value]),
            )
            return RayBundleSensorProtocol.FramePointCloud(
                motion_compensation=True,
                xyz_m_start=motion_compensation_result.xyz_s_sensorend if with_start_points else None,
                xyz_m_end=motion_compensation_result.xyz_e_sensorend,
            )

        @override
        def get_frame_ray_bundle_return_distance(
            self, frame_index: int, return_index: int = 0
        ) -> npt.NDArray[np.float32]:
            """Returns the per-ray measured distances for the ray bundle return for a specific frame"""

            # V4 stores non-motion-compensated point cloud in 'xyz_m' field, so we can use it's norm
            # as the measured distance
            return self.ray_bundle_reader.get_frame_ray_bundle_return_data(
                self.frames_timestamps_us[frame_index, FrameTimepoint.END.value].item(),
                "distance_m",
                return_index=return_index,
            )

    class LidarSensor(RayBundleSensor, LidarSensorProtocol):
        """Lidar sensor implementation for V4 data.

        Args:
            reader: Lidar component reader
            pose_graph: Pose graph interpolator for coordinate transformations
            model_parameters: Lidar intrinsic model parameters, if available
        """

        def __init__(
            self,
            reader: LidarSensorComponent.Reader,
            pose_graph: PoseGraphInterpolator,
            model_parameters: Optional[types.ConcreteLidarModelParametersUnion],
        ):
            super().__init__(reader, pose_graph)

            self._model_parameters: Optional[types.ConcreteLidarModelParametersUnion] = model_parameters

        @property
        def lidar_reader(self) -> LidarSensorComponent.Reader:
            return cast(LidarSensorComponent.Reader, self._reader)

        @property
        @override
        def model_parameters(self) -> Optional[types.ConcreteLidarModelParametersUnion]:
            """Returns parameters specific to the lidar's intrinsic model, if available"""
            return self._model_parameters

        @override
        def get_frame_ray_bundle_model_element(self, frame_index: int) -> Optional[npt.NDArray[np.uint16]]:
            """Returns the per-ray model elements for a ray bundle for a specific frame, if available"""

            if self.lidar_reader.has_frame_ray_bundle_data(
                timestamp_us := self.frames_timestamps_us[frame_index, FrameTimepoint.END.value].item(), "model_element"
            ):
                return self.lidar_reader.get_frame_ray_bundle_data(
                    timestamp_us,
                    "model_element",
                )

            return None

        @override
        def get_frame_ray_bundle_return_intensity(
            self, frame_index: int, return_index: int = 0
        ) -> npt.NDArray[np.float32]:
            """Returns the per-ray measured intensities for a ray bundle return for a specific frame"""

            return self.lidar_reader.get_frame_ray_bundle_return_data(
                self.frames_timestamps_us[frame_index, FrameTimepoint.END.value].item(),
                "intensity",
                return_index=return_index,
            )

    @override
    def get_lidar_sensor(self, sensor_id: str) -> LidarSensorProtocol:
        return self.LidarSensor(
            reader=unpack_optional(
                self._lidars_readers.get(sensor_id),
                msg=f"Lidar sensor '{sensor_id}' not available in loaded lidar sensor groups",
            ),
            pose_graph=self._pose_graph,
            model_parameters=self._intrinsics_reader.get_lidar_model_parameters(sensor_id),
        )

    class RadarSensor(RayBundleSensor, RadarSensorProtocol):
        """Radar sensor implementation for V4 data.

        Args:
            reader: Radar component reader
            pose_graph: Pose graph interpolator for coordinate transformations
        """

        def __init__(
            self,
            reader: RadarSensorComponent.Reader,
            pose_graph: PoseGraphInterpolator,
        ):
            super().__init__(reader, pose_graph)

        @property
        def radar_reader(self) -> RadarSensorComponent.Reader:
            return cast(RadarSensorComponent.Reader, self._reader)

    @override
    def get_radar_sensor(self, sensor_id: str) -> RadarSensorProtocol:
        return self.RadarSensor(
            reader=unpack_optional(
                self._radars_readers.get(sensor_id),
                msg=f"Radar sensor '{sensor_id}' not available in loaded radar sensor groups",
            ),
            pose_graph=self._pose_graph,
        )

    @override
    def get_cuboid_track_observations(self) -> Generator[CuboidTrackObservation]:
        """Returns all cuboid track observations in the sequence"""
        yield from self._cuboids_reader.get_observations()


class SequenceLoaderV3(SequenceLoaderProtocol):
    """SequenceLoader implementation for NCore V3 data.

    Provides a unified interface to access V3 format (shard-based) sequence data,
    wrapping the ShardDataLoader with the SequenceLoaderProtocol interface.

    Args:
        loader: Shard data loader for V3 data
    """

    def __init__(
        self,
        loader: ShardDataLoader,
    ):
        self._loader: ShardDataLoader = loader

        # init pose graph from static / dynamic edges
        poses = loader.get_poses()
        edges = [
            PoseGraphInterpolator.Edge(
                source_node="rig",
                target_node="world",
                # In V4 we use float32 for local world poses (sufficiently accurate)
                T_source_target=poses.T_rig_worlds.astype(np.float32),
                timestamps_us=poses.T_rig_world_timestamps_us,
            ),
            PoseGraphInterpolator.Edge(
                source_node="world",
                target_node="world_global",
                # Require float64 for global world poses (e.g., ECEF)
                T_source_target=poses.T_rig_world_base.astype(np.float64),
                timestamps_us=None,
            ),
        ]
        for sensor_id in loader.get_sensor_ids():
            edges.append(
                PoseGraphInterpolator.Edge(
                    source_node=sensor_id,
                    target_node="rig",
                    # Extrinsics are sufficiently accurate in float32
                    T_source_target=loader.get_sensor(sensor_id).get_T_sensor_rig().astype(np.float32),
                    timestamps_us=None,
                )
            )
        self._pose_graph: PoseGraphInterpolator = PoseGraphInterpolator(edges=edges)

    @property
    @override
    def sequence_id(self) -> str:
        return self._loader.get_sequence_id(with_shard_range=False)

    @property
    @override
    def generic_meta_data(self) -> Dict[str, JsonLike]:
        return self._loader.get_generic_meta_data()

    @property
    @override
    def sequence_timestamp_interval_us(self) -> HalfClosedInterval:
        # Note: V3 does not store sequence time bounds directly, but they correspond to rig pose timestamps ranges
        T_rig_world_timestamps_us = self._loader.get_poses().T_rig_world_timestamps_us
        return HalfClosedInterval.from_start_end(
            int(T_rig_world_timestamps_us[0].item()),
            int(T_rig_world_timestamps_us[-1].item()),
        )

    @property
    @override
    def sequence_paths(self) -> List[UPath]:
        return self._loader.get_shard_paths()

    @override
    def reload_resources(self) -> None:
        self._loader.reload_store_resources()

    @override
    def get_sequence_meta(self) -> Dict[str, JsonLike]:
        return self._loader.get_sequence_meta()

    @property
    @override
    def camera_ids(self) -> List[str]:
        return self._loader.get_camera_ids()

    @property
    @override
    def lidar_ids(self) -> List[str]:
        return self._loader.get_lidar_ids()

    @property
    @override
    def radar_ids(self) -> List[str]:
        return self._loader.get_radar_ids()

    @property
    @override
    def pose_graph(self) -> PoseGraphInterpolator:
        return self._pose_graph

    class Sensor(SensorProtocol):
        """Base sensor implementation for V3 data providing common sensor functionality.

        Args:
            sensor: V3 sensor instance (camera, lidar, or radar)
            pose_graph: Pose graph interpolator for coordinate transformations
        """

        def __init__(
            self,
            sensor: Union[data3.CameraSensor, data3.LidarSensor, data3.RadarSensor],
            pose_graph: PoseGraphInterpolator,
        ):
            self.set_pose_graph(pose_graph)

            self._sensor = sensor

            # preload frame timestamps once [need to iterate through all frames to get start timestamp also in V3]
            self._frames_timestamps_us = np.array(
                [
                    [
                        sensor.get_frame_timestamp_us(frame_idx, FrameTimepoint.START),
                        sensor.get_frame_timestamp_us(frame_idx, FrameTimepoint.END),
                    ]
                    for frame_idx in sensor.get_frame_index_range()
                ],
                dtype=np.uint64,
            )

        @property
        @override
        def sensor_id(self) -> str:
            return self._sensor.get_sensor_id()

        @property
        @override
        def frames_count(self) -> int:
            return self._sensor.get_frames_count()

        @property
        @override
        def frames_timestamps_us(self) -> npt.NDArray[np.uint64]:
            return self._frames_timestamps_us

        # Generic per-frame data
        @override
        def get_frame_generic_data_names(self, frame_index: int) -> List[str]:
            """List of all generic frame-data names"""

            return self._sensor.get_frame_generic_data_names(frame_index)

        @override
        def has_frame_generic_data(self, frame_index: int, name: str) -> bool:
            """Signals if named generic frame-data exists"""

            return self._sensor.has_frame_generic_data(frame_index, name)

        @override
        def get_frame_generic_data(self, frame_index: int, name: str) -> npt.NDArray[Any]:
            """Returns generic frame-data for a specific frame and name"""

            return self._sensor.get_frame_generic_data(frame_index, name)

        @override
        def get_frame_generic_meta_data(self, frame_index: int) -> Dict[str, JsonLike]:
            """Returns generic frame meta-data for a specific frame"""

            return self._sensor.get_frame_generic_meta_data(frame_index)

        # Compat-API pose access
        @property
        @override
        def T_sensor_rig(self) -> Optional[npt.NDArray[np.floating]]:
            # Return static extrinsic (unconditionally available in V3)
            return self._sensor.get_T_sensor_rig()

        @override
        def get_frame_T_sensor_world(
            self, frame_index: int, frame_timepoint: FrameTimepoint = FrameTimepoint.END
        ) -> npt.NDArray[np.floating]:
            # Rely on hardcoded poses in V3 API
            return self._sensor.get_frame_T_sensor_world(frame_index, frame_timepoint)

    class CameraSensor(Sensor, CameraSensorProtocol):
        """Camera sensor implementation for V3 data.

        Args:
            sensor: V3 camera sensor instance
            pose_graph: Pose graph interpolator for coordinate transformations
        """

        def __init__(
            self,
            sensor: data3.CameraSensor,
            pose_graph: PoseGraphInterpolator,
        ):
            super().__init__(sensor, pose_graph)

        @property
        def camera_sensor(self) -> data3.CameraSensor:
            return cast(data3.CameraSensor, self._sensor)

        @property
        @override
        def model_parameters(self) -> types.ConcreteCameraModelParametersUnion:
            """Returns parameters specific to the camera's intrinsic model"""
            return self.camera_sensor.get_camera_model_parameters()

        @override
        def get_mask_images(self) -> Dict[str, PILImage.Image]:
            """Returns all named camera mask images"""

            # V3 only has a single optional ego-mask - name it 'ego' if present
            if (mask_image := self.camera_sensor.get_camera_mask_image()) is not None:
                return {"ego": mask_image}

            return {}

        @override
        def get_frame_handle(self, frame_index: int) -> CameraSensorProtocol.EncodedImageDataHandleProtocol:
            """Returns the frame's encoded image data"""
            return self.camera_sensor.get_frame_handle(frame_index)

    @override
    def get_camera_sensor(self, sensor_id: str) -> CameraSensorProtocol:
        return self.CameraSensor(
            sensor=unpack_optional(
                self._loader.get_camera_sensor(sensor_id), msg=f"Camera sensor '{sensor_id}' not found"
            ),
            pose_graph=self.pose_graph,
        )

    class RayBundleSensor(Sensor, RayBundleSensorProtocol):
        """Base ray bundle sensor implementation for V3 data (lidar/radar).

        Args:
            sensor: V3 ray bundle sensor instance (lidar or radar)
            pose_graph: Pose graph interpolator for coordinate transformations and motion compensation
        """

        def __init__(
            self,
            sensor: Union[data3.LidarSensor, data3.RadarSensor],
            pose_graph: PoseGraphInterpolator,
        ):
            super().__init__(sensor, pose_graph)

            self._motion_compensator: MotionCompensator = MotionCompensator(pose_graph)

        @property
        def point_cloud_sensor(self) -> data3.PointCloudSensor:
            return cast(data3.PointCloudSensor, self._sensor)

        @override
        def get_frame_point_cloud(
            self, frame_index: int, motion_compensation: bool, with_start_points: bool, return_index: int = 0
        ) -> RayBundleSensorProtocol.FramePointCloud:
            """Returns motion-compensated or non-motion-compensated point-cloud for a specific frame"""

            assert return_index == 0, "V3 point-cloud sensors do not support multiple returns"

            # V3 stores motion-compensated point-clouds in 'xyz_e' field
            xyz_e = self.point_cloud_sensor.get_frame_data(frame_index, "xyz_e")

            if motion_compensation:
                return RayBundleSensorProtocol.FramePointCloud(
                    motion_compensation=True,
                    xyz_m_start=self.point_cloud_sensor.get_frame_data(frame_index, "xyz_s")
                    if with_start_points
                    else None,
                    xyz_m_end=xyz_e,
                )

            # Apply motion de-compensation
            frame_timestamps_us = self.frames_timestamps_us[frame_index, :]
            xyz_m_end = self._motion_compensator.motion_decompensate_points(
                sensor_id=self.sensor_id,
                xyz_sensorend=xyz_e,
                timestamp_us=self.point_cloud_sensor.get_frame_data(frame_index, "timestamp_us"),
                frame_start_timestamp_us=int(frame_timestamps_us[FrameTimepoint.START.value]),
                frame_end_timestamp_us=int(frame_timestamps_us[FrameTimepoint.END.value]),
            )

            return RayBundleSensorProtocol.FramePointCloud(
                motion_compensation=False,
                xyz_m_start=np.zeros_like(xyz_m_end) if with_start_points else None,
                xyz_m_end=xyz_m_end,
            )

        @override
        def get_frame_ray_bundle_count(self, frame_index: int) -> int:
            """Returns the number of rays for a specific frame without decoding it"""
            return self.point_cloud_sensor.get_frame_point_count(frame_index)

        @override
        def get_frame_ray_bundle_direction(self, frame_index):
            """Returns the per-ray directions for the ray-bundle for a specific frame"""

            # V3 stores motion-compensated point clouds - we need to undo motion compensation to get directions

            pc = self.get_frame_point_cloud(
                frame_index, motion_compensation=False, with_start_points=False, return_index=0
            )

            direction = pc.xyz_m_end / np.linalg.norm(pc.xyz_m_end, axis=1, keepdims=True)

            return direction

        @override
        def get_frame_ray_bundle_timestamp_us(self, frame_index: int) -> npt.NDArray[np.uint64]:
            """Returns the per-ray timestamps for the ray-bundle for a specific frame if available (V3 lidar), otherwise replicates frame timestamp (V3 radar)"""
            if self.point_cloud_sensor.has_frame_data(frame_index, "timestamp_us"):
                return self.point_cloud_sensor.get_frame_data(frame_index, "timestamp_us")
            else:
                return np.array(
                    [self.point_cloud_sensor.get_frame_timestamp_us(frame_index, FrameTimepoint.END)]
                    * self.get_frame_ray_bundle_count(frame_index),
                    dtype=np.uint64,
                )

        @override
        def get_frame_ray_bundle_return_count(self, frame_index: int) -> int:
            """Returns the number of different ray returns for a specific frame without decoding it"""
            return 1  # V3 point-cloud sensors do not support multiple returns

        @override
        def get_frame_ray_bundle_return_distance(
            self, frame_index: int, return_index: int = 0
        ) -> npt.NDArray[np.float32]:
            """Returns the per-ray measured distances for the ray bundle return for a specific frame"""

            assert return_index == 0, "V3 point-cloud sensors do not support multiple returns"

            # V3 stores motion-compensated ray stated/end points, use their difference for per-point distances
            xyz_s = self.point_cloud_sensor.get_frame_data(frame_index, "xyz_s")
            xyz_e = self.point_cloud_sensor.get_frame_data(frame_index, "xyz_e")

            return np.linalg.norm(
                xyz_e - xyz_s,
                axis=1,
            )

    class LidarSensor(RayBundleSensor, LidarSensorProtocol):
        """Lidar sensor implementation for V3 data.

        Args:
            sensor: V3 lidar sensor instance
            pose_graph: Pose graph interpolator for coordinate transformations
        """

        def __init__(
            self,
            sensor: data3.LidarSensor,
            pose_graph: PoseGraphInterpolator,
        ):
            super().__init__(sensor, pose_graph)

        @property
        def lidar_sensor(self) -> data3.LidarSensor:
            return cast(data3.LidarSensor, self._sensor)

        @property
        @override
        def model_parameters(self) -> Optional[types.ConcreteLidarModelParametersUnion]:
            """Returns parameters specific to the lidar's intrinsic model"""
            return self.lidar_sensor.get_lidar_model_parameters()

        @override
        def get_frame_ray_bundle_model_element(self, frame_index: int) -> Optional[npt.NDArray[np.uint16]]:
            """Returns the per-ray model elements for a ray bundle for a specific frame, if available"""

            if self.lidar_sensor.has_frame_data(frame_index, "model_element"):
                return self.lidar_sensor.get_frame_data(frame_index, "model_element")

            return None

        @override
        def get_frame_ray_bundle_return_intensity(
            self, frame_index: int, return_index: int = 0
        ) -> npt.NDArray[np.float32]:
            """Returns the per-ray measured intensities for a ray bundle return for a specific frame"""

            assert return_index == 0, "V3 point-cloud sensors do not support multiple returns"

            return self.point_cloud_sensor.get_frame_data(frame_index, "intensity")

    @override
    def get_lidar_sensor(self, sensor_id: str) -> LidarSensorProtocol:
        return self.LidarSensor(
            sensor=unpack_optional(
                self._loader.get_lidar_sensor(sensor_id), msg=f"Lidar sensor '{sensor_id}' not found"
            ),
            pose_graph=self._pose_graph,
        )

    class RadarSensor(RayBundleSensor, RadarSensorProtocol):
        """Radar sensor implementation for V3 data.

        Args:
            sensor: V3 radar sensor instance
            pose_graph: Pose graph interpolator for coordinate transformations
        """

        def __init__(
            self,
            sensor: data3.RadarSensor,
            pose_graph: PoseGraphInterpolator,
        ):
            super().__init__(sensor, pose_graph)

        @property
        def radar_sensor(self) -> data3.RadarSensor:
            return cast(data3.RadarSensor, self._sensor)

    @override
    def get_radar_sensor(self, sensor_id: str) -> RadarSensorProtocol:
        return self.RadarSensor(
            sensor=unpack_optional(
                self._loader.get_radar_sensor(sensor_id), msg=f"Radar sensor '{sensor_id}' not found"
            ),
            pose_graph=self._pose_graph,
        )

    @override
    def get_cuboid_track_observations(self) -> Generator[CuboidTrackObservation]:
        """Returns all cuboid track observations in the sequence"""

        # V3 stores cuboids with each lidar frame, so iterate through all lidar sensors
        # and collect cuboid observations

        def process_frame(lidar_sensor: LidarSensor, frame_index: int) -> List[CuboidTrackObservation]:
            """Process a single frame and return cuboid observations.

            Args:
                lidar_sensor: Lidar sensor instance
                frame_index: Index of the frame to process

            Returns:
                List of cuboid track observations from this frame
            """
            frame_observations = []
            reference_frame_timestamp_us = lidar_sensor.get_frame_timestamp_us(frame_index, FrameTimepoint.END)
            for frame_label in lidar_sensor.get_frame_labels(frame_index):
                frame_observations.append(
                    # convert V3 FrameLabel3 to CuboidTrackObservation
                    CuboidTrackObservation(
                        track_id=frame_label.track_id,
                        class_id=frame_label.label_class,
                        timestamp_us=frame_label.timestamp_us,
                        # V3 cuboids labels are unconditionally relative to the
                        # sensor frame at frame end time
                        reference_frame_id=lidar_id,
                        reference_frame_timestamp_us=reference_frame_timestamp_us,
                        bbox3=frame_label.bbox3,
                        source=frame_label.source,
                        source_version=frame_label.source_version,
                    )
                )
            return frame_observations

        # Collect all frame tasks
        frame_tasks: List[Tuple[LidarSensor, int]] = []
        for lidar_id in self.lidar_ids:
            lidar_sensor = self._loader.get_lidar_sensor(lidar_id)
            for frame_index in lidar_sensor.get_frame_index_range():
                frame_tasks.append((lidar_sensor, frame_index))

        # Process frames in parallel
        with ThreadPoolExecutor() as executor:
            future_to_frame = {
                executor.submit(process_frame, lidar_sensor, frame_index): (lidar_sensor, frame_index)
                for lidar_sensor, frame_index in frame_tasks
            }

            for future in as_completed(future_to_frame):
                yield from future.result()
