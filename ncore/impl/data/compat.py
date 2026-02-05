# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Abstract compatibility layer for unified access to different NCore sequence data formats.

This module provides protocol-based interfaces and adapter implementations that enable
unified access to both different NCore data formats through a common API.

For instance, while deprecated V3 is a monolithic shard-based format, V4 is a more
general component-based format.

Key Components:
    - SequenceLoaderProtocol: Unified interface for sequence-level data access
    - SensorProtocol: Common interface for sensor data (cameras, lidars, radars)
    - CameraSensorProtocol: Camera-specific extensions
    - RayBundleSensorProtocol: Ray bundle sensor interface (lidar/radar)
    - LidarSensorProtocol: Lidar-specific extensions

The compatibility layer handles differences between different NCore data formats including:
    - Different storage mechanisms (shards vs. components)
    - Motion compensation conventions (e.g., deprecated V3 stores compensated, V4 stores uncompensated)
    - Pose graph construction and transformation APIs
    - Frame indexing and timestamp access patterns
    - Metadata retrieval

Example:
    # Load V4 data
    reader = SequenceComponentGroupsReader([Path("file.zarr.itar"), Path("some/folder.zarr")])
    loader = SequenceLoaderV4(reader)

    # Load V3 data [deprecated]
    shard_loader = ShardDataLoader(["data_shard_*.zarr.itar"])
    loader = SequenceLoaderV3(shard_loader)

    # Use unified API for either format
    camera = loader.get_camera_sensor("camera_front")
    image = camera.get_frame_image(0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Protocol, Union

import numpy as np
import PIL.Image as PILImage

from typing_extensions import runtime_checkable
from upath import UPath

from ncore.impl.common.transformations import HalfClosedInterval, PoseGraphInterpolator
from ncore.impl.data.types import (
    ConcreteCameraModelParametersUnion,
    ConcreteLidarModelParametersUnion,
    CuboidTrackObservation,
    EncodedImageData,
    FrameTimepoint,
    JsonLike,
)
from ncore.impl.data.util import closest_index_sorted


if TYPE_CHECKING:
    import numpy.typing as npt  # type: ignore[import-not-found]


@runtime_checkable
class SequenceLoaderProtocol(Protocol):
    """SequenceLoaderProtocol provides unified access to a relevant subset of common NCore sequence data APIs"""

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
        """Returns all available cuboid track observations in the sequence"""
        ...


@runtime_checkable
class SensorProtocol(Protocol):
    """SensorProtocol provides unified access to a relevant subset of common NCore sensor data APIs"""

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

    ## Poses API (uses pose-graph exclusively)
    @property
    def pose_graph(self) -> PoseGraphInterpolator:
        """Access the sensor-associated pose graph (usually the sequence-wide one, unless overwritten)"""
        return self._pose_graph

    def set_pose_graph(self, pose_graph: PoseGraphInterpolator) -> None:
        """Assigns a new pose graph to the sensor instance (e.g., to overwrite / use a sensor-exclusive pose graph)"""
        self._pose_graph = pose_graph

    ## Convenience pose accessors for common sensor transforms
    @property
    def T_sensor_rig(self, rig_node: str = "rig") -> Optional[npt.NDArray[np.floating]]:
        """Return static extrinsic transformation from sensor to named rig coordinate frame.

        Returns the 4x4 homogeneous transformation matrix T_sensor_rig that transforms
        points from the sensor coordinate frame to the rig coordinate frame.

        Args:
            rig_node: Name of the rig coordinate frame (default: "rig")

        Returns:
            4x4 transformation matrix if the static transformation exists, None otherwise
        """
        try:
            return self._pose_graph.evaluate_poses(
                source_node=self.sensor_id,
                target_node=rig_node,
                timestamps_us=np.empty((), dtype=np.uint64),
            )
        except KeyError:
            return None

    # Generic relative pose evaluation
    def get_frames_T_source_sensor(
        self,
        source_node: str,
        frame_indices: Union[int, npt.NDArray[np.integer]],
        frame_timepoint: Optional[FrameTimepoint] = FrameTimepoint.END,
    ) -> npt.NDArray[np.floating]:
        """Evaluates relative sensor-relative poses at timestamps inferred from frame indices.

        Computes transformation matrices T_source_sensor that transform points from the
        source coordinate frame to the sensor coordinate frame at specified frame times.

        Args:
            source_node: Name of the source coordinate frame
            frame_indices: Individual index or array of frame indices at which to evaluate poses
            frame_timepoint: Frame-relative timepoint (START or END). If None, returns both

        Returns:
            Transformation matrices with shape [frame_indices-shape,2,4,4] if frame_timepoint
            is None (both start and end poses), else [frame_indices-shape,4,4] (single timepoint)
        """

        return self.get_frames_T_source_target(
            source_node=source_node,
            target_node=self.sensor_id,
            frame_indices=frame_indices,
            frame_timepoint=frame_timepoint,
        )

    def get_frames_T_sensor_target(
        self,
        target_node: str,
        frame_indices: Union[int, npt.NDArray[np.integer]],
        frame_timepoint: Optional[FrameTimepoint] = FrameTimepoint.END,
    ) -> npt.NDArray[np.floating]:
        """Evaluates relative poses of the sensor at timestamps inferred from frame indices.

        Computes transformation matrices T_sensor_target that transform points from the
        sensor coordinate frame to the target coordinate frame at specified frame times.

        Args:
            target_node: Name of the target coordinate frame
            frame_indices: Individual index or array of frame indices at which to evaluate poses
            frame_timepoint: Frame-relative timepoint (START or END). If None, returns both

        Returns:
            Transformation matrices with shape [frame_indices-shape,2,4,4] if frame_timepoint
            is None (both start and end poses), else [frame_indices-shape,4,4] (single timepoint)
        """

        return self.get_frames_T_source_target(
            source_node=self.sensor_id,
            target_node=target_node,
            frame_indices=frame_indices,
            frame_timepoint=frame_timepoint,
        )

    def get_frames_T_source_target(
        self,
        source_node: str,
        target_node: str,
        frame_indices: Union[int, npt.NDArray[np.integer]],
        frame_timepoint: Optional[FrameTimepoint] = FrameTimepoint.END,
    ) -> npt.NDArray[np.floating]:
        """Evaluates relative poses at timestamps inferred from frame indices.

        Computes transformation matrices T_source_target that transform points from the
        source coordinate frame to the target coordinate frame at specified frame times.

        Args:
            source_node: Name of the source coordinate frame (defaults to sensor_id if None)
            target_node: Name of the target coordinate frame
            frame_indices: Individual index or array of frame indices at which to evaluate poses
            frame_timepoint: Frame-relative timepoint (START or END). If None, returns both

        Returns:
            Transformation matrices with shape [frame_indices-shape,2,4,4] if frame_timepoint
            is None (both start and end poses), else [frame_indices-shape,4,4] (single timepoint)
        """

        if isinstance(frame_indices, int):
            frame_indices = np.array(frame_indices, dtype=np.int64)

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

        return closest_index_sorted(target_timestamps_us, timestamp_us)


@runtime_checkable
class CameraSensorProtocol(SensorProtocol, Protocol):
    """CameraSensorProtocol provides unified access to a relevant subset of common NCore camera sensor APIs"""

    @property
    def model_parameters(self) -> ConcreteCameraModelParametersUnion:
        """Returns parameters specific to the camera's intrinsic model"""
        ...

    def get_mask_images(self) -> Dict[str, PILImage.Image]:
        """Returns all available named camera mask images"""
        ...

    # Image Frame Data
    @runtime_checkable
    class EncodedImageDataHandleProtocol(Protocol):
        """References encoded image data without loading it"""

        def get_data(self) -> EncodedImageData:
            """Loads the referenced encoded image data to memory"""
            ...

    def get_frame_handle(self, frame_index: int) -> EncodedImageDataHandleProtocol:
        """Returns the frame's encoded image data"""
        ...

    def get_frame_data(self, frame_index: int) -> EncodedImageData:
        """Returns the frame's encoded image data"""
        return self.get_frame_handle(frame_index).get_data()

    def get_frame_image(self, frame_index: int) -> PILImage.Image:
        """Returns the frame's decoded image data"""
        return self.get_frame_data(frame_index).get_decoded_image()

    def get_frame_image_array(self, frame_index: int) -> npt.NDArray[np.uint8]:
        """Returns decoded image data as array [W,H,C]"""
        return np.asarray(self.get_frame_image(frame_index))


@runtime_checkable
class RayBundleSensorProtocol(SensorProtocol, Protocol):
    """RayBundleSensorProtocol provides unified access to a relevant subset of common NCore ray-bundle sensor APIs"""

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

    def get_frame_ray_bundle_return_distance_m(
        self, frame_index: int, return_index: int = 0
    ) -> npt.NDArray[np.float32]:
        """Returns the per-ray measured metric distances for the ray bundle returns of a specific frame.

        Args:
            frame_index: Index of the frame
            return_index: Index of the ray bundle return to retrieve (for multi-return sensors)

        Returns:
            Array of per-ray metric distances [N,]
        """
        ...

    def get_frame_ray_bundle_return_valid_mask(self, frame_index: int, return_index: int = 0) -> npt.NDArray[np.bool_]:
        """Returns the per-ray valid mask for the ray bundle returns of a specific frame.

        Args:
            frame_index: Index of the frame
            return_index: Index of the ray bundle return to retrieve (for multi-return sensors)

        Returns:
            Array of per-ray return valid masks [N]
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
        self,
        frame_index: int,
        motion_compensation: bool,
        with_start_points: bool,
        return_index: int = 0,
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


@runtime_checkable
class LidarSensorProtocol(RayBundleSensorProtocol, Protocol):
    """LidarSensorProtocol provides unified access to a relevant subset of common NCore lidar sensor APIs"""

    @property
    def model_parameters(self) -> Optional[ConcreteLidarModelParametersUnion]:
        """Returns parameters specific to the lidar's intrinsic model (optional as not mandatory)"""
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


@runtime_checkable
class RadarSensorProtocol(RayBundleSensorProtocol, Protocol):
    """RadarSensorProtocol provides unified access to a relevant subset of common NCore radar sensor APIs"""

    ...
