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
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Protocol, Tuple, Union, cast

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
    LabelCategory,
    LabelSchema,
    LabelType,
    PointCloud,
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

    def get_cuboid_track_observations(
        self, timestamp_interval_us: Optional[HalfClosedInterval] = None
    ) -> Generator[CuboidTrackObservation]:
        """Returns all available cuboid track observations in the sequence.

        Args:
            timestamp_interval_us: If provided, only observations whose ``timestamp_us``
                falls within this half-closed interval ``[start, stop)`` are returned.
                When ``None`` (default), all observations are returned.
        """
        ...

    @property
    def point_clouds_ids(self) -> List[str]:
        """All native point-clouds source IDs in the sequence"""
        ...

    def get_point_clouds_source(self, source_id: str, *, return_index: int = 0) -> PointCloudsSourceProtocol:
        """Returns a point-clouds source for a given source id.

        For native point-clouds sources, ``return_index`` is ignored.
        For lidar/radar-adapted sources, ``return_index`` selects the ray-bundle return.
        """
        ...

    @property
    def camera_labels_ids(self) -> List[str]:
        """List of all camera label instance IDs."""
        ...

    def get_camera_labels(self, camera_label_id: str) -> CameraLabelsProtocol:
        """Get a camera labels source by instance ID."""
        ...

    def query_camera_labels(
        self,
        camera_id: str,
        label_type: Optional[LabelType] = None,
        label_category: Optional[LabelCategory] = None,
    ) -> List[CameraLabelsProtocol]:
        """Query camera label sources matching filters.

        Parameters
        ----------
        camera_id
            Camera ID to match.
        label_type
            If provided, only return sources with this exact label type.
        label_category
            If provided, only return sources whose label type category matches.
        """
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


@runtime_checkable
class PointCloudsSourceProtocol(Protocol):
    """Uniform access to point clouds from any source (native component, lidar, or radar).

    A point-clouds source exposes an indexed sequence of :class:`PointCloud` snapshots,
    each with its own reference frame and timestamp.  Per-pc generic data and metadata
    are available alongside the typed :class:`PointCloud` attributes.
    """

    @property
    def point_clouds_source_id(self) -> str:
        """Unique identifier of this point-clouds source."""
        ...

    @property
    def pcs_count(self) -> int:
        """Total number of point-cloud snapshots in this source."""
        ...

    @property
    def pc_timestamps_us(self) -> npt.NDArray[np.uint64]:
        """Snapshot timestamps in microseconds, shape ``(pcs_count,)``."""
        ...

    def get_pc(self, pc_index: int) -> PointCloud:
        """Return the point cloud at *pc_index* as an immutable :class:`PointCloud` instance.

        Schema-declared attributes are available via :meth:`PointCloud.get_attribute`.
        """
        ...

    def get_pc_generic_data_names(self, pc_index: int) -> List[str]:
        """Return the names of all generic data arrays stored for the given point cloud."""
        ...

    def has_pc_generic_data(self, pc_index: int, name: str) -> bool:
        """Check whether a named generic data array exists for the given point cloud."""
        ...

    def get_pc_generic_data(self, pc_index: int, name: str) -> npt.NDArray:
        """Return a generic data array by name for the given point cloud.

        Unlike :class:`PointCloud` attributes, generic data arrays are not necessarily
        per-point and do not participate in coordinate-frame transforms.
        """
        ...

    def get_pc_generic_meta_data(self, pc_index: int) -> Dict[str, JsonLike]:
        """Returns generic point cloud meta-data for a specific point-cloud."""
        ...

    def get_pc_index_range(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
    ) -> range:
        """Return a range of pc indices following ``start:stop:step`` slice conventions.

        Absent bounds default to the full ``[0, pcs_count)`` range.
        """

        return range(*slice(start, stop, step).indices(self.pcs_count))


class RayBundleSensorPointCloudsSourceAdapter:
    """Adapts a :class:`RayBundleSensorProtocol` (lidar or radar) into :class:`PointCloudsSourceProtocol`.

    Each sensor *frame* maps to one point cloud.  Per-point sensor data (timestamps,
    valid mask, and -- for lidar -- intensity and model element) are exposed as lazy
    :class:`PointCloud.Attribute` instances.  Sensor-level ``generic_data`` is forwarded
    through :meth:`get_pc_generic_data`.

    Additionally, generic_data entries listed in ``promote_generic_data`` are
    **auto-promoted** to :class:`PointCloud.Attribute` instances when their name
    matches (case-insensitive).  By default, ``"rgb"`` with shape ``(N, 3)``
    is promoted.  Shape validation happens lazily on first access.

    Args:
        sensor: The ray-bundle sensor to adapt.
        return_index: Which ray-bundle return to use for multi-return sensors.
        promote_generic_data: Generic-data entries to auto-promote to
            :class:`PointCloud.Attribute`.  Defaults to :data:`DEFAULT_GENERIC_DATA_PROMOTIONS`.
    """

    @dataclass(frozen=True)
    class GenericDataPromotion:
        """Shape suffix and transform type for a promoted ``generic_data`` frame entry.

        Used as the *value* in the ``promote_generic_data`` dict whose *keys*
        are the normalised (lowercase) attribute names.  Shape validation
        ``(N,) + shape_suffix`` happens lazily on first attribute access.
        """

        shape_suffix: Tuple[int, ...]
        transform_type: PointCloud.AttributeTransformType = PointCloud.AttributeTransformType.INVARIANT

    DEFAULT_GENERIC_DATA_PROMOTIONS: Dict[str, GenericDataPromotion] = {
        "rgb": GenericDataPromotion(shape_suffix=(3,)),
    }

    def __init__(
        self,
        sensor: RayBundleSensorProtocol,
        return_index: int = 0,
        promote_generic_data: Dict[str, GenericDataPromotion] = DEFAULT_GENERIC_DATA_PROMOTIONS,
    ) -> None:
        self._sensor = sensor
        self._return_index = return_index
        # Normalise keys to lowercase for case-insensitive matching
        self._promote_generic_data = {k.lower(): v for k, v in promote_generic_data.items()}

    # -- PointCloudsSourceProtocol properties -----------------------------------

    @property
    def point_clouds_source_id(self) -> str:
        return self._sensor.sensor_id

    @property
    def pcs_count(self) -> int:
        return self._sensor.frames_count

    @property
    def pc_timestamps_us(self) -> "npt.NDArray[np.uint64]":
        return self._sensor.frames_timestamps_us[:, FrameTimepoint.END.value]

    # -- PointCloudsSourceProtocol methods --------------------------------------

    def get_pc(self, pc_index: int) -> PointCloud:
        sensor = self._sensor
        return_index = self._return_index

        # xyz from motion-compensated point cloud
        frame_pc = sensor.get_frame_point_cloud(
            pc_index,
            motion_compensation=True,
            with_start_points=False,
            return_index=return_index,
        )

        # Build lazy attributes
        attributes: Dict[str, PointCloud.Attribute] = {}

        # timestamp_us -- per-ray timestamps (INVARIANT)
        def _load_timestamp_us(s: RayBundleSensorProtocol = sensor, i: int = pc_index) -> "npt.NDArray":
            return s.get_frame_ray_bundle_timestamp_us(i)

        attributes["timestamp_us"] = PointCloud.Attribute(
            loader=_load_timestamp_us,
            transform_type=PointCloud.AttributeTransformType.INVARIANT,
        )

        # valid_mask -- per-ray valid mask for the selected return (INVARIANT)
        def _load_valid_mask(
            s: RayBundleSensorProtocol = sensor,
            i: int = pc_index,
            ri: int = return_index,
        ) -> "npt.NDArray":
            return s.get_frame_ray_bundle_return_valid_mask(i, ri)

        attributes["valid_mask"] = PointCloud.Attribute(
            loader=_load_valid_mask,
            transform_type=PointCloud.AttributeTransformType.INVARIANT,
        )

        # Lidar-specific attributes (determined at runtime)
        if isinstance(sensor, LidarSensorProtocol):
            lidar = cast(LidarSensorProtocol, sensor)

            # intensity -- per-ray intensity for the selected return (INVARIANT)
            def _load_intensity(
                s: LidarSensorProtocol = lidar,
                i: int = pc_index,
                ri: int = return_index,
            ) -> "npt.NDArray":
                return s.get_frame_ray_bundle_return_intensity(i, ri)

            attributes["intensity"] = PointCloud.Attribute(
                loader=_load_intensity,
                transform_type=PointCloud.AttributeTransformType.INVARIANT,
            )

            # model_element -- per-ray model element, only if present (INVARIANT)
            model_element = lidar.get_frame_ray_bundle_model_element(pc_index)
            if model_element is not None:
                captured_model_element: "npt.NDArray" = model_element

                def _load_model_element(c: "npt.NDArray" = captured_model_element) -> "npt.NDArray":
                    return c

                attributes["model_element"] = PointCloud.Attribute(
                    loader=_load_model_element,
                    transform_type=PointCloud.AttributeTransformType.INVARIANT,
                )

        # Auto-promote matching generic_data entries to PointCloud attributes
        N = frame_pc.xyz_m_end.shape[0]
        for gd_name in sensor.get_frame_generic_data_names(pc_index):
            gd_name_normalized = gd_name.lower()
            promotion = self._promote_generic_data.get(gd_name_normalized)
            if promotion is not None and gd_name_normalized not in attributes:
                promo_suffix = promotion.shape_suffix
                promo_tt = promotion.transform_type

                def _load_promoted(
                    s: RayBundleSensorProtocol = sensor,
                    i: int = pc_index,
                    n: str = gd_name,
                    expected_n: int = N,
                    suffix: Tuple[int, ...] = promo_suffix,
                ) -> "npt.NDArray":
                    arr = s.get_frame_generic_data(i, n)
                    expected_shape = (expected_n,) + suffix
                    assert arr.shape == expected_shape, (
                        f"Promoted generic_data '{n}' shape {arr.shape} != expected {expected_shape}"
                    )
                    return arr

                attributes[gd_name_normalized] = PointCloud.Attribute(
                    loader=_load_promoted,
                    transform_type=promo_tt,
                )

        # Reference frame: sensor coordinate frame at end-of-frame time
        ref_frame_timestamp_us = int(sensor.frames_timestamps_us[pc_index, FrameTimepoint.END.value])

        return PointCloud(
            _xyz=frame_pc.xyz_m_end,
            reference_frame_id=sensor.sensor_id,
            reference_frame_timestamp_us=ref_frame_timestamp_us,
            coordinate_unit=PointCloud.CoordinateUnit.METERS,
            _attributes=attributes,
        )

    def get_pc_generic_data_names(self, pc_index: int) -> List[str]:
        return self._sensor.get_frame_generic_data_names(pc_index)

    def has_pc_generic_data(self, pc_index: int, name: str) -> bool:
        return self._sensor.has_frame_generic_data(pc_index, name)

    def get_pc_generic_data(self, pc_index: int, name: str) -> "npt.NDArray":
        return self._sensor.get_frame_generic_data(pc_index, name)

    def get_pc_generic_meta_data(self, pc_index: int) -> Dict[str, JsonLike]:
        return self._sensor.get_frame_generic_meta_data(pc_index)

    def get_pc_index_range(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
    ) -> range:
        return range(*slice(start, stop, step).indices(self.pcs_count))


@runtime_checkable
class CameraLabelsProtocol(Protocol):
    """Protocol for accessing camera-associated image labels.

    Each instance provides labels of one type for one camera, with independently-managed timestamps.
    """

    @property
    def camera_id(self) -> str:
        """Camera ID this label instance is associated with."""
        ...

    @property
    def label_type(self) -> LabelType:
        """Resolved label type."""
        ...

    @property
    def label_schema(self) -> LabelSchema:
        """Schema describing the label data format."""
        ...

    @property
    def labels_count(self) -> int:
        """Number of stored labels."""
        ...

    @property
    def label_timestamps_us(self) -> npt.NDArray[np.uint64]:
        """Timestamps of all stored labels, sorted ascending."""
        ...

    @property
    def labels_generic_meta_data(self) -> Dict[str, JsonLike]:
        """Generic metadata associated with all labels."""
        ...

    @runtime_checkable
    class CameraLabelHandleProtocol(Protocol):
        """Protocol for a single camera label at a specific timestamp.

        Returned by :meth:`CameraLabelsProtocol.get_label` and provides deferred
        access to the label data and metadata.
        """

        @property
        def schema(self) -> LabelSchema:
            """Schema describing the label data format."""
            ...

        @property
        def timestamp_us(self) -> int:
            """Timestamp of this label in microseconds."""
            ...

        @property
        def generic_meta_data(self) -> Dict[str, JsonLike]:
            """Per-label generic metadata."""
            ...

        def get_data(self) -> npt.NDArray[Any]:
            """Load and return the label data as a numpy array."""
            ...

        def get_encoded_data(self) -> Optional[bytes]:
            """Return raw encoded bytes for IMAGE_ENCODED labels, or None for RAW."""
            ...

    def get_label(self, timestamp_us: int) -> CameraLabelHandleProtocol:
        """Return a lazy handle to the label data at the given timestamp."""
        ...
