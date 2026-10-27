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

import asyncio

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Sequence, Union, cast

import numpy as np
import PIL.Image as PILImage

from typing_extensions import override
from upath import UPath

from ncore.impl.common.transformations import HalfClosedInterval, MotionCompensator, PoseGraphInterpolator
from ncore.impl.common.util import unpack_optional
from ncore.impl.data.compat import (
    AsyncCameraLabelsProtocol,
    AsyncCameraSensorProtocol,
    AsyncLidarSensorProtocol,
    AsyncPointCloudsSourceProtocol,
    AsyncRadarSensorProtocol,
    AsyncRayBundleSensorPointCloudsSourceAdapter,
    AsyncRayBundleSensorProtocol,
    AsyncSensorProtocol,
    AsyncSequenceLoaderProtocol,
    RayBundleSensorProtocol,
)
from ncore.impl.data.json_types import JsonDocument
from ncore.impl.data.types import (
    CameraLabelDescriptor,
    CameraModelParameters,
    CuboidTrackObservation,
    FrameTimepoint,
    LabelCategory,
    LabelType,
    LidarModelParameters,
    PointCloud,
)
from ncore.impl.data.v4.components import (
    BaseRayBundleSensorComponentReader,
    CameraLabelsComponent,
    CameraSensorComponent,
    CuboidsComponent,
    IntrinsicsComponent,
    LidarSensorComponent,
    MasksComponent,
    PointCloudsComponent,
    PosesComponent,
    RadarSensorComponent,
    SequenceComponentGroupsReader,
)


if TYPE_CHECKING:
    import numpy.typing as npt


class SequenceLoaderV4(AsyncSequenceLoaderProtocol):
    """SequenceLoader implementation for NCore V4 data (with asynchronous reads, see :class:`AsyncSequenceLoaderProtocol`).

    The constructor takes an opened (synchronously or asynchronously) :class:`SequenceComponentGroupsReader`, see
    :meth:`open_async` for opening a sequence's component stores and the loader asynchronously in one step.

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
        masks_component_group_name: Optional[str] = "default",
        cuboids_component_group_name: Optional[str] = "default",
    ):
        self._reader: SequenceComponentGroupsReader = reader

        # open all mandatory component readers
        poses_readers = self._reader.open_component_readers(PosesComponent.Reader)
        assert poses_component_group_name in poses_readers, (
            f"PosesComponent group '{poses_component_group_name}' not found"
        )
        self._poses_reader: PosesComponent.Reader = poses_readers[poses_component_group_name]

        intrinsics_readers = self._reader.open_component_readers(IntrinsicsComponent.Reader)
        assert intrinsics_component_group_name in intrinsics_readers, (
            f"IntrinsicsComponent group '{intrinsics_component_group_name}' not found"
        )
        self._intrinsics_reader: IntrinsicsComponent.Reader = intrinsics_readers[intrinsics_component_group_name]

        # open optional component readers if available
        self._masks_reader: Optional[MasksComponent.Reader] = None
        if masks_component_group_name is not None:
            masks_readers = self._reader.open_component_readers(MasksComponent.Reader)
            assert masks_component_group_name in masks_readers, (
                f"MasksComponent group '{masks_component_group_name}' not found"
            )
            self._masks_reader = masks_readers[masks_component_group_name]

        self._cuboids_reader: Optional[CuboidsComponent.Reader] = None
        if cuboids_component_group_name is not None:
            cuboids_readers = self._reader.open_component_readers(CuboidsComponent.Reader)
            assert cuboids_component_group_name in cuboids_readers, (
                f"CuboidsComponent group '{cuboids_component_group_name}' not found"
            )
            self._cuboids_reader = cuboids_readers[cuboids_component_group_name]
        self._cuboid_observations_cache: Optional[List[CuboidTrackObservation]] = None

        self._cameras_readers: Dict[str, CameraSensorComponent.Reader] = self._reader.open_component_readers(
            CameraSensorComponent.Reader
        )
        self._lidars_readers: Dict[str, LidarSensorComponent.Reader] = self._reader.open_component_readers(
            LidarSensorComponent.Reader
        )
        self._radars_readers: Dict[str, RadarSensorComponent.Reader] = self._reader.open_component_readers(
            RadarSensorComponent.Reader
        )

        self._point_clouds_readers: Dict[str, PointCloudsComponent.Reader] = self._reader.open_component_readers(
            PointCloudsComponent.Reader
        )

        self._camera_labels_readers: Dict[str, CameraLabelsComponent.Reader] = self._reader.open_component_readers(
            CameraLabelsComponent.Reader
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

    @classmethod
    async def open_async(
        cls,
        component_group_paths: Union[List[UPath], List[Path]],
        poses_component_group_name: str = "default",
        intrinsics_component_group_name: str = "default",
        masks_component_group_name: Optional[str] = "default",
        cuboids_component_group_name: Optional[str] = "default",
        open_consolidated: bool = True,
        itar_index_tail_read_size: int = 1 << 20,
        itar_remote_range_reads: bool = False,
        max_concurrent_requests: Union[int, asyncio.Semaphore, None] = None,
        node_cache_size: Optional[int] = None,
    ) -> SequenceLoaderV4:
        """Opens a sequence asynchronously, without blocking the running event loop (see
        :meth:`SequenceComponentGroupsReader.open_async` and :meth:`__init__` for the parameters).

        Opening many sequences concurrently overlaps their requests, e.g. with a shared bound of the requests in
        flight:

        .. code-block:: python

            limit = asyncio.Semaphore(64)
            loaders = await asyncio.gather(
                *(SequenceLoaderV4.open_async([path], max_concurrent_requests=limit) for path in sequence_paths)
            )
        """
        reader = await SequenceComponentGroupsReader.open_async(
            component_group_paths,
            open_consolidated=open_consolidated,
            itar_index_tail_read_size=itar_index_tail_read_size,
            itar_remote_range_reads=itar_remote_range_reads,
            max_concurrent_requests=max_concurrent_requests,
            node_cache_size=node_cache_size,
        )
        # (initializing the loader reads no data: component readers are initialized from the opened hierarchies)
        return cls(
            reader,
            poses_component_group_name=poses_component_group_name,
            intrinsics_component_group_name=intrinsics_component_group_name,
            masks_component_group_name=masks_component_group_name,
            cuboids_component_group_name=cuboids_component_group_name,
        )

    @property
    @override
    def sequence_id(self) -> str:
        return self._reader.sequence_id

    @property
    @override
    def generic_meta_data(self) -> JsonDocument:
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
    def get_sequence_meta(self) -> JsonDocument:
        return cast(JsonDocument, self._reader.get_sequence_meta().to_dict())

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

    class Sensor(AsyncSensorProtocol):
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
        async def get_frame_generic_data_async(self, frame_index: int, name: str) -> npt.NDArray[Any]:
            """Returns generic frame-data for a specific frame and name, read asynchronously"""

            return await self._reader.get_frame_generic_data_async(
                self.frames_timestamps_us[frame_index, FrameTimepoint.END.value].item(), name
            )

        @override
        def get_frame_generic_meta_data(self, frame_index: int) -> JsonDocument:
            """Returns generic frame meta-data for a specific frame"""

            return self._reader.get_frame_generic_meta_data(
                self.frames_timestamps_us[frame_index, FrameTimepoint.END.value].item()
            )

    class CameraSensor(Sensor, AsyncCameraSensorProtocol):
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
            mask_reader: Optional[MasksComponent.Reader],
            model_parameters: CameraModelParameters,
            pose_graph: PoseGraphInterpolator,
        ):
            super().__init__(reader, pose_graph)

            self._mask_reader: Optional[MasksComponent.Reader] = mask_reader
            self._model_parameters: CameraModelParameters = model_parameters

        @property
        def camera_reader(self) -> CameraSensorComponent.Reader:
            return cast(CameraSensorComponent.Reader, self._reader)

        @property
        @override
        def model_parameters(self) -> CameraModelParameters:
            """Returns parameters specific to the camera's intrinsic model"""
            return self._model_parameters

        @override
        def get_mask_images(self) -> Dict[str, PILImage.Image]:
            """Returns all available named camera mask images"""

            # No mask reader available, return empty dict
            if self._mask_reader is None:
                return {}

            return dict(self._mask_reader.get_camera_mask_images(self.sensor_id))

        @override
        async def get_mask_images_async(self) -> Dict[str, PILImage.Image]:
            """Returns all available named camera mask images, read asynchronously (concurrently)"""

            if (mask_reader := self._mask_reader) is None:
                return {}

            mask_names = mask_reader.get_camera_mask_names(self.sensor_id)
            images = await asyncio.gather(
                *(mask_reader.get_camera_mask_image_async(self.sensor_id, mask_name) for mask_name in mask_names)
            )
            return dict(zip(mask_names, images))

        @override
        def get_frame_handle(self, frame_index: int) -> AsyncCameraSensorProtocol.EncodedImageDataHandleProtocol:
            """Returns the frame's encoded image data"""
            return self.camera_reader.get_frame_handle(
                self.frames_timestamps_us[frame_index, FrameTimepoint.END.value].item()
            )

    @override
    def get_camera_sensor(self, sensor_id: str) -> AsyncCameraSensorProtocol:
        return self.CameraSensor(
            reader=unpack_optional(
                self._cameras_readers.get(sensor_id),
                msg=f"Camera sensor '{sensor_id}' not available in loaded camera sensor groups",
            ),
            mask_reader=self._masks_reader,
            pose_graph=self._pose_graph,
            model_parameters=self._intrinsics_reader.get_camera_model_parameters(sensor_id),
        )

    class RayBundleSensor(Sensor, AsyncRayBundleSensorProtocol):
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
        def get_frame_ray_bundle_direction(self, frame_index: int) -> "npt.NDArray[np.float32]":
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
        async def get_frame_ray_bundle_direction_async(self, frame_index: int) -> "npt.NDArray[np.float32]":
            """Returns the per-ray directions for the ray-bundle for a specific frame, read asynchronously"""
            return await self.ray_bundle_reader.get_frame_ray_bundle_data_async(
                self.frames_timestamps_us[frame_index, FrameTimepoint.END.value].item(), "direction"
            )

        @override
        async def get_frame_ray_bundle_timestamp_us_async(self, frame_index: int) -> npt.NDArray[np.uint64]:
            """Returns the per-ray timestamps for the ray-bundle for a specific frame, read asynchronously"""
            return await self.ray_bundle_reader.get_frame_ray_bundle_data_async(
                self.frames_timestamps_us[frame_index, FrameTimepoint.END.value].item(), "timestamp_us"
            )

        @override
        def get_frame_point_cloud(
            self, frame_index: int, motion_compensation: bool, with_start_points: bool, return_index: int = 0
        ) -> RayBundleSensorProtocol.FramePointCloud:
            """Returns motion-compensated or non-motion-compensated point-cloud for a specific frame"""
            end_timestamp_us = int(self.frames_timestamps_us[frame_index, FrameTimepoint.END.value])
            reader = self.ray_bundle_reader

            # V4 stores non-motion-compensated ray directions in 'direction' field and return-specific distances
            return self._frame_point_cloud(
                frame_index,
                motion_compensation,
                with_start_points,
                direction=reader.get_frame_ray_bundle_data(end_timestamp_us, "direction"),
                distance_m=reader.get_frame_ray_bundle_return_data(
                    end_timestamp_us, "distance_m", return_index=return_index
                ),
                timestamp_us=(
                    reader.get_frame_ray_bundle_data(end_timestamp_us, "timestamp_us") if motion_compensation else None
                ),
            )

        @override
        async def get_frame_point_cloud_async(
            self, frame_index: int, motion_compensation: bool, with_start_points: bool, return_index: int = 0
        ) -> RayBundleSensorProtocol.FramePointCloud:
            """Returns a frame point cloud like :meth:`get_frame_point_cloud`, with its data read asynchronously
            (concurrently)"""
            end_timestamp_us = int(self.frames_timestamps_us[frame_index, FrameTimepoint.END.value])
            reader = self.ray_bundle_reader

            async def no_timestamps() -> None:
                return None

            direction, distance_m, timestamp_us = await asyncio.gather(
                reader.get_frame_ray_bundle_data_async(end_timestamp_us, "direction"),
                reader.get_frame_ray_bundle_return_data_async(
                    end_timestamp_us, "distance_m", return_index=return_index
                ),
                reader.get_frame_ray_bundle_data_async(end_timestamp_us, "timestamp_us")
                if motion_compensation
                else no_timestamps(),
            )
            return self._frame_point_cloud(
                frame_index,
                motion_compensation,
                with_start_points,
                direction=direction,
                distance_m=distance_m,
                timestamp_us=timestamp_us,
            )

        def _frame_point_cloud(
            self,
            frame_index: int,
            motion_compensation: bool,
            with_start_points: bool,
            direction: npt.NDArray[np.float32],
            distance_m: npt.NDArray[np.float32],
            timestamp_us: Optional[npt.NDArray[np.uint64]],
        ) -> RayBundleSensorProtocol.FramePointCloud:
            """The (optionally motion-compensated) point cloud of a frame from its ray data (shared by the synchronous
            and asynchronous reads).

            Args:
                frame_index: Index of the frame
                motion_compensation: Whether to motion-compensate the points
                with_start_points: Whether to include the ray segment start points
                direction: Per-ray directions [N,3]
                distance_m: Per-ray distances of the requested return [N,]
                timestamp_us: Per-ray timestamps [N,], required if `motion_compensation` (unused otherwise, may be
                    None)
            """
            frame_timestamps_us = self.frames_timestamps_us[frame_index, :]

            xyz_m = direction * distance_m[:, np.newaxis]

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
                timestamp_us=unpack_optional(
                    timestamp_us, msg="per-ray timestamps are required for motion compensation"
                ),
                frame_start_timestamp_us=int(frame_timestamps_us[FrameTimepoint.START.value]),
                frame_end_timestamp_us=int(frame_timestamps_us[FrameTimepoint.END.value]),
            )
            return RayBundleSensorProtocol.FramePointCloud(
                motion_compensation=True,
                xyz_m_start=motion_compensation_result.xyz_s_reftime if with_start_points else None,
                xyz_m_end=motion_compensation_result.xyz_e_reftime,
            )

        @override
        def get_frame_ray_bundle_return_distance_m(
            self, frame_index: int, return_index: int = 0
        ) -> npt.NDArray[np.float32]:
            """Returns the per-ray measured metric distances for the ray bundle return for a specific frame"""

            # V4 stores non-motion-compensated point cloud in 'xyz_m' field, so we can use it's norm
            # as the measured distance
            return self.ray_bundle_reader.get_frame_ray_bundle_return_data(
                self.frames_timestamps_us[frame_index, FrameTimepoint.END.value].item(),
                "distance_m",
                return_index=return_index,
            )

        @override
        def get_frame_ray_bundle_return_valid_mask(
            self, frame_index: int, return_index: int = 0
        ) -> npt.NDArray[np.bool_]:
            """Returns the per-ray valid mask of the ray bundle returns for a specific frame"""

            return self.ray_bundle_reader.get_frame_ray_bundle_return_valid_mask(
                self.frames_timestamps_us[frame_index, FrameTimepoint.END.value].item(),
            )[return_index]

        @override
        async def get_frame_ray_bundle_return_distance_m_async(
            self, frame_index: int, return_index: int = 0
        ) -> npt.NDArray[np.float32]:
            """Returns the per-ray measured metric distances for the ray bundle return for a specific frame, read
            asynchronously"""
            return await self.ray_bundle_reader.get_frame_ray_bundle_return_data_async(
                self.frames_timestamps_us[frame_index, FrameTimepoint.END.value].item(),
                "distance_m",
                return_index=return_index,
            )

        @override
        async def get_frame_ray_bundle_return_valid_mask_async(
            self, frame_index: int, return_index: int = 0
        ) -> npt.NDArray[np.bool_]:
            """Returns the per-ray valid mask of the ray bundle returns for a specific frame, read asynchronously"""
            return (
                await self.ray_bundle_reader.get_frame_ray_bundle_return_valid_mask_async(
                    self.frames_timestamps_us[frame_index, FrameTimepoint.END.value].item(),
                )
            )[return_index]

    class LidarSensor(RayBundleSensor, AsyncLidarSensorProtocol):
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
            model_parameters: Optional[LidarModelParameters],
        ):
            super().__init__(reader, pose_graph)

            self._model_parameters: Optional[LidarModelParameters] = model_parameters

        @property
        def lidar_reader(self) -> LidarSensorComponent.Reader:
            return cast(LidarSensorComponent.Reader, self._reader)

        @property
        @override
        def model_parameters(self) -> Optional[LidarModelParameters]:
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
        async def get_frame_ray_bundle_model_element_async(self, frame_index: int) -> Optional[npt.NDArray[np.uint16]]:
            """Returns the per-ray model elements for a ray bundle for a specific frame (None if not available), read
            asynchronously"""

            if self.lidar_reader.has_frame_ray_bundle_data(
                timestamp_us := self.frames_timestamps_us[frame_index, FrameTimepoint.END.value].item(), "model_element"
            ):
                return await self.lidar_reader.get_frame_ray_bundle_data_async(timestamp_us, "model_element")

            return None

        @override
        async def get_frame_ray_bundle_return_intensity_async(
            self, frame_index: int, return_index: int = 0
        ) -> npt.NDArray[np.float32]:
            """Returns the per-ray measured intensities for a ray bundle return for a specific frame, read
            asynchronously"""

            return await self.lidar_reader.get_frame_ray_bundle_return_data_async(
                self.frames_timestamps_us[frame_index, FrameTimepoint.END.value].item(),
                "intensity",
                return_index=return_index,
            )

    @override
    def get_lidar_sensor(self, sensor_id: str) -> AsyncLidarSensorProtocol:
        return self.LidarSensor(
            reader=unpack_optional(
                self._lidars_readers.get(sensor_id),
                msg=f"Lidar sensor '{sensor_id}' not available in loaded lidar sensor groups",
            ),
            pose_graph=self._pose_graph,
            model_parameters=self._intrinsics_reader.get_lidar_model_parameters(sensor_id),
        )

    class RadarSensor(RayBundleSensor, AsyncRadarSensorProtocol):
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
    def get_radar_sensor(self, sensor_id: str) -> AsyncRadarSensorProtocol:
        return self.RadarSensor(
            reader=unpack_optional(
                self._radars_readers.get(sensor_id),
                msg=f"Radar sensor '{sensor_id}' not available in loaded radar sensor groups",
            ),
            pose_graph=self._pose_graph,
        )

    class CameraLabels(AsyncCameraLabelsProtocol):
        """Wraps a :class:`CameraLabelsComponent.Reader` to implement :class:`AsyncCameraLabelsProtocol`."""

        def __init__(self, reader: CameraLabelsComponent.Reader) -> None:
            self._reader = reader

        @property
        @override
        def label_descriptor(self) -> CameraLabelDescriptor:
            return self._reader.label_descriptor

        @property
        @override
        def labels_count(self) -> int:
            return self._reader.labels_count

        @property
        @override
        def label_timestamps_us(self) -> npt.NDArray[np.uint64]:
            return self._reader.timestamps_us

        @property
        @override
        def labels_generic_meta_data(self) -> JsonDocument:
            return self._reader.generic_meta_data

        @override
        def get_label(self, timestamp_us: int) -> CameraLabelsComponent.Reader.CameraLabelHandle:
            return self._reader.get_label(timestamp_us)

    class PointCloudsSource(AsyncPointCloudsSourceProtocol):
        """Native point-clouds source wrapping a :class:`PointCloudsComponent.Reader`.

        Args:
            reader: Component reader for the point-clouds data.
        """

        def __init__(self, reader: PointCloudsComponent.Reader) -> None:
            self._reader = reader

        @property
        @override
        def point_clouds_source_id(self) -> str:
            return self._reader.instance_name

        @property
        @override
        def pcs_count(self) -> int:
            return self._reader.pcs_count

        @property
        @override
        def pc_timestamps_us(self) -> npt.NDArray[np.uint64]:
            return self._reader.pc_timestamps_us

        @override
        def get_pc(self, pc_index: int) -> PointCloud:
            return self._point_cloud(pc_index, self._reader.get_pc_xyz(pc_index))

        @override
        async def get_pc_async(self, pc_index: int) -> PointCloud:
            return self._point_cloud(pc_index, await self._reader.get_pc_xyz_async(pc_index))

        def _point_cloud(self, pc_index: int, xyz: npt.NDArray[np.float32]) -> PointCloud:
            """The point cloud of a pc index with its points `xyz` (attributes are loaded lazily)"""
            reader = self._reader

            # Build lazy attributes from the declared schema
            attributes: Dict[str, PointCloud.Attribute] = {}
            for attr_name in reader.attribute_names:
                schema = reader.get_attribute_schema(attr_name)

                def _load_attr(
                    r: PointCloudsComponent.Reader = reader,
                    idx: int = pc_index,
                    name: str = attr_name,
                ) -> "npt.NDArray":
                    return r.get_pc_attribute(idx, name)

                attributes[attr_name] = PointCloud.Attribute(
                    loader=_load_attr,
                    transform_type=schema.transform_type,
                )

            return PointCloud(
                _xyz=xyz,
                reference_frame_id=reader.get_pc_reference_frame_id(pc_index),
                reference_frame_timestamp_us=reader.get_pc_reference_frame_timestamp_us(pc_index),
                coordinate_unit=reader.coordinate_unit,
                _attributes=attributes,
            )

        @override
        def get_pc_generic_data_names(self, pc_index: int) -> List[str]:
            return self._reader.get_pc_generic_data_names(pc_index)

        @override
        def has_pc_generic_data(self, pc_index: int, name: str) -> bool:
            return self._reader.has_pc_generic_data(pc_index, name)

        @override
        def get_pc_generic_data(self, pc_index: int, name: str) -> npt.NDArray:
            return self._reader.get_pc_generic_data(pc_index, name)

        @override
        async def get_pc_generic_data_async(self, pc_index: int, name: str) -> npt.NDArray:
            return await self._reader.get_pc_generic_data_async(pc_index, name)

        @override
        def get_pc_generic_meta_data(self, pc_index: int) -> JsonDocument:
            return self._reader.get_pc_generic_meta_data(pc_index)

    @property
    @override
    def point_clouds_ids(self) -> List[str]:
        return list(self._point_clouds_readers.keys())

    @override
    def get_point_clouds_source(self, source_id: str, *, return_index: int = 0) -> AsyncPointCloudsSourceProtocol:
        # 1. Check native point-clouds sources first
        if source_id in self._point_clouds_readers:
            return self.PointCloudsSource(self._point_clouds_readers[source_id])

        # 2. Check lidar sensors (adapted)
        if source_id in self._lidars_readers:
            return AsyncRayBundleSensorPointCloudsSourceAdapter(
                sensor=self.get_lidar_sensor(source_id),
                return_index=return_index,
            )

        # 3. Check radar sensors (adapted)
        if source_id in self._radars_readers:
            return AsyncRayBundleSensorPointCloudsSourceAdapter(
                sensor=self.get_radar_sensor(source_id),
                return_index=return_index,
            )

        raise KeyError(f"Point-clouds source '{source_id}' not found in native sources, lidars, or radars")

    @property
    @override
    def camera_labels_ids(self) -> List[str]:
        return list(self._camera_labels_readers.keys())

    @override
    def get_camera_labels(self, camera_label_id: str) -> AsyncCameraLabelsProtocol:
        if camera_label_id not in self._camera_labels_readers:
            raise KeyError(f"Camera labels '{camera_label_id}' not found")
        return self.CameraLabels(self._camera_labels_readers[camera_label_id])

    @override
    def query_camera_labels(
        self,
        camera_id: str,
        label_type: Optional[LabelType] = None,
        label_category: Optional[LabelCategory] = None,
    ) -> Sequence[AsyncCameraLabelsProtocol]:
        results: List[AsyncCameraLabelsProtocol] = []
        for reader in self._camera_labels_readers.values():
            if reader.label_descriptor.camera_id != camera_id:
                continue
            reader_lt = reader.label_descriptor.label_type
            if label_type is not None and reader_lt != label_type:
                continue
            if label_category is not None and reader_lt.category != label_category:
                continue
            results.append(self.CameraLabels(reader))
        return results

    @override
    def get_cuboid_track_observations(
        self, timestamp_interval_us: Optional[HalfClosedInterval] = None
    ) -> Generator[CuboidTrackObservation]:
        """Returns all available cuboid track observations in the sequence.

        Args:
            timestamp_interval_us: If provided, only observations whose ``timestamp_us``
                falls within this half-closed interval ``[start, stop)`` are returned.
                When ``None`` (default), all observations are returned.
        """

        # No cuboids reader available, return empty generator
        if self._cuboids_reader is None:
            return

        # Lazily cache all observations on first call to avoid re-reading from storage
        if self._cuboid_observations_cache is None:
            self._cuboid_observations_cache = list(self._cuboids_reader.get_observations())

        for obs in self._cuboid_observations_cache:
            if timestamp_interval_us is None or obs.timestamp_us in timestamp_interval_us:
                yield obs
