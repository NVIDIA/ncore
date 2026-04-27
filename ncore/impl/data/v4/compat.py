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

from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Union, cast

import numpy as np
import PIL.Image as PILImage

from typing_extensions import override
from upath import UPath

from ncore.impl.common.transformations import HalfClosedInterval, MotionCompensator, PoseGraphInterpolator
from ncore.impl.common.util import unpack_optional
from ncore.impl.data.compat import (
    CameraLabelsProtocol,
    CameraSensorProtocol,
    LidarSensorProtocol,
    PointCloudsSourceProtocol,
    RadarSensorProtocol,
    RayBundleSensorPointCloudsSourceAdapter,
    RayBundleSensorProtocol,
    SensorProtocol,
    SequenceLoaderProtocol,
)
from ncore.impl.data.types import (
    ConcreteCameraModelParametersUnion,
    ConcreteLidarModelParametersUnion,
    CuboidTrackObservation,
    FrameTimepoint,
    JsonLike,
    LabelSchema,
    LabelType,
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
    import numpy.typing as npt  # type: ignore[import-not-found]


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
        masks_component_group_name: Optional[str] = "default",
        cuboids_component_group_name: Optional[str] = "default",
    ):
        self._reader: SequenceComponentGroupsReader = reader

        # open all mandatory component readers
        assert poses_component_group_name in (
            poses_readers := self._reader.open_component_readers(PosesComponent.Reader)
        ), f"PosesComponent group '{poses_component_group_name}' not found"
        self._poses_reader: PosesComponent.Reader = poses_readers[poses_component_group_name]

        assert intrinsics_component_group_name in (
            intrinsics_readers := self._reader.open_component_readers(IntrinsicsComponent.Reader)
        ), f"IntrinsicsComponent group '{intrinsics_component_group_name}' not found"
        self._intrinsics_reader: IntrinsicsComponent.Reader = intrinsics_readers[intrinsics_component_group_name]

        # open optional component readers if available
        self._masks_reader: Optional[MasksComponent.Reader] = None
        if masks_component_group_name is not None:
            assert (masks_readers := self._reader.open_component_readers(MasksComponent.Reader)), (
                f"MasksComponent group '{masks_component_group_name}' not found"
            )
            self._masks_reader = masks_readers[masks_component_group_name]

        self._cuboids_reader: Optional[CuboidsComponent.Reader] = None
        if cuboids_component_group_name is not None:
            assert (cuboids_readers := self._reader.open_component_readers(CuboidsComponent.Reader)), (
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
            mask_reader: Optional[MasksComponent.Reader],
            model_parameters: ConcreteCameraModelParametersUnion,
            pose_graph: PoseGraphInterpolator,
        ):
            super().__init__(reader, pose_graph)

            self._mask_reader: Optional[MasksComponent.Reader] = mask_reader
            self._model_parameters: ConcreteCameraModelParametersUnion = model_parameters

        @property
        def camera_reader(self) -> CameraSensorComponent.Reader:
            return cast(CameraSensorComponent.Reader, self._reader)

        @property
        @override
        def model_parameters(self) -> ConcreteCameraModelParametersUnion:
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
            model_parameters: Optional[ConcreteLidarModelParametersUnion],
        ):
            super().__init__(reader, pose_graph)

            self._model_parameters: Optional[ConcreteLidarModelParametersUnion] = model_parameters

        @property
        def lidar_reader(self) -> LidarSensorComponent.Reader:
            return cast(LidarSensorComponent.Reader, self._reader)

        @property
        @override
        def model_parameters(self) -> Optional[ConcreteLidarModelParametersUnion]:
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

    class CameraLabelsSource(CameraLabelsProtocol):
        """Wraps a :class:`CameraLabelsComponent.Reader` to implement :class:`CameraLabelsProtocol`."""

        def __init__(self, reader: CameraLabelsComponent.Reader) -> None:
            self._reader = reader

        @property
        @override
        def camera_id(self) -> str:
            return self._reader.camera_id

        @property
        @override
        def label_type(self) -> LabelType:
            return self._reader.label_type

        @property
        @override
        def label_type_name(self) -> str:
            return self._reader.label_type_name

        @property
        @override
        def label_schema(self) -> LabelSchema:
            return self._reader.label_schema

        @property
        @override
        def labels_count(self) -> int:
            return self._reader.labels_count

        @property
        @override
        def timestamps_us(self) -> npt.NDArray[np.uint64]:
            return self._reader.timestamps_us

        @property
        @override
        def generic_meta_data(self) -> Dict[str, JsonLike]:
            return self._reader.generic_meta_data

        @override
        def get_label_handle(self, timestamp_us: int) -> Any:
            return self._reader.get_label_handle(timestamp_us)

        @override
        def get_label_data(self, timestamp_us: int) -> npt.NDArray[Any]:
            return self._reader.get_label_data(timestamp_us)

        @override
        def get_label_encoded_data(self, timestamp_us: int) -> Optional[bytes]:
            return self._reader.get_label_encoded_data(timestamp_us)

        @override
        def get_label_generic_meta_data(self, timestamp_us: int) -> Dict[str, JsonLike]:
            return self._reader.get_label_generic_meta_data(timestamp_us)

        @override
        def has_label_at(self, timestamp_us: int) -> bool:
            return timestamp_us in self._reader._timestamp_to_index

    class PointCloudsSource(PointCloudsSourceProtocol):
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
                _xyz=reader.get_pc_xyz(pc_index),
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
        def get_pc_generic_meta_data(self, pc_index: int) -> Dict[str, JsonLike]:
            return self._reader.get_pc_generic_meta_data(pc_index)

    @property
    @override
    def point_clouds_ids(self) -> List[str]:
        return list(self._point_clouds_readers.keys())

    @override
    def get_point_clouds_source(self, source_id: str, *, return_index: int = 0) -> PointCloudsSourceProtocol:
        # 1. Check native point-clouds sources first
        if source_id in self._point_clouds_readers:
            return self.PointCloudsSource(self._point_clouds_readers[source_id])

        # 2. Check lidar sensors (adapted)
        if source_id in self._lidars_readers:
            return RayBundleSensorPointCloudsSourceAdapter(
                sensor=self.get_lidar_sensor(source_id),
                return_index=return_index,
            )

        # 3. Check radar sensors (adapted)
        if source_id in self._radars_readers:
            return RayBundleSensorPointCloudsSourceAdapter(
                sensor=self.get_radar_sensor(source_id),
                return_index=return_index,
            )

        raise KeyError(f"Point-clouds source '{source_id}' not found in native sources, lidars, or radars")

    @property
    @override
    def camera_label_ids(self) -> List[str]:
        return list(self._camera_labels_readers.keys())

    @override
    def get_camera_labels_source(self, camera_label_id: str) -> CameraLabelsProtocol:
        if camera_label_id not in self._camera_labels_readers:
            raise KeyError(f"Camera labels source '{camera_label_id}' not found")
        return self.CameraLabelsSource(self._camera_labels_readers[camera_label_id])

    @override
    def get_camera_label_ids_for_camera(self, camera_id: str) -> List[str]:
        return [
            inst_name
            for inst_name, reader in self._camera_labels_readers.items()
            if reader.camera_id == camera_id
        ]

    @override
    def get_camera_label_ids_for_type(self, label_type: Union[LabelType, str]) -> List[str]:
        target_name = label_type.name if isinstance(label_type, LabelType) else label_type
        return [
            inst_name
            for inst_name, reader in self._camera_labels_readers.items()
            if reader.label_type_name == target_name
        ]

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
