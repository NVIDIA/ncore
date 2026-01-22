# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Tuple, Union, cast

import numpy as np
import PIL.Image as PILImage

from typing_extensions import override
from upath import UPath

from ncore.impl.common.transformations import HalfClosedInterval, MotionCompensator, PoseGraphInterpolator
from ncore.impl.common.util import unpack_optional
from ncore.impl.data.compat import (
    CameraSensorProtocol,
    LidarSensorProtocol,
    RadarSensorProtocol,
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
)
from ncore.impl.data.v3.shards import CameraSensor, LidarSensor, PointCloudSensor, RadarSensor, ShardDataLoader


if TYPE_CHECKING:
    import numpy.typing as npt  # type: ignore[import-not-found]


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
            sensor: Union[CameraSensor, LidarSensor, RadarSensor],
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

    class CameraSensor(Sensor, CameraSensorProtocol):
        """Camera sensor implementation for V3 data.

        Args:
            sensor: V3 camera sensor instance
            pose_graph: Pose graph interpolator for coordinate transformations
        """

        def __init__(
            self,
            sensor: CameraSensor,
            pose_graph: PoseGraphInterpolator,
        ):
            super().__init__(sensor, pose_graph)

        @property
        def camera_sensor(self) -> CameraSensor:
            return cast(CameraSensor, self._sensor)

        @property
        @override
        def model_parameters(self) -> ConcreteCameraModelParametersUnion:
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
            sensor: Union[LidarSensor, RadarSensor],
            pose_graph: PoseGraphInterpolator,
        ):
            super().__init__(sensor, pose_graph)

            self._motion_compensator: MotionCompensator = MotionCompensator(pose_graph)

        @property
        def point_cloud_sensor(self) -> PointCloudSensor:
            return cast(PointCloudSensor, self._sensor)

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
            frame_start_timestamp_us = int(frame_timestamps_us[FrameTimepoint.START.value])
            frame_end_timestamp_us = int(frame_timestamps_us[FrameTimepoint.END.value])

            if frame_start_timestamp_us != frame_end_timestamp_us:
                # regular case for non-instant frames
                xyz_m_end = self._motion_compensator.motion_decompensate_points(
                    sensor_id=self.sensor_id,
                    xyz_sensorend=xyz_e,
                    timestamp_us=self.get_frame_ray_bundle_timestamp_us(frame_index),
                    frame_start_timestamp_us=frame_start_timestamp_us,
                    frame_end_timestamp_us=frame_end_timestamp_us,
                )
            else:
                # instant frames don't require motion-decompensation as point coordinates
                # are relative to a single frame pose already
                xyz_m_end = xyz_e

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
            sensor: LidarSensor,
            pose_graph: PoseGraphInterpolator,
        ):
            super().__init__(sensor, pose_graph)

        @property
        def lidar_sensor(self) -> LidarSensor:
            return cast(LidarSensor, self._sensor)

        @property
        @override
        def model_parameters(self) -> Optional[ConcreteLidarModelParametersUnion]:
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
            sensor: RadarSensor,
            pose_graph: PoseGraphInterpolator,
        ):
            super().__init__(sensor, pose_graph)

        @property
        def radar_sensor(self) -> RadarSensor:
            return cast(RadarSensor, self._sensor)

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
