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

"""Data access layer wrapping :class:`SequenceLoaderProtocol` for the interactive viewer."""

from __future__ import annotations

import dataclasses
import functools
import logging

from typing import List, Literal, Optional

import numpy as np
import pandas as pd

from ncore.impl.common.transformations import HalfClosedInterval, PoseGraphInterpolator
from ncore.impl.data.compat import CameraSensorProtocol, LidarSensorProtocol, SensorProtocol, SequenceLoaderProtocol
from ncore.impl.data.types import CuboidTrackObservation, FrameTimepoint, LabelSource
from tools.ncore_vis.tracks import CuboidTrack


logger = logging.getLogger(__name__)


class DataLoader:
    """Convenience wrapper around :class:`SequenceLoaderProtocol` for visualization.

    Provides higher-level helpers such as trajectory sampling, cross-sensor frame
    synchronization, and cuboid observation querying.
    """

    def __init__(
        self,
        loader: SequenceLoaderProtocol,
        rig_frame_id: Optional[str] = "rig",
        world_frame_id: str = "world",
    ) -> None:
        self._loader: SequenceLoaderProtocol = loader
        self._rig_frame_id: Optional[str] = rig_frame_id
        self._world_frame_id: str = world_frame_id
        self._cuboid_df: Optional[pd.DataFrame] = None
        self._cuboid_tracks: Optional[List[CuboidTrack]] = None

    # ------------------------------------------------------------------
    # Sensor access
    # ------------------------------------------------------------------

    @property
    def camera_ids(self) -> List[str]:
        """All camera sensor IDs in the sequence."""
        return self._loader.camera_ids

    @property
    def lidar_ids(self) -> List[str]:
        """All lidar sensor IDs in the sequence."""
        return self._loader.lidar_ids

    @property
    def pose_graph(self) -> PoseGraphInterpolator:
        """The sequence-wide pose graph."""
        return self._loader.pose_graph

    @property
    def rig_frame_id(self) -> Optional[str]:
        """Pose graph frame ID for the rig/vehicle body."""
        return self._rig_frame_id

    @property
    def world_frame_id(self) -> str:
        """Pose graph frame ID for the world/map reference."""
        return self._world_frame_id

    @functools.lru_cache(maxsize=None)
    def get_camera_sensor(self, camera_id: str) -> CameraSensorProtocol:
        """Return a camera sensor by ID (cached)."""
        return self._loader.get_camera_sensor(camera_id)

    @functools.lru_cache(maxsize=None)
    def get_lidar_sensor(self, lidar_id: str) -> LidarSensorProtocol:
        """Return a lidar sensor by ID (cached)."""
        return self._loader.get_lidar_sensor(lidar_id)

    # ------------------------------------------------------------------
    # Cross-sensor frame synchronization
    # ------------------------------------------------------------------

    def _get_sensor(self, sensor_id: str) -> SensorProtocol:
        """Return a sensor by ID, looking up cameras first then lidars."""
        if sensor_id in self.camera_ids:
            return self.get_camera_sensor(sensor_id)
        return self.get_lidar_sensor(sensor_id)

    def get_sensor_frame_interval_us(self, sensor_id: str, frame_index: int) -> HalfClosedInterval:
        """Return the ``[start, stop)`` timestamp interval in microseconds for a sensor frame.

        Works for both cameras and lidars.  The returned interval uses half-closed
        semantics: ``start`` is the frame's start timestamp and ``stop`` is the
        frame's end timestamp **+ 1**, so that ``timestamp_us in interval`` correctly
        includes the end timestamp.
        """
        sensor = self._get_sensor(sensor_id)
        return HalfClosedInterval.from_start_end(
            sensor.get_frame_timestamp_us(frame_index, FrameTimepoint.START),
            sensor.get_frame_timestamp_us(frame_index, FrameTimepoint.END),
        )

    # ------------------------------------------------------------------
    # Trajectory
    # ------------------------------------------------------------------

    def get_trajectory_poses(self) -> np.ndarray:
        """Sample rig-to-world poses at the sequence's dynamic pose timestamps.

        Returns:
            Array of shape ``[N, 4, 4]`` with rig-to-world transformation matrices.
        """
        if self._rig_frame_id is None:
            return np.empty((0, 4, 4), dtype=np.float64)

        interval = self._loader.sequence_timestamp_interval_us
        # Sample trajectory at ~10 Hz (every 100 ms = 100_000 us)
        sample_step_us = 100_000
        n_samples = max(1, (interval.stop - interval.start) // sample_step_us)
        timestamps = np.linspace(interval.start, interval.end, num=n_samples, dtype=np.uint64)

        try:
            poses = self.pose_graph.evaluate_poses(self._rig_frame_id, self._world_frame_id, timestamps)
        except KeyError:
            logger.warning(
                "Pose graph does not contain '%s' -> '%s' path; trajectory unavailable.",
                self._rig_frame_id,
                self._world_frame_id,
            )
            return np.empty((0, 4, 4), dtype=np.float64)

        return poses

    def get_rig_pose_at_frame(self, sensor_id: str, frame_index: int) -> Optional[np.ndarray]:
        """Return the 4x4 rig-to-world pose at the end timestamp of the given sensor frame.

        Args:
            sensor_id: The sensor whose frame timestamp to use.
            frame_index: 0-based frame index.

        Returns:
            4x4 ``T_rig_world`` matrix, or ``None`` if the pose graph cannot resolve the path.
        """
        if self._rig_frame_id is None:
            return None

        timestamp_us = self._get_sensor(sensor_id).get_frame_timestamp_us(frame_index, FrameTimepoint.END)
        return self.get_rig_pose_at_timestamp(timestamp_us)

    def get_rig_pose_at_timestamp(self, timestamp_us: int) -> Optional[np.ndarray]:
        """Return the 4x4 rig-to-world pose at the given timestamp.

        Args:
            timestamp_us: Timestamp in microseconds.

        Returns:
            4x4 ``T_rig_world`` matrix, or ``None`` if the pose graph cannot resolve the path.
        """
        if self._rig_frame_id is None:
            return None

        try:
            poses = self.pose_graph.evaluate_poses(
                self._rig_frame_id, self._world_frame_id, np.array([timestamp_us], dtype=np.uint64)
            )
            return poses[0]  # type: ignore[return-value]
        except KeyError:
            logger.warning(
                "Pose graph does not contain '%s' -> '%s' path; rig pose unavailable.",
                self._rig_frame_id,
                self._world_frame_id,
            )
            return None

    # ------------------------------------------------------------------
    # Cuboid observations
    # ------------------------------------------------------------------

    @property
    def cuboid_sources(self) -> List[str]:
        """Available label source names derived from the :class:`LabelSource` enum."""
        return [s.name for s in LabelSource]

    def _ensure_cuboid_df(self) -> pd.DataFrame:
        """Lazily load cuboid observations into a DataFrame for efficient querying."""
        if self._cuboid_df is None:
            observations = list(self._loader.get_cuboid_track_observations())
            if observations:
                self._cuboid_df = pd.DataFrame.from_records(
                    [obs.to_dict() for obs in observations],
                    columns=[field.name for field in dataclasses.fields(CuboidTrackObservation)],
                )
            else:
                self._cuboid_df = pd.DataFrame(
                    columns=[field.name for field in dataclasses.fields(CuboidTrackObservation)]
                )
        return self._cuboid_df

    def get_cuboid_observations_in_world(
        self,
        interval_us: HalfClosedInterval,
        target_time: Literal["end-of-interval", "observation-time"],
        source_filter: Optional[str] = None,
    ) -> List[CuboidTrackObservation]:
        """Return cuboid observations within a time window, transformed to world coordinates.

        Selects all observations whose ``timestamp_us`` falls within *interval_us*,
        then transforms each one into the world coordinate frame at the specified
        target time via the pose graph.  The returned observations have
        ``reference_frame_id == self.world_frame_id`` and their ``bbox3`` expressed
        in world coordinates, ready for direct rendering.

        This method is sensor-agnostic: observations can originate from any reference
        frame in the pose graph (lidar, camera, rig, etc.) and will be correctly
        transformed to world.

        Args:
            interval_us: Half-closed ``[start, stop)`` timestamp interval in
                microseconds.  Typically obtained from
                :meth:`get_sensor_frame_interval_us`.
            target_time: Whether to evaluate the pose graph at the end of the interval
                or at the observation's own timestamp.
            source_filter: If set, only return observations from this :class:`LabelSource` name.

        Returns:
            List of :class:`CuboidTrackObservation` in world coordinates.
        """
        cuboid_df = self._ensure_cuboid_df()
        if cuboid_df.empty:
            return []

        # Half-closed interval: start <= timestamp_us < stop
        mask = (cuboid_df["timestamp_us"] >= interval_us.start) & (cuboid_df["timestamp_us"] < interval_us.stop)
        if source_filter is not None:
            # to_dict() encodes LabelSource as its .name string (e.g. "AUTOLABEL"),
            # so we compare against the name string directly.
            mask = mask & (cuboid_df["source"] == source_filter)

        matched_rows = cuboid_df.loc[mask]

        pose_graph = self.pose_graph
        world_frame_id = self._world_frame_id
        result: List[CuboidTrackObservation] = []
        for _, row in matched_rows.iterrows():
            obs = CuboidTrackObservation.from_dict(row.to_dict())
            obs = obs.transform(
                target_frame_id=world_frame_id,
                target_frame_timestamp_us=interval_us.end if target_time == "end-of-interval" else obs.timestamp_us,
                pose_graph=pose_graph,
            )
            result.append(obs)
        return result

    def get_cuboid_tracks(self) -> List[CuboidTrack]:
        """Return all cuboid tracks for the sequence, lazily built and cached.

        Tracks are constructed once from the full set of cuboid observations
        (i.e. the entire sequence, not filtered by any time interval) and then
        cached for the lifetime of this :class:`DataLoader`.  Each
        :class:`~tools.ncore_vis.tracks.CuboidTrack` covers all labelled
        keyframes for one tracked object.

        Callers that need the interpolated cuboid pose at a specific timestamp
        (e.g. a camera frame's mid-of-frame time) should call
        :meth:`~tools.ncore_vis.tracks.CuboidTrack.interpolate_at` on each
        returned track.

        Returns:
            List of :class:`~tools.ncore_vis.tracks.CuboidTrack`, one per
            unique ``track_id`` in the sequence.  Returns an empty list when
            no cuboid observations are available.
        """
        if self._cuboid_tracks is None:
            cuboid_df = self._ensure_cuboid_df()
            if cuboid_df.empty:
                self._cuboid_tracks = []
            else:
                all_obs = [CuboidTrackObservation.from_dict(row.to_dict()) for _, row in cuboid_df.iterrows()]
                self._cuboid_tracks = CuboidTrack.from_observations(all_obs)
        return self._cuboid_tracks
