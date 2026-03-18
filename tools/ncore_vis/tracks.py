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

"""Cuboid track type with temporal interpolation.

A :class:`CuboidTrack` groups all :class:`~ncore.impl.data.types.CuboidTrackObservation`
instances that share the same ``track_id`` and exposes
:meth:`CuboidTrack.interpolate_at` to obtain an interpolated observation at any
target timestamp.  Translation is interpolated linearly and rotation via SLERP,
using :class:`~ncore.impl.common.transformations.PoseInterpolator`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional

import numpy as np

from ncore.impl.common.transformations import PoseInterpolator, bbox_pose, pose_bbox
from ncore.impl.data.types import BBox3, CuboidTrackObservation


@dataclass
class CuboidTrack:
    """Temporal sequence of cuboid observations for a single tracked object.

    All observations must share the same ``track_id``.  The sequence need not be
    dense; observations represent the labelled keyframes provided by the label source.

    Use :meth:`interpolate_at` to obtain a smoothly blended observation at any
    target timestamp within the track's time range.  Queries outside the range
    return ``None``.

    Use the :meth:`from_observations` factory to build a list of tracks from a
    flat, heterogeneous list of observations (e.g. the full sequence cache).
    """

    observations: List[CuboidTrackObservation]
    """Observations sorted by ``timestamp_us`` ascending (enforced in ``__post_init__``)."""

    def __post_init__(self) -> None:
        if not self.observations:
            raise ValueError("CuboidTrack requires at least one observation")
        track_ids = {obs.track_id for obs in self.observations}
        if len(track_ids) != 1:
            raise ValueError(f"CuboidTrack observations must all share the same track_id, got: {sorted(track_ids)}")
        # Ensure chronological order (may arrive unsorted from the loader)
        self.observations = sorted(self.observations, key=lambda o: o.timestamp_us)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def track_id(self) -> str:
        """Unique track identifier shared by all observations."""
        return self.observations[0].track_id

    @property
    def class_id(self) -> str:
        """Object class label (from the first observation; assumed stable across the track)."""
        return self.observations[0].class_id

    @property
    def source(self):
        """Label source (from the first observation)."""
        return self.observations[0].source

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def interpolate_at(self, timestamp_us: int) -> Optional[CuboidTrackObservation]:
        """Return an interpolated observation at *timestamp_us*, or ``None`` if out of range.

        The interpolated bbox pose (centroid + orientation) is computed by
        finding the two observations that bracket *timestamp_us* and blending
        them via :class:`~ncore.impl.common.transformations.PoseInterpolator`
        (linear translation + SLERP rotation).  ``bbox3.dim`` (dimensions) are
        taken from the earlier bracketing observation and are **not** interpolated.

        Out-of-range queries return ``None``:

        * Before the first observation's timestamp → ``None``.
        * After the last observation's timestamp → ``None``.

        For a single-observation track, only a query exactly at that observation's
        timestamp returns a result; all other times return ``None``.

        Args:
            timestamp_us: Target timestamp in microseconds.

        Returns:
            A :class:`~ncore.impl.data.types.CuboidTrackObservation` whose
            ``timestamp_us`` equals *timestamp_us* and whose ``bbox3`` is
            interpolated to that time, or ``None`` if *timestamp_us* is outside
            the track's time range.
        """

        obs = self.observations

        # --- Out-of-range: return None ---
        if timestamp_us < obs[0].timestamp_us or timestamp_us > obs[-1].timestamp_us:
            return None

        # --- Exact endpoint match, also handles single-observation tracks if matching ---
        if timestamp_us == obs[0].timestamp_us:
            return replace(obs[0], timestamp_us=timestamp_us, reference_frame_timestamp_us=timestamp_us)
        if timestamp_us == obs[-1].timestamp_us:
            return replace(obs[-1], timestamp_us=timestamp_us, reference_frame_timestamp_us=timestamp_us)

        # --- Find bracketing pair via binary search ---

        # searchsorted(..., side="right") gives the first index strictly greater than timestamp_us
        idx_after = int(np.searchsorted([o.timestamp_us for o in obs], timestamp_us, side="right"))
        idx_before = idx_after - 1
        obs_before = obs[idx_before]
        obs_after = obs[idx_after]

        alpha = (float(timestamp_us) - obs_before.timestamp_us) / (
            obs_after.timestamp_us - obs_before.timestamp_us
        )  # in (0, 1)

        # Build SE3 matrices from both bboxes and let PoseInterpolator blend them.
        # We parametrise the two keyframes at t=0.0 and t=1.0, query at alpha.
        poses = np.stack(
            [
                bbox_pose(obs_before.bbox3.to_array()),
                bbox_pose(obs_after.bbox3.to_array()),
            ],
            axis=0,
        )  # (2, 4, 4)
        blended_pose = PoseInterpolator(poses, [0.0, 1.0]).interpolate_to_timestamps(alpha)[0]  # (4, 4)

        # Reconstruct BBox3: preserve dimensions from the earlier observation
        blended_bbox = BBox3.from_array(pose_bbox(blended_pose, np.array(obs_before.bbox3.dim)))

        return replace(
            obs_before,
            timestamp_us=timestamp_us,
            reference_frame_timestamp_us=timestamp_us,
            bbox3=blended_bbox,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def from_observations(observations: List[CuboidTrackObservation]) -> List["CuboidTrack"]:
        """Group a flat list of observations into per-track :class:`CuboidTrack` objects.

        Observations are grouped by ``track_id``.  The order of tracks in the
        returned list is deterministic (insertion order of first occurrence).

        Args:
            observations: Flat list of :class:`~ncore.impl.data.types.CuboidTrackObservation`
                instances, potentially spanning multiple tracks.

        Returns:
            One :class:`CuboidTrack` per unique ``track_id`` found in *observations*.
            Returns an empty list when *observations* is empty.
        """
        by_track: Dict[str, List[CuboidTrackObservation]] = {}
        for obs in observations:
            by_track.setdefault(obs.track_id, []).append(obs)
        return [CuboidTrack(observations=group) for group in by_track.values()]
