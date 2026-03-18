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

import unittest

import numpy as np

from ncore.impl.data.types import BBox3, CuboidTrackObservation, LabelSource
from tools.ncore_vis.tracks import CuboidTrack


def _make_obs(
    track_id: str,
    timestamp_us: int,
    x: float,
    y: float = 0.0,
    z: float = 0.0,
    yaw: float = 0.0,
) -> CuboidTrackObservation:
    return CuboidTrackObservation(
        track_id=track_id,
        class_id="Vehicle",
        timestamp_us=timestamp_us,
        reference_frame_id="world",
        reference_frame_timestamp_us=timestamp_us,
        bbox3=BBox3(
            centroid=(x, y, z),
            dim=(4.0, 2.0, 1.5),
            rot=(0.0, 0.0, yaw),
        ),
        source=LabelSource.GT_ANNOTATION,
    )


class TestCuboidTrackConstruction(unittest.TestCase):
    """Test CuboidTrack construction and validation"""

    def test_rejects_empty_observations(self):
        with self.assertRaises(ValueError):
            CuboidTrack(observations=[])

    def test_rejects_mixed_track_ids(self):
        obs_a = _make_obs("a", 0, 0.0)
        obs_b = _make_obs("b", 1_000_000, 1.0)
        with self.assertRaises(ValueError):
            CuboidTrack(observations=[obs_a, obs_b])

    def test_properties(self):
        obs0 = _make_obs("t1", 0, 0.0)
        obs1 = _make_obs("t1", 1_000_000, 10.0)
        track = CuboidTrack(observations=[obs0, obs1])
        self.assertEqual(track.track_id, "t1")
        self.assertEqual(track.class_id, "Vehicle")
        self.assertEqual(len(track.observations), 2)

    def test_observations_sorted_by_timestamp(self):
        obs1 = _make_obs("t1", 1_000_000, 10.0)
        obs0 = _make_obs("t1", 0, 0.0)
        track = CuboidTrack(observations=[obs1, obs0])  # reversed order
        self.assertEqual(track.observations[0].timestamp_us, 0)
        self.assertEqual(track.observations[1].timestamp_us, 1_000_000)


class TestCuboidTrackInterpolation(unittest.TestCase):
    """Test CuboidTrack.interpolate_at"""

    def _make_track(self) -> CuboidTrack:
        obs0 = _make_obs("t1", 0, 0.0)
        obs1 = _make_obs("t1", 1_000_000, 10.0)
        return CuboidTrack(observations=[obs0, obs1])

    def test_interpolate_exact_start(self):
        track = self._make_track()
        result = track.interpolate_at(0)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result.bbox3.centroid, (0.0, 0.0, 0.0), atol=1e-5)

    def test_interpolate_exact_end(self):
        track = self._make_track()
        result = track.interpolate_at(1_000_000)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result.bbox3.centroid, (10.0, 0.0, 0.0), atol=1e-5)

    def test_interpolate_midpoint_centroid(self):
        track = self._make_track()
        result = track.interpolate_at(500_000)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result.bbox3.centroid, (5.0, 0.0, 0.0), atol=1e-5)

    def test_interpolate_before_range_returns_none(self):
        track = self._make_track()
        result = track.interpolate_at(-500_000)
        self.assertIsNone(result)

    def test_interpolate_after_range_returns_none(self):
        track = self._make_track()
        result = track.interpolate_at(2_000_000)
        self.assertIsNone(result)

    def test_interpolate_before_range_clamp_within_limit(self):
        track = self._make_track()  # range [0, 1_000_000]
        result = track.interpolate_at(-100_000, max_clamp_us=200_000)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result.bbox3.centroid, (0.0, 0.0, 0.0), atol=1e-5)
        self.assertEqual(result.timestamp_us, -100_000)

    def test_interpolate_before_range_clamp_exceeds_limit(self):
        track = self._make_track()
        result = track.interpolate_at(-500_000, max_clamp_us=200_000)
        self.assertIsNone(result)

    def test_interpolate_after_range_clamp_within_limit(self):
        track = self._make_track()
        result = track.interpolate_at(1_100_000, max_clamp_us=200_000)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result.bbox3.centroid, (10.0, 0.0, 0.0), atol=1e-5)
        self.assertEqual(result.timestamp_us, 1_100_000)

    def test_interpolate_after_range_clamp_exceeds_limit(self):
        track = self._make_track()
        result = track.interpolate_at(2_000_000, max_clamp_us=200_000)
        self.assertIsNone(result)

    def test_single_observation_exact_match(self):
        obs = _make_obs("t1", 500_000, 3.0, 4.0, 5.0)
        track = CuboidTrack(observations=[obs])
        result = track.interpolate_at(500_000)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result.bbox3.centroid, (3.0, 4.0, 5.0), atol=1e-5)

    def test_single_observation_before_returns_none(self):
        obs = _make_obs("t1", 500_000, 3.0, 4.0, 5.0)
        track = CuboidTrack(observations=[obs])
        self.assertIsNone(track.interpolate_at(0))

    def test_single_observation_after_returns_none(self):
        obs = _make_obs("t1", 500_000, 3.0, 4.0, 5.0)
        track = CuboidTrack(observations=[obs])
        self.assertIsNone(track.interpolate_at(1_000_000))

    def test_single_observation_before_clamp_within_limit(self):
        obs = _make_obs("t1", 500_000, 3.0, 4.0, 5.0)
        track = CuboidTrack(observations=[obs])
        result = track.interpolate_at(400_000, max_clamp_us=200_000)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result.bbox3.centroid, (3.0, 4.0, 5.0), atol=1e-5)

    def test_single_observation_before_clamp_exceeds_limit(self):
        obs = _make_obs("t1", 500_000, 3.0, 4.0, 5.0)
        track = CuboidTrack(observations=[obs])
        self.assertIsNone(track.interpolate_at(0, max_clamp_us=200_000))

    def test_single_observation_after_clamp_within_limit(self):
        obs = _make_obs("t1", 500_000, 3.0, 4.0, 5.0)
        track = CuboidTrack(observations=[obs])
        result = track.interpolate_at(600_000, max_clamp_us=200_000)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result.bbox3.centroid, (3.0, 4.0, 5.0), atol=1e-5)

    def test_single_observation_after_clamp_exceeds_limit(self):
        obs = _make_obs("t1", 500_000, 3.0, 4.0, 5.0)
        track = CuboidTrack(observations=[obs])
        self.assertIsNone(track.interpolate_at(1_000_000, max_clamp_us=200_000))

    def test_dimensions_unchanged(self):
        track = self._make_track()
        result = track.interpolate_at(500_000)
        self.assertIsNotNone(result)
        self.assertEqual(result.bbox3.dim, (4.0, 2.0, 1.5))

    def test_result_timestamp_matches_query(self):
        track = self._make_track()
        result = track.interpolate_at(750_000)
        self.assertIsNotNone(result)
        self.assertEqual(result.timestamp_us, 750_000)
        self.assertEqual(result.reference_frame_timestamp_us, 750_000)

    def test_interpolate_rotation_midpoint(self):
        import math

        obs0 = _make_obs("t1", 0, 0.0, yaw=0.0)
        obs1 = _make_obs("t1", 1_000_000, 0.0, yaw=math.pi / 2)
        track = CuboidTrack(observations=[obs0, obs1])
        result = track.interpolate_at(500_000)
        self.assertIsNotNone(result)
        # Mid-point yaw should be ~pi/4
        np.testing.assert_allclose(result.bbox3.rot[2], math.pi / 4, atol=1e-4)

    def test_interpolate_quarter_point(self):
        track = self._make_track()
        result = track.interpolate_at(250_000)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result.bbox3.centroid, (2.5, 0.0, 0.0), atol=1e-5)

    def test_result_reference_frame_id_preserved(self):
        track = self._make_track()
        result = track.interpolate_at(500_000)
        self.assertIsNotNone(result)
        self.assertEqual(result.reference_frame_id, "world")

    def test_result_class_id_preserved(self):
        track = self._make_track()
        result = track.interpolate_at(500_000)
        self.assertIsNotNone(result)
        self.assertEqual(result.class_id, "Vehicle")


class TestCuboidTrackFromObservations(unittest.TestCase):
    """Test CuboidTrack.from_observations factory"""

    def test_groups_by_track_id(self):
        obs_t1_a = _make_obs("t1", 0, 0.0)
        obs_t1_b = _make_obs("t1", 1_000_000, 1.0)
        obs_t2_a = _make_obs("t2", 0, 5.0)
        tracks = CuboidTrack.from_observations([obs_t1_a, obs_t1_b, obs_t2_a])
        self.assertEqual(len(tracks), 2)
        track_ids = {t.track_id for t in tracks}
        self.assertIn("t1", track_ids)
        self.assertIn("t2", track_ids)

    def test_single_track(self):
        obs = _make_obs("t1", 0, 0.0)
        tracks = CuboidTrack.from_observations([obs])
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track_id, "t1")

    def test_empty_list(self):
        tracks = CuboidTrack.from_observations([])
        self.assertEqual(tracks, [])

    def test_observation_counts_per_track(self):
        obs = [_make_obs("t1", i * 100_000, float(i)) for i in range(5)]
        obs += [_make_obs("t2", 0, 99.0)]
        tracks = CuboidTrack.from_observations(obs)
        by_id = {t.track_id: t for t in tracks}
        self.assertEqual(len(by_id["t1"].observations), 5)
        self.assertEqual(len(by_id["t2"].observations), 1)
