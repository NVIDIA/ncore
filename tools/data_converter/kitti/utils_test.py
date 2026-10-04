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

"""Data-free unit tests for the KITTI converter utilities (run in CI)."""

from __future__ import annotations

import unittest

import numpy as np

from tools.data_converter.kitti.utils import compute_velodyne_timestamps_us


# Realistic KITTI raw spin bounds: absolute epoch microseconds (2011-09-26) and a ~103 ms spin
START_US = 1317020000000000
END_US = START_US + 103_000


class TestComputeVelodyneTimestamps(unittest.TestCase):
    def _points(self, xy: np.ndarray) -> np.ndarray:
        points = np.zeros((len(xy), 4), dtype=np.float32)
        points[:, :2] = xy
        return points

    def test_azimuth_to_spin_fraction(self):
        """Rear / left / front / right map to 0 / 1/4 / 1/2 / 3/4 of the spin, exactly at microsecond scale."""
        # (x, y) = rear (y = -0.0 so atan2 is -pi, i.e. the end of the spin), left, front, right
        points = self._points(np.array([[-1.0, -0.0], [0.0, 1.0], [1.0, 0.0], [0.0, -1.0]], dtype=np.float32))

        # Spin bounds are python ints (as returned by load_timestamps)
        timestamps_us = compute_velodyne_timestamps_us(points, START_US, END_US)

        self.assertEqual(timestamps_us.dtype, np.uint64)
        np.testing.assert_array_equal(
            timestamps_us.astype(np.int64) - START_US,
            np.array([103_000, 25_750, 51_500, 77_250], dtype=np.int64),
        )

    def test_timestamps_within_spin_bounds(self):
        """All per-point timestamps lie within the spin interval for absolute epoch timestamps."""
        rng = np.random.default_rng(0)
        points = rng.standard_normal((10_000, 4)).astype(np.float32)

        timestamps_us = compute_velodyne_timestamps_us(points, START_US, END_US)

        self.assertGreaterEqual(int(timestamps_us.min()), START_US)
        self.assertLessEqual(int(timestamps_us.max()), END_US)
        # The random azimuths cover (almost) the full spin
        self.assertLess(int(timestamps_us.min()) - START_US, 100)
        self.assertLess(END_US - int(timestamps_us.max()), 100)


if __name__ == "__main__":
    unittest.main()
