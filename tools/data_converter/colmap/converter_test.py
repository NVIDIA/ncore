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

"""Data-free unit tests for the COLMAP converter's synthetic timestamps.

These run in CI (no external dataset needed). COLMAP scenes carry no capture
times, so the converter synthesizes them at one second per frame, offset by
``--start-time-sec``. That offset also determines the declared sequence
interval, so a camera whose timestamps ignore it produces frames outside the
interval and the writer rejects them.
"""

from __future__ import annotations

import unittest

import numpy as np
import pycolmap

from upath import UPath

from tools.data_converter.colmap.converter import ColmapCamera


def _camera(n_images: int, start_time_sec: float) -> ColmapCamera:
    """Builds a ColmapCamera with n_images frames.

    The camera model and image path are placeholders: the timestamps under test
    depend only on the image count and the start time.
    """
    return ColmapCamera(
        camera_id="camera_0",
        colmap_camera=pycolmap.Camera("PINHOLE", 640, 480, [500.0, 500.0, 320.0, 240.0]),
        image_path=UPath("/nonexistent"),
        image_names=[f"{i:04d}.jpg" for i in range(n_images)],
        start_time_sec=start_time_sec,
    )


class TestColmapTimestamps(unittest.TestCase):
    def test_timestamps_are_one_second_apart(self) -> None:
        timestamps_us = _camera(n_images=5, start_time_sec=0.0).timestamps_us
        np.testing.assert_array_equal(timestamps_us, np.arange(5, dtype=np.uint64) * 1_000_000)

    def test_start_time_sec_offsets_frame_timestamps(self) -> None:
        """The offset must reach the camera frames, not only the sequence interval."""
        start_time_sec = 10.0
        timestamps_us = _camera(n_images=4, start_time_sec=start_time_sec).timestamps_us

        expected = int(1e6 * start_time_sec) + np.arange(4, dtype=np.uint64) * 1_000_000
        np.testing.assert_array_equal(timestamps_us, expected)

    def test_frames_fall_inside_the_declared_sequence_interval(self) -> None:
        """Regression: frames used to start at 0 regardless of --start-time-sec.

        The converter declares the sequence interval as
        ``[start_time_sec, start_time_sec + n_images)`` seconds. Frames outside
        it are rejected by the writer rather than clamped.
        """
        start_time_sec, n_images = 10.0, 4
        timestamps_us = _camera(n_images=n_images, start_time_sec=start_time_sec).timestamps_us

        interval_start_us = int(1e6 * start_time_sec)
        interval_stop_us = int(1e6 * (start_time_sec + n_images))

        self.assertGreaterEqual(int(timestamps_us[0]), interval_start_us)
        self.assertLess(int(timestamps_us[-1]), interval_stop_us)

    def test_empty_camera_yields_no_timestamps(self) -> None:
        self.assertEqual(len(_camera(n_images=0, start_time_sec=3.0).timestamps_us), 0)


if __name__ == "__main__":
    unittest.main()
