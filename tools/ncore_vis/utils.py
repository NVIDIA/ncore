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

"""Shared utility functions for the NCore interactive viewer."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from scipy.spatial.transform import Rotation


def se3_to_position_wxyz(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract translation and wxyz quaternion from a 4x4 SE3 matrix.

    Args:
        T: Homogeneous 4x4 rigid transformation matrix.

    Returns:
        Tuple of (position [3,], wxyz quaternion [4,]).
    """
    position = T[:3, 3].astype(np.float64)
    rotation = Rotation.from_matrix(T[:3, :3].astype(np.float64))
    xyzw = rotation.as_quat()  # scipy convention: [x, y, z, w]
    wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
    return position, wxyz


def se3_from_centroid_euler(centroid: Tuple[float, float, float], rot: Tuple[float, float, float]) -> np.ndarray:
    """Build a 4x4 SE3 matrix from centroid position and XYZ Euler angles.

    Args:
        centroid: Translation (x, y, z) in meters.
        rot: Rotation as XYZ Euler angles in radians.

    Returns:
        Homogeneous 4x4 transformation matrix.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_euler("XYZ", rot).as_matrix()
    T[:3, 3] = centroid
    return T
