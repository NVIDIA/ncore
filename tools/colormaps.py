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

"""Colormap utilities backed by OpenCV lookup tables.

Provides :func:`jet` and :func:`turbo` functions that map float arrays in
``[0, 1]`` to ``[N, 3]`` uint8 RGB arrays, suitable for direct use with
OpenCV drawing primitives (``cv2.circle``, ``cv2.line``, etc.).

Usage::

    from tools.colormaps import jet, turbo

    colors_rgb = jet(normalized_values)   # [N, 3] uint8
    colors_rgb = turbo(normalized_values) # [N, 3] uint8
"""

import cv2
import numpy as np


# Pre-build 256-entry RGB lookup tables from OpenCV colormaps at import time.
# OpenCV returns BGR; we convert to RGB and store contiguous copies.
_GRAY_RAMP = np.arange(256, dtype=np.uint8).reshape(1, -1)
_JET_LUT: np.ndarray = cv2.applyColorMap(_GRAY_RAMP, cv2.COLORMAP_JET)[0][:, ::-1].copy()
_TURBO_LUT: np.ndarray = cv2.applyColorMap(_GRAY_RAMP, cv2.COLORMAP_TURBO)[0][:, ::-1].copy()


def jet(t: np.ndarray) -> np.ndarray:
    """Apply the jet colormap to normalized values.

    Args:
        t: Array of float values in [0, 1].

    Returns:
        uint8 RGB array of shape ``[N, 3]``.
    """
    indices = np.clip((np.asarray(t, dtype=np.float64) * 255).astype(np.intp), 0, 255)
    return _JET_LUT[indices]


def turbo(t: np.ndarray) -> np.ndarray:
    """Apply the turbo colormap to normalized values.

    Args:
        t: Array of float values in [0, 1].

    Returns:
        uint8 RGB array of shape ``[N, 3]``.
    """
    indices = np.clip((np.asarray(t, dtype=np.float64) * 255).astype(np.intp), 0, 255)
    return _TURBO_LUT[indices]
