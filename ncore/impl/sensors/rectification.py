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

"""Rectification utilities mapping a source camera image into a target camera domain"""

from __future__ import annotations

from typing import Literal, Union

import numpy as np
import torch

from ncore.impl.sensors.camera import CameraModel
from ncore.impl.sensors.common import to_torch


class Rectificator:
    """Maps imagery from a source camera into a target camera's image domain

    Rectification is a generic "from-camera -> to-camera" intrinsic remap within a shared
    camera frame. The most common use is mapping a distorted source camera to a
    distortion-free (ideal pinhole) target, but the target may itself be distorted.

    On construction a backward sample map (target -> source direction) is built once:
    for every target pixel, the corresponding ray is unprojected with the target model
    and re-projected with the source model, giving the source image coordinate to sample
    from. Resampling a source image into the target domain is then a single gather /
    interpolation (:meth:`apply`). The raw map (:attr:`sample_map`, :attr:`valid_mask`)
    is exposed so callers can run their own (e.g. GPU or OpenCV) resampling.
    """

    def __init__(self, source: CameraModel, target: CameraModel):
        """Build the rectification map from a source/target camera model pair

        Args:
            source: the source camera model the imagery was captured with.
            target: the target camera model to rectify the imagery into.
        """
        self.source = source
        self.target = target

        device = target.device
        dtype = target.dtype

        target_width = int(target.resolution[0].item())
        target_height = int(target.resolution[1].item())
        self._target_width = target_width
        self._target_height = target_height

        # Build the continuous target image-point grid (pixel centers)
        ys, xs = torch.meshgrid(
            torch.arange(target_height, device=device, dtype=dtype),
            torch.arange(target_width, device=device, dtype=dtype),
            indexing="ij",
        )
        target_image_points = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1) + 0.5

        # Target image points -> target rays -> source image points
        target_rays = target.image_points_to_camera_rays(target_image_points)
        source = self.source
        source_proj = source.camera_rays_to_image_points(target_rays)

        sample_map = source_proj.image_points.reshape(target_height, target_width, 2)
        valid_mask = source_proj.valid_flag.reshape(target_height, target_width)

        self._sample_map = sample_map.contiguous()
        self._valid_mask = valid_mask.contiguous()

    @property
    def sample_map(self) -> torch.Tensor:
        """Source image coordinates to sample for each target pixel, shape ``[H, W, 2]``

        ``sample_map[v, u]`` is the continuous source image point ``(x, y)`` that target
        pixel ``(u, v)`` maps to. Invalid entries (see :attr:`valid_mask`) hold whatever
        the source projection produced and should not be relied upon.
        """
        return self._sample_map

    @property
    def valid_mask(self) -> torch.Tensor:
        """Boolean mask of valid target pixels, shape ``[H, W]``

        A target pixel is valid when its ray projects to a point inside the source image.
        """
        return self._valid_mask

    def apply(
        self,
        image: Union[torch.Tensor, np.ndarray],
        mode: Literal["bilinear", "nearest", "bicubic"] = "bilinear",
        padding_mode: Literal["zeros", "border", "reflection"] = "zeros",
    ) -> torch.Tensor:
        """Resample a source image into the target camera domain

        Args:
            image: source image, either ``[H_src, W_src]``, ``[H_src, W_src, C]`` or
                   ``[N, H_src, W_src, C]`` (channels-last, as commonly stored). Integer
                   images are converted to float for interpolation.
            mode: interpolation mode passed to ``grid_sample``, one of ``"bilinear"``,
                  ``"nearest"`` or ``"bicubic"``.
            padding_mode: out-of-bounds sampling mode passed to ``grid_sample``, one of
                  ``"zeros"``, ``"border"`` or ``"reflection"``.

        Returns:
            the rectified image in the target domain, channels-last, with the same number
            of leading/trailing dimensions as the (broadcast) input. Invalid target
            pixels are set to zero.
        """
        image_t = to_torch(image, device=self.target.device)

        # Normalize to [N, H, W, C]
        squeeze_batch = False
        squeeze_channel = False
        if image_t.ndim == 2:
            image_t = image_t[None, :, :, None]
            squeeze_batch = True
            squeeze_channel = True
        elif image_t.ndim == 3:
            image_t = image_t[None]
            squeeze_batch = True
        elif image_t.ndim != 4:
            raise ValueError(f"Unsupported image shape {tuple(image_t.shape)}")

        src_height, src_width = image_t.shape[1], image_t.shape[2]

        was_floating = image_t.is_floating_point()
        image_f = image_t.to(self.target.dtype)

        # [N, C, H, W] for grid_sample
        image_nchw = image_f.permute(0, 3, 1, 2)
        batch = image_nchw.shape[0]

        # Normalize sample coordinates to [-1, 1] (align_corners=False, pixel centers)
        grid = self._sample_map.clone()
        grid_x = 2.0 * grid[..., 0] / src_width - 1.0
        grid_y = 2.0 * grid[..., 1] / src_height - 1.0
        norm_grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        norm_grid = norm_grid[None].expand(batch, -1, -1, -1)

        sampled = torch.nn.functional.grid_sample(
            image_nchw,
            norm_grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=False,
        )  # [N, C, H_tgt, W_tgt]

        # Back to channels-last and zero out invalid target pixels
        result = sampled.permute(0, 2, 3, 1)
        result = result * self._valid_mask[None, :, :, None].to(result.dtype)

        if not was_floating:
            result = result.round().to(image_t.dtype)

        if squeeze_channel:
            result = result[..., 0]
        if squeeze_batch:
            result = result[0]
        return result

    def source_points_to_target(self, points: Union[torch.Tensor, np.ndarray]) -> CameraModel.ImagePointsReturn:
        """Map continuous source image points into the target camera domain

        Args:
            points: source image points, shape ``[n, 2]``.

        Returns:
            target image points and a valid flag (see
            :class:`~ncore.impl.sensors.camera.CameraModel.ImagePointsReturn`).
        """
        rays = self.source.image_points_to_camera_rays(points)
        return self.target.camera_rays_to_image_points(rays)

    def target_points_to_source(self, points: Union[torch.Tensor, np.ndarray]) -> CameraModel.ImagePointsReturn:
        """Map continuous target image points back into the source camera domain

        Args:
            points: target image points, shape ``[n, 2]``.

        Returns:
            source image points and a valid flag (see
            :class:`~ncore.impl.sensors.camera.CameraModel.ImagePointsReturn`).
        """
        rays = self.target.image_points_to_camera_rays(points)
        return self.source.camera_rays_to_image_points(rays)
