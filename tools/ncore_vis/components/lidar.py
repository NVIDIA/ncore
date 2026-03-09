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

"""Lidar point cloud visualization component with fusing and motion compensation."""

from __future__ import annotations

import concurrent.futures
import logging

from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import viser

from ncore.impl.common.transformations import HalfClosedInterval, transform_point_cloud
from ncore.impl.data.types import FrameTimepoint
from tools.ncore_vis.components.base import VisualizationComponent, register_component


logger = logging.getLogger(__name__)

_DEFAULT_POINT_COLOR: np.ndarray = np.array([255, 0, 0], dtype=np.uint8)

# Pre-fetch colormaps once at module level.
_JET_CMAP: matplotlib.colors.Colormap = matplotlib.colormaps["jet"]
_TURBO_CMAP: matplotlib.colors.Colormap = matplotlib.colormaps["turbo"]

# Supported color styles.
_COLOR_STYLES: List[str] = [
    "Intensity",
    "Intensity \u03b3=1/2",
    "Intensity \u03b3=1/4",
    "Range (jet)",
    "Height (turbo)",
    "Timestamp",
    "Model Row",
    "Model Column",
]

# Color styles that require per-frame data (cannot be computed from positions alone).
_PER_FRAME_COLOR_STYLES: frozenset[str] = frozenset(
    [
        "Intensity",
        "Intensity \u03b3=1/2",
        "Intensity \u03b3=1/4",
        "Timestamp",
        "Model Row",
        "Model Column",
    ]
)


@register_component
class LidarComponent(VisualizationComponent):
    """Lidar point cloud visualization with intensity coloring, fusing, and motion compensation.

    For each lidar sensor, provides controls for frame selection, point cloud coloring
    style, point size, multi-frame fusing, and a toggle between motion-compensated and
    non-motion-compensated point clouds.
    """

    def create_gui(self, tab_group: viser.GuiTabGroupHandle) -> None:  # noqa: C901
        self._enabled: bool = True
        self._point_clouds: Dict[str, viser.PointCloudHandle] = {}

        self._frame_sliders: Dict[str, viser.GuiInputHandle[int]] = {}
        self._color_style: Dict[str, str] = {}
        self._point_size: Dict[str, float] = {}
        self._show_pc: Dict[str, bool] = {}
        self._is_fused: Dict[str, bool] = {}
        self._fused_frame_step: Dict[str, int] = {}
        self._fused_range: Dict[str, Tuple[int, int]] = {}
        self._motion_comp: Dict[str, bool] = {}
        self._range_cycle: Dict[str, float] = {}
        self._height_range: Dict[str, Tuple[float, float]] = {}
        # Per-sensor generic_data fields that can drive point color.
        self._metadata_color_fields: Dict[str, List[str]] = {}

        with tab_group.add_tab("Lidars"):
            enabled_checkbox = self.client.gui.add_checkbox(
                "Enabled", initial_value=True, hint="Enable lidar point cloud visualization"
            )

            @enabled_checkbox.on_update
            def _(_: viser.GuiEvent) -> None:
                self._enabled = enabled_checkbox.value
                for lid in self._point_clouds:
                    self._point_clouds[lid].visible = enabled_checkbox.value and self._show_pc.get(lid, True)

            for lidar_id in self.data_loader.lidar_ids:
                sensor = self.data_loader.get_lidar_sensor(lidar_id)
                frame_count = sensor.frames_count
                max_frame = max(0, frame_count - 1)

                metadata_fields = self._scan_metadata_color_fields(lidar_id)
                self._metadata_color_fields[lidar_id] = metadata_fields
                sensor_color_styles = _COLOR_STYLES + metadata_fields

                self._color_style[lidar_id] = "Intensity \u03b3=1/2"
                self._point_size[lidar_id] = 0.025
                self._show_pc[lidar_id] = True
                self._is_fused[lidar_id] = False
                self._fused_range[lidar_id] = (0, max_frame)
                self._motion_comp[lidar_id] = True
                self._range_cycle[lidar_id] = 50.0
                self._height_range[lidar_id] = (-5.0, 15.0)

                frame_step_init = min(40, max_frame) if max_frame > 0 else 0
                self._fused_frame_step[lidar_id] = frame_step_init

                with self.client.gui.add_folder(lidar_id):
                    frame_slider = self.client.gui.add_slider(
                        "Frame",
                        min=0,
                        max=max_frame,
                        step=1,
                        initial_value=0,
                    )
                    self._frame_sliders[lidar_id] = frame_slider

                    pc_checkbox = self.client.gui.add_checkbox(
                        "Show Lidar", initial_value=True, hint="Show point cloud for this lidar"
                    )

                    with self.client.gui.add_folder("Point Cloud Settings"):
                        color_dropdown = self.client.gui.add_dropdown(
                            "Color Style",
                            options=sensor_color_styles,
                            initial_value="Intensity \u03b3=1/2",
                        )
                        point_size = self.client.gui.add_slider(
                            "Point Size Radius (cm)",
                            min=0,
                            max=50,
                            step=1,
                            initial_value=25,
                        )
                        range_cycle_slider = self.client.gui.add_slider(
                            "Range Cycle (m)",
                            min=5.0,
                            max=200.0,
                            step=1.0,
                            initial_value=50.0,
                        )
                        height_range_slider = self.client.gui.add_multi_slider(
                            "Height Range (m)",
                            min=-50.0,
                            max=100.0,
                            step=0.5,
                            initial_value=(-5.0, 15.0),
                        )
                        fused_checkbox = self.client.gui.add_checkbox("Fuse", initial_value=False)
                        fused_frame_step = self.client.gui.add_slider(
                            "Frame Step (Fused)",
                            min=min(1, max_frame),
                            max=max(1, max_frame),
                            step=1,
                            initial_value=frame_step_init,
                        )
                        fused_range = self.client.gui.add_multi_slider(
                            "Fused Range",
                            min=0,
                            max=max_frame,
                            step=1,
                            initial_value=(0, max_frame),
                        )
                        motion_comp_checkbox = self.client.gui.add_checkbox(
                            "Motion Compensation",
                            initial_value=True,
                        )

                    self._bind_lidar_callbacks(
                        lidar_id,
                        frame_slider,
                        color_dropdown,
                        point_size,
                        range_cycle_slider,
                        height_range_slider,
                        pc_checkbox,
                        fused_checkbox,
                        fused_frame_step,
                        fused_range,
                        motion_comp_checkbox,
                    )

    def get_frame_sliders(self) -> Dict[str, viser.GuiInputHandle[int]]:
        return dict(self._frame_sliders)

    def populate_scene(self) -> None:
        if not self._enabled:
            return
        futures = [self.renderer._executor.submit(self._update_lidar, lid) for lid in self.data_loader.lidar_ids]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception:
                logger.exception("Error populating lidar scene")

    def on_reference_frame_change(self, interval_us: HalfClosedInterval) -> None:
        if not self._enabled:
            return
        center_us = interval_us.start + (interval_us.end - interval_us.start) // 2
        for lidar_id in self.data_loader.lidar_ids:
            if not self._show_pc.get(lidar_id, True):
                continue
            sensor = self.data_loader.get_lidar_sensor(lidar_id)
            self._frame_sliders[lidar_id].value = sensor.get_closest_frame_index(center_us, relative_frame_time=0.5)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scan_metadata_color_fields(self, lidar_id: str) -> List[str]:
        """Return generic_data field names from frame 0 whose shape qualifies for coloring.

        A field qualifies if it has exactly 1, 3, or 4 values per point.
        """
        qualifying: List[str] = []
        try:
            sensor = self.data_loader.get_lidar_sensor(lidar_id)
            ts = sensor.frames_timestamps_us[0, 1].item()
            n_points = sensor.get_frame_ray_bundle_count(0)
            for name in sensor.get_frame_generic_data_names(ts):
                data = np.asarray(sensor.get_frame_generic_data(ts, name))
                if data.ndim == 1 and data.shape[0] == n_points:
                    qualifying.append(name)
                elif data.ndim == 2 and data.shape[0] == n_points and data.shape[1] in [1, 3, 4]:
                    qualifying.append(name)
        except Exception:
            logger.debug("Could not scan metadata color fields for lidar '%s'", lidar_id)
        return qualifying

    def _bind_lidar_callbacks(
        self,
        lidar_id: str,
        frame_slider: viser.GuiInputHandle[int],
        color_dropdown: viser.GuiDropdownHandle,
        point_size: viser.GuiInputHandle[int],
        range_cycle_slider: viser.GuiInputHandle[float],
        height_range_slider: Any,
        pc_checkbox: viser.GuiInputHandle[bool],
        fused_checkbox: viser.GuiInputHandle[bool],
        fused_frame_step: viser.GuiInputHandle[int],
        fused_range: Any,
        motion_comp_checkbox: viser.GuiInputHandle[bool],
    ) -> None:
        @frame_slider.on_update
        def _(_: viser.GuiEvent, _lid: str = lidar_id) -> None:
            if not self._is_fused[_lid]:
                self._update_lidar(_lid)

        @color_dropdown.on_update
        def _(_: viser.GuiEvent, _lid: str = lidar_id) -> None:
            self._color_style[_lid] = color_dropdown.value
            self._update_lidar(_lid)

        @point_size.on_update
        def _(_: viser.GuiEvent, _lid: str = lidar_id) -> None:
            self._point_size[_lid] = point_size.value / 1000.0
            self._update_lidar(_lid)

        @range_cycle_slider.on_update
        def _(_: viser.GuiEvent, _lid: str = lidar_id) -> None:
            self._range_cycle[_lid] = range_cycle_slider.value
            if self._color_style[_lid] == "Range (jet)":
                self._update_lidar(_lid)

        @height_range_slider.on_update
        def _(_: viser.GuiEvent, _lid: str = lidar_id) -> None:
            self._height_range[_lid] = height_range_slider.value
            if self._color_style[_lid] == "Height (turbo)":
                self._update_lidar(_lid)

        @pc_checkbox.on_update
        def _(_: viser.GuiEvent, _lid: str = lidar_id) -> None:
            was_hidden = not self._show_pc.get(_lid, True)
            self._show_pc[_lid] = pc_checkbox.value
            if pc_checkbox.value and was_hidden:
                self._update_lidar(_lid)
            elif _lid in self._point_clouds:
                self._point_clouds[_lid].visible = pc_checkbox.value

        @fused_checkbox.on_update
        def _(_: viser.GuiEvent, _lid: str = lidar_id) -> None:
            self._is_fused[_lid] = fused_checkbox.value
            self._update_lidar(_lid)

        @fused_frame_step.on_update
        def _(_: viser.GuiEvent, _lid: str = lidar_id) -> None:
            self._fused_frame_step[_lid] = fused_frame_step.value
            if self._is_fused[_lid]:
                self._update_lidar(_lid)

        @fused_range.on_update
        def _(_: viser.GuiEvent, _lid: str = lidar_id) -> None:
            self._fused_range[_lid] = fused_range.value
            if self._is_fused[_lid]:
                self._update_lidar(_lid)

        @motion_comp_checkbox.on_update
        def _(_: viser.GuiEvent, _lid: str = lidar_id) -> None:
            self._motion_comp[_lid] = motion_comp_checkbox.value
            self._update_lidar(_lid)

    def _update_lidar(self, lidar_id: str) -> None:
        """Re-render point cloud for *lidar_id* using current GUI state."""
        if not self._enabled:
            return
        if not self._show_pc.get(lidar_id, True):
            return
        with self.client.atomic():
            # Remove existing scene node
            if point_cloud := self._point_clouds.pop(lidar_id, None):
                point_cloud.remove()

            frame = self._frame_sliders[lidar_id].value
            is_fused = self._is_fused[lidar_id]
            fused_range = self._fused_range[lidar_id]
            fused_step = self._fused_frame_step[lidar_id]
            point_size = self._point_size[lidar_id]
            visible = self._show_pc[lidar_id]

            pc_handle = self._create_point_cloud(
                lidar_id,
                frame,
                is_fused,
                fused_range,
                fused_step,
                point_size,
                visible,
            )
            self._point_clouds[lidar_id] = pc_handle
        self.client.flush()

    def _create_point_cloud(
        self,
        lidar_id: str,
        frame: int,
        is_fused: bool,
        fused_range: Tuple[int, int],
        fused_step: int,
        point_size: float,
        visible: bool,
    ) -> viser.PointCloudHandle:
        """Create a single viser point cloud node."""
        color_style = self._color_style[lidar_id]
        is_metadata_style = color_style in self._metadata_color_fields.get(lidar_id, [])

        if is_fused:
            handle_name = f"/lidars/{lidar_id}/fused_point_cloud"
            needs_per_frame = color_style in _PER_FRAME_COLOR_STYLES or is_metadata_style
            if needs_per_frame:
                points_world, colors = self._get_fused_point_cloud_with_colors(
                    lidar_id, fused_range[0], fused_range[1], fused_step
                )
            else:
                points_world, points_sensor = self._get_fused_point_cloud(
                    lidar_id, fused_range[0], fused_range[1], fused_step
                )
                colors = self._colorize_points(lidar_id, frame, points_sensor, points_world)
        else:
            handle_name = f"/lidars/{lidar_id}/point_cloud"
            points_sensor = self._get_point_cloud_sensor(lidar_id, frame)
            points_world = self._transform_to_world(lidar_id, frame, points_sensor)
            colors = self._colorize_points(lidar_id, frame, points_sensor, points_world)

        return self.client.scene.add_point_cloud(
            handle_name,
            points=points_world,
            colors=colors,
            point_size=point_size,
            point_shape="circle",
            visible=visible,
        )

    # ------------------------------------------------------------------
    # Point cloud loading
    # ------------------------------------------------------------------

    def _get_point_cloud_sensor(self, lidar_id: str, frame: int) -> np.ndarray:
        """Load a single-frame point cloud in sensor coordinates."""
        sensor = self.data_loader.get_lidar_sensor(lidar_id)
        motion_comp = self._motion_comp[lidar_id]
        pc = sensor.get_frame_point_cloud(frame, motion_compensation=motion_comp, with_start_points=False)
        return pc.xyz_m_end

    def _transform_to_world(self, lidar_id: str, frame: int, points_sensor: np.ndarray) -> np.ndarray:
        """Transform sensor-frame points to world coordinates."""
        sensor = self.data_loader.get_lidar_sensor(lidar_id)
        T_sensor_world = sensor.get_frames_T_sensor_target(self.data_loader.world_frame_id, frame, FrameTimepoint.END)
        return transform_point_cloud(points_sensor, T_sensor_world)

    def _get_fused_point_cloud(
        self,
        lidar_id: str,
        start_frame: int,
        end_frame: int,
        step: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Accumulate point clouds over a frame range (parallel frame loading).

        Returns:
            Tuple of (world-frame points, sensor-frame points).
        """
        frames = list(range(start_frame, end_frame + 1, max(1, step)))

        def _load(frame: int) -> Tuple[np.ndarray, np.ndarray]:
            pts_sensor = self._get_point_cloud_sensor(lidar_id, frame)
            pts_world = self._transform_to_world(lidar_id, frame, pts_sensor)
            return pts_world, pts_sensor

        results = list(self.renderer._executor.map(_load, frames))
        all_world = [r[0] for r in results]
        all_sensor = [r[1] for r in results]
        return np.concatenate(all_world), np.concatenate(all_sensor)

    def _get_fused_point_cloud_with_colors(
        self,
        lidar_id: str,
        start_frame: int,
        end_frame: int,
        step: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Accumulate point clouds with per-frame coloring over a frame range (parallel).

        Used for color styles that require per-frame data (intensity, timestamp,
        model element).

        Returns:
            Tuple of (world-frame points, colors).
        """
        frames = list(range(start_frame, end_frame + 1, max(1, step)))

        def _load_and_color(frame: int) -> Tuple[np.ndarray, np.ndarray]:
            pts_sensor = self._get_point_cloud_sensor(lidar_id, frame)
            pts_world = self._transform_to_world(lidar_id, frame, pts_sensor)
            colors = self._colorize_points(lidar_id, frame, pts_sensor, pts_world)
            return pts_world, colors

        results = list(self.renderer._executor.map(_load_and_color, frames))
        all_world = [r[0] for r in results]
        all_colors = [r[1] for r in results]
        return np.concatenate(all_world), np.concatenate(all_colors)

    # ------------------------------------------------------------------
    # Coloring
    # ------------------------------------------------------------------

    def _colorize_points(
        self,
        lidar_id: str,
        frame: int,
        points_sensor: np.ndarray,
        points_world: np.ndarray,
    ) -> np.ndarray:
        """Compute per-point RGB colors based on the active color style.

        Args:
            lidar_id: Lidar sensor ID.
            frame: Current frame index (used for per-frame data lookups).
            points_sensor: Point cloud in sensor coordinates ``[N, 3]``.
            points_world: Point cloud in world coordinates ``[N, 3]``.

        Returns:
            ``uint8`` color array of shape ``[N, 3]``.
        """
        color_style = self._color_style[lidar_id]
        n_points = points_world.shape[0]

        if color_style == "Range (jet)":
            return self._color_range_jet(points_sensor, lidar_id)
        if color_style == "Height (turbo)":
            return self._color_height_turbo(points_world, lidar_id)
        if color_style == "Timestamp":
            return self._color_timestamp(lidar_id, frame, n_points)
        if color_style == "Model Row":
            return self._color_model_element(lidar_id, frame, n_points, column=0)
        if color_style == "Model Column":
            return self._color_model_element(lidar_id, frame, n_points, column=1)
        if color_style in self._metadata_color_fields.get(lidar_id, []):
            return self._color_metadata_field(lidar_id, frame, color_style, n_points)

        # Intensity-based modes
        return self._color_intensity(lidar_id, frame, color_style, n_points)

    def _color_intensity(self, lidar_id: str, frame: int, style: str, n_points: int) -> np.ndarray:
        """Intensity-based coloring with optional gamma correction."""
        sensor = self.data_loader.get_lidar_sensor(lidar_id)
        try:
            intensity = sensor.get_frame_ray_bundle_return_intensity(frame, return_index=0)
        except Exception:
            return np.tile(_DEFAULT_POINT_COLOR, (n_points, 1))

        intensity = np.clip(intensity, 0.0, 1.0)

        if style == "Intensity \u03b3=1/2":
            intensity = np.power(intensity, 0.5)
        elif style == "Intensity \u03b3=1/4":
            intensity = np.power(intensity, 0.25)

        gray = (intensity * 255.0).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=1)

    def _color_range_jet(self, points_sensor: np.ndarray, lidar_id: str) -> np.ndarray:
        """Color by range from sensor origin using the jet colormap."""
        cycle = max(1.0, self._range_cycle[lidar_id])
        ranges = np.linalg.norm(points_sensor, axis=1)
        normalized = (ranges % cycle) / cycle
        rgba = _JET_CMAP(normalized)  # [N, 4] float in [0, 1]
        return (rgba[:, :3] * 255.0).astype(np.uint8)

    def _color_height_turbo(self, points_world: np.ndarray, lidar_id: str) -> np.ndarray:
        """Color by world-frame Z height using the turbo colormap."""
        z_min, z_max = self._height_range[lidar_id]
        z_range = max(z_max - z_min, 0.01)
        z = points_world[:, 2]
        normalized = np.clip((z - z_min) / z_range, 0.0, 1.0)
        rgba = _TURBO_CMAP(normalized)  # [N, 4] float in [0, 1]
        return (rgba[:, :3] * 255.0).astype(np.uint8)

    def _color_timestamp(self, lidar_id: str, frame: int, n_points: int) -> np.ndarray:
        """Color by per-point timestamp within the frame using the turbo colormap."""
        sensor = self.data_loader.get_lidar_sensor(lidar_id)
        try:
            timestamps = sensor.get_frame_ray_bundle_timestamp_us(frame)
        except Exception:
            return np.tile(_DEFAULT_POINT_COLOR, (n_points, 1))

        ts_min = float(timestamps.min())
        ts_max = float(timestamps.max())
        ts_range = max(ts_max - ts_min, 1.0)
        normalized = (timestamps.astype(np.float64) - ts_min) / ts_range
        rgba = _TURBO_CMAP(normalized)
        return (rgba[:, :3] * 255.0).astype(np.uint8)

    def _color_model_element(self, lidar_id: str, frame: int, n_points: int, column: int) -> np.ndarray:
        """Color by model element index (row or column) using the turbo colormap.

        Args:
            lidar_id: Lidar sensor ID.
            frame: Frame index.
            n_points: Number of points (for fallback).
            column: 0 for row index, 1 for column index.
        """
        sensor = self.data_loader.get_lidar_sensor(lidar_id)
        model_elements: Optional[np.ndarray] = None
        try:
            model_elements = sensor.get_frame_ray_bundle_model_element(frame)
        except Exception:
            pass

        if model_elements is None:
            return np.tile(_DEFAULT_POINT_COLOR, (n_points, 1))

        indices = model_elements[:, column].astype(np.float64)
        idx_max = max(float(indices.max()), 1.0)
        normalized = indices / idx_max
        rgba = _TURBO_CMAP(normalized)
        return (rgba[:, :3] * 255.0).astype(np.uint8)

    def _color_metadata_field(self, lidar_id: str, frame: int, field_name: str, n_points: int) -> np.ndarray:
        """Color by a generic_data metadata field.

        Single-channel fields are treated as intensity (grayscale).
        Three-channel fields are treated as RGB.
        Four-channel fields are treated as RGBA (alpha-premultiplied onto black).

        Float data is assumed to be in [0, 1] and scaled to uint8.
        Integer data is scaled linearly by its dtype maximum.
        """
        try:
            sensor = self.data_loader.get_lidar_sensor(lidar_id)
            ts = sensor.frames_timestamps_us[frame, 1].item()
            data = np.asarray(sensor.get_frame_generic_data(ts, field_name))
        except Exception:
            return np.tile(_DEFAULT_POINT_COLOR, (n_points, 1))

        # Convert to uint8
        if data.dtype == np.uint8:
            data_u8 = data
        elif np.issubdtype(data.dtype, np.floating):
            data_u8 = (np.clip(data, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            data_u8 = (data.astype(np.float64) / np.iinfo(data.dtype).max * 255.0).astype(np.uint8)

        if data_u8.ndim == 1 or (data_u8.ndim == 2 and data_u8.shape[1] == 1):
            gray = data_u8.ravel()
            return np.stack([gray, gray, gray], axis=1)
        if data_u8.shape[1] == 3:
            return data_u8
        # 4-channel RGBA: premultiply RGB by normalised alpha onto black background
        rgb = data_u8[:, :3].astype(np.float32)
        alpha = data_u8[:, 3:4].astype(np.float32) / 255.0
        return np.clip(rgb * alpha, 0, 255).astype(np.uint8)
