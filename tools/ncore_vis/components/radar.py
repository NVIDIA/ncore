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

"""Radar point cloud visualization component with fusing and motion compensation.

Unlike the lidar component which duplicates settings per sensor, the radar component
uses a single set of shared visualization settings (color style, point size, fusing,
motion compensation) applied to all radar sensors.  Each sensor only has its own
frame slider and show/hide toggle.
"""

from __future__ import annotations

import concurrent.futures
import logging

from typing import Any, Dict, List, Tuple

import matplotlib
import numpy as np
import viser

from ncore.impl.common.transformations import HalfClosedInterval, transform_point_cloud
from ncore.impl.data.types import FrameTimepoint
from tools.ncore_vis.components.base import VisualizationComponent, register_component


logger = logging.getLogger(__name__)

_DEFAULT_POINT_COLOR: np.ndarray = np.array([0, 255, 0], dtype=np.uint8)

# Pre-fetch colormaps once at module level.
_JET_CMAP: matplotlib.colors.Colormap = matplotlib.colormaps["jet"]
_TURBO_CMAP: matplotlib.colors.Colormap = matplotlib.colormaps["turbo"]

# Supported color styles (no intensity or model element styles for radar).
_COLOR_STYLES: List[str] = [
    "Range (jet)",
    "Height (turbo)",
    "Timestamp",
]

# Color styles that require per-frame data (cannot be computed from positions alone).
_PER_FRAME_COLOR_STYLES: frozenset[str] = frozenset(
    [
        "Timestamp",
    ]
)


@register_component
class RadarComponent(VisualizationComponent):
    """Radar point cloud visualization with shared settings, fusing, and motion compensation.

    A single set of visualization controls (color style, point size, fusing, motion
    compensation) is shared across all radar sensors.  Each sensor only exposes a
    frame slider and a show/hide toggle.
    """

    def __init__(self, client: Any, data_loader: Any, renderer: Any) -> None:
        super().__init__(client, data_loader, renderer)

        # Scan for per-sensor generic_data fields that can drive point color.
        self._metadata_color_fields: Dict[str, List[str]] = {
            radar_id: self._scan_metadata_color_fields(radar_id) for radar_id in self.data_loader.radar_ids
        }

        # Build the union of all metadata color fields across sensors.
        seen: dict[str, None] = {}
        for fields in self._metadata_color_fields.values():
            for f in fields:
                seen.setdefault(f, None)
        self._all_metadata_color_fields: List[str] = list(seen)

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------

    def create_gui(self, tab_group: viser.GuiTabGroupHandle) -> None:  # noqa: C901
        self._enabled: bool = True
        self._point_clouds: Dict[str, viser.PointCloudHandle] = {}

        # Per-sensor state (only frame index and visibility).
        self._frame_sliders: Dict[str, viser.GuiInputHandle[int]] = {}
        self._show_pc: Dict[str, bool] = {}

        # Shared visualization settings (single values applied to all radars).
        default_color = "radial_velocity" if "radial_velocity" in self._all_metadata_color_fields else "Range (jet)"
        self._color_style: str = default_color
        self._point_size: float = 0.05
        self._is_fused: bool = False
        self._fused_frame_step: int = 0
        self._fused_range: Tuple[int, int] = (0, 0)
        self._motion_comp: bool = True
        self._range_cycle: float = 50.0
        self._height_range: Tuple[float, float] = (-5.0, 15.0)

        # Early return if no radar sensors available (no empty tab).
        if not self.data_loader.radar_ids:
            return

        # Determine global max frame across all radars (for fusing sliders).
        global_max_frame = 0
        for radar_id in self.data_loader.radar_ids:
            sensor = self.data_loader.get_radar_sensor(radar_id)
            global_max_frame = max(global_max_frame, max(0, sensor.frames_count - 1))

        self._fused_range = (0, global_max_frame)
        self._fused_frame_step = min(40, global_max_frame) if global_max_frame > 0 else 0

        with tab_group.add_tab("Radars"):
            enabled_checkbox = self.client.gui.add_checkbox(
                "Enabled", initial_value=True, hint="Enable radar point cloud visualization"
            )

            @enabled_checkbox.on_update
            def _(_: viser.GuiEvent) -> None:
                self._enabled = enabled_checkbox.value
                for rid in self._point_clouds:
                    self._point_clouds[rid].visible = enabled_checkbox.value and self._show_pc.get(rid, True)

            # -- Shared settings folder --
            with self.client.gui.add_folder("Settings"):
                color_dropdown = self.client.gui.add_dropdown(
                    "Color Style",
                    options=_COLOR_STYLES + self._all_metadata_color_fields,
                    initial_value=default_color,
                )
                point_size_slider = self.client.gui.add_slider(
                    "Point Size Radius (cm)",
                    min=0,
                    max=100,
                    step=1,
                    initial_value=50,
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
                motion_comp_checkbox = self.client.gui.add_checkbox(
                    "Motion Compensation",
                    initial_value=True,
                )
                fused_checkbox = self.client.gui.add_checkbox("Fuse", initial_value=False)
                fused_frame_step_slider = self.client.gui.add_slider(
                    "Frame Step (Fused)",
                    min=min(1, global_max_frame),
                    max=max(1, global_max_frame),
                    step=1,
                    initial_value=self._fused_frame_step,
                )
                fused_range_slider = self.client.gui.add_multi_slider(
                    "Fused Range",
                    min=0,
                    max=global_max_frame,
                    step=1,
                    initial_value=(0, global_max_frame),
                )

            self._bind_shared_settings(
                color_dropdown=color_dropdown,
                point_size_slider=point_size_slider,
                range_cycle_slider=range_cycle_slider,
                height_range_slider=height_range_slider,
                motion_comp_checkbox=motion_comp_checkbox,
                fused_checkbox=fused_checkbox,
                fused_frame_step_slider=fused_frame_step_slider,
                fused_range_slider=fused_range_slider,
            )

            # -- Per-radar sensor folders (frame slider + show/hide only) --
            for radar_id in self.data_loader.radar_ids:
                sensor = self.data_loader.get_radar_sensor(radar_id)
                max_frame = max(0, sensor.frames_count - 1)
                self._show_pc[radar_id] = True

                with self.client.gui.add_folder(radar_id):
                    frame_slider = self.client.gui.add_slider("Frame", min=0, max=max_frame, step=1, initial_value=0)
                    self._frame_sliders[radar_id] = frame_slider

                    show_checkbox = self.client.gui.add_checkbox(
                        "Show", initial_value=True, hint=f"Show point cloud for {radar_id}"
                    )

                self._bind_per_radar_callbacks(radar_id, frame_slider, show_checkbox)

    def get_frame_sliders(self) -> Dict[str, viser.GuiInputHandle[int]]:
        return dict(self._frame_sliders)

    def populate_scene(self) -> None:
        if not self._enabled:
            return
        futures = [self.renderer._executor.submit(self._update_radar, rid) for rid in self.data_loader.radar_ids]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception:
                logger.exception("Error populating radar scene")

    def on_reference_frame_change(self, interval_us: HalfClosedInterval) -> None:
        if not self._enabled:
            return
        center_us = interval_us.start + (interval_us.end - interval_us.start) // 2
        for radar_id in self.data_loader.radar_ids:
            if not self._show_pc.get(radar_id, True):
                continue
            if radar_id not in self._frame_sliders:
                continue
            sensor = self.data_loader.get_radar_sensor(radar_id)
            self._frame_sliders[radar_id].value = sensor.get_closest_frame_index(center_us, relative_frame_time=0.5)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scan_metadata_color_fields(self, radar_id: str) -> List[str]:
        """Return generic_data field names from frame 0 whose shape qualifies for coloring."""
        sensor = self.data_loader.get_radar_sensor(radar_id)

        if sensor.frames_count == 0:
            return []

        n_points = sensor.get_frame_ray_bundle_count(0)
        if n_points == 0:
            return []

        color_metadata_fields: List[str] = []
        for name in sensor.get_frame_generic_data_names(0):
            data = np.asarray(sensor.get_frame_generic_data(0, name))
            if data.ndim == 1 and data.shape[0] == n_points:
                color_metadata_fields.append(name)
            elif data.ndim == 2 and data.shape[0] == n_points and data.shape[1] in [1, 3, 4]:
                color_metadata_fields.append(name)
        return color_metadata_fields

    def _bind_shared_settings(
        self,
        color_dropdown: viser.GuiDropdownHandle,
        point_size_slider: viser.GuiInputHandle[int],
        range_cycle_slider: viser.GuiInputHandle[float],
        height_range_slider: Any,
        motion_comp_checkbox: viser.GuiInputHandle[bool],
        fused_checkbox: viser.GuiInputHandle[bool],
        fused_frame_step_slider: viser.GuiInputHandle[int],
        fused_range_slider: Any,
    ) -> None:
        """Wire up shared visualization setting callbacks — each triggers a refresh of all radars."""

        @color_dropdown.on_update
        def _(_: viser.GuiEvent) -> None:
            self._color_style = color_dropdown.value
            self._refresh_all()

        @point_size_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            self._point_size = point_size_slider.value / 1000.0
            self._refresh_all()

        @range_cycle_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            self._range_cycle = range_cycle_slider.value
            if self._color_style == "Range (jet)":
                self._refresh_all()

        @height_range_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            self._height_range = height_range_slider.value
            if self._color_style == "Height (turbo)":
                self._refresh_all()

        @motion_comp_checkbox.on_update
        def _(_: viser.GuiEvent) -> None:
            self._motion_comp = motion_comp_checkbox.value
            self._refresh_all()

        @fused_checkbox.on_update
        def _(_: viser.GuiEvent) -> None:
            self._is_fused = fused_checkbox.value
            self._refresh_all()

        @fused_frame_step_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            self._fused_frame_step = fused_frame_step_slider.value
            if self._is_fused:
                self._refresh_all()

        @fused_range_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            self._fused_range = fused_range_slider.value
            if self._is_fused:
                self._refresh_all()

    def _bind_per_radar_callbacks(
        self,
        radar_id: str,
        frame_slider: viser.GuiInputHandle[int],
        show_checkbox: viser.GuiInputHandle[bool],
    ) -> None:
        """Wire up per-radar callbacks (frame change, show/hide)."""

        @frame_slider.on_update
        def _(_: viser.GuiEvent, _rid: str = radar_id) -> None:
            if not self._is_fused:
                self._update_radar(_rid)

        @show_checkbox.on_update
        def _(_: viser.GuiEvent, _rid: str = radar_id) -> None:
            was_hidden = not self._show_pc.get(_rid, True)
            self._show_pc[_rid] = show_checkbox.value
            if show_checkbox.value and was_hidden:
                self._update_radar(_rid)
            elif _rid in self._point_clouds:
                self._point_clouds[_rid].visible = show_checkbox.value and self._enabled

    def _refresh_all(self) -> None:
        """Re-render all visible radar point clouds."""
        for radar_id in self.data_loader.radar_ids:
            if self._show_pc.get(radar_id, True):
                self._update_radar(radar_id)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _update_radar(self, radar_id: str) -> None:
        """Re-render point cloud for *radar_id* using current GUI state."""
        if not self._enabled:
            return
        if not self._show_pc.get(radar_id, True):
            return
        with self.client.atomic():
            if point_cloud := self._point_clouds.pop(radar_id, None):
                point_cloud.remove()

            frame_idx = self._frame_sliders[radar_id].value
            visible = self._show_pc[radar_id]

            pc_handle = self._create_point_cloud(radar_id, frame_idx, visible)
            self._point_clouds[radar_id] = pc_handle
        self.client.flush()

    def _create_point_cloud(
        self,
        radar_id: str,
        frame_idx: int,
        visible: bool,
    ) -> viser.PointCloudHandle:
        """Create a single viser point cloud node."""
        color_style = self._color_style
        is_metadata_style = color_style in self._metadata_color_fields.get(radar_id, [])

        if self._is_fused:
            handle_name = f"/radars/{radar_id}/fused_point_cloud"
            needs_per_frame = color_style in _PER_FRAME_COLOR_STYLES or is_metadata_style
            if needs_per_frame:
                points_world, colors = self._get_fused_point_cloud_with_colors(
                    radar_id, self._fused_range[0], self._fused_range[1], self._fused_frame_step
                )
            else:
                points_world, points_sensor = self._get_fused_point_cloud(
                    radar_id, self._fused_range[0], self._fused_range[1], self._fused_frame_step
                )
                colors = self._colorize_points(radar_id, frame_idx, points_sensor, points_world)
        else:
            handle_name = f"/radars/{radar_id}/point_cloud"
            points_sensor = self._get_point_cloud_sensor(radar_id, frame_idx)
            points_world = self._transform_to_world(radar_id, frame_idx, points_sensor)
            colors = self._colorize_points(radar_id, frame_idx, points_sensor, points_world)

        return self.client.scene.add_point_cloud(
            handle_name,
            points=points_world,
            colors=colors,
            point_size=self._point_size,
            point_shape="circle",
            visible=visible,
        )

    # ------------------------------------------------------------------
    # Point cloud loading
    # ------------------------------------------------------------------

    def _get_point_cloud_sensor(self, radar_id: str, frame_idx: int) -> np.ndarray:
        """Load a single-frame point cloud in sensor coordinates."""
        sensor = self.data_loader.get_radar_sensor(radar_id)
        pc = sensor.get_frame_point_cloud(frame_idx, motion_compensation=self._motion_comp, with_start_points=False)
        return pc.xyz_m_end

    def _transform_to_world(self, radar_id: str, frame_idx: int, points_sensor: np.ndarray) -> np.ndarray:
        """Transform sensor-frame points to world coordinates."""
        sensor = self.data_loader.get_radar_sensor(radar_id)
        T_sensor_world = sensor.get_frames_T_sensor_target(
            self.data_loader.world_frame_id, frame_idx, FrameTimepoint.END
        )
        return transform_point_cloud(points_sensor, T_sensor_world)

    def _get_fused_point_cloud(
        self,
        radar_id: str,
        start_frame_idx: int,
        end_frame_idx: int,
        step: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Accumulate point clouds over a frame range.

        Returns:
            Tuple of (world-frame points, sensor-frame points).
        """
        sensor = self.data_loader.get_radar_sensor(radar_id)
        max_frame = max(0, sensor.frames_count - 1)
        # Clamp range to this sensor's actual frame count.
        end = min(end_frame_idx, max_frame)
        frame_idxs = list(range(start_frame_idx, end + 1, max(1, step)))

        def _load(frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
            pts_sensor = self._get_point_cloud_sensor(radar_id, frame_idx)
            pts_world = self._transform_to_world(radar_id, frame_idx, pts_sensor)
            return pts_world, pts_sensor

        results = list(self.renderer._executor.map(_load, frame_idxs))
        all_world = [r[0] for r in results]
        all_sensor = [r[1] for r in results]
        return np.concatenate(all_world), np.concatenate(all_sensor)

    def _get_fused_point_cloud_with_colors(
        self,
        radar_id: str,
        start_frame_idx: int,
        end_frame_idx: int,
        step: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Accumulate point clouds with per-frame coloring over a frame range.

        Returns:
            Tuple of (world-frame points, colors).
        """
        sensor = self.data_loader.get_radar_sensor(radar_id)
        max_frame = max(0, sensor.frames_count - 1)
        end = min(end_frame_idx, max_frame)
        frame_idxs = list(range(start_frame_idx, end + 1, max(1, step)))

        def _load_and_color(frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
            pts_sensor = self._get_point_cloud_sensor(radar_id, frame_idx)
            pts_world = self._transform_to_world(radar_id, frame_idx, pts_sensor)
            colors = self._colorize_points(radar_id, frame_idx, pts_sensor, pts_world)
            return pts_world, colors

        results = list(self.renderer._executor.map(_load_and_color, frame_idxs))
        all_world = [r[0] for r in results]
        all_colors = [r[1] for r in results]
        return np.concatenate(all_world), np.concatenate(all_colors)

    # ------------------------------------------------------------------
    # Coloring
    # ------------------------------------------------------------------

    def _colorize_points(
        self,
        radar_id: str,
        frame_idx: int,
        points_sensor: np.ndarray,
        points_world: np.ndarray,
    ) -> np.ndarray:
        """Compute per-point RGB colors based on the active color style."""
        color_style = self._color_style
        n_points = points_world.shape[0]

        if color_style == "Range (jet)":
            return self._color_range_jet(points_sensor)
        if color_style == "Height (turbo)":
            return self._color_height_turbo(points_world)
        if color_style == "Timestamp":
            return self._color_timestamp(radar_id, frame_idx, n_points)
        if color_style in self._metadata_color_fields.get(radar_id, []):
            return self._color_metadata_field(radar_id, frame_idx, color_style, n_points)

        return np.tile(_DEFAULT_POINT_COLOR, (n_points, 1))

    def _color_range_jet(self, points_sensor: np.ndarray) -> np.ndarray:
        """Color by range from sensor origin using the jet colormap."""
        cycle = max(1.0, self._range_cycle)
        ranges = np.linalg.norm(points_sensor, axis=1)
        normalized = (ranges % cycle) / cycle
        rgba = _JET_CMAP(normalized)
        return (rgba[:, :3] * 255.0).astype(np.uint8)

    def _color_height_turbo(self, points_world: np.ndarray) -> np.ndarray:
        """Color by world-frame Z height using the turbo colormap."""
        z_min, z_max = self._height_range
        z_range = max(z_max - z_min, 0.01)
        z = points_world[:, 2]
        normalized = np.clip((z - z_min) / z_range, 0.0, 1.0)
        rgba = _TURBO_CMAP(normalized)
        return (rgba[:, :3] * 255.0).astype(np.uint8)

    def _color_timestamp(self, radar_id: str, frame_idx: int, n_points: int) -> np.ndarray:
        """Color by per-point timestamp within the frame using the turbo colormap."""
        sensor = self.data_loader.get_radar_sensor(radar_id)
        try:
            timestamps = sensor.get_frame_ray_bundle_timestamp_us(frame_idx)
        except Exception:
            return np.tile(_DEFAULT_POINT_COLOR, (n_points, 1))

        ts_min = float(timestamps.min())
        ts_max = float(timestamps.max())
        ts_range = max(ts_max - ts_min, 1.0)
        normalized = (timestamps.astype(np.float64) - ts_min) / ts_range
        rgba = _TURBO_CMAP(normalized)
        return (rgba[:, :3] * 255.0).astype(np.uint8)

    def _color_metadata_field(self, radar_id: str, frame_idx: int, field_name: str, n_points: int) -> np.ndarray:
        """Color by a generic_data metadata field.

        Single-channel fields use turbo colormap (better for Doppler, RCS, etc.).
        """
        try:
            sensor = self.data_loader.get_radar_sensor(radar_id)
            data = np.asarray(sensor.get_frame_generic_data(frame_idx, field_name))
        except (KeyError, IndexError):
            return np.tile(_DEFAULT_POINT_COLOR, (n_points, 1))

        # Single-channel: use turbo colormap.
        if data.ndim == 1 or (data.ndim == 2 and data.shape[1] == 1):
            values = data.ravel().astype(np.float64)
            v_min = float(values.min())
            v_max = float(values.max())
            v_range = max(v_max - v_min, 1e-9)
            normalized = (values - v_min) / v_range
            rgba = _TURBO_CMAP(normalized)
            return (rgba[:, :3] * 255.0).astype(np.uint8)

        # Multi-channel uint8: return as-is.
        if data.dtype == np.uint8:
            return data[:, :3] if data.shape[1] >= 3 else np.tile(_DEFAULT_POINT_COLOR, (n_points, 1))

        # Multi-channel float: normalize and convert.
        if not np.issubdtype(data.dtype, np.floating):
            data = data.astype(np.float32)
        min_val = data.min()
        max_val = data.max()
        drange = max_val - min_val
        if drange == 0.0:
            return np.tile(_DEFAULT_POINT_COLOR, (n_points, 1))
        return ((data[:, :3] - min_val) / drange * 255.0).astype(np.uint8)
