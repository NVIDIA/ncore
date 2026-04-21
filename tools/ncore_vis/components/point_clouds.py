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

"""Point cloud visualization component for native PointCloudsSource data."""

from __future__ import annotations

import logging

from typing import Any, Dict, List, Tuple

import matplotlib
import numpy as np
import viser

from ncore.impl.common.transformations import HalfClosedInterval
from tools.ncore_vis.components.base import VisualizationComponent, register_component


logger = logging.getLogger(__name__)

_DEFAULT_POINT_COLOR: np.ndarray = np.array([0, 128, 255], dtype=np.uint8)

# Pre-fetch colormaps once at module level.
_TURBO_CMAP: matplotlib.colors.Colormap = matplotlib.colormaps["turbo"]
_JET_CMAP: matplotlib.colors.Colormap = matplotlib.colormaps["jet"]

# Base color styles (always available).
_BASE_COLOR_STYLES: List[str] = [
    "Uniform",
    "Height (turbo)",
    "Range (jet)",
]


@register_component
class PointCloudsComponent(VisualizationComponent):
    """Visualization component for native point-clouds sources.

    For each point-clouds source, provides controls for point cloud index selection,
    coloring style (uniform, rgb attribute, height-based), and point size.
    """

    def __init__(self, client: Any, data_loader: Any, renderer: Any) -> None:
        super().__init__(client, data_loader, renderer)

        # Discover which sources have an 'rgb' attribute by probing pc 0.
        self._has_rgb: Dict[str, bool] = {}
        for source_id in self.data_loader.point_clouds_ids:
            source = self.data_loader.get_point_clouds_source(source_id)
            if source.pcs_count > 0:
                pc = source.get_pc(0)
                self._has_rgb[source_id] = pc.has_attribute("rgb")
            else:
                self._has_rgb[source_id] = False

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------

    def create_gui(self, tab_group: viser.GuiTabGroupHandle) -> None:  # noqa: C901
        self._enabled: bool = True
        self._point_clouds: Dict[str, viser.PointCloudHandle] = {}

        # Per-source state.
        self._pc_sliders: Dict[str, viser.GuiInputHandle[int]] = {}
        self._color_style: Dict[str, str] = {}
        self._point_size: Dict[str, float] = {}
        self._show_pc: Dict[str, bool] = {}
        self._height_range: Dict[str, Tuple[float, float]] = {}
        self._range_cycle: Dict[str, float] = {}

        # Early return if no point-clouds sources available (no empty tab).
        if not self.data_loader.point_clouds_ids:
            return

        with tab_group.add_tab("Point Clouds"):
            enabled_checkbox = self.client.gui.add_checkbox(
                "Enabled", initial_value=True, hint="Enable point cloud visualization"
            )

            @enabled_checkbox.on_update
            def _(_: viser.GuiEvent) -> None:
                self._enabled = enabled_checkbox.value
                for sid in self._point_clouds:
                    self._point_clouds[sid].visible = enabled_checkbox.value and self._show_pc.get(sid, True)

            for source_id in self.data_loader.point_clouds_ids:
                source = self.data_loader.get_point_clouds_source(source_id)
                pcs_count = source.pcs_count
                max_pc = max(0, pcs_count - 1)

                # Build color style options for this source.
                color_options = list(_BASE_COLOR_STYLES)
                if self._has_rgb.get(source_id, False):
                    color_options.insert(0, "RGB")

                # Default to RGB if available, otherwise Uniform.
                default_color = "RGB" if self._has_rgb.get(source_id, False) else "Uniform"

                self._color_style[source_id] = default_color
                self._point_size[source_id] = 0.025
                self._show_pc[source_id] = True
                self._height_range[source_id] = (-5.0, 15.0)
                self._range_cycle[source_id] = 50.0

                with self.client.gui.add_folder(source_id):
                    pc_slider = self.client.gui.add_slider(
                        "PC Index",
                        min=0,
                        max=max_pc,
                        step=1,
                        initial_value=0,
                    )
                    self._pc_sliders[source_id] = pc_slider

                    show_checkbox = self.client.gui.add_checkbox(
                        "Show", initial_value=True, hint=f"Show point cloud for {source_id}"
                    )

                    with self.client.gui.add_folder("Settings"):
                        color_dropdown = self.client.gui.add_dropdown(
                            "Color Style",
                            options=color_options,
                            initial_value=default_color,
                        )
                        point_size_slider = self.client.gui.add_slider(
                            "Point Size Radius (cm)",
                            min=0,
                            max=50,
                            step=1,
                            initial_value=25,
                        )
                        height_range_slider = self.client.gui.add_multi_slider(
                            "Height Range (m)",
                            min=-50.0,
                            max=100.0,
                            step=0.5,
                            initial_value=(-5.0, 15.0),
                        )
                        range_cycle_slider = self.client.gui.add_slider(
                            "Range Cycle (m)",
                            min=5.0,
                            max=200.0,
                            step=1.0,
                            initial_value=50.0,
                        )

                    self._bind_callbacks(
                        source_id,
                        pc_slider,
                        show_checkbox,
                        color_dropdown,
                        point_size_slider,
                        height_range_slider,
                        range_cycle_slider,
                    )

    def get_frame_sliders(self) -> Dict[str, viser.GuiInputHandle[int]]:
        return dict(self._pc_sliders)

    def populate_scene(self) -> None:
        if not self._enabled:
            return
        for source_id in self.data_loader.point_clouds_ids:
            try:
                self._update_point_cloud(source_id)
            except Exception:
                logger.exception("Error populating point cloud scene for %s", source_id)

    def on_reference_frame_change(self, interval_us: HalfClosedInterval) -> None:
        if not self._enabled:
            return
        center_us = interval_us.start + (interval_us.end - interval_us.start) // 2
        for source_id in self.data_loader.point_clouds_ids:
            if not self._show_pc.get(source_id, True):
                continue
            if source_id not in self._pc_sliders:
                continue
            source = self.data_loader.get_point_clouds_source(source_id)
            if source.pcs_count == 0:
                continue
            # Find closest pc by timestamp.
            timestamps = source.pc_timestamps_us
            idx = int(np.argmin(np.abs(timestamps.astype(np.int64) - int(center_us))))
            self._pc_sliders[source_id].value = idx

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bind_callbacks(
        self,
        source_id: str,
        pc_slider: viser.GuiInputHandle[int],
        show_checkbox: viser.GuiInputHandle[bool],
        color_dropdown: viser.GuiDropdownHandle,
        point_size_slider: viser.GuiInputHandle[int],
        height_range_slider: Any,
        range_cycle_slider: viser.GuiInputHandle[float],
    ) -> None:
        @pc_slider.on_update
        def _(_: viser.GuiEvent, _sid: str = source_id) -> None:
            self._update_point_cloud(_sid)

        @show_checkbox.on_update
        def _(_: viser.GuiEvent, _sid: str = source_id) -> None:
            was_hidden = not self._show_pc.get(_sid, True)
            self._show_pc[_sid] = show_checkbox.value
            if show_checkbox.value and was_hidden:
                self._update_point_cloud(_sid)
            elif _sid in self._point_clouds:
                self._point_clouds[_sid].visible = show_checkbox.value and self._enabled

        @color_dropdown.on_update
        def _(_: viser.GuiEvent, _sid: str = source_id) -> None:
            self._color_style[_sid] = color_dropdown.value
            self._update_point_cloud(_sid)

        @point_size_slider.on_update
        def _(_: viser.GuiEvent, _sid: str = source_id) -> None:
            self._point_size[_sid] = point_size_slider.value / 1000.0
            self._update_point_cloud(_sid)

        @height_range_slider.on_update
        def _(_: viser.GuiEvent, _sid: str = source_id) -> None:
            self._height_range[_sid] = height_range_slider.value
            if self._color_style[_sid] == "Height (turbo)":
                self._update_point_cloud(_sid)

        @range_cycle_slider.on_update
        def _(_: viser.GuiEvent, _sid: str = source_id) -> None:
            self._range_cycle[_sid] = range_cycle_slider.value
            if self._color_style[_sid] == "Range (jet)":
                self._update_point_cloud(_sid)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _update_point_cloud(self, source_id: str) -> None:
        """Re-render point cloud for *source_id* using current GUI state."""
        if not self._enabled:
            return
        if not self._show_pc.get(source_id, True):
            return
        with self.client.atomic():
            if point_cloud := self._point_clouds.pop(source_id, None):
                point_cloud.remove()

            pc_idx = self._pc_sliders[source_id].value
            point_size = self._point_size[source_id]
            visible = self._show_pc[source_id]

            source = self.data_loader.get_point_clouds_source(source_id)
            pc = source.get_pc(pc_idx)

            # Transform point cloud to world coordinates.
            points_world = pc.transform(
                target_frame_id=self.data_loader.world_frame_id,
                target_frame_timestamp_us=pc.reference_frame_timestamp_us,
                pose_graph=self.data_loader.pose_graph,
            ).xyz

            colors = self._colorize_points(source_id, pc, points_world)

            handle_name = f"/point_clouds/{source_id}/point_cloud"
            pc_handle = self.client.scene.add_point_cloud(
                handle_name,
                points=points_world,
                colors=colors,
                point_size=point_size,
                point_shape="circle",
                visible=visible,
            )
            self._point_clouds[source_id] = pc_handle
        self.client.flush()

    # ------------------------------------------------------------------
    # Coloring
    # ------------------------------------------------------------------

    def _colorize_points(
        self,
        source_id: str,
        pc: Any,
        points_world: np.ndarray,
    ) -> np.ndarray:
        """Compute per-point RGB colors based on the active color style.

        Args:
            source_id: Point-clouds source ID.
            pc: The :class:`PointCloud` instance (for attribute access).
            points_world: Point cloud in world coordinates ``[N, 3]``.

        Returns:
            ``uint8`` color array of shape ``[N, 3]``.
        """
        color_style = self._color_style[source_id]
        n_points = points_world.shape[0]

        if color_style == "RGB":
            return self._color_rgb(pc, n_points)
        if color_style == "Height (turbo)":
            return self._color_height_turbo(points_world, source_id)
        if color_style == "Range (jet)":
            return self._color_range_jet(points_world, source_id)

        # Uniform color fallback.
        return np.tile(_DEFAULT_POINT_COLOR, (n_points, 1))

    def _color_rgb(self, pc: Any, n_points: int) -> np.ndarray:
        """Color from the point cloud's ``rgb`` attribute."""
        try:
            rgb = pc.get_attribute("rgb")
            # Ensure uint8 [N, 3].
            if rgb.dtype != np.uint8:
                if np.issubdtype(rgb.dtype, np.floating):
                    rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
                else:
                    rgb = rgb.astype(np.uint8)
            if rgb.ndim == 2 and rgb.shape[1] >= 3:
                return rgb[:, :3]
            return np.tile(_DEFAULT_POINT_COLOR, (n_points, 1))
        except Exception:
            return np.tile(_DEFAULT_POINT_COLOR, (n_points, 1))

    def _color_height_turbo(self, points_world: np.ndarray, source_id: str) -> np.ndarray:
        """Color by world-frame Z height using the turbo colormap."""
        z_min, z_max = self._height_range[source_id]
        z_range = max(z_max - z_min, 0.01)
        z = points_world[:, 2]
        normalized = np.clip((z - z_min) / z_range, 0.0, 1.0)
        rgba = _TURBO_CMAP(normalized)
        return (rgba[:, :3] * 255.0).astype(np.uint8)

    def _color_range_jet(self, points_world: np.ndarray, source_id: str) -> np.ndarray:
        """Color by range from origin using the jet colormap."""
        cycle = max(1.0, self._range_cycle[source_id])
        ranges = np.linalg.norm(points_world, axis=1)
        normalized = (ranges % cycle) / cycle
        rgba = _JET_CMAP(normalized)
        return (rgba[:, :3] * 255.0).astype(np.uint8)
