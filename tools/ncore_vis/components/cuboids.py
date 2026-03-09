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

"""3D cuboid bounding box visualization component queried by reference timestamp."""

from __future__ import annotations

from typing import List

import viser

from ncore.impl.common.transformations import HalfClosedInterval
from ncore.impl.data.types import LabelSource
from tools.ncore_vis.components.base import VisualizationComponent, register_component
from tools.ncore_vis.utils import se3_from_centroid_euler, se3_to_position_wxyz


@register_component
class CuboidsComponent(VisualizationComponent):
    """3D wireframe cuboid bounding box visualization driven by the scene reference timestamp.

    Renders cuboid bounding boxes as wireframe meshes in world coordinates.
    Observations are queried at the current reference timestamp and transformed
    to world coordinates via the pose graph, independent of any specific lidar
    sensor's frame index.
    """

    def create_gui(self, tab_group: viser.GuiTabGroupHandle) -> None:
        self._cuboid_boxes: List[viser.BoxHandle] = []
        self._cuboid_labels: List[viser.LabelHandle] = []

        self._cuboid_source: str = LabelSource.AUTOLABEL.name
        self._enabled: bool = True
        self._show_labels: bool = False

        default_source = LabelSource.AUTOLABEL.name

        with tab_group.add_tab("Cuboids"):
            cuboid_checkbox = self.client.gui.add_checkbox(
                "Enabled", initial_value=True, hint="Enable cuboid bounding box visualization"
            )
            source_dropdown = self.client.gui.add_dropdown(
                "Cuboid Source",
                options=[s.name for s in LabelSource],
                initial_value=default_source,
                hint="Label source for cuboid observations",
            )
            label_checkbox = self.client.gui.add_checkbox(
                "Show Labels", initial_value=False, hint="Show track/class text labels on cuboids"
            )

            @source_dropdown.on_update
            def _(_: viser.GuiEvent) -> None:
                self._cuboid_source = source_dropdown.value
                self._update_cuboids()

            @cuboid_checkbox.on_update
            def _(_: viser.GuiEvent) -> None:  # type: ignore[no-redef]
                self._enabled = cuboid_checkbox.value
                for box in self._cuboid_boxes:
                    box.visible = cuboid_checkbox.value
                for label in self._cuboid_labels:
                    label.visible = cuboid_checkbox.value

            @label_checkbox.on_update
            def _(_: viser.GuiEvent) -> None:  # type: ignore[no-redef]
                self._show_labels = label_checkbox.value
                self._update_cuboids()

    def populate_scene(self) -> None:
        self._update_cuboids()

    def on_reference_frame_change(self, interval_us: HalfClosedInterval) -> None:
        if not self._enabled:
            return
        self._update_cuboids()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_cuboids(self) -> None:
        """Re-render cuboid bounding boxes at the current reference timestamp."""
        with self.client.atomic():
            # Remove existing scene nodes
            old_boxes, self._cuboid_boxes = self._cuboid_boxes, []
            old_labels, self._cuboid_labels = self._cuboid_labels, []
            
            for box in old_boxes:
                box.remove()
            for label in old_labels:
                label.remove()

            if not self._enabled:
                self.client.flush()
                return

            interval_us = self.renderer.reference_frame_interval_us
            source = self._cuboid_source
            visible = self._enabled
            show_labels = self._show_labels

            observations = self.data_loader.get_cuboid_observations_in_world(interval_us, source_filter=source)

            # Observations are already in world coordinates.
            for i, obs in enumerate(observations):
                bbox = obs.bbox3
                T_bbox_world = se3_from_centroid_euler(bbox.centroid, bbox.rot)
                position, wxyz = se3_to_position_wxyz(T_bbox_world)
                color = self.renderer.get_class_color(obs.class_id)

                box_handle = self.client.scene.add_box(
                    name=f"/cuboids/cuboid_{i}",
                    dimensions=(bbox.dim[0], bbox.dim[1], bbox.dim[2]),
                    color=color,
                    position=position,
                    wxyz=wxyz,
                    wireframe=True,
                    visible=visible,
                )
                self._cuboid_boxes.append(box_handle)

                if show_labels:
                    node_name = f"{obs.track_id}[{obs.class_id}]"
                    label_position = (position[0], position[1], position[2] + bbox.dim[2] / 2.0)
                    label_handle = self.client.scene.add_label(
                        f"/cuboid_labels/label_{i}",
                        node_name,
                        position=label_position,
                        visible=visible,
                    )
                    self._cuboid_labels.append(label_handle)
        self.client.flush()
