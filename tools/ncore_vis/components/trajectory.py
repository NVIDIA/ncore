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

"""Rig trajectory visualization component."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import viser

from scipy.spatial.transform import Rotation

from ncore.impl.common.transformations import HalfClosedInterval
from tools.ncore_vis.components.base import VisualizationComponent, register_component
from tools.ncore_vis.utils import se3_to_position_wxyz


# Pre-computed rotation matrices for directional arrow branches.
_T_ROTATE_LEFT: np.ndarray = np.eye(4, dtype=np.float64)
_T_ROTATE_LEFT[:3, :3] = Rotation.from_euler("xyz", [0, 0, 50], degrees=True).as_matrix()
_T_TRANSLATE_LEFT: np.ndarray = np.eye(4, dtype=np.float64)
_T_TRANSLATE_LEFT[1, 3] = 0.1

_T_ROTATE_RIGHT: np.ndarray = np.eye(4, dtype=np.float64)
_T_ROTATE_RIGHT[:3, :3] = Rotation.from_euler("xyz", [0, 0, -50], degrees=True).as_matrix()
_T_TRANSLATE_RIGHT: np.ndarray = np.eye(4, dtype=np.float64)
_T_TRANSLATE_RIGHT[1, 3] = -0.1


@register_component
class TrajectoryComponent(VisualizationComponent):
    """Rig trajectory visualization as colored boxes with directional arrows.

    Renders the rig poses sampled from the pose graph as red rectangles along
    the trajectory path.  Every 10th pose additionally gets arrow-shaped boxes
    indicating the forward direction.
    """

    def create_gui(self, tab_group: viser.GuiTabGroupHandle) -> None:
        # No dedicated tab; GUI is created via create_sequence_gui.
        self._handles: List[viser.SceneNodeHandle] = []
        self._visible: bool = True
        self._rig_frame_handle: Optional[viser.FrameHandle] = None
        self._show_rig_frame: bool = True
        self._world_frame_handle: Optional[viser.FrameHandle] = None
        self._show_world_frame: bool = False

    def create_sequence_gui(self, sequence_tab: viser.GuiTabHandle) -> None:
        checkbox = self.client.gui.add_checkbox(
            "Rig Trajectory", initial_value=True, hint="Show trajectory path as red boxes"
        )

        @checkbox.on_update
        def _(_: viser.GuiEvent) -> None:
            self._visible = checkbox.value
            with self.client.atomic():
                for handle in self._handles:
                    handle.visible = self._visible
            self.client.flush()

        rig_frame_checkbox = self.client.gui.add_checkbox(
            "Show Rig Frame", initial_value=True, hint="Show coordinate frame tripod at current rig pose"
        )

        @rig_frame_checkbox.on_update
        def _(_: viser.GuiEvent) -> None:
            self._show_rig_frame = rig_frame_checkbox.value
            if self._rig_frame_handle is not None:
                self._rig_frame_handle.visible = self._show_rig_frame

        world_frame_checkbox = self.client.gui.add_checkbox(
            "Show World Frame", initial_value=False, hint="Show coordinate frame tripod at world origin"
        )

        @world_frame_checkbox.on_update
        def _(_: viser.GuiEvent) -> None:
            self._show_world_frame = world_frame_checkbox.value
            if self._world_frame_handle is not None:
                self._world_frame_handle.visible = self._show_world_frame

    def populate_scene(self) -> None:
        self._world_frame_handle = self.client.scene.add_frame(
            name="world_origin_frame",
            axes_length=0.3,
            axes_radius=0.015,
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
            visible=self._show_world_frame,
        )

        if self.data_loader.rig_frame_id is None:
            return

        poses = self.data_loader.get_trajectory_poses()
        if poses.shape[0] == 0:
            return

        with self.client.atomic():
            for i in range(poses.shape[0]):
                T = poses[i]
                position, wxyz = se3_to_position_wxyz(T)

                box = self.client.scene.add_box(
                    name=f"rig_trajectory/{i}",
                    color=(1.0, 0.0, 0.0),
                    dimensions=(0.5, 0.1, 0.05),
                    wxyz=wxyz,
                    position=position,
                    visible=self._visible,
                )
                self._handles.append(box)

                # Directional arrows every 10th pose
                if i % 10 == 0:
                    T_left = T @ _T_TRANSLATE_LEFT @ _T_ROTATE_LEFT
                    pos_l, wxyz_l = se3_to_position_wxyz(T_left)
                    arrow_left = self.client.scene.add_box(
                        name=f"rig_trajectory/{i}_left_arrow",
                        color=(1.0, 0.0, 0.0),
                        dimensions=(0.1, 0.35, 0.05),
                        wxyz=wxyz_l,
                        position=pos_l,
                        visible=self._visible,
                    )
                    self._handles.append(arrow_left)

                    T_right = T @ _T_TRANSLATE_RIGHT @ _T_ROTATE_RIGHT
                    pos_r, wxyz_r = se3_to_position_wxyz(T_right)
                    arrow_right = self.client.scene.add_box(
                        name=f"rig_trajectory/{i}_right_arrow",
                        color=(1.0, 0.0, 0.0),
                        dimensions=(0.1, 0.35, 0.05),
                        wxyz=wxyz_r,
                        position=pos_r,
                        visible=self._visible,
                    )
                    self._handles.append(arrow_right)
        self.client.flush()

        # Show rig frame tripod at the initial pose (frame 0 of the first sensor).
        self._update_rig_frame_at_initial_pose()

    def on_reference_frame_change(self, interval_us: HalfClosedInterval) -> None:
        """Update the rig frame tripod to the current reference frame's pose."""
        if self.data_loader.rig_frame_id is None:
            return
        pose = self.data_loader.get_rig_pose_at_timestamp(interval_us.end)
        self._place_rig_frame(pose)

    def _update_rig_frame_at_initial_pose(self) -> None:
        """Create the rig frame tripod at frame 0 so it is visible on load."""
        if self.data_loader.rig_frame_id is None:
            return
        sensor_ids = self.data_loader.camera_ids or self.data_loader.lidar_ids
        if not sensor_ids:
            return
        pose = self.data_loader.get_rig_pose_at_frame(sensor_ids[0], 0)
        self._place_rig_frame(pose)

    def _place_rig_frame(self, pose: Optional[np.ndarray]) -> None:
        """Create or update the rig frame tripod at the given 4x4 pose."""
        if pose is None:
            if self._rig_frame_handle is not None:
                self._rig_frame_handle.visible = False
            return

        position, wxyz = se3_to_position_wxyz(pose)

        if self._rig_frame_handle is None:
            self._rig_frame_handle = self.client.scene.add_frame(
                name="rig_pose_frame",
                axes_length=0.3,
                axes_radius=0.015,
                wxyz=wxyz,
                position=position,
                visible=self._show_rig_frame,
            )
        else:
            self._rig_frame_handle.wxyz = wxyz
            self._rig_frame_handle.position = position
            self._rig_frame_handle.visible = self._show_rig_frame
