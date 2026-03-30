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

"""Camera frustum visualization component with optional cuboid and lidar projection overlays."""

from __future__ import annotations

import concurrent.futures
import logging

from typing import Any, Dict, List, Tuple

import cv2
import matplotlib
import numpy as np
import torch
import viser

from scipy.spatial.transform import Rotation as RotLib

from ncore.impl.common.transformations import HalfClosedInterval, transform_point_cloud
from ncore.impl.data.types import FrameTimepoint, LabelSource
from ncore.impl.sensors.camera import CameraModel
from tools.ncore_vis.components.base import VisualizationComponent, register_component
from tools.ncore_vis.utils import se3_to_position_wxyz


logger = logging.getLogger(__name__)

# Default field-of-view (full angle in radians) used for camera frustum display.
_DEFAULT_CAMERA_FOV: float = 1.2
_DEFAULT_CAMERA_SCALE: float = 0.15

# Cuboid edge indices defining the 12 wireframe edges of a box with 8 corners.
_CUBOID_EDGES: List[Tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),  # bottom ring
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),  # top ring
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),  # verticals
]

# Unit cube corners scaled by bounding box dimensions.
_UNIT_CUBE_CORNERS: np.ndarray = np.array(
    [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ],
    dtype=np.float32,
)

# Projection mode choices for lidar overlay.
_PROJECTION_MODES: List[str] = ["rolling-shutter", "mean", "start", "end"]

# Pre-fetch jet colormap for lidar projection coloring.
_JET_CMAP: matplotlib.colors.Colormap = matplotlib.colormaps["jet"]


@register_component
class CameraComponent(VisualizationComponent):
    """Camera frustum visualization with RGB images and optional cuboid / lidar projection overlays.

    For each camera sensor, renders a frustum showing the decoded image, a coordinate
    frame indicating the camera pose, and an optional label.  When cuboid overlay is
    enabled, 3D bounding box edges are projected onto the image using rolling-shutter-
    aware camera models.  When lidar projection is enabled, a point cloud is projected
    onto the camera image with range-based jet-colormap coloring.
    """

    def create_gui(self, tab_group: viser.GuiTabGroupHandle) -> None:  # noqa: C901
        self._frusta: Dict[str, viser.SceneNodeHandle] = {}
        self._poses: Dict[str, viser.FrameHandle] = {}
        self._labels: Dict[str, viser.LabelHandle] = {}

        self._frame_sliders: Dict[str, viser.GuiInputHandle[int]] = {}
        self._visible: Dict[str, bool] = {}
        self._labels_visible: bool = True
        self._enabled: bool = True

        # Shared overlay settings (single values applied to all cameras)
        self._overlay_cuboids: bool = False
        self._cuboid_source: str = LabelSource.AUTOLABEL.name
        self._project_lidar: bool = False
        self._project_lidar_id: str = ""
        self._project_mode: str = "rolling-shutter"
        self._project_point_size: int = 1
        self._project_range_cycle: float = 50.0
        self._project_radar: bool = False
        self._project_radar_id: str = "All"
        self._show_mask: bool = False
        self._mask_name: str = ""
        self._mask_opacity: float = 0.3

        # Camera model device and cached CameraModel instances (one per camera sensor)
        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._camera_models: Dict[str, CameraModel] = {}
        self._build_camera_models()

        # Cache camera aspect ratios and load static masks
        self._aspects: Dict[str, float] = {}
        self._masks: Dict[str, Dict[str, np.ndarray]] = {}
        all_mask_names: List[str] = []
        for camera_id in self.data_loader.camera_ids:
            cam = self.data_loader.get_camera_sensor(camera_id)
            width, height = cam.model_parameters.resolution
            self._aspects[camera_id] = float(width) / float(height)

            mask_images = cam.get_mask_images()
            cam_masks: Dict[str, np.ndarray] = {}
            for name, pil_img in mask_images.items():
                cam_masks[name] = np.asarray(pil_img)
                if name not in all_mask_names:
                    all_mask_names.append(name)
            self._masks[camera_id] = cam_masks

        if all_mask_names:
            self._mask_name = all_mask_names[0]
        self._all_mask_names: List[str] = all_mask_names

        lidar_ids = self.data_loader.lidar_ids
        if lidar_ids:
            self._project_lidar_id = lidar_ids[0]

        radar_ids = self.data_loader.radar_ids

        for camera_id in self.data_loader.camera_ids:
            self._visible[camera_id] = True

        with tab_group.add_tab("Cameras"):
            toggle_cameras = self.client.gui.add_checkbox("Enabled", initial_value=True)
            show_labels = self.client.gui.add_checkbox("Show Labels", initial_value=True)

            # -- Shared overlay settings folder --
            with self.client.gui.add_folder("Overlay Settings", expand_by_default=False):
                # -- Projection device --
                _device_options = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
                device_dropdown = self.client.gui.add_dropdown(
                    "Projection Device",
                    options=_device_options,
                    initial_value=self._device,
                    hint="Torch device used for camera model projections",
                )

                @device_dropdown.on_update
                def _(_: viser.GuiEvent) -> None:
                    self._device = device_dropdown.value
                    self._build_camera_models()
                    self._refresh_all_cameras()

                # -- Cuboid overlay --
                overlay_cuboids_checkbox = self.client.gui.add_checkbox(
                    "Overlay Cuboids", initial_value=False, hint="Project 3D cuboid edges onto all cameras"
                )
                source_dropdown = self.client.gui.add_dropdown(
                    "Cuboid Source",
                    options=[s.name for s in LabelSource],
                    initial_value=self._cuboid_source,
                    hint="Label source for cuboid overlays on all cameras",
                )
                self._bind_overlay_settings(
                    overlay_cuboids_checkbox=overlay_cuboids_checkbox,
                    source_dropdown=source_dropdown,
                )

                # -- Lidar projection --
                if lidar_ids:
                    project_lidar_checkbox = self.client.gui.add_checkbox(
                        "Project Lidar", initial_value=False, hint="Project lidar points onto all cameras"
                    )
                    proj_lidar_dropdown = self.client.gui.add_dropdown(
                        "Lidar Sensor",
                        options=lidar_ids,
                        initial_value=self._project_lidar_id,
                        hint="Lidar sensor to project onto camera images",
                    )
                    proj_mode_dropdown = self.client.gui.add_dropdown(
                        "Projection Mode",
                        options=_PROJECTION_MODES,
                        initial_value="rolling-shutter",
                        hint="Camera pose interpolation for lidar projection",
                    )
                    proj_point_size_slider = self.client.gui.add_slider(
                        "Point Size", min=1, max=10, step=1, initial_value=1, hint="Pixel radius of projected points"
                    )
                    proj_range_cycle_slider = self.client.gui.add_slider(
                        "Range Cycle (m)",
                        min=5.0,
                        max=200.0,
                        step=1.0,
                        initial_value=50.0,
                        hint="Range in meters before jet colormap wraps",
                    )
                    self._bind_lidar_projection_settings(
                        project_lidar_checkbox=project_lidar_checkbox,
                        proj_lidar_dropdown=proj_lidar_dropdown,
                        proj_mode_dropdown=proj_mode_dropdown,
                        proj_point_size_slider=proj_point_size_slider,
                        proj_range_cycle_slider=proj_range_cycle_slider,
                    )

                # -- Radar projection --
                if radar_ids:
                    project_radar_checkbox = self.client.gui.add_checkbox(
                        "Project Radar", initial_value=False, hint="Project radar points onto all cameras"
                    )
                    proj_radar_dropdown = self.client.gui.add_dropdown(
                        "Radar Sensor",
                        options=["All"] + radar_ids,
                        initial_value="All",
                        hint="Radar sensor to project (or 'All' for every radar)",
                    )
                    self._bind_radar_projection_settings(
                        project_radar_checkbox=project_radar_checkbox,
                        proj_radar_dropdown=proj_radar_dropdown,
                    )

                # -- Mask overlay --
                if all_mask_names:
                    show_mask_checkbox = self.client.gui.add_checkbox(
                        "Show Mask", initial_value=False, hint="Overlay static camera mask on all cameras"
                    )
                    mask_dropdown = self.client.gui.add_dropdown(
                        "Mask Name",
                        options=all_mask_names,
                        initial_value=self._mask_name,
                        hint="Static camera mask to overlay",
                    )
                    mask_opacity_slider = self.client.gui.add_slider(
                        "Mask Opacity",
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        initial_value=0.3,
                        hint="Transparency of the mask color tint",
                    )
                    self._bind_mask_settings(
                        show_mask_checkbox=show_mask_checkbox,
                        mask_dropdown=mask_dropdown,
                        mask_opacity_slider=mask_opacity_slider,
                    )

            # -- Per-camera folders --
            for camera_id in self.data_loader.camera_ids:
                cam = self.data_loader.get_camera_sensor(camera_id)
                frame_count = cam.frames_count

                with self.client.gui.add_folder(camera_id):
                    slider = self.client.gui.add_slider(
                        "Frame", min=0, max=max(0, frame_count - 1), step=1, initial_value=0
                    )
                    self._frame_sliders[camera_id] = slider

                    show_checkbox = self.client.gui.add_checkbox("Show Camera", initial_value=True)
                    go_to_frame = self.client.gui.add_button("Go to Frame")

                    # Wire up per-camera callbacks
                    self._bind_camera_callbacks(camera_id, slider, show_checkbox, go_to_frame)

            @show_labels.on_update
            def _(_: viser.GuiEvent) -> None:
                self._labels_visible = show_labels.value
                for cid, label in self._labels.items():
                    label.visible = show_labels.value
                    if cid in self._poses:
                        label.wxyz = self._poses[cid].wxyz

            @toggle_cameras.on_update
            def _(_: viser.GuiEvent) -> None:
                self._enabled = toggle_cameras.value
                for cid in self._frusta:
                    self._set_camera_visibility(cid, toggle_cameras.value)

    def get_frame_sliders(self) -> Dict[str, viser.GuiInputHandle[int]]:
        return dict(self._frame_sliders)

    def populate_scene(self) -> None:
        futures = [self.renderer._executor.submit(self._update_camera, cid) for cid in self.data_loader.camera_ids]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception:
                logger.exception("Error populating camera scene")

    def on_reference_frame_change(self, interval_us: HalfClosedInterval) -> None:
        if not self._enabled:
            return
        center_us = interval_us.start + (interval_us.end - interval_us.start) // 2
        for camera_id in self.data_loader.camera_ids:
            if not self._visible.get(camera_id, True):
                continue
            sensor = self.data_loader.get_camera_sensor(camera_id)
            self._frame_sliders[camera_id].value = sensor.get_closest_frame_index(center_us, relative_frame_time=0.5)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bind_camera_callbacks(
        self,
        camera_id: str,
        slider: viser.GuiInputHandle[int],
        show_checkbox: viser.GuiInputHandle[bool],
        go_to_frame: viser.GuiButtonHandle,
    ) -> None:
        @slider.on_update
        def _(_: viser.GuiEvent, _cid: str = camera_id) -> None:
            self._update_camera(_cid)

        @show_checkbox.on_update
        def _(_: viser.GuiEvent, _cid: str = camera_id) -> None:
            was_hidden = not self._visible.get(_cid, True)
            self._set_camera_visibility(_cid, show_checkbox.value)
            if show_checkbox.value and was_hidden:
                self._update_camera(_cid)

        @go_to_frame.on_click
        def _(_: viser.GuiEvent, _cid: str = camera_id) -> None:
            if _cid in self._poses:
                self.client.camera.wxyz = self._poses[_cid].wxyz
                self.client.camera.position = self._poses[_cid].position

    def _bind_overlay_settings(
        self,
        overlay_cuboids_checkbox: viser.GuiInputHandle[bool],
        source_dropdown: viser.GuiInputHandle[str],
    ) -> None:
        """Wire up cuboid overlay shared-setting callbacks."""

        @overlay_cuboids_checkbox.on_update
        def _(_: viser.GuiEvent) -> None:
            self._overlay_cuboids = overlay_cuboids_checkbox.value
            self._refresh_all_cameras()

        @source_dropdown.on_update
        def _(_: viser.GuiEvent) -> None:  # type: ignore[no-redef]
            self._cuboid_source = source_dropdown.value
            self._refresh_all_cameras()

    def _bind_lidar_projection_settings(
        self,
        project_lidar_checkbox: viser.GuiInputHandle[bool],
        proj_lidar_dropdown: viser.GuiInputHandle[str],
        proj_mode_dropdown: viser.GuiInputHandle[str],
        proj_point_size_slider: viser.GuiInputHandle[int],
        proj_range_cycle_slider: viser.GuiInputHandle[float],
    ) -> None:
        """Wire up lidar projection shared-setting callbacks."""

        @project_lidar_checkbox.on_update
        def _(_: viser.GuiEvent) -> None:
            self._project_lidar = project_lidar_checkbox.value
            self._refresh_all_cameras()

        @proj_lidar_dropdown.on_update
        def _(_: viser.GuiEvent) -> None:  # type: ignore[no-redef]
            self._project_lidar_id = proj_lidar_dropdown.value
            self._refresh_all_cameras()

        @proj_mode_dropdown.on_update
        def _(_: viser.GuiEvent) -> None:  # type: ignore[no-redef]
            self._project_mode = proj_mode_dropdown.value
            self._refresh_all_cameras()

        @proj_point_size_slider.on_update
        def _(_: viser.GuiEvent) -> None:  # type: ignore[no-redef]
            self._project_point_size = proj_point_size_slider.value
            self._refresh_all_cameras()

        @proj_range_cycle_slider.on_update
        def _(_: viser.GuiEvent) -> None:  # type: ignore[no-redef]
            self._project_range_cycle = proj_range_cycle_slider.value
            self._refresh_all_cameras()

    def _bind_radar_projection_settings(
        self,
        project_radar_checkbox: viser.GuiInputHandle[bool],
        proj_radar_dropdown: viser.GuiInputHandle[str],
    ) -> None:
        """Wire up radar projection shared-setting callbacks."""

        @project_radar_checkbox.on_update
        def _(_: viser.GuiEvent) -> None:
            self._project_radar = project_radar_checkbox.value
            self._refresh_all_cameras()

        @proj_radar_dropdown.on_update
        def _(_: viser.GuiEvent) -> None:  # type: ignore[no-redef]
            self._project_radar_id = proj_radar_dropdown.value
            self._refresh_all_cameras()

    def _bind_mask_settings(
        self,
        show_mask_checkbox: viser.GuiInputHandle[bool],
        mask_dropdown: viser.GuiInputHandle[str],
        mask_opacity_slider: viser.GuiInputHandle[float],
    ) -> None:
        """Wire up mask overlay shared-setting callbacks."""

        @show_mask_checkbox.on_update
        def _(_: viser.GuiEvent) -> None:
            self._show_mask = show_mask_checkbox.value
            self._refresh_all_cameras()

        @mask_dropdown.on_update
        def _(_: viser.GuiEvent) -> None:  # type: ignore[no-redef]
            self._mask_name = mask_dropdown.value
            self._refresh_all_cameras()

        @mask_opacity_slider.on_update
        def _(_: viser.GuiEvent) -> None:  # type: ignore[no-redef]
            self._mask_opacity = mask_opacity_slider.value
            self._refresh_all_cameras()

    def _build_camera_models(self) -> None:
        """Build (or rebuild) the per-camera :class:`CameraModel` cache using ``self._device``."""
        self._camera_models = {
            camera_id: CameraModel.from_parameters(
                self.data_loader.get_camera_sensor(camera_id).model_parameters,
                device=self._device,
                dtype=torch.float32,
            )
            for camera_id in self.data_loader.camera_ids
        }

    def _refresh_all_cameras(self) -> None:
        """Re-render all cameras in parallel (used when a shared overlay setting changes)."""
        if not self._enabled:
            return
        visible_ids = [cid for cid in self.data_loader.camera_ids if self._visible.get(cid, True)]
        futures = [self.renderer._executor.submit(self._update_camera, cid) for cid in visible_ids]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception:
                logger.exception("Error refreshing camera")

    def _set_camera_visibility(self, camera_id: str, visible: bool, update_labels: bool = True) -> None:
        self._visible[camera_id] = visible
        if camera_id in self._frusta:
            self._frusta[camera_id].visible = visible
        if camera_id in self._poses:
            self._poses[camera_id].visible = visible
        if update_labels and camera_id in self._labels:
            self._labels[camera_id].visible = visible and self._labels_visible

    def _update_camera(self, camera_id: str) -> None:
        """Re-render camera frustum, pose frame, and label for *camera_id*."""
        if not self._visible.get(camera_id, True):
            return
        with self.client.atomic():
            if frustum := self._frusta.pop(camera_id, None):
                frustum.remove()
            if pose := self._poses.pop(camera_id, None):
                pose.remove()
            if label := self._labels.pop(camera_id, None):
                label.remove()

            frame_idx = self._frame_sliders[camera_id].value
            visible = self._visible[camera_id]

            cam = self.data_loader.get_camera_sensor(camera_id)
            T_camera_world = cam.get_frames_T_sensor_target(
                self.data_loader.world_frame_id, frame_idx, FrameTimepoint.END
            )
            position, wxyz = se3_to_position_wxyz(T_camera_world)

            # Pose frame
            pose_handle = self.client.scene.add_frame(
                f"/cameras/{camera_id}/pose",
                wxyz=wxyz,
                position=position,
                axes_length=0.003,
                axes_radius=0.0005,
                visible=visible,
            )
            self._poses[camera_id] = pose_handle

            # Image (with optional overlays)
            image = cam.get_frame_image_array(frame_idx)

            if self._project_lidar:
                try:
                    image = self._overlay_lidar_projection(camera_id, frame_idx, image)
                except Exception:
                    logger.debug("Lidar projection overlay failed for %s frame %d", camera_id, frame_idx, exc_info=True)

            if self._project_radar:
                try:
                    image = self._overlay_radar_projection(camera_id, frame_idx, image)
                except Exception:
                    logger.debug("Radar projection overlay failed for %s frame %d", camera_id, frame_idx, exc_info=True)

            if self._overlay_cuboids:
                try:
                    image = self._overlay_cuboids_on_image(camera_id, frame_idx, image)
                except Exception:
                    logger.debug("Cuboid overlay failed for %s frame %d", camera_id, frame_idx, exc_info=True)

            if self._show_mask:
                try:
                    image = self._overlay_mask(camera_id, image)
                except Exception:
                    logger.debug("Mask overlay failed for %s frame %d", camera_id, frame_idx, exc_info=True)

            frustum_handle = self.client.scene.add_camera_frustum(
                f"/cameras/{camera_id}/pose/frustum",
                fov=_DEFAULT_CAMERA_FOV,
                aspect=self._aspects[camera_id],
                scale=_DEFAULT_CAMERA_SCALE,
                image=image,
                visible=visible,
            )
            self._frusta[camera_id] = frustum_handle

            # Click-to-fly
            @frustum_handle.on_click
            def _(_event: Any, _pose: viser.FrameHandle = pose_handle) -> None:
                self.client.camera.wxyz = _pose.wxyz
                self.client.camera.position = _pose.position

            # Label
            label_handle = self.client.scene.add_label(
                f"/labels/{camera_id}",
                camera_id,
                wxyz=wxyz,
                position=position,
                visible=visible and self._labels_visible,
            )
            self._labels[camera_id] = label_handle
        self.client.flush()

    # ------------------------------------------------------------------
    # Mask overlay
    # ------------------------------------------------------------------

    def _overlay_mask(self, camera_id: str, image: np.ndarray) -> np.ndarray:
        """Blend a semi-transparent color tint over masked regions of the image.

        Args:
            camera_id: Camera sensor ID.
            image: RGB image array (H, W, 3), uint8.

        Returns:
            Copy of the image with mask tint applied.
        """
        mask_name = self._mask_name
        cam_masks = self._masks.get(camera_id, {})
        if mask_name not in cam_masks:
            return image

        mask_raw = cam_masks[mask_name]
        # Convert mask to boolean: handle both 2D and 3D masks
        if mask_raw.ndim == 3:
            mask_bool = mask_raw[:, :, 0] > 0
        else:
            mask_bool = mask_raw > 0

        # Resize mask if dimensions don't match image
        h, w = image.shape[:2]
        mh, mw = mask_bool.shape[:2]
        if mh != h or mw != w:
            mask_resized = cv2.resize(mask_bool.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        else:
            mask_resized = mask_bool

        # Apply semi-transparent magenta tint over masked regions
        output = image.copy()
        tint_color = np.array([255, 0, 255], dtype=np.uint8)
        alpha = self._mask_opacity
        output[mask_resized] = (
            (1.0 - alpha) * output[mask_resized].astype(np.float32) + alpha * tint_color.astype(np.float32)
        ).astype(np.uint8)

        return output

    # ------------------------------------------------------------------
    # Lidar projection overlay
    # ------------------------------------------------------------------

    def _overlay_lidar_projection(self, camera_id: str, frame_idx: int, image: np.ndarray) -> np.ndarray:
        """Project a lidar point cloud onto the camera image with range-based coloring.

        Args:
            camera_id: Camera sensor to project onto.
            frame_idx: Camera frame index.
            image: RGB image array (H, W, 3), uint8.

        Returns:
            Copy of the image with projected lidar points drawn.
        """
        lidar_id = self._project_lidar_id
        if not lidar_id:
            return image

        cam = self.data_loader.get_camera_sensor(camera_id)
        lidar_sensor = self.data_loader.get_lidar_sensor(lidar_id)
        camera_model = self._camera_models[camera_id]

        # Find closest lidar frame to the camera frame (by center-of-frame timestamp)
        cam_interval = self.data_loader.get_sensor_frame_interval_us(camera_id, frame_idx)
        cam_center_us = cam_interval.start + (cam_interval.end - cam_interval.start) // 2
        lidar_frame_idx = lidar_sensor.get_closest_frame_index(cam_center_us, relative_frame_time=0.5)

        # Load point cloud and transform to world coordinates
        pc_sensor = lidar_sensor.get_frame_point_cloud(
            lidar_frame_idx, motion_compensation=True, with_start_points=False
        )
        world_id = self.data_loader.world_frame_id
        T_lidar_world = lidar_sensor.get_frames_T_sensor_target(world_id, lidar_frame_idx, FrameTimepoint.END)
        pc_world = transform_point_cloud(pc_sensor.xyz_m_end, T_lidar_world)

        # Get camera world-to-sensor transforms (T_world_camera)
        T_world_camera_start = cam.get_frames_T_source_sensor(world_id, frame_idx, FrameTimepoint.START)
        T_world_camera_end = cam.get_frames_T_source_sensor(world_id, frame_idx, FrameTimepoint.END)

        # Project world points to image coordinates
        mode = self._project_mode
        projection = self._project_points(camera_model, pc_world, T_world_camera_start, T_world_camera_end, mode)

        if projection.valid_indices is None or projection.image_points.shape[0] == 0:
            return image

        image_coords = projection.image_points.cpu().numpy()
        valid_idx = projection.valid_indices.cpu().numpy()

        # Compute per-point range (distance from camera origin in camera frame)
        if projection.T_world_sensors is not None:
            T_w2c = projection.T_world_sensors.cpu().numpy()
            transformed = transform_point_cloud(pc_world[valid_idx, None, :], T_w2c).squeeze(1)
            ranges = np.linalg.norm(transformed, axis=1)
        else:
            # Fallback: use distance from lidar sensor
            ranges = np.linalg.norm(pc_sensor.xyz_m_end[valid_idx], axis=1)

        # Draw range-colored points on image
        output_image = image.copy()
        cycle = max(1.0, self._project_range_cycle)
        point_radius = max(1, self._project_point_size)

        # Compute colors for all points at once
        normalized = (ranges % cycle) / cycle
        rgba = _JET_CMAP(normalized)  # [N, 4]
        colors_bgr = (rgba[:, [2, 1, 0]] * 255.0).astype(np.uint8)

        for i in range(image_coords.shape[0]):
            px = int(round(image_coords[i, 0]))
            py = int(round(image_coords[i, 1]))
            color = (int(colors_bgr[i, 0]), int(colors_bgr[i, 1]), int(colors_bgr[i, 2]))
            cv2.circle(output_image, (px, py), point_radius, color, thickness=-1, lineType=cv2.LINE_AA)

        return output_image

    # ------------------------------------------------------------------
    # Radar projection overlay
    # ------------------------------------------------------------------

    def _overlay_radar_projection(self, camera_id: str, frame_idx: int, image: np.ndarray) -> np.ndarray:
        """Project radar point clouds onto the camera image with range-based coloring.

        When ``self._project_radar_id`` is ``"All"``, every radar sensor is projected.

        Args:
            camera_id: Camera sensor to project onto.
            frame_idx: Camera frame index.
            image: RGB image array (H, W, 3), uint8.

        Returns:
            Copy of the image with projected radar points drawn.
        """
        if self._project_radar_id == "All":
            radar_ids = self.data_loader.radar_ids
        else:
            radar_ids = [self._project_radar_id]

        output_image = image.copy()
        for radar_id in radar_ids:
            output_image = self._project_single_radar(camera_id, frame_idx, radar_id, output_image)
        return output_image

    def _project_single_radar(self, camera_id: str, frame_idx: int, radar_id: str, image: np.ndarray) -> np.ndarray:
        """Project a single radar sensor's point cloud onto *image* (mutates in-place)."""
        cam = self.data_loader.get_camera_sensor(camera_id)
        radar_sensor = self.data_loader.get_radar_sensor(radar_id)
        camera_model = self._camera_models[camera_id]

        # Find closest radar frame to the camera frame (by center-of-frame timestamp)
        cam_interval = self.data_loader.get_sensor_frame_interval_us(camera_id, frame_idx)
        cam_center_us = cam_interval.start + (cam_interval.end - cam_interval.start) // 2
        radar_frame_idx = radar_sensor.get_closest_frame_index(cam_center_us, relative_frame_time=0.5)

        # Load point cloud and transform to world coordinates
        pc_sensor = radar_sensor.get_frame_point_cloud(
            radar_frame_idx, motion_compensation=True, with_start_points=False
        )
        world_id = self.data_loader.world_frame_id
        T_radar_world = radar_sensor.get_frames_T_sensor_target(world_id, radar_frame_idx, FrameTimepoint.END)
        pc_world = transform_point_cloud(pc_sensor.xyz_m_end, T_radar_world)

        # Get camera world-to-sensor transforms (T_world_camera)
        T_world_camera_start = cam.get_frames_T_source_sensor(world_id, frame_idx, FrameTimepoint.START)
        T_world_camera_end = cam.get_frames_T_source_sensor(world_id, frame_idx, FrameTimepoint.END)

        # Project world points to image coordinates
        mode = self._project_mode
        projection = self._project_points(camera_model, pc_world, T_world_camera_start, T_world_camera_end, mode)

        if projection.valid_indices is None or projection.image_points.shape[0] == 0:
            return image

        image_coords = projection.image_points.cpu().numpy()
        valid_idx = projection.valid_indices.cpu().numpy()

        # Compute per-point range (distance from camera origin in camera frame)
        if projection.T_world_sensors is not None:
            T_w2c = projection.T_world_sensors.cpu().numpy()
            transformed = transform_point_cloud(pc_world[valid_idx, None, :], T_w2c).squeeze(1)
            ranges = np.linalg.norm(transformed, axis=1)
        else:
            ranges = np.linalg.norm(pc_sensor.xyz_m_end[valid_idx], axis=1)

        # Draw range-colored points (slightly larger than lidar to distinguish)
        cycle = max(1.0, self._project_range_cycle)
        point_radius = max(2, self._project_point_size + 1)

        normalized = (ranges % cycle) / cycle
        rgba = _JET_CMAP(normalized)
        colors_bgr = (rgba[:, [2, 1, 0]] * 255.0).astype(np.uint8)

        for i in range(image_coords.shape[0]):
            px = int(round(image_coords[i, 0]))
            py = int(round(image_coords[i, 1]))
            color = (int(colors_bgr[i, 0]), int(colors_bgr[i, 1]), int(colors_bgr[i, 2]))
            cv2.circle(image, (px, py), point_radius, color, thickness=-1, lineType=cv2.LINE_AA)

        return image

    def _project_points(
        self,
        camera_model: CameraModel,
        pc_world: np.ndarray,
        T_world_camera_start: np.ndarray,
        T_world_camera_end: np.ndarray,
        mode: str,
        return_all_projections: bool = False,
    ) -> CameraModel.WorldPointsToImagePointsReturn:
        """Project world points using the specified projection mode."""
        if mode == "rolling-shutter":
            return camera_model.world_points_to_image_points_shutter_pose(
                pc_world,
                T_world_camera_start,
                T_world_camera_end,
                return_valid_indices=True,
                return_T_world_sensors=True,
                return_all_projections=return_all_projections,
            )
        if mode == "mean":
            return camera_model.world_points_to_image_points_mean_pose(
                pc_world,
                T_world_camera_start,
                T_world_camera_end,
                return_valid_indices=True,
                return_T_world_sensors=True,
                return_all_projections=return_all_projections,
            )
        if mode == "start":
            return camera_model.world_points_to_image_points_static_pose(
                pc_world,
                T_world_camera_start,
                return_valid_indices=True,
                return_T_world_sensors=True,
                return_all_projections=return_all_projections,
            )
        # "end"
        return camera_model.world_points_to_image_points_static_pose(
            pc_world,
            T_world_camera_end,
            return_valid_indices=True,
            return_T_world_sensors=True,
            return_all_projections=return_all_projections,
        )

    # ------------------------------------------------------------------
    # Cuboid overlay projection
    # ------------------------------------------------------------------

    def _overlay_cuboids_on_image(self, camera_id: str, frame_idx: int, image: np.ndarray) -> np.ndarray:
        """Project 3D cuboid edges onto the camera image, interpolated to the mid-of-frame time.

        Each cuboid track is interpolated to the camera frame's mid-of-frame timestamp so
        that the projected box position reflects the object's estimated location at the
        moment the camera was actually exposing.  The interpolated observation is then
        transformed to world coordinates and projected using the shared projection mode
        (rolling-shutter / mean / start / end).

        Args:
            camera_id: Camera sensor to project onto.
            frame_idx: Current frame index for this camera.
            image: RGB image array (H, W, 3), uint8.

        Returns:
            Copy of the image with visible cuboid edges drawn.
        """
        cam = self.data_loader.get_camera_sensor(camera_id)
        camera_model = self._camera_models[camera_id]

        world_id = self.data_loader.world_frame_id
        pose_graph = self.data_loader.pose_graph

        output_image = image.copy()
        image_height, image_width = output_image.shape[:2]
        image_rect = (0, 0, image_width, image_height)

        # Approximate the track / camera association with mid-frame interpolation
        timestamp_start_us = cam.get_frame_timestamp_us(frame_idx, FrameTimepoint.START)
        timestamp_end_us = cam.get_frame_timestamp_us(frame_idx, FrameTimepoint.END)
        mid_timestamp_us = (timestamp_start_us + timestamp_end_us) // 2

        # Use the reference-time range as the clamp boundary so tracks are
        # currently selected remain visible at the scene boundary
        ref_interval = self.renderer.reference_frame_interval_us
        max_clamp_us = ref_interval.stop - ref_interval.start

        # Camera poses at start/end of frame for rolling-shutter-aware projection
        T_world_camera_start = cam.get_frames_T_source_sensor(world_id, frame_idx, FrameTimepoint.START)
        T_world_camera_end = cam.get_frames_T_source_sensor(world_id, frame_idx, FrameTimepoint.END)

        # Iterate over all tracks; interpolate each to the mid-frame time.
        for track in self.data_loader.get_cuboid_tracks():
            # Filter by label source
            if track.source.name != self._cuboid_source:
                continue

            if (obs := track.interpolate_at(mid_timestamp_us, max_clamp_us=max_clamp_us)) is None:
                continue

            # Transform the interpolated observation at mid-of-frame time to world coordinates at mid-frame time
            obs = obs.transform(
                target_frame_id=world_id,
                target_frame_timestamp_us=mid_timestamp_us,
                pose_graph=pose_graph,
            )
            bbox = obs.bbox3

            # Compute 8 corners in world coordinates
            dimensions = np.array(bbox.dim, dtype=np.float32)
            corners_local = _UNIT_CUBE_CORNERS * dimensions
            rotation = RotLib.from_euler("XYZ", bbox.rot).as_matrix().astype(np.float32)
            translation = np.array(bbox.centroid, dtype=np.float32)
            corners_world = (corners_local @ rotation.T + translation).astype(np.float32)

            # Project using the shared projection mode (rolling-shutter / mean / start / end)
            projection = self._project_points(
                camera_model,
                corners_world,
                T_world_camera_start,
                T_world_camera_end,
                self._project_mode,
                return_all_projections=True,
            )

            if projection.valid_indices is None or projection.image_points.shape[0] == 0:
                continue

            projected_pts = projection.image_points.cpu().numpy()
            valid_mask = np.zeros(projected_pts.shape[0], dtype=bool)
            valid_mask[projection.valid_indices.cpu().numpy()] = True

            # Deterministic color per class
            line_color = self.renderer.get_class_color(obs.class_id)

            # Draw the 12 edges of the cuboid if either corner is valid (visible);
            # use OpenCV's clipLine to handle partially visible edges
            for corner_a, corner_b in _CUBOID_EDGES:
                if not (valid_mask[corner_a] or valid_mask[corner_b]):
                    continue
                p1 = (int(round(projected_pts[corner_a, 0])), int(round(projected_pts[corner_a, 1])))
                p2 = (int(round(projected_pts[corner_b, 0])), int(round(projected_pts[corner_b, 1])))
                ok, clipped_p1, clipped_p2 = cv2.clipLine(image_rect, p1, p2)
                if ok:
                    cv2.line(output_image, clipped_p1, clipped_p2, color=line_color, thickness=2)

        return output_image
