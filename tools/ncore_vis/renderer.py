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

"""Per-client renderer that orchestrates visualization components."""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time

from typing import Dict, List, Literal, Tuple, cast

import numpy as np
import viser

from viser import Icon

from ncore.impl.common.transformations import HalfClosedInterval
from tools.ncore_vis.components import VisualizationComponent, get_registered_components
from tools.ncore_vis.data_loader import DataLoader


logger = logging.getLogger(__name__)

# Maximum number of worker threads for parallel sensor updates.
_MAX_WORKERS: int = 8


class NCoreVisRenderer:
    """Per-client rendering state.

    Creates the GUI tab group, instantiates all registered visualization
    components, and coordinates the reference-sensor frame synchronization
    that keeps all sensors in temporal lock-step.
    """

    def __init__(
        self,
        client: viser.ClientHandle,
        data_loader: DataLoader,
    ) -> None:
        self.client: viser.ClientHandle = client
        self.data_loader: DataLoader = data_loader

        # Shared state between renderer and components
        self.sensor_frame_handles: Dict[str, viser.GuiInputHandle[int]] = {}
        self.reference_sensor: str = ""
        self.reference_frame_interval_us: HalfClosedInterval = HalfClosedInterval(0, 1)

        # Deterministic class-color mapping (guarded by lock for thread safety)
        self._class_colors: Dict[str, Tuple[int, int, int]] = {}
        self._class_colors_lock: threading.Lock = threading.Lock()
        self._rng: np.random.RandomState = np.random.RandomState(42)

        # Shared thread pool for parallel sensor updates.
        self._executor: concurrent.futures.ThreadPoolExecutor = concurrent.futures.ThreadPoolExecutor(
            max_workers=_MAX_WORKERS
        )

        # Playback state
        self._playing: bool = False
        self._playback_fps: float = 5.0
        self._loop: bool = True

        self._setup()

    # ------------------------------------------------------------------
    # Public helpers used by components
    # ------------------------------------------------------------------

    def get_class_color(self, class_id: str) -> Tuple[int, int, int]:
        """Return a deterministic RGB color tuple for *class_id*.

        Colors are generated lazily and cached for the lifetime of the renderer.
        Thread-safe: multiple parallel component updates may request colors concurrently.
        """
        with self._class_colors_lock:
            if class_id not in self._class_colors:
                color = self._rng.randint(0, 256, size=3)
                self._class_colors[class_id] = (int(color[0]), int(color[1]), int(color[2]))
            return self._class_colors[class_id]

    def update_all_sensor_frames(self, ref_frame: int) -> None:
        """Synchronize all sensor frame sliders to the reference sensor's *ref_frame*.

        Computes the reference frame interval from the reference sensor and
        triggers ``on_reference_frame_change`` on each component.
        """
        self.reference_frame_interval_us = self.data_loader.get_sensor_frame_interval_us(
            self.reference_sensor, ref_frame
        )
        for component in self._components:
            component.on_reference_frame_change(self.reference_frame_interval_us)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        self.client.gui.set_panel_label("NCore Data Controls")
        self.client.scene.set_background_image(np.full((1000, 1000, 3), 40, dtype=np.uint8))

        tab_group = self.client.gui.add_tab_group()

        # Phase 1: create the Sequence tab FIRST so it appears at the top
        sequence_tab = self._create_sequence_tab(tab_group)

        # Phase 2: instantiate and create per-component GUIs (dedicated tabs)
        self._components: List[VisualizationComponent] = []
        for component_cls in get_registered_components():
            component = component_cls(
                client=self.client,
                data_loader=self.data_loader,
                renderer=self,
            )
            self._components.append(component)

        for component in self._components:
            component.create_gui(tab_group)

        # Collect sensor frame handles from lidar + camera components
        # so that cuboids and recording can cross-reference them.
        for component in self._components:
            self.sensor_frame_handles.update(component.get_frame_sliders())

        # Phase 3: let components add to the Sequence tab
        with sequence_tab:
            for component in self._components:
                component.create_sequence_gui(sequence_tab)

        # Phase 4: populate initial 3D scene (parallel across components)
        futures = [self._executor.submit(c.populate_scene) for c in self._components]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception:
                logger.exception("Error during populate_scene")

    def _create_sequence_tab(self, tab_group: viser.GuiTabGroupHandle) -> viser.GuiTabHandle:
        """Build the Sequence tab with reference sensor controls and playback."""
        # Prefer lidars as default reference sensor (better temporal coverage).
        all_sensor_ids = self.data_loader.lidar_ids + self.data_loader.camera_ids
        if not all_sensor_ids:
            all_sensor_ids = ["(none)"]
        self.reference_sensor = all_sensor_ids[0]

        # Determine initial frame range from first sensor
        max_frame = self._get_sensor_max_frame(self.reference_sensor)

        # Set initial reference frame interval from frame 0.
        self.reference_frame_interval_us = self.data_loader.get_sensor_frame_interval_us(self.reference_sensor, 0)

        sequence_tab = tab_group.add_tab("Sequence")
        with sequence_tab:
            self._ref_frame_slider = self.client.gui.add_slider(
                "Reference Frame",
                min=0,
                max=max_frame,
                initial_value=0,
                step=1,
                hint="0-based frame index of the reference sensor",
            )

            reference_dropdown = self.client.gui.add_dropdown(
                label="Reference Sensor",
                options=all_sensor_ids,
                initial_value=all_sensor_ids[0],
                hint="Sensor whose timeline drives all other sensors",
            )

            self._ref_frame_slider.on_update(self._on_ref_frame_slider_update)

            # -- Playback controls --
            play_button = self.client.gui.add_button(
                "Play", icon=Icon.PLAYER_PLAY, hint="Start/stop automatic frame advance"
            )
            playback_fps = self.client.gui.add_number(
                "Playback FPS", initial_value=5, min=1, max=30, step=1, hint="Frames per second during playback"
            )
            loop_checkbox = self.client.gui.add_checkbox(
                "Loop", initial_value=True, hint="Loop back to frame 0 after the last frame"
            )

            @play_button.on_click
            def _(_: viser.GuiEvent) -> None:
                if self._playing:
                    # Pause
                    self._playing = False
                    play_button.icon = Icon.PLAYER_PLAY
                    play_button.label = "Play"
                else:
                    # Play
                    self._playing = True
                    play_button.icon = Icon.PLAYER_PAUSE
                    play_button.label = "Pause"
                    self._run_playback()

            @playback_fps.on_update
            def _(_: viser.GuiEvent) -> None:
                self._playback_fps = max(1.0, float(playback_fps.value))

            @loop_checkbox.on_update
            def _(_: viser.GuiEvent) -> None:
                self._loop = loop_checkbox.value

            # -- Up direction --
            up_direction_dropdown = self.client.gui.add_dropdown(
                label="Up Direction",
                options=["+z", "-z", "+y", "-y", "+x", "-x"],
                initial_value="+z",
                hint="Global up direction for camera controls",
            )

            @up_direction_dropdown.on_update
            def _(_: viser.GuiEvent) -> None:
                self.client.scene.set_up_direction(
                    cast(Literal["+x", "+y", "+z", "-x", "-y", "-z"], up_direction_dropdown.value)
                )

            # -- Reference sensor change --
            @reference_dropdown.on_update
            def _on_ref_sensor_update(_: viser.GuiEvent) -> None:
                self.reference_sensor = reference_dropdown.value
                new_max = self._get_sensor_max_frame(reference_dropdown.value)

                # Recreate the slider with the new range
                self._ref_frame_slider.remove()
                self._ref_frame_slider = self.client.gui.add_slider(
                    "Reference Frame",
                    min=0,
                    max=new_max,
                    initial_value=0,
                    step=1,
                    hint="0-based frame index of the reference sensor",
                )
                self._ref_frame_slider.on_update(self._on_ref_frame_slider_update)

        return sequence_tab

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def _run_playback(self) -> None:
        """Advance the reference frame at the configured FPS until paused or stopped."""
        while self._playing:
            max_frame = self._get_sensor_max_frame(self.reference_sensor)
            current = self._ref_frame_slider.value
            next_frame = current + 1

            if next_frame > max_frame:
                if self._loop:
                    next_frame = 0
                else:
                    self._playing = False
                    # Reset button to Play state — access via the GUI handle
                    break

            self._ref_frame_slider.value = next_frame
            self.update_all_sensor_frames(next_frame)
            time.sleep(1.0 / max(1.0, self._playback_fps))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _on_ref_frame_slider_update(self, _: viser.GuiEvent) -> None:
        """Handle reference frame slider value changes."""
        self.update_all_sensor_frames(self._ref_frame_slider.value)

    def _get_sensor_max_frame(self, sensor_id: str) -> int:
        """Get the maximum frame index for a sensor."""
        if sensor_id in self.data_loader.camera_ids:
            return max(0, self.data_loader.get_camera_sensor(sensor_id).frames_count - 1)
        if sensor_id in self.data_loader.lidar_ids:
            return max(0, self.data_loader.get_lidar_sensor(sensor_id).frames_count - 1)
        return 0
