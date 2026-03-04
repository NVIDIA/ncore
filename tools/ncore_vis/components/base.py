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

"""Base class and decorator-based registry for visualization components.

To create a new visualization component:
    1. Subclass :class:`VisualizationComponent`.
    2. Decorate the subclass with :func:`register_component`.
    3. Import the module in ``components/__init__.py`` to trigger registration.

Components are instantiated once per connected client by the renderer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Type

import viser

from ncore.impl.common.transformations import HalfClosedInterval


if TYPE_CHECKING:
    from tools.ncore_vis.data_loader import DataLoader
    from tools.ncore_vis.renderer import NCoreVisRenderer  # type: ignore[import-not-found]

# ---------------------------------------------------------------------------
# Component registry
# ---------------------------------------------------------------------------

_COMPONENT_REGISTRY: List[Type[VisualizationComponent]] = []


def register_component(cls: Type[VisualizationComponent]) -> Type[VisualizationComponent]:
    """Class decorator that registers a visualization component.

    Registered components are automatically instantiated for every connected
    client.  Registration order determines GUI tab ordering.
    """
    _COMPONENT_REGISTRY.append(cls)
    return cls


def get_registered_components() -> List[Type[VisualizationComponent]]:
    """Return all registered component classes in registration order."""
    return list(_COMPONENT_REGISTRY)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class VisualizationComponent(ABC):
    """Base class for pluggable visualization components.

    Each component owns a slice of the GUI (typically a tab) and the
    corresponding 3D scene elements.  The lifecycle is:

    1. ``__init__``          -- store references, no side-effects.
    2. ``create_gui``        -- build GUI elements inside the tab group.
    3. ``create_sequence_gui``  -- (optional) add controls to the shared Sequence tab.
    4. ``populate_scene``    -- render initial 3D objects.
    5. ``on_reference_frame_change`` -- called when the global frame slider moves.
    """

    def __init__(
        self,
        client: viser.ClientHandle,
        data_loader: DataLoader,
        renderer: NCoreVisRenderer,
    ) -> None:
        self.client: viser.ClientHandle = client
        self.data_loader: DataLoader = data_loader
        self.renderer: NCoreVisRenderer = renderer

    @abstractmethod
    def create_gui(self, tab_group: viser.GuiTabGroupHandle) -> None:
        """Create component-specific GUI elements (tabs, sliders, etc.)."""
        ...

    def create_sequence_gui(self, sequence_tab: viser.GuiTabHandle) -> None:
        """Add controls to the shared *Sequence* tab (optional).

        Override this when the component needs a small presence in the Sequence tab
        rather than its own dedicated tab.
        """

    @abstractmethod
    def populate_scene(self) -> None:
        """Render initial 3D scene objects after all GUIs have been created."""
        ...

    def get_frame_sliders(self) -> Dict[str, viser.GuiInputHandle[int]]:
        """Return per-sensor frame slider handles owned by this component.

        Components that manage frame sliders (e.g. camera, lidar) should override
        this method so the renderer can collect all sensor frame handles without
        resorting to dynamic attribute access.
        """
        return {}

    def on_reference_frame_change(self, interval_us: HalfClosedInterval) -> None:
        """React to a change in the global reference frame.

        Args:
            interval_us: Half-closed ``[start, stop)`` timestamp interval of the
                reference sensor's current frame.  Components can derive the
                center timestamp for closest-frame lookups or the inclusive end
                timestamp (``interval_us.end``) for pose evaluation.

        Override to synchronize per-sensor frame sliders or scene elements.
        """
