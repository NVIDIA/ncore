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

"""Visualization components for the NCore interactive viewer.

To add a custom component:

1. Create a new module in this package.
2. Subclass :class:`VisualizationComponent` and decorate with :func:`register_component`.
3. Import the module below to trigger registration.
4. Add the module to ``BUILD.bazel`` ``srcs``.

Registration order determines GUI tab ordering.
"""

# Import built-in components to trigger registration (order = tab order).
from tools.ncore_vis.components import (
    camera,  # noqa: F401
    cuboids,  # noqa: F401
    lidar,  # noqa: F401
    point_clouds,  # noqa: F401
    radar,  # noqa: F401
    trajectory,  # noqa: F401
)
from tools.ncore_vis.components.base import VisualizationComponent, get_registered_components, register_component


__all__ = [
    "VisualizationComponent",
    "get_registered_components",
    "register_component",
]
