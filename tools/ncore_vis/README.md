<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NCore Interactive 3D Viewer

Browser-based interactive 3D viewer for NCore V4 sequence data, built on
[viser](https://github.com/nerfstudio-project/viser).

## Quick Start

```bash
bazel run //tools/ncore_vis -- v4 --component-group=<SEQUENCE_META.json>
```

Then open the URL printed in the terminal (default `http://0.0.0.0:8080`).

## Features

- **Scene tab with playback**: Play/Pause button to auto-advance frames at
  configurable FPS with optional looping; all controls have hover tooltips
- **Camera frustums** with decoded RGB images and shared overlay settings
  for cuboid projection, lidar projection, and mask overlays
- **Camera mask overlay**: semi-transparent color tint on static per-sensor
  masks with configurable opacity
- **Lidar point clouds** with 8 coloring modes (intensity, gamma, range jet,
  height turbo, timestamp, model row/column), multi-frame fusing, and motion
  compensation toggle
- **3D cuboid bounding boxes** as wireframe meshes with per-class coloring
  and text labels
- **Rig trajectory** visualization as directional boxes with a live
  coordinate frame tripod at the current rig pose
- **Configurable frame IDs**: `--rig-frame-id` and `--world-frame-id` CLI
  options (defaults: `rig`, `world`)
- **Video recording** from the client camera viewpoint (MP4 via OpenCV)

## Architecture

The viewer uses a **component-based plugin architecture**. Each visualization
feature is implemented as a `VisualizationComponent` subclass registered with
the `@register_component` decorator. Components are automatically instantiated
for every connected client and own their own GUI tabs and 3D scene elements.

## Extending the Viewer

To add a new visualization:

1. Create a new Python module in `tools/ncore_vis/components/`.
2. Subclass `VisualizationComponent` from `components.base` and decorate with
   `@register_component`.
3. Implement `create_gui()` and `populate_scene()`.
4. Import the module in `components/__init__.py` to trigger registration.
5. Add the file to `BUILD.bazel` under `pylib_components` srcs.

See the full documentation for details:

```bash
bazel run //docs:view_ncore
```
