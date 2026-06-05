..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Lidar Model Evaluation Tool
============================

The ``ncore_evaluate_lidar_model`` tool measures the quality of a structured lidar
model by comparing model-predicted ray directions against stored native directions.
It works with any NCore V4 sequence that has a structured lidar model (nuScenes,
Waymo, PAI, or custom datasets).

Metrics
-------

The tool reports:

- **Angular error** (degrees): arccos of the dot product between native and model
  direction vectors. Reported as mean, median, p95, and max across all valid points
  and separately for far-range points (>20m).
- **Systematic azimuth shift** (degrees): mean signed azimuth difference for
  far-range points. Indicates whether the model's column grid is consistently
  rotated relative to reality.
- **Pixel-equivalent error**: angular error converted to pixel units using the
  camera's focal length (when ``--camera-id`` is specified).
- **Per-row breakdown**: mean error per beam row, revealing systematic issues
  with specific beams.
- **Per-frame breakdown**: mean error per frame, revealing alignment instability.

Quality Reference
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - Dataset
     - Mean (far)
     - Pixel equiv
     - Notes
   * - Waymo (ground truth)
     - 0.010 deg
     - 0.35 px
     - Theoretical floor
   * - PAI (extraction)
     - 0.029 deg
     - ~0.5 px
     - Factory-quality extraction
   * - nuScenes (4x nominal)
     - 0.029 deg
     - 0.7 px
     - Nominal model + 4x resolution
   * - nuScenes (1x empirical)
     - 0.083 deg
     - 1.8 px
     - Empirical model, native resolution

Image Output
------------

When ``--camera-id`` and ``--output-dir`` are specified, the tool produces two
images per frame:

- ``{frame}_overlay.png``: Native points (cyan) and model points (red) projected
  onto the camera image. Where the model is accurate, colors overlap. Where it
  diverges, you see the separation direction and magnitude.
- ``{frame}_error.png``: Points colored by angular error magnitude using the
  turbo colormap (blue = low error, red = high error at ``--max-error-deg``).

Usage
-----

.. code-block:: bash

   bazel run //tools:ncore_evaluate_lidar_model -- \
       --source-id lidar_top \
       --camera-id camera_front \
       --output-dir /tmp/eval_output \
       --step-frame 10 \
       --point-size 2 \
       --device cpu \
       v4 --component-group /path/to/scene.json

CLI Options
-----------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Flag
     - Default
     - Description
   * - ``--source-id``
     - (required)
     - Lidar sensor ID to evaluate
   * - ``--camera-id``
     - None
     - Camera for pixel metrics and image output
   * - ``--output-dir``
     - None
     - Output directory for images (requires ``--camera-id``)
   * - ``--start-frame``
     - None
     - First lidar frame to evaluate
   * - ``--stop-frame``
     - None
     - Past-the-end frame
   * - ``--step-frame``
     - None
     - Frame step for subsampling
   * - ``--far-range-m``
     - 20.0
     - Distance threshold for far-range metrics
   * - ``--point-size``
     - 1.5
     - Point radius for rendered images
   * - ``--max-error-deg``
     - 1.0
     - Error colormap ceiling (degrees)
   * - ``--device``
     - cpu
     - Torch device (cpu or cuda)
   * - ``--pose``
     - rolling-shutter
     - Pose mode for projection (rolling-shutter, mean, start, end)

Examples
--------

Evaluate nuScenes with nominal 4x model:

.. code-block:: bash

   # Convert first
   bazel run //tools/data_converter/nuscenes -- \
       --root-dir /path/to/nuscenes --output-dir /tmp/ns_out \
       nuscenes-v4 --version v1.0-mini --scene-name scene-0061 \
       --store-type directory \
       --lidar-model-source nominal \
       --lidar-model-resolution 4 \
       --lidar-model-optimization-passes 1

   # Evaluate
   bazel run //tools:ncore_evaluate_lidar_model -- \
       --source-id lidar_top --camera-id camera_front \
       --output-dir /tmp/eval --step-frame 10 --device cpu \
       v4 --component-group /tmp/ns_out/scene-0061/scene-0061.json

Evaluate Waymo (baseline -- ground truth model from range image metadata):

.. code-block:: bash

   bazel run //tools:ncore_evaluate_lidar_model -- \
       --source-id lidar_top --camera-id camera_front_50fov \
       --output-dir /tmp/waymo_eval --step-frame 10 --device cpu \
       v4 --component-group /path/to/waymo_scene.json
