.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Visualization & Export
======================

Data stored in NCore-specific dataformats can be visualized and exported using
the tools described below.

Rolling-Shutter Point-Cloud to Camera Projections
--------------------------------------------------

The tool ``//tools:ncore_project_pc_to_img`` visualizes projections of
point-clouds into camera images, applying sensor-specific rolling-shutter
compensation. This verifies the extrinsics of the point-cloud sensor, the
extrinsics of the cameras, the intrinsics of the cameras, as well as the
trajectories of the rig.

Example invocation::

    bazel run //tools:ncore_project_pc_to_img \
        -- \
        --source-id=lidar00 \
        --camera-id=camera01 \
        v4 \
        --component-group=<SEQUENCE_META.json>

Or with multiple component groups::

    bazel run //tools:ncore_project_pc_to_img \
        -- \
        --source-id=lidar00 \
        --camera-id=camera01 \
        v4 \
        --component-group=<COMPONENT_GROUP0> \
        --component-group=<COMPONENT_GROUP1>

The ``--source-id`` flag accepts any point cloud source: a native point cloud
ID (e.g. ``sfm_points``), a lidar ID, or a radar ID.


.. figure:: proj0.png
   :figwidth: 80%
   :width: 100%

   Point-to-camera projection on NV Hyperion data

.. figure:: proj1.png
   :figwidth: 80%
   :width: 100%

   Point-to-camera projection on Waymo-Open data

.. figure:: proj2.gif
   :figwidth: 80%
   :width: 100%

   Point-to-camera projection on Physical-AI-AV data


Point-Cloud Export
------------------

The tool ``//tools:ncore_export_ply`` exports point-clouds into common
``.ply`` format, transforming points into different frames. Specifying
``--frame=world`` allows to visualize multiple frames in a common frame to
verify the extrinsics of the point-cloud sensor, as well as the trajectories of
the rig.

Example invocation::

    bazel run //tools:ncore_export_ply \
        -- \
        --output-dir=<OUTPUT_FOLDER> \
        --source-id=lidar00 \
        --frame=world \
        v4 \
        --component-group=<COMPONENT_GROUP0> \
        --component-group=<COMPONENT_GROUP1>

The ``--source-id`` flag accepts any point cloud source (native, lidar, or
radar).

.. figure:: pc.png
   :figwidth: 80%
   :width: 100%

   Differently colored point clouds exported to a common world frame


Colored Point-Cloud Export
--------------------------

The tool ``//tools:ncore_export_colored_pc`` projects point clouds onto a
camera image to obtain per-point RGB colors, then exports the result as ``.ply``
files. This combines rolling-shutter-aware projection with PLY export to produce
colored point clouds useful for visual inspection and downstream processing.

If the point cloud source already has an RGB attribute (e.g. COLMAP SfM points),
the ``--use-source-rgb`` flag skips camera projection and uses the native colors.

Example invocation::

    bazel run //tools:ncore_export_colored_pc \
        -- \
        --output-dir=<OUTPUT_FOLDER> \
        --source-id=lidar00 \
        --camera-id=camera01 \
        v4 \
        --component-group=<SEQUENCE_META.json>

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Option
     - Default
     - Description
   * - ``--output-dir``
     - (required)
     - Directory for output PLY files
   * - ``--source-id``
     - (first available)
     - Point cloud source to export (native, lidar, or radar ID)
   * - ``--camera-id``
     - ``camera_front_wide_120fov``
     - Camera sensor used for coloring
   * - ``--device``
     - ``cuda``
     - Torch device (``cuda`` or ``cpu``)
   * - ``--camera-pose``
     - ``rolling-shutter``
     - Projection pose mode (``rolling-shutter``, ``mean``, ``start``, ``end``)
   * - ``--point-cloud-space``
     - ``world``
     - Output coordinate space (``world`` or ``sensor``)
   * - ``--lidar-return-index``
     - ``0``
     - Lidar ray bundle return index
   * - ``--output-filepattern``
     - ``frame-index``
     - Filename pattern (``frame-index`` or ``timestamps-us``)
   * - ``--start-pc``
     - all
     - First pc index to export
   * - ``--stop-pc``
     - all
     - Past-the-end pc index
   * - ``--step-pc``
     - 1
     - Step for downsampling point clouds


Camera Frame Export
-------------------

The tool ``//tools:ncore_export_camera`` exports camera frames to image files
for introspection, optionally encoding them as MP4 video.

Example invocation::

    bazel run //tools:ncore_export_camera \
        -- \
        --output-dir=<OUTPUT_FOLDER> \
        --camera-id=camera00 \
        v4 \
        --component-group=<SEQUENCE_META.json>

Or with multiple component groups::

    bazel run //tools:ncore_export_camera \
        -- \
        --output-dir=<OUTPUT_FOLDER> \
        --camera-id=camera00 \
        v4 \
        --component-group=<COMPONENT_GROUP0> \
        --component-group=<COMPONENT_GROUP1>
