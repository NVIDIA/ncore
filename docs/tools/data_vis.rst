.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Visualization
=============

Data stored in NCore-specific dataformats can be visualized using different
tools provided by the project

Rolling-Shutter Point-Cloud to Camera Projections
-------------------------------------------------

The tool ``//tools:ncore_project_pc_to_img`` visualizes projections of
point-clouds into camera images, applying sensor-specific rolling-shutter
compensation. This verifies the extrinsics of the point-cloud sensor, the
extrinsics of the cameras, the intrinsics of the cameras, as well as the
trajectories of the rig.

Example invocation::

    bazel run //tools:ncore_project_pc_to_img \
        -- \
        --sensor-id=lidar00 \
        --camera-id=camera01 \
        v4 \
        --component-group=<SEQUENCE_META.json>

Or with multiple component groups::

    bazel run //tools:ncore_project_pc_to_img \
        -- \
        --sensor-id=lidar00 \
        --camera-id=camera01 \
        v4 \
        --component-group=<COMPONENT_GROUP0> \
        --component-group=<COMPONENT_GROUP1>


.. figure:: proj0.png
   :figwidth: 50%
   :width: 80%

   Point-to-camera projection on NV Hyperion data

.. figure:: proj1.png
   :figwidth: 50%
   :width: 80%

   Point-to-camera projection on Waymo-Open data


Frame-Exporting
---------------
The tool ``//tools:ncore_export_ply`` exports point-clouds into common
``.ply`` format, transforming points into different frames. Specifying
``--frame=world`` allows to visualize multiple frames in a common frame to
verify the extrinsics of the point-cloud sensor, as well as the trajectories of
the rig.

Example invocation::

    bazel run //tools:ncore_export_ply \
        -- \
        --output-dir=<OUTPUT_FOLDER> \
        --sensor-id=lidar00 \
        --frame=world \
        v4 \
        --component-group=<COMPONENT_GROUP0> \
        --component-group=<COMPONENT_GROUP1>

.. figure:: pc.png
   :figwidth: 50%
   :width: 80%

   Differently colored point clouds exported to a common world frame

---------------

Likewise, the tool ``//tools:ncore_export_image`` allows exporting specific
camera-frame ranges into image files for introspection.

Example invocation::

    bazel run //tools:ncore_export_image \
        -- \
        --output-dir=<OUTPUT_FOLDER> \
        --camera-id=camera00 \
        v4 \
        --component-group=<SEQUENCE_META.json>

Or with multiple component groups::

    bazel run //tools:ncore_export_image \
        -- \
        --output-dir=<OUTPUT_FOLDER> \
        --camera-id=camera00 \
        v4 \
        --component-group=<COMPONENT_GROUP0> \
        --component-group=<COMPONENT_GROUP1>
