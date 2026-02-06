.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION
.. SPDX-License-Identifier: Apache-2.0

Visualization
=============

Data stored in NCore-specific dataformats can be visualized using different
tools provided by the project

Rolling-Shutter Point-Cloud to Camera Projections
-------------------------------------------------

The tool ``//scripts:ncore_project_pc_to_img`` visualizes projections of
point-clouds into camera images, applying sensor-specific rolling-shutter
compensation. This verifies the extrinsics of the point-cloud sensor, the
extrinsics of the cameras, the intrinsics of the cameras, as well as the
trajectories of the rig.

Example invocation::

    bazel run //scripts:ncore_project_pc_to_img \
        -- \
        --sensor-id=lidar00 \
        --camera-id=camera01 \
        v4 \
        --component-group=<SEQUENCE_META.json>

Or with multiple component groups::

    bazel run //scripts:ncore_project_pc_to_img \
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


Point-Cloud and Label Visualization
-----------------------------------
The tool ``//scripts:ncore_visualize_labels`` visualize 3D point-cloud
properties (like timestamps / per-object dynamic flags) as well as label cuboid
bounds, enabling label verification relative to the point-cloud sensor.

Example invocation::

    bazel run //scripts:ncore_visualize_labels \
        -- \
        v4 \
        --component-group=<SEQUENCE_META.json>

.. figure:: pc0.png
   :figwidth: 50%
   :width: 80%

   Color-coded per-point timestamps and 3D cuboid labels

.. figure:: pc1.png
   :figwidth: 50%
   :width: 80%

   Color-coded per-point dynamic-object flags (optional, red indicating dynamic points)

Frame-Exporting
---------------
The tool ``//scripts:ncore_export_ply`` exports point-clouds into common
``.ply`` format, transforming points into different frames. Specifying
``--frame=world`` allows to visualize multiple frames in a common frame to
verify the extrinsics of the point-cloud sensor, as well as the trajectories of
the rig.

Example invocation::

    bazel run //scripts:ncore_export_ply \
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

Likewise, the tool ``//scripts:ncore_export_image`` allows exporting specific
camera-frame ranges into image files for introspection.

Example invocation::

    bazel run //scripts:ncore_export_image \
        -- \
        --output-dir=<OUTPUT_FOLDER> \
        --camera-id=camera00 \
        v4 \
        --component-group=<SEQUENCE_META.json>

Or with multiple component groups::

    bazel run //scripts:ncore_export_image \
        -- \
        --output-dir=<OUTPUT_FOLDER> \
        --camera-id=camera00 \
        v4 \
        --component-group=<COMPONENT_GROUP0> \
        --component-group=<COMPONENT_GROUP1>
