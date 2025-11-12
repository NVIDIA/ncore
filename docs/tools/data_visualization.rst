.. Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

Visualization
=============

Data stored in NCore-specific dataformats can be visualized using different tools provided by the SDK

Rolling-Shutter Point-Cloud to Camera Projections
-------------------------------------------------

The tool ``//scripts:ncore_project_pc_to_img`` visualizes projections of point-clouds into camera images, applying sensor-specific rolling-shutter compensation.
This verifies the extrinsics of the point-cloud sensor, the extrinsics of the cameras, the intrinsics of the cameras, as well as the trajectories of the rig.

Example invocation::

    bazel run //scripts:ncore_project_pc_to_img \
        -- \
        v3 \
        --shard-file-pattern=<SHARD_FILE_PATTERN> \
        --sensor-id=lidar00 \
        --camera-id=camera01

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
The tool ``//scripts:ncore_visualize_labels`` visualize 3D point-cloud properties (like timestamps / per-object dynamic flags) as well as label cuboid bounds,
enabling label verification relative to the point-cloud sensor.

Example invocation::

    bazel run //scripts:ncore_visualize_labels \
        -- \
        v3 \
        --shard-file-pattern=<SHARD_FILE_PATTERN>

.. figure:: pc0.png
   :figwidth: 50%
   :width: 80%

   Color-coded per-point timestamps and 3D cuboid labels

.. figure:: pc1.png
   :figwidth: 50%
   :width: 80%

   Color-coded per-point dynamic-object flags
   (red indicating dynamic points)

Frame-Exporting
---------------
The tool ``//scripts:ncore_export_ply`` exports point-clouds into common ``.ply`` format, transforming points into different frames.
Specifying ``--frame=world`` allows to visualize multiple frames in a common frame to verify the extrinsics of the point-cloud sensor, as well as the trajectories of the rig.

Example invocation::

    bazel run //scripts:ncore_export_ply \
        -- \
        --shard-file-pattern=<SHARD_FILE_PATTERN> \
        --output-dir=<OUTPUT_FOLDER> \
        --sensor-id=lidar00 \
        --frame=world

.. figure:: pc.png
   :figwidth: 50%
   :width: 80%

   Differently colored point clouds exported to a common world frame

---------------

Likewise, the tool ``//scripts:ncore_export_image`` allows exporting specific camera-frame ranges into image files for introspection.

Example invocation::

    bazel run //scripts:ncore_export_image \
        -- \
        --shard-file-pattern=<SHARD_FILE_PATTERN> \
        --output-dir=<OUTPUT_FOLDER> \
        --camera-id=camera00
