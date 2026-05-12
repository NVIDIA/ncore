.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

KITTI Dataset
=============

The NCore KITTI tool converts data from the
`KITTI Vision Benchmark Suite <https://www.cvlibs.net/datasets/kitti/>`_
raw data format (synced+rectified) into NCore V4 format.

.. _kitti_data_conventions:

Conventions
-----------

The KITTI raw dataset provides data from 6 sensors:

Camera Sensors
^^^^^^^^^^^^^^
    1. **Left Grayscale (camera_gray_left)** -- Point Grey Flea 2, rectified
    2. **Right Grayscale (camera_gray_right)** -- Point Grey Flea 2, rectified
    3. **Left Color (camera_color_left)** -- Point Grey Flea 2, rectified
    4. **Right Color (camera_color_right)** -- Point Grey Flea 2, rectified

All cameras use CCD sensors (global shutter) and images are provided
rectified with zero distortion. The camera intrinsics are stored using
:class:`~ncore.data.OpenCVPinholeCameraModelParameters` with distortion
coefficients set to zero.

LiDAR Sensor
^^^^^^^^^^^^^
    1. **Top LiDAR (lidar_top)** -- Velodyne HDL-64E, 64 layers, ~100k points/frame

Point clouds are stored as unstructured ray-bundle data (no structured
spinning lidar model, no intrinsic sensor model). The original KITTI binary
format provides only raw ``(x, y, z, reflectance)`` per point without
row/column structure or per-beam calibration, so ``model_element`` is not
set. Approximate per-point timestamps are reconstructed from azimuth angles
using the known spin timing of the Velodyne HDL-64E (10 Hz,
counter-clockwise rotation).

GPS/IMU
^^^^^^^
The OXTS RT 3003 GPS/INS provides 30-field measurements at 10 Hz. Ego poses
are computed via Mercator projection (first frame as origin) and stored as
dynamic ``("rig", "world")`` poses. Raw OXTS measurements are preserved as
component-level generic data on the poses component.

3D Annotations
^^^^^^^^^^^^^^
Tracklet labels (``tracklet_labels.xml``) are parsed and stored as
:class:`~ncore.data.v4.CuboidsComponent` observations in the velodyne
coordinate frame. The viewer transforms them to world coordinates at
runtime via the pose graph.

Usage
-----

.. code-block:: bash

   bazel run //tools/data_converter/kitti -- \
       --root-dir /path/to/2011_09_26 \
       --output-dir /path/to/output \
       kitti-v4

See ``tools/data_converter/kitti/README.md`` for full option documentation.
