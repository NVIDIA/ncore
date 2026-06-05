.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

nuScenes Dataset
================

The NCore nuScenes tool converts data from the
`nuScenes <https://www.nuscenes.org/>`_ autonomous driving dataset into
NCore V4 format. All dataset versions are supported (v1.0-mini,
v1.0-trainval, v1.0-test).

.. _nuscenes_data_conventions:

Conventions
-----------

The nuScenes dataset provides data from 6 cameras, 1 lidar, and 5
radars. The converter handles all sensor modalities and 3D annotations.

Camera Sensors
^^^^^^^^^^^^^^
    1. **Front (camera_front)** -- 1600x900, 70 deg FOV
    2. **Front Left (camera_front_left)** -- 1600x900, 70 deg FOV
    3. **Front Right (camera_front_right)** -- 1600x900, 70 deg FOV
    4. **Back (camera_back)** -- 1600x900, 110 deg FOV
    5. **Back Left (camera_back_left)** -- 1600x900, 70 deg FOV
    6. **Back Right (camera_back_right)** -- 1600x900, 70 deg FOV

All cameras use Basler acA1600-60gc sensors (global shutter). Images are
provided undistorted with zero distortion coefficients. Camera intrinsics
are stored using :class:`~ncore.data.OpenCVPinholeCameraModelParameters`
with ``ShutterType.GLOBAL``.

LiDAR Sensor
^^^^^^^^^^^^^
    1. **Top LiDAR (lidar_top)** -- Velodyne HDL-32E, 32 layers, ~34k points/frame

Point clouds in nuScenes are motion-compensated to the sensor frame at the
sweep reference timestamp. The converter decompensates them back to
per-point-time sensor frames (raw measurements) before storing, since
NCore V4 expects non-motion-compensated ray-bundle data.

Per-point timestamps are derived from the column structure of the .bin
file: each file contains 32-point columns (one per beam) in sequential
firing order. Timestamps use ``column_index / n_model_columns * frame_duration``
(fencepost convention: next frame starts at frame_end, not the last column).

A structured lidar model (``RowOffsetStructuredSpinningLidarModelParameters``)
is stored as intrinsics. Two derivation modes are available:

- **Nominal** (``--lidar-model-source nominal``, recommended): Model parameters
  from the HDL-32E spec -- uniform column azimuths, spec elevation angles,
  analytical firing offsets. No circular data dependency.
- **Empirical** (``--lidar-model-source empirical``): Model derived from a
  decompensated reference frame. Row offsets are blended with analytical values
  for beams that lack far-range observations (steep downward-facing beams).

Resolution upsampling (``--lidar-model-resolution 4``, recommended) interpolates
column azimuths to 4x native resolution (4340 columns). This compensates for
per-revolution azimuth drift in the mechanical spinning and reduces alignment
quantization error from ~0.10 deg to ~0.03 deg.

Optional multi-frame optimization (``--lidar-model-optimization-passes 1``)
adjusts column azimuths and row offsets from median residuals across all frames,
further reducing systematic error.

Model parameters stored:

- ``row_elevations_rad``: per-beam elevation angles
- ``column_azimuths_rad``: per-column azimuth angles (n_columns depends on resolution)
- ``row_azimuth_offsets_rad``: per-beam azimuth offsets from intra-column firing
  sequence (~0.25 deg total range; two 16-beam banks at 1.152 us pair intervals)
- ``spinning_direction``: clockwise ("cw")
- ``spinning_frequency_hz``: derived from inter-sweep timestamps (~20 Hz)

Accuracy (4x nominal + optimization): 0.029 deg far-range angular error,
comparable to PAI-level extraction quality.

The ``model_element`` field is populated with ``[ring_index, column_index]``
per point, where ``column_index`` addresses the (possibly upsampled) model.
Column indices are assigned via iterative per-column alignment with fine-grained
sub-column refinement at the model's resolution.

The minimum distance filter (1.0 m) matches the ``remove_close``
default used by the nuScenes devkit to discard sensor housing
reflections.

Radar Sensors
^^^^^^^^^^^^^
    1. **Front (radar_front)** -- Continental ARS 408
    2. **Front Left (radar_front_left)** -- Continental ARS 408
    3. **Front Right (radar_front_right)** -- Continental ARS 408
    4. **Back Left (radar_back_left)** -- Continental ARS 408
    5. **Back Right (radar_back_right)** -- Continental ARS 408

Radar detections are sparse (typically 10-100 per sweep). Each detection
provides position (x, y, z), ego-motion-compensated velocity, and radar
cross section (RCS). Per-frame generic data fields:

- ``radial_velocity_m_s`` (float32, [N]) -- radial velocity in m/s
  (positive = moving away from sensor), computed by projecting the
  ego-motion-compensated velocity vector onto the detection direction.
- ``rcs_dBsm`` (float32, [N]) -- radar cross section in dBsm.

Radar is not a spinning sensor; all detections in a frame share a single
timestamp.

Ego Poses
^^^^^^^^^
Ego poses are derived from the per-sweep ``ego_pose`` records in the
nuScenes database (GPS/INS-based). Poses are stored as dynamic
``("rig", "world")`` poses relative to the first frame. The absolute
first-frame pose is preserved as a static ``("world", "world_global")``
transform.

3D Annotations
^^^^^^^^^^^^^^
Cuboid annotations are stored in the ``world_global`` coordinate frame
(the nuScenes global map frame) as
:class:`~ncore.data.v4.CuboidsComponent` observations. Only keyframe
annotations are included. The :meth:`~ncore.data.CuboidTrackObservation.transform`
method can re-project them to any sensor frame at runtime via the pose
graph.

Category mapping from nuScenes to NCore class IDs:

- vehicle.car -> car
- vehicle.truck -> truck
- vehicle.bus.* -> bus
- vehicle.construction -> construction_vehicle
- vehicle.motorcycle -> motorcycle
- vehicle.bicycle -> bicycle
- vehicle.trailer -> trailer
- vehicle.emergency.* -> emergency_vehicle
- human.pedestrian.* -> pedestrian
- movable_object.barrier -> barrier
- movable_object.trafficcone -> traffic_cone

Usage
-----

.. code-block:: bash

   bazel run //tools/data_converter/nuscenes -- \
       --root-dir /path/to/nuscenes \
       --output-dir /path/to/output \
       nuscenes-v4 \
       --version v1.0-trainval

Convert a single scene by name:

.. code-block:: bash

   bazel run //tools/data_converter/nuscenes -- \
       --root-dir /path/to/nuscenes \
       --output-dir /path/to/output \
       nuscenes-v4 \
       --version v1.0-mini \
       --scene-name scene-0061

See ``tools/data_converter/nuscenes/README.md`` for full option documentation.
