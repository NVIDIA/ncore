.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Argoverse 2 Dataset
===================

The NCore Argoverse 2 tool converts data from the
`Argoverse 2 <https://www.argoverse.org/av2.html>`_ Sensor Dataset into NCore
V4 format. The converter reads the Argoverse 2 on-disk Apache Feather files
directly with ``pyarrow`` and deliberately avoids the heavy ``av2`` devkit
(which pulls in torch, kornia, numba, polars and PyAV). Quaternion handling uses
``scipy`` (already an ncore dependency), so no extra dependency is introduced.

.. _argoverse2_data_conventions:

Conventions
-----------

Argoverse 2 provides data from 9 cameras and 2 lidars; it has no radar. The
converter handles all sensor modalities and 3D cuboid annotations.

Camera Sensors
^^^^^^^^^^^^^^
    1. **ring_front_center** -- 2048x1550 (portrait)
    2. **ring_front_left** -- 1550x2048
    3. **ring_front_right** -- 1550x2048
    4. **ring_side_left** -- 1550x2048
    5. **ring_side_right** -- 1550x2048
    6. **ring_rear_left** -- 1550x2048
    7. **ring_rear_right** -- 1550x2048
    8. **stereo_front_left** -- 1550x2048
    9. **stereo_front_right** -- 1550x2048

The released imagery for all nine cameras is already undistorted -- the official
av2 devkit projects with the intrinsic matrix ``K`` only and does not load the
distortion columns -- so camera intrinsics are stored using
:class:`~ncore.data.IdealPinholeCameraModelParameters`. Because the imagery is
already undistorted, **global shutter is assumed** (``ShutterType.GLOBAL``). The
``k1, k2, k3`` coefficients present in ``intrinsics.feather`` describe the
original lens (for re-distorting into the raw frame) and are intentionally not
applied to the released images; they are preserved per camera in the camera
component ``generic_meta_data`` under ``av2_original_distortion`` so the original
calibration is not lost.

LiDAR Sensors
^^^^^^^^^^^^^
    1. **up_lidar** -- Velodyne VLP-32C, 32 beams, 10 Hz
    2. **down_lidar** -- Velodyne VLP-32C, 32 beams, 10 Hz

Argoverse 2 sweeps are egomotion-compensated to the sweep reference timestamp
and provided in the egovehicle frame, with real per-point timestamps
(``offset_ns``). The two stacked VLP-32C units are stored separately, each with
its own static extrinsic. Points are split per unit by ``laser_number``,
mapped into the unit's own sensor frame, and decompensated using the real
per-point timestamps so that NCore stores raw per-point-time ray directions.
Because the sensor extrinsic is static, this decompensation is independent of
whether the source data applied ego-motion before or after the sensor
transform.

A structured VLP-32C model is stored per unit as lidar intrinsics, with per-point
``model_element`` (row, column). Argoverse 2 provides no native firing-column
index, so the firing pattern is reconstructed from ``offset_ns`` (firing columns --
one VLP-32C revolution at 10 Hz) and ``laser_number`` (the beam, mapped to an
elevation-sorted row). The geometry is derived per log from the *decompensated*
reference sweep: elevations, the laser-to-row map, column timing, per-column
azimuths, and per-row azimuth offsets (the 32 beams of a firing column span several
degrees of azimuth, so the per-row offset is fit empirically). The two stacked
units fire in opposite phase, so they spin oppositely in their own frames (one
``cw``, one ``ccw``), which is detected from the data. The column grid is upsampled
4x so per-frame alignment is not column-quantized, and each sweep is re-aligned to
the model by a per-frame affine column remap -- a constant phase (the spin phase at
a given ``offset_ns`` drifts ~1 deg between sweeps) plus a linear term (the spin
rate drifts slightly within a sweep on some scenes). Steep downward beams that only
return at near range (no far data) have their azimuth offset fit from near-range
returns. Deriving from the decompensated cloud (not the ego-motion-smeared
compensated one) plus these steps gives ~0.03 deg median far-range reconstruction
across scenes (validated on 38 val logs / 76 units, all sub-0.08 deg median with no
systematic azimuth or elevation bias), on par with native-column sensors. Pass
``--lidar-model-source none`` to store raw ray bundles only.

The ``laser_number`` to up/down unit split is not documented by Argoverse 2. The
two units occupy the two laser-number halves (``< 32`` and ``>= 32``); the unit
*label* is recovered from extrinsic geometry by per-beam elevation flatness -- a
laser ring traces a constant-elevation cone only in its own sensor frame, so the
wrong extrinsic tilts the cone and inflates the per-ring elevation spread. The
decision is made once per log and is stable with a wide (~2-10x) margin.

Annotations
^^^^^^^^^^^

3D cuboid annotations are native to the egovehicle frame at the sweep reference
time. They are stored in the ``rig`` frame at that timestamp with no ego pose
baked in, so the egovehicle motion stays out of the stored coordinates and
remains swappable downstream (a V4 feature); the pose graph places the cuboids
using the active ego trajectory. The full 3-DOF box orientation is preserved (the
AV2 quaternion is converted to the ``BBox3`` ``xyz``-Euler convention, not reduced
to yaw). The ``track_uuid`` is used as the track ID.

Coordinate Frames
^^^^^^^^^^^^^^^^^

The first ego pose's ``city_SE3_egovehicle`` is stored as the static
``world -> world_global`` pose, so ``world_global`` is the Argoverse 2 city
frame. All absolute city coordinates remain recoverable for later alignment
with the Argoverse 2 HD map (which the converter does not export).

Usage
-----

.. code-block:: bash

    bazel run //tools/data_converter/argoverse2 -- \
        --root-dir /path/to/argoverse2/sensor \
        --output-dir /path/to/output \
        argoverse2-v4 \
        --split val

Convert a single log:

.. code-block:: bash

    bazel run //tools/data_converter/argoverse2 -- \
        --root-dir /path/to/argoverse2/sensor \
        --output-dir /path/to/output \
        argoverse2-v4 \
        --split val \
        --log-id 02678d04-cc9f-3148-9f95-1ba66347dff9

Testing
-------

.. code-block:: bash

    AV2_DIR=/path/to/argoverse2/sensor AV2_SPLIT=val \
        bazel test //tools/data_converter/argoverse2:pytest_converter
