.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Waymo-Open Dataset
==================

The NCore Waymo tool converts data from the
`Waymo-Open <https://waymo.com/open/>`_ format (**.tfrecords**) into NCore V4
format.

.. _waymo_data_conventions:

Conventions
-----------

Waymo-Open's data convention consists of 6 sensors:

Camera Sensors
^^^^^^^^^^^^^^
    1. **Front Camera 50deg FOV (camera_front_50fov)**
    2. **Front Left Camera 50deg FOV (camera_front_left_50fov)**
    3. **Front Right Camera 50deg FOV (camera_front_right_50fov)**
    4. **Side Left Camera 50deg FOV (camera_side_left_50fov)**
    5. **Side Right Camera 50deg FOV (camera_side_right_50fov)**

The camera intrinsics are compatible with the
:class:`~ncore.data.OpenCVPinholeCameraModelParameters` model.

The Waymo camera frame convention is:
    - Principal axis along local camera's +x axis
    - Local y-axis points left
    - Local z-axis points up

**Note:** The converter transforms this to NCore's camera convention (principal
axis +z, x-axis right, y-axis down).

Each camera provides panoptic segmentation data with 29 semantic classes.

LiDAR Sensors
^^^^^^^^^^^^^
    1. **Top Lidar (lidar_top)**

LiDAR data includes point clouds with multi-return support (primary and
secondary returns) and 3D bounding box labels for 5 classes: unknown, vehicle,
pedestrian, sign, cyclist.

For more information, see the `Waymo Open Dataset paper
<https://openaccess.thecvf.com/content_CVPR_2020/papers/Sun_Scalability_in_Perception_for_Autonomous_Driving_Waymo_Open_Dataset_CVPR_2020_paper.pdf>`_.


.. _waymo_data_conversions:

Conversion
----------

The converter uses NCore V4's component-based architecture. Each sequence is
parsed from ``.tfrecord`` files and written to NCore format via
:class:`~ncore.data.v4.SequenceComponentGroupsWriter` with specialized component
writers for poses, intrinsics, lidar, cameras, masks, and 3D labels.

Usage
^^^^^

Run the converter with Bazel from the repository root:

.. code-block:: bash

    bazel run //tools/data_converter/waymo:convert -- \
        --root-dir <PATH_TO_TFRECORDS> \
        --output-dir <PATH_TO_OUTPUT> \
        waymo-v4

To convert only a time slice of each sequence, use ``--seek-sec`` and/or
``--duration-sec``:

.. code-block:: bash

    # Convert 10 seconds starting 5 seconds into each sequence
    bazel run //tools/data_converter/waymo:convert -- \
        --root-dir <PATH_TO_TFRECORDS> \
        --output-dir <PATH_TO_OUTPUT> \
        waymo-v4 --seek-sec 5.0 --duration-sec 10.0

**Base arguments** (required):

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Argument
     - Description
   * - ``--root-dir PATH``
     - Path to the directory containing ``.tfrecord`` sequence files
   * - ``--output-dir PATH``
     - Path where converted NCore V4 sequences will be written

**Base arguments** (optional):

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Argument
     - Description
   * - ``--no-cameras``
     - Disable exporting all camera sensors
   * - ``--camera-id ID``
     - Export only the specified camera (repeatable; defaults to all cameras)
   * - ``--no-lidars``
     - Disable exporting all lidar sensors
   * - ``--lidar-id ID``
     - Export only the specified lidar (repeatable; defaults to all lidars)
   * - ``--verbose``
     - Enable debug-level logging

**Subcommand arguments** (``waymo-v4``):

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Argument
     - Default
     - Description
   * - ``--store-type {itar,directory}``
     - ``itar``
     - Output store format. ``itar`` produces an indexed tar archive;
       ``directory`` writes plain zarr directories
   * - ``--profile {default,separate-sensors,separate-all}``
     - ``default``
     - Component group layout. ``default`` groups all sensors together;
       ``separate-sensors`` gives each sensor its own group; ``separate-all``
       splits every component type into its own group
   * - ``--sequence-meta`` / ``--no-sequence-meta``
     - enabled
     - Whether to write a JSON metadata file alongside each converted sequence
   * - ``--world-global-mode {none,identity,localized}``
     - ``none``
     - Controls whether a ``("world", "world_global")`` static pose is stored.
       ``none`` omits it (default); ``identity`` stores an identity matrix;
       ``localized`` rebases poses relative to the first frame (matching,
       e.g., the PAI converter pattern) and stores the original first pose
       as ``world_global`` in float64
   * - ``--seek-sec FLOAT``
     - ``None``
     - Time to skip from the start of the sequence (in seconds). When set,
       frames before this offset are excluded from the output. The full pose
       range is still used internally for motion compensation
   * - ``--duration-sec FLOAT``
     - ``None``
     - Restrict total duration of the converted sequence (in seconds).
       Measured from the (possibly seeked) start time. When both
       ``--seek-sec`` and ``--duration-sec`` are given, only frames in the
       window ``[seek, seek + duration]`` are converted

For the complete implementation, see
``tools/data_converter/waymo/converter.py``.

API Reference
^^^^^^^^^^^^^

**V4 Components** (:mod:`ncore.data.v4`):

- :class:`~ncore.data.v4.SequenceComponentGroupsWriter` - Main writer for V4
  sequences
- :class:`~ncore.data.v4.PosesComponent` - Static and dynamic pose storage
- :class:`~ncore.data.v4.IntrinsicsComponent` - Camera and lidar intrinsics
- :class:`~ncore.data.v4.LidarSensorComponent` - Lidar frame data
- :class:`~ncore.data.v4.CameraSensorComponent` - Camera frame data
- :class:`~ncore.data.v4.CuboidsComponent` - 3D cuboid track observations
- :class:`~ncore.data.v4.MasksComponent` - Camera masks

**Data Converter** (:mod:`ncore.data_converter`):

- :class:`~ncore.data_converter.BaseDataConverter` - Abstract base class for
  converters
- :class:`~ncore.data_converter.BaseDataConverterConfig` - Base configuration
  dataclass

**Sensor Models** (:mod:`ncore.data`):

- :class:`~ncore.data.OpenCVPinholeCameraModelParameters` - Camera intrinsics
  model
- :class:`~ncore.data.RowOffsetStructuredSpinningLidarModelParameters` - Lidar
  intrinsics model
