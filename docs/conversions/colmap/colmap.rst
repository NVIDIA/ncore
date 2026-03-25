.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Colmap Dataset
==============

The NCore Colmap tool converts data from COLMAP data representations into NCore
V4 format.

.. _colmap_data_conventions:

Conventions
-----------

COLMAP's data format represents an arbitrary number of camera frames with
associated poses, but without timestamp information. Because NCore is designed
for applications with timestamped data, logical timestamps are assigned to
images at a rate of 1 FPS starting from a configurable start time.

Camera Sensors
^^^^^^^^^^^^^^

COLMAP cameras are identified by integer IDs. The converter maps them to NCore
sensor IDs using a configurable prefix (default: ``camera``). Downsampled
image directories (``images_2``, ``images_4``, ``images_8``) are optionally
included as additional camera sensor instances, each with their own individually
scaled intrinsics.

Example sensor IDs:

- **camera1** — full-resolution images from COLMAP camera 1
- **camera1_2** — 2× downsampled images from camera 1
- **camera1_4** — 4× downsampled images from camera 1
- **camera2** — full-resolution images from COLMAP camera 2

Camera intrinsics are compatible with the
:class:`~ncore.data.OpenCVPinholeCameraModelParameters` model for COLMAP camera
types 0–4 (``SIMPLE_PINHOLE``, ``PINHOLE``, ``SIMPLE_RADIAL``, ``RADIAL``,
``OPENCV``), and :class:`~ncore.data.OpenCVFisheyeCameraModelParameters` for
COLMAP camera type 5 (``OPENCV_FISHEYE``). COLMAP uses the same local camera
convention as NCore:

- Principal axis along the camera's +z axis
- x-axis points right, y-axis points down

Per-Image Masks
^^^^^^^^^^^^^^^

The converter automatically detects per-image mask files and stores them as
per-frame ``mask`` properties in the ``generic_data`` of each camera frame
(grayscale ``uint8``, shape ``[H, W]``).

Three mask-file conventions are supported, checked in priority order:

1. **Explicit masks directory** — ``<sequence_dir>/<masks_dir>/<stem>.png``
   when ``--masks-dir`` is configured.
2. **Co-located mask** — ``<image_dir>/<stem>_mask.png``
   alongside the image file.
3. **Separate masks directory** — ``<sequence_dir>/masks/<image_filename>``.

If no mask file is found for a given image, the frame is stored without a
``mask`` entry.  The number of masks found per camera is logged at INFO level.

LiDAR Sensors
^^^^^^^^^^^^^

- **virtual_lidar** — single static frame from the COLMAP SfM point cloud
  (optional)

If the COLMAP reconstruction contains 3D points, they are optionally exported as
a single frame of a virtual LiDAR sensor at the world origin. Each point carries
its reconstructed RGB color as generic per-point data. The virtual lidar has no
intrinsics and its extrinsic is an identity pose (sensor frame = world frame).

.. _data_conversions:

Conversion
----------

The converter uses NCore V4's component-based architecture. Each COLMAP scene is
parsed from a COLMAP reconstruction directory (default: ``sparse/0/``) and written to NCore format via
:class:`~ncore.data.v4.SequenceComponentGroupsWriter` with specialized component
writers for poses, intrinsics, cameras, and optionally a virtual lidar.

Usage
^^^^^

Run the converter with Bazel from the repository root:

.. code-block:: bash

    bazel run //tools/data_converter/colmap:convert -- \
        --root-dir <PATH_TO_COLMAP_SCENE> \
        --output-dir <PATH_TO_OUTPUT> \
        colmap-v4

If ``--root-dir`` points to a parent directory containing multiple scenes, each
subdirectory is treated as a separate sequence.

**Base arguments** (required):

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Argument
     - Description
   * - ``--root-dir PATH``
     - Path to a single COLMAP scene directory or a parent directory containing
       multiple scenes
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
     - Disable exporting the virtual lidar sensor
   * - ``--verbose``
     - Enable debug-level logging

**Subcommand arguments** (``colmap-v4``):

.. list-table::
   :widths: 35 20 45
   :header-rows: 1

   * - Argument
     - Default
     - Description
   * - ``--store-type {itar,directory}``
     - ``itar``
     - Output store format. ``itar`` produces an indexed tar archive;
       ``directory`` writes plain zarr directories
   * - ``--profile {default,separate-sensors,separate-all}``
     - ``separate-sensors``
     - Component group layout. ``default`` groups all sensors together;
       ``separate-sensors`` gives each sensor its own group; ``separate-all``
       splits every component type into its own group
   * - ``--sequence-meta`` / ``--no-sequence-meta``
     - enabled
     - Whether to write a JSON metadata file alongside each converted sequence
   * - ``--start-time-sec FLOAT``
     - ``0.0``
     - Logical start time in seconds assigned to the first image frame
   * - ``--camera-prefix TEXT``
     - ``camera``
     - Prefix prepended to COLMAP integer camera IDs to form NCore sensor IDs
       (e.g. ``camera1``)
   * - ``--include-downsampled-images`` / ``--no-include-downsampled-images``
     - enabled
     - Include downsampled image directories (``images_2``, ``images_4``,
       ``images_8``) as additional camera sensors
   * - ``--include-3d-points`` / ``--no-include-3d-points``
     - enabled
     - Include the SfM point cloud as a single frame of a ``virtual_lidar``
       sensor
   * - ``--colmap-dir TEXT``
     - ``sparse/0``
     - Relative path to the COLMAP reconstruction directory within each
       sequence
   * - ``--images-dir TEXT``
     - ``images``
     - Relative path to the image directory within each sequence
   * - ``--masks-dir TEXT``
     - (auto-detect)
     - Explicit masks directory relative to each sequence root. When set,
       looks for ``<masks_dir>/<stem>.png``

For the complete implementation, see
``tools/data_converter/colmap/converter.py``.

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

- :class:`~ncore.data.OpenCVPinholeCameraModelParameters` - Pinhole camera
  intrinsics model
- :class:`~ncore.data.OpenCVFisheyeCameraModelParameters` - Fisheye camera
  intrinsics model

ScanNet++ Conversion
^^^^^^^^^^^^^^^^^^^^

The ``scannetpp-v4`` subcommand converts ScanNet++ DSLR scenes using the resized
fisheye images (``dslr/resized_images/``) with the COLMAP ``OPENCV_FISHEYE``
camera model. Train/test split metadata from ``train_test_lists.json`` is stored
in the sequence-level ``generic_meta_data``.

.. code-block:: bash

    bazel run //tools/data_converter/colmap:convert -- \
        --root-dir /path/to/scannetpp/scene_id \
        --output-dir /path/to/output \
        scannetpp-v4

When ``--root-dir`` points to a parent directory containing multiple scenes,
each subdirectory with a ``dslr/colmap/`` directory is treated as a separate
scene.

**Subcommand arguments** (``scannetpp-v4``):

.. list-table::
   :widths: 35 20 45
   :header-rows: 1

   * - Argument
     - Default
     - Description
   * - ``--store-type {itar,directory}``
     - ``itar``
     - Output store format
   * - ``--profile {default,separate-sensors,separate-all}``
     - ``separate-sensors``
     - Component group layout
   * - ``--include-3d-points`` / ``--no-include-3d-points``
     - enabled
     - Include COLMAP SfM point cloud as virtual lidar
