.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Physical AI (PAI) Dataset
==========================

The NCore PAI tool converts data from the
`NVIDIA PhysicalAI-Autonomous-Vehicles <https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles>`_
HuggingFace dataset into NCore V4 format.

.. _pai_data_conventions:

Conventions
-----------

The PAI dataset is collected on the Hyperion 8 / Hyperion 8.1 sensor platform and provides
timestamped data from 7 cameras, 1 top lidar, egomotion, and cuboid obstacle labels per clip.

Camera Sensors
^^^^^^^^^^^^^^

    1. **Front Wide Camera 120° FOV (camera_front_wide_120fov)**
    2. **Front Tele Camera 30° FOV (camera_front_tele_30fov)**
    3. **Cross Left Camera 120° FOV (camera_cross_left_120fov)**
    4. **Cross Right Camera 120° FOV (camera_cross_right_120fov)**
    5. **Rear Left Camera 70° FOV (camera_rear_left_70fov)**
    6. **Rear Right Camera 70° FOV (camera_rear_right_70fov)**
    7. **Rear Tele Camera 30° FOV (camera_rear_tele_30fov)**

Camera video is stored as MP4 and extracted to individual JPEG frames during conversion.
Per-frame timestamps and optional blur-box metadata are provided.

Camera intrinsics are compatible with the :class:`~ncore.data.FThetaCameraModelParameters`,
:class:`~ncore.data.OpenCVFisheyeCameraModelParameters`, or
:class:`~ncore.data.OpenCVPinholeCameraModelParameters` models depending on the sensor.
Each camera model also carries a ``shutter_delay_us`` field that is used to compute
per-frame rolling shutter start timestamps.

LiDAR Sensors
^^^^^^^^^^^^^

    1. **Top Lidar 360° FOV (lidar_top_360fov)**

Lidar scans are converted from DRACO-compressed point clouds [#draco]_.
Each row is one spin frame and includes ``spin_start_timestamp``, ``spin_end_timestamp``, and
per-point attributes: XYZ position, intensity (0–255), per-point timestamp, and model element index.
Points inside the ego vehicle bounding box are filtered out during conversion.

Lidar intrinsics use the :class:`~ncore.data.RowOffsetStructuredSpinningLidarModelParameters` model.

Egomotion and Labels
^^^^^^^^^^^^^^^^^^^^

Rig-to-world poses are converted from a ``qx, qy, qz, qw, x, y, z, timestamp`` parquet. The first
pose in the selected time window defines the local world reference frame; all subsequent poses
are expressed relative to it.

Cuboid obstacle labels are stored in a per-clip ``obstacle.parquet`` with bounding box extents,
orientation quaternions, track IDs, class IDs, and label source annotations.

.. _pai_data_access:

Data Access
-----------

The converter supports two data access modes:

**Local mode** (``pai-v4``)
    Clips are first downloaded to disk using the ``pai-clip-dl`` tool, then converted from
    the local directory.

**Streaming mode** (``pai-stream-v4``)
    Clips are streamed directly from HuggingFace — no full download required. Calibration
    parquet files are downloaded per dataset chunk and filtered to the target clip. Video
    files are temporarily written to disk and cleaned up after each clip.

    When available, the streaming provider automatically uses pre-processed ``.offline`` variants
    of features (e.g. calibration, egomotion, obstacle) in place of the online features.

Prerequisites
^^^^^^^^^^^^^

- A HuggingFace account with the PAI dataset license accepted
- A HuggingFace API token (set ``HF_TOKEN`` or pass ``--hf-token``)
- For local mode: clips downloaded with the ``pai-clip-dl`` tool

Downloading Clips (Local Mode)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``pai-clip-dl`` tool manages downloads from the HuggingFace dataset:

.. code-block:: bash

    # Download one or more clips to a local directory
    bazel run //tools/data_converter/pai/pai_remote:pai-clip-dl -- \
        download <clip-id> [<clip-id> ...] \
        --output-dir /path/to/data \
        --revision ncore_test

    # Show clip metadata and sensor presence
    bazel run //tools/data_converter/pai/pai_remote:pai-clip-dl -- \
        info <clip-id>

    # List all available features in the dataset
    bazel run //tools/data_converter/pai/pai_remote:pai-clip-dl -- \
        list-features

The ``download`` command accepts ``--features`` (repeatable) to selectively download specific
feature types rather than the full clip. Omitting ``--features`` downloads all available features.

The downloaded clip directory has this layout::

    {output_dir}/{clip_id}/
    ├── calibration/
    │   ├── camera_intrinsics.parquet
    │   ├── sensor_extrinsics.parquet
    │   ├── vehicle_dimensions.parquet
    │   └── lidar_intrinsics.parquet      (optional)
    ├── labels/
    │   ├── {clip_id}.egomotion.parquet
    │   └── {clip_id}.obstacle.parquet    (optional)
    ├── camera/
    │   ├── {clip_id}.{camera_id}.mp4
    │   ├── {clip_id}.{camera_id}.timestamps.parquet
    │   └── {clip_id}.{camera_id}.blurred_boxes.parquet
    ├── lidar/
    │   └── {clip_id}.lidar_top_360fov.parquet
    └── metadata/
        ├── sensor_presence.parquet
        ├── data_collection.parquet
        └── provenance.json               (download source, optional)

.. _pai_data_conversions:

Conversion
----------

The converter uses NCore V4's component-based architecture. 
Data is written to NCore format via :class:`~ncore.data.v4.SequenceComponentGroupsWriter` with
specialized component writers for poses, intrinsics, cameras, lidar, masks, and cuboid labels.

Usage
^^^^^

**Local mode** — convert clips previously downloaded with ``pai-clip-dl``:

.. code-block:: bash

    bazel run //tools/data_converter/pai:convert -- \
        --root-dir <PATH_TO_CLIPS> \
        --output-dir <PATH_TO_OUTPUT> \
        pai-v4

**Streaming mode** — convert clips directly from HuggingFace without downloading:

.. code-block:: bash

    bazel run //tools/data_converter/pai:convert -- \
        --output-dir <PATH_TO_OUTPUT> \
        pai-stream-v4 \
        --clip-id <clip-id> \
        --hf-token <your-hf-token>

The output for each clip is written to::

    <output-dir>/pai_<clip-id>/pai_<clip-id>.ncore4.zarr.itar

**Base arguments** (required):

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Argument
     - Description
   * - ``--root-dir PATH``
     - Directory containing clip subdirectories (local mode). Ignored in streaming mode
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
   * - ``--verbose``
     - Enable debug-level logging

**Shared subcommand arguments** (``pai-v4`` and ``pai-stream-v4``):

.. list-table::
   :widths: 35 20 45
   :header-rows: 1

   * - Argument
     - Default
     - Description
   * - ``--clip-id ID``
     - all clips
     - Specific clip ID(s) to convert (repeatable). Required for streaming mode; filters discovered directories in local mode
   * - ``--seek-sec FLOAT``
     - none
     - Skip this many seconds from the start of each clip before converting
   * - ``--duration-sec FLOAT``
     - none
     - Limit the converted duration of each clip to this many seconds
   * - ``--store-type {itar,directory}``
     - ``itar``
     - Output store format. ``itar`` produces an indexed tar archive; ``directory`` writes plain zarr directories
   * - ``--profile {default,separate-sensors,separate-all}``
     - ``separate-sensors``
     - Component group layout. ``default`` groups all sensors together; ``separate-sensors`` gives each sensor its own group; ``separate-all`` splits every component type into its own group
   * - ``--sequence-meta`` / ``--no-sequence-meta``
     - enabled
     - Whether to write a JSON metadata file alongside each converted sequence

**Additional arguments** (``pai-stream-v4`` only):

.. list-table::
   :widths: 35 20 45
   :header-rows: 1

   * - Argument
     - Default
     - Description
   * - ``--hf-token TEXT``
     - ``$HF_TOKEN``
     - HuggingFace API token. Reads from the ``HF_TOKEN`` environment variable if not provided
   * - ``--revision TEXT``
     - ``main``
     - HuggingFace dataset branch or tag to stream from. Note: the ``pai-clip-dl`` download tool uses ``ncore_test`` as its default revision

For the complete implementation, see ``tools/data_converter/pai/converter.py``.

API Reference
^^^^^^^^^^^^^

**V4 Components** (:mod:`ncore.data.v4`):

- :class:`~ncore.data.v4.SequenceComponentGroupsWriter` - Main writer for V4 sequences
- :class:`~ncore.data.v4.PosesComponent` - Static and dynamic pose storage
- :class:`~ncore.data.v4.IntrinsicsComponent` - Camera and lidar intrinsics
- :class:`~ncore.data.v4.LidarSensorComponent` - Lidar frame data
- :class:`~ncore.data.v4.CameraSensorComponent` - Camera frame data
- :class:`~ncore.data.v4.CuboidsComponent` - 3D cuboid track observations
- :class:`~ncore.data.v4.MasksComponent` - Camera ego masks

**Data Converter** (:mod:`ncore.data_converter`):

- :class:`~ncore.data_converter.BaseDataConverter` - Abstract base class for converters
- :class:`~ncore.data_converter.BaseDataConverterConfig` - Base configuration dataclass

**Sensor Models** (:mod:`ncore.data`):

- :class:`~ncore.data.FThetaCameraModelParameters` - FTheta (equidistant) camera model
- :class:`~ncore.data.OpenCVFisheyeCameraModelParameters` - Kannala-Brandt fisheye camera model
- :class:`~ncore.data.OpenCVPinholeCameraModelParameters` - Radial/tangential pinhole camera model
- :class:`~ncore.data.RowOffsetStructuredSpinningLidarModelParameters` - Spinning lidar model

.. rubric:: Footnotes

.. [#draco] The PAI lidar feature requires the `DracoPy <https://pypi.org/project/DracoPy/>`_ library
    for decompressing Google Draco-encoded point clouds.
