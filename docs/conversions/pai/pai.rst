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
Per-frame timestamps and optional blur-box metadata are provided in companion parquet files.

Camera intrinsics are compatible with the :class:`~ncore.data.FThetaCameraModelParameters`,
:class:`~ncore.data.OpenCVFisheyeCameraModelParameters`, or
:class:`~ncore.data.OpenCVPinholeCameraModelParameters` models depending on the sensor.
Each camera model also carries a ``shutter_delay_us`` field that is used to compute
per-frame rolling shutter start timestamps.

LiDAR Sensors
^^^^^^^^^^^^^

    1. **Top Lidar 360° FOV (lidar_top_360fov)**

Lidar scans are stored as DRACO-compressed point clouds [#draco]_ in a per-clip parquet file.
Each row is one spin frame and includes ``spin_start_timestamp``, ``spin_end_timestamp``, and
per-point attributes: XYZ position, intensity (0–255), per-point timestamp, and model element index.
Points inside the ego vehicle bounding box are filtered out during conversion.

Lidar intrinsics use the :class:`~ncore.data.RowOffsetStructuredSpinningLidarModelParameters` model,
loaded either from a companion ``lidar_intrinsics.parquet`` or from the Arrow schema metadata
embedded in the lidar parquet file.

Egomotion and Labels
^^^^^^^^^^^^^^^^^^^^

Rig-to-world poses are stored as a ``qx, qy, qz, qw, x, y, z, timestamp`` parquet. The first
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
    Clips are streamed directly from HuggingFace via HTTP range requests — no full download
    required. Video data is temporarily written to disk (ffmpeg requires local file paths)
    and cleaned up after each clip.

Both modes use the same ``ClipDataProvider`` protocol, so the conversion logic is identical
regardless of data source.

Prerequisites
^^^^^^^^^^^^^

- A HuggingFace account with the PAI dataset license accepted
- A HuggingFace API token (set ``HF_TOKEN`` or pass ``--hf-token``)
- For local mode: clips downloaded with the ``pai-clip-dl`` tool

.. code-block:: bash

    # Download a single clip with pai-clip-dl (Bazel)
    bazel run //tools/data_converter/pai/pai-clip-dl:pai-clip-dl -- \
        download <clip-id> -o /path/to/data

    # Or using uv directly from the pai-clip-dl subdirectory
    cd tools/data_converter/pai/pai-clip-dl
    uv run pai-clip-dl download <clip-id> -o /path/to/data

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
        └── data_collection.parquet

.. _pai_data_conversions:

Conversion
----------

Overview
^^^^^^^^

The converter uses NCore V4's component-based architecture. Data is written through specialized
component writers registered with a ``SequenceComponentGroupsWriter``:

The conversion extracts four key data types from each PAI clip:

    1. **Poses** - Rig-to-world transformations over time (from egomotion parquet)
    2. **Camera data** - JPEG frames extracted from MP4 with intrinsics, extrinsics, and optional ego masks
    3. **LiDAR data** - Per-point direction/distance from DRACO-compressed scans with per-point timestamps
    4. **Cuboid labels** - 3D obstacle tracks with class IDs, bounding boxes, and source annotations

Configuration
^^^^^^^^^^^^^

The converter is configured via ``PaiConverter4Config``:

.. code-block:: python

    @dataclass
    class PaiConverter4Config(BaseDataConverterConfig):
        seek_sec: float | None = None
        duration_sec: float | None = None
        clip_id: list[str] = []
        store_type: Literal["itar", "directory"] = "itar"
        component_group_profile: Literal["default", "separate-sensors", "separate-all"] = "separate-sensors"
        store_sequence_meta: bool = True

- **seek_sec**: Skip this many seconds from the start of the clip (default: no skip)
- **duration_sec**: Restrict total converted duration in seconds (default: full clip)
- **clip_id**: Specific clip UUIDs to convert. In local mode, filters the discovered directories;
  in streaming mode this is required.
- **store_type**: Output format — ``"itar"`` (indexed tar archive, default) or ``"directory"`` (plain zarr)
- **component_group_profile**: How sensor components are grouped in the output store
- **store_sequence_meta**: Whether to generate a JSON metadata sidecar file (default: ``True``)

The streaming mode adds two further fields via ``PaiStreamConverterConfig``:

- **hf_token**: HuggingFace API token (reads from ``HF_TOKEN`` env var if not set)
- **revision**: Dataset branch or tag to use (default: ``"main"``)

Convert Sequence
^^^^^^^^^^^^^^^^

The ``convert_sequence`` method implements the core conversion logic for a single clip.

**Step 1: Load metadata**

Calibration and platform information are loaded through the ``ClipDataProvider``:

.. code-block:: python

    def load_metadata(self) -> None:
        self.sensor_presence = self.provider.get_sensor_presence()

        intrinsics_df = self.provider.load_parquet("camera_intrinsics")
        self.camera_intrinsics = parse_camera_intrinsics(intrinsics_df, self.clip_id, self.sensor_presence)

        extrinsics_df = self.provider.load_parquet("sensor_extrinsics")
        self.sensor_extrinsics = parse_sensor_extrinsics(extrinsics_df, self.clip_id, self.sensor_presence)

        dimensions_df = self.provider.load_parquet("vehicle_dimensions")
        self.vehicle_dimensions = parse_vehicle_dimensions(dimensions_df, self.clip_id)

**Step 2: Parse egomotion and apply time bounds**

.. code-block:: python

    ego_df = self.provider.load_parquet("egomotion")
    T_rig_worlds, T_rig_world_timestamps_us = parse_egomotion_parquet(ego_df)

    # Apply seek_sec / duration_sec clipping
    start_us, end_us = time_bounds(T_rig_world_timestamps_us.tolist(), self.seek_sec, self.duration_sec)

    # First pose defines the local world reference frame
    T_world_world_global = T_rig_worlds[0].copy()
    T_rig_worlds = se3_inverse(T_world_world_global) @ T_rig_worlds

**Step 3: Initialize the writer and register component writers**

.. code-block:: python

    self.component_groups = ComponentGroupAssignments.create(
        camera_ids=active_camera_ids,
        lidar_ids=active_lidar_ids,
        radar_ids=[],
        profile=self.component_group_profile,
    )

    self.store_writer = SequenceComponentGroupsWriter(
        output_dir_path=self.output_dir / sequence_id,
        store_base_name=sequence_id,
        sequence_id=sequence_id,
        sequence_timestamp_interval_us=self.sequence_timestamp_interval_us,
        store_type=self.store_type,
        generic_meta_data={...},   # converter version, source clip/repo provenance
    )

    self.poses_writer     = self.store_writer.register_component_writer(PosesComponent.Writer, ...)
    self.intrinsics_writer = self.store_writer.register_component_writer(IntrinsicsComponent.Writer, ...)
    self.masks_writer     = self.store_writer.register_component_writer(MasksComponent.Writer, ...)

**Step 4: Store poses**

.. code-block:: python

    # Dynamic rig->world poses (float32, relative to local world frame)
    self.poses_writer.store_dynamic_pose(
        source_frame_id="rig",
        target_frame_id="world",
        poses=T_rig_worlds.astype(np.float32),
        timestamps_us=T_rig_world_timestamps_us,
    )

    # Static world->world_global (float64, absolute base transform)
    self.poses_writer.store_static_pose(
        source_frame_id="world",
        target_frame_id="world_global",
        pose=T_world_world_global,
    )

**Step 5: Decode and store camera data**

Frames are extracted from MP4 via ffmpeg (``imageio``). Each camera gets its own component writer:

.. code-block:: python

    camera_writer = self.store_writer.register_component_writer(
        CameraSensorComponent.Writer,
        component_instance_name=camera_id,
        group_name=self.component_groups.camera_component_groups.get(camera_id),
    )

    camera_writer.store_frame(
        image_binary_data=jpeg_bytes,                     # JPEG-encoded at quality=93
        image_format="jpeg",
        frame_timestamps_us=np.array([                    # rolling-shutter start/end
            frame_end_timestamp_us - shutter_delay_us,
            frame_end_timestamp_us,
        ], dtype=np.uint64),
        generic_data={},
        generic_meta_data={"blur_boxes": ...},            # optional privacy blur regions
    )

Camera intrinsics, ego masks, and extrinsics are stored alongside:

.. code-block:: python

    # Intrinsics (FTheta, OpenCV Fisheye, or OpenCV Pinhole depending on sensor)
    self.intrinsics_writer.store_camera_intrinsics(
        camera_id=camera_id,
        camera_model_parameters=camera_model_parameters,
    )

    # Ego mask image (PNG, stored if provided in camera_intrinsics parquet)
    self.masks_writer.store_camera_masks(
        camera_id=camera_id,
        mask_images={"ego": ego_mask_image},  # empty dict if not available
    )

    # Static sensor->rig extrinsic
    self.poses_writer.store_static_pose(
        source_frame_id=camera_id,
        target_frame_id="rig",
        pose=T_sensor_rig.astype(np.float32),
    )

**Step 6: Decode and store lidar data**

DRACO-compressed point clouds are decoded per frame. Ego vehicle points are filtered out
using the vehicle bounding box before storage:

.. code-block:: python

    lidar_writer = self.store_writer.register_component_writer(
        LidarSensorComponent.Writer,
        component_instance_name=lidar_id,
        group_name=self.component_groups.lidar_component_groups.get(lidar_id),
    )

    pc = DracoPy.decode(row["draco_encoded_pointcloud"])
    valid = filter_ego_vehicle_points(xyz_m, T_sensor_rig, self.vehicle_dimensions)

    lidar_writer.store_frame(
        direction=direction[valid],            # [N, 3] unit ray directions in sensor frame
        timestamp_us=point_timestamps_us[valid],  # [N] per-point timestamps (µs)
        model_element=model_element[valid],    # [N, 2] row/column indices into lidar model
        distance_m=distance_m[valid][np.newaxis],  # [1, N] distances (single return)
        intensity=intensity[valid][np.newaxis],    # [1, N] normalized to [0, 1]
        frame_timestamps_us=np.array([spin_start, spin_end], dtype=np.uint64),
        generic_data={},
        generic_meta_data={},
    )

**Step 7: Store cuboid labels**

.. code-block:: python

    cuboids_writer = self.store_writer.register_component_writer(
        CuboidsComponent.Writer,
        "default",
        self.component_groups.cuboid_track_observations_component_group,
    )
    cuboids_writer.store_observations(self._load_cuboid_track_observations())

**Step 8: Finalize and optionally generate sequence metadata**

.. code-block:: python

    ncore_4_paths = self.store_writer.finalize()

    if self.store_sequence_meta:
        sequence_component_reader = SequenceComponentGroupsReader(ncore_4_paths)
        sequence_meta_path = self.output_dir / sequence_id / f"{sequence_id}.json"
        with sequence_meta_path.open("w") as f:
            json.dump(sequence_component_reader.get_sequence_meta().to_dict(), f, indent=2)

Summary
^^^^^^^

The PAI V4 conversion follows this pattern:

1. Load calibration via ``ClipDataProvider`` (local disk or HuggingFace streaming)
2. Parse egomotion parquet into SE3 poses; apply optional seek/duration clipping
3. Initialize ``SequenceComponentGroupsWriter`` and register component writers
4. Store dynamic rig→world poses and static world→world_global anchor via ``PosesComponent.Writer``
5. Decode MP4 video frames with ``imageio``/ffmpeg and store as JPEG via ``CameraSensorComponent.Writer``
6. Store camera intrinsics via ``IntrinsicsComponent.Writer``, ego masks via ``MasksComponent.Writer``,
   and sensor→rig extrinsics as static poses
7. Decode DRACO lidar scans, filter ego vehicle points, and store via ``LidarSensorComponent.Writer``
8. Store lidar intrinsics (from ``lidar_intrinsics.parquet`` or parquet schema metadata) via
   ``IntrinsicsComponent.Writer``
9. Store cuboid obstacle labels via ``CuboidsComponent.Writer``
10. Finalize writer and optionally generate a sequence metadata JSON sidecar

For the complete implementation, see ``tools/data_converter/pai/converter.py`` (local and streaming)
and ``tools/data_converter/pai/pai_stream.py`` (streaming-specific initialization).

Usage
^^^^^

**Local mode** — convert clips downloaded with ``pai-clip-dl``:

.. code-block:: bash

    bazel run //scripts:convert_raw_data -- \
        --root-dir=/path/to/data \
        --output-dir=/path/to/output \
        pai-v4 \
        --clip-id <clip-id>

**Streaming mode** — convert directly from HuggingFace:

.. code-block:: bash

    bazel run //scripts:convert_raw_data -- \
        --root-dir=/tmp/unused \
        --output-dir=/path/to/output \
        pai-stream-v4 \
        --clip-id <clip-id> \
        --hf-token <your-hf-token>

The output for each clip is written to::

    <output-dir>/pai_<clip-id>/pai_<clip-id>.ncore4.zarr.itar

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
