.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Colmap Dataset
==============

The NCore Colmap tool converts data from Colmap into NCore V4 format.
It also serves as a reference implementation for creating data conversion flows to NCore.

.. _colmap_data_conventions:

Conventions
-----------

Colmap's data format represents an arbitrary number of instantaneous camera frames with associated poses, but without timestamps information.
Because NCore is designed for applications with timestamped data, we add use logical timestamps to identify images at a rate of 1 FPS.
These cameras may additionally have downsampled data, which can be represented as additional camera sensor instances, because they
have their individually associated intrinsics.

Example Camera Sensors
^^^^^^^^^^^^^^^^^^^^^^
    1. **camera1 (the camera prefix 'camera' can be customized in the cli)**
    2. **camera1_2 (the camera with the 2x downsampled images from camera1)**
    3. **camera1_4 (the camera with the 4x downsampled images from camera1)**
    4. **camera2 (a second camera)**

The camera intrinsics are compatible with the :class:`~ncore.data.OpenCVPinholeCameraModelParameters` model. [#opencv_fisheye]_

Colmap cameras use the same local camera sensor conventions as NCore:
    - Principal axis along local camera's positive z axis
    - Local x-axis points right
    - Local y-axis points down

LiDAR Sensors
^^^^^^^^^^^^^
    1. **virtual_lidar (single frame static point cloud)**

Colmap datasets additionally may expose a single static point cloud of SfM
points, which are optionally represented in NCore as a single frame of a virtual
lidar.

.. _colmap_data_conversions:

Conversion
----------

Overview
^^^^^^^^

The converter uses NCore V4's component-based architecture. Data is written through specialized component writers
registered with a ``SequenceComponentGroupsWriter``:

The conversion extracts three key data types from each Colmap scene:
    1. **Poses** - Rig-to-world transformations over (arbitrary) time
    2. **LiDAR data** - A single 3D SfM point cloud with per-point colors, represented as a single frame of a virtual LiDAR sensor
    3. **Camera data** - Images with intrinsics

Configuration
^^^^^^^^^^^^^

The converter is configured via ``ColmapConverter4Config``:

.. code-block:: python

    @dataclass
    class ColmapConverter4Config(BaseDataConverterConfig):
        store_type: Literal["itar", "directory"] = "itar"
        component_group_profile: Literal["default", "separate-sensors", "separate-all"] = "separate-sensors"
        store_sequence_meta: bool = True
        start_time_sec: float = 0.0
        camera_prefix: str = "camera"
        include_downsampled_images: bool = True
        include_3d_points: bool = True

- **store_type**: Output format - ``"itar"`` (indexed tar archive, default) or ``"directory"`` (plain zarr directory)
- **component_group_profile**: How components are grouped in the output store
- **store_sequence_meta**: Whether to generate a JSON metadata file for the sequence (default is True)
- **start_time_sec**: A time offset to add to all timestamps (default is 0)
- **camera_prefix**: Colmap cameras are identified with integer IDs, this prefix is prepended to identify the corresponding NCore camera sensor (default is ``"camera"``)
- **include_downsampled_images**: Whether to include downsampled images as extra camera sensors (default is True)
- **include_3d_points**: Whether to include 3d points as a single frame of a ``"virtual_lidar"`` sensor (default is True)

Convert Sequence
^^^^^^^^^^^^^^^^

The ``convert_sequence`` method implements the core conversion logic:

**Step 1: Parse frames and decode poses**

.. code-block:: python

    def convert_sequence(self, sequence_path: UPath) -> None:

        bin_path = self.sequence_path / "sparse" / "0"
        self.scene_manager = pycolmap.SceneManager(str(bin_path))
        self.scene_manager.load_cameras()
        self.scene_manager.load_images()
        self.scene_manager.load_points3D()

        self.cameras = self.populate_camera_data(
            parent_dir=self.sequence_path,
            camera_prefix=self.camera_prefix,
            downsample=self.include_downsampled_images,
        )

**Step 2: Initialize the writer and register component writers**

.. code-block:: python

    # Create component group assignments
    self.component_groups = ComponentGroupAssignments.create(
        camera_ids=self.camera_ids,
        lidar_ids=self.lidar_ids,
        radar_ids=[],
        profile=self.component_group_profile,
    )

    # Create main store writer
    self.store_writer = SequenceComponentGroupsWriter(
        output_dir_path=self.output_dir / sequence_name,
        store_base_name=sequence_name,
        sequence_id=sequence_name,
        sequence_timestamp_interval_us=sequence_timestamp_interval_us,
        store_type=self.store_type,
        generic_meta_data={},
    )

    # Register default component writers
    self.poses_writer = self.store_writer.register_component_writer(
        PosesComponent.Writer,
        component_instance_name="default",
        group_name=self.component_groups.poses_component_group,
    )

    self.intrinsics_writer = self.store_writer.register_component_writer(
        IntrinsicsComponent.Writer,
        component_instance_name="default",
        group_name=self.component_groups.intrinsics_component_group,
    )

    self.masks_writer = self.store_writer.register_component_writer(
        MasksComponent.Writer,
        component_instance_name="default",
        group_name=self.component_groups.masks_component_group,
    )

**Step 3: Store poses**

Camera poses are stored as dynamic (time-varying) transforms, while other poses are static:

.. code-block:: python

    # Store static world->world_global pose (high precision base transform)
    self.poses_writer.store_static_pose(
        source_frame_id="world",
        target_frame_id="world_global",
        pose=self.T_world_world_global.astype(np.float64),
    )

**Step 4: Decode and store lidar data**

The virtual lidar sensor is associated with its own component writer:

.. code-block:: python

    lidar_writer = self.store_writer.register_component_writer(
        LidarSensorComponent.Writer,
        component_instance_name=lidar_ncore_id,
        group_name=self.component_groups.lidar_component_groups.get(lidar_ncore_id),
    )

Lidar frames use direction/distance format with multi-return support (only 1 return is needed for Colmap):

.. code-block:: python

    lidar_writer.store_frame(
        direction=direction,                      # [N, 3] ray directions (non-motion-compensated)
        timestamp_us=point_timestamps_us,         # [N] per-point timestamps
        model_element=None,                       # [N, 2] indices into lidar model
        distance_m=distance_m,                    # [2, N] distances (2 returns)
        intensity=intensity,                      # [2, N] intensities (2 returns)
        frame_timestamps_us=frame_timestamps_us,  # [2] start/end timestamps
        generic_data={"rgb": self.scene_manager.point3D_colors}, # Custom per-point color data
        generic_meta_data={},
    )

The virtual lidar sensor has associated intrinsics, its world-pose is represented as an identity pose:

.. code-block:: python

    # Store extrinsics as static pose
    self.poses_writer.store_static_pose(
        source_frame_id=lidar_ncore_id,
        target_frame_id="world",
        pose=np.eye(4, dtype=np.float64),
    )

**Step 5: Decode and store camera data**

Each camera sensor is associated with its own component writer:

.. code-block:: python

    camera_writer = self.store_writer.register_component_writer(
        CameraSensorComponent.Writer,
        component_instance_name=camera_ncore_id,
        group_name=self.component_groups.camera_component_groups.get(camera_ncore_id),
    )

    camera_writer.store_frame(
        image_binary_data=image_bytes,
        image_format="jpeg",
        frame_timestamps_us=np.array([frame_start_timestamp_us, frame_end_timestamp_us], dtype=np.uint64),
        generic_data={},
        generic_meta_data={},
    )

Camera intrinsics (using :class:`~ncore.data.OpenCVPinholeCameraModelParameters`), masks, and extrinsics:

.. code-block:: python

    # Store camera intrinsics
    self.intrinsics_writer.store_camera_intrinsics(
        camera_id=camera_ncore_id,
        camera_model_parameters=OpenCVPinholeCameraModelParameters(...),
    )

    # Store empty masks (none available in Colmap dataset)
    self.masks_writer.store_camera_masks(
        camera_id=camera_ncore_id,
        mask_images={},
    )

    # Camera reference frames are either "world" or, for downsampled cameras, the base camera.
    self.poses_writer.store_dynamic_pose(
        source_frame_id=camera_ncore_id,
        target_frame_id=colmap_camera.reference_frame,
        poses=colmap_camera.T_camera_ref,
        timestamps_us=colmap_camera.timestamps_us,
        require_sequence_time_coverage=False,  # Some cameras may have more poses than others
    )

**Step 6: Finalize and optionally generate sequence metadata**

.. code-block:: python

    ncore_4_paths = self.store_writer.finalize()

Summary
^^^^^^^

The Colmap V4 conversion follows this pattern:

1. Parse Colmap scene information
2. Initialize ``SequenceComponentGroupsWriter`` and register component writers
3. Store poses via ``PosesComponent.Writer`` (dynamic + static transforms)
4. Store lidar data via ``LidarSensorComponent.Writer`` (direction/distance format)
5. Store lidar intrinsics via ``IntrinsicsComponent.Writer``
6. Store camera data via ``CameraSensorComponent.Writer``
7. Store camera intrinsics via ``IntrinsicsComponent.Writer`` and masks via ``MasksComponent.Writer``
8. Store all extrinsics as static poses
9. Finalize writer

For the complete implementation, see ``colmap/converter.py``.

API Reference
^^^^^^^^^^^^^

**V4 Components** (:mod:`ncore.data.v4`):

- :class:`~ncore.data.v4.SequenceComponentGroupsWriter` - Main writer for V4 sequences
- :class:`~ncore.data.v4.PosesComponent` - Static and dynamic pose storage
- :class:`~ncore.data.v4.IntrinsicsComponent` - Camera and lidar intrinsics
- :class:`~ncore.data.v4.LidarSensorComponent` - Lidar frame data
- :class:`~ncore.data.v4.CameraSensorComponent` - Camera frame data
- :class:`~ncore.data.v4.CuboidsComponent` - 3D cuboid track observations
- :class:`~ncore.data.v4.MasksComponent` - Camera masks

**Data Converter** (:mod:`ncore.data_converter`):

- :class:`~ncore.data_converter.BaseDataConverter` - Abstract base class for converters
- :class:`~ncore.data_converter.BaseDataConverterConfig` - Base configuration dataclass

**Sensor Models** (:mod:`ncore.data`):

- :class:`~ncore.data.OpenCVPinholeCameraModelParameters` - Camera intrinsics model

.. rubric:: Footnotes

.. [#opencv_fisheye] Support for
    :class:`~ncore.data.OpenCVFisheyeCameraModelParameters` could be added in
    the future to additionally handle fisheye lens distortion models from
    Colmap.
