.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Waymo-Open Dataset
==================

The NCore Waymo tool converts data from the Waymo-Open format (**.tfrecords**) into NCore V4 format.
It serves as a reference implementation for creating data conversion flows to NCore.

.. _data_conventions:

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

The camera intrinsics are compatible with the :class:`~ncore.data.OpenCVPinholeCameraModelParameters` model.

The Waymo camera frame convention is:
    - Principal axis along local camera's +x axis
    - Local y-axis points left
    - Local z-axis points up

**Note:** The converter transforms this to NCore's camera convention (principal axis +z, x-axis right, y-axis down).

Each camera provides panoptic segmentation data with 29 semantic classes (see ``CAMERA_LABEL_CLASS_ID_STRING_MAP`` in the converter source).

LiDAR Sensors
^^^^^^^^^^^^^
    1. **Top Lidar (lidar_top)**

LiDAR data includes point clouds with multi-return support (primary and secondary returns) and 3D bounding box labels for 5 classes: unknown, vehicle, pedestrian, sign, cyclist.

For more information, see the `Waymo Open Dataset paper <https://openaccess.thecvf.com/content_CVPR_2020/papers/Sun_Scalability_in_Perception_for_Autonomous_Driving_Waymo_Open_Dataset_CVPR_2020_paper.pdf>`_.


.. _data_conversions:

Conversion
----------

Overview
^^^^^^^^

The converter uses NCore V4's component-based architecture. Data is written through specialized component writers
registered with a ``SequenceComponentGroupsWriter``:

.. figure:: waymo_conversion_flow.png

The conversion extracts three key data types from each Waymo sequence:
    1. **Poses** - Rig-to-world transformations over time
    2. **LiDAR data** - Point clouds with direction/distance format and multi-return support
    3. **Camera data** - Images with intrinsics and optional segmentation

Configuration
^^^^^^^^^^^^^

The converter is configured via ``WaymoConverter4Config``:

.. code-block:: python

    @dataclass
    class WaymoConverter4Config(BaseDataConverterConfig):
        store_type: Literal["itar", "directory"] = "itar"
        component_group_profile: Literal["default", "separate-sensors", "separate-all"] = "separate-sensors"
        store_sequence_meta: bool = True

- **store_type**: Output format - ``"itar"`` (indexed tar archive, default) or ``"directory"`` (plain zarr directory)
- **component_group_profile**: How components are grouped in the output store
- **store_sequence_meta**: Whether to generate a JSON metadata file for the sequence

Convert Sequence
^^^^^^^^^^^^^^^^

The ``convert_sequence`` method implements the core conversion logic:

**Step 1: Parse frames and decode poses**

.. code-block:: python

    def convert_sequence(self, sequence_path: UPath) -> None:
        dataset = tf.data.TFRecordDataset(sequence_path, compression_type="")

        frames: list[dataset_pb2.Frame] = []
        sequence_name = ""
        for data in dataset:
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            if not frames:
                sequence_name = frame.context.name
            frames.append(frame)

        # Decode poses first (needed for timestamp interval)
        self.decode_poses(frames)

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

    # Register component writers
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

Poses are stored as dynamic (time-varying) and static transforms:

.. code-block:: python

    # Store dynamic rig->world poses
    self.poses_writer.store_dynamic_pose(
        source_frame_id="rig",
        target_frame_id="world",
        poses=self.pose_interpolator.poses.astype(np.float32),
        timestamps_us=self.pose_interpolator.timestamps,
    )

    # Store static world->world_global pose (high precision base transform)
    self.poses_writer.store_static_pose(
        source_frame_id="world",
        target_frame_id="world_global",
        pose=self.T_world_world_global.astype(np.float64),
    )

**Step 4: Decode and store lidar data**

Each lidar sensor gets its own component writer:

.. code-block:: python

    lidar_writer = self.store_writer.register_component_writer(
        LidarSensorComponent.Writer,
        component_instance_name=lidar_ncore_id,
        group_name=self.component_groups.lidar_component_groups.get(lidar_ncore_id),
    )

Lidar frames use direction/distance format with multi-return support:

.. code-block:: python

    lidar_writer.store_frame(
        direction=direction,                    # [N, 3] ray directions (non-motion-compensated)
        timestamp_us=point_timestamps_us,       # [N] per-point timestamps
        model_element=model_element,            # [N, 2] indices into lidar model
        distance_m=distance_m,                  # [2, N] distances (2 returns)
        intensity=intensity,                    # [2, N] intensities (2 returns)
        frame_timestamps_us=frame_timestamps_us,  # [2] start/end timestamps
        generic_data={"elongation": elongation},
        generic_meta_data={},
    )

Intrinsics and extrinsics are stored separately. The ``lidar_model_parameters`` uses :class:`~ncore.data.RowOffsetStructuredSpinningLidarModelParameters`:

.. code-block:: python

    # Store lidar intrinsics
    self.intrinsics_writer.store_lidar_intrinsics(
        lidar_id=lidar_ncore_id,
        lidar_model_parameters=lidar_model_parameters,
    )

    # Store extrinsics as static pose
    self.poses_writer.store_static_pose(
        source_frame_id=lidar_ncore_id,
        target_frame_id="rig",
        pose=T_sensor_rig,
    )

3D labels are stored as cuboid track observations:

.. code-block:: python

    cuboids_writer = self.store_writer.register_component_writer(
        CuboidsComponent.Writer,
        "default",
        self.component_groups.cuboid_track_observations_component_group,
    )
    cuboids_writer.store_observations(cuboid_track_observations)

**Step 5: Decode and store camera data**

Each camera sensor gets its own component writer:

.. code-block:: python

    camera_writer = self.store_writer.register_component_writer(
        CameraSensorComponent.Writer,
        component_instance_name=camera_ncore_id,
        group_name=self.component_groups.camera_component_groups.get(camera_ncore_id),
    )

    camera_writer.store_frame(
        image_binary_data=image.image,
        image_format="jpeg",
        frame_timestamps_us=np.array([frame_start_timestamp_us, frame_end_timestamp_us], dtype=np.uint64),
        generic_data=generic_data,      # e.g., panoptic segmentation
        generic_meta_data=generic_meta_data,
    )

Camera intrinsics (using :class:`~ncore.data.OpenCVPinholeCameraModelParameters` with sensor-specific shutter directions), masks, and extrinsics:

.. code-block:: python

    # Store camera intrinsics
    self.intrinsics_writer.store_camera_intrinsics(
        camera_id=camera_ncore_id,
        camera_model_parameters=OpenCVPinholeCameraModelParameters(...),
    )

    # Store empty masks (none available in Waymo dataset)
    self.masks_writer.store_camera_masks(
        camera_id=camera_ncore_id,
        mask_images={},
    )

    # Store extrinsics as static pose
    self.poses_writer.store_static_pose(
        source_frame_id=camera_ncore_id,
        target_frame_id="rig",
        pose=T_sensor_rig,
    )

**Step 6: Finalize and optionally generate sequence metadata**

.. code-block:: python

    ncore_4_paths = self.store_writer.finalize()

    if self.store_sequence_meta:
        sequence_component_reader = SequenceComponentGroupsReader(ncore_4_paths)
        sequence_meta_path = self.output_dir / sequence_name / f"{sequence_component_reader.sequence_id}.json"

        with sequence_meta_path.open("w") as f:
            json.dump(sequence_component_reader.get_sequence_meta().to_dict(), f, indent=2)

Summary
^^^^^^^

The Waymo V4 conversion follows this pattern:

1. Parse ``.tfrecord`` frames using TensorFlow and ``waymo_open_dataset``
2. Initialize ``SequenceComponentGroupsWriter`` and register component writers
3. Store poses via ``PosesComponent.Writer`` (dynamic + static transforms)
4. Store lidar data via ``LidarSensorComponent.Writer`` (direction/distance format, multi-return)
5. Store lidar intrinsics via ``IntrinsicsComponent.Writer``
6. Store 3D labels via ``CuboidsComponent.Writer``
7. Store camera data via ``CameraSensorComponent.Writer``
8. Store camera intrinsics via ``IntrinsicsComponent.Writer`` and masks via ``MasksComponent.Writer``
9. Store all extrinsics as static poses
10. Finalize writer and optionally generate sequence metadata JSON

For the complete implementation, see ``ncore_waymo/converter.py``.

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
- :class:`~ncore.data.RowOffsetStructuredSpinningLidarModelParameters` - Lidar intrinsics model
