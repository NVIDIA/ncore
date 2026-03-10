.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _data_formats:


Format
======

The current NCore data format is **V4 (Component-based Format)** -- a modular
format that separates data into independent component stores. Each component
(poses, intrinsics, sensors, labels, etc.) is stored as a separate zarr
component that can be independently managed, versioned, and combined. This
format enables:

* Flexible data composition from multiple sources
* Independent component updates without reprocessing entire sequences
* Parallel access and distributed storage optimization
* Extensibility through custom component types
* Fine-grained access control and data sharing

The format uses coordinate system conventions and transformations described
in :ref:`data_conventions`.

.. _v4-data-format:

V4: Component Store Hierarchy (Component-Based Format)
------------------------------------------------------

The component-based V4 data format represents sequences as collections of
*component groups*. V4 distributes data across modular components that can
be independently managed, versioned, and combined to form virtual sequences.

Component Architecture
~~~~~~~~~~~~~~~~~~~~~~

Each component group is a zarr store (either a ``.zarr.itar`` archive or a
directory-based ``.zarr`` store) containing a specific number of data component
instances. The NCore library provides the following default component types:

* :class:`~ncore.data.v4.PosesComponent` - Static and dynamic pose
  transformations between named coordinate frames
* :class:`~ncore.data.v4.IntrinsicsComponent` - Camera and lidar intrinsic
  calibration parameters
* :class:`~ncore.data.v4.MasksComponent` - Static masks associated with sensors
* :class:`~ncore.data.v4.CameraSensorComponent` - Camera frame data including
  images
* :class:`~ncore.data.v4.LidarSensorComponent` - Lidar frame data including
  point clouds
* :class:`~ncore.data.v4.RadarSensorComponent` - Radar frame data including
  detections
* :class:`~ncore.data.v4.CuboidsComponent` - 3D cuboid track observations and
  annotations

The component architecture is extensible, allowing custom component types to be
defined for application-specific data.

.. _itar_store_format:

Indexed Tar Archive Format (``.itar``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each component group is a `zarr <https://zarr.readthedocs.io/en/stable/>`_
group stored either as a directory-based ``.zarr`` store or as a single-file
``.zarr.itar`` (indexed tar) archive. The itar format packages all zarr chunks
as sequential tar members in a single file and appends a compressed index at
the end, combining the streaming efficiency of tar with random-access
capability.

.. figure:: itar.svg
   :width: 100%

   Comparison of regular tar files (as used by
   `WebDataset <https://github.com/webdataset/webdataset>`_, supporting fast
   linear streaming but no random access) with the indexed tar format, which
   appends a compressed index enabling O(1) key lookups and direct seeks to
   any chunk.

The itar store implements the zarr ``Store`` interface, so it can be used as a
drop-in replacement for directory stores in all NCore APIs. Via
`UPath <https://github.com/fsspec/universal_pathlib>`_,
itar containers can also be accessed transparently from cloud storage backends
(e.g., S3, GCS) without requiring a local copy.

**Tradeoffs:**

* **itar** (single file) -- efficient for distribution, cloud storage, and
  atomic transfers; supports both sequential streaming and random access via
  the appended index
* **directory store** -- individual chunk files on disk; simpler for debugging
  and incremental updates

Both formats are accessed through the same
:class:`~ncore.data.v4.SequenceComponentGroupsReader` and
:class:`~ncore.data.v4.SequenceComponentGroupsWriter` APIs.

Component Group Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~

Each component group has the following root-level structure:

.. code-block:: text

   ncore4[-{component_group_name}].zarr[.itar]/
   │
   ├── {sequence_meta_data}
   │   ├── sequence_id: str
   │   ├── version: str (currently "v4")
   │   ├── sequence_timestamp_interval_us: {start, stop}
   │   ├── generic_meta_data: {...}
   │   └── component_group_name: str
   │
   └── {component_type}/
       └── {component_instance_name}/
            ├── {component_meta_data}
            │   ├── component_name: str
            │   ├── component_instance_name: str
            │   ├── component_version: str
            │   └── generic_meta_data: {...}
            │
            └── {component_specific_data}...

Poses Component
~~~~~~~~~~~~~~~

The poses component stores both static (time-invariant) and dynamic
(time-dependent) rigid transformations between named coordinate frames:

.. code-block:: text

   poses/
   └── {component_instance_name}/
       ├── static_poses/
       │   └── {attrs}
       │       └── ("source_frame", "target_frame"):
       │           ├── pose: [[4,4]] float32/64
       │           └── dtype: str
       │
       └── dynamic_poses/
           └── {attrs}
               └── ("source_frame", "target_frame"):
                   ├── poses: [[N,4,4]] float32/64
                   ├── timestamps_us: [N] uint64
                   └── dtype: str

For ego-vehicle trajectories, the rig-to-world transformation is typically
stored as a dynamic pose under the key ``("rig", "world")``. Transformations
from local world to global world frames (like ECEF) are represented by the
``("world", "world_global")`` record.

Static poses are used for sensor extrinsic calibrations. For example, a
camera-to-rig transformation would be stored under the key
``("camera_front_wide_120fov", "rig")``.

Intrinsics Component
~~~~~~~~~~~~~~~~~~~~

Camera and lidar intrinsic model parameters:

.. code-block:: text

   intrinsics/
   └── {component_instance_name}/
        ├── cameras/
        │   └── {camera_id}/
        │       └── {attrs}
        │           ├── camera_model_type: str
        │           ├── camera_model_parameters: {...}
        │           └── external_distortion_type: str  (optional)
       │
       └── lidars/
           └── {lidar_id}/
               └── {attrs}
                   ├── lidar_model_type: str
                   └── lidar_model_parameters: {...}

Model types include ``ftheta``, ``opencv-pinhole``, and ``opencv-fisheye`` for
camera sensors, and ``row-offset-spinning`` for lidar sensors. For detailed
model parameterizations and mathematical specifications, see
:ref:`sensor_models`.

Masks Component
~~~~~~~~~~~~~~~

Static masks for sensors are stored per sensor instance (currently only cameras
are supported):

.. code-block:: text

   masks/
   └── {component_instance_name}/
       └── cameras/
           └── {camera_id}/
               ├── {attrs}
               │   └── mask_names: [str, ...]
               └── {mask_name} () |Sx  (encoded image, attrs: format: str)

Sensor Components
~~~~~~~~~~~~~~~~~

Sensor components (cameras, lidars, radars) share a common frame-based structure:

.. code-block:: text

   {sensor_type}/
   └── {sensor_id}/
       ├── {component_meta_data}
       │
       └── frames/
           ├── {attrs}
           │   └── frames_timestamps_us: [N, 2] uint64  (start, end per frame)
           │
           └── {frame_name}/  (keyed by end-of-frame timestamp)
               ├── {sensor_specific_data}
               └── generic_data/
                   ├── {attrs: generic_meta_data}
                   └── {named datasets}...

*Camera Sensor Frames*:

.. code-block:: text

   cameras/{camera_id}/frames/{frame_name}/
   ├── image () |Sx  (encoded image, attrs: format: str)
   └── generic_data/...

*Lidar Sensor Frames*:

Lidar and radar data structures separate ray geometry (``ray_bundle/``) from
multi-return properties (``ray_bundle_returns/``) for flexible data organization.
Non-existing values are indicated via NaNs and must be consistent across all
return datasets to define a coherent ``[R,N]`` valid-return mask. This mask is
stored in bit-packed form as ``ray_bundle_returns_valid_mask_packed``.

.. code-block:: text

   lidars/{lidar_id}/frames/{frame_name}/
   ├── ray_bundle/
   │   ├── {attrs: n_rays: int}
   │   ├── direction: [N,3] float32     (per-ray normalized ray directions in sensor coordinates)
   │   ├── timestamp_us: [N] uint64     (per-ray timestamps of ray measurement time in us)
   │   └── model_element: [N,2] uint16  (optional: model-element indices of each ray)
   │
   ├── ray_bundle_returns/
   │   ├── {attrs: n_returns: int}
   │   ├── distance_m: [R,N] float32    (per-return measured metric distances along rays)
   │   ├── intensity: [R,N] float32     (per-return measured return intensity values [0,1])
   │   └── ...                          (may include additional return datasets)
   │
   └── ray_bundle_returns_valid_mask_packed () uint8  (bit-packed [R,N] valid mask, attrs: n_returns, n_rays)

*Radar Sensor Frames*:

.. code-block:: text

   radars/{radar_id}/frames/{frame_name}/
   ├── ray_bundle/
   │   ├── {attrs: n_rays: int}
   │   ├── direction: [N,3] float32  (per-ray normalized ray directions in sensor coordinates)
   │   └── timestamp_us: [N] uint64  (per-ray timestamps of ray measurement time in us)
   │
   ├── ray_bundle_returns/
   │   ├── {attrs: n_returns: int}
   │   ├── distance_m: [R,N] float32  (per-return measured metric distances along rays)
   │   └── ...                        (may include radial velocities, RCS)
   │
   └── ray_bundle_returns_valid_mask_packed () uint8  (bit-packed [R,N] valid mask, attrs: n_returns, n_rays)

Cuboids Component
~~~~~~~~~~~~~~~~~

3D cuboid track observations are stored in a structured format:

.. code-block:: text

   cuboids/
   └── {component_instance_name}/
       └── cuboids/
           └── {attrs}
               └── cuboid_track_observations: [N] (JSON-serialized list)

Each observation is a JSON-serializable object containing:

* ``track_id`` - Unique track identifier (str)
* ``class_id`` - Object class label (str)
* ``timestamp_us`` - Observation timestamp in us (int)
* ``reference_frame_id`` - Reference frame identifier (str)
* ``reference_frame_timestamp_us`` - Reference frame timestamp in us (int)
* ``bbox3`` - 3D bounding box in reference frame coordinates
* ``source`` - Label source (e.g., :attr:`~ncore.data.LabelSource.AUTOLABEL`,
  :attr:`~ncore.data.LabelSource.GT_SYNTHETIC`)
* ``source_version`` - Optional source version identifier (str)

Observations can be transformed between reference frames using the pose
graph and support motion compensation across different sensor frames.

Component Groups
~~~~~~~~~~~~~~~~

Multiple component instances can coexist using different *component instance names*.
This enables scenarios such as:

* Multiple calibrations (e.g., "factory", "online_refined")
* Multiple label sources (e.g., "auto_labels", "human_verified")
* Different processing versions (e.g., "v1", "v2")

The default component group name is ``default``. Component stores with different
group names are stored in separate zarr archives following the naming pattern:
``ncore4-{component_group_name}.zarr[.itar]``.


Custom Components
~~~~~~~~~~~~~~~~~

The component architecture is extensible: define a
:class:`~ncore.data.v4.ComponentWriter` / :class:`~ncore.data.v4.ComponentReader`
pair with a unique ``COMPONENT_NAME`` and version string, then register
instances through :class:`~ncore.data.v4.SequenceComponentGroupsWriter`.

To avoid name clashes with built-in or third-party components, use a
reverse-domain naming convention for custom component names, e.g.
``com.myorg.velocity``.

A minimal custom component looks like:

.. code-block:: python

   from ncore.data.v4 import ComponentWriter, ComponentReader

   class VelocityComponent:
       COMPONENT_NAME = "com.myorg.velocity"

       class Writer(ComponentWriter):
           @staticmethod
           def get_component_name() -> str:
               return VelocityComponent.COMPONENT_NAME

           @staticmethod
           def get_component_version() -> str:
               return "v1"

           def store_velocity(self, velocity, timestamp_us):
               ...  # collect data

           def finalize(self):
               ...  # write zarr datasets to self._group

       class Reader(ComponentReader):
           @staticmethod
           def get_component_name() -> str:
               return VelocityComponent.COMPONENT_NAME

           @staticmethod
           def supports_component_version(version: str) -> bool:
               return version == "v1"

           def get_velocities(self):
               return self._group["velocities"][:], self._group["timestamps_us"][:]

Writers must ensure that all stored timestamps fall within the sequence's
``sequence_timestamp_interval_us`` time range. Existing datasets can be extended
with new components by creating a writer via
:meth:`SequenceComponentGroupsWriter.from_reader() <ncore.data.v4.SequenceComponentGroupsWriter.from_reader>`
and finalizing the additional stores. For a complete working example (including
component versioning and backward-compatible readers), see
`TestDataNewComponent <https://github.com/NVIDIA/ncore/blob/main/ncore/impl/data/v4/components_test.py>`_.

Loading V4 Data
~~~~~~~~~~~~~~~

V4 sequences are loaded by specifying one or more component store paths:

.. code-block:: python

   from ncore.data.v4 import SequenceComponentGroupsReader
   from pathlib import Path
   
   # Load sequence from multiple component stores
   reader = SequenceComponentGroupsReader([
       Path("ncore4.zarr.itar"),           # default components
       Path("ncore4-calibv2.zarr.itar"),   # alternative calibration
   ])
   
   # Access specific components
   poses_readers = reader.open_component_readers(PosesComponent.Reader)
   camera_readers = reader.open_component_readers(CameraSensorComponent.Reader)


