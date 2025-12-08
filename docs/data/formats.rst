.. Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

.. _data_formats:


Data Formats
============

This document describes the NCORE data format specifications for storing
sensor data, poses, calibrations, and annotations.


Data Format Versions
--------------------

NCORE supports two data format versions, each optimized for different use cases:

**V3 (Shard-based Format)** - The established production-verified format in which sequences
are partitioned into self-contained shard archives. Each shard is a monolithic unit
containing all sensor data, poses, and labels for a portion of a sequence, stored as
zarr groups within indexed tar archives (``.itar`` files). This format is optimal for:

* Sequential processing of large datasets consisting of multiple shards
* Backwards compatibility with existing pipelines
* Simple deployment in which all data is accessed together

**V4 (Component-based Format)** - The next-generation modular format that
separates data into independent component stores. Each component (poses,
intrinsics, sensors, labels, etc.) is stored as a separate zarr store that can
be independently managed, versioned, and combined. This format enables:

* Flexible data composition from multiple sources
* Independent component updates without reprocessing entire sequences
* Parallel access and distributed storage optimization
* Extensibility through custom component types
* Fine-grained access control and data sharing

Both formats use the same coordinate system conventions and transformations described
in :ref:`data_conventions`. The choice between V3 and V4 depends on workflow requirements,
with V4 recommended for new development requiring modularity and scalability.


.. _v3-data-format:

V3: Shard Data Hierarchy (Shard-Based Format)
----------------------------------------------

The per-shard data converted using NCORE V3 converters is represented as
*self-contained* shard archives [#f1]_.

Within each shard archive the data is represented in the following
dataset hierarchy [#f2]_:

.. code-block:: text

   /
   │
   ├── {generic_meta_data}...
   │
   ├── poses
   │   ├── T_rig_world_base (4, 4) float64
   │   ├── T_rig_world_timestamps_us (I,) uint64
   │   └── T_rig_worlds (I, 4, 4) float64
   │
   ├── {labels}
   │   └── track_labels () object
   │
   └── sensors
      ├── cameras
      │   ├── camera_front_wide_120fov
      │   ⁞   ├── frame_timestamps_us (J,) uint64
      │   ⁞   │
      │   ⁞   ├── {mask} () |Sx
      │   ⁞   │
      │   ⁞   ├── {generic_meta_data}...
      │   ⁞   │
      │   ⁞   ├── 000000
      │   ⁞   │   ├── T_rig_worlds (2, 4, 4) float32
      │   ⁞   │   ├── timestamps_us (2,) uint64
      │   ⁞   │   │
      │   ⁞   │   ├── image () |Sx
      │   ⁞   │   │      
      │   ⁞   │   ├── {generic_data}...
      │   ⁞   │   └── {generic_meta_data}...
      │   ⁞   │
      │   ⁞   └── 000001...
      │
      ├── lidars
      │   ├── lidar_gt_top_p128_v4p5
      │   ⁞   ├── frame_timestamps_us (K,) uint64
      │   ⁞   │
      │   ⁞   ├── {generic_meta_data}...
      │   ⁞   │
      │   ⁞   ├── 000000
      │   ⁞   │   ├── T_rig_worlds (2, 4, 4) float32
      │   ⁞   │   ├── timestamps_us (2,) uint64
      │   ⁞   │   │
      │   ⁞   │   ├── frame_labels (M,) object
      │   ⁞   │   │
      │   ⁞   │   ├── dynamic_flag (N,) int8 [deprecated - see footnote 4]
      │   ⁞   │   ├── intensity (N,) float32
      │   ⁞   │   ├── timestamp_us (N,) uint64
      │   ⁞   │   ├── xyz_e (N, 3) float32
      │   ⁞   │   ├── xyz_s (N, 3) float32
      │   ⁞   │   │
      │   ⁞   │   ├── {generic_data}...
      │   ⁞   │   └── {generic_meta_data}...
      │   ⁞   │
      │   ⁞   └── 000001...
      │
      └── {radars}
          ├── radar_corner_front_left
          ⁞   ├── frame_timestamps_us (K,) uint64
          ⁞   │
          ⁞   ├── {generic_meta_data}...
          ⁞   │
          ⁞   └── 000000...

For instance, the dataset representing the encoded image of the first
frame of the camera with ID ``camera_front_wide_120fov`` is referenced
by the path ``/sensors/cameras/camera_front_wide_120fov/000000/image``
within this dataset hierarchy.

Pose Data
~~~~~~~~~

The timestamped trajectory of the rig frame is stored in the dataset
``/poses/T_rig_worlds`` as SE3 transformation matrices from the rig to
the local world coordinate system (float64, [n,4,4]) with associated
timestamps ``/poses/T_rig_world_timestamps_us`` (in microseconds
(uint64, [n])).

All of these poses are relative to the single base pose stored in the
dataset ``/poses/T_rig_world_base`` (float64, [4,4], see details in
:ref:`data_conventions`).

Sensor Data
~~~~~~~~~~~

**Images**

Camera-associated image data is saved either in binary `jpeg` (raw
sensor data) or in the `png` (masks / semantic segmentations) formats.
The concrete format of image data is available in the dataset
attributes, e.g.,
``/sensors/cameras/camera_front_wide_120fov/mask.attrs['format']`` is
the image format of the static mask associated with the camera
``camera_front_wide_120fov``.

**Point clouds**

Lidar and radar point clouds are saved as compressed and named per-point
datasets per frame.

Lidar data contains the following columns:

* ``xyz_s`` - 3D coordinate of the *start* of the ray in the sensor's
  end-of-spin reference frame (float32, [n,3])
* ``xyz_e`` - Motion-compensated 3D coordinate of the *end* of the ray
  in the sensor's end-of-spin reference frame (float32, [n,3])
* ``intensity`` - measured intensity (float32, [n]): normalized to
  ``[0.0, 1.0]`` range
* ``dynamic_flag`` - dynamic flag (int8, [n]): ``-1`` if not evaluated,
  ``0`` for static points, and ``1`` for dynamic points. Note: points
  are classified as dynamic if the associated object-track is dynamic
  at *any* point in time of the sequence [deprecated [#f3]_]
* ``timestamp_us`` - point timestamp in microseconds (uint64, [n])

Radar data contains the following columns:

* ``xyz_s`` - 3D coordinate of the start of the ray in the sensor's
  end-of-frame reference frame (float32, [n,3])
* ``xyz_e`` - 3D coordinate of the end of the ray in the sensor's
  end-of-frame reference frame (float32, [n,3])

Additional radar readings (like radial velocities / radar-cross-section
measurements) are exposed in dataset-specific generic per-frame data.

**Metadata**

Metadata is available per frame, but also for individual sensors and for
the general sequence.

Per-frame metadata contains the following datasets for all sensors and
all frames:

* ``timestamps_us`` - timestamps of the frame's start and end point in
  microseconds (uint64, [2,])
* ``T_rig_worlds`` - SE3 transformation matrices from the rig to the
  world coordinate system at the start and end timestamp of the frame
  (float32, [2,4,4])
* ``generic_data`` - a group with non-enforced dataset members
  (usually data-source dependent, encoding data-source specific
  per-frame data not common to all data-sources)

For individual sensor types, we also save sequence-wide sensor-specific
meta data either as attributes of the sensor-group or individual datasets:

*All Sensors*:

* ``T_sensor_rig``- SE3 transformation matrix from the sensor to the rig
  coordinate system (float32, [4,4])
* ``frame_timestamps_us`` - end-of-frame timestamps of all the sensor's
  frames in microseconds (uint64, [J,])
* ``generic_meta_data`` - a group with non-enforced members
  (usually data-source dependent, encoding data-source specific sensor
  meta data not common to all data-sources)

*Cameras*:

* ``camera_model_type`` - camera model type (str, one of [ftheta,
  opencv-pinhole, opencv-fisheye])

The field ``camera_model_parameters`` will unconditionally contain:

* ``resolution`` - width and height of the image in pixels (uint32,
  [2,])
* ``shutter_type`` - shutter type of the camera's imaging sensor (str, one of
  [ROLLING_TOP_TO_BOTTOM, ROLLING_LEFT_TO_RIGHT, ROLLING_BOTTOM_TO_TOP,
  ROLLING_RIGHT_TO_LEFT, GLOBAL])

If ``camera_model_type = 'ftheta'`` the following intrinsic parameters
will additionally be available in ``camera_model_parameters``:

* ``principal_point`` - u and v coordinate of the principal point,
  following the NVIDIA default convention for FTheta camera models
  in which the pixel indices represent the center of the pixel
  (not the top-left corners). NOTE: principal point coordinates
  will be adapted internally in camera model APIs to reflect
  the :ref:`image coordinate conventions
  <image_coordinate_conventions>` (float32, [2,])

* ``reference_poly`` - indicating which of the two polynomials is the
  *reference* polynomial - the other polynomial is only an approximation
  of the inverse of the reference polynomial (str, one of
  [PIXELDIST_TO_ANGLE, ANGLE_TO_PIXELDIST])
* ``pixeldist_to_angle_poly`` - coefficients of the backward distortion
  polynomial (conditionally approximate, depending on
  ``reference_poly``), mapping pixel-distances to angles [rad] (float32,
  [6,])
* ``angle_to_pixeldist_poly`` - coefficients of the forward distortion
  polynomial (conditionally approximate, depending on
  ``reference_poly``), mapping angles [rad] to pixel-distances (float32,
  [6,])
* ``max_angle`` - maximal extrinsic ray angle [rad] with the principal
  direction (float32)
* ``linear_cde`` - Coefficients of the constrained linear term
  :math:`\begin{bmatrix} c & d \\ e & 1 \end{bmatrix}` transforming between
  sensor coordinates (in mm) to image coordinates (in px) (float32, [3,])

If ``camera_model_type` = 'opencv-pinhole'`` the following intrinsic parameters
will additionally be available in ``camera_model_parameters``:

* ``principal_point`` - u and v coordinate of the principal point,
  following the :ref:`image coordinate conventions
  <image_coordinate_conventions>` (float32, [2,])
* ``focal_length`` - focal lengths in u and v direction, resp., mapping
  (distorted) normalized camera coordinates to image coordinates relative
  to the principal point (float32, [2,])
* ``radial_coeffs`` - radial distortion coefficients
  ``[k1,k2,k3,k4,k5,k6]`` parameterizing the rational radial distortion
  factor :math:`\frac{1 + k_1r^2 + k_2r^4 + k_3r^6}{1 + k_4r^2 + k_5r^4
  + k_6r^6}` for squared norms :math:`r^2` of normalized camera
  coordinates (float32, [6,])
* ``tangential_coeffs`` - tangential distortion coefficients ``[p1,p2]``
  parameterizing the tangential distortion components
  :math:`\begin{bmatrix} 2p_1x'y' + p_2 \left(r^2 + 2{x'}^2 \right) \\
  p_1 \left(r^2 + 2{y'}^2 \right) + 2p_2x'y' \end{bmatrix}` for
  normalized camera coordinates :math:`\begin{bmatrix} x' \\ y'
  \end{bmatrix}` (float32, [2,])
* ``thin_prism_coeffs`` - thins prism distortion coefficients
  ``[s1,s2,s3,s4]`` parameterizing the thin prism distortion components
  :math:`\begin{bmatrix} s_1r^2 + s_2r^4 \\ s_3r^2 + s_4r^4
  \end{bmatrix}` for squared norms :math:`r^2` of normalized camera
  coordinates (float32, [4,])

If ``camera_model_type` = 'opencv-fisheye'`` the following intrinsic parameters
will additionally be available in ``camera_model_parameters``:

* ``principal_point`` - u and v coordinate of the principal point,
  following the :ref:`image coordinate conventions
  <image_coordinate_conventions>` (float32, [2,])
* ``focal_length`` - focal lengths in u and v direction, resp., mapping
  (distorted) normalized camera coordinates to image coordinates relative
  to the principal point (float32, [2,])
* ``radial_coeffs`` - radial distortion coefficients representing
  OpenCV-style ``[k1,k2,k3,k4]`` parameters of the
  fisheye distortion polynomial :math:`\theta(1 + k_1\theta^2 +
  k_2\theta^4 + k_3\theta^6 + k_4\theta^8)` for extrinsic camera ray
  angles :math:`\theta` with the principal direction (float32, [4,])
* ``max_angle`` - maximal extrinsic ray angle [rad] with the principal
  direction (float32)

Finally, we also save general metadata related to the session (input
data and versioning):

* ``version`` - version of the dataset (str)
* ``egomotion_type`` - type of ego-motion that was used to generate the
  data (str)
* ``calibration_type`` - type of sensor calibration that was used to
  generate the data (str)
* ``camera_ids`` / ``lidar_ids`` / ``radar_ids`` - individual lists of
  sensor names processed available in the data ([str])
* ``sequence_id`` - the identifier of the source dataset, if available
  (str)
* ``shard_id`` - the index of the current shard within all shards
  generated for a given sequence
* ``shard_count`` - the total number of shards generated for a given
  sequence

Labels
~~~~~~

**4D autolabels**

Annotation data is stored in a segmented format:

* All observations of each unique track instance is stored in a
  session-wide format in the ``/labels/track_labels`` dataset by
  *references* to the individual sensor frames that are observing the
  track's instance.
* Individual time-associated instances of each track observation (e.g.,
  3D bounding boxes in lidar frames or 2D bounding boxes in camera
  frames) are stored in the per-frame meta data of each sensor's frame.
  For instance, the dataset
  ``/sensors/lidars/lidar_gt_top_p128_v4p5/000000/frame_labels`` contains
  the observed track instances in the first frame of a given lidar.


.. _v4-data-format:

V4: Component Store Hierarchy (Component-Based Format)
-------------------------------------------------------

The component-based V4 data format represents sequences as collections of
*component groups* [#f4]_. Unlike V3's monolithic shards, V4
distributes data across modular components that can be independently managed,
versioned, and combined to form virtual sequences.

Component Architecture
~~~~~~~~~~~~~~~~~~~~~~

Each component group is a zarr store (either a ``.zarr.itar`` archive or a
directory-based ``.zarr`` store) containing a specific number of data component
instances. The NCore library provides the following default component types:

* **PosesComponent** - Static and dynamic pose transformations between named
  coordinate frames
* **IntrinsicsComponent** - Camera and lidar intrinsic calibration parameters
* **MasksComponent** - Static masks associated with sensors
* **CameraSensorComponent** - Camera frame data including images
* **LidarSensorComponent** - Lidar frame data including point clouds
* **RadarSensorComponent** - Radar frame data including detections
* **CuboidsComponent** - 3D cuboid track observations and annotations

The component architecture is extensible, allowing custom component types to be
defined for application-specific data.

Component Group Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~

Each component group has the following root-level structure:

.. code-block:: text

   ncore4[-{component_group_name}].zarr[.itar]/
   │
   ├── {sequence_meta_data}
   │   ├── sequence_id: str
   │   ├── version: str (currently "4.0")
   │   ├── sequence_timestamp_interval_us: {start, stop}
   │   ├── generic_meta_data: {...}
   │   └── component_group_name: str
   │
   └── {component_type}/
       └── {component_instance_name}/
           ├── {component_meta_data}
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
stored as a dynamic pose under the key ``("rig", "world")``, with the base
world coordinate frame maintained for consistency with V3. transformations
from local world to global world frames (like ECEF) are
represented by the ``("world", "world_global")`` record.

Static poses are used for sensor extrinsic calibrations. For example, a
camera-to-rig transformation would be stored under the key
``("camera_front_wide_120fov", "rig")``, analogous to the ``T_sensor_rig``
calibration in the V3 format.

Intrinsics Component
~~~~~~~~~~~~~~~~~~~~

Camera and lidar intrinsic model parameters follow the same specifications as V3:

.. code-block:: text

   intrinsics/
   └── {component_instance_name}/
       ├── cameras/
       │   └── {camera_id}/
       │       └── {attrs}
       │           ├── camera_model_type: str
       │           └── camera_model_parameters: {...}
       │
       └── lidars/
           └── {lidar_id}/
               └── {attrs}
                   ├── lidar_model_type: str
                   └── lidar_model_parameters: {...}

Model types and parameters are identical to those specified in the V3 format
(`ftheta`, `opencv-pinhole`, `opencv-fisheye` for camera sensors;
`row-offset-spinning` for lidar sensors).

Masks Component
~~~~~~~~~~~~~~~

Static masks for sensors are stored per sensor instance (currently only cameras
are supported):

.. code-block:: text

   masks/
   └── {component_instance_name}/
       └── cameras/
           └── {camera_id}/
               └── {mask_name} () |Sx  (encoded image)

Sensor Components
~~~~~~~~~~~~~~~~~

Sensor components (cameras, lidars, radars) share a common frame-based structure:

.. code-block:: text

   {sensor_type}/
   └── {sensor_id}/
       ├── {sensor_meta_data}
       │   └── generic_meta_data: {...}
       │
       ├── frames_timestamps_us: [N] uint64
       │
       └── frames/
           └── {frame_name}/  (using end-of-frame timestamps)
               ├── timestamps_us: [2] uint64  (start, end)
               ├── {sensor_specific_data}
               ├── {generic_data}/...
               └── {generic_meta_data}

*Camera Sensor Frames*:

.. code-block:: text

   cameras/{camera_id}/frames/{frame_name}/
   ├── image () |Sx  (encoded jpeg/png)
   └── {generic_data}/...

*Lidar Sensor Frames*:

Lidar and radar data structures separate ray geometry (ray_bundle) from multi-return properties
(ray_bundle_returns) for flexible data organization. Non-existing values are indicated via NaNs.

.. code-block:: text

   lidars/{lidar_id}/frames/{frame_name}/
   ├── ray_bundle/
   │   ├── direction: [N,3] float32  (normalized ray directions)
   │   ├── timestamp_us: [N] uint64  (timestamps of ray measurement time)
   │   └── {generic_data}/...
   │
   └── [ray_bundle_returns]
       ├── distance: [N,3] float32  (measured distances along rays)
       ├── intensity: [N] float32
       └── {generic_data}/...

*Radar Sensor Frames*:

.. code-block:: text

   radars/{radar_id}/frames/{frame_name}/
   ├── ray_bundle/
   │   ├── direction: [N,3] float32  (normalized ray directions)
   │   ├── timestamp_us: [N] uint64  (timestamps of ray measurement time)
   │   └── {generic_data}/...  
   │
   └── [ray_bundle_returns]
       ├── distance: [N,3] float32  (measured distances along rays)
       └── {generic_data}/...       (may include radial velocities, RCS)

Cuboids Component
~~~~~~~~~~~~~~~~~

3D cuboid track observations are stored in a structured format:

.. code-block:: text

   cuboids/
   └── {component_instance_name}/
       └── observations: [N] object

Each observation is a JSON-serializable object containing:

* ``track_id`` - Unique track identifier (str)
* ``class_id`` - Object class label (str)
* ``timestamp_us`` - Observation timestamp (int)
* ``reference_frame_id`` - Reference frame identifier (str)
* ``reference_frame_timestamp_us`` - Reference frame timestamp (int)
* ``bbox3`` - 3D bounding box in reference frame coordinates
* ``source`` - Label source (e.g., AUTOLABEL, GT_SYNTHETIC)
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


Loading V4 Data
~~~~~~~~~~~~~~~

V4 sequences are loaded by specifying one or more component store paths:

.. code-block:: python

   from ncore.unstable.data.v4 import SequenceComponentStoreReader
   from pathlib import Path
   
   # Load sequence from multiple component stores
   reader = SequenceComponentStoreReader([
       Path("ncore4.zarr.itar"),           # default components
       Path("ncore4-calibv2.zarr.itar"),   # alternative calibration
   ])
   
   # Access specific components
   poses_readers = reader.open_component_readers(PosesComponent.Reader)
   camera_readers = reader.open_component_readers(CameraSensorComponent.Reader)

Compatibility and Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

V3 data can be converted to V4 format using the `ncore_3to4` conversion
utilities, which preserve all sensor data, poses, intrinsics, and annotations
while reorganizing them into the component-based structure. The conversion
supports selective sensor and time range extraction, component group assignment,
and preservation of all metadata and calibrations.


.. rubric:: Footnotes

.. [#f1] NCore V3 data shards are represented by
         `zarr <https://zarr.readthedocs.io/en/stable/>`_
         groups within a custom ``.itar`` archive format that can be
         loaded most easily using the ``ShardDataLoader`` type, which
         also supports shard concatenation to reconstruct full source
         sequences.
.. [#f2] Curly brackets denote optional data.
.. [#f3] Deprecated property, might be available as generic frame
         data still
.. [#f4] NCore V4 component stores are represented by
         `zarr <https://zarr.readthedocs.io/en/stable/>`_ groups within
         either a custom ``.zarr.itar`` archive format or plain directory
         stores. The ``SequenceComponentStoreReader`` and
         ``SequenceComponentStoreWriter`` types provide the primary APIs for
         loading and creating V4 data.
