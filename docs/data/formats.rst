.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION
.. SPDX-License-Identifier: Apache-2.0

.. _data_formats:


Data Formats
============

This document describes the NCore data format specifications for storing
sensor data, poses, calibrations, and annotations.


Data Format
-----------

NCore current data format version is **V4 (Component-based Format)** - a modular format that
separates data into independent component stores. Each component (poses,
intrinsics, sensors, labels, etc.) is stored as a separate zarr component that can
be independently managed, versioned, and combined. This format enables:

* Flexible data composition from multiple sources
* Independent component updates without reprocessing entire sequences
* Parallel access and distributed storage optimization
* Extensibility through custom component types
* Fine-grained access control and data sharing

The format uses coordinate system conventions and transformations described
in :ref:`data_conventions`.

.. _v4-data-format:

V4: Component Store Hierarchy (Component-Based Format)
-------------------------------------------------------

The component-based V4 data format represents sequences as collections of
*component groups* [#f1]_. V4 distributes data across modular components
that can be independently managed, versioned, and combined to form virtual
sequences.

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
       │           └── camera_model_parameters: {...}
       │
       └── lidars/
           └── {lidar_id}/
               └── {attrs}
                   ├── lidar_model_type: str
                   └── lidar_model_parameters: {...}

Model types include `ftheta`, `opencv-pinhole`, and `opencv-fisheye` for
camera sensors, and `row-offset-spinning` for lidar sensors.

.. _camera_model_parameterizations:

Camera Model Parameterizations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Camera models may optionally include external distortion parameters to account for distortion sources outside the camera itself (e.g., windshields). See :ref:`external_distortion_models` for details.

The ``camera_model_parameters`` dictionary will unconditionally contain:

* ``resolution`` - width and height of the image in pixels (uint32, [2,])
* ``shutter_type`` - shutter type of the camera's imaging sensor (str, one of
  [ROLLING_TOP_TO_BOTTOM, ROLLING_LEFT_TO_RIGHT, ROLLING_BOTTOM_TO_TOP,
  ROLLING_RIGHT_TO_LEFT, GLOBAL])

.. _ftheta_camera_model:

**FTheta Camera Model**

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

**Mathematical Model:**

The FTheta model projects 3D camera rays to image coordinates through the following steps:

1. **Ray to Angle**: Convert camera ray direction [x, y, z] to angle θ:
   
   .. math::
   
      \theta = \arctan2(\sqrt{x^2 + y^2}, z)

2. **Angle to Pixel Distance**: Apply the forward polynomial f(θ):
   
   .. math::
   
      \delta = f(\theta) = c_0 + c_1\theta + c_2\theta^2 + c_3\theta^3 + c_4\theta^4 + c_5\theta^5

3. **Linear Transformation**: Apply constrained linear transformation A and add principal point:
   
   .. math::
   
      \begin{bmatrix} u \\ v \end{bmatrix} = 
      \begin{bmatrix} c & d \\ e & 1 \end{bmatrix}
      \frac{\delta}{\sqrt{x^2 + y^2}}
      \begin{bmatrix} x \\ y \end{bmatrix} + 
      \begin{bmatrix} u_0 \\ v_0 \end{bmatrix}

The inverse operation uses the backward polynomial (either reference or approximate inverse).

**Example:** For a typical FTheta model, both forward and backward polynomials contain 6 coefficients each (float32, [6,]).

.. _opencv_pinhole_camera_model:

**OpenCV Pinhole Camera Model**

If ``camera_model_type = 'opencv-pinhole'`` the following intrinsic parameters
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

**Mathematical Model:**

The OpenCV Pinhole model applies radial, tangential, and thin prism distortions:

1. **Normalize**: Convert camera ray [x, y, z] to normalized coordinates:
   
   .. math::
   
      x' = \frac{x}{z}, \quad y' = \frac{y}{z}, \quad r^2 = {x'}^2 + {y'}^2

2. **Radial Distortion**: Apply rational radial distortion factor:
   
   .. math::
   
      d_r = \frac{1 + k_1r^2 + k_2r^4 + k_3r^6}{1 + k_4r^2 + k_5r^4 + k_6r^6}

3. **Tangential Distortion**:
   
   .. math::
   
      \begin{bmatrix} \Delta x_t \\ \Delta y_t \end{bmatrix} = 
      \begin{bmatrix} 
      2p_1x'y' + p_2(r^2 + 2{x'}^2) \\ 
      p_1(r^2 + 2{y'}^2) + 2p_2x'y' 
      \end{bmatrix}

4. **Thin Prism Distortion**:
   
   .. math::
   
      \begin{bmatrix} \Delta x_p \\ \Delta y_p \end{bmatrix} = 
      \begin{bmatrix} 
      s_1r^2 + s_2r^4 \\ 
      s_3r^2 + s_4r^4 
      \end{bmatrix}

5. **Project to Image**: Apply focal length and principal point:
   
   .. math::
   
      \begin{bmatrix} u \\ v \end{bmatrix} = 
      \begin{bmatrix} f_u(x'd_r + \Delta x_t + \Delta x_p) + u_0 \\ 
      f_v(y'd_r + \Delta y_t + \Delta y_p) + v_0 \end{bmatrix}

**Example:** Typical parameter sizes: radial_coeffs [6,], tangential_coeffs [2,], thin_prism_coeffs [4,], all float32.

.. _opencv_fisheye_camera_model:

**OpenCV Fisheye Camera Model**

If ``camera_model_type = 'opencv-fisheye'`` the following intrinsic parameters
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

**Mathematical Model:**

The OpenCV Fisheye model uses a fisheye distortion polynomial:

1. **Ray to Angle**: Convert camera ray direction [x, y, z] to angle θ and normalized direction:
   
   .. math::
   
      \theta = \arctan2(\sqrt{x^2 + y^2}, z)
   
   .. math::
   
      x_n = \frac{x}{\sqrt{x^2 + y^2}}, \quad y_n = \frac{y}{\sqrt{x^2 + y^2}}

2. **Fisheye Distortion Polynomial**: Apply distortion to angle:
   
   .. math::
   
      \delta(\theta) = \theta(1 + k_1\theta^2 + k_2\theta^4 + k_3\theta^6 + k_4\theta^8)

3. **Project to Image**: Scale by focal length and add principal point:
   
   .. math::
   
      \begin{bmatrix} u \\ v \end{bmatrix} = 
      \begin{bmatrix} f_u \delta(\theta) x_n + u_0 \\ 
      f_v \delta(\theta) y_n + v_0 \end{bmatrix}

The inverse operation uses Newton-Raphson iteration to invert the ninth-degree polynomial. **Example:** The radial_coeffs parameter contains [k₁, k₂, k₃, k₄] (float32, [4,]).

.. _external_distortion_models:

External Distortion Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Camera models can optionally include external distortion sources that affect rays before they reach the camera lens (e.g., windshields). If present, the ``camera_model_parameters`` dictionary will contain:

* ``external_distortion_parameters`` - parameters for the external distortion model (optional, may be None)

When ``external_distortion_parameters`` is present, it will contain:

* ``external_distortion_type`` - type of external distortion (str)

.. _bivariate_windshield_model:

**Bivariate Windshield Model**

If ``external_distortion_type = 'bivariate-windshield'``, this models optical distortion caused by a vehicle's windshield. This model is only applicable when the camera's entire field of view projects through the windshield.

**Mathematical Model:**

The distortion operates on spherical angle representations of sensor rays with direction [x, y, z]:

1. **Convert to Spherical Angles**:
   
   .. math::
   
      \phi = \arcsin\left(\frac{x}{\sqrt{x^2 + y^2 + z^2}}\right) \quad \text{(horizontal angle)}
   
   .. math::
   
      \theta = \arcsin\left(\frac{y}{\sqrt{x^2 + y^2 + z^2}}\right) \quad \text{(vertical angle)}

2. **Apply Bivariate Polynomial Distortion**:
   
   Distortion is applied via separate polynomials for horizontal and vertical components:
   
   .. math::
   
      \phi' = P_h(\phi, \theta)
   
   .. math::
   
      \theta' = P_v(\phi, \theta)
   
   Where each polynomial has the bivariate form (example for order N=2):
   
   .. math::
   
      P(\phi, \theta) = c_0 + c_1\phi + c_2\phi^2 + (c_3 + c_4\phi)\theta + c_5\theta^2
   
   General form for order N:
   
   .. math::
   
      P(\phi, \theta) = \sum_{j=0}^{N} \left(\sum_{i=0}^{N-j} c_{i,j}\phi^i\right)\theta^j

3. **Reconstruct Distorted Ray**:
   
   .. math::
   
      [x', y', z'] = \left[\sin(\phi'), \sin(\theta'), \sqrt{1 - \sin^2(\phi') - \sin^2(\theta')}\right]

**Parameters:**

The following parameters will be available in ``external_distortion_parameters``:

* ``reference_poly`` - indicates which polynomial is the reference (str, one of [FORWARD, BACKWARD]). The reference polynomial is exact; the other is an approximate inverse computed during calibration.
* ``horizontal_poly`` - coefficients for forward horizontal distortion polynomial P_h (float32, [(N+1)·(N+2)/2,])
* ``vertical_poly`` - coefficients for forward vertical distortion polynomial P_v (float32, [(M+1)·(M+2)/2,])
* ``horizontal_poly_inverse`` - coefficients for backward horizontal distortion (float32, [(N+1)·(N+2)/2,])
* ``vertical_poly_inverse`` - coefficients for backward vertical distortion (float32, [(M+1)·(M+2)/2,])

**Example:** For a 2nd-order bivariate polynomial (N=2), the coefficient arrays contain 6 elements: [c₀, c₁, c₂, c₃, c₄, c₅]. For a 3rd-order polynomial (N=3), they contain 10 elements. The number of coefficients for order N is given by: (N+1)·(N+2)/2

**Direction:**

- **Forward distortion** (distort_camera_rays): transforms rays from external (world) to internal (camera sensor)
- **Backward distortion** (undistort_camera_rays): transforms rays from internal (camera sensor) to external (world)

.. _lidar_model_parameterizations:

Lidar Model Parameterizations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``lidar_model_parameters`` dictionary contains model-specific parameters for lidar sensors. Currently, the supported model type is ``row-offset-spinning``.

.. _row_offset_spinning_lidar_model:

**Row-Offset Structured Spinning Lidar Model**

If ``lidar_model_type = 'row-offset-spinning'``, this models structured spinning lidar sensors with per-row azimuth offsets (compatible with sensors like Hesai P128).

**Mathematical Model:**

The Row-Offset Structured Spinning Lidar model parameterizes lidar returns using per-row elevation angles and per-column azimuth angles with row-specific offsets.

**Notation:**

- α (alpha): elevation angle in radians
- β (beta): azimuth angle in radians
- E: row_elevations_rad array - elevation angles per row
- B: column_azimuths_rad array - base azimuth angles per column
- Δ (Delta): row_azimuth_offsets_rad array - azimuth offsets per row
- i: row index
- j: column index

1. **Element to Angles**: For element (i, j):
   
   .. math::
   
      \alpha = E[i]
   
   .. math::
   
      \beta = B[j] + \Delta[i]

2. **Angles to Sensor Ray**: Convert spherical angles to 3D ray direction:
   
   .. math::
   
      x = \cos(\beta) \cdot \cos(\alpha)
   
   .. math::
   
      y = \sin(\beta) \cdot \cos(\alpha)
   
   .. math::
   
      z = \sin(\alpha)

3. **Sensor Ray to Angles**: Convert 3D ray [x, y, z] to angles:
   
   .. math::
   
      \alpha = \arcsin(z)
   
   .. math::
   
      \beta = \arctan2(y, x)

**Parameters:**

The following parameters will be available in ``lidar_model_parameters``:

* ``row_elevations_rad`` - elevation angle for each row (float32, [n_rows,])
* ``column_azimuths_rad`` - base azimuth angle for each column (float32, [n_columns,])
* ``row_azimuth_offsets_rad`` - azimuth offset for each row (float32, [n_rows,])
* ``spinning_frequency_hz`` - rotation frequency in Hz (float32)
* ``spinning_direction`` - rotation direction (str, one of ['cw', 'ccw'])
* ``n_rows`` - number of vertical laser channels (int)
* ``n_columns`` - number of azimuth divisions per revolution (int)

**Example:** A Hesai P128 lidar would have n_rows=128 (for 128 laser channels), with row_elevations_rad containing 128 elevation angles and row_azimuth_offsets_rad containing 128 azimuth offsets.

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

Lidar and radar data structures separate ray geometry (ray_bundle) from
multi-return properties (ray_bundle_returns) for flexible data organization.
Non-existing values are indicated via NaNs and need to be consistent accross all
individual return signals to define a consistent [R,N] mask of valid returns.

.. code-block:: text

   lidars/{lidar_id}/frames/{frame_name}/
   ├── ray_bundle/
   │   ├── direction: [N,3] float32     (per-ray normalized ray directions in sensor coordinates)
   │   ├── timestamp_us: [N] uint64     (per-ray timestamps of ray measurement time in us)
   │   ├── model_element: [N,2] uint16  (optional: model-element indices of each ray)
   │   └── {generic_data}/...
   │
   └── [R ray_bundle_returns]
       ├── distance_m: [N] float32      (per-return measured metric distances along rays)
       ├── intensity: [N] float32       (per-return measured return intensity values [0,1])
       └── {generic_data}/...           (may include additional return values, like elongation, reflectivity, etc.)

*Radar Sensor Frames*:

.. code-block:: text

   radars/{radar_id}/frames/{frame_name}/
   ├── ray_bundle/
   │   ├── direction: [N,3] float32  (per-ray normalized ray directions in sensor coordinates)
   │   ├── timestamp_us: [N] uint64  (per-ray timestamps of ray measurement time in us)
   │   └── {generic_data}/...  
   │
   └── [R ray_bundle_returns]
       ├── distance_m: [N] float32  (per-return measured metric distances along rays)
       └── {generic_data}/...       (may include radial velocities, RCS)

Cuboids Component
~~~~~~~~~~~~~~~~~

3D cuboid track observations are stored in a structured format:

.. code-block:: text

   cuboids/
   └── {component_instance_name}/
       └── observations: [N] (cuboid track observations)

Each observation is a JSON-serializable object containing:

* ``track_id`` - Unique track identifier (str)
* ``class_id`` - Object class label (str)
* ``timestamp_us`` - Observation timestamp in us (int)
* ``reference_frame_id`` - Reference frame identifier (str)
* ``reference_frame_timestamp_us`` - Reference frame timestamp in us (int)
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




.. rubric:: Footnotes

.. [#f1] NCore V4 component stores are represented by
         `zarr <https://zarr.readthedocs.io/en/stable/>`_ groups within
         either a custom ``.zarr.itar`` archive format or plain directory
         stores. The ``SequenceComponentGroupsReader`` and
         ``SequenceComponentGroupsWriter`` types provide the primary APIs for
         loading and creating V4 data.
