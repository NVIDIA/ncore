.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _sensor_models:


Sensor Models
=============

NCore supports multiple sensor models for representing camera and lidar
intrinsics and distortion parameters. Each model provides different
parameterizations suited for various sensor types and use cases.


Camera Models
-------------

The following camera models are supported:

* :ref:`FTheta Camera Model <ftheta_camera_model>` - NVIDIA's FTheta camera
  model with polynomial distortion parameterization, suitable for both
  perspective and wide field-of-view cameras
* :ref:`OpenCV Pinhole Camera Model <opencv_pinhole_camera_model>` - Standard
  pinhole camera model with rational radial, tangential, and thin prism
  distortion coefficients
* :ref:`OpenCV Fisheye Camera Model <opencv_fisheye_camera_model>` - OpenCV's
  fisheye camera model with polynomial distortion for ultra-wide angle and
  fisheye lenses

.. _camera_model_parameterizations:

Camera Model Parameterizations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
--------------------------

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


Lidar Models
------------

NCore supports the following lidar models:

* :ref:`Row-Offset Structured Spinning Lidar <row_offset_spinning_lidar_model>`
  - Structured spinning lidar model with per-row azimuth offsets (compatible
  with sensors like Hesai P128)

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
