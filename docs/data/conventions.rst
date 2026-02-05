.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION
.. SPDX-License-Identifier: Apache-2.0

.. _data_conventions: 


Conventions and Specification
=============================

All data has to be stored following these NCore data specifications. If modules
use a different convention internally, the conversion should be done within the
module, and output data should be converted back to this convention.


Coordinate Systems and Transformations
--------------------------------------
There are several coordinate systems that are used in practice - NCore uses the
following conventions.


**Transformations between Coordinate Systems**

All the transformations are stored in the form of 4x4 ``SE3`` matrices, where
the top left 3x3 elements represent the rotation matrix ``R`` and the first
three rows of the last column denote the translation ``t`` in meters. They are
stored using the convention ``T_a_b``, which denotes the transformation matrix
that transforms the points from the coordinate system ``a`` to the coordinate
system ``b``. For example a point :math:`\mathbf{p}_a` in coordinate system
``a`` can be transformed to point :math:`\mathbf{p}_b` as

.. math::
    \mathbf{p}_b = \mathbf{R}_a^b \mathbf{p}_a + \mathbf{t}_a^b


**Pose Graph Representation**

NCore represents all transformations in a pose graph that can be evaluated at
any valid target timestamp in the dataset. Each node in the pose graph
represents a coordinate system, and each edge represents a transformation
between two coordinate systems. Time-dependent transformations are represented
as trajectories that can be interpolated to obtain smooth transformations. By
convention, in NCore AV / robotics applications, `rig` is the name of a
reference coordinate system fixed to the vehicle / robot, `world` represents a
local coordinate system fixed to the data sequence, and sensor-associated
coordinate system fixed to the recording device are named equal to the sensor
IDs.

**World / Global Coordinate Frame**

.. figure:: ecef.png
   :figwidth: 40%
   :width: 50%

The position and orientation of the ego frame, as well as other objects in the
scene, can be expressed in the earth-centered-earth-fixed (ECEF) `world_global`
coordinate system in the form of high-precision SE3 matrices. To avoid very
large local coordinates and numerical precision issues, NCore uses a local
`world` coordinate system where one usually stores the first pose of the ego
frame in the sequence to an identity matrix and express all other poses relative
to it. The conversion between the local `world` and `world_global` reference frames
is known as ``T_world_world_global``.


**Rig Coordinate Frame**

.. figure:: rig.png
   :figwidth: 40%
   :width: 50%

A vehicle / robot-local ``rig`` coordinate system is defined as a right-handed
coordinate system with the x-axis pointing to the front of the car, y is
pointing left, and z up. The origin of the coordinate system is located in the
middle of the rear axis on the nominal ground.

A time-dependent trajectory of the ego frame is expressed as ``T_rig_world``.
This can be interpolated to obtain smooth transformations for the start and end
timestamps for each sensor frame (e.g., the start- and end-times of a
rolling-shutter camera frame / spinning lidar frame).


**Camera and Image Coordinate System**

.. figure:: camera.jpg
   :figwidth: 40%
   :width: 80%

Both extrinsic camera and intrinsic image coordinate systems are right-handed
coordinate systems. The axes of the extrinsic camera coordinate system are
defined such that the camera's principal axis is along the +z axis, the x-axis
points to the right, and the y-axis points down. The principal point corresponds
to the optical center of the camera.

.. _image_coordinate_conventions:

The image coordinate system is defined such that the u-axis points to the right
and the v-axis down. The origin of the image coordinate system is in the top
left corner of the image, and the units are pixels. Continuous pixel coordinates
start with ``[0.0, 0.0]`` at the top-left corner of the top-left pixel in the
image, i.e., both the u and v coordinates of the first pixel span the range
``[0.0, 1.0]``.


Sensor Models
-------------

NCore supports multiple sensor models for representing camera / lidar intrinsics
and distortion parameters. Each model provides different parameterizations
suited for various camera types and use cases.

**Camera Models**

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

For detailed mathematical specifications and parameter descriptions of each
camera model, please refer to :ref:`camera_model_parameterizations`.

**Lidar Models**

NCore supports the following lidar models:

* :ref:`Row-Offset Structured Spinning Lidar <row_offset_spinning_lidar_model>`
  - Structured spinning lidar model with per-row azimuth offsets (compatible
  with sensors like Hesai P128)

For detailed mathematical specifications and parameter descriptions, please
refer to :ref:`lidar_model_parameterizations`.

**External Distortion Models**

Camera sensors may include external distortion models to account for distortion
sources outside the camera itself:

* :ref:`Bivariate Windshield Model <bivariate_windshield_model>` - Models
  optical distortion from vehicle windshields using bivariate polynomial
  deflection in spherical coordinates

Data Format Specifications
---------------------------

For detailed information about how data is organized and stored in NCore's
component-based format, including data hierarchies, metadata schemas, and
loading patterns, please refer to :ref:`data_formats`.
