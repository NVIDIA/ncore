.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _data_conventions: 


Specification
=============

All data is stored following the NCore data specification. If modules use a
different convention internally, the conversion should be done within the
module, and output data should be converted back to this convention.


Coordinate Systems and Transformations
--------------------------------------
There are several coordinate systems that are used in practice - NCore uses the
following conventions.


Transformations between Coordinate Systems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All relative transformations are stored in the form of homogeneous 4x4
:math:`SE(3)` matrices, where the top left 3x3 elements represent a rotation
matrix :math:`\mathbf{R}` and the first three rows of the last column denote a
translation :math:`\mathbf{t}` in the metric units of the sequence. They are
denoted using the identifier ``T_a_b`` representing :math:`T_a^b`, which
corresponds to the transformation matrix that maps the points from the
coordinate system :math:`a` to the coordinate system :math:`b`.

For instance, a point :math:`\mathbf{p}_a` in coordinate system :math:`a` can be
transformed to point :math:`\mathbf{p}_b` as

.. math::
    \mathbf{p}_b = \mathbf{R}_a^b \, \mathbf{p}_a + \mathbf{t}_a^b

Transformations are chained together by matrix multiplication. For example,
given the transformations :math:`T_a^b` and :math:`T_b^c`, the transformation
from :math:`a` to :math:`c` is obtained as

.. math::
    T_a^c = T_b^c \, T_a^b


Pose Graph Representation
^^^^^^^^^^^^^^^^^^^^^^^^^

NCore represents all spatial relationships in a **pose graph** -- a tree of
named coordinate frames connected by rigid transformations. Nodes represent
coordinate frames and edges represent :math:`SE(3)` transformations between
them. The graph forms a tree (no cycles), enabling unambiguous traversal between
any pair of frames by composing transformations along the unique connecting
path. Edges can be traversed in either direction via :math:`SE(3)` inverse.

Edges are either **static** or **dynamic**:

* **Static edges** are time-invariant 4x4 :math:`SE(3)` matrices. These
  represent relationships that do not change over time, such as sensor extrinsic
  calibrations (e.g., ``T_camera_rig``), or the mapping between local and global
  world frames (``T_world_world_global``).

* **Dynamic edges** are time-dependent trajectories -- sequences of
  :math:`SE(3)` matrices with associated timestamps. They can be interpolated to
  arbitrary timestamps using SLERP for the rotation component and linear
  interpolation for translation. The ego-vehicle trajectory (e.g.,
  ``T_rig_world``) is a typical dynamic edge.

A typical pose graph for an AV recording has the following structure:

.. figure:: pose_graph.svg
   :width: 60%

   Example pose graph for a vehicle with an articulated trailer. Dashed edges
   are static (time-invariant) transforms; bold edges are dynamic
   (time-dependent) trajectories.

The well-known coordinate frames are:

* ``rig`` -- the vehicle / robot body frame (see below)
* ``world`` -- a sequence-local reference frame (see below)
* ``world_global`` -- the earth-centered-earth-fixed (ECEF) global frame
* Sensor frames -- named by their sensor ID (e.g., ``camera_front_wide_120fov``)

World / Global Coordinate Frame
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ecef.png
   :figwidth: 40%
   :width: 50%

The position and orientation of the ego frame, as well as other objects in the
scene, can be expressed in the earth-centered-earth-fixed (ECEF) `world_global`
coordinate system in the form of high-precision :math:`SE(3)` matrices. To avoid
very large local coordinates and numerical precision issues, NCore uses a local
`world` coordinate system where one usually stores the first pose of the ego
frame in the sequence to an identity matrix and express all other poses relative
to it. The conversion between the local `world` and `world_global` reference
frames is known as ``T_world_world_global``.


Rig Coordinate Frame
^^^^^^^^^^^^^^^^^^^^

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


Camera and Image Coordinate System
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

.. _camera_ray_conventions:

Camera Rays and Camera Points
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All quantities below are expressed in the **extrinsic camera frame** defined
above, i.e. relative to the camera frame's origin, with the principal axis along
:math:`+z`.

Unprojection: camera rays
"""""""""""""""""""""""""

Unprojecting an image point yields a *camera ray*, whose representation depends
on whether the camera model is central:

* **Central models** (FTheta, ideal pinhole, OpenCV pinhole, OpenCV fisheye)
  project through a single centre. A ray is a 3d **direction**
  ``[dx, dy, dz]``, relative to the camera frame origin, and that origin is
  where the ray starts, so it does not have to be stored. These directions are
  returned normalized.
* **Non-central models** (the ideal orthographic model) have rays that share no
  common origin, so a direction alone does not identify a ray. A ray is 6d, an
  explicit origin followed by a direction,
  ``[ox, oy, oz, dx, dy, dz]``. The origin is a point in the camera frame,
  stated relative to the same camera frame origin that a central model's
  direction is measured from.

:attr:`~ncore.sensors.CameraModel.camera_ray_dim` reports which representation a
model uses. World rays (as returned by the ``image_points_to_world_rays_*`` and
``pixels_to_world_rays_*`` methods) are always 6d ``[origin, direction]``, for
every model.

Projection: camera points
"""""""""""""""""""""""""

The forward direction takes a 3d camera-frame **point** for every model, which
is why it is called
:meth:`~ncore.sensors.CameraModel.camera_points_to_image_points`. What each
model does with that point's magnitude differs, however:

* A **central** model's projection is **scale-invariant**: it divides by depth
  (pinhole) or takes an angle about the projection centre (FTheta, fisheye), so
  every point along a given ray projects to the same image point. Passing a
  direction of any length is therefore equivalent to passing an actual point.
* A **non-central** model's projection is **not** scale-invariant. An
  orthographic camera drops the depth rather than dividing by it, so the
  point's ``x`` and ``y`` *are* the quantity being projected.

.. warning::

   It follows that a normalized direction is **not** a valid input for a
   non-central model: normalizing rescales ``x`` and ``y`` and silently moves
   the projected location. Pass the actual camera-frame point. For central
   models either works.

.. note::

   :meth:`~ncore.sensors.CameraModel.camera_points_to_image_points` and
   :meth:`~ncore.sensors.CameraModel.camera_points_to_pixels` were previously
   named ``camera_rays_to_image_points`` and ``camera_rays_to_pixels``. The old
   names remain as deprecated aliases. They were renamed because the argument
   was always a point; with only scale-invariant models the distinction was not
   observable, and the first non-central model made it so.


Sensor Models
-------------

NCore supports multiple sensor models for camera and lidar intrinsics and
distortion parameters. For the full list of supported models, mathematical
specifications, and parameter descriptions, see :ref:`sensor_models`.
