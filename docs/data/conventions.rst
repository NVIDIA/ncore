.. Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

.. _data_conventions: 


Conventions and Specification
=============================

All data has to be stored following these NCORE data specifications.
If modules use a different convention internally, the conversion should be done within the module,
and output data should be converted back to this convention.


Coordinate Systems and Transformations
--------------------------------------
There are several coordinate systems that are used in practice -
NCORE uses the following conventions.


**Transformations between the Coordinate Systems**

All the transformations are stored in the form of 4x4 ``SE3`` matrices, where
the top left 3x3 elements represent the rotation matrix ``R`` and the
first three rows of the last column denote the translation ``t`` in
meters. They are stored using the convention ``T_a_b``, which denotes the
transformation matrix that transforms the points from the coordinate
system ``a`` to the coordinate system ``b``. For example a point
:math:`\mathbf{p}_a` in coordinate system ``a`` can be transformed to
point :math:`\mathbf{p}_b` as

.. math::
    \mathbf{p}_b = \mathbf{R}_a^b \mathbf{p}_a + \mathbf{t}_a^b


**Global Coordinate Frame**

.. figure:: ecef.png
   :figwidth: 40%
   :width: 50%

The position and orientation of the ego car, as well as other objects in
the scene, are expressed in the earth-centered-earth-fixed (ECEF)
coordinate system in the form of SE3 matrices. To avoid very large
coordinates, we set the first pose of the ego car in the sequence to an
identity matrix and express all other poses relative to it. This
reference pose is available as ``T_rig_world_base`` [#f1]_.


**Rig Coordinate Frame**

.. figure:: rig.png
   :figwidth: 40%
   :width: 50%

The ``rig`` coordinate system is defined as a right-handed coordinate
system with the x-axis pointing to the front of the car, y is pointing
left, and z up. The origin of the coordinate system is located in the
middle of the rear axis on the nominal ground. All poses of the ego car
are always expressed as ``T_rig_world`` transformations with two
associated start and end timestamps (e.g., the start- and end-poses
corresponding to a lidar spin).


**Camera and Image Coordinate System**

.. figure:: camera.jpg
   :figwidth: 40%
   :width: 80%

Both extrinsic camera and intrinsic image coordinate systems are
right-handed coordinate systems. The axes of the extrinsic camera
coordinate system are defined such that the camera's principal axis is
along the +z axis, the x-axis points to the right, and the y-axis points
down. The principal point corresponds to the optical center of the
camera.

.. _image_coordinate_conventions:

The image coordinate system is defined such that the u-axis points to
the right and the v-axis down. The origin of the image coordinate system
is in the top left corner of the image, and the units are pixel s.
Continuous pixel coordinates start with ``[0.0, 0.0]`` at the top-left corner of
the top-left pixel in the image, i.e., both the u and v coordinates of
the first pixel span the range ``[0.0, 1.0]``.


Data Format Specifications
---------------------------

For detailed information about how data is organized and stored in NCORE's
V3 (shard-based) and V4 (component-based) formats, including data hierarchies,
metadata schemas, and loading patterns, please refer to :ref:`data_formats`.


.. rubric:: Footnotes

.. [#f1] The ``T_rig_world_base`` of the Maglev processed datasets will
         by default be an identity matrix if DeepMap poses are not used.
         The coordinate system is hence a local 3D cartesian system and
         not ECEF.