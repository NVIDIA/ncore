.. _conventions: 

Conventions
===========

Coordinate systems and transformations
--------------------------------------
There are several coordinate systems that are used in this system, by default we use the following conventions. 
All data should be saved/exported in the correct format, if some modules use a different convention internally, the conversion should be done within the module, and output data should be converted back to this convention.


**Global coordinate frame**

.. figure:: ../images/ecef.png
   :figwidth: 40%
   :width: 50%
   :align: right

The position and orientation of the ego car, as well as other objects in the scene, are expressed in the earth-centered-earth-fixed (ECEF) coordinate system in the form of SE3 matrices. 
To avoid very large coordinates, we set the first pose of the ego car in the sequence to an identity matrix ``base_pose`` and express all other poses relative to it. [#f1]_


**Rig coordinate frame**

.. figure:: ../images/rig.png
   :figwidth: 40%
   :width: 50%
   :align: right

The rig coordinate system is defined as a right-handed coordinate system with the x-axis pointing to the front of the car, y is pointing left, and z up. The origin of the coordinate system is located in the middle of the rear axis on the nominal ground.

**Camera and image coordinate system**

.. figure:: ../images/camera.jpg
   :figwidth: 40%
   :width: 80%
   :align: right

Both camera and image coordinate systems are right-handed coordinate systems. The axes of the camera coordinate system are defined as follows, the camera looks down the +z axis, the x-axis points to the right, and the y-axis points down. The origin is at the optical center of the camera. The image coordinate system is defined such that the u-axis points to the right and the v-axis down. The origin of the image coordinate system is in the top left corner of the image, and the units are pixels.


**Transformations between the coordinate systems**

All the transformations are saved in the form of ``SE3`` matrices, where the top left 3x3 elements represent the rotation matrix ``R`` and the first three rows of the last column denote the translation ``t`` in meters. They are saved using the convention ``T_a_b``, which denotes the transformation matrix that transforms the points from the coordinate system ``a`` to the coordinate system ``b``. For example a point :math:`\mathbf{p}_a` in coordinate system ``a`` can be transformed to point :math:`\mathbf{p}_b` as

.. math::
    \mathbf{p}_b = \mathbf{R\_a\_b}_b * \mathbf{p}_a + \mathbf{t\_a\_b}



Folder structure
----------------

The data processed using DSAI library on Maglev will be made available in the following folder structure [#f2]_:

.. code-block:: text

   session-id/
    ├-lidars/
    │ ├─lidar_gt_top_p128_v4p5/
    │ │ ├-000000.hdf5
    │ │ ├-000000.json
    │ │ ├-000000_labels.json
    │ │ ├-000001.hdf5
    │ │ ├-000001.json
    │ │ ├-000001_labels.json

    │ │ ├─...
    │ │ └-meta.json
    │ │
    │ ├─{lidar_parking_gt_front_p128/}
    │ │ ├-000000.hdf5
    │ │ ├-000000.json
    │ │ ├-000000_labels.json
    │ │ ├─...
    │ │ └-meta.json
    │ │
    │ └─...
    │
    ├-cameras/
    │ ├─camera_front_wide_120fov/
    │ │ ├-000000.jpg
    │ │ ├-000000.json
    │ │ ├-000000_mask.png
    │ │ ├-000000_sem.png
    │ │ ├-000000_inst.png
    │ │ ├─...
    │ │ └-meta.json
    │ │
    │ ├─camera_front_fisheye_200fov/
    │ │ ├-000000.jpg
    │ │ ├-000000.json
    │ │ ├-000000_mask.png
    │ │ ├-000000_sem.png
    │ │ ├-000000_inst.png
    │ │ ├─...
    │ │ └-meta.json
    │ │
    │ └─...
    │
    ├-{radars/}
    │ ├─radar_front_center/
    │ │ ├-000000.hdf5
    │ │ ├-000000.json
    │ │ ├─...
    │ │ └-meta.json
    │ │
    │ ├─radar_front_left/
    │ │ ├-000000.hdf5
    │ │ ├-000000.json
    │ │ ├─...
    │ │ └-meta.json
    │ │
    │ └─...
    │
    ├-poses.json
    ├-labels.json
    └-meta.json


Sensor data
-----------

**Images**

The images are saved either in the `*.jpg` (raw sensor data) or in the `*.png` format (mask, semantic or instance labels).

**Point clouds**

Lidar and radar point clouds are saved in the `*.hdf5` files, in the tabular format with named columns.

Lidar data contains the following columns:

* ``xyz_s`` - 3D coordinate of the start of the ray in the sensor reference frame (float32, [n,3])
* ``xyz_e`` - 3D coordinate of the end of the ray in the sensor reference frame (float32, [n,3])
* ``intensity`` - measured intensity (uint8, [n])
* ``dynamic_flag`` - dynamic flag (bool, [n])

Radar data contains the following columns:

* ``xyz_s`` - 3D coordinate of the start of the ray in the sensor reference frame (float32, [n,3])
* ``xyz_e`` - 3D coordinate of the end of the ray in the sensor reference frame (float32, [n,3]) 


**Metadata**

Metadata is available per frame, but also for individual sensors and for the general sequence. They are all stored in form of ``*.json`` files.

Per-frame metadata contains the following entries for all sensors: 

* ``T_sensor_rig`` - SE3 transformation matrix from the sensor to the rig coordinate system (np.array, [4,4], float32)
* ``timestamps`` - timestamp of the frame's start and end point in microseconds (np.array, [2,], uint64)
* ``T_rig_world`` - SE3 transformation matrices from the rig to the world coordinate system at the start and end timestamp of the frame (np.array, [2,4,4], float32)



For individual sensors we also save session wise metadata:

*Cameras*: 

* ``img_resolution`` - width and height of the image in pixels (np.array, [2,], int)
* ``rolling_shutter_direction`` - direction of the rolling shutter (int, 1 = TOP_TO_BOTTOM, 2 = LEFT_TO_RIGHT, 3 = BOTTOM_TO_TOP, 4 = RIGHT_TO_LEFT )
* ``camera_model`` - camera model type (str)
* ``exposure_time`` - exposure time of the camera (float32)
* ``principal_point`` - x and y coordinate of the principal point (np.array, [2,], float32)

If ``camera_model` = 'f_theta'`` the following intrinsic parameters will be available: 

* ``bw_poly`` - coefficients of the backward polynomial (np.array, [6,], float32)
* ``fw_poly`` - coefficients of the forward polynomial (np.array, [6,], float32)

If ``camera_model` = 'pinhole'`` the following intrinsic parameters will be available: 

* ``fl_u`` - focal length in u direction (float32)
* ``fl_v`` - focal length in v direction (float32)
* ``distortion_coefficients`` - p1, p2, k1, k2, k3 distortion coefficients (np.array, [5,], float32)


*Lidars*: 

* ``{sampling_pattern}`` - sampling pattern of the lidar sensor in terms of elevation and azimuth angles (np.array, [n,m], float32)
* 

Finally, we also save general metadata related to the session (input data and versioning):

* ``version`` - version of the dataset (str)
* ``ego_motion_type`` - type of ego-motion that was used to generate the data (str) 
* ``calibration_type`` - type of sensor calibration that was used to generate the data (str)
  

Labels
------

**4D autolabels**


.. rubric:: Footnotes

.. [#f1] The ``base_pose`` of the Maglev processed datasets will by default be an identity matrix if DeepMap poses are not used. The coordinate system is hence a local 3D cartesian system and not ECEF.
.. [#f2] Curly brackets denote optional data.
