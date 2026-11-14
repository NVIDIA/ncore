.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _upgrading_20:

Upgrading from NCore 19.x
=========================

NCore 20 keeps the data format: data written by NCore 19.x is read by NCore 20, and data written by NCore 20 in zarr
format v2 (the default) is read by NCore 19.x. This page lists the API changes of NCore 20 relative to the latest
NCore 19.x release (19.9), by area; changes that already shipped in an earlier 19.x release are marked with that
version. The `GitHub releases <https://github.com/NVIDIA/ncore/releases>`_ hold the release notes of each version,
including the changes between the pre-releases of NCore 20; the repository no longer keeps a ``CHANGELOG.md``.

Most code reading data through :class:`~ncore.data.v4.SequenceLoaderV4` or the component readers keeps working;
check the sections below for the APIs you use. Common changes:

* optional parameters of many APIs are keyword-only;
* camera / lidar ray generation returns valid flags;
* model factories are module-level functions (``ncore.sensors.camera_model_from_parameters``);
* invalid input raises ``ValueError`` instead of ``AssertionError``;
* the storage APIs below the components are new (:mod:`ncore.data.stores`).

Dependencies
------------

* zarr-python 3 (>= 3.1.4, on Python >= 3.11) is supported next to zarr-python 2, and installed by default on Python
  >= 3.11. The implementation is selected at import time. See :ref:`zarr_versions`. On Python >= 3.11, zarr-python 2
  requires >= 2.18.7 (``"zarr<3"`` resolves to it); older Python versions keep zarr-python 2 >= 2.12.
* zarr format v3 stores can be written opt-in (``zarr_format=3`` of
  :class:`~ncore.data.v4.SequenceComponentGroupsWriter`, ``--zarr-format 3`` of the data converters), and need
  zarr-python 3 for writing and reading. zarr format v2 stays the default.
* Python 3.8 remains supported (with zarr-python 2). ``numcodecs`` and ``fsspec`` are declared dependencies, and the
  minimum versions of ``pillow`` (>= 12.2.0, Python >= 3.10) and ``cbor2`` (>= 5.9.0, Python >= 3.9) carry security
  fixes.

Public packages
---------------

Import public names from the public packages instead of ``ncore.impl``:

* :mod:`ncore.data.stores` (new): indexed tar archives, zarr hierarchy stores, groups / arrays (see
  :ref:`stores_api`).
* :mod:`ncore.geometry` (new): ``PoseInterpolator``, ``PoseGraphInterpolator``, ``MotionCompensator``,
  ``HalfClosedInterval``, ``time_bounds``, and the rigid transformation functions (``se3_inverse``,
  ``so3_trans_2_se3``, ``transform_point_cloud``, ``bbox_pose``, ``pose_bbox``, ``transform_bbox``,
  ``is_within_3d_bboxes``; names and signatures unchanged from ``ncore.impl.common.transformations``).
* :mod:`ncore.data`: the JSON types ``Json``, ``JsonDocument``, ``JsonLike`` and ``JsonDocumentLike`` (``JsonLike``
  was in ``ncore.impl.data.types``), ``FOV``, ``RayBundleSensorProtocol``, the asynchronous protocols
  (``AsyncSequenceLoaderProtocol`` and friends) and ``AsyncRayBundleSensorPointCloudsSourceAdapter``.
* :mod:`ncore.data.v4`: ``BaseSensorComponentReader`` / ``BaseSensorComponentWriter``,
  ``BaseRayBundleSensorComponentReader`` / ``BaseRayBundleSensorComponentWriter``, ``SequenceMeta``,
  ``ComponentGroupAssignments`` and ``IndexTailRetention``.
* ``ncore.impl.data.v4.compat`` no longer re-exports the compatibility protocols (``SequenceLoaderProtocol``,
  ``CameraSensorProtocol``, ...): import them from :mod:`ncore.data`.
* ``ncore.data.EncodedImageHandle`` is ``CameraSensorProtocol.EncodedImageDataHandleProtocol`` (same ``get_data()``
  method, now runtime-checkable).

Component readers
-----------------

* Arrays returned by component readers are **read-only** (e.g., frame data, ray bundles, timestamps, generic data,
  and the cached ``frames_timestamps_us``, which returns the same array on each access): copy them before modifying
  them in place (``np.array(data)``).
* All pose getters of ``PosesComponent.Reader`` (``get_static_pose`` / ``get_dynamic_pose`` and the plural
  generators) return poses with their **stored dtype** (float32 or float64). This is kept by design, like the
  read-only arrays; convert poses if a specific dtype is required. (The singular getters return the stored dtype
  since 19.8.1; earlier 19.x versions always returned float64.)
* All reads of array data have asynchronous variants (``*_async``), see :ref:`async_reads`.
* ``PosesComponent.validate_frame_name(name)`` replaces ``ncore.impl.data.v4.components.validate_frame_name`` and
  raises ``ValueError``.
* ``PointCloudsComponent.Writer.store_pc`` / ``CameraLabelsComponent.Writer.store_label`` and
  ``set_generic_data`` of all component writers return the writer (chainable).
* Keyword-only arguments: ``store_type`` and ``zarr_format`` of ``SequenceComponentGroupsWriter`` and
  ``SequenceComponentGroupsWriter.from_reader``; ``attribute_schemas`` of ``PointCloudsComponent.Writer``; the
  ``attributes``, ``generic_data`` and ``generic_meta_data`` of ``store_pc`` and the ``generic_meta_data`` of
  ``store_label``. Arguments forwarded to writer constructors by ``register_component_writer`` follow its
  ``group_name`` and ``generic_meta_data`` parameters: pass them by keyword.
* Errors: ``PointCloudsComponent.Reader.get_attribute_schema`` and ``CameraLabelsComponent.Reader.get_label`` raise
  ``KeyError`` for unknown names / timestamps; ``register_component_writer`` raises ``ValueError`` for empty or
  duplicate names; a duplicate component instance raises ``RuntimeError`` when opening a sequence; a mismatching shape
  of promoted generic data raises ``ValueError``.
* Invalid input to the component writers (poses, sensor frames, cuboids, point clouds, camera labels, and
  ``finalize``) and to the validated data types of :mod:`ncore.data` (camera / lidar model parameters, ``BBox3``,
  ``LabelType``, ``LabelSchema``, ``QuantizationParams``, ``CuboidTrackObservation``, ``CameraLabelDescriptor``)
  raises ``ValueError`` instead of ``AssertionError``, also under ``python -O``.
* Custom components access their data via ``self._group``, a :class:`ncore.data.stores.Group` (instead of a
  zarr-python group, which is available as ``self._group.zarr``), see :ref:`stores_api`.

Opening sequences
-----------------

* Optional parameters are keyword-only: ``SequenceComponentGroupsReader(paths, *, open_consolidated=...,
  itar_index_tail_read_size=..., max_threads=..., ...)`` and ``SequenceLoaderV4(reader, *,
  poses_component_group_name=..., ...)``. For example, replace ``SequenceComponentGroupsReader(paths, False)`` with
  ``SequenceComponentGroupsReader(paths, open_consolidated=False)``.
* ``SequenceLoaderV4.open(paths, ...)`` / ``open_async(paths, ...)`` open the component stores and the loader in one
  step (the reader + loader construction keeps working). ``open_consolidated`` accepts ``"auto"``, and
  ``component_group_paths`` any sequence of ``UPath`` / ``Path``.
* A missing component group raises ``KeyError``, an empty path list ``ValueError``.
* ``.itar`` paths are no longer checked for existence up front: a missing archive raises ``FileNotFoundError`` when
  its index is read.
* Readers release the index tail read of each ``.itar`` store after opening it from its consolidated metadata, so
  records near the end of an archive are read from storage when accessed; ``itar_index_tail_retention="keep"``
  keeps the previous behaviour (see :ref:`index_tail_retention`).

Compatibility API (``SequenceLoaderV4``)
----------------------------------------

* ``SequenceLoaderProtocol.query_camera_labels`` returns a ``Sequence`` instead of a ``List``.
* ``SequenceLoaderV4.Sensor`` / ``CameraSensor`` / ``RayBundleSensor`` / ``LidarSensor`` / ``RadarSensor``: all
  arguments after the reader are keyword-only, ``Sensor(sensor_reader=...)`` is ``Sensor(reader, pose_graph=...)``,
  ``CameraSensor``'s ``mask_reader`` and ``LidarSensor``'s ``model_parameters`` default to None. The constructors
  assign the pose graph directly: subclass overrides of ``set_pose_graph`` are not called during construction.
* Ray bundle sensors take an optional ``motion_compensator`` (the loader passes its own, configured by
  ``motion_compensator_parameters``, see :ref:`motion_compensation`); without one they use
  ``MotionCompensator(pose_graph)``, exposed by the new ``motion_compensator`` property. ``set_pose_graph`` replaces
  the compensator by one of the new pose graph with the same parameters, so point clouds of a sensor with a replaced
  pose graph are motion-compensated with that pose graph (19.x kept compensating with the original pose graph).
* ``get_camera_sensor`` / ``get_lidar_sensor`` / ``get_radar_sensor`` raise ``KeyError`` for unknown sensor ids
  (19.x raised ``ValueError``), like the other getters.
* ``RayBundleSensorPointCloudsSourceAdapter(sensor, *, return_index=..., promote_generic_data=...)`` and its
  asynchronous twin take their optional arguments by keyword.
* Sensors provide ``get_T_sensor_rig(rig_node=...)`` for named rig frames; the ``T_sensor_rig`` property is the
  ``"rig"`` shortcut.
* ``RayBundleSensorProtocol.FramePointCloud`` has an optional ``timestamp_us`` field (the per-point timestamps,
  read-only); lidar / radar point-cloud sources adapted via ``get_point_clouds_source`` reuse it.
* ``get_closest_frame_index`` raises ``ValueError`` for an out-of-range ``relative_frame_time``.

Storage
-------

The storage APIs below the components were rewritten (:mod:`ncore.data.stores`, see :ref:`stores_api`):

* ``IndexedTarStore`` is replaced by :class:`~ncore.data.stores.IndexedTarArchive` (the records of an archive,
  independent of zarr-python) and :class:`~ncore.data.stores.IndexedTarZarrStore` (a zarr hierarchy in an archive).
  Other zarr-python stores are used via :class:`~ncore.data.stores.ZarrStoreAdapter` (e.g.,
  ``ZarrStoreAdapter.directory(path)``). An ``IndexedTarZarrStore`` is not a zarr-python store itself: pass
  ``store.zarr`` to zarr-python, or open the hierarchy via ``store.open_root_group()``.
* Removed: ``ConsolidatedCompressedMetadataStore``, the ``metadata_key=`` / ``mode=`` / ``**kwargs`` arguments of the
  consolidation functions, ``IndexedTarStore.mutex`` / ``.index`` / ``.TarRecord`` / ``.TailBuffer``. The records of
  an archive are ``IndexedTarArchive.records`` (``TarRecord.offset_data`` is ``Record.offset``).
* ``consolidate_compressed_metadata`` / ``open_compressed_consolidated`` are replaced by ``Store.consolidate()`` /
  ``Store.open_root_group()``; ``Store.is_consolidated_metadata_key(key)`` identifies consolidated metadata records.
* Optional parameters are keyword-only, e.g. ``IndexedTarZarrStore(path, mode="w")`` instead of
  ``IndexedTarZarrStore(path, "w")``, and ``open_root_group(consolidated=False)``.
* Writes through hierarchies opened for reading raise ``ValueError``, also ``Store.create_root_group`` and
  ``Store.consolidate`` of stores opened for reading. ``Group.require_group`` of an existing array raises
  ``ValueError``.
* Opening consolidated metadata of an unsupported format raises ``ValueError`` (19.x raised zarr-python's
  ``MetadataError``).
* Node metadata records of ``.itar`` archives are appended when closing an archive, so archives are not byte-identical
  to archives written by NCore 19.8 (their records are), and NCore 19.8 reads them.

Sensor models
-------------

* ``CameraModel.from_parameters(...)`` is removed: use ``ncore.sensors.camera_model_from_parameters(...)``. Likewise
  ``ExternalDistortionModel.from_parameters`` -> ``external_distortion_model_from_parameters``, and
  ``LidarModel.maybe_from_parameters`` / ``StructuredLidarModel.maybe_from_parameters`` ->
  ``maybe_lidar_model_from_parameters`` (or ``lidar_model_from_parameters``; check
  ``isinstance(model, StructuredLidarModel)`` where a structured model is required). The factories forward extra
  keyword arguments to the model constructor (e.g. ``newton_iterations=...``), and factories registered with
  ``register_*_model`` receive them as ``**model_kwargs``.
* Ray generation returns valid flags (see :ref:`ray_validity`):

  * ``image_points_to_camera_rays`` / ``pixels_to_camera_rays`` return a ``CameraModel.CameraRaysReturn`` (use
    ``.camera_rays``, mask with ``.valid_flag``) instead of a tensor. ``camera_rays=`` of the world-ray unprojections
    takes a ``CameraRaysReturn`` (``CameraRaysReturn.all_valid(rays)`` for rays known to be valid), and
    ``CameraModel.WorldRaysReturn`` has a ``valid_flag`` field. Camera models implementing
    ``_image_points_to_camera_rays_impl`` return a ``CameraRaysReturn``.
  * ``StructuredLidarModel.elements_to_sensor_rays`` returns ``LidarModel.SensorRaysReturn`` (``.sensor_rays``),
    ``elements_to_sensor_points`` returns ``LidarModel.SensorPointsReturn`` (``.sensor_points``), and
    ``elements_to_sensor_angles`` returns ``LidarModel.SensorAnglesReturn`` (``.sensor_angles``; custom structured
    models return it). ``sensor_rays=`` of ``elements_to_world_rays_shutter_pose`` takes a ``SensorRaysReturn``, and
    ``LidarModel.WorldRaysReturn`` has a ``valid_flag`` field. ``LidarModel.SensorRayReturn`` is renamed
    ``LidarModel.SensorRaysReturn``. ``sensor_rays_to_sensor_angles`` also accepts a ``SensorRaysReturn`` (custom
    lidar models implementing it should accept that type).
  * ``WorldPointsToImagePointsReturn``, ``WorldPointsToPixelsReturn`` and ``WorldPointsToSensorAnglesReturn`` have a
    required ``valid_flag`` field over all input points. ``valid_indices`` are always 1-D (since 19.8.6; earlier 19.x
    versions returned a 0-dim tensor when exactly one point projected).

* Keyword-only arguments: the ``return_*`` flags of all ``world_points_to_*`` methods; ``camera_rays``,
  ``return_T_sensor_worlds`` and ``return_timestamps`` of the ``pixels_to_world_rays_*`` /
  ``image_points_to_world_rays_*`` methods, and ``sensor_rays``, ``return_T_sensor_worlds`` and ``return_timestamps``
  of ``elements_to_world_rays_shutter_pose`` (e.g., replace ``pixels_to_world_rays_static_pose(pixels, T, ts,
  rays)`` with ``pixels_to_world_rays_static_pose(pixels, T, ts, camera_rays=rays)``); the model arguments after
  ``parameters, device, dtype`` (``newton_iterations``, ``min_2d_norm``, ``max_normalized_radius_squared``,
  ``tangential_iterations``); for ``RowOffsetStructuredSpinningLidarModel``, ``device`` and ``dtype`` now directly
  follow the parameters, and the ``angles_to_columns_map_*`` / ``fov_eps_factor`` arguments are keyword-only.
* ``camera_rays_to_image_points(cam_rays=...)`` / ``camera_rays_to_pixels(cam_rays=...)`` are renamed
  ``camera_rays=``; subclasses overriding ``_camera_rays_to_image_points_impl`` should rename the parameter.
* OpenCV pinhole: validity follows the first extremum of the radial distortion instead of fixed limits of the
  distortion factor (override with ``max_normalized_radius_squared=``, see :ref:`ray_validity`), and the
  undistortion iterates until every point converged, so image points near the image corners stay valid when
  unprojected with central points. ``OpenCVPinholeCameraModel.compute_max_normalized_radius_squared`` is
  :meth:`OpenCVPinholeCameraModelParameters.compute_max_normalized_radius_squared
  <ncore.data.OpenCVPinholeCameraModelParameters.compute_max_normalized_radius_squared>`.
* OpenCV fisheye: ``inverse_max_angle`` is private. (Since 19.8.4, the inverse is bracketed on the increasing branch of
  the polynomial, and ``newton_iterations`` is an upper bound with default 20 instead of 3.)
* FTheta: ``FThetaCameraModel`` takes a keyword-only ``tangential_iterations`` argument (an attribute in 19.9), and
  its ``tangential_coeffs`` buffer is persistent: state dicts contain it (load state dicts written by NCore 20 into
  older versions with ``strict=False``). (``FThetaCameraModelParameters.tangential_coeffs`` were added in 19.9.0:
  default zeros, omitted when stored as zeros, non-zero coefficients require ``max_angle < pi/2``, readers older than
  19.9 ignore them.)
* Private hooks for subclasses: ``_has_tangentials`` / ``_has_tangential`` -> ``_has_tangential_distortion``,
  ``_has_thin_prism`` -> ``_has_thin_prism_distortion``; the OpenCV pinhole hooks ``_compute_distortion`` /
  ``_iterative_undistort`` are single-underscore methods. The undistortion stop parameter
  ``stop_mean_of_squares_error_px2`` is the keyword-only ``stop_squared_residual``, which applies per point (instead
  of to the mean over all points).
* Invalid input to the camera and lidar models (constructor arguments, shapes and dtypes of poses / points / image
  points / elements, missing or reversed timestamps with ``return_timestamps=True``, ...) raises ``ValueError``
  instead of ``AssertionError``, also for rays passed for reuse (``camera_rays=`` / ``sensor_rays=``) that don't
  match the input points. The constructors validate ``newton_iterations`` (FTheta, OpenCV fisheye) and
  ``angles_to_columns_map_dtype`` (row-offset spinning lidar), which were checked on first use.
  ``RowOffsetStructuredSpinningLidarModelParameters`` rejects column azimuths one full revolution or more away from
  the first column.
* :class:`~ncore.sensors.Rectificator`'s ``valid_mask`` also requires a valid ray of the target camera.

Geometry
--------

* ``MotionCompensator.motion_decompensate_points(...)`` returns a ``MotionCompensator.MotionDecompensationResult``:
  replace ``xyz = mc.motion_decompensate_points(...)`` with ``xyz = mc.motion_decompensate_points(...).xyz_pointtime``.
* ``MotionCompensator.Parameters`` selects the motion compensation method (``EXACT``, the default, or the faster
  ``INTERPOLATED``), see :ref:`motion_compensation`.
* ``PoseInterpolator.slerp``, ``.f_t`` and ``.last_row`` are private: use ``interpolate_to_timestamps`` /
  ``extrapolate_to_timestamps`` (``poses`` and ``timestamps`` remain public). Interpolation accepts any strictly
  increasing time domain; extrapolation requires integer microseconds.
* ``MotionCompensator(pose_graph, *, parameters=...)``: ``parameters`` (new) is keyword-only, and so are
  ``reference_timestamp_us`` and ``anchor_frame_id`` of ``motion_compensate_points`` / ``motion_decompensate_points``
  (e.g., replace ``mc.motion_compensate_points(sensor_id, xyz, ts, start, end, ref)`` with
  ``mc.motion_compensate_points(sensor_id, xyz, ts, start, end, reference_timestamp_us=ref)``).
* ``time_bounds``, ``HalfClosedInterval``, ``PoseInterpolator``, ``is_within_3d_bboxes``,
  ``MotionCompensator.motion_compensate_points`` / ``motion_decompensate_points`` and ``PoseGraphInterpolator.Edge``
  raise ``ValueError`` instead of ``AssertionError`` for invalid input.

Data converters
---------------

* ``BaseDataConverterConfig.zarr_format`` (default 2) selects the zarr format of the written stores
  (``--zarr-format`` of the converter CLIs). On Python < 3.10, fields added by subclasses of
  ``BaseDataConverterConfig`` need defaults.
