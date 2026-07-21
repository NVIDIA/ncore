.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

LiDAR Model Estimation
======================

This tutorial shows how to represent a structured spinning LiDAR in NCore,
estimate its intrinsic model from point clouds when factory calibration is not
available, assign a ``model_element`` index to every ray, evaluate the result,
and store both artifacts in an NCore V4 sequence.

The model and geometry classes are part of the ``nvidia-ncore`` package. The
estimation and alignment utilities are reference implementations in the NCore
repository's ``tools/data_converter`` directory and should be copied or adapted
for a downstream converter.

Model and per-ray indices
-------------------------

A structured spinning LiDAR is represented as a table of ray directions. Each
table element is addressed by ``(row, column)``:

.. math::

   \begin{aligned}
   \mathrm{elevation} &= \mathrm{row\_elevations\_rad}[\mathrm{row}], \\
   \mathrm{azimuth} &= \mathrm{column\_azimuths\_rad}[\mathrm{column}]
       + \mathrm{row\_azimuth\_offsets\_rad}[\mathrm{row}], \\
   \mathrm{ray} &= [\cos(a)\cos(e),\;\sin(a)\cos(e),\;\sin(e)].
   \end{aligned}

``RowOffsetStructuredSpinningLidarModelParameters`` stores the static table.
Every frame additionally stores an ``[N, 2]`` ``uint16`` array named
``model_element``. Entry ``i`` maps ray ``i`` to its model row and column.
Together with distance, the index reconstructs the point from the model:

.. code-block:: python

   model = LidarModel.maybe_from_parameters(params, device="cpu")
   reconstructed_xyz = model.elements_to_sensor_points(model_element, distance_m)

See :ref:`Row-Offset Structured Spinning Lidar
<row_offset_spinning_lidar_model>` for the complete parameterization and
:ref:`v4-data-format` for the frame layout.

Create a nominal model
----------------------

When the sensor data sheet provides the vertical beam angles and horizontal
sampling pattern, start with a nominal model. A nominal model is deterministic,
independent of the captured scene, and often sufficient when factory calibration
is available.

The following example creates a partial-FOV, clockwise model. Replace the angles,
frequency, dimensions, and offsets with values from the sensor specification:

.. code-block:: python

   import numpy as np

   from ncore.data import RowOffsetStructuredSpinningLidarModelParameters
   from ncore.sensors import LidarModel

   n_rows = 128
   n_columns = 1200

   params = RowOffsetStructuredSpinningLidarModelParameters(
       spinning_frequency_hz=10.0,
       spinning_direction="cw",
       n_rows=n_rows,
       n_columns=n_columns,
       # Rows are always stored in strictly descending elevation order.
       row_elevations_rad=np.deg2rad(
           np.linspace(12.93, -12.47, n_rows)
       ).astype(np.float32),
       # Clockwise models use decreasing column azimuths.
       column_azimuths_rad=np.deg2rad(
           np.linspace(60.0, -60.0, n_columns, endpoint=False)
       ).astype(np.float32),
       row_azimuth_offsets_rad=np.zeros(n_rows, dtype=np.float32),
   )

   model = LidarModel.maybe_from_parameters(params, device="cpu")
   assert model is not None

The required ordering is independent of the sensor firing order. Rows must have
strictly descending elevation. Columns must follow ``spinning_direction``:
decreasing azimuth for ``"cw"`` and increasing azimuth for ``"ccw"``. NCore
validates both invariants when the parameter object is created.

Use zero row offsets when no trustworthy values are available. NCore defines the
effective azimuth as ``column_azimuth + row_offset``; convert a vendor's sign
convention before storing nonzero offsets.

Estimate a model from points
----------------------------

If factory calibration is unavailable, estimate the table from a representative
decompensated frame. Here, *decompensated* means that each point is expressed in
the LiDAR sensor frame at its own measurement time, rather than motion-compensated
to a common frame timestamp.

The reference helper expects a dense ``[n_columns * n_beams, 3]`` array in firing
order, with all beams of one firing column adjacent. It derives median elevation
per beam, median azimuth per column, row azimuth offsets, and the permutation from
sensor firing order to NCore's descending-elevation row order:

.. code-block:: python

   from tools.data_converter.structured_lidar_model import (
       derive_model_from_decompensated,
   )

   params = derive_model_from_decompensated(
       xyz_decompensated=xyz_decompensated,
       n_beams_per_column=128,
       n_target_cols=1200,
       spinning_direction="cw",
       spinning_frequency_hz=10.0,
       # Set this only when the sensor firing schedule is known.
       beam_pair_interval_us=0.0,
   )
   if params is None:
       raise ValueError(
           "The reference frame does not contain the expected structured scan"
       )

Use a frame with returns in at least 90 percent of the expected columns and with
enough distant structure to stabilize azimuth estimates. Invalid or no-return
points should have a distance below the helper's ``min_valid_distance_m`` value.
For sensors with mechanical spin-phase variation, consider
``upsample_model(params, resolution_factor=4)`` before aligning frames; the finer
grid reduces column-quantization error.

Assign ``model_element`` for every frame
----------------------------------------

A static model does not by itself identify the table cell of every measured ray.
The per-frame alignment step creates that mapping while accounting for spin phase
and vehicle motion. Prefer a native beam or firing ID for the row instead of a
nearest-elevation lookup.

``align_frame`` accepts a motion-compensated point cloud, beam IDs, intensity,
frame timestamps, and an initialized
``ncore.impl.common.transformations.MotionCompensator``. It iteratively aligns
the scan to the model, decompensates the points, and returns per-ray timestamps
and indices:

.. code-block:: python

   from tools.data_converter.structured_lidar_model import align_frame

   frame_data = align_frame(
       xyz_mc=xyz_motion_compensated,
       ring_index=ring_index,
       intensity=intensity,
       n_beams_per_column=128,
       model_params=params,
       motion_compensator=motion_compensator,
       sensor_id="lidar_top",
       frame_start_us=frame_start_us,
       frame_end_us=frame_end_us,
       # Pass native per-point timestamps when the source dataset provides them.
       timestamps_us=point_timestamps_us,
       model_resolution_factor=1,
   )
   if frame_data is None:
       raise ValueError("Insufficient valid columns to align this frame")

   xyz_decompensated = frame_data.xyz_decompensated
   point_timestamps_us = frame_data.timestamps_us
   model_element = frame_data.model_element  # [N, 2], uint16

Dataset converters with native firing timestamps can also reconstruct columns
directly from their timing metadata. The Argoverse 2
``reconstruct_model_elements`` implementation is an example: it maps native
laser IDs to model rows, derives columns from firing offsets, and estimates a
per-frame phase correction from decompensated far-range points.

Evaluate model quality
----------------------

Always compare model-predicted directions with the native, decompensated ray
directions before writing a full dataset. The reference helper returns the mean
angular error for all points, the mean error beyond a configurable distance, and
the systematic far-range azimuth shift:

.. code-block:: python

   from tools.data_converter.structured_lidar_model import (
       compute_model_consistency,
   )

   mean_all_deg, mean_far_deg, mean_az_shift_deg = compute_model_consistency(
       directions=native_directions,
       model_element=model_element,
       distances=distance_m,
       model_params=params,
       far_range_m=20.0,
   )
   print(f"mean angular error: {mean_all_deg:.4f} deg")
   print(f"far-range angular error: {mean_far_deg:.4f} deg")
   print(f"far-range azimuth shift: {mean_az_shift_deg:.4f} deg")

For an end-to-end NCore V4 sequence, use the evaluation CLI. It reports mean,
median, p95, per-row, and per-frame errors and can render camera overlays:

.. code-block:: bash

   bazel run //tools:ncore_evaluate_lidar_model -- \
       --source-id lidar_top \
       --camera-id camera_front \
       --output-dir /tmp/lidar_model_eval \
       --warn-threshold-deg 0.05 \
       v4 --component-group /path/to/sequence-ncore4.json

Treat ``0.05`` degrees as a starting warning threshold, not a universal sensor
specification. Select the acceptance threshold from the sensor's angular
resolution and the downstream projection error budget. Far-range error and
systematic azimuth bias are generally more diagnostic than an all-points mean,
which can be dominated by close-range motion-compensation artifacts. See
:doc:`../tools/lidar_model_eval` for all metrics and options.

Write the model to NCore V4
---------------------------

Store the static parameters once in the sequence intrinsics component. Store the
``model_element`` indices, decompensated unit directions, measurement timestamps,
and return values with every LiDAR frame:

.. code-block:: python

   intrinsics_writer.store_lidar_intrinsics(
       lidar_id="lidar_top",
       lidar_model_parameters=params,
   )

   distance_m = np.linalg.norm(xyz_decompensated, axis=1).astype(np.float32)
   direction = np.zeros_like(xyz_decompensated, dtype=np.float32)
   valid = distance_m > 0
   direction[valid] = xyz_decompensated[valid] / distance_m[valid, None]

   lidar_writer.store_frame(
       direction=direction,
       timestamp_us=point_timestamps_us.astype(np.uint64),
       model_element=model_element.astype(np.uint16),
       distance_m=distance_m[None, :],
       intensity=normalized_intensity.astype(np.float32)[None, :],
       frame_timestamps_us=np.array(
           [frame_start_us, frame_end_us], dtype=np.uint64
       ),
       generic_data={},
       generic_meta_data={},
   )

The writer requires unit-norm ``float32`` directions, ``uint64`` timestamps
inside the frame interval, ``uint16`` model indices, nonnegative distances, and
intensity normalized to ``[0, 1]``. For multiple returns, use arrays of shape
``[R, N]`` for distance and intensity while keeping one ``model_element`` per
ray.

Practical checks
----------------

Before converting the complete dataset, verify the following on multiple scenes:

* The row and column arrays satisfy NCore's strict ordering checks.
* Every ``model_element[:, 0]`` is smaller than ``n_rows`` and every
  ``model_element[:, 1]`` is smaller than ``n_columns``.
* Native directions and model-reconstructed directions use the same sensor axis
  convention: x forward, y left, z up.
* Elevation estimation and frame alignment use decompensated rays.
* Per-frame timestamps fall within the stored frame interval.
* Far-range angular error, systematic azimuth bias, and camera-overlay error meet
  the downstream quality target.

The nuScenes and Argoverse 2 converters provide complete integration examples
for empirical model estimation, per-frame alignment, optimization, evaluation,
and V4 writing.
