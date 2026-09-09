<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# V4 invariants not covered by the repo docs

Enough to read a failure and to know what an adapted converter must satisfy.
Everything here is either absent from `docs/` or stated only in code. For
anything else, prefer the specification: `docs/data/conventions.rst` for frames
and transformations, `docs/data/formats.rst` for the component layout,
`docs/data/sensor_models.rst` for camera and lidar models.

## Time

- The sequence interval is **half-open**, `[start, stop)`, in microseconds. The
  inclusive last timestamp is therefore `stop - 1`. `docs/` gives the field
  name but not the semantics; see `HalfClosedInterval` in
  `ncore/impl/common/transformations.py`.
- `store_dynamic_pose(..., require_sequence_time_coverage=True)` is the
  default, and asserts the first timestamp equals the sequence start and the
  last equals `stop - 1`. Setting it `False` relaxes only that endpoint
  exactness. Containment, strict increase, and a minimum of two poses are
  enforced either way. The COLMAP converter passes `False` because per-camera
  pose tracks are ragged.
- Frames are stored as `[start-of-frame, end-of-frame]` per frame, and keyed by
  the end-of-frame timestamp. The format does not define these as exposure or
  sweep bounds; a global-shutter camera legitimately stores `start == end`, as
  COLMAP does.
- Out-of-range frames are **not clamped**. The writer asserts both endpoints
  lie inside the sequence interval and fails with `Frame start timestamp must
  be contained in the sequence time range`. Trim frames or widen the interval
  yourself. Converters generally skip such frames rather than adjusting them.

## Poses

- Pose dtype is not fixed by the format. Both `store_static_pose` and
  `store_dynamic_pose` accept and round-trip float32 and float64. If a
  downstream consumer raises `RuntimeError: double != float`, match that
  consumer rather than assuming V4 requires float32. Note the singular getters
  return float64 regardless of what was stored, while the plural generators
  honour the stored dtype.
- Pose **density** drives motion compensation, independently of the coverage
  rule above. Synchronized cameras share one trigger time, so N cameras and M
  frames dedupe to about M unique waypoints. Adding cameras does not densify
  the trajectory.
- Re-referencing poses to the first pose is a precision measure, not a V4 rule.
  KITTI, PAI, nuScenes and Argoverse 2 do it unconditionally; Waymo only under
  `--world-global-mode localized`; COLMAP never. The rationale and the
  `T_world_world_global` edge are in `docs/data/conventions.rst`.
- The ego edge is typical, not required. The COLMAP converter stores
  `source_frame_id=<camera_id>` against the camera's own reference frame.

## Cameras

- FTheta stores its principal point in **pixel-center** convention, where an
  index names the center of a pixel. The runtime adds half a pixel to reach the
  corner-origin image convention that the rest of the library uses, and
  subtracts it again on the way out. So for a 1920x1080 image whose optical
  center is the image center, the **stored** value is `[959.5, 539.5]` and the
  **runtime** value is `[960.0, 540.0]`. Getting this backwards is a
  half-pixel error in exactly the direction the convention exists to prevent.
  See `docs/data/sensor_models.rst` for the convention and
  `ncore/impl/sensors/camera.py` for the offset.
- No camera model carries a per-row shutter-delay field. Rolling-shutter timing
  comes entirely from the frame's `[start, end]` interval, with per-row times
  interpolated from the row index. PAI's source data has a `shutter_delay_us`,
  but the converter consumes it to derive the frame start and does not persist
  it.

## Lidar

- `model_element` is optional on `LidarSensorComponent.store_frame`: pass
  `None` for a sensor without a row and column grid, and per-ray `timestamp_us`
  still works. It has no default, so it must be passed explicitly.
- `row_elevations_rad` must be **strictly decreasing**, row 0 highest. The
  check is expressed as a clockwise relative-angle comparison, so the failure
  message talks about descending order.
- `spinning_direction` controls azimuth ordering and sweep interpretation only.
  Ray `z` is `sin(elevation)`, so a wrong value cannot flip the cloud
  vertically. Derive it per unit from the data where units may differ, as
  Argoverse 2 does.
- Per-column firing time uses two deliberately different conventions. The
  runtime lidar model maps a column onto the **closed** frame interval, so
  column `N-1` lands exactly on the frame end and the divisor is `n_columns - 1`.
  The converter helper treats a revolution as **half-open**, with the frame end
  belonging to the next frame's column 0, so its divisor is `n_columns`. Each
  is correct for its consumer. Follow whichever the code you are working in
  already uses, and do not port the expression between them.

## Components

- `MasksComponent` stores one untimestamped named mask set per camera and
  cannot represent anything varying per frame. For per-frame typed labels such
  as depth or semantic segmentation, prefer `CameraLabelsComponent`, which is
  independently timestamped. That is a recommendation, not the only route: the
  COLMAP converter carries per-image masks as per-frame `CameraSensor`
  `generic_data["mask"]` and writes an empty static mask set.
- `PointCloudsComponent` can carry per-point `timestamp_us` as a
  schema-declared `INVARIANT` attribute, so it does not lose timing. Prefer
  `LidarSensorComponent` for ray-bundle semantics and automatic per-ray motion
  compensation, not because PointClouds cannot hold timestamps.
- Cuboid `BBox3.centroid` is the **geometric center**, with `dim` a symmetric
  extent about it and `rot` XYZ Euler angles. If a source is bottom-centered,
  add `dim_z / 2`. Check the sign first: a ground-resting box gives
  `mean(z) - mean(h)/2 approx. -mean(h)/2` when bottom-centered and approx. 0
  when already centered.
- IMU is not a first-class component. Raw IMU samples are not SE(3) poses:
  integrate them to estimate or densify the ego trajectory, and keep the raw
  GPS/IMU stream in the Poses component's `generic_data`, which the format
  explicitly supports and KITTI already uses for OXTS.

## Motion compensation

Not documented outside docstrings; `MotionCompensator` lives in
`ncore/impl/common/transformations.py` and is not exported from a public
package.

`motion_decompensate_points` expects `xyz_reftime` **already in the sensor
frame** at one reference timestamp, namely the timestamp the data was
compensated to. Transform world or ego-frame points into that frame first, and
pass the matching `reference_timestamp_us` and `anchor_frame_id`, where the
anchor must match the one used for compensation. Feeding world-frame points
straight in silently corrupts both range and direction rather than raising.

## Sensor ids differ between converters

Ids are not portable, so a `--source-id` copied from another dataset's example
will not resolve:

| Converter | Lidar | Cameras |
|---|---|---|
| KITTI, nuScenes | `lidar_top` | nuScenes: `camera_front`, `camera_front_left`, `camera_back_left`, and so on |
| PAI | `lidar_top_360fov` | `camera_front_wide_120fov`, `camera_rear_left_70fov`, and so on |

Note nuScenes says `back` where PAI says `rear`, and PAI's lidar carries a
field-of-view suffix. Read the actual ids off `//tools:ncore_sequence_meta`.
