---
name: ncore-data-conversion
description: >-
  Use when getting sensor data into an NCore V4 store: running a built-in
  converter (PAI, Waymo, COLMAP/ScanNet++, KITTI, nuScenes, Argoverse 2),
  adapting the nearest built-in converter for a dataset NCore does not ship
  one for, or checking a converted store before something downstream reads
  it. Covers V4 conventions for poses, camera and lidar models, cuboids and
  timestamps. Do NOT use to train reconstructions or to extract per-object
  3D assets.
license: Apache-2.0
metadata:
  author: NVIDIA NCore
  version: "0.3.0"
  tags: ncore, data-conversion, sensors, v4, zarr, itar
  upstream: https://github.com/NVIDIA/ncore
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NCore V4 data conversion

Get sensor data into an **NCore V4** store, and check it before anything
downstream consumes it. A V4 store is a general-purpose sensor-data format:
NuRec is one consumer, so are `ncore_vis`, gsplat-based research code, and
your own tooling. Nothing here assumes a NuRec pipeline.

Three jobs, in order of preference: run a built-in converter; adapt the
nearest built-in converter when none fits; check the result either way.

This skill ships inside the ncore repository, so paths like
`docs/data/conventions.rst` refer to files you already have. Prefer them over
the rendered site, since they match your checkout.

## Prerequisites

- **Bazel**, via `bazelisk`. Not optional: converters and inspection tools are
  Bazel targets. The `nvidia-ncore` wheel packages only the `ncore` library and
  declares no console entry points, so `ncore_vis` and friends are reachable
  only from a checkout. See `CONTRIBUTING.md`.
- **A GitHub PAT with `read:packages` in `~/.netrc`**, per `CONTRIBUTING.md`.
  External test-data and docs archives are fetched from
  `maven.pkg.github.com`; without it the first `bazel build` fails in a way
  that looks unrelated to what you were doing.
- **An NVIDIA GPU for the PAI converter.** Its camera path decodes H.264 on the
  GPU through PyNvVideoCodec, imported at module scope, so the binary needs the
  GPU stack even to start. The other five converters have no such requirement.
- **For PAI on a gated clip**: `HF_TOKEN` set, on an account that has accepted
  the `nvidia/PhysicalAI-Autonomous-Vehicles` license. A valid token without
  accepted access fails as an opaque HTTP error, not a helpful message.

## Choose a converter

Six converters under `tools/data_converter/`. Pick by dataset; if none matches,
see [references/adapting-a-converter.md](references/adapting-a-converter.md).

| Dataset | Target | Subcommand |
|---|---|---|
| PAI clip (HuggingFace or local) | `//tools/data_converter/pai:convert` | `pai-stream-v4`, `pai-v4` |
| Waymo `.tfrecord` | `//tools/data_converter/waymo:convert` | `waymo-v4` |
| COLMAP scene, ScanNet++ DSLR | `//tools/data_converter/colmap:convert` | `colmap-v4`, `scannetpp-v4` |
| KITTI raw | `//tools/data_converter/kitti:convert` | `kitti-v4` |
| nuScenes | `//tools/data_converter/nuscenes:convert` | `nuscenes-v4` |
| Argoverse 2 Sensor | `//tools/data_converter/argoverse2:convert` | `argoverse2-v4` |

Flag documentation is split between two places, and which one is fuller
depends on the converter. Prefer `--help` over both:

| Converter | Fuller flag reference |
|---|---|
| Waymo, COLMAP, PAI | `docs/conversions/<name>/<name>.rst` |
| KITTI, nuScenes, Argoverse 2 | `tools/data_converter/<name>/README.md` (the `.rst` defers to it) |

Behavior worth knowing before you pick or adapt one:

- **nuScenes** treats cameras as global shutter: one capture timestamp per
  image, no rolling-shutter metadata, so frame start equals frame end. Its
  lidar model is derived from the data by default (`--lidar-model-source`,
  default `empirical`) from 1085 native firing columns, upsampled 4x. Cuboid
  positions are already geometric centers. Spin direction is hard-coded `cw`.
- **Argoverse 2** aggregates two stacked units that spin oppositely in their
  own frames, so `spinning_direction` is derived per unit and can be `ccw`.
  `--lidar-model-source none` stores raw ray bundles with no structured model
  and no lidar intrinsics.
- **COLMAP** timestamps are synthetic, one second per frame. It never infers a
  frame rate from filenames.
- **PAI** uses Hyperion 8 sensor ids (`camera_front_wide_120fov`,
  `lidar_top_360fov`, and so on) and needs the GPU noted above.

## Run a converter

Each `convert` binary takes shared base flags (`--root-dir`, `--output-dir`,
`--verbose`, and `--no-cameras` / `--camera-id` / `--no-lidars` / `--lidar-id`
/ `--no-radars` / `--radar-id` to restrict which sensors are processed),
followed by a subcommand carrying format-specific flags. The sensor-restriction
flags are for narrowing a run during development, not for making an unsupported
host work.

Common subcommand flags, though not every subcommand offers every one:

| Flag | Default | Meaning |
|---|---|---|
| `--store-type {itar,directory}` | `itar` | `itar` is compact and self-contained; `directory` is easier to inspect |
| `--profile {default,separate-sensors,separate-all}` | `separate-sensors` | Component group layout. Every converter defaults to `separate-sensors`, including Waymo |
| `--sequence-meta` / `--no-sequence-meta` | enabled | Writes `<output-dir>/<sequence_id>/<sequence_id>.json`, the file that expands to every shard. `scannetpp-v4` always writes it and offers no switch |
| `--world-global-mode` | `none` | Only `waymo-v4` and `colmap-v4` take it; `localized` is Waymo-only. The other four converters store a `world` to `world_global` pose unconditionally |

Assign real values before running; never paste angle-bracket placeholders into
a shell, which reads them as redirections.

Waymo:

```bash
OUT_DIR=                          # FILL IN
TFRECORD_DIR=                     # FILL IN
: "${OUT_DIR:?set OUT_DIR}" "${TFRECORD_DIR:?set TFRECORD_DIR}"

bazel run //tools/data_converter/waymo:convert -- \
    --root-dir "$TFRECORD_DIR" \
    --output-dir "$OUT_DIR" \
    waymo-v4
```

nuScenes, one scene:

```bash
NUSCENES_ROOT=                    # FILL IN
OUT_DIR=                          # FILL IN
: "${NUSCENES_ROOT:?set NUSCENES_ROOT}" "${OUT_DIR:?set OUT_DIR}"

bazel run //tools/data_converter/nuscenes:convert -- \
    --root-dir "$NUSCENES_ROOT" \
    --output-dir "$OUT_DIR" \
    nuscenes-v4 \
        --version v1.0-trainval \
        --scene-name scene-0001   # or --scene-token; omit for all scenes
```

PAI, streaming, which needs no `--root-dir`. The converter reads the token from
`HF_TOKEN`; `--hf-token` exists on `pai-stream-v4` and defaults to that
variable, so do not pass it explicitly. Doing so puts the secret in the
process's argv, readable from `/proc/<pid>/cmdline` for the whole run, and in
your shell history.

```bash
OUT_DIR=                          # FILL IN
CLIP_ID=                          # FILL IN
: "${OUT_DIR:?set OUT_DIR}" "${CLIP_ID:?set CLIP_ID}"

bazel run //tools/data_converter/pai:convert -- \
    --output-dir "$OUT_DIR" \
    --camera-id camera_front_wide_120fov \
    pai-stream-v4 \
        --clip-id "$CLIP_ID"
```

Different sequences coexist safely, since converters write under
`<output-dir>/<sequence_id>/`. Re-running the *same* sequence to an existing
`itar` store truncates it before producing anything, so write to a fresh path
or back up first. The `directory` store type does not have this behavior.

## Check the result

There is no validator, and no single check establishes correctness. What
follows is a progression from cheap to thorough.

Reading a store back is itself a check: opening a sequence enforces version
agreement, rejects duplicate component groups, and verifies that sequence ids
and timestamp intervals agree across shards.

Under the default `separate-sensors` profile a sequence is split into one shard
per sensor plus a default shard holding poses, intrinsics, masks and cuboids.
Only the sequence `.json` expands to all of them. A single bare `.zarr` or
`.zarr.itar` path works only if that one store holds every component you need;
otherwise pass the JSON, or repeat `--component-group` once per shard.

**1. Dump the metadata.** Fastest way to see what actually landed:

```bash
SEQ_META=                         # FILL IN: path to the sequence .json
OUT_DIR=                          # FILL IN
: "${SEQ_META:?set SEQ_META}" "${OUT_DIR:?set OUT_DIR}"

bazel run //tools:ncore_sequence_meta -- \
    --output-dir="$OUT_DIR" \
    v4 --component-group="$SEQ_META"
```

**2. Look at it.** `ncore_vis` makes wrong extrinsics and rotated cameras
obvious. It defaults to `--host=0.0.0.0` and its Viser server is
unauthenticated, which on a shared or cloud host exposes your sensor imagery
and geometry to the network. Bind loopback unless you mean otherwise.

```bash
bazel run //tools/ncore_vis -- \
    --host=127.0.0.1 \
    v4 --component-group="$SEQ_META"
```

See `docs/tools/ncore_vis.rst` for the viewer's own options.

**3. Project lidar onto camera**, where the sequence has both. This exercises
extrinsics, intrinsics, per-ray timestamps and pose density together, which is
what makes it useful and also what makes a failure ambiguous. See
[projections that do not line up](#when-projections-do-not-line-up) for
separating the causes.

```bash
SOURCE_ID=                        # FILL IN, e.g. lidar_top
CAMERA_ID=                        # FILL IN, e.g. camera_front
: "${SOURCE_ID:?set SOURCE_ID}" "${CAMERA_ID:?set CAMERA_ID}"

bazel run //tools:ncore_project_pc_to_img -- \
    --source-id="$SOURCE_ID" \
    --camera-id="$CAMERA_ID" \
    --output-dir="$OUT_DIR" \
    --device=cpu \
    v4 --component-group="$SEQ_META"
```

Use ids your store actually contains. No converter emits `lidar00` or
`camera01`; KITTI and nuScenes write `lidar_top`, nuScenes cameras are
`camera_front`, `camera_back_left` and so on, while PAI writes
`lidar_top_360fov` and `camera_rear_left_70fov`. Read them off
`ncore_sequence_meta` if unsure.

Two defaults to keep in mind: `--no-lidar-model` means the projection uses
stored ray directions and does not exercise the structured lidar model, and
`--no-external-distortion` means a camera carrying external distortion can look
misaligned until you pass `--external-distortion`. `--device` defaults to
`cuda` here, but to `cpu` in `ncore_evaluate_lidar_model`.

## When projections do not line up

Projected lidar points that do not sit on image features are the common
failure. It is usually inherited from the source data rather than introduced by
conversion, and three independent causes produce the same picture: extrinsics,
intrinsics, and timestamps. The appearance alone does not tell you which, so
separate them by choosing what to project. The lists below are common cases,
not an exhaustive set.

### Separate calibration from timing: project at standstill

Find frames where the rig is not moving and project only those, using
`--start-frame` and `--stop-frame`. Without ego motion, timestamp errors have
no geometric effect, so any misalignment that survives is calibration.

- Unaligned at standstill: extrinsics or intrinsics.
- Aligned at standstill, unaligned once moving: timing.

`--pose {start,end,mean,rolling-shutter}` is the second lever. If switching
between `start` and `end` visibly changes the projection, the frame interval is
doing real work and its endpoints are worth checking.

### Unaligned at standstill

- The whole cloud is offset or rotated rigidly against otherwise consistent
  image content. Look at the lidar-to-rig static extrinsic, and at the source
  data's own lidar frame convention. The source is the more common cause, and
  conversion cannot correct it.
- Alignment is good near the image center and degrades toward the edges.
  Intrinsics: distortion coefficients, or the FTheta principal-point
  convention, which is stored pixel-centered with the runtime adding half a
  pixel.
- Model-predicted ray directions disagree with the stored native ones. Measure
  it rather than guessing: `//tools:ncore_evaluate_lidar_model` reports angular
  error and any systematic azimuth shift, and
  `docs/tools/lidar_model_eval.rst` gives expected magnitudes. Remember the
  projector does not use the model unless you pass `--lidar-model`.

### Aligned at standstill, unaligned in motion

- Misalignment grows with vehicle speed. Per-ray timestamps or the frame
  interval. A sweep midpoint stored as the frame start shifts every ray by half
  a sweep; `frame_timestamps_us[0]` is the real start of frame.
- Misalignment grows with rotation rate rather than speed, or points appear
  drawn out along the motion direction. The pose trajectory is too sparse to
  interpolate through. Combine every available pose source and densify from
  IMU or odometry. Adding more cameras does not help: synchronized cameras
  share a trigger time, so N cameras and M frames still dedupe to about M
  unique waypoints.
- A rolling-shutter camera shows internal inconsistency rather than a uniform
  offset. Either one global timestamp is being stored for the whole frame, or
  `ShutterType.GLOBAL` is set on a rolling sensor. Store a real
  `[start, end]` interval per camera and the matching enum.

### The writer rejected the data

- `Frame start timestamp must be contained in the sequence time range`. The
  frame falls outside the half-open sequence interval. Trim the frame or widen
  the interval; the writer does not clamp.
- An assertion that row elevations must be sorted descending. Sort
  `row_elevations_rad` descending and separate duplicates by about 1e-6 rad.
- Dynamic poses must cover the full sequence time range. The pose track starts
  after the sequence start or ends before its last microsecond. Extend it, or
  pass `require_sequence_time_coverage=False` if per-sensor pose tracks are
  legitimately ragged, as the COLMAP converter does.

### Other symptoms

- Cuboids sit half-buried in the ground. A bottom-centered origin was stored as
  a centroid. Add `dim_z / 2`, after checking the sign: a ground-resting box
  gives `mean(z) - mean(h)/2 approx. -mean(h)/2` when bottom-centered and
  approx. 0 when already centered.
- The ego vehicle is reconstructed as part of the scene. No ego mask. Store a
  per-camera binary mask through `MasksComponent.store_camera_masks`.
- Most rays in a Livox-style scan have range 0. A structured spinning model was
  forced onto a non-repetitive scan pattern. Keep `LidarSensorComponent` and
  pass `model_element=None`.
- `RuntimeError: double != float` in a downstream consumer. That consumer wants
  float32. V4 stores either, so cast to what the failing consumer expects
  rather than assuming the format requires it.
- Sub-centimetre detail is lost at scene scale. Large global coordinates
  (UTM, ECEF) narrowed to float32. Re-reference poses to the first pose, but
  only while preserving `T_world_world_global`; otherwise keep float64. See
  `docs/data/conventions.rst` on the local `world` frame.

## Scope and constraints

- Conversion is one way. No tool reconstructs the original dataset from a V4
  store, so keep the source.
- Cuboid dimensions come from source annotations. Nothing here estimates them.
- Checks here are inspection, not verification. They catch common structural
  and calibration errors; they do not establish every schema invariant.

## Reference

- V4 facts not covered by the repo docs:
  [references/v4-invariants.md](references/v4-invariants.md)
- Writing a converter for an unsupported dataset:
  [references/adapting-a-converter.md](references/adapting-a-converter.md)
- Format specification: `docs/data/conventions.rst`, `docs/data/formats.rst`
- Camera and lidar models: `docs/data/sensor_models.rst`
- Store types and performance: `docs/data/storage_and_access.rst`
- Tools: `docs/tools/` (`ncore_vis`, `data_vis`, `ncore_sequence_meta`,
  `lidar_model_eval`)
- Per-converter guides: `docs/conversions/`, `tools/data_converter/*/README.md`
- Contributing, build setup and repo gates: `CONTRIBUTING.md`
- Published docs: <https://nvidia.github.io/ncore/>
- Library only, without the tools: `pip install nvidia-ncore`
