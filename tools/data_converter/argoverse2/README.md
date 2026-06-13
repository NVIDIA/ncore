# Argoverse 2 to NCore V4 Converter

Converts [Argoverse 2](https://www.argoverse.org/av2.html) Sensor Dataset logs to
NCore V4 format.

The converter reads the Argoverse 2 on-disk Apache Feather files directly with
`pyarrow`, deliberately avoiding the heavy `av2` devkit (which pulls in torch,
kornia, numba, polars and PyAV). Quaternion handling uses `scipy` (already an
ncore dependency), so no extra dependency is introduced.

## Requirements

- Argoverse 2 Sensor Dataset downloaded locally, organised as
  `{root}/{split}/{log_id}/...`
- Python packages: `pyarrow` (plus `scipy`, already an ncore dependency)

## Usage

```bash
bazel run //tools/data_converter/argoverse2 -- \
    --root-dir /path/to/argoverse2/sensor \
    --output-dir /path/to/output \
    argoverse2-v4 \
    --split val
```

### Convert a single log

```bash
bazel run //tools/data_converter/argoverse2 -- \
    --root-dir /path/to/argoverse2/sensor \
    --output-dir /path/to/output \
    argoverse2-v4 \
    --split val \
    --log-id 02678d04-cc9f-3148-9f95-1ba66347dff9
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--split` | val | Split directory under `--root-dir` (train, val, test) |
| `--log-id` | None | Filter to a single log by ID |
| `--store-type` | itar | Output store format (itar or directory) |
| `--profile` | separate-sensors | Component group assignment profile |
| `--sequence-meta/--no-sequence-meta` | enabled | Generate sequence meta JSON |

## Sensor Assumptions

- **Cameras**: 9 cameras (7 ring + 2 stereo). AV2 imagery is shipped already
  undistorted -- the official av2 devkit projects with the intrinsic matrix `K`
  only and does not load the `k1, k2, k3` columns -- so the stored model is an
  ideal (distortion-free) pinhole (`IdealPinholeCameraModelParameters`). Because
  the imagery is already undistorted, global shutter is assumed
  (`ShutterType.GLOBAL`). The `k1, k2, k3` coefficients in `intrinsics.feather`
  describe the original lens (for re-distorting into the raw frame) and must not be
  applied to the released images, so they are not used for projection -- but they
  are preserved per camera in the camera component `generic_meta_data` under
  `av2_original_distortion` so the original calibration is not lost.
- **Lidar**: two stacked Velodyne VLP-32C units (`up_lidar` / `down_lidar`, 10 Hz).
  The source sweep is egomotion-compensated to the sweep reference timestamp and
  expressed in the egovehicle frame. Real per-point timestamps are available via
  `offset_ns`. Each unit is stored separately with its own extrinsic. Points are
  mapped into each unit's sensor frame and decompensated using the real per-point
  timestamps so the stored directions are raw per-point-time measurements. Because
  the extrinsic is static, this is independent of whether AV2 applied ego-motion
  before or after the sensor transform.
  - A structured VLP-32C model is stored per unit as lidar intrinsics, with
    per-point `model_element` (row, column). AV2 provides no native firing-column
    index, so the firing pattern is reconstructed from `offset_ns` (firing columns --
    one VLP-32C revolution at 10 Hz) and `laser_number` (the beam, mapped to an
    elevation-sorted row). The geometry is derived per log from the *decompensated*
    reference sweep: elevations, the laser->row map, column timing, per-column
    azimuths, and per-row azimuth offsets (the 32 beams of a firing column span
    several degrees of azimuth, so the per-row offset is fit empirically rather than
    assumed). The two stacked units fire in opposite phase, so they spin oppositely
    in their own frames (one `cw`, one `ccw`); this is detected from the data. The
    column grid is upsampled 4x so the per-frame alignment is not column-quantized.
    Each sweep is re-aligned to the model by a per-frame affine column remap (a
    constant phase plus a linear term): the spin phase at a given `offset_ns` drifts
    ~1 deg between sweeps (the constant), and the spin rate drifts slightly within a
    sweep on some scenes (the linear term). A fixed mapping, or a phase-only shift,
    would leave some frames off by up to ~1 deg / ~0.25 deg respectively.
    Steep downward beams that only return at near range (no far data, e.g. the
    lowest laser at ~-25 deg) have their azimuth offset fit from near-range returns.
    Deriving from the decompensated cloud (not the ego-motion-smeared compensated
    one) plus these steps gives ~0.03-0.05 deg median far-range reconstruction across
    scenes, on par with native-column sensors. Pass `--lidar-model-source none` to
    store raw ray bundles only.
  - The `laser_number` to up/down unit split is not documented by AV2. The two
    units occupy the two laser-number halves (`< 32` and `>= 32`); the unit *label*
    is recovered from extrinsic geometry by per-beam elevation flatness (a laser
    ring traces a constant-elevation cone only in its own sensor frame, so the
    wrong extrinsic tilts the cone and inflates the per-ring elevation spread). The
    decision is made once per log and is stable with a wide (~2-10x) margin.
- **Radar**: AV2 has no radar.
- **Cuboid annotations**: native to the egovehicle frame at the sweep reference
  time, stored in the `rig` frame at that timestamp with no ego pose baked in. This
  keeps the egovehicle motion out of the stored coordinates so it stays swappable
  downstream (a V4 feature); the pose graph places the cuboids using the active ego
  trajectory. The full 3-DOF box orientation is preserved (the AV2 quaternion is
  converted to the BBox3 `xyz`-Euler convention, not reduced to yaw). `track_uuid`
  is used as track ID.

## Coordinate frames

The first ego pose's `city_SE3_egovehicle` is stored as the static
`world -> world_global` pose, so `world_global` is the AV2 city frame. All absolute
city coordinates remain recoverable for later alignment with the AV2 HD map (which
the converter does not export).

## Testing

```bash
AV2_DIR=/path/to/argoverse2/sensor AV2_SPLIT=val \
    bazel test //tools/data_converter/argoverse2:pytest_converter
```
