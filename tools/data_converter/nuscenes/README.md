# nuScenes to NCore V4 Converter

Converts [nuScenes](https://www.nuscenes.org/) dataset scenes to NCore V4 format.

## Requirements

- nuScenes dataset downloaded locally (any version: v1.0-mini, v1.0-trainval, v1.0-test)
- Python packages: `nuscenes-devkit`, `pyquaternion`

## Usage

```bash
bazel run //tools/data_converter/nuscenes -- \
    --root-dir /path/to/nuscenes \
    --output-dir /path/to/output \
    nuscenes-v4 \
    --version v1.0-trainval
```

### Convert a single scene by token

```bash
bazel run //tools/data_converter/nuscenes -- \
    --root-dir /path/to/nuscenes \
    --output-dir /path/to/output \
    nuscenes-v4 \
    --version v1.0-mini \
    --scene-token cc8c0bf57f984915a77078b10eb33198
```

### Convert a single scene by name

```bash
bazel run //tools/data_converter/nuscenes -- \
    --root-dir /path/to/nuscenes \
    --output-dir /path/to/output \
    nuscenes-v4 \
    --version v1.0-mini \
    --scene-name scene-0061
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--version` | v1.0-trainval | nuScenes version string |
| `--scene-token` | None | Filter to a single scene by token |
| `--scene-name` | None | Filter to a single scene by name |
| `--store-type` | itar | Output store format (itar or directory) |
| `--profile` | separate-sensors | Component group assignment profile |
| `--sequence-meta/--no-sequence-meta` | enabled | Generate sequence meta JSON |
| `--lidar-model-source` | empirical | Model derivation: `empirical` (from data) or `nominal` (from HDL-32E spec) |
| `--lidar-model-resolution` | 4 | Model column resolution factor (1/2/4). Higher = finer alignment. |
| `--lidar-model-optimization-passes` | 1 | Multi-frame optimization iterations (0 to disable) |

Recommended for best quality: `--lidar-model-source empirical --lidar-model-resolution 4 --lidar-model-optimization-passes 1`.
Across the 10 v1.0-mini scenes plus a trainval sample, `empirical` reproduces the point
cloud up to ~15x more accurately at far range than `nominal` (and never worse), so it is
the default. Use `nominal` only when a data-independent model is required. Expected mean
far-range (> 20 m) angular error, measured with `//tools:ncore_evaluate_lidar_model`:

| `--lidar-model-source` | Mean far-range error | Systematic azimuth bias |
|------------------------|----------------------|-------------------------|
| `empirical` (default)  | ~0.02 - 0.04 deg     | < ~0.005 deg            |
| `nominal`              | ~0.02 - 0.6 deg      | up to ~0.4 deg          |

## Sensor Assumptions

- **Cameras**: Treated as global shutter (ShutterType.GLOBAL). nuScenes provides a single
  capture timestamp per image with no rolling-shutter metadata. Images are already undistorted,
  so all distortion coefficients are zero.
- **Lidar**: Velodyne HDL-32E spinning lidar at 20 Hz. Source point clouds are
  motion-compensated; the converter decompensates them to raw per-point-time
  measurements. Per-point timestamps are derived from the 32-beam column structure.
  A structured lidar model is stored as intrinsics with configurable derivation:
  - *Empirical* (`--lidar-model-source empirical`, default): derived from a decompensated
    reference frame with analytical blending for data-poor beams. Best accuracy.
  - *Nominal* (`--lidar-model-source nominal`): from HDL-32E spec (uniform azimuths,
    spec elevations, analytical firing offsets). No circular data dependency, but
    lower fidelity than empirical.
  - *Resolution upsampling* (`--lidar-model-resolution 4`): interpolates the model
    to 4x column resolution, reducing alignment quantization from ~0.09 to ~0.02 deg.
  - *Optimization* (`--lidar-model-optimization-passes 1`): multi-frame median
    correction of azimuths and offsets.
- **Cuboid annotations**: Stored in the world coordinate frame. Only keyframe annotations
  are included.

## Testing

```bash
NUSCENES_DIR=/path/to/nuscenes NUSCENES_VERSION=v1.0-mini \
    bazel test //tools/data_converter/nuscenes:pytest_converter
```
