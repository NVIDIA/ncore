<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NCore KITTI Converter

Convert KITTI Raw dataset to NCore V4 format.

## Overview

This module provides tooling for converting KITTI Raw synchronized+rectified data to NCore V4
format. It is a standalone Bazel module that depends on the parent `ncore` module.

Supported sensors:
- 4 cameras (2 grayscale, 2 color) with global-shutter intrinsics
- 1 Velodyne HDL-64E lidar with per-point azimuth-based timestamps
- OXTS/GPS ego poses (Mercator-projected, rebased to first frame)
- 3D tracklet annotations (cuboids in Velodyne frame)

## Prerequisites

- NCore build requirements (see <CONTRIBUTING.md>)
- KITTI Raw data (download from <https://www.cvlibs.net/datasets/kitti/raw_data.php>)
  - Synced+rectified data
  - Calibration files
  - Tracklet labels (optional)

## Usage

```bash
bazel run //tools/data_converter/kitti -- \
    --root-dir /path/to/kitti/2011_09_26 \
    --output-dir /path/to/output/ncore \
    kitti-v4
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--root-dir` | Path to date directory containing calibration files and drive sequences | Required |
| `--output-dir` | Path where converted data will be saved | Required |
| `--verbose` | Enable debug logging | False |
| `--no-cameras` | Disable exporting cameras | False |
| `--camera-id` | Camera IDs to export (all if not specified) | All |
| `--no-lidars` | Disable exporting lidars | False |
| `--lidar-id` | Lidar IDs to export (all if not specified) | All |
| `--store-type` | Output store type (`itar` or `directory`) | `itar` |
| `--profile` | Component group profile (`default`, `separate-sensors`, `separate-all`) | `separate-sensors` |
| `--sequence-meta` / `--no-sequence-meta` | Generate sequence meta-data file | True |

### Examples

Convert all sequences in a date directory:

```bash
bazel run //tools/data_converter/kitti -- \
    --root-dir /data/kitti/2011_09_26 \
    --output-dir /data/ncore/kitti \
    kitti-v4
```

Convert only color cameras and lidar:

```bash
bazel run //tools/data_converter/kitti -- \
    --root-dir /data/kitti/2011_09_26 \
    --output-dir /data/ncore/kitti \
    --camera-id camera_color_left \
    --camera-id camera_color_right \
    kitti-v4
```

Convert to directory format:

```bash
bazel run //tools/data_converter/kitti -- \
    --root-dir /data/kitti/2011_09_26 \
    --output-dir /data/ncore/kitti \
    --store-type directory \
    kitti-v4
```

## Sensor IDs

| KITTI Directory | NCore Sensor ID |
|----------------|-----------------|
| `image_00` | `camera_gray_left` |
| `image_01` | `camera_gray_right` |
| `image_02` | `camera_color_left` |
| `image_03` | `camera_color_right` |
| `velodyne_points` | `lidar_top` |

## License

Apache 2.0 - See LICENSE file in the repository root.
