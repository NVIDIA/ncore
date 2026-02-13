<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION
SPDX-License-Identifier: Apache-2.0
-->

# NCore Waymo Converter

Convert Waymo Open Dataset tfrecords to NCore V4 format.

## Overview

This module provides tooling for converting Waymo Open Dataset tfrecords to NCore V4 format.
It is a standalone Bazel module that depends on the parent `ncore` module.

## Prerequisites

- Bazel 8.5.0+ (via Bazelisk)
- Waymo Open Dataset tfrecords (download from <https://waymo.com/intl/en_us/open/download/>)

## Usage

```bash
cd modules/ncore_waymo
bazel run convert -- \
    --root-dir /path/to/waymo/tfrecords \
    --output-dir /path/to/output/ncore
    waymo-v4
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--root-dir` | Path to raw data sequences (directory with tfrecords) | Required |
| `--output-dir` | Path where converted data will be saved | Required |
| `--verbose` | Enable debug logging | False |
| `--no-cameras` | Disable exporting cameras | False |
| `--camera-id` | Camera IDs to export (all if not specified) | All |
| `--no-lidars` | Disable exporting lidars | False |
| `--lidar-id` | Lidar IDs to export (all if not specified) | All |
| `--no-radars` | Disable exporting radars | False |
| `--radar-id` | Radar IDs to export (all if not specified) | All |
| `--store-type` | Output store type (`itar` or `directory`) | `itar` |
| `--profile` | Component group profile (`default`, `separate-sensors`, `separate-all`) | `separate-sensors` |
| `--sequence-meta` / `--no-sequence-meta` | Generate sequence meta-data file | True |

### Examples

Convert all sequences with default settings:

```bash
bazel run //ncore_waymo:convert -- \
    --root-dir /data/waymo/training \
    --output-dir /data/ncore/waymo
```

Convert only front camera and top lidar:

```bash
bazel run //ncore_waymo:convert -- \
    --root-dir /data/waymo/training \
    --output-dir /data/ncore/waymo \
    --camera-id camera_front_50fov \
    --lidar-id lidar_top
```

Convert to directory format (instead of itar):

```bash
bazel run //ncore_waymo:convert -- \
    --root-dir /data/waymo/training \
    --output-dir /data/ncore/waymo \
    --store-type directory
```

## Sensor Mapping

### Cameras

| Waymo Name | NCore ID |
|------------|----------|
| FRONT | camera_front_50fov |
| FRONT_LEFT | camera_front_left_50fov |
| FRONT_RIGHT | camera_front_right_50fov |
| SIDE_LEFT | camera_side_left_50fov |
| SIDE_RIGHT | camera_side_right_50fov |

### Lidars

| Waymo Name | NCore ID |
|------------|----------|
| TOP | lidar_top |

## Development

### Regenerating Locked Requirements

```bash
cd modules/ncore_waymo
bazel run //3rdparty/python:requirements_3_11
```

### Running with Type Checking

```bash
bazel build //ncore_waymo:convert --config=mypy
```

## License

Apache 2.0 - See LICENSE file in the repository root.
