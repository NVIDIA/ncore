<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NCore Waymo Converter

Convert Waymo Open Dataset tfrecords to NCore V4 format.

## Overview

This module provides tooling for converting Waymo Open Dataset tfrecords to NCore V4 format.
It is a standalone Bazel module that depends on the parent `ncore` module.

## Prerequisites

- NCore build requirements (see <CONTRIBUTING.md>)
- Waymo Open Dataset tfrecords (download from <https://waymo.com/intl/en_us/open/download/>)

## Usage

```bash
bazel run //tools/data_converter/waymo -- \
    --root-dir /path/to/waymo/tfrecords \
    --output-dir /path/to/output/ncore \
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
| `--world-global-mode` | Controls `("world", "world_global")` static pose storage: `none` (omit), `identity` (store identity matrix), `localized` (rebase poses relative to the first frame, matching, e.g., the PAI converter pattern) | `none` |

### Examples

Convert all sequences with default settings:

```bash
bazel run //tools/data_converter/waymo -- \
    --root-dir /data/waymo/training \
    --output-dir /data/ncore/waymo \
    waymo-v4
```

Convert only front camera and top lidar:

```bash
bazel run //tools/data_converter/waymo -- \
    --root-dir /data/waymo/training \
    --output-dir /data/ncore/waymo \
    --camera-id camera_front_50fov \
    --lidar-id lidar_top \
    waymo-v4
```

Convert to directory format (instead of itar):

```bash
bazel run //tools/data_converter/waymo -- \
    --root-dir /data/waymo/training \
    --output-dir /data/ncore/waymo \
    --store-type directory \
    waymo-v4
```

## License

Apache 2.0 - See LICENSE file in the repository root.
