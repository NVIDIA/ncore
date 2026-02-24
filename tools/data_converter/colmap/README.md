<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NCore Colmap Converter

Convert Colmap Dataset to NCore V4 format.

## Overview

This module provides tooling for converting Colmap Datasets to NCore V4 format.
It is a standalone Bazel module that depends on the parent `ncore` module.
Because Colmap data does not adhere to the usual AV data conventions, we have made some
design decisions to ease this conversion:

- Camera images have been assigned timestamps at 1FPS.
- The pointcloud has been assigned to a single lidar frame at timestamp 0, with an initial pose that aligns it with the other data.
- Downsampled images can be added as if they are separate cameras (with camera id suffixes `_2`, `_4`, and `_8`)

## Prerequisites

- NCore build requirements (see <CONTRIBUTING.md>)
- Colmap data from a Colmap reconstruction as documented here: <https://colmap.github.io/format.html>

## Usage

Note: If `--root-dir` ends with a `/`, all subfolders will be processed individually.

```bash
bazel run //tools/data_converter/colmap -- \
    --root-dir /path/to/colmap/data \
    --output-dir /path/to/output/ncore \
    colmap-v4
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--root-dir` | Path to the Colmap data folder or parent folder | Required |
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
bazel run //tools/data_converter/colmap -- \
    --root-dir /data/colmap/scenes \
    --output-dir /data/ncore/colmap \
    colmap-v4
```

Exclude downsampled images and the 3d points.

```bash
bazel run //tools/data_converter/colmap -- \
    --root-dir /data/colmap/scenes \
    --output-dir /data/ncore/colmap \
    colmap-v4 \
    --include-3d-points false \
    --include-downsampled-images false
```

Convert to directory format (instead of itar):

```bash
bazel run //tools/data_converter/colmap -- \
    --root-dir /data/colmap/training \
    --output-dir /data/ncore/colmap \
    --store-type directory \
    colmap-v4
```

## License

Apache 2.0 - See LICENSE file in the repository root.
