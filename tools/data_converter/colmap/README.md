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
- The pointcloud has been assigned to a single lidar frame at timestamp 0 - point colors are stored as generic `rgb` frame data.
- Downsampled images can be added as if they are separate cameras (with camera id suffixes `_2`, `_4`, and `_8`)

## Per-Image Masks

The converter automatically detects per-image mask files and stores them as
per-frame `mask` properties in the `generic_data` of each camera frame
(grayscale, `uint8`, shape `[H, W]`).

Two mask-file conventions are supported, checked in priority order:

1. **Co-located mask** — `<image_dir>/<stem>_mask.png`
   alongside the image file.
2. **Separate masks directory** — `<sequence_dir>/masks/<image_filename>`.

If no mask file is found for a given image, the frame is stored without a
`mask` entry.

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
| `--colmap-dir` | Relative path to COLMAP reconstruction directory | `sparse/0` |
| `--images-dir` | Relative path to images directory | `images` |
| `--masks-dir` | Explicit masks directory (auto-detect if not set) | None |
| `--world-global-mode` | Controls `("world", "world_global")` static pose storage: `none` (omit), `identity` (store identity matrix) | `none` |

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
    --no-include-3d-points \
    --no-include-downsampled-images
```

Convert to directory format (instead of itar):

```bash
bazel run //tools/data_converter/colmap -- \
    --root-dir /data/colmap/training \
    --output-dir /data/ncore/colmap \
    --store-type directory \
    colmap-v4
```

### ScanNet++ Conversion

The `scannetpp-v4` subcommand converts ScanNet++ DSLR scenes. It uses the
resized fisheye images (`dslr/resized_images/`) with the COLMAP `OPENCV_FISHEYE`
camera model and stores train/test split metadata from `train_test_lists.json`.

Convert a single scene:

```bash
bazel run //tools/data_converter/colmap:convert -- \
    --root-dir /path/to/scannetpp/scene_id \
    --output-dir /path/to/output \
    scannetpp-v4
```

Convert all scenes under a parent directory:

```bash
bazel run //tools/data_converter/colmap:convert -- \
    --root-dir /path/to/scannetpp/ \
    --output-dir /path/to/output \
    scannetpp-v4
```

You can also convert ScanNet++ data manually via the generic `colmap-v4` subcommand
with explicit path options:

```bash
bazel run //tools/data_converter/colmap:convert -- \
    --root-dir /path/to/scannetpp/scene_id \
    --output-dir /path/to/output \
    colmap-v4 \
    --colmap-dir dslr/colmap \
    --images-dir dslr/resized_images \
    --masks-dir dslr/resized_anon_masks \
    --camera-prefix dslr \
    --no-include-downsampled-images
```

## License

Apache 2.0 - See LICENSE file in the repository root.
