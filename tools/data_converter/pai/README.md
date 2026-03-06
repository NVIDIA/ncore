<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Physical AI (PAI) Data Converter

Convert clip data from the [NVIDIA PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) HuggingFace dataset into NCore V4 format.

## Prerequisites

- A [HuggingFace](https://huggingface.co/) account with the PAI dataset license accepted
- A HuggingFace API token with read permissions on the dataset:

  ```bash
  export HF_TOKEN=hf_...
  # or pass --token hf_... / --hf-token hf_... to individual commands
  ```

## Dataset Overview

The dataset contains 306,152 clips organized into ~3,146 chunks of ~100 clips each.

| Category        | Features | Format | Typical size per chunk |
|-----------------|----------|--------|------------------------|
| **Labels**      | egomotion, egomotion.offline, obstacle.offline | zip (per-clip parquets) | ~16 KB |
| **Calibration** | camera_intrinsics, sensor_extrinsics, vehicle_dimensions, lidar_intrinsics | parquet (chunk-level) | ~14 KB |
| **Camera**      | 7 cameras (front wide, front tele, cross L/R, rear L/R, rear tele) | zip (mp4 + timestamps + blurred_boxes per clip) | ~2 GB |
| **Lidar**       | lidar_top_360fov | zip (per-clip parquets) | ~20 GB |
| **Radar**       | 19 radar sensors | zip (per-clip parquets) | varies |
| **Metadata**    | data_collection, sensor_presence | parquet (global) | ~11 MB |

Some features have `.offline` variants. These are used automatically by the streaming converter when available.

## pai-clip-dl: Download Tool

`pai-clip-dl` downloads and inspects clip data from HuggingFace. All commands run via Bazel:

```bash
bazel run //tools/data_converter/pai/pai_remote:pai-clip-dl -- <command> [options]
```

### Download clips

```bash
# Download all data for a single clip
bazel run //tools/data_converter/pai/pai_remote:pai-clip-dl -- \
    download <clip-id> -o /path/to/data

# Download multiple clips at once (batched by chunk automatically)
bazel run //tools/data_converter/pai/pai_remote:pai-clip-dl -- \
    download <clip-id-1> <clip-id-2> -o /path/to/data

# Download only specific features
bazel run //tools/data_converter/pai/pai_remote:pai-clip-dl -- \
    download <clip-id> -o /path/to/data -f egomotion -f camera_front_wide_120fov

# Skip metadata parquets
bazel run //tools/data_converter/pai/pai_remote:pai-clip-dl -- \
    download <clip-id> -o /path/to/data --no-metadata
```

### Inspect clips and features

```bash
# Show chunk, split, and sensor presence for a clip
bazel run //tools/data_converter/pai/pai_remote:pai-clip-dl -- info <clip-id>

# List all available features in the dataset
bazel run //tools/data_converter/pai/pai_remote:pai-clip-dl -- list-features
```

### Stream individual files (no full download)

```bash
# List files available for a clip within a feature's zip
bazel run //tools/data_converter/pai/pai_remote:pai-clip-dl -- \
    stream <clip-id> -f camera_front_wide_120fov

# Extract a single file to disk
bazel run //tools/data_converter/pai/pai_remote:pai-clip-dl -- \
    stream <clip-id> -f camera_front_wide_120fov --file timestamps.parquet -o out.parquet

# Pipe binary content to stdout
bazel run //tools/data_converter/pai/pai_remote:pai-clip-dl -- \
    stream <clip-id> -f camera_front_wide_120fov --file .mp4 > video.mp4
```

### Downloaded directory layout

```
{output_dir}/{clip_id}/
├── calibration/
│   ├── camera_intrinsics.parquet
│   ├── sensor_extrinsics.parquet          (may include .offline variant)
│   ├── vehicle_dimensions.parquet
│   └── lidar_intrinsics.parquet           (optional)
├── labels/
│   ├── {clip_id}.egomotion.parquet
│   ├── {clip_id}.egomotion.offline.parquet (if available)
│   └── {clip_id}.obstacle.offline.parquet  (if available)
├── camera/
│   ├── {clip_id}.{camera_id}.mp4
│   ├── {clip_id}.{camera_id}.timestamps.parquet
│   └── {clip_id}.{camera_id}.blurred_boxes.parquet
├── lidar/
│   └── {clip_id}.lidar_top_360fov.parquet
├── radar/
│   └── {clip_id}.radar_*.parquet          (present sensors only)
└── metadata/
    ├── sensor_presence.parquet
    ├── data_collection.parquet
    └── provenance.json                    (download source, optional)
```

## Converting to NCore V4

There are two conversion modes. Both produce output at:

```
<output-dir>/pai_<clip-id>/pai_<clip-id>.ncore4.zarr.itar
```

### Local mode (`pai-v4`)

Convert clips previously downloaded with `pai-clip-dl`. Point `--root-dir` at the directory
containing the downloaded clip folders.

```bash
# Convert a specific clip
bazel run //tools/data_converter/pai:convert -- \
    --root-dir /path/to/data \
    --output-dir /path/to/output \
    pai-v4 \
    --clip-id <clip-id>

# Convert all clips under root-dir (omit --clip-id)
bazel run //tools/data_converter/pai:convert -- \
    --root-dir /path/to/data \
    --output-dir /path/to/output \
    pai-v4
```

### Streaming mode (`pai-stream-v4`)

Convert clips directly from HuggingFace without a prior download. `--root-dir` is required by
the base CLI but ignored in this mode. `--clip-id` is required.

```bash
bazel run //tools/data_converter/pai:convert -- \
    --root-dir /tmp/unused \
    --output-dir /path/to/output \
    pai-stream-v4 \
    --clip-id <clip-id> \
    --hf-token <your-hf-token>
```

`--revision` selects the HuggingFace dataset branch/tag (default: `main`). Video data is
temporarily written to disk and cleaned up after each clip.

### Shared conversion options

Both subcommands support:

| Option | Default | Description |
|--------|---------|-------------|
| `--clip-id ID` | all clips | Clip ID(s) to convert (repeatable) |
| `--seek-sec FLOAT` | none | Skip this many seconds from the start of each clip |
| `--duration-sec FLOAT` | none | Limit converted duration to this many seconds |
| `--store-type {itar,directory}` | `itar` | Output store format |
| `--profile {default,separate-sensors,separate-all}` | `separate-sensors` | Component group layout |
| `--sequence-meta / --no-sequence-meta` | enabled | Write a JSON metadata sidecar |

## How It Works

1. **Index resolution** — `clip_index.parquet` maps each clip UUID to a chunk ID. `features.csv` describes all feature types and their chunk path templates.

2. **Sensor filtering** — `sensor_presence.parquet` records which sensors were active for each clip. Features with absent sensors are skipped automatically.

3. **Offline variants** — when streaming, `.offline` pre-processed variants of features (e.g. `egomotion.offline`, `sensor_extrinsics.offline`) are preferred over raw features when available.

4. **Chunk batching** — when downloading multiple clips, they are grouped by chunk. Each remote zip is opened only once and all relevant clips are extracted in a single pass.

5. **Streaming via range requests** — zip archives are accessed without full download. The central directory is read from the end of the file (~64 KB), then individual entries are fetched on demand. Files are stored uncompressed (`STORE`), so range requests map directly to entry bytes.

6. **CDN URL caching** — HuggingFace's `/resolve/` endpoint redirects to a signed CDN URL. These are cached for 50 minutes to avoid redundant requests.

## Configuration

| Environment variable | Description | Default |
|----------------------|-------------|---------|
| `HF_TOKEN` | HuggingFace API token | (required) |
| `pai_remote_CACHE` | Local cache directory for index files | `~/.cache/pai-clip-dl` |

## Module Structure

```
pai/
├── converter.py       # Main converter; pai-v4 and pai-stream-v4 CLI entry points
├── data_provider.py   # ClipDataProvider protocol; LocalClipDataProvider and StreamingClipDataProvider
├── utils.py           # Calibration parsing, ego-vehicle point filtering
└── pai_remote/
    ├── config.py      # Config dataclass, constants, env var handling
    ├── remote.py      # HFRemote: authenticated HTTP, CDN URL resolution, downloads
    ├── index.py       # ClipIndex: clip_index.parquet, features.csv, sensor_presence
    ├── streaming.py   # StreamingZipAccess: HTTP range access into remote zips
    ├── downloader.py  # ClipDownloader: orchestrates multi-clip download with batching
    └── cli.py         # pai-clip-dl CLI: download, info, list-features, stream
```
