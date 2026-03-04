<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Convert Physical AI (PAI) data to the NCore format

## Prerequisites

- [Hugging Face](https://huggingface.co/) account with the PAI dataset license accepted
- Hugging Face CLI authenticated (`huggingface-cli login`)
- The `pai-clip-dl` tool (located in `pai-clip-dl/` subdirectory)

## 1. Download PAI clips with pai-clip-dl

Use the `pai-clip-dl` tool to download per-clip data from Hugging Face. See
`pai-clip-dl/README.md` for full documentation.

```bash
# Install the tool (one-time)
cd pai-clip-dl && uv sync && cd ..

# Download a single clip
pai-clip-dl download <clip-id> -o /path/to/data

# Download multiple clips
pai-clip-dl download <clip-id-1> <clip-id-2> -o /path/to/data

# Download only specific features
pai-clip-dl download <clip-id> -o /path/to/data -f camera_front_wide_120fov -f egomotion
```

This creates a per-clip directory structure under the output path:

```
/path/to/data/
  <clip-id>/
    calibration/
    camera/
    labels/
    lidar/
    metadata/
    radar/
```

## 2. Convert PAI data to NCore

There are two conversion modes:

- **`pai-v4`** -- converts clips previously downloaded to disk with `pai-clip-dl`.
- **`pai-stream-v4`** -- streams clip data directly from HuggingFace without a prior download.

### Local mode (`pai-v4`)

Point `--root-dir` at the directory containing the downloaded clip folders.

#### Single clip

```bash
bazel run //scripts:convert_raw_data -- \
    --root-dir=/path/to/data \
    --output-dir=/path/to/output \
    pai-v4 \
    --clip-id <clip-id>
```

#### All downloaded clips

Omit `--clip-id` to convert every clip directory found under `--root-dir`:

```bash
bazel run //scripts:convert_raw_data -- \
    --root-dir=/path/to/data \
    --output-dir=/path/to/output \
    pai-v4
```

### Streaming mode (`pai-stream-v4`)

Converts clips directly from HuggingFace. No prior download required.

```bash
bazel run //scripts:convert_raw_data -- \
    --root-dir=/tmp/unused \
    --output-dir=/path/to/output \
    pai-stream-v4 \
    --clip-id <clip-id> \
    --hf-token <your-hf-token>
```

- `--root-dir` is required by the base CLI but ignored in streaming mode (set to any placeholder).
- `--clip-id` is **required** and can be specified multiple times.
- `--hf-token` reads from the `HF_TOKEN` environment variable if not provided on the command line.
- `--revision` selects the HuggingFace dataset branch/tag (default: `main`).

Video data is temporarily written to disk (ffmpeg requires local file paths) and
automatically cleaned up after conversion.

### Shared options

Both subcommands support these options:

- `--seek-sec <float>` -- skip this many seconds from the start
- `--duration-sec <float>` -- restrict total duration in seconds
- `--clip-id` can be specified multiple times to convert a subset of clips
- `--store-type itar|directory` -- output format (default: `itar`)
- `--profile default|separate-sensors|separate-all` -- component group layout
- `--sequence-meta / --no-sequence-meta` -- generate sequence metadata JSON

The output is an NCore v4 shard per clip at:
```
<output-dir>/pai_<clip-id>/pai_<clip-id>.ncore4.zarr.itar
```

## Next steps

The NCore v4 shard can be visualized (via the [`ncore`](https://github.com/NVIDIA/ncore) tools) or as input to a [NuRec (NRE) reconstruction](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nre/containers/nre?version=26.01.00).
