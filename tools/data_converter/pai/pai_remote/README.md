# pai_remote

Download and stream clip data from the [NVIDIA PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) HuggingFace dataset.

## Features

- **Clip-level download** — given one or more clip UUIDs, resolves the chunk, downloads all associated files (labels, calibration, camera, lidar, radar, metadata), and saves them organized by clip.
- **Multi-clip batching** — clips sharing the same chunk are grouped so each remote zip archive is opened only once.
- **Streaming access** — read individual files from multi-GB zip archives via HTTP range requests without downloading the full archive. A 15 MB video can be extracted from a 2 GB camera zip in ~0.2 s.
- **Sensor presence filtering** — automatically skips features where the sensor was not present for a given clip.
- **Local caching** — index files (`clip_index.parquet`, `features.csv`, `sensor_presence.parquet`) are cached locally to avoid redundant downloads.

## Authentication

All access requires a HuggingFace token with read permissions on the dataset. Provide it via:

```bash
export HF_TOKEN=hf_...
# or pass --token hf_... to every command
```

## CLI Usage

All commands follow the pattern:

```bash
bazel run //tools/data_converter/pai/pai_remote:pai-clip-dl -- <command> [options]
```

### Download clips

```bash
# Download all data for a single clip
bazel run //tools/data_converter/pai/pai_remote:pai-clip-dl -- download <clip_id> -o ./data

# Download multiple clips at once (batches by chunk automatically)
bazel run //tools/data_converter/pai/pai_remote:pai-clip-dl -- download <clip_id_1> <clip_id_2> <clip_id_3> -o ./data

# Download only specific features (labels + one camera)
bazel run //tools/data_converter/pai/pai_remote:pai-clip-dl -- download <clip_id> -f egomotion -f camera_front_wide_120fov -o ./data

# Skip metadata parquets
bazel run //tools/data_converter/pai/pai_remote:pai-clip-dl -- download <clip_id> --no-metadata -o ./data
```

### Clip info

```bash
# Show chunk, split, and sensor presence for one or more clips
bazel run //tools/data_converter/pai/pai_remote:pai-clip-dl -- info <clip_id>
bazel run //tools/data_converter/pai/pai_remote:pai-clip-dl -- info <clip_id_1> <clip_id_2>
```

### List features

```bash
bazel run //tools/data_converter/pai/pai_remote:pai-clip-dl -- list-features
```

### Stream (no full download)

```bash
# List files available for a clip within a feature's zip
bazel run //tools/data_converter/pai/pai_remote:pai-clip-dl -- stream <clip_id> -f camera_front_wide_120fov

# Extract a single file from a remote zip to a local file
bazel run //tools/data_converter/pai/pai_remote:pai-clip-dl -- stream <clip_id> -f camera_front_wide_120fov --file timestamps.parquet -o out.parquet

# Pipe binary content to stdout
bazel run //tools/data_converter/pai/pai_remote:pai-clip-dl -- stream <clip_id> -f camera_front_wide_120fov --file .mp4 > video.mp4
```

## Python API

```python
from tools.data_converter.pai.pai_remote.config import Config
from tools.data_converter.pai.pai_remote.remote import HFRemote
from tools.data_converter.pai.pai_remote.index import ClipIndex
from tools.data_converter.pai.pai_remote.downloader import ClipDownloader
from tools.data_converter.pai.pai_remote.streaming import StreamingZipAccess

config = Config.from_env()  # reads HF_TOKEN from environment
remote = HFRemote(config)
index  = ClipIndex(remote)
```

### Look up clip metadata

```python
clip_id  = "25cd4769-5dcf-4b53-a351-bf2c5deb6124"
chunk_id = index.get_chunk_id(clip_id)   # -> 0
split    = index.get_split(clip_id)      # -> "train"
presence = index.get_sensor_presence(clip_id)  # -> {"camera_front_wide_120fov": True, ...}
```

### Download clips to disk

```python
from pathlib import Path

dl = ClipDownloader(remote, index)

# Single clip
dl.download_clip(clip_id, Path("./data"))

# Multiple clips (batches by chunk)
dl.download_clips(
    ["25cd4769-...", "2edf278f-...", "908654ce-..."],
    Path("./data"),
    features=["egomotion", "camera_front_wide_120fov"],  # optional filter
)
```

### Streaming zip access (no download)

```python
import pandas as pd

feature  = index.get_feature("camera_front_wide_120fov")
chunk_id = index.get_chunk_id(clip_id)

with StreamingZipAccess.from_feature(remote, feature, chunk_id) as sza:
    # List entries for this clip
    entries = sza.get_clip_entries(clip_id)

    # Get seekable BytesIO handles (fetched via HTTP range requests)
    handles = sza.stream_clip_files(clip_id)
    timestamps = pd.read_parquet(handles["...timestamps.parquet"])
    video_bytes = handles["...mp4"].read()
```

## Output directory structure

```
{output_dir}/{clip_id}/
├── labels/
│   ├── {clip_id}.egomotion.parquet
│   ├── {clip_id}.egomotion.offline.parquet
│   └── {clip_id}.obstacle.offline.parquet
├── calibration/
│   ├── camera_intrinsics.parquet          (filtered to clip)
│   ├── sensor_extrinsics.offline.parquet  (filtered to clip)
│   ├── vehicle_dimensions.parquet         (filtered to clip)
│   └── ...
├── camera/
│   ├── {clip_id}.camera_front_wide_120fov.mp4
│   ├── {clip_id}.camera_front_wide_120fov.timestamps.parquet
│   ├── {clip_id}.camera_front_wide_120fov.blurred_boxes.parquet
│   └── ... (all 7 cameras)
├── lidar/
│   └── {clip_id}.lidar_top_360fov.parquet
├── radar/
│   └── {clip_id}.radar_*.parquet  (present sensors only)
└── metadata/
    ├── data_collection.parquet   (filtered to clip)
    └── sensor_presence.parquet   (filtered to clip)
```

## Dataset overview

The [PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) dataset (branch `ncore_test`) contains 306,152 clips organized into ~3,146 chunks of ~100 clips each.

| Category        | Features | Format  | Typical size per chunk |
|-----------------|----------|---------|------------------------|
| **Labels**      | egomotion, egomotion.offline, obstacle.offline | zip (per-clip parquets inside) | ~16 KB |
| **Calibration** | camera_intrinsics, sensor_extrinsics, vehicle_dimensions, ... | parquet (chunk-level) | ~14 KB |
| **Camera**      | 7 cameras (front wide, front tele, cross L/R, rear L/R, rear tele) | zip (mp4 + timestamps + blurred_boxes per clip) | ~2 GB |
| **Lidar**       | lidar_top_360fov | zip (per-clip parquets) | ~20 GB |
| **Radar**       | 19 radar sensors | zip (per-clip parquets) | varies |
| **Metadata**    | data_collection, sensor_presence | parquet (global) | ~11 MB |

## How it works

1. **Index resolution** — `clip_index.parquet` maps each clip UUID to a chunk ID. `features.csv` describes all feature types and their chunk path templates.

2. **Sensor filtering** — `sensor_presence.parquet` records which sensors were active for each clip. Features with absent sensors are skipped automatically.

3. **Chunk batching** — when downloading multiple clips, they are grouped by chunk. Each remote zip is opened only once and all relevant clips are extracted in a single pass.

4. **Streaming via `remotezip`** — zip archives are accessed without full download. The library reads the central directory from the end of the file (~64 KB via HTTP range request), then fetches individual entries on demand. Files in the dataset zips are stored uncompressed (`STORE`), so range requests map directly to the entry bytes.

5. **CDN URL caching** — HuggingFace's `/resolve/` endpoint redirects to a signed CDN URL. These are cached for 50 minutes to avoid redundant HEAD requests.

## Configuration

| Environment variable  | Description | Default |
|-----------------------|-------------|---------|
| `HF_TOKEN`            | HuggingFace API token | (required) |
| `pai_remote_CACHE`   | Local cache directory for index files | `~/.cache/pai-clip-dl` |

## Project structure

```
pai_remote/
├── config.py         # Config dataclass, constants, env var handling
├── remote.py         # HFRemote: authenticated HTTP, CDN URL resolution, downloads
├── index.py          # ClipIndex: clip_index.parquet, features.csv, sensor_presence
├── streaming.py      # StreamingZipAccess: remotezip wrapper for HTTP range access
├── downloader.py     # ClipDownloader: orchestrates multi-clip download with batching
└── cli.py            # Click CLI: download, info, list-features, stream
```
