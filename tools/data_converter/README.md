<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Data-conversion Entrypoint

This packages contains common and abstract functionality to implement NCore data-converters
in downstream repositories.

## Structured Lidar Model Extraction

`structured_lidar_model.py` provides generic utilities for deriving structured spinning
lidar models from point cloud data. It is designed for datasets that provide
motion-compensated point clouds without raw sensor timestamps (e.g., nuScenes).

The library supports:

- **Model creation**: From sensor spec (nominal model with uniform azimuths) or from
  empirical measurement of a decompensated reference frame. The empirical model is
  substantially more accurate -- see the expected-accuracy table below.
- **Resolution upsampling**: Interpolate column azimuths to 2x/4x resolution for
  sub-column alignment precision (~0.02 deg at 4x vs ~0.09 deg at native). This is
  required because mechanical spinning introduces per-frame azimuth drift -- the sensor
  does not fire at exactly the same angles each revolution. The upsampled model allows
  alignment to snap to the actual firing position rather than the nearest nominal column.
- **Per-frame alignment**: Iterative column alignment + motion decompensation,
  with optional fine-grained sub-column refinement.
- **Multi-frame optimization**: Median-based correction of column azimuths and
  row offsets across many frames.
- **Pre-computed timestamps**: When per-point timestamps are already available
  (e.g., from raw sensor data), the alignment step can use them directly instead
  of approximating from column indices.

### Usage

```python
from tools.data_converter.structured_lidar_model import (
    derive_nominal_hdl32e,
    upsample_model,
    align_frame,
    optimize_model,
)

# 1. Create model (HDL-32E example)
model = derive_nominal_hdl32e(spinning_frequency_hz=20.0, start_azimuth_rad=1.5)
model = upsample_model(model, resolution_factor=4)

# 2. Per-frame alignment
frame_data = align_frame(
    xyz_mc, ring_index, intensity,
    n_beams_per_column=32,
    model_params=model,
    motion_compensator=mc,
    sensor_id="lidar_top",
    frame_start_us=t0, frame_end_us=t1,
    model_resolution_factor=4,
)

# 3. With pre-computed timestamps (no column-index approximation needed)
frame_data = align_frame(
    ...,
    timestamps_us=per_point_timestamps,  # from raw sensor data
    model_resolution_factor=4,
)
```

### Expected Model Accuracy

Measured with `//tools:ncore_evaluate_lidar_model` across the nuScenes v1.0-mini
scenes plus a v1.0-trainval sample (HDL-32E, 4x resolution, 1 optimization pass).
The far-range mean (points > 20 m) and the median are the meaningful quality
indicators; the combined "all points" mean is dominated by close-range
motion-compensation artifacts and occasional glitch frames.

| Model source | Mean far-range error | Systematic azimuth bias |
|--------------|----------------------|-------------------------|
| `empirical` (default) | ~0.02 - 0.04 deg | < ~0.005 deg |
| `nominal` | ~0.02 - 0.6 deg | up to ~0.4 deg |

The empirical model reproduces the point cloud up to ~15x more accurately than
the spec-derived nominal model (and never worse), so it is the default. Use
`nominal` only when a data-independent model is required.

### HDL-32E Presets

The library includes constants and a factory for the Velodyne HDL-32E:
- 32 beams, 1085 columns/revolution, 50 ms scan duration
- Spec elevation angles (-30.67 to +10.67 deg, non-uniform)
- Analytical intra-column firing offsets (2 x 16-beam banks at 1.152 us pair interval)

### Testing

```bash
bazel test //tools/data_converter:pytest_structured_lidar_model_3_11
```
