<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Adapting a converter for an unsupported dataset

NCore does not ship a converter for every dataset; PandaSet is a common
example. There is no scaffold or template, and no porting guide elsewhere in
the repository. The in-tree converters are the reference implementations, so
adapt the nearest one rather than starting from scratch.

## 1. Pick the nearest by shape, not by name

Match on input layout and sensor model:

| Your data looks like | Start from |
|---|---|
| Per-frame files, spinning lidar, calibrated cameras, per-frame ego poses | nuScenes or Argoverse 2 |
| A single aggregated sweep from stacked lidar units | Argoverse 2 |
| Poses and images from structure-from-motion, no lidar | COLMAP |
| Sequential logs with an OXTS-style GPS/IMU stream | KITTI |
| Record-oriented container files | Waymo |
| Streamed from a remote object store rather than a local tree | PAI |

## 2. Read it

For your chosen converter:

- `tools/data_converter/<name>/converter.py` is the substance in every case.
- `tools/data_converter/<name>/main.py` exists for KITTI, nuScenes and
  Argoverse 2, and is a three-line registration shim. PAI, Waymo and COLMAP
  are driven from `converter.py` directly.
- `tools/data_converter/cli.py` defines the shared base flags for all six.
- `ncore/impl/data_converter/base.py` holds the config dataclasses, including
  the `--root-dir` requirement that file-based converters inherit and
  streaming ones do not.
- The matching page under `docs/conversions/`, plus
  `tools/data_converter/<name>/README.md`.

Only KITTI, nuScenes and Argoverse 2 ship converter tests, and those are
dataset-gated: they carry the `manual` Bazel tag and skip without their
dataset environment variable, so CI never runs them. Argoverse 2 additionally
ships a data-free unit test for its lidar-model derivation which does run in
CI, and which is the closest thing to an executable example of the geometry
code.

## 3. Keep the structure, change the edges

What varies between converters is narrower than it looks. In practice you are
replacing:

- **The reader.** How frames, calibration and annotations are enumerated from
  the source layout.
- **Sensor-model construction.** Which camera model parameter class fits, and
  whether a structured lidar model can be derived at all.
- **Timestamp derivation.** Where per-frame intervals and per-ray times come
  from, or how they are synthesised when the source has none.

What should not vary is the writer sequence, the component-group profile
handling, and the sequence-meta output. If you find yourself changing those,
check whether the dataset really is closest to the converter you picked.

Two decisions worth making early:

- **A structured lidar model is optional.** If the scan pattern is not a
  repeating row and column grid, pass `model_element=None` and store raw ray
  bundles with per-ray timestamps. Forcing a spinning model onto a
  non-repetitive pattern produces rays at range 0. Argoverse 2's
  `--lidar-model-source none` is the in-tree precedent.
- **Derive `spinning_direction` from the data** rather than assuming, if the
  dataset has more than one lidar unit or you have not verified it. nuScenes
  hard-codes `cw`; Argoverse 2 derives per unit because its two stacked units
  spin oppositely in their own frames.

## 4. Work on a branch, then check the gates

Adapt on a feature branch in your own fork. Before proposing anything:

```bash
bazel run //:format.check
bazel test //tools/data_converter/...   # dataset-gated tests will skip
```

Then check the store itself, which is the part that applies regardless of
which converter you started from. See the "Check the result" section of
`SKILL.md`.

`CONTRIBUTING.md` covers the rest of the repository's expectations: conventional
commits, GPG signing, SPDX headers, rebase-only history.
