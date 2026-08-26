<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nvidia-ncore

A unified data format and library for autonomous vehicle and robotics sensor data.

## Features

- **Neural Reconstruction** - Designed for data-driven neural 3D reconstruction and simulation applications
- **Data Format** - Canonical and extensible component-based data format optimized for efficient training
- **Sensor Models** - GPU-accelerated Camera and LiDAR intrinsic models
- **Converters** - Transform external datasets into a common representation

## Installation

```bash
pip install nvidia-ncore
```

To convert raw [PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
clips without a source checkout, install the optional PAI converter extra:

```bash
pip install "nvidia-ncore[pai]"
ncore-convert --help
```

See the [PAI data converter README](https://github.com/NVIDIA/ncore/blob/main/tools/data_converter/pai/README.md) for usage.

## Documentation

<https://nvidia.github.io/ncore>
