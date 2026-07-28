<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NCore

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CI](https://github.com/NVIDIA/ncore/actions/workflows/ci.yml/badge.svg)](https://github.com/NVIDIA/ncore/actions/workflows/ci.yml)

NVIDIA NCore is an open, self-contained multi-sensor data platform with a focus on robotics and autonomous vehicle data. It defines a canonical component-based data format, GPU-accelerated camera and LiDAR sensor models, dataset converters, and APIs for reconstruction and simulation. NCore is used by NVIDIA Omniverse NuRec and other research and development workflows.

**Project Site:** [research.nvidia.com/labs/sil/projects/ncore](https://research.nvidia.com/labs/sil/projects/ncore)

## Key Features

- **Neural Reconstruction** - Designed for data-driven neural 3D reconstruction and simulation applications
- **Data Format** - Canonical and extensible component-based data format optimized for efficient training
- **Sensor Models** - GPU-accelerated Camera and LiDAR intrinsic models
- **Converters** - Transform external datasets into a common representation

## Installation

```bash
pip install nvidia-ncore
```

## Documentation

Full documentation is available at [nvidia.github.io/ncore](https://nvidia.github.io/ncore/).

## Contributing

Interested in contributing to NCore? See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed instructions on setting up your development environment, coding guidelines, and how to submit your contributions.

## Third Party Dependencies

This project will download and install additional third-party open source software as dependencies. Review the license terms of these open source projects before use. See [ATTRIBUTIONS.md](ATTRIBUTIONS.md) for a list of direct dependencies and their licenses.

## Support

**`NCore` code-level bugs, documentation issues, and feature requests**: file a [GitHub issue](../../issues/new/choose) using the appropriate template (Bug Report, Documentation Request, or Feature Request). The relevant NVIDIA responder is auto-assigned via the template's `assignees:` field.

**Usage and how-to questions** related to _NuRec_ / _Omniverse_: please post on the [NVIDIA Developer Forum (Omniverse / NuRec)](https://forums.developer.nvidia.com/c/omniverse/platform/nurec/752). Such questions are not tracked in this repository.

**Security vulnerabilities**: please use [NVIDIA's Vulnerability Disclosure Program](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail) (see [SECURITY.md](SECURITY.md)). Do not file security issues publicly here.

## License

NCore is provided under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full license text.
