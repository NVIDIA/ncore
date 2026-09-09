<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Description: <br>
Converts sensor data into an NCore V4 store using NCore's built-in dataset converters, guides adaptation of the nearest converter when no built-in one fits, and checks the resulting store. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA NCore <br>

### License/Terms of Use: <br>
Apache-2.0, per the [repository LICENSE](../../LICENSE). <br>

Adapted from NVIDIA-authored material in [NVIDIA/nurec-skills](https://github.com/NVIDIA/nurec-skills) at commit `5b9d287`, narrowed to the built-in converters and re-verified against this repository. <br>

## Use Case: <br>
Engineers converting autonomous-vehicle and scene-capture datasets (PAI, Waymo, COLMAP/ScanNet++, KITTI, nuScenes, Argoverse 2) into the NCore V4 sensor-data format, adapting an existing converter for a dataset NCore does not support, or diagnosing calibration and timing problems in a converted store. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [API key] <br>

A GitHub personal access token with `read:packages` is required to build the repository. A Hugging Face token is required only for the PAI converter, on an account that has accepted the `nvidia/PhysicalAI-Autonomous-Vehicles` dataset license. The skill instructs reading tokens from the environment rather than passing them as command-line arguments, so they are not exposed through process arguments or shell history. <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Re-running a converter over an existing indexed-tar store truncates it before writing, so an interrupted re-run can leave neither the previous store nor a complete new one. <br>
Mitigation: The skill directs output to a fresh path or a backup, and notes that the directory store type does not behave this way. <br>

Risk: The `ncore_vis` viewer binds all interfaces by default and its server is unauthenticated, exposing sensor imagery and geometry on shared or cloud hosts. <br>
Mitigation: The skill's invocation binds loopback explicitly and states the reason. <br>

Risk: Conversion cannot correct calibration or timing errors present in the source dataset, and a plausible-looking store may still be wrong. <br>
Mitigation: The skill states that its checks are inspection rather than verification, and gives a procedure that separates calibration faults from timing faults instead of guessing. <br>

Risk: Review before execution, as generated commands could be incorrect for a given dataset layout. <br>
Mitigation: Review commands before running; the skill uses explicit fill-in variables with guard checks rather than emitting ready-to-paste paths. <br>

## Reference(s): <br>
- [NVIDIA NCore](https://github.com/NVIDIA/ncore) <br>
- [NCore documentation](https://nvidia.github.io/ncore/) <br>
- [V4 invariants](references/v4-invariants.md) <br>
- [Adapting a converter](references/adapting-a-converter.md) <br>

## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Tasks: <br>
6 evaluation tasks (4 positive, 2 negative) defined in [evals/evals.json](evals/evals.json). <br>
