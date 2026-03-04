# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI entry point for the NCore interactive 3D viewer.

Usage::

    bazel run //tools/ncore_vis -- v4 --component-group=<SEQUENCE_META.json>

"""

from __future__ import annotations

import logging

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import click

from ncore.impl.data.compat import SequenceLoaderProtocol
from ncore.impl.data.v4.compat import SequenceLoaderV4
from ncore.impl.data.v4.components import SequenceComponentGroupsReader
from tools.cli import OptionalStrParamType
from tools.debug import breakpoint as debug_breakpoint
from tools.ncore_vis.server import NCoreVisServer


@dataclass(frozen=True, slots=True, kw_only=True)
class CLIBaseParams:
    """Global CLI parameters shared across format sub-commands."""

    host: str
    port: int
    rig_frame_id: str
    world_frame_id: str
    debug: bool
    debug_port: int


@click.group()
@click.option("--host", type=str, default="0.0.0.0", help="Server host address")
@click.option("--port", type=int, default=8080, help="Server port")
@click.option("--rig-frame-id", type=str, default="rig", help="Pose graph frame ID for the rig/vehicle body")
@click.option("--world-frame-id", type=str, default="world", help="Pose graph frame ID for the world/map reference")
@click.option("--debug", is_flag=True, default=False, help="Start a debugpy remote debugging session")
@click.option("--debug-port", type=int, default=5678, help="Port on which debugpy will wait for a client to connect")
@click.pass_context
def cli(ctx: click.Context, **kwargs: object) -> None:
    """Interactive 3D viewer for NCore sequence data."""
    ctx.obj = CLIBaseParams(**kwargs)  # type: ignore[arg-type]


@cli.command()
@click.option(
    "component_groups",
    "--component-group",
    multiple=True,
    type=str,
    help="Data component group / sequence meta paths",
    required=True,
)
@click.option("--poses-component-group", type=str, help="Component group for 'poses'", default="default")
@click.option("--intrinsics-component-group", type=str, help="Component group for 'intrinsics'", default="default")
@click.option(
    "--masks-component-group",
    type=OptionalStrParamType(),
    help="Component group for 'masks' (use 'none' to disable)",
    default="default",
)
@click.option(
    "--cuboids-component-group",
    type=OptionalStrParamType(),
    help="Component group for 'cuboids' (use 'none' to disable)",
    default="default",
)
@click.pass_context
def v4(
    ctx: click.Context,
    component_groups: Tuple[str, ...],
    poses_component_group: str,
    intrinsics_component_group: str,
    masks_component_group: Optional[str],
    cuboids_component_group: Optional[str],
) -> None:
    """Launch the viewer with NCore V4 (component-based) sequence data."""
    params: CLIBaseParams = ctx.obj

    reader = SequenceComponentGroupsReader(
        [Path(group_path) for group_path in component_groups],
    )

    run(
        params,
        SequenceLoaderV4(
            reader,
            poses_component_group_name=poses_component_group,
            intrinsics_component_group_name=intrinsics_component_group,
            masks_component_group_name=masks_component_group,
            cuboids_component_group_name=cuboids_component_group,
        ),
    )


def run(params: CLIBaseParams, loader: SequenceLoaderProtocol) -> None:
    """Start the interactive viewer server.

    Args:
        params: Global CLI parameters.
        loader: Sequence loader providing unified data access.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    if params.debug:
        debug_breakpoint(port=params.debug_port)

    server = NCoreVisServer(
        loader=loader,
        host=params.host,
        port=params.port,
        rig_frame_id=params.rig_frame_id,
        world_frame_id=params.world_frame_id,
    )
    server.start()


if __name__ == "__main__":
    cli(show_default=True)
