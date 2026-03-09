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

import logging

from types import SimpleNamespace

import click


try:
    from tools.debug import breakpoint
except ImportError:
    # if included externally as 'ncore_repo' use fully-evaluated path
    try:
        from ncore_repo.tools.debug import breakpoint  # type: ignore
    except ImportError:
        # fall back to 'external'-prefixed 'ncore_repo' reference if required
        from external.ncore_repo.tools.debug import breakpoint  # type: ignore


@click.group()
@click.option(
    "--root-dir",
    type=str,
    default=None,
    help="Path to the raw data sequences (required for file-based converters only)",
)
@click.option("--output-dir", type=str, help="Path where the converted data will be saved", required=True)
@click.option("--verbose", is_flag=True, default=False, help="Enables debug logging outputs")
@click.option("--debug", is_flag=True, default=False, help="Start a debugpy remote debugging sessions to listen on")
@click.option(
    "--debug-port", type=int, default=5678, help="The port on which debugpy will wait for a client to connect"
)
@click.option("--no-cameras", is_flag=True, default=False, help="Disable exporting any camera sensor")
@click.option(
    "--camera-id",
    "camera_ids",
    multiple=True,
    type=str,
    help="Cameras to be exported (multiple value option, all if not specified)",
    default=None,
)
@click.option("--no-lidars", is_flag=True, default=False, help="Disable exporting any lidar sensor")
@click.option(
    "--lidar-id",
    "lidar_ids",
    multiple=True,
    type=str,
    help="Lidars to be exported (multiple value option, all if not specified)",
    default=None,
)
@click.option("--no-radars", is_flag=True, default=False, help="Disable exporting any radar sensor")
@click.option(
    "--radar-id",
    "radar_ids",
    multiple=True,
    type=str,
    help="Radars to be exported (multiple value option, all if not specified)",
    default=None,
)
@click.pass_context
def cli(ctx, *_, **kwargs):
    """Data Preprocessing Pipeline

    Source data format is selected via subcommands, for which dedicated options can be specified.

    Example invocation for 'NV maglev' data

    \b
    ./convert_raw_data.py
      --root-dir <FOLDER WITH SOURCE DATASETS>
      --output-dir <FOLDER DATA WILL BE PRODUCED>
      <your-data-variant>
    """

    # Create a config dict-like object and remember it as the context object. From
    # this point onwards other commands can refer to it by using the
    # @click.pass_context decorator.
    ctx.obj = SimpleNamespace(**kwargs)

    # Initialize basic top-level logger configuration
    logging.basicConfig(
        level=logging.DEBUG if ctx.obj.verbose else logging.INFO,
        format="<%(asctime)s|%(levelname)s|%(filename)s:%(lineno)d|%(name)s> %(message)s",
    )

    # If the debug flag is set, add a breakpoint and wait for remote debugger
    if ctx.obj.debug:
        breakpoint(port=ctx.obj.debug_port)
