# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import json
import logging

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import click

from ncore.impl.data.v3.compat import SequenceLoaderV3
from ncore.impl.data.v3.shards import ShardDataLoader
from ncore.impl.data.v4.compat import SequenceLoaderProtocol, SequenceLoaderV4
from ncore.impl.data.v4.components import SequenceComponentGroupsReader


logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True, frozen=True)
class CLIBaseParams:
    """Parameters passed to non-command-based CLI part.

    Attributes:
        output_dir: Directory path where the output JSON file will be written
        output_file: Optional custom filename for the output. If None, uses <sequence_id>.json
        open_consolidated: Whether to pre-load consolidated zarr metadata for faster access
        debug: Enable debug-level logging output
    """

    output_dir: str
    output_file: Optional[str]
    open_consolidated: bool
    debug: bool


@click.group()
@click.option("--output-dir", type=str, help="Path to the output folder", required=True)
@click.option(
    "--output-file",
    type=str,
    default=None,
    help="Filename of generated file (json) - <sequence_id>.json will be used by default if not provided",
    required=False,
)
@click.option("--open-consolidated/--no-open-consolidated", default=True, help="Pre-load shard meta-data?")
@click.option("--debug", is_flag=True, default=False, help="Enables debug logging outputs")
@click.pass_context
def cli(ctx, **kwargs) -> None:
    """Main CLI entry point for sequence metadata extraction."""
    ctx.obj = CLIBaseParams(**kwargs)

    # Initialize the logger
    logging.basicConfig(
        level=logging.DEBUG if ctx.obj.debug else logging.INFO,
    )


@cli.command()
@click.option(
    "--shard-file-pattern", type=str, help="Data shard pattern to load (supports range expansion)", required=True
)
@click.option(
    "--shard-file-skip-suffix",
    "shard_file_skip_suffixes",
    multiple=True,
    type=str,
    help="Suffixes to skip when evaluating shard file pattern",
    default=None,
)
@click.pass_context
def v3(
    ctx,
    shard_file_pattern: str,
    shard_file_skip_suffixes: Tuple[str],
) -> None:
    """Extract metadata from NCore V3 (shard-based) sequence data.

    Args:
        shard_file_pattern: Glob pattern for shard files (supports range expansion like shard_[0-10].zarr)
        shard_file_skip_suffixes: File suffixes to exclude when matching the pattern
    """
    params: CLIBaseParams = ctx.obj

    loader = ShardDataLoader(
        ShardDataLoader.evaluate_shard_file_pattern(shard_file_pattern, skip_suffixes=shard_file_skip_suffixes),
        open_consolidated=params.open_consolidated,
    )

    run(params, SequenceLoaderV3(loader))


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
@click.option("--masks-component-group", type=str, help="Component group for 'masks'", default="default")
@click.option(
    "--cuboids-component-group",
    type=str,
    help="Component group for 'cuboids'",
    default="default",
)
@click.pass_context
def v4(
    ctx,
    component_groups: Tuple[str, ...],
    poses_component_group: str,
    intrinsics_component_group: str,
    masks_component_group: str,
    cuboids_component_group: str,
) -> None:
    """Extract metadata from NCore V4 (component-based) sequence data.

    Args:
        component_groups: Paths to V4 component groups (can specify multiple)
        poses_component_group: Name of the poses component group to use
        intrinsics_component_group: Name of the intrinsics component group to use
        masks_component_group: Name of the masks component group to use
        cuboids_component_group: Name of the cuboids component group to use
    """
    params: CLIBaseParams = ctx.obj

    loader = SequenceComponentGroupsReader(
        [Path(group_path) for group_path in component_groups],
        open_consolidated=params.open_consolidated,
    )

    run(
        params,
        SequenceLoaderV4(
            loader,
            poses_component_group_name=poses_component_group,
            intrinsics_component_group_name=intrinsics_component_group,
            masks_component_group_name=masks_component_group,
            cuboids_component_group_name=cuboids_component_group,
        ),
    )


def run(params: CLIBaseParams, loader: SequenceLoaderProtocol) -> None:
    """Extracts sequence metadata and exports it as JSON.

    Collects comprehensive metadata from the sequence including:
    - Sequence ID and timestamp range
    - Component store information with MD5 checksums
    - Component versions and configurations
    - Generic metadata fields

    Args:
        params: CLI parameters specifying output location and options
        loader: Sequence loader (V3 or V4) providing unified data access
    """

    ## Collect sequence-wide information
    output = loader.get_sequence_meta()

    ## Serialize output
    output_path = Path(params.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if params.output_file:
        output_path /= params.output_file
    else:
        output_path /= f"{loader.sequence_id}.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Wrote meta data {str(output_path)}")


if __name__ == "__main__":
    cli(show_default=True)
