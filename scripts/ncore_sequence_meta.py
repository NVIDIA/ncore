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

from pathlib import Path
from typing import Optional, Tuple

import click

from ncore.impl.data.data3 import ShardDataLoader


@click.command()
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
def ncore_sequence_meta(
    shard_file_pattern: str,
    shard_file_skip_suffixes: Tuple[str],
    output_dir: str,
    output_file: Optional[str],
    open_consolidated: bool,
    debug: bool,
):
    """Summarizes and exports data-ranges within a virtual shard sequence"""

    # Initialize the logger
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
    )
    logger = logging.getLogger(__name__)

    loader = ShardDataLoader(
        ShardDataLoader.evaluate_shard_file_pattern(shard_file_pattern, skip_suffixes=shard_file_skip_suffixes),
        open_consolidated=open_consolidated,
    )

    ## Collect sequence-wide information
    output = loader.get_sequence_meta()

    ## Serialize output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if output_file:
        output_path /= output_file
    else:
        output_path /= f"{loader.get_sequence_id()}.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Wrote meta data {str(output_path)}")


if __name__ == "__main__":
    ncore_sequence_meta()
