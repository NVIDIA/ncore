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

"""Command-line interface for pai-clip-dl."""

from __future__ import annotations

import logging
import sys

from pathlib import Path

import click

from rich.console import Console
from rich.table import Table

from pai_clip_dl.config import Config
from pai_clip_dl.downloader import ClipDownloader
from pai_clip_dl.index import ClipIndex
from pai_clip_dl.remote import HFRemote
from pai_clip_dl.streaming import StreamingZipAccess


console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _make_components(token: str | None, revision: str) -> tuple[Config, HFRemote, ClipIndex]:
    config = Config.from_env(token=token, revision=revision)
    remote = HFRemote(config)
    index = ClipIndex(remote)
    return config, remote, index


def _validate_clip_ids(index: ClipIndex, clip_ids: list[str]) -> list[str]:
    """Validate clip IDs and return the valid ones, printing errors for invalid."""
    valid: list[str] = []
    for cid in clip_ids:
        if index.clip_exists(cid):
            valid.append(cid)
        else:
            console.print(f"[red]Error:[/red] clip_id {cid!r} not found in index.")
    return valid


@click.group()
@click.version_option(version="0.1.0", prog_name="pai-clip-dl")
def main() -> None:
    """Download and stream clip data from the NVIDIA PhysicalAI-Autonomous-Vehicles dataset."""


@main.command()
@click.argument("clip_ids", nargs=-1, required=True)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("."),
    help="Root output directory. Files are written to OUTPUT_DIR/CLIP_ID/.",
)
@click.option("--features", "-f", multiple=True, help="Feature names to download. Repeat for multiple. Default: all.")
@click.option("--token", envvar="HF_TOKEN", help="HuggingFace API token.")
@click.option("--revision", default="ncore_test", help="Git branch/revision of the dataset.")
@click.option("--no-skip-missing", is_flag=True, default=False, help="Don't skip features where the sensor is absent.")
@click.option("--no-metadata", is_flag=True, default=False, help="Skip downloading metadata parquets.")
@click.option("--verbose", "-v", is_flag=True, default=False)
def download(
    clip_ids: tuple[str, ...],
    output_dir: Path,
    features: tuple[str, ...],
    token: str | None,
    revision: str,
    no_skip_missing: bool,
    no_metadata: bool,
    verbose: bool,
) -> None:
    """Download all data for one or more clips to a local directory.

    Pass one or more CLIP_IDS as positional arguments.  Clips sharing the
    same chunk are batched so each remote zip is opened only once.
    """
    _setup_logging(verbose)
    config, remote, index = _make_components(token, revision)

    valid = _validate_clip_ids(index, list(clip_ids))
    if not valid:
        console.print("[red]No valid clip IDs provided.[/red]")
        sys.exit(1)

    dl = ClipDownloader(remote, index)
    dl.download_clips(
        valid,
        output_dir,
        features=list(features) if features else None,
        skip_missing_sensors=not no_skip_missing,
        include_metadata=not no_metadata,
    )


@main.command()
@click.argument("clip_ids", nargs=-1, required=True)
@click.option("--token", envvar="HF_TOKEN", help="HuggingFace API token.")
@click.option("--revision", default="ncore_test", help="Git branch/revision of the dataset.")
@click.option("--verbose", "-v", is_flag=True, default=False)
def info(clip_ids: tuple[str, ...], token: str | None, revision: str, verbose: bool) -> None:
    """Show information about one or more clips (chunk, split, sensors)."""
    _setup_logging(verbose)
    config, remote, index = _make_components(token, revision)

    valid = _validate_clip_ids(index, list(clip_ids))
    if not valid:
        sys.exit(1)

    for i, clip_id in enumerate(valid):
        if i > 0:
            console.print()  # blank line between clips

        chunk_id = index.get_chunk_id(clip_id)
        split = index.get_split(clip_id)
        presence = index.get_sensor_presence(clip_id)

        console.print(f"[bold]Clip ID:[/bold]  {clip_id}")
        console.print(f"[bold]Chunk:[/bold]    {chunk_id}")
        console.print(f"[bold]Split:[/bold]    {split}")
        console.print()

        table = Table(title="Sensor Presence")
        table.add_column("Sensor", style="cyan")
        table.add_column("Present", style="green")
        for sensor, present in sorted(presence.items()):
            style = "green" if present else "dim red"
            table.add_row(sensor, str(present), style=style)
        console.print(table)


@main.command("list-features")
@click.option("--token", envvar="HF_TOKEN", help="HuggingFace API token.")
@click.option("--revision", default="ncore_test", help="Git branch/revision of the dataset.")
@click.option("--verbose", "-v", is_flag=True, default=False)
def list_features(token: str | None, revision: str, verbose: bool) -> None:
    """List all available features in the dataset."""
    _setup_logging(verbose)
    config, remote, index = _make_components(token, revision)

    table = Table(title="Available Features")
    table.add_column("Feature", style="cyan")
    table.add_column("Category", style="yellow")
    table.add_column("Format", style="green")
    table.add_column("Clip Files", style="white")

    for f in index.features:
        fmt = "zip" if f.is_zip else "parquet"
        clip_files = ", ".join(f.clip_files.keys()) if f.clip_files else "(chunk-level)"
        table.add_row(f.name, f.directory, fmt, clip_files)

    console.print(table)


@main.command()
@click.argument("clip_id")
@click.option("--feature", "-f", required=True, help="Feature name (e.g. camera_front_wide_120fov).")
@click.option(
    "--file", "filename", default=None, help="Specific file inside the zip to read. If omitted, lists available files."
)
@click.option(
    "--output", "-o", type=click.Path(path_type=Path), default=None, help="Save output to file instead of stdout."
)
@click.option("--token", envvar="HF_TOKEN", help="HuggingFace API token.")
@click.option("--revision", default="ncore_test", help="Git branch/revision of the dataset.")
@click.option("--verbose", "-v", is_flag=True, default=False)
def stream(
    clip_id: str,
    feature: str,
    filename: str | None,
    output: Path | None,
    token: str | None,
    revision: str,
    verbose: bool,
) -> None:
    """Stream a single file from a remote zip without full download.

    If --file is not given, lists the clip's files in the zip archive.
    """
    _setup_logging(verbose)
    config, remote, index = _make_components(token, revision)

    if not index.clip_exists(clip_id):
        console.print(f"[red]Error:[/red] clip_id {clip_id!r} not found in index.")
        sys.exit(1)

    chunk_id = index.get_chunk_id(clip_id)
    feat = index.get_feature(feature)

    if not feat.is_zip:
        console.print(f"[red]Error:[/red] Feature {feature!r} is parquet-based. Use 'download' to fetch it.")
        sys.exit(1)

    with StreamingZipAccess.from_feature(remote, feat, chunk_id) as sza:
        clip_entries = sza.get_clip_entries(clip_id)

        if filename is None:
            # List mode
            console.print(f"[bold]Files for clip {clip_id} in {feature}:[/bold]")
            for entry in clip_entries:
                info_obj = sza._ensure_open().getinfo(entry)
                size_kb = info_obj.file_size / 1024
                if size_kb > 1024:
                    size_str = f"{size_kb / 1024:.1f} MB"
                else:
                    size_str = f"{size_kb:.1f} KB"
                console.print(f"  {entry}  ({size_str})")
            return

        # Resolve the filename -- user can pass the full name or just the
        # logical part
        target = None
        for entry in clip_entries:
            if entry == filename or entry.endswith(filename):
                target = entry
                break
        if target is None:
            console.print(
                f"[red]Error:[/red] File {filename!r} not found for clip {clip_id}.\nAvailable: {clip_entries}"
            )
            sys.exit(1)

        data = sza.read(target)

        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(data)
            console.print(f"Written {len(data)} bytes to {output}")
        else:
            # Write binary to stdout
            sys.stdout.buffer.write(data)


if __name__ == "__main__":
    main()
