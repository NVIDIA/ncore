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

"""Orchestrates downloading all data for one or more clips to a local directory."""

from __future__ import annotations

import io
import json
import logging

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from tools.data_converter.pai.pai_remote.index import ClipIndex, FeatureSpec
from tools.data_converter.pai.pai_remote.remote import HFRemote
from tools.data_converter.pai.pai_remote.streaming import StreamingZipAccess


log = logging.getLogger(__name__)
console = Console()


class ClipDownloader:
    """Download all data associated with one or more clips to a local directory.

    Given a list of clip_ids, this:

    1. Resolves each clip's chunk_id via :class:`ClipIndex`.
    2. Groups clips by chunk so that each remote zip is opened at most once.
    3. Checks sensor presence to skip features where the sensor is absent.
    4. For **zip features** (camera, lidar, radar, labels): uses
       :class:`StreamingZipAccess` to extract only the requested clips' files.
    5. For **parquet features** (calibration): downloads the chunk parquet,
       filters to the clips' rows, and saves a filtered result per clip.
    6. Downloads and filters metadata parquets.

    Output structure::

        {output_dir}/{clip_id}/
        ├── labels/
        │   ├── {clip_id}.egomotion.parquet
        │   └── ...
        ├── calibration/
        │   ├── camera_intrinsics.parquet
        │   └── ...
        ├── camera/
        │   ├── {clip_id}.camera_front_wide_120fov.mp4
        │   └── ...
        ├── lidar/
        │   └── ...
        ├── radar/
        │   └── ...
        └── metadata/
            ├── data_collection.parquet
            └── feature_presence.parquet
    """

    def __init__(self, remote: HFRemote, index: ClipIndex) -> None:
        self.remote = remote
        self.index = index

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download_clips(
        self,
        clip_ids: Sequence[str],
        output_dir: Path,
        *,
        features: Sequence[str] | None = None,
        skip_missing_sensors: bool = True,
        include_metadata: bool = True,
    ) -> list[Path]:
        """Download all (or selected) data for multiple clips.

        Clips sharing the same chunk are batched so that each remote zip
        archive is opened only once.

        Parameters
        ----------
        clip_ids:
            One or more clip UUIDs.
        output_dir:
            Root output directory.  Per-clip files go into
            ``output_dir/<clip_id>/``.
        features:
            If given, only download these feature names.  Otherwise all.
        skip_missing_sensors:
            Check feature_presence and skip features where the sensor is
            absent for a given clip.
        include_metadata:
            Also download and filter the global metadata parquets.

        Returns
        -------
        List of per-clip output directories.
        """
        clip_ids = list(clip_ids)

        # Resolve chunks and group clips by chunk
        chunk_map: dict[str, int] = {}  # clip_id -> chunk_id
        by_chunk: dict[int, list[str]] = defaultdict(list)  # chunk_id -> [clip_ids]
        for cid in clip_ids:
            chunk_id = self.index.get_chunk_id(cid)
            chunk_map[cid] = chunk_id
            by_chunk[chunk_id].append(cid)

        console.print(f"[bold]Clips:[/bold]  {len(clip_ids)}  [bold]Chunks:[/bold] {len(by_chunk)}")

        # Resolve feature list
        all_features = self.index.features
        if features is not None:
            feature_set = set(features)
            all_features = [f for f in all_features if f.name in feature_set]
            missing = feature_set - {f.name for f in all_features}
            if missing:
                console.print(f"[yellow]Warning:[/yellow] Unknown features: {missing}")

        # Total work units = features * chunks (each chunk may serve N clips)
        total_units = len(all_features) * len(by_chunk)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Downloading...", total=total_units)

            for feature in all_features:
                for chunk_id, chunk_clip_ids in by_chunk.items():
                    # Filter to clips that actually have this sensor
                    if skip_missing_sensors:
                        active_clips = [
                            cid for cid in chunk_clip_ids if self.index.is_feature_present(cid, feature.name)
                        ]
                    else:
                        active_clips = chunk_clip_ids

                    if not active_clips:
                        progress.update(task, advance=1)
                        continue

                    desc = f"{feature.name} chunk {chunk_id:04d} ({len(active_clips)} clips)"
                    progress.update(task, description=desc)

                    try:
                        if feature.is_zip:
                            self._download_zip_feature_batch(
                                active_clips,
                                chunk_id,
                                feature,
                                output_dir,
                            )
                        elif feature.is_parquet:
                            self._download_parquet_feature_batch(
                                active_clips,
                                chunk_id,
                                feature,
                                output_dir,
                            )
                    except Exception as e:
                        console.print(f"[yellow]Warning:[/yellow] {feature.name} chunk {chunk_id:04d}: {e}")
                        log.warning(
                            "Failed %s chunk %04d: %s",
                            feature.name,
                            chunk_id,
                            e,
                        )

                    progress.update(task, advance=1)

        # Metadata (downloaded once, filtered per clip)
        if include_metadata:
            self._download_metadata_batch(clip_ids, output_dir)

        # Write provenance metadata per clip
        self._write_provenance(clip_ids, output_dir)

        clip_dirs = [output_dir / cid for cid in clip_ids]
        console.print(f"[green]Done![/green] {len(clip_ids)} clip(s) saved under {output_dir}")
        return clip_dirs

    def download_clip(
        self,
        clip_id: str,
        output_dir: Path,
        *,
        features: Sequence[str] | None = None,
        skip_missing_sensors: bool = True,
        include_metadata: bool = True,
    ) -> Path:
        """Download all (or selected) data for a single clip.

        Convenience wrapper around :meth:`download_clips`.
        """
        dirs = self.download_clips(
            [clip_id],
            output_dir,
            features=features,
            skip_missing_sensors=skip_missing_sensors,
            include_metadata=include_metadata,
        )
        return dirs[0]

    # ------------------------------------------------------------------
    # Zip features — batch by chunk
    # ------------------------------------------------------------------

    def _download_zip_feature_batch(
        self,
        clip_ids: list[str],
        chunk_id: int,
        feature: FeatureSpec,
        output_dir: Path,
    ) -> None:
        """Open one remote zip and extract files for all *clip_ids* in the chunk."""
        with StreamingZipAccess.from_feature(self.remote, feature, chunk_id) as sza:
            for clip_id in clip_ids:
                dest_dir = output_dir / clip_id / feature.directory
                dest_dir.mkdir(parents=True, exist_ok=True)

                clip_files = sza.get_clip_entries(clip_id)
                if not clip_files:
                    log.warning(
                        "No files for clip %s in %s chunk %04d",
                        clip_id,
                        feature.name,
                        chunk_id,
                    )
                    continue

                for filename in clip_files:
                    data = sza.read(filename)
                    dest = dest_dir / filename
                    dest.write_bytes(data)
                    log.debug("Wrote %s (%d bytes)", dest, len(data))

    # ------------------------------------------------------------------
    # Parquet features — batch by chunk
    # ------------------------------------------------------------------

    def _download_parquet_feature_batch(
        self,
        clip_ids: list[str],
        chunk_id: int,
        feature: FeatureSpec,
        output_dir: Path,
    ) -> None:
        """Download one chunk parquet, then filter and save per clip."""
        repo_path = feature.chunk_path(chunk_id)
        raw = self.remote.download_bytes(repo_path)
        df = pd.read_parquet(io.BytesIO(raw))

        for clip_id in clip_ids:
            dest_dir = output_dir / clip_id / feature.directory
            dest_dir.mkdir(parents=True, exist_ok=True)

            filtered = _filter_parquet_to_clip(df, clip_id)
            dest = dest_dir / f"{feature.name}.parquet"
            filtered.to_parquet(dest)
            log.debug("Wrote %s (%d rows)", dest, len(filtered))

    # ------------------------------------------------------------------
    # Metadata — downloaded once, split per clip
    # ------------------------------------------------------------------

    def _download_metadata_batch(
        self,
        clip_ids: list[str],
        output_dir: Path,
    ) -> None:
        """Download each global metadata parquet once, then filter per clip."""
        for name in ("data_collection", "feature_presence"):
            repo_path = f"metadata/{name}.parquet"
            try:
                raw = self.remote.download_bytes(repo_path)
                df = pd.read_parquet(io.BytesIO(raw))
            except Exception as e:
                log.warning("Failed to download metadata/%s: %s", name, e)
                continue

            for clip_id in clip_ids:
                meta_dir = output_dir / clip_id / "metadata"
                meta_dir.mkdir(parents=True, exist_ok=True)
                filtered = _filter_parquet_to_clip(df, clip_id)
                dest = meta_dir / f"{name}.parquet"
                filtered.to_parquet(dest)
                log.debug("Wrote %s", dest)

    # ------------------------------------------------------------------
    # Provenance — records where the data came from
    # ------------------------------------------------------------------

    def _write_provenance(
        self,
        clip_ids: list[str],
        output_dir: Path,
    ) -> None:
        """Write a ``provenance.json`` into each clip's ``metadata/`` directory.

        Records the HuggingFace repo, revision, resolved commit SHA, and
        download timestamp so that downstream tools can trace data lineage.
        """
        provenance_base = {
            "repo_id": self.remote.config.repo_id,
            "revision": self.remote.config.revision,
            "commit_sha": self.remote.commit_sha,
            "download_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        for clip_id in clip_ids:
            meta_dir = output_dir / clip_id / "metadata"
            meta_dir.mkdir(parents=True, exist_ok=True)
            dest = meta_dir / "provenance.json"
            provenance = {**provenance_base, "clip_id": clip_id}
            dest.write_text(json.dumps(provenance, indent=2))
            log.debug("Wrote %s", dest)


# ======================================================================
# Helpers
# ======================================================================


def _filter_parquet_to_clip(df: pd.DataFrame, clip_id: str) -> pd.DataFrame:
    """Filter a DataFrame to rows belonging to *clip_id*.

    Handles three index layouts:
    - MultiIndex with a ``clip_id`` level (calibration files).
    - Simple ``clip_id`` index (metadata, vehicle_dimensions, …).
    - ``clip_id`` as a regular column.
    - Falls back to returning the whole DataFrame when none match.
    """
    if isinstance(df.index, pd.MultiIndex):
        if "clip_id" in df.index.names:
            try:
                return df.xs(clip_id, level="clip_id", drop_level=False)
            except KeyError:
                return df.iloc[0:0]  # empty with same schema
    elif df.index.name == "clip_id":
        if clip_id in df.index:
            return df.loc[[clip_id]]
        return df.iloc[0:0]
    elif "clip_id" in df.columns:
        return df[df["clip_id"] == clip_id]
    return df
