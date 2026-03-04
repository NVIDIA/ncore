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

"""Clip index and feature manifest: resolve clip_id -> chunk_id and list available features."""

from __future__ import annotations

import csv
import io
import json
import logging

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from pai_clip_dl.remote import HFRemote


log = logging.getLogger(__name__)


@dataclass
class FeatureSpec:
    """A single row from features.csv describing a data feature."""

    name: str
    directory: str  # top-level category: labels, calibration, camera, lidar, radar
    chunk_path_template: str  # e.g. "labels/egomotion/egomotion.chunk_{chunk_id:04d}.zip"
    clip_files: dict[str, str]  # logical_name -> filename template with {clip_id}

    @property
    def is_zip(self) -> bool:
        return self.chunk_path_template.endswith(".zip")

    @property
    def is_parquet(self) -> bool:
        return self.chunk_path_template.endswith(".parquet")

    def chunk_path(self, chunk_id: int) -> str:
        """Resolve the chunk path for a given chunk_id."""
        return self.chunk_path_template.format(chunk_id=chunk_id)

    def clip_filenames(self, clip_id: str) -> dict[str, str]:
        """Resolve per-clip filenames inside a zip for a given clip_id."""
        return {logical: template.format(clip_id=clip_id) for logical, template in self.clip_files.items()}


class ClipIndex:
    """Loads and queries the clip index, feature manifest, and sensor presence data.

    Usage::

        remote = HFRemote(config)
        index = ClipIndex(remote)
        chunk_id = index.get_chunk_id("25cd4769-5dcf-4b53-a351-bf2c5deb6124")
        features = index.get_features()
        presence = index.get_sensor_presence("25cd4769-...")
    """

    def __init__(self, remote: HFRemote) -> None:
        self._remote = remote
        self._clip_df: pd.DataFrame | None = None
        self._features: list[FeatureSpec] | None = None
        self._sensor_presence_df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # clip_index.parquet
    # ------------------------------------------------------------------

    @property
    def clip_df(self) -> pd.DataFrame:
        """Lazy-load clip_index.parquet (cached locally)."""
        if self._clip_df is None:
            path = self._remote.get_cached_or_download("clip_index.parquet")
            self._clip_df = pd.read_parquet(path)
            log.info("Loaded clip index: %d clips", len(self._clip_df))
        return self._clip_df

    def get_chunk_id(self, clip_id: str) -> int:
        """Look up the chunk_id for a clip_id."""
        df = self.clip_df
        if clip_id not in df.index:
            raise KeyError(f"clip_id {clip_id!r} not found in clip index")
        return int(df.loc[clip_id, "chunk"])  # pyright: ignore[reportArgumentType]

    def get_split(self, clip_id: str) -> str:
        """Look up the split (train/val/test) for a clip_id."""
        df = self.clip_df
        if clip_id not in df.index:
            raise KeyError(f"clip_id {clip_id!r} not found in clip index")
        return str(df.loc[clip_id, "split"])

    def clip_exists(self, clip_id: str) -> bool:
        return clip_id in self.clip_df.index

    # ------------------------------------------------------------------
    # features.csv
    # ------------------------------------------------------------------

    @property
    def features(self) -> list[FeatureSpec]:
        """Lazy-load and parse features.csv."""
        if self._features is None:
            path = self._remote.get_cached_or_download("features.csv")
            self._features = _parse_features_csv(path.read_text())
            log.info("Loaded %d feature specs", len(self._features))
        return self._features

    def get_feature(self, name: str) -> FeatureSpec:
        """Get a FeatureSpec by name."""
        for f in self.features:
            if f.name == name:
                return f
        available = [f.name for f in self.features]
        raise KeyError(f"Feature {name!r} not found. Available: {available}")

    def get_features_by_directory(self, directory: str) -> list[FeatureSpec]:
        """Get all features in a directory category (labels, camera, etc.)."""
        return [f for f in self.features if f.directory == directory]

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]

    @property
    def directories(self) -> list[str]:
        seen: list[str] = []
        for f in self.features:
            if f.directory not in seen:
                seen.append(f.directory)
        return seen

    # ------------------------------------------------------------------
    # sensor_presence.parquet
    # ------------------------------------------------------------------

    @property
    def sensor_presence_df(self) -> pd.DataFrame:
        """Lazy-load metadata/sensor_presence.parquet."""
        if self._sensor_presence_df is None:
            path = self._remote.get_cached_or_download("metadata/sensor_presence.parquet")
            self._sensor_presence_df = pd.read_parquet(path)
            log.info("Loaded sensor presence: %d clips", len(self._sensor_presence_df))
        return self._sensor_presence_df

    def get_sensor_presence(self, clip_id: str) -> dict[str, bool]:
        """Return a dict mapping sensor/feature name -> is_present for a clip."""
        df = self.sensor_presence_df
        if clip_id not in df.index:
            raise KeyError(f"clip_id {clip_id!r} not found in sensor_presence")
        row = df.loc[clip_id]
        return {col: bool(row[col]) for col in df.columns}

    def is_feature_present(self, clip_id: str, feature_name: str) -> bool:
        """Check if a specific feature/sensor is present for a clip.

        Features that don't appear in sensor_presence (e.g. labels,
        calibration, metadata) are assumed to always be present.
        """
        df = self.sensor_presence_df
        # The sensor_presence columns correspond to sensor feature names
        if feature_name not in df.columns:
            return True  # Not a sensor feature -> always present
        if clip_id not in df.index:
            return True  # Unknown clip -> optimistically assume present
        return bool(df.loc[clip_id, feature_name])


def _parse_features_csv(text: str) -> list[FeatureSpec]:
    """Parse the features.csv manifest into FeatureSpec objects."""
    features: list[FeatureSpec] = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        clip_files_raw = row.get("clip_files_in_zip", "").strip()
        if clip_files_raw:
            clip_files = json.loads(clip_files_raw)
        else:
            clip_files = {}
        features.append(
            FeatureSpec(
                name=row["feature"],
                directory=row["directory"],
                chunk_path_template=row["chunk_path"],
                clip_files=clip_files,
            )
        )
    return features
