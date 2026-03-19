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

"""Data provider abstraction for PAI clip data access.

Defines a :class:`ClipDataProvider` protocol and two implementations:

- :class:`LocalClipDataProvider` -- reads from a local pai-clip-dl output directory.
- :class:`StreamingClipDataProvider` -- streams from HuggingFace without prior download.
"""

from __future__ import annotations

import io
import json
import logging

from pathlib import Path
from typing import Any, Dict, Protocol, cast, runtime_checkable

import pandas as pd
import pyarrow.parquet as pq

from upath import UPath

from tools.data_converter.pai.pai_remote.streaming import StreamingZipAccess


logger = logging.getLogger(__name__)


# ======================================================================
# Protocol
# ======================================================================


@runtime_checkable
class ClipDataProvider(Protocol):
    """Provides data access for a single PAI clip.

    Implementations hide whether data comes from local disk or a remote
    streaming source.  The converter code depends only on this interface.
    """

    @property
    def clip_id(self) -> str: ...

    def load_parquet(self, key: str) -> pd.DataFrame:
        """Load a parquet file by logical key.

        Keys match those returned by :func:`find_clip_files`:
        ``"egomotion"``, ``"camera_intrinsics"``, ``"sensor_extrinsics"``,
        ``"vehicle_dimensions"``, ``"obstacle"``, ``"lidar_intrinsics"``,
        ``"lidar_top_360fov"``, ``"<camera_id>_timestamps"``,
        ``"<camera_id>_blurred_boxes"``.
        """
        ...

    def load_parquet_schema(self, key: str) -> Any:
        """Load the Arrow schema for a parquet file (used for lidar metadata fallback)."""
        ...

    def get_video_path(self, camera_id: str) -> Path:
        """Return a local filesystem path to the MP4 for *camera_id*.

        For local providers this is the original file; for streaming providers
        the video is downloaded to a temporary file first.
        """
        ...

    def has_file(self, key: str) -> bool:
        """Return True if data exists for *key*."""
        ...

    def get_sensor_presence(self) -> pd.Series:
        """Return sensor-presence flags for this clip (Series indexed by sensor name)."""
        ...

    def get_platform_class(self) -> str:
        """Return the platform class string (e.g. ``"hyperion_8"``)."""
        ...

    def get_source_metadata(self) -> Dict[str, Any]:
        """Return provenance metadata about the data source.

        The returned dict may include keys such as ``clip_id``,
        ``repo_id``, ``revision``, ``commit_sha``, and
        ``download_timestamp``.  Missing keys indicate the information
        is not available (e.g. older local downloads without
        ``provenance.json``).
        """
        ...

    def close(self) -> None:
        """Close any open resources."""
        ...


# ======================================================================
# Local provider (pai-clip-dl output on disk)
# ======================================================================


class LocalClipDataProvider(ClipDataProvider):
    """Read clip data from a local pai-clip-dl output directory.

    Expects the directory layout produced by ``pai-clip-dl download``::

        <clip_dir>/
            calibration/  camera/  labels/  lidar/  metadata/
    """

    def __init__(self, clip_dir: UPath, clip_id: str) -> None:
        from tools.data_converter.pai.utils import find_clip_files

        self._clip_id = clip_id
        self._clip_dir = clip_dir
        self._files = find_clip_files(clip_dir, clip_id)
        self._metadata_dir = clip_dir / "metadata"

    @property
    def clip_id(self) -> str:
        return self._clip_id

    def load_parquet(self, key: str) -> pd.DataFrame:
        return pd.read_parquet(str(self._files[key]))

    def load_parquet_schema(self, key: str) -> Any:
        return pq.read_schema(str(self._files[key]))

    def get_video_path(self, camera_id: str) -> Path:
        return Path(str(self._files[f"{camera_id}_video"]))

    def has_file(self, key: str) -> bool:
        return key in self._files and self._files[key].exists()

    def get_sensor_presence(self) -> pd.Series:
        sp_path = self._metadata_dir / "feature_presence.parquet"
        if sp_path.exists():
            df = pd.read_parquet(str(sp_path))
            return df.loc[self._clip_id]  # type: ignore[return-value]
        # Fallback: infer from extrinsics
        logger.info("feature_presence.parquet not found, inferring from extrinsics")
        ext_df = self.load_parquet("sensor_extrinsics")
        sensor_names = ext_df.index.get_level_values("sensor_name").unique()
        return pd.Series(True, index=sensor_names)

    def get_platform_class(self) -> str:
        dc_path = self._metadata_dir / "data_collection.parquet"
        df = pd.read_parquet(str(dc_path))
        return str(df.platform_class[self._clip_id])

    def get_source_metadata(self) -> Dict[str, Any]:
        provenance_path = self._metadata_dir / "provenance.json"
        if provenance_path.exists():
            return json.loads(provenance_path.read_text())  # type: ignore[no-any-return]
        # Fallback for older downloads that lack provenance.json
        logger.info("provenance.json not found, returning minimal source metadata")
        return {"clip_id": self._clip_id}

    def close(self) -> None:
        """Close any open resources."""
        pass


# ======================================================================
# Streaming provider (direct HuggingFace access)
# ======================================================================


class StreamingClipDataProvider:
    """Stream clip data directly from HuggingFace without a prior download.

    Uses ``pai_remote`` library classes (:class:`HFRemote`,
    :class:`ClipIndex`, :class:`StreamingZipAccess`) for data access.

    Calibration parquet files (per-chunk) are downloaded in full and filtered
    to the target clip.  Zip-based features (camera, lidar, labels) are
    fetched via HTTP range requests.  Video files are written to a temporary
    directory since ffmpeg requires a local file path.
    """

    def __init__(
        self,
        clip_id: str,
        remote: Any,  # pai_remote.HFRemote
        index: Any,  # pai_remote.ClipIndex
        temp_dir: Path,
        chunk_parquet_cache: Dict[str, pd.DataFrame] | None = None,
    ) -> None:
        self._clip_id = clip_id
        self._remote = remote
        self._index = index
        self._temp_dir = Path(temp_dir)
        self._temp_dir.mkdir(parents=True, exist_ok=True)

        self._chunk_id: int = index.get_chunk_id(clip_id)

        # Shared cache for chunk-level parquet DataFrames (avoids re-downloading
        # the same chunk parquet when converting multiple clips from the same chunk).
        self._chunk_parquet_cache: Dict[str, pd.DataFrame] = (
            chunk_parquet_cache if chunk_parquet_cache is not None else {}
        )

        # Cache for streamed zip handles (one per feature)
        self._zip_cache: Dict[str, Any] = {}  # feature_name -> StreamingZipAccess

        # Cache for video temp file paths
        self._video_paths: Dict[str, Path] = {}

        # Determine which features prefer ".offline" variants
        self._feature_names = {f.name for f in self._index.features}

        logger.info(f"Streaming provider for clip {clip_id} (chunk {self._chunk_id})")

    @property
    def clip_id(self) -> str:
        return self._clip_id

    # ------------------------------------------------------------------
    # Feature resolution helpers
    # ------------------------------------------------------------------

    def _prefer_offline_feature(self, base_name: str) -> str:
        """Return ``"base_name.offline"`` if available in the feature manifest, else *base_name*."""
        offline = f"{base_name}.offline"
        if offline in self._feature_names:
            return offline
        return base_name

    def _resolve_feature_key(self, key: str) -> tuple[str, str | None]:
        """Map a logical key to a (feature_name, clip_filename_or_None) pair.

        For calibration (parquet-based) features, the clip_filename is None.
        For zip-based features, it is the specific file inside the zip.

        Returns:
            Tuple of (feature_name, clip_filename_inside_zip_or_None).
        """
        # Calibration / metadata parquet features
        if key in ("camera_intrinsics", "sensor_extrinsics", "vehicle_dimensions", "lidar_intrinsics"):
            return self._prefer_offline_feature(key), None

        # Label features
        if key == "egomotion":
            feat_name = self._prefer_offline_feature("egomotion")
            feat = self._index.get_feature(feat_name)
            clip_files = feat.clip_filenames(self._clip_id)
            # Get the first (and only) clip file
            filename = next(iter(clip_files.values()))
            return feat_name, filename

        if key == "obstacle":
            feat_name = self._prefer_offline_feature("obstacle")
            feat = self._index.get_feature(feat_name)
            clip_files = feat.clip_filenames(self._clip_id)
            filename = next(iter(clip_files.values()))
            return feat_name, filename

        # Camera features: <camera_id>_timestamps, <camera_id>_blurred_boxes
        for suffix, clip_file_key in [("_timestamps", "frame_timestamps"), ("_blurred_boxes", "blurred_boxes")]:
            if key.endswith(suffix):
                camera_id = key[: -len(suffix)]
                feat = self._index.get_feature(camera_id)
                clip_files = feat.clip_filenames(self._clip_id)
                return camera_id, clip_files[clip_file_key]

        # Lidar
        if key == "lidar_top_360fov":
            feat = self._index.get_feature("lidar_top_360fov")
            clip_files = feat.clip_filenames(self._clip_id)
            filename = next(iter(clip_files.values()))
            return "lidar_top_360fov", filename

        raise KeyError(f"Unknown data key: {key!r}")

    # ------------------------------------------------------------------
    # Zip access helpers
    # ------------------------------------------------------------------

    def _get_zip_access(self, feature_name: str) -> Any:
        """Get or create a StreamingZipAccess for *feature_name*."""

        if feature_name not in self._zip_cache:
            feat = self._index.get_feature(feature_name)
            sza = StreamingZipAccess.from_feature(self._remote, feat, self._chunk_id)
            sza.open()
            self._zip_cache[feature_name] = sza
        return self._zip_cache[feature_name]

    # ------------------------------------------------------------------
    # Chunk parquet helpers
    # ------------------------------------------------------------------

    def _load_chunk_parquet(self, feature_name: str) -> pd.DataFrame:
        """Download a chunk-level parquet and filter to this clip."""
        cache_key = f"{feature_name}__chunk_{self._chunk_id}"
        if cache_key not in self._chunk_parquet_cache:
            feat = self._index.get_feature(feature_name)
            repo_path = feat.chunk_path(self._chunk_id)
            logger.info(f"Downloading chunk parquet: {repo_path}")
            raw = self._remote.download_bytes(repo_path)
            df = pd.read_parquet(io.BytesIO(raw))
            self._chunk_parquet_cache[cache_key] = df

        df = self._chunk_parquet_cache[cache_key]
        return _filter_parquet_to_clip(df, self._clip_id)

    # ------------------------------------------------------------------
    # ClipDataProvider interface
    # ------------------------------------------------------------------

    def load_parquet(self, key: str) -> pd.DataFrame:
        feature_name, clip_filename = self._resolve_feature_key(key)
        feat = self._index.get_feature(feature_name)

        if feat.is_parquet:
            # Calibration: download chunk parquet, filter to clip
            return self._load_chunk_parquet(feature_name)
        else:
            # Zip-based: stream the specific file
            assert clip_filename is not None
            sza = self._get_zip_access(feature_name)
            data = sza.read(clip_filename)
            return pd.read_parquet(io.BytesIO(data))

    def load_parquet_schema(self, key: str) -> Any:
        feature_name, clip_filename = self._resolve_feature_key(key)
        feat = self._index.get_feature(feature_name)

        if feat.is_parquet:
            raw = self._remote.download_bytes(feat.chunk_path(self._chunk_id))
            return pq.read_schema(io.BytesIO(raw))
        else:
            assert clip_filename is not None
            sza = self._get_zip_access(feature_name)
            data = sza.read(clip_filename)
            return pq.read_schema(io.BytesIO(data))

    def get_video_path(self, camera_id: str) -> Path:
        if camera_id in self._video_paths:
            return self._video_paths[camera_id]

        feat = self._index.get_feature(camera_id)
        clip_files = feat.clip_filenames(self._clip_id)
        video_filename = clip_files["video"]

        sza = self._get_zip_access(camera_id)

        logger.info(f"Streaming video {video_filename} to temp file")
        data = sza.read(video_filename)

        # Write to a persistent temp file (not auto-deleted)
        video_path = self._temp_dir / video_filename
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(data)

        self._video_paths[camera_id] = video_path
        logger.info(f"Wrote {len(data) / 1024 / 1024:.1f} MB video to {video_path}")
        return video_path

    def has_file(self, key: str) -> bool:
        try:
            feature_name, clip_filename = self._resolve_feature_key(key)
        except KeyError:
            return False

        feat = self._index.get_feature(feature_name)

        if feat.is_parquet:
            # Calibration parquets always exist at chunk level;
            # check if the clip has rows after filtering.
            try:
                df = self._load_chunk_parquet(feature_name)
                return len(df) > 0
            except Exception:
                return False
        else:
            # Zip-based: check if the clip file is in the zip
            if clip_filename is None:
                return False
            try:
                sza = self._get_zip_access(feature_name)
                return clip_filename in sza.namelist()
            except Exception:
                return False

    def get_sensor_presence(self) -> pd.Series:
        sp = self._index.get_sensor_presence(self._clip_id)
        return pd.Series(sp)

    def get_platform_class(self) -> str:
        repo_path = "metadata/data_collection.parquet"
        cache_key = "__metadata__data_collection"
        if cache_key not in self._chunk_parquet_cache:
            logger.info(f"Downloading metadata: {repo_path}")
            raw = self._remote.download_bytes(repo_path)
            self._chunk_parquet_cache[cache_key] = pd.read_parquet(io.BytesIO(raw))

        df = self._chunk_parquet_cache[cache_key]
        filtered = _filter_parquet_to_clip(df, self._clip_id)
        return str(filtered.platform_class.iloc[0])

    def get_source_metadata(self) -> Dict[str, Any]:
        return {
            "clip_id": self._clip_id,
            "repo_id": self._remote.config.repo_id,
            "revision": self._remote.config.revision,
            "commit_sha": self._remote.commit_sha,
        }

    def close(self) -> None:
        """Close any open streaming zip handles."""
        for sza in self._zip_cache.values():
            try:
                sza.close()
            except Exception:
                pass
        self._zip_cache.clear()


# ======================================================================
# Helpers
# ======================================================================


def _filter_parquet_to_clip(df: pd.DataFrame, clip_id: str) -> pd.DataFrame:
    """Filter a DataFrame to rows belonging to *clip_id*.

    Handles multiple index layouts (same logic as pai_remote.downloader).
    """
    if isinstance(df.index, pd.MultiIndex):
        if "clip_id" in df.index.names:
            try:
                return cast(pd.DataFrame, df.xs(clip_id, level="clip_id", drop_level=False))
            except KeyError:
                return df.iloc[0:0]
    elif df.index.name == "clip_id":
        if clip_id in df.index:
            return df.loc[[clip_id]]
        return df.iloc[0:0]
    elif "clip_id" in df.columns:
        return df[df["clip_id"] == clip_id]
    return df
