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

"""Authenticated access to HuggingFace-hosted files with CDN URL resolution."""

from __future__ import annotations

import io
import logging
import time

from pathlib import Path
from typing import BinaryIO

import requests

from tools.data_converter.pai.pai_remote.config import Config


log = logging.getLogger(__name__)

# CDN URLs are typically valid for ~1 hour; we re-resolve after 50 minutes.
_CDN_TTL_SECONDS = 50 * 60


class HFRemote:
    """Handles authenticated HTTP access to HuggingFace dataset files.

    The HF ``/resolve/`` endpoint redirects to a signed CDN URL.  We cache
    these resolved URLs so that repeated accesses to the same file (e.g. the
    same chunk zip for multiple clip extractions) don't incur extra HEAD
    requests.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {config.token}"})
        # Cache: repo_path -> (cdn_url, resolved_at)
        self._cdn_cache: dict[str, tuple[str, float]] = {}
        # Commit SHA resolved from the first HF response header
        self._commit_sha: str | None = None

    # ------------------------------------------------------------------
    # URL helpers
    # ------------------------------------------------------------------

    def _repo_url(self, repo_path: str) -> str:
        """Full resolve URL for a path inside the repo."""
        return f"{self.config.resolve_base}/{repo_path}"

    def _capture_commit_sha(self, resp: requests.Response) -> None:
        """Extract the commit SHA from HuggingFace response headers (if present)."""
        if self._commit_sha is not None:
            return
        # HF returns the resolved commit SHA in the X-Repo-Commit header
        # on /resolve/ requests (both initial and redirect chain).
        sha = resp.headers.get("X-Repo-Commit")
        if sha is None:
            # Check the redirect history (the initial 302 carries the header)
            for r in resp.history:
                sha = r.headers.get("X-Repo-Commit")
                if sha is not None:
                    break
        if sha is not None:
            self._commit_sha = sha
            log.info("Resolved commit SHA: %s", sha)

    @property
    def commit_sha(self) -> str | None:
        """The HuggingFace commit SHA for the configured revision, or None if not yet resolved."""
        return self._commit_sha

    def resolve_cdn_url(self, repo_path: str) -> str:
        """Follow HF redirect to obtain a signed CDN URL.

        The CDN URL does not require an Authorization header (the signature
        is embedded in query parameters).  We cache it for ``_CDN_TTL_SECONDS``.
        """
        cached = self._cdn_cache.get(repo_path)
        if cached is not None:
            url, ts = cached
            if time.time() - ts < _CDN_TTL_SECONDS:
                return url

        resp = self._session.head(self._repo_url(repo_path), allow_redirects=True, timeout=30)
        resp.raise_for_status()
        self._capture_commit_sha(resp)
        cdn_url = resp.url
        self._cdn_cache[repo_path] = (cdn_url, time.time())
        log.debug("Resolved %s -> %s", repo_path, cdn_url[:120])
        return cdn_url

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------

    def download_bytes(self, repo_path: str) -> bytes:
        """Download a file entirely into memory."""
        url = self._repo_url(repo_path)
        resp = self._session.get(url, allow_redirects=True, timeout=120)
        resp.raise_for_status()
        self._capture_commit_sha(resp)
        return resp.content

    def download_file(self, repo_path: str, dest: Path, *, chunk_size: int = 8 * 1024 * 1024) -> Path:
        """Stream-download a file to disk."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = self._repo_url(repo_path)
        with self._session.get(url, allow_redirects=True, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as fout:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    fout.write(chunk)
        return dest

    def download_to_buffer(self, repo_path: str) -> BinaryIO:
        """Download a file and return a seekable BytesIO."""
        return io.BytesIO(self.download_bytes(repo_path))

    # ------------------------------------------------------------------
    # Cached index files
    # ------------------------------------------------------------------

    def get_cached_or_download(self, repo_path: str) -> Path:
        """Return a local cached copy of a repo file, downloading if needed.

        Files are cached under ``config.cache_dir / revision / repo_path``.
        To force a refresh, delete the cached file.
        """
        local = self.config.cache_dir / self.config.revision / repo_path
        if local.exists():
            log.debug("Using cached %s", local)
            return local
        log.info("Downloading %s ...", repo_path)
        self.download_file(repo_path, local)
        return local
