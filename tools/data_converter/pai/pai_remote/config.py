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

"""Configuration constants and token handling."""

from __future__ import annotations

import os

from dataclasses import dataclass, field
from pathlib import Path


REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"
REPO_TYPE = "dataset"
DEFAULT_REVISION = "main"
BASE_URL = "https://huggingface.co/datasets"


def _default_cache_dir() -> Path:
    return Path(os.environ.get("pai_remote_CACHE", Path.home() / ".cache" / "pai-clip-dl"))


@dataclass
class Config:
    """Runtime configuration for pai-clip-dl."""

    token: str
    repo_id: str = REPO_ID
    revision: str = DEFAULT_REVISION
    cache_dir: Path = field(default_factory=_default_cache_dir)

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, token: str | None = None, **kwargs) -> Config:
        """Create config, reading HF_TOKEN from environment if not provided."""
        if token is None:
            token = os.environ.get("HF_TOKEN", "")
        if not token:
            raise ValueError("HuggingFace token required. Pass --token or set HF_TOKEN env var.")
        return cls(token=token, **kwargs)

    @property
    def resolve_base(self) -> str:
        """Base URL for file resolution (e.g. https://huggingface.co/datasets/nvidia/.../resolve/main)."""
        return f"{BASE_URL}/{self.repo_id}/resolve/{self.revision}"
