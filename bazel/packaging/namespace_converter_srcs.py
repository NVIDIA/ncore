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

"""Rewrite in-repo tools.* converter sources into ncore.converters for the wheel."""

from __future__ import annotations

import argparse
import sys

from pathlib import Path
from typing import List, Optional


_SPDX_INIT = """# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""

_REPLACEMENTS = (
    ("from tools.data_converter", "from ncore.converters"),
    ("import tools.data_converter", "import ncore.converters"),
    ("from tools.debug", "from ncore.converters.debug"),
    ("from ncore_repo.tools.debug", "from ncore.converters.debug"),
    ("from external.ncore_repo.tools.debug", "from ncore.converters.debug"),
)


def _dest_for(src: Path) -> Path:
    parts = src.parts
    if src.name == "debug.py":
        return Path("debug.py")
    if "pai_remote" in parts:
        return Path("pai") / "pai_remote" / src.name
    if "pai" in parts:
        return Path("pai") / src.name
    if src.name == "cli.py":
        return Path("cli.py")
    raise ValueError(f"no wheel path mapping for {src}")


def _rewrite(text: str) -> str:
    for old, new in _REPLACEMENTS:
        text = text.replace(old, new)
    return text


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("srcs", nargs="+", type=Path)
    args = parser.parse_args(argv)

    written: set[Path] = set()
    for src in args.srcs:
        dest = args.out_root / _dest_for(src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(_rewrite(src.read_text(encoding="utf-8")), encoding="utf-8")
        written.add(dest)

    for init_dir in (
        args.out_root,
        args.out_root / "pai",
        args.out_root / "pai" / "pai_remote",
    ):
        init_dir.mkdir(parents=True, exist_ok=True)
        init_file = init_dir / "__init__.py"
        if init_file not in written:
            init_file.write_text(_SPDX_INIT, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
