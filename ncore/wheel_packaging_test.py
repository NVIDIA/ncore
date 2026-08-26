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

import configparser
import os
import unittest
import zipfile

from pathlib import Path


def _wheel_path() -> Path:
    srcdir = Path(os.environ["TEST_SRCDIR"])
    matches = list(srcdir.glob("*/ncore/nvidia_ncore-*-py3-none-any.whl"))
    if not matches:
        matches = list(srcdir.glob("ncore/nvidia_ncore-*-py3-none-any.whl"))
    assert matches, "nvidia-ncore wheel not found in test runfiles"
    assert len(matches) == 1, matches
    return matches[0]


class TestPaiWheelPackaging(unittest.TestCase):
    def test_ncore_wheel_includes_pai_converter_entry_point(self) -> None:
        with zipfile.ZipFile(_wheel_path()) as wheel:
            names = wheel.namelist()

            top_level = {name.split("/", 1)[0] for name in names}
            self.assertIn("ncore", top_level)
            self.assertNotIn("tools", top_level)

            self.assertIn("ncore/converters/cli.py", names)
            self.assertIn("ncore/converters/debug.py", names)
            self.assertIn("ncore/converters/pai/converter.py", names)
            self.assertIn("ncore/converters/pai/pai_remote/remote.py", names)
            self.assertNotIn("ncore/converters/pai/pai_remote/downloader.py", names)

            dist_info = next(name for name in names if name.endswith(".dist-info/entry_points.txt"))
            entry_points = configparser.ConfigParser()
            entry_points.read_string(wheel.read(dist_info).decode("utf-8"))
            self.assertEqual(
                entry_points["console_scripts"]["ncore-convert"],
                "ncore.converters.pai.converter:cli",
            )

            metadata_name = dist_info.replace("entry_points.txt", "METADATA")
            metadata = wheel.read(metadata_name).decode("utf-8")
            self.assertIn("Requires-Dist: click; extra == 'pai'", metadata)
            self.assertIn("Requires-Dist: DracoPy>=2.0.0; extra == 'pai'", metadata)
            self.assertIn("Requires-Dist: PyNvVideoCodec; extra == 'pai'", metadata)
            self.assertNotIn("Requires-Dist: rich; extra == 'pai'", metadata)
