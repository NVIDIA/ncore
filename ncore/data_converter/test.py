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


def test_import_package():
    """Make sure package can be imported"""

    import ncore.data_converter  # noqa: F401


def test_config_defaults():
    """Converter configs write zarr format 2 by default, file-based configs require a root directory"""

    import pytest

    from ncore.data_converter import BaseDataConverterConfig, FileBasedDataConverterConfig

    def base_config(**kwargs) -> BaseDataConverterConfig:
        return BaseDataConverterConfig(
            output_dir="/tmp/out",
            no_cameras=False,
            camera_ids=None,
            no_lidars=False,
            lidar_ids=None,
            no_radars=False,
            radar_ids=None,
            verbose=False,
            debug=False,
            debug_port=5678,
            **kwargs,
        )

    def file_based_config(**kwargs) -> FileBasedDataConverterConfig:
        return FileBasedDataConverterConfig(
            output_dir="/tmp/out",
            no_cameras=False,
            camera_ids=None,
            no_lidars=False,
            lidar_ids=None,
            no_radars=False,
            radar_ids=None,
            verbose=False,
            debug=False,
            debug_port=5678,
            **kwargs,
        )

    assert base_config().zarr_format == 2
    assert base_config(zarr_format=3).zarr_format == 3
    assert file_based_config(root_dir="/tmp/in").zarr_format == 2

    with pytest.raises(ValueError, match="--root-dir is required"):
        file_based_config()
