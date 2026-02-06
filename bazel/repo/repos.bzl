# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module defines repository rules for including external dependencies required by the project.
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", _http_archive = "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def http_archive(name, **kwargs):
    maybe(_http_archive, name = name, **kwargs)

def register_repositories():
    """
    Registers all the necessary repositories for the project when included in a WORKSPACE file.
    """

    ## 3rdparty
    http_archive(
        name = "rules_python",
        sha256 = "4f7e2aa1eb9aa722d96498f5ef514f426c1f55161c3c9ae628c857a7128ceb07",
        strip_prefix = "rules_python-1.0.0",
        url = "https://github.com/bazelbuild/rules_python/releases/download/1.0.0/rules_python-1.0.0.tar.gz",
    )

    http_archive(
        name = "test-data-v3-shards",
        sha256 = "cdb766ba178548f9307b1458bb408f4f92ba1557dd34bb30dac2a303749ab783",
        urls = ["https://gitlab-master.nvidia.com/api/v4/projects/61004/packages/generic/test-data-v3-shards/0.7/test-data-v3-shards.tar.gz"],
    )

    http_archive(
        name = "test-data-v4",
        sha256 = "4be8c5e6409a772dac27389612f224d5c214f89d46a545bd87cd524212a7889f",
        urls = ["https://gitlab-master.nvidia.com/api/v4/projects/61004/packages/generic/test-data-v4/0.1/test-data-v4.tar.gz"],
    )

    http_archive(
        name = "cgrindel_bazel_starlib",
        sha256 = "43e375213dabe0c3928e65412ea7ec16850db93285c8c6f8b0eaa41cacd0f882",
        urls = [
            "https://github.com/cgrindel/bazel-starlib/releases/download/v0.21.0/bazel-starlib.v0.21.0.tar.gz",
        ],
    )

    http_archive(
        name = "aspect_bazel_lib",
        sha256 = "6d758a8f646ecee7a3e294fbe4386daafbe0e5966723009c290d493f227c390b",
        strip_prefix = "bazel-lib-2.7.7",
        url = "https://github.com/aspect-build/bazel-lib/releases/download/v2.7.7/bazel-lib-v2.7.7.tar.gz",
    )

    http_archive(
        name = "bazel_features",
        sha256 = "06f02b97b6badb3227df2141a4b4622272cdcd2951526f40a888ab5f43897f14",
        strip_prefix = "bazel_features-1.9.0",
        url = "https://github.com/bazel-contrib/bazel_features/releases/download/v1.9.0/bazel_features-v1.9.0.tar.gz",
    )
