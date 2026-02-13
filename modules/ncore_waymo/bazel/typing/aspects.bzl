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
Mypy aspect for ncore_waymo that type-checks against @ncore dependencies.

Uses the reusable mypy_with_first_party_deps factory from the parent ncore module
to properly include ncore sources in MYPYPATH for type checking.
"""

load("@ncore//bazel/typing:aspects.bzl", "mypy_with_first_party_deps")

mypy_aspect = mypy_with_first_party_deps(
    mypy_cli = "//bazel/typing:mypy",
    mypy_ini = "//bazel/typing:mypy.ini",
    first_party_repos = ["ncore+"],
)
