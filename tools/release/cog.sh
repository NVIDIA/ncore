#!/usr/bin/env bash
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

# Runs the pinned cocogitto (see bazel/multitool.lock.json) in the workspace, with the repository's
# configuration (.cog.toml):
#
#   bazel run //:cog -- check
#
# Exports its path as COG, so the release notes tool run by the cog bump hooks uses the same cog.

# --- begin runfiles.bash initialization v3 ---
# Copy-pasted from the Bazel Bash runfiles library v3.
set -uo pipefail; set +e; f=bazel_tools/tools/bash/runfiles/runfiles.bash
# shellcheck disable=SC1090
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null || \
source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null || \
source "$0.runfiles/$f" 2>/dev/null || \
source "$(grep -sm1 "^$f " "$0.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
source "$(grep -sm1 "^$f " "$0.exe.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
{ echo>&2 "ERROR: cannot find $f"; exit 1; }; f=; set -e
# --- end runfiles.bash initialization v3 ---

COG="$(rlocation multitool/tools/cog/cog)"
if [[ -z "${COG}" ]]; then
    echo "error: cog not found in the runfiles" >&2
    exit 1
fi
COG="$(cd "$(dirname "${COG}")" && pwd)/$(basename "${COG}")"
export COG

if [[ -z "${BUILD_WORKSPACE_DIRECTORY:-}" ]]; then
    echo "error: run with 'bazel run //:cog -- <args>'" >&2
    exit 2
fi
cd "${BUILD_WORKSPACE_DIRECTORY}"

# Use the repository's configuration unless one is given (cog accepts `--config` only once, before the command;
# without it, cog silently uses defaults, e.g. no `v` tag prefix)
for arg in "$@"; do
    if [[ "${arg}" == --config || "${arg}" == --config=* ]]; then
        exec "${COG}" "$@"
    fi
done
exec "${COG}" --config .cog.toml "$@"
