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

# Bazel workspace status (`--workspace_status_command`, see .bazelrc): the git commit the build is made from and the
# number of commits since the release commit, read by the documentation build. STABLE_ keys: targets reading them
# rebuild when the commit changes, not on every build.
# Outside a git checkout (e.g. a source archive) the key is omitted.

set -euo pipefail

if commit="$(git rev-parse HEAD 2>/dev/null)"; then
    echo "STABLE_GIT_COMMIT ${commit}"
    # Commits since the release commit of the version (the commit that last changed `embed_label`): 0 for the
    # release commit itself, which the release tag points to
    release_commit="$(git log -1 --format=%H -G '^build --embed_label=' HEAD -- .bazelrc 2>/dev/null || true)"
    if [[ -n "${release_commit}" ]]; then
        echo "STABLE_GIT_COMMITS_SINCE_RELEASE $(git rev-list --count "${release_commit}..HEAD")"
    fi
fi
