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

# Prints the GitHub release notes for a release commit to stdout.
#
# Usage: .github/release/release-notes.sh <version> <release-commit>
#
# The notes consist of
# - the body of the release commit (`chore(version): vX.Y.Z`) as highlights, if it has one, and
# - the change list since the previous release tag reachable from the release commit, rendered by
#   cocogitto with release-notes.tera.
#
# For a final release (no pre-release part), pre-release tags are ignored when looking for the
# previous release, so the notes of e.g. v20.0.0 cover everything since the last 19.x release
# reachable from it, not only the changes since the last release candidate.
#
# Requires git and cog, and a clone with full history and tags. Read-only: it does not modify the
# repository it runs in (a final release is rendered in a temporary clone without pre-release tags).

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <version> <release-commit>" >&2
    exit 2
fi

version="$1"
target="$(git rev-parse --verify "$2^{commit}")"
tag="v${version}"

repo_root="$(git rev-parse --show-toplevel)"
template="${repo_root}/.github/release/release-notes.tera"
repo_url="https://github.com/NVIDIA/ncore"

# Previous release tag reachable from the release commit (excluding the release commit itself)
describe_args=(--tags --abbrev=0 --match 'v[0-9]*')
if [[ "${version}" != *-* ]]; then
    describe_args+=(--exclude 'v*-*')
fi
previous="$(git describe "${describe_args[@]}" "${target}^")"

# Highlights: body of the release commit without trailers
highlights="$(git log -1 --format='%b' "${target}" | sed -E '/^(Signed-off-by|Co-authored-by):/d')"

render_changes() {
    (cd "$1" && cog --config .cog.toml changelog --template "${template}" "${previous}..${target}")
}

if [[ "${version}" != *-* ]] && [[ -n "$(git tag --list 'v*-*' --merged "${target}")" ]]; then
    # cog renders one section per tag in the range; drop pre-release tags in a throwaway clone so a
    # final release gets a single change list
    scratch="$(mktemp -d)"
    trap 'rm -rf "${scratch}"' EXIT
    git clone --quiet --no-checkout "${repo_root}" "${scratch}/repo"
    git -C "${scratch}/repo" fetch --quiet "${repo_root}" "${target}"
    git -C "${scratch}/repo" checkout --quiet --detach "${target}"
    git -C "${scratch}/repo" tag --list 'v*-*' | xargs -r git -C "${scratch}/repo" tag --delete >/dev/null
    changes="$(render_changes "${scratch}/repo")"
else
    changes="$(render_changes "${repo_root}")"
fi

if [[ -n "${highlights//[[:space:]]/}" ]]; then
    printf '### Highlights\n\n%s\n\n' "${highlights}"
fi
printf '### Changes\n%s\n\n' "${changes}"
printf '**Full diff**: %s/compare/%s...%s\n' "${repo_url}" "${previous}" "${tag}"
