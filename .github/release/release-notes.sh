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

# Release notes of NCore releases.
#
# The release notes are the markdown body of the release commit (`chore(version): vX.Y.Z`), so they
# are reviewed and edited in the release PR.
#
# Usage:
#   release-notes.sh draft <version>
#       Fill the empty body of the release commit at HEAD with the change list since the previous
#       release (run by the cog post-bump hook, see .cog.toml).
#   release-notes.sh changes <version> <commit>
#       Print the change list of the release at <commit> since the previous release.
#   release-notes.sh publish <version> <commit>
#       Print the GitHub release text of the release at <commit> (used by CI): the commit body
#       without trailers and with wrapped lines joined, followed by a link to the full diff.
#
# The previous release is the latest release tag reachable from the release commit. For a final
# release (no pre-release part), pre-release tags are ignored, so the notes of e.g. v20.0.0 cover
# everything since the last 19.x release reachable from it, not only the changes since the last
# release candidate.
#
# Requires git, and cog for `draft` and `changes`, in a clone with full history and tags.

set -euo pipefail

usage() {
    sed -n '/^# Usage:/,/^#$/s/^# \{0,1\}//p' "$0" >&2
    exit 2
}

[[ $# -ge 2 ]] || usage
mode="$1"
version="$2"
tag="v${version}"

repo_root="$(git rev-parse --show-toplevel)"
template="${repo_root}/.github/release/release-notes.tera"
repo_url="https://github.com/NVIDIA/ncore"

# Latest release tag reachable from the parent of the release commit <commit>
previous_tag() {
    local args=(--tags --abbrev=0 --match 'v[0-9]*')
    if [[ "${version}" != *-* ]]; then
        args+=(--exclude 'v*-*')
    fi
    git describe "${args[@]}" "$1^"
}

# Change list of the release at <commit>, rendered by cog with release-notes.tera
changes() {
    local target previous dir scratch
    target="$(git rev-parse --verify "$1^{commit}")"
    previous="$(previous_tag "${target}")"
    dir="${repo_root}"
    if [[ "${version}" != *-* ]] && [[ -n "$(git tag --list 'v*-*' --merged "${target}")" ]]; then
        # cog renders one section per tag in the range; drop pre-release tags in a throwaway clone so
        # a final release gets a single change list
        scratch="$(mktemp -d)"
        trap 'rm -rf "${scratch}"' EXIT
        git clone --quiet --no-checkout "${repo_root}" "${scratch}/repo"
        git -C "${scratch}/repo" fetch --quiet "${repo_root}" "${target}"
        git -C "${scratch}/repo" checkout --quiet --detach "${target}"
        git -C "${scratch}/repo" tag --list 'v*-*' | xargs -r git -C "${scratch}/repo" tag --delete >/dev/null
        dir="${scratch}/repo"
    fi
    printf '### Changes\n\n'
    (cd "${dir}" && cog --config .cog.toml changelog --template "${template}" "${previous}..${target}") |
        sed -e '/./,$!d' | sed -e ':a' -e '/^\n*$/{$d;N;ba' -e '}'
}

# Join wrapped lines of a markdown document into single lines, so GitHub (which renders single line
# breaks in release text as hard breaks) shows regular paragraphs and list items. Lines are not
# joined across blank lines, list items, headings, quotes, tables, code blocks, or explicit line
# breaks (two trailing spaces or a trailing backslash).
unwrap() {
    awk '
        function flush() { if (have) print buf; have = 0; buf = "" }
        function standalone(s) { return s ~ /^[ \t]*(#|>|\||[-*+][ \t]|[0-9]+[.)][ \t])/ }
        /^[ \t]*(```|~~~)/ { flush(); print; fence = !fence; next }
        fence { print; next }
        /^[ \t]*$/ { flush(); print; next }
        {
            if (have && !standalone($0) && !hard) {
                line = $0; sub(/^[ \t]+/, "", line); buf = buf " " line
            } else {
                flush(); buf = $0; have = 1
            }
            hard = ($0 ~ /(  |\\)$/) || ($0 ~ /^[ \t]*(#|\|)/)
        }
        END { flush() }
    '
}

case "${mode}" in
    draft)
        [[ $# -eq 2 ]] || usage
        if [[ -n "$(git log -1 --format='%b' HEAD | tr -d '[:space:]')" ]]; then
            echo "error: the release commit at HEAD already has a body, not overwriting it" >&2
            exit 1
        fi
        subject="$(git log -1 --format='%s' HEAD)"
        if [[ "${subject}" != "chore(version): ${tag}" ]]; then
            echo "error: HEAD is not the release commit of ${tag} ('${subject}')" >&2
            exit 1
        fi
        message="$(mktemp)"
        printf '%s\n\n%s\n' "${subject}" "$(changes HEAD)" > "${message}"
        # verbatim: keep the `###` headings, which git would otherwise strip as comments
        git commit --quiet --amend --cleanup=verbatim --file "${message}"
        rm -f "${message}"
        echo "Release commit body drafted. Add highlights and edit it with 'git commit --amend --cleanup=scissors'."
        ;;
    changes)
        [[ $# -eq 3 ]] || usage
        changes "$3"
        ;;
    publish)
        [[ $# -eq 3 ]] || usage
        target="$(git rev-parse --verify "$3^{commit}")"
        body="$(git log -1 --format='%b' "${target}" | sed -E '/^(Signed-off-by|Co-authored-by):/d')"
        if [[ -z "${body//[[:space:]]/}" ]]; then
            # No release notes written: fall back to the generated change list
            body="$(changes "${target}")"
        fi
        printf '%s\n\n' "${body}" | unwrap
        printf '**Full diff**: %s/compare/%s...%s\n' "${repo_url}" "$(previous_tag "${target}")" "${tag}"
        ;;
    *)
        usage
        ;;
esac
