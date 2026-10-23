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
#       Fill the empty body of the release commit at HEAD with the change list of the release
#       (run by the cog post-bump hook, see .cog.toml).
#   release-notes.sh changes <version> <commit>
#       Print the change list of the release at <commit>.
#   release-notes.sh publish <version> <commit>
#       Print the GitHub release text of the release at <commit> (used by CI): the commit body
#       without trailers and with wrapped lines joined, followed by links to the diffs.
#
# Change lists cover the whole release: all changes since the previous final release (pre-release
# tags are ignored, e.g. v20.0.0 and v20.0.0-rc.1 list everything since the last 19.x release).
# Pre-releases following an earlier pre-release of the same release also list the changes since that
# one separately:
#
# - alpha / beta: the changes since the previous pre-release, then all changes of the release
#   (collapsed)
# - rc: all changes of the release, then the changes since the previous pre-release
#
# and link both diffs.
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

# Latest final release tag reachable from the parent of the release commit <commit>
previous_release() {
    git describe --tags --abbrev=0 --match 'v[0-9]*' --exclude 'v*-*' "$1^" 2>/dev/null || true
}

# Pre-release tags of the release of <commit> reachable from its parent (since the previous final
# release), newest first (by ancestry: the commits between them are walked from <commit> backwards)
previous_prereleases() {
    local commit="$1" base range
    base="$(previous_release "${commit}")"
    range="${base:+${base}..}${commit}^"
    git log --format='%H' "${range}" | while read -r c; do
        git tag --points-at "${c}" --list 'v[0-9]*-*'
    done
}

# Change list (cog, release-notes.tera) of the commits in <from>..<to>, headings demoted by <level>
# extra levels (0: `####` per commit type)
change_list() {
    local from="$1" to="$2" level="${3:-0}" dir="${repo_root}" scratch="" range inner
    # (commit ids, the pre-release tags of the range may be dropped below)
    to="$(git rev-parse --verify "${to}^{commit}")"
    [[ -z "${from}" ]] || from="$(git rev-parse --verify "${from}^{commit}")"
    range="${from:+${from}..}${to}"
    inner="$(git log --format='%H' "${range}^" 2>/dev/null | while read -r c; do git tag --points-at "${c}" --list 'v*-*'; done)"
    if [[ -n "${inner}" ]]; then
        # cog renders one section per tag in the range; drop pre-release tags in a throwaway clone so
        # the range gets a single change list
        scratch="$(mktemp -d)"
        git clone --quiet --no-checkout "${repo_root}" "${scratch}/repo"
        git -C "${scratch}/repo" fetch --quiet "${repo_root}" "${to}"
        git -C "${scratch}/repo" checkout --quiet --detach "${to}"
        git -C "${scratch}/repo" tag --list 'v*-*' | xargs -r git -C "${scratch}/repo" tag --delete >/dev/null
        dir="${scratch}/repo"
    fi
    (cd "${dir}" && cog --config "${repo_root}/.cog.toml" changelog --template "${template}" "${range}") |
        sed -e '/./,$!d' | sed -e ':a' -e '/^\n*$/{$d;N;ba' -e '}' |
        { if [[ "${level}" -gt 0 ]]; then sed -E "s/^(#+)/\1$(printf '#%.0s' $(seq "${level}"))/"; else cat; fi }
    [[ -z "${scratch}" ]] || rm -rf "${scratch}"
}

compare() { printf '%s/compare/%s...%s' "${repo_url}" "$1" "$2"; }

# Change list of the release at <commit> (see the header)
changes() {
    local target base prereleases previous kind
    target="$(git rev-parse --verify "$1^{commit}")"
    base="$(previous_release "${target}")"
    mapfile -t prereleases < <(previous_prereleases "${target}")
    previous="${prereleases[0]:-}"

    # final releases, and the first pre-release of a release: a single change list
    if [[ "${version}" != *-* ]] || [[ -z "${previous}" ]]; then
        printf '### Changes%s\n\n' "${base:+ since ${base}}"
        change_list "${base}" "${target}"
        return
    fi

    kind="${version#*-}"
    kind="${kind%%.*}" # alpha / beta / rc

    if [[ "${kind}" == rc ]]; then
        printf '### Changes%s\n\n' "${base:+ since ${base}}"
        change_list "${base}" "${target}"
        printf '\n### New since %s\n\n' "${previous}"
        change_list "${previous}" "${target}" 1
    else
        printf '### Changes since %s\n\n' "${previous}"
        change_list "${previous}" "${target}"
        printf '\n<details>\n<summary>All changes%s</summary>\n\n' "${base:+ since ${base}}"
        change_list "${base}" "${target}"
        printf '\n</details>\n'
    fi
}

# Join wrapped lines of a markdown document into single lines, so GitHub (which renders single line
# breaks in release text as hard breaks) shows regular paragraphs and list items. Lines are not
# joined across blank lines, list items, headings, quotes, tables, code blocks, HTML, or explicit
# line breaks (two trailing spaces or a trailing backslash).
unwrap() {
    awk '
        function flush() { if (have) print buf; have = 0; buf = "" }
        function standalone(s) { return s ~ /^[ \t]*(#|>|\||<|[-*+][ \t]|[0-9]+[.)][ \t])/ }
        /^[ \t]*(```|~~~)/ { flush(); print; fence = !fence; next }
        fence { print; next }
        /^[ \t]*$/ { flush(); print; next }
        {
            if (have && !standalone($0) && !hard) {
                line = $0; sub(/^[ \t]+/, "", line); buf = buf " " line
            } else {
                flush(); buf = $0; have = 1
            }
            hard = ($0 ~ /(  |\\)$/) || ($0 ~ /^[ \t]*(#|\||<)/)
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
        base="$(previous_release "${target}")"
        mapfile -t prereleases < <(previous_prereleases "${target}")
        previous="${prereleases[0]:-}"
        if [[ "${version}" == *-* ]] && [[ -n "${previous}" ]]; then
            printf '**Changes since %s**: %s  \n' "${previous}" "$(compare "${previous}" "${tag}")"
        fi
        if [[ -n "${base}" ]]; then
            printf '**Full diff**: %s\n' "$(compare "${base}" "${tag}")"
        fi
        ;;
    *)
        usage
        ;;
esac
