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

# Tests release-notes.sh on a scratch repository with a final release, pre-releases (alpha, beta,
# rc) of the next release, and the next final release. Requires git and cog.
#
# Usage: .github/release/release-notes_test.sh

set -euo pipefail

source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${source_dir}/../.." && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf "${scratch}"' EXIT

cd "${scratch}"
git init --quiet --initial-branch=main
git config user.name "Dev One"
git config user.email "one@example.com"
git config commit.gpgSign false
git config tag.gpgSign false
git config core.excludesFile /dev/null
mkdir -p .github/release
cp "${source_dir}/release-notes.sh" "${source_dir}/release-notes.tera" .github/release/
cp "${repo_root}/.cog.toml" .
printf '\n[[changelog.authors]]\nsignature = "Dev One"\nusername = "dev-one"\n' >> .cog.toml

day=0
commit() {
    day=$((day + 1))
    echo "$1" >> changes.txt
    git add --force changes.txt .cog.toml .github
    GIT_AUTHOR_DATE="2026-01-${day}T10:00:00" GIT_COMMITTER_DATE="2026-01-${day}T10:00:00" \
        git commit --quiet --message "$1"
}
release() {
    day=$((day + 1))
    echo "build --embed_label=$1" > .bazelrc
    git add --force .bazelrc
    GIT_AUTHOR_DATE="2026-01-${day}T12:00:00" GIT_COMMITTER_DATE="2026-01-${day}T12:00:00" \
        git commit --quiet --message "chore(version): v$1"
    git tag --annotate "v$1" --message "v$1"
}

commit "feat: initial"
release 1.0.0
commit "feat(data)!: breaking feature"
commit "perf(data): faster reads"
release 2.0.0-alpha.1
commit "fix(data): alpha fix"
release 2.0.0-alpha.2
commit "fix(data): beta fix"
release 2.0.0-beta.1
commit "fix(data): rc fix"
release 2.0.0-rc.1
commit "fix(data): second rc fix"
release 2.0.0-rc.2
commit "docs: final docs"
release 2.0.0

failures=0
notes() { .github/release/release-notes.sh publish "$1" "v$1"; }
# (outputs are captured before matching: `grep --quiet` stops reading at the first match, which fails
# the writer of a pipe with SIGPIPE under `set -o pipefail`)
matches() { grep --quiet --perl-regexp --null-data -- "$2" <<< "$1"; }
expect() {
    local version="$1" description="$2" pattern="$3"
    if ! matches "$(notes "${version}")" "${pattern}"; then
        echo "FAIL ${version}: ${description}" >&2
        notes "${version}" | sed 's/^/    /' >&2
        failures=$((failures + 1))
    fi
}
expect_not() {
    local version="$1" description="$2" pattern="$3"
    if matches "$(notes "${version}")" "${pattern}"; then
        echo "FAIL ${version}: ${description}" >&2
        notes "${version}" | sed 's/^/    /' >&2
        failures=$((failures + 1))
    fi
}

# Section of the notes of <version> from the line matching <heading> to the next `###` heading or link
section() {
    awk -v heading="$2" '
        index($0, heading) == 1 { found = 1; print; next }
        found && (/^### / || /^<\/?details>/ || /^\*\*/) { found = 0 }
        found { print }
    ' <<< "$(notes "$1")"
}
expect_section() {
    local version="$1" heading="$2" description="$3" pattern="$4"
    if ! matches "$(section "${version}" "${heading}")" "${pattern}"; then
        echo "FAIL ${version}: ${description}" >&2
        section "${version}" "${heading}" | sed 's/^/    /' >&2
        failures=$((failures + 1))
    fi
}
expect_section_not() {
    local version="$1" heading="$2" description="$3" pattern="$4"
    if matches "$(section "${version}" "${heading}")" "${pattern}"; then
        echo "FAIL ${version}: ${description}" >&2
        section "${version}" "${heading}" | sed 's/^/    /' >&2
        failures=$((failures + 1))
    fi
}

# First pre-release of a release: a single change list since the previous final release
expect 2.0.0-alpha.1 "single change list" '^### Changes since v1\.0\.0\n'
expect_section 2.0.0-alpha.1 "### Changes since v1.0.0" "lists its changes" '(?=(?s).*breaking feature)(?=(?s).*faster reads)'
expect_not 2.0.0-alpha.1 "no incremental section" 'All changes|New since|Changes since v2'
expect 2.0.0-alpha.1 "full diff" '\*\*Full diff\*\*: \S+/compare/v1\.0\.0\.\.\.v2\.0\.0-alpha\.1\n'

# alpha / beta: changes since the previous pre-release first, all changes collapsed
expect 2.0.0-alpha.2 "incremental section first" '^### Changes since v2\.0\.0-alpha\.1\n'
expect_section 2.0.0-alpha.2 "### Changes since v2.0.0-alpha.1" "incremental changes" '\(\*\*data\*\*\) alpha fix \(\w+\) - @dev-one'
expect_section_not 2.0.0-alpha.2 "### Changes since v2.0.0-alpha.1" "no earlier changes" 'breaking feature|faster reads'
expect 2.0.0-alpha.2 "all changes collapsed" '<details>\n<summary>All changes since v1\.0\.0</summary>\n(?=(?s).*breaking feature)(?=(?s).*alpha fix)(?s).*</details>'
expect_section 2.0.0-beta.1 "### Changes since v2.0.0-alpha.2" "incremental since alpha.2" '^(?!(?s).*alpha fix)(?s).*beta fix'
expect 2.0.0-beta.1 "both diffs" '\*\*Changes since v2\.0\.0-alpha\.2\*\*: \S+/compare/v2\.0\.0-alpha\.2\.\.\.v2\.0\.0-beta\.1  \n\*\*Full diff\*\*: \S+/compare/v1\.0\.0\.\.\.v2\.0\.0-beta\.1\n'

# rc: all changes first, then the changes since the previous pre-release
expect 2.0.0-rc.1 "all changes first" '^### Changes since v1\.0\.0\n(?s).*\n### New since v2\.0\.0-beta\.1\n'
expect_section 2.0.0-rc.1 "### Changes since v1.0.0" "all changes" '(?=(?s).*breaking feature)(?=(?s).*alpha fix)(?=(?s).*beta fix)(?=(?s).*rc fix)'
expect_section 2.0.0-rc.1 "### New since v2.0.0-beta.1" "new since previous pre-release" '^(?!(?s).*beta fix)(?s)\#\#\# New since v2\.0\.0-beta\.1\n\n##### .*rc fix'
expect_not 2.0.0-rc.1 "not collapsed" '<details>'
expect_section 2.0.0-rc.2 "### New since v2.0.0-rc.1" "new since rc.1" '^(?!(?s).*[^d] rc fix)(?s).*second rc fix'

# Final release: a single change list since the previous final release, including the pre-releases
expect 2.0.0 "single change list" '^### Changes since v1\.0\.0\n'
expect_section 2.0.0 "### Changes since v1.0.0" "all changes" '(?=(?s).*breaking feature)(?=(?s).*alpha fix)(?=(?s).*beta fix)(?=(?s).*second rc fix)(?=(?s).*final docs)'
expect_not 2.0.0 "no pre-release sections" 'New since|All changes|Changes since v2|<details>'
expect 2.0.0 "full diff only" '\n\*\*Full diff\*\*: \S+/compare/v1\.0\.0\.\.\.v2\.0\.0\n$'
expect_not 2.0.0 "no release commits listed" 'chore\(version\)|v2\.0\.0-rc\.1 \('

# Change lists are single lists (cog renders one section per tag in a range, separated by `- - -`,
# unless the pre-release tags are dropped), with each commit type once
for version in 2.0.0-alpha.2 2.0.0-beta.1 2.0.0-rc.1 2.0.0-rc.2 2.0.0; do
    expect_not "${version}" "single list" '(?m)^- - -$'
    if [[ "$(grep --count '^#### .*Fixed' <<< "$(notes "${version}")")" -gt 2 ]]; then
        echo "FAIL ${version}: commit types listed more than once per change list" >&2
        failures=$((failures + 1))
    fi
done

if [[ "${failures}" -gt 0 ]]; then
    echo "${failures} failure(s)" >&2
    exit 1
fi
echo "release-notes.sh: all checks passed"
