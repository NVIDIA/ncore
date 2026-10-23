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

"""Release notes of NCore releases.

The release notes are the markdown body of the release commit (`chore(version): vX.Y.Z`), so they are reviewed and
edited in the release PR.

Usage (`bazel run //tools/release:release_notes -- <args>`, or `python3 tools/release/release_notes.py <args>`):

    draft <version>
        Fill the empty body of the release commit at HEAD with the change list of the release (run by the cog
        post-bump hook, see .cog.toml).
    changes <version> <commit>
        Print the change list of the release at <commit>.
    publish <version> <commit>
        Print the GitHub release text of the release at <commit> (used by CI): the commit body without trailers and
        with wrapped lines joined, followed by links to the diffs.

Change lists cover the whole release: all changes since the previous final release (pre-release tags are ignored,
e.g. v20.0.0 and v20.0.0-rc.1 list everything since the last 19.x release). Pre-releases following an earlier
pre-release of the same release also list the changes since that one separately:

- alpha / beta: the changes since the previous pre-release, then all changes of the release (collapsed)
- rc: all changes of the release, then the changes since the previous pre-release

and link both diffs.

Requires git, and cocogitto for `draft` and `changes`, in a clone with full history and tags. cog is looked up in this
order: the `--cog` option, the `COG` environment variable (set by `bazel run //:cog`), the Bazel runfiles (pinned
version, see bazel/multitool.lock.json), and the `PATH`.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile

from pathlib import Path


# Change list template of cog (see release-notes.tera)
TEMPLATE = Path(__file__).absolute().parent / "release-notes.tera"

REPO_URL = "https://github.com/NVIDIA/ncore"

# Trailers of release commits, not part of the release notes
TRAILER = re.compile(r"^(Signed-off-by|Co-authored-by):")


class Error(Exception):
    """Error reported without a traceback."""


def git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
    if result.returncode != 0:
        raise Error(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def git_lines(*args: str) -> list[str]:
    """Output lines of a git command, no lines if it fails (e.g. for a range starting before the first commit)."""
    result = subprocess.run(["git", *args], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, encoding="utf-8")
    return result.stdout.splitlines() if result.returncode == 0 else []


def rev_parse(rev: str) -> str:
    """Commit id of <rev>."""
    return git("rev-parse", "--verify", f"{rev}^{{commit}}").strip()


def find_cog(option: str | None) -> str:
    if option:
        return option
    if os.environ.get("COG"):
        return os.environ["COG"]
    try:
        from python.runfiles import Runfiles

        runfiles = Runfiles.Create()
        if runfiles is not None:
            path = runfiles.Rlocation("multitool/tools/cog/cog")
            if path and os.path.isfile(path):
                return path
    except ImportError:
        pass
    path = shutil.which("cog")
    if path is None:
        raise Error("cog not found: use `bazel run //tools/release:release_notes`, or set COG or the PATH")
    return path


def previous_release(commit: str) -> str:
    """Latest final release tag reachable from the parent of the release commit <commit> ('' if none)."""
    lines = git_lines("describe", "--tags", "--abbrev=0", "--match", "v[0-9]*", "--exclude", "v*-*", f"{commit}^")
    return lines[0] if lines else ""


def previous_prereleases(commit: str) -> list[str]:
    """Pre-release tags of the release of <commit> reachable from its parent (since the previous final release).

    Newest first, by ancestry: the commits between them are walked from <commit> backwards.
    """
    base = previous_release(commit)
    tags: list[str] = []
    for c in git_lines("log", "--format=%H", f"{base}..{commit}^" if base else f"{commit}^"):
        tags += git_lines("tag", "--points-at", c, "--list", "v[0-9]*-*")
    return tags


class Notes:
    """Release notes of the repository in the working directory."""

    def __init__(self, version: str, cog: str | None = None) -> None:
        self.version = version
        self.tag = f"v{version}"
        self._cog_option = cog
        self._cog: str | None = None
        self.repo_root = Path(git("rev-parse", "--show-toplevel").strip())
        # (in a shallow clone, the previous releases are missing and the change lists would cover the whole history)
        if git("rev-parse", "--is-shallow-repository").strip() == "true":
            raise Error("shallow clone, the release notes need the full history and tags")

    @property
    def cog(self) -> str:
        # (looked up on first use: publishing written release notes does not need cog)
        if self._cog is None:
            self._cog = find_cog(self._cog_option)
        return self._cog

    def change_list(self, start: str, end: str, level: int = 0) -> str:
        """Change list (cog, release-notes.tera) of the commits in <start>..<end>.

        Headings are demoted by <level> extra levels (0: `####` per commit type).
        """
        # (commit ids, the pre-release tags of the range may be dropped below)
        end = rev_parse(end)
        start = rev_parse(start) if start else ""
        # (cog rejects a single commit as range, `..<end>` is everything up to <end>)
        commits = f"{start}..{end}"
        inner = [
            tag
            for c in git_lines("log", "--format=%H", f"{start}..{end}^" if start else f"{end}^")
            for tag in git_lines("tag", "--points-at", c, "--list", "v*-*")
        ]
        with tempfile.TemporaryDirectory() as scratch:
            cwd = self.repo_root
            if inner:
                # cog renders one section per tag in the range; drop pre-release tags in a throwaway clone so the range
                # gets a single change list
                cwd = Path(scratch) / "repo"
                git("clone", "--quiet", "--no-checkout", str(self.repo_root), str(cwd))
                git("fetch", "--quiet", str(self.repo_root), end, cwd=cwd)
                git("checkout", "--quiet", "--detach", end, cwd=cwd)
                prereleases = git("tag", "--list", "v*-*", cwd=cwd).split()
                if prereleases:
                    git("tag", "--delete", *prereleases, cwd=cwd)
            # (the configuration of the repository: the clone has no untracked or modified configuration)
            config = str(self.repo_root / ".cog.toml")
            out = subprocess.run(
                [self.cog, "--config", config, "changelog", "--template", str(TEMPLATE), commits],
                cwd=cwd,
                check=True,
                stdout=subprocess.PIPE,
                encoding="utf-8",
            ).stdout

        lines = out.split("\n")
        if out.endswith("\n"):
            lines.pop()
        # strip leading and trailing empty lines
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()
        if level > 0:
            lines = [re.sub(r"^(#+)", r"\1" + "#" * level, line) for line in lines]
        return "".join(line + "\n" for line in lines)

    def changes(self, commit: str) -> str:
        """Change list of the release at <commit> (see the module docstring)."""
        target = rev_parse(commit)
        base = previous_release(target)
        prereleases = previous_prereleases(target)
        previous = prereleases[0] if prereleases else ""
        since_base = f" since {base}" if base else ""

        # final releases, and the first pre-release of a release: a single change list
        if "-" not in self.version or not previous:
            return f"### Changes{since_base}\n\n" + self.change_list(base, target)

        kind = self.version.split("-", 1)[1].split(".", 1)[0]  # alpha / beta / rc
        if kind == "rc":
            return (
                f"### Changes{since_base}\n\n"
                + self.change_list(base, target)
                + f"\n### New since {previous}\n\n"
                + self.change_list(previous, target, 1)
            )
        return (
            f"### Changes since {previous}\n\n"
            + self.change_list(previous, target)
            + f"\n<details>\n<summary>All changes{since_base}</summary>\n\n"
            + self.change_list(base, target)
            + "\n</details>\n"
        )

    def draft(self) -> int:
        """Fill the empty body of the release commit at HEAD with the change list of the release."""
        if git("log", "-1", "--format=%b", "HEAD").strip():
            print("error: the release commit at HEAD already has a body, not overwriting it", file=sys.stderr)
            return 1
        subject = git("log", "-1", "--format=%s", "HEAD").rstrip("\n")
        if subject != f"chore(version): {self.tag}":
            print(f"error: HEAD is not the release commit of {self.tag} ('{subject}')", file=sys.stderr)
            return 1
        message = f"{subject}\n\n{self.changes('HEAD').rstrip(chr(10))}\n"
        # verbatim: keep the `###` headings, which git would otherwise strip as comments
        subprocess.run(
            ["git", "commit", "--quiet", "--amend", "--cleanup=verbatim", "--file", "-"],
            input=message,
            check=True,
            encoding="utf-8",
        )
        print("Release commit body drafted. Add highlights and edit it with 'git commit --amend --cleanup=scissors'.")
        return 0

    def publish(self, commit: str) -> str:
        """GitHub release text of the release at <commit>."""
        target = rev_parse(commit)
        body = git("log", "-1", "--format=%b", target)
        body = "".join(line for line in body.splitlines(keepends=True) if not TRAILER.match(line)).rstrip("\n")
        if not body.strip():
            # No release notes written: fall back to the generated change list
            body = self.changes(target).rstrip("\n")
        text = unwrap(f"{body}\n\n")
        base = previous_release(target)
        prereleases = previous_prereleases(target)
        previous = prereleases[0] if prereleases else ""
        if "-" in self.version and previous:
            text += f"**Changes since {previous}**: {compare(previous, self.tag)}  \n"
        if base:
            text += f"**Full diff**: {compare(base, self.tag)}\n"
        return text


def compare(start: str, end: str) -> str:
    return f"{REPO_URL}/compare/{start}...{end}"


FENCE = re.compile(r"^[ \t]*(```|~~~)")
BLANK = re.compile(r"^[ \t]*$")
STANDALONE = re.compile(r"^[ \t]*(#|>|\||<|[-*+][ \t]|[0-9]+[.)][ \t])")
HARD_END = re.compile(r"(  |\\)$")
HARD_START = re.compile(r"^[ \t]*(#|\||<)")


def unwrap(text: str) -> str:
    """Join wrapped lines of a markdown document into single lines.

    GitHub renders single line breaks in release text as hard breaks; joined lines show as regular paragraphs and list
    items. Lines are not joined across blank lines, list items, headings, quotes, tables, code blocks, HTML, or
    explicit line breaks (two trailing spaces or a trailing backslash).
    """
    lines = text.split("\n")
    if text.endswith("\n"):
        lines.pop()
    out: list[str] = []
    buf: str | None = None
    fence = hard = False

    def flush() -> None:
        nonlocal buf
        if buf is not None:
            out.append(buf)
        buf = None

    for line in lines:
        if FENCE.match(line):
            flush()
            out.append(line)
            fence = not fence
            continue
        if fence:
            out.append(line)
            continue
        if BLANK.match(line):
            flush()
            out.append(line)
            continue
        if buf is not None and not STANDALONE.match(line) and not hard:
            buf += " " + line.lstrip(" \t")
        else:
            flush()
            buf = line
        hard = bool(HARD_END.search(line) or HARD_START.match(line))
    flush()
    return "".join(line + "\n" for line in out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0] if __doc__ else None,
        epilog="See the module docstring of tools/release/release_notes.py for details.",
    )
    parser.add_argument("--cog", help="cocogitto executable")
    commands = parser.add_subparsers(dest="command", required=True)
    draft = commands.add_parser("draft", help="fill the empty body of the release commit at HEAD")
    draft.add_argument("version")
    for name, summary in [
        ("changes", "print the change list of the release at <commit>"),
        ("publish", "print the GitHub release text of the release at <commit>"),
    ]:
        command = commands.add_parser(name, help=summary)
        command.add_argument("version")
        command.add_argument("commit")
    args = parser.parse_args(argv)

    try:
        notes = Notes(args.version, args.cog)
        if args.command == "draft":
            return notes.draft()
        sys.stdout.write(notes.changes(args.commit) if args.command == "changes" else notes.publish(args.commit))
    except (Error, subprocess.CalledProcessError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    # under `bazel run`, operate on the workspace
    if "BUILD_WORKSPACE_DIRECTORY" in os.environ:
        os.chdir(os.environ["BUILD_WORKSPACE_DIRECTORY"])
    # (change lists contain emojis and author names, independent of the locale)
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
