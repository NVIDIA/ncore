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

"""Tests of release_notes.py on scratch repositories (requires git and cog, see BUILD.bazel)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess

from pathlib import Path
from typing import Iterator

import pytest

from tools.release import release_notes


# The repository's cog configuration (commit types, changelog settings), next to this file in the source tree and in the
# Bazel runfiles
COG_CONFIG = Path(__file__).parent.parent.parent / ".cog.toml"

# Releases of the scratch history, in order
VERSIONS = ["1.0.0", "2.0.0-alpha.1", "2.0.0-alpha.2", "2.0.0-beta.1", "2.0.0-rc.1", "2.0.0-rc.2", "2.0.0"]

URL = "https://github.com/NVIDIA/ncore/compare"


class Repo:
    """Scratch git repository with deterministic commit dates."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.day = 0

    def git(self, *args: str, env: dict[str, str] | None = None) -> str:
        return subprocess.run(
            ["git", *args], cwd=self.path, check=True, capture_output=True, text=True, env={**os.environ, **(env or {})}
        ).stdout

    def _commit(self, message: str, hour: int) -> None:
        self.day += 1
        date = f"2026-01-{self.day:02d}T{hour}:00:00"
        # (verbatim: keep markdown headings and explicit line breaks of release notes)
        self.git(
            "commit",
            "--quiet",
            "--cleanup=verbatim",
            "--message",
            message if message.endswith("\n") else message + "\n",
            env={"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date},
        )

    def commit(self, message: str) -> None:
        with open(self.path / "changes.txt", "a") as f:
            f.write(message.splitlines()[0] + "\n")
        self.git("add", "--force", "changes.txt")
        self._commit(message, 10)

    def release(self, version: str, body: str = "", tag: bool = True) -> None:
        (self.path / ".bazelrc").write_text(f"build --embed_label={version}\n")
        self.git("add", "--force", ".bazelrc")
        self._commit(f"chore(version): v{version}" + (f"\n\n{body}" if body else ""), 12)
        if tag:
            self.git("tag", "--annotate", f"v{version}", "--message", f"v{version}")


def init_repo(path: Path) -> Repo:
    path.mkdir()
    repo = Repo(path)
    repo.git("init", "--quiet", "--initial-branch=main")
    for key, value in [
        ("user.name", "Dev One"),
        ("user.email", "one@example.com"),
        ("commit.gpgSign", "false"),
        ("tag.gpgSign", "false"),
        # (a user's global gitignore must not ignore files of the scratch repository)
        ("core.excludesFile", "/dev/null"),
    ]:
        repo.git("config", key, value)
    shutil.copyfile(COG_CONFIG, path / ".cog.toml")
    with open(path / ".cog.toml", "a") as f:
        f.write('\n[[changelog.authors]]\nsignature = "Dev One"\nusername = "dev-one"\n')
    repo.git("add", "--force", ".cog.toml")
    return repo


def make_history(path: Path) -> Repo:
    """A final release, pre-releases (alpha, beta, rc) of the next release, and the next final release."""
    repo = init_repo(path)
    repo.commit("feat: initial")
    repo.release("1.0.0")
    repo.commit("feat(data)!: breaking feature")
    repo.commit("perf(data): faster reads")
    repo.release("2.0.0-alpha.1")
    repo.commit("fix(data): alpha fix")
    repo.release("2.0.0-alpha.2")
    repo.commit("fix(data): beta fix")
    repo.release("2.0.0-beta.1")
    repo.commit("fix(data): rc fix")
    repo.release("2.0.0-rc.1")
    repo.commit("fix(data): second rc fix")
    repo.release("2.0.0-rc.2")
    repo.commit("docs: final docs")
    repo.release("2.0.0")
    return repo


@pytest.fixture(autouse=True)
def isolated_git(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ignore the user's and the system's git configuration."""
    monkeypatch.setenv("HOME", str(tmp_path_factory.mktemp("home")))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.delenv("GIT_CONFIG_GLOBAL", raising=False)


@pytest.fixture(scope="module")
def history(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Repo]:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("HOME", str(tmp_path_factory.mktemp("home")))
        mp.setenv("GIT_CONFIG_NOSYSTEM", "1")
        mp.delenv("GIT_CONFIG_GLOBAL", raising=False)
        yield make_history(tmp_path_factory.mktemp("history") / "repo")


def run(repo: Repo, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, *args: str) -> str:
    """Output of the release notes tool run in <repo>, with commit ids and the emojis of change list headings masked."""
    monkeypatch.chdir(repo.path)
    capsys.readouterr()
    assert release_notes.main(list(args)) == 0
    out = capsys.readouterr().out
    out = re.sub(r"\([0-9a-f]{7}\)", "(<id>)", out)
    return re.sub(r"^(#{4,}) \S+ ", r"\1 ", out, flags=re.MULTILINE)


ADDED = "#### Added\n- **BREAKING** (**data**) breaking feature (<id>) - @dev-one\n"
PERF = "#### Performance\n- (**data**) faster reads (<id>) - @dev-one\n"

# GitHub release text of each release of the scratch history (the release commits have no body, so the generated change
# list is published)
EXPECTED_NOTES = {
    # first release: everything since the start of the history
    "1.0.0": "### Changes\n\n#### Added\n- initial (<id>) - @dev-one\n\n",
    # first pre-release of a release: a single change list since the previous final release
    "2.0.0-alpha.1": f"""\
### Changes since v1.0.0

{ADDED}
{PERF}
**Full diff**: {URL}/v1.0.0...v2.0.0-alpha.1
""",
    # alpha / beta: the changes since the previous pre-release first, all changes collapsed
    "2.0.0-alpha.2": f"""\
### Changes since v2.0.0-alpha.1

#### Fixed
- (**data**) alpha fix (<id>) - @dev-one

<details>
<summary>All changes since v1.0.0</summary>

{ADDED}
#### Fixed
- (**data**) alpha fix (<id>) - @dev-one

{PERF}
</details>

**Changes since v2.0.0-alpha.1**: {URL}/v2.0.0-alpha.1...v2.0.0-alpha.2  
**Full diff**: {URL}/v1.0.0...v2.0.0-alpha.2
""",
    "2.0.0-beta.1": f"""\
### Changes since v2.0.0-alpha.2

#### Fixed
- (**data**) beta fix (<id>) - @dev-one

<details>
<summary>All changes since v1.0.0</summary>

{ADDED}
#### Fixed
- (**data**) beta fix (<id>) - @dev-one
- (**data**) alpha fix (<id>) - @dev-one

{PERF}
</details>

**Changes since v2.0.0-alpha.2**: {URL}/v2.0.0-alpha.2...v2.0.0-beta.1  
**Full diff**: {URL}/v1.0.0...v2.0.0-beta.1
""",
    # rc: all changes first, then the changes since the previous pre-release (headings demoted)
    "2.0.0-rc.1": f"""\
### Changes since v1.0.0

{ADDED}
#### Fixed
- (**data**) rc fix (<id>) - @dev-one
- (**data**) beta fix (<id>) - @dev-one
- (**data**) alpha fix (<id>) - @dev-one

{PERF}
### New since v2.0.0-beta.1

##### Fixed
- (**data**) rc fix (<id>) - @dev-one

**Changes since v2.0.0-beta.1**: {URL}/v2.0.0-beta.1...v2.0.0-rc.1  
**Full diff**: {URL}/v1.0.0...v2.0.0-rc.1
""",
    "2.0.0-rc.2": f"""\
### Changes since v1.0.0

{ADDED}
#### Fixed
- (**data**) second rc fix (<id>) - @dev-one
- (**data**) rc fix (<id>) - @dev-one
- (**data**) beta fix (<id>) - @dev-one
- (**data**) alpha fix (<id>) - @dev-one

{PERF}
### New since v2.0.0-rc.1

##### Fixed
- (**data**) second rc fix (<id>) - @dev-one

**Changes since v2.0.0-rc.1**: {URL}/v2.0.0-rc.1...v2.0.0-rc.2  
**Full diff**: {URL}/v1.0.0...v2.0.0-rc.2
""",
    # final release: a single change list since the previous final release, including the pre-releases
    "2.0.0": f"""\
### Changes since v1.0.0

{ADDED}
#### Fixed
- (**data**) second rc fix (<id>) - @dev-one
- (**data**) rc fix (<id>) - @dev-one
- (**data**) beta fix (<id>) - @dev-one
- (**data**) alpha fix (<id>) - @dev-one

{PERF}
#### Documentation
- final docs (<id>) - @dev-one

**Full diff**: {URL}/v1.0.0...v2.0.0
""",
}


@pytest.mark.parametrize("version", VERSIONS)
def test_publish_generated_notes(
    version: str, history: Repo, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert run(history, capsys, monkeypatch, "publish", version, f"v{version}") == EXPECTED_NOTES[version]


def test_changes(history: Repo, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    notes = EXPECTED_NOTES["2.0.0-rc.1"]
    expected = notes[: notes.index("\n**Changes since")]
    assert run(history, capsys, monkeypatch, "changes", "2.0.0-rc.1", "v2.0.0-rc.1") == expected


def test_publish_commit_body(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = init_repo(tmp_path / "repo")
    repo.commit("feat: initial")
    repo.release("1.0.0")
    repo.commit("fix: a fix")
    body = """\
### Highlights

A paragraph wrapped
  over three
lines.

- A list item wrapped
  over two lines.
- Another item

```
code is
kept
```

Explicit break  
kept.

Signed-off-by: Dev One <one@example.com>
Co-authored-by: Dev Two <two@example.com>
"""
    repo.release("1.0.1", body=body)
    assert (
        run(repo, capsys, monkeypatch, "publish", "1.0.1", "v1.0.1")
        == f"""\
### Highlights

A paragraph wrapped over three lines.

- A list item wrapped over two lines.
- Another item

```
code is
kept
```

Explicit break  
kept.

**Full diff**: {URL}/v1.0.0...v1.0.1
"""
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a\nb\n", "a b\n"),
        ("a\n\nb\n", "a\n\nb\n"),
        ("a\n   b\n", "a b\n"),
        # list items, headings, quotes, tables, and HTML start a new line
        ("a\n- b\n* c\n+ d\n1. e\n2) f\n", "a\n- b\n* c\n+ d\n1. e\n2) f\n"),
        ("a\n# b\n> c\n| d |\n<e>\n", "a\n# b\n> c\n| d |\n<e>\n"),
        # (but continue a list item)
        ("- a\n  b\n", "- a b\n"),
        # headings, tables, and HTML end a line
        ("# a\nb\n", "# a\nb\n"),
        ("| a |\nb\n", "| a |\nb\n"),
        ("<a>\nb\n", "<a>\nb\n"),
        # explicit line breaks
        ("a  \nb\n", "a  \nb\n"),
        ("a\\\nb\n", "a\\\nb\n"),
        # code blocks are kept as is
        ("~~~\na\nb\n~~~\nc\nd\n", "~~~\na\nb\n~~~\nc d\n"),
        ("```py\na\n\nb\n```\n", "```py\na\n\nb\n```\n"),
    ],
)
def test_unwrap(text: str, expected: str) -> None:
    assert release_notes.unwrap(text) == expected


def test_draft(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    repo = make_history(tmp_path / "repo")
    repo.commit("fix(data): patch fix")
    # (the cog bump post-bump hooks delete the tag of the release commit before drafting the notes)
    repo.release("2.0.1", tag=False)
    run(repo, capsys, monkeypatch, "draft", "2.0.1")
    message = re.sub(r"\([0-9a-f]{7}\)", "(<id>)", repo.git("log", "-1", "--format=%B"))
    message = re.sub(r"^(#{4,}) \S+ ", r"\1 ", message, flags=re.MULTILINE)
    # (headings are kept, git would strip them as comments by default)
    assert message == (
        "chore(version): v2.0.1\n\n### Changes since v2.0.0\n\n#### Fixed\n- (**data**) patch fix (<id>) - @dev-one\n\n"
    )
    assert repo.git("status", "--porcelain") == ""


def test_draft_refuses_existing_body(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = init_repo(tmp_path / "repo")
    repo.commit("feat: initial")
    repo.release("1.0.0", body="Hand-written notes", tag=False)
    monkeypatch.chdir(repo.path)
    assert release_notes.main(["draft", "1.0.0"]) == 1
    assert "already has a body" in capsys.readouterr().err
    assert repo.git("log", "-1", "--format=%B") == "chore(version): v1.0.0\n\nHand-written notes\n\n"


def test_refuses_shallow_clone(
    history: Repo, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # (a shallow clone has no previous releases, its notes would silently cover the whole fetched history)
    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "--quiet", "--depth=2", "--no-local", str(history.path), str(clone)],
        check=True,
        capture_output=True,
    )
    monkeypatch.chdir(clone)
    assert release_notes.main(["publish", "2.0.0", "HEAD"]) == 1
    assert capsys.readouterr().err == "error: shallow clone, the release notes need the full history and tags\n"


def test_reports_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    repo = init_repo(tmp_path / "repo")
    repo.commit("feat: initial")
    monkeypatch.chdir(repo.path)
    assert release_notes.main(["publish", "1.0.0", "v1.0.0"]) == 1
    assert capsys.readouterr().err.startswith("error: ")


def test_draft_refuses_other_commit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = init_repo(tmp_path / "repo")
    repo.commit("feat: initial")
    monkeypatch.chdir(repo.path)
    assert release_notes.main(["draft", "1.0.0"]) == 1
    assert "is not the release commit of v1.0.0" in capsys.readouterr().err
    assert repo.git("log", "-1", "--format=%B") == "feat: initial\n\n"
