# How to Contribute

We'd love to receive your patches and contributions to NCore. Please keep your PRs as draft until you're ready for review.

## Code Reviews

All submissions, including submissions by project members, require review. We use GitHub pull requests for this purpose. Consult [GitHub Help](https://help.github.com/articles/about-pull-requests/) for more information on using pull requests.

## Development Setup

### Install System Packages

In addition to nvidia drivers / cuda runtime (>11.1), the following system packages should be installed to build all parts of the project:

```bash
sudo apt-get install pandoc
```

Additionally, the following packages should be installed as dependencies for tools:

```bash
sudo apt-get install qt6-base-dev libxcb-cursor0
```

### Setup GitHub Personal Access Token / Docker Credentials

Create a github personal access token with `read:packages` scope at [create-new-token](https://github.com/settings/tokens/new) and register the new token in `~/.netrc` file as

```netrc
machine maven.pkg.github.com
  login <GITHUB_USER>
  password <GITHUB_TOKEN>
```

by replacing `<GITHUB_USER>` with the user-name and `<GITHUB_TOKEN>` with the created token string.

[one-time operation]

### Cloning the Repository

```bash
git clone https://github.com/NVIDIA/ncore.git
```

### Install Bazel

The repository uses `bazel` as the core build-system (see `.bazelversion` for the required version).

The correct `bazel` version is most easily invoked using the official `bazelisk` wrapper.

`bazelisk` can be installed with one of the methods listed at [bazelisk-installation](https://github.com/bazelbuild/bazelisk#installation), or simply by running:

```bash
sudo wget -O /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
sudo chmod +x /usr/local/bin/bazel
```

### Build and Test

```bash
# Build all targets
bazel build //...

# Run tests
bazel test //...
```

## Coding Guidelines

### Formatting

Format all code before submitting:

```bash
# Format all files
bazel run //:format

# Check for violations (used in CI)
bazel run //:format.check
```

This project uses:

- **Python**: Ruff (120 character line length)
- **Bazel files**: Buildifier

Note a special case for Bazel: the commands above are used both for traditional formatting and for linter warnings. Bazel attempts to fix some of the linter warnings automatically as part of the `//:format` target, but others will be left untouched and need to be manually corrected by the user.

### Static Type Checking

All Python code is checked with `mypy`, which runs automatically as a Bazel aspect during builds.

#### Speeding Up Local mypy Execution

The repository makes use of `mypy` for static-code validation of the important components. These are executed as part of aspects associated with all python targets. In order to speed up _local_ execution of `mypy`, consider making use of a _persistent_ user-cache available to `mypy` (as due to the way bazel sandboxes are setup, `mypy` is not able to access an external cache folder for faster analysis).

To enable separate local mypy caching, we are using a patched version which enables using local cache folders, which can be enabled by setting

```bazel
# Make use of local mypy cache
build --sandbox_writable_path=<ABSOLUTE-PATH-TO>/.mypy_cache
build --action_env=MYPY_CACHE_DIR=<ABSOLUTE-PATH-TO>/.mypy_cache
```

in `.bazelrc.user`.

Without these options there will be no caching of intermediate incremental mypy results (bazel caching of final test states is not affected by this and still active before).

### License Headers

All source files must include SPDX license headers:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

### Commit Messages

This project follows the [Conventional Commits](https://www.conventionalcommits.org/) specification. All commits must follow this format:

```text
<type>[optional (scope)]: <description>

[optional body]

[optional footer(s)]
```

With optional [breaking-change indicators](https://www.conventionalcommits.org/en/v1.0.0/#commit-message-with--to-draw-attention-to-breaking-change).

**Common types:**

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code (formatting, etc.)
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `build`: Changes that affect the build system or external dependencies
- `ci`: Changes to CI configuration files and scripts
- `chore`: Other changes that don't modify src or test files

Commit messages are validated in CI using [cocogitto](https://github.com/cocogitto/cocogitto). PRs with non-compliant commits or merge commits will fail the conventional commits check.

### Branch Naming

Branches must follow the naming pattern:

```text
<type>/<username>/<topic>
```

Where:

- `<type>` is a conventional commit type (e.g., `feat`, `fix`, `docs`, `refactor`, `dev`)
- `<username>` is your GitHub username or identifier
- `<topic>` is a short, descriptive name for the change (use hyphens for spaces)

### Pull Requests

- Keep PRs focused and concise; address a _single_ atomic concern per PR
- Avoid committing commented-out code
- Ensure the CI passes with no warnings or errors
- Ensure library versions are updated appropriately for changing public-facing APIs
- Include full test coverage for new or updated functionality

### Linear History

This repository enforces a **linear commit history** of commits with conventional commit messages.

PRs are merged via **rebase only** (no merge commits).

Before submitting your PR, ensure your branch is *rebased* on the latest `main` and is free of merge commits.

## Signing Your Work

- We require that all contributors "sign-off" on their commits. This certifies
  that the contribution is your original work, or you have rights to submit it
  under the same license, or a compatible license.

  - Any contribution which contains commits that are not Signed-Off will not be
    accepted.

- To sign off on a commit you simply use the `--signoff` (or `-s`) option when
  committing your changes:

  ```bash
  git commit -s -m "Add cool feature."
  ```

  This will append the following to your commit message:

  ```bash
  Signed-off-by: Your Name <your@email.com>
  ```

- Full text of the DCO:

  ```text
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this
    license document, but changing it is not allowed.
  ```

  ```text
    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have
        the right to submit it under the open source license indicated in the file; or

    (b) The contribution is based upon previous work that, to the best of my knowledge,
        is covered under an appropriate open source license and I have the right under
        that license to submit that work with modifications, whether created in whole
        or in part by me, under the same open source license (unless I am permitted
        to submit under a different license), as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who
        certified (a), (b) or (c) and I have not modified it.

    (d) I understand and agree that this project and the contribution are public and
        that a record of the contribution (including all personal information I submit
        with it, including my sign-off) is maintained indefinitely and may be
        redistributed consistent with this project or the open source license(s) involved.
  ```
