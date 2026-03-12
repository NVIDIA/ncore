# How to Contribute

We'd love to receive your patches and contributions to NCore. These guidelines are intended to help you get your contribution accepted as quickly as possible.

## Development Setup

### Install System Packages

In addition to nvidia drivers / cuda runtime (>11.1), the following system packages should be installed to build all parts of the project:

```bash
sudo apt-get install pandoc
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

## Code Reviews

All submissions, including submissions by project members, require review. We use GitHub pull requests from forks for this purpose. Consult [GitHub Help](https://help.github.com/articles/about-pull-requests/) for more information on using pull requests.

### Pull Requests

- Keep PRs as draft until you're ready for review
- Keep PRs focused and concise; address a _single_ atomic concern per PR
- If applicable, reference relevant issues that the PR is closing
- Ensure the CI passes with no warnings or errors, and that required approvals are obtained
- Ensure library versions are updated appropriately for changing public-facing APIs
- Include full test coverage for new or updated functionality
- Avoid committing commented-out code

### Linear History

This repository enforces a **linear commit history** of commits with conventional commit messages.

PRs are merged via **rebase only** (no merge commits).

Before submitting your PR, ensure your branch is _rebased_ on the latest `main` and is free of merge commits.

## Releases & Versioning

NCore uses [Semantic Versioning](https://semver.org/) with versions determined automatically from [Conventional Commits](https://www.conventionalcommits.org/) via [cocogitto](https://docs.cocogitto.io/).

### Cocogitto Configuration

Cocogitto is configured via `.cog.toml` in the repository root. All `cog` commands must reference this config:

```bash
cog --config .cog.toml check                   # Validate commit messages
cog --config .cog.toml bump --auto --dry-run   # Preview next version
cog --config .cog.toml changelog               # Generate changelog
```

## Commit Signatures

We require that all commits are GPG-signed. This ensures commit authenticity and certifies your agreement with the Developer Certificate of Origin (DCO) below.

Any contribution which contains commits that are not signed will not be accepted.

### Setting Up GPG Signing

1. **Generate a GPG key** (if you don't have one):
   See [GitHub: Generating a new GPG key](https://docs.github.com/en/authentication/managing-commit-signature-verification/generating-a-new-gpg-key)

2. **Add your GPG key to your GitHub account**:
   See [GitHub: Adding a GPG key to your account](https://docs.github.com/en/authentication/managing-commit-signature-verification/adding-a-gpg-key-to-your-github-account)

3. **Configure Git to use your signing key**:

   ```bash
   # List your GPG keys to find your key ID
   gpg --list-secret-keys --keyid-format=long

   # Configure Git with your key ID
   git config --global user.signingkey <your-key-id>

   # Enable automatic signing for all commits (recommended)
   git config --global commit.gpgsign true
   ```

### Signing Commits

If you haven't enabled automatic signing, use the `-S` flag:

```bash
git commit -S -m "feat: add cool feature"
```

Your commits will show as "Verified" on GitHub once properly configured.

### Developer Certificate of Origin

By signing your commits, you certify your agreement with the following:

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
    with it, including my signature) is maintained indefinitely and may be
    redistributed consistent with this project or the open source license(s) involved.
```
