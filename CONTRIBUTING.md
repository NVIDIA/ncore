# How to Contribute

We'd love to receive your patches and contributions to  `NCore`. Please keep your PRs as draft until  ou're ready for review.

## Code Reviews

All submissions, including submissions by project members, require review. We use GitHub pull requests for this purpose. Consult [GitHubHelp](https://help.github.com/articles/bout-pull-requests/) for more information on using pull requests.

## Development Setup

See [README.md](README.md) for detailed installation instructions. Quick reference:

```bash
# Install Bazelisk
sudo wget -O /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
sudo chmod +x /usr/local/bin/bazel

# Build
bazel build ...

# Test
bazel test ...
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

### Type Checking

All Python code is checked with `mypy`, which runs automatically as a Bazel aspect during builds.

### License Headers

All source files must include SPDX license headers:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION
# SPDX-License-Identifier: Apache-2.0
```

### Pull Requests

- Keep PRs focused and concise; address a _single_ atomic concern per PR
- Avoid committing commented-out code
- Ensure the build passes with no warnings or errors
- Ensure library versions are updated appropriately for changing public-facing APIs
- Include full test coverage for new or updated functionality

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
