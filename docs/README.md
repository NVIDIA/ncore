<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Documentation

## Building and Viewing

NCore's documentation is sphinx-based. A HTML version of the documentation can be build using

```
bazel build //docs:ncore
```

The build results will be available in the output folder `bazel-bin/docs/ncore_html`.

The footer of every page names the version the documentation describes: the wheel version (`--embed_label` in
`.bazelrc`) and the git commit it is built from (`tools/release/workspace_status.sh`). The published documentation is
built from `main`: commits after a release commit are marked (e.g. `20.0.0rc1+ (abcdef123, 3 commits after the
release)`), as they describe changes towards the next release.

The HTML version can also be directly build and opened in a web-browser by running the

```
bazel run //docs:view_ncore
```

target.
