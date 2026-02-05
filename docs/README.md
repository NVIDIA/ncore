<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION
SPDX-License-Identifier: Apache-2.0
-->

# Documentation

## Building and Viewing

NCore's documentation is sphinx-based. A HTML version of the documentation can be build using

```
bazel build //docs:ncore
```

, which will be outputted into the output folder `bazel-bin/docs/ncore_html`.

The HTML version can also be directly build and opened in a web-browser by running the

```
bazel run //docs:view_ncore
```

target.
