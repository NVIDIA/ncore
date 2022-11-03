<!-- Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved. -->

# Documentation 

## Building and Viewing

DSAI's documentation is sphinx-based. A HTML version of the documentation can be build using

```
bazel build //docs:dsai
```

, which will be outputted into the output folder `bazel-bin/docs/dsai_html`.

The HTML version can also be directly build and opened in a web-browser by running the

```
bazel run //docs:view_dsai
```

target.
