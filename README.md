<!-- Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved. -->

# NCORE

NOTE: This codebase is under active development and the APIs may thus still change. If you build upon this repository, consider forking it to prevent such issues.

# Installation

## Install system-packages

In addition to nvidia drivers / cuda runtime (>11.1), the following system packages should be installed to build the project:

```
sudo apt-get install gcc g++ clang libgl1 xz-utils jupyter-nbconvert
```

Alternatively, builds can be executed within the `gitlab-master.nvidia.com:5005/nrs/ncore_repos/ncore:dev` docker image, which has these packages pre-installed.

Additionally, the following packages should be installed as dependencies for scripts:

```
sudo apt-get install qt6-base-dev libxcb-cursor0
```

## Install / setup git-lfs

Large files within the repository are tracked via the `git-lfs` extension. To install the package and register git-lfs, execute

```
sudo apt-get install git-lfs
git lfs install
```

before cloning the repository.

[one-time operation]

## Setup gitlab personal access token / docker credentials

Create a gitlab-master personal access token with `api` scope at [link](https://gitlab-master.nvidia.com/-/profile/personal_access_tokens) and register the new token token in `~/.netrc` file as

```
machine gitlab-master.nvidia.com
login oauth2
password <GITLAB_TOKEN>
```

by replacing `<TOKEN>` with the created token string.

Additionally, the local docker daemon needs to be authenticated against gitlab's image registry via

```
docker login gitlab-master.nvidia.com:5005 -u oauth2
```

using the same `<TOKEN>` to access development and base images.

[one-time operation]

## Cloning the repo

```
git clone https://gitlab-master.nvidia.com/Toronto_DL_Lab/nrs/ncore.git
```

## Install bazel

The repository is using `bazel` as the core build-system (see `.bazelversion` for the required version).

The correct `bazel` version is most easily invoked using the official `bazelisk` wrapper.

`bazelisk` can be installed with one of the methods listed at [bazelisk-installation](https://github.com/bazelbuild/bazelisk#installation), or simply by running:

```
sudo wget -O /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
sudo chmod +x /usr/local/bin/bazel
```

# Execution

## Format all code files

Execute

```
bazel run //:format
```

to format all source files. Use `bazel run //:format.check` to only check for
code-formatting violations.

Note a special case for Bazel: the commands above are used both for traditional
formatting and for linter warnings. Bazel attempts to fix some of the linter
warnings automatically as part of the `//:format` target, but others will be
left untouched and need to be manually corrected by the user.

## Static Code Analysis

The repository makes use of `mypy` for static-code validation of the important components.
These are executed as part of aspects associated with all python targets.
In order to speed up _local_ execution of `mypy`, consider making use of a _persistent_ user-cache available to `mypy` (as due to the way bazel sandboxes are setup, `mypy` is not able to access an external cache folder for faster analysis).

To enable separate local mypy caching, we are using a patched version which enables using local cache folders, which can be enabled by setting

```
# Make use of local mypy cache
build --sandbox_writable_path=<ABSOLUTE-PATH-TO>/.mypy_cache
build --action_env=MYPY_CACHE_DIR=<ABSOLUTE-PATH-TO>/.mypy_cache
```

in `.bazelrc.user`.

Without these options there will be no caching of intermediate incremental mypy results (bazel caching of final test states is not
affected by this and still active before).
