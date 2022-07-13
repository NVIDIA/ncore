# DriveSim-AI

DISCLAIMER: THIS REPOSITORY IS NVIDIA INTERNAL/CONFIDENTIAL. DO NOT SHARE EXTERNALLY.
IF YOU PLAN TO USE THIS CODEBASE FOR YOUR RESEARCH, PLEASE CONTACT ZAN GOJCIC <zgojcic@nvidia.com> / JANICK MARTINEZ ESTURO <janickm@nvidia.com> / OR LITANY <olitany@nvidia.com>.

NOTE: This codebase is under active development and the APIs may thus still change. If you build upon this repository, consider forking it to prevent such issues.

# Installation 

## Install system-packages

In addition to nvidia drivers / cuda runtime (>11.1), the following system packages should be installed to build the project:

```
sudo apt-get install gcc g++ clang libgl1 xz-utils
```

Alternatively, builds can be executed within the `gitlab-master.nvidia.com:5005/zgojcic/drivesim-ai:dev` docker image, which has these packages pre-installed.

## Install / setup git-lfs

Large files within the repository are tracked via the `git-lfs` extension. To install the package and register git-lfs, execute

```
sudo apt-get install git-lfs
git lfs install
```

before cloning the repository.

[one-time operation]

## Setup gitlab personal access token

Create a gitlab-master personal access token with `api` scope at [link](https://gitlab-master.nvidia.com/-/profile/personal_access_tokens) and register the new toekn token in `~/.netrc` file as

```
machine gitlab-master.nvidia.com
login oauth2
password <TOKEN>
```

by replacing `<TOKEN>` with the created token string.

[one-time operation]

## Cloning the repo

```
git clone https://gitlab-master.nvidia.com/zgojcic/drivesim-ai.git
```

## Install bazel

The repository is using `bazel` as the core build-system (see `.bazelversion` for the required version).

The correct `bazel` version can either be installed locally, or incorporated automatically with the [bazelisk](https://github.com/bazelbuild/bazelisk) wrapper 

with 

`go install github.com/bazelbuild/bazelisk@latest`

(make sure to also add `$(go env GOPATH)/bin` to your local `PATH` environment variable).

# Execution

## Format all bazel files

Execute

```
bazel run //:update_all
```

to format all bazel source files (`//:bzlformat_missing_pkgs_fix` can be used to register new files)

## Example of building / running a target with bazel

Build targets can be seamlessly build and executed using the bazel driver (either `bazel` or `bazelisk`) via, e.g.,

```
bazel run //scripts:convert_raw_data --
  --help
```

In this command, `run` is the bazel command to run (other common alternatives are `build` / `test`), `//scripts:convert_raw_data` is the label of the target `convert_raw_data` living in the `//scripts` package (corresponding to the `<repo-root>/scripts` folder), and `-- --help` are arguments passed to the executed target (not the intermediate `--` separator).
