# DriveSim-AI

DISCLAIMER: THIS REPOSITORY IS NVIDIA INTERNAL/CONFIDENTIAL. DO NOT SHARE EXTERNALLY.
IF YOU PLAN TO USE THIS CODEBASE FOR YOUR RESEARCH, PLEASE CONTACT ZAN GOJCIC <zgojcic@nvidia.com>/OR LITANY <olitany@nvidia.com>.

NOTE: This codebase is under active development and the APIs may thus still change. If you build upon this repository, consider forking it to prevent such issues.

# Installation 

## Install git-lfs

```
sudo apt-get install git-lfs
git lfs install
```
[one-time operation]

## Clone repo with submodules

```
git clone --recursive https://gitlab-master.nvidia.com/zgojcic/drivesim-ai.git
```

## Install bazel

The repository is using `bazel` as the core build-system (see `.bazelversion` for the required version).

The correct `bazel` version can either be installed locally, or incorporated automatically with the [bazelisk](https://github.com/bazelbuild/bazelisk) wrapper 

with 

`go install github.com/bazelbuild/bazelisk@latest`

(make sure to also add `$(go env GOPATH)/bin` to your local `PATH` environment variable).

## Example of building / running a target with bazel

Build targets can be seamlessly build and executed using the bazel driver (either `bazel` or `bazelisk`) via, e.g.,

```
bazelisk run //scripts:convert_raw_data --
  --help
```

In this command, `bazelisk` is the bazel driver, `run` is the bazel command to run (other common alternatives are `build` / `test`), `//scripts:convert_raw_data` is the label of the target `convert_raw_data` living in the `//scripts` package (corresponding to the `<repo-root>/scripts` folder), and `-- --help` are arguments passed to the executed target (not the intermediate `--` separator).

## Create a virtual environment 

Install `apex` as 

```
cd dependencies/apex
pip install -v --disable-pip-version-check --no-cache-dir ./
cd ../..
```
