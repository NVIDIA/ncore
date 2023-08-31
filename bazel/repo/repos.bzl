# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

load("@bazel_tools//tools/build_defs/repo:http.bzl", _http_archive = "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", _git_repository = "git_repository", _new_git_repository = "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def http_archive(name, **kwargs):
    maybe(_http_archive, name = name, **kwargs)

def new_git_repository(name, **kwargs):
    maybe(_new_git_repository, name = name, **kwargs)

def git_repository(name, **kwargs):
    maybe(_git_repository, name = name, **kwargs)

def register_repositories():
    ## 3rdparty
    new_git_repository(
        name = "PoissonRecon",
        build_file = "//3rdparty/PoissonRecon:PoissonRecon.BUILD",
        commit = "f1c71fe491828d00d34f95de9c264d4bf3481d61",  # v13.74
        patch_args = ["-p1"],
        patches = ["//3rdparty/PoissonRecon:PoissonRecon.patch"],
        remote = "https://github.com/mkazhdan/PoissonRecon",
        shallow_since = "1650856848 -0400",
    )

    new_git_repository(
        name = "numpyeigen",
        build_file = "@ncore_repo//3rdparty/numpyeigen:numpyeigen.BUILD",
        commit = "e2ac64034fde35cb70da39aaefbf5331df6015f7",
        remote = "https://github.com/fwilliams/numpyeigen",
        shallow_since = "1673738105 -0500",
    )

    new_git_repository(
        name = "numpyeigen_pybind11",
        build_file = "@ncore_repo//3rdparty/numpyeigen_pybind11:numpyeigen_pybind11.BUILD",
        commit = "c230777e92be50e509f955e025d1f56f42e847fa",
        remote = "https://github.com/fwilliams/pybind11",
        shallow_since = "1688567160 -0400",
    )

    new_git_repository(
        name = "semantic-segmentation",
        build_file = "//3rdparty/semantic-segmentation:semantic-segmentation.BUILD",
        commit = "7726b144c2cc0b8e09c67eabb78f027efdf3f0fa",
        patch_args = ["-p1"],
        patches = ["//3rdparty/semantic-segmentation:semantic-segmentation.patch"],
        remote = "https://github.com/NVIDIA/semantic-segmentation.git",
        shallow_since = "1614912803 -0800",
    )

    http_archive(
        name = "semantic-segmentation-models",
        sha256 = "6ff6f5f1d80ad5742fad3f81651e1053e37a50d9bf79c561dd07a6ec5f88f5e2",
        urls = ["https://gitlab-master.nvidia.com/api/v4/projects/61004/packages/generic/semantic_segmentation_pretrained_models/0.1/semantic_segmentation_pretrained_models.tar.xz"],
    )

    http_archive(
        name = "instance-segmentation-models",
        sha256 = "b2db69e0c6e409ec137d953217f8481def9e7f3af84255f705e8dd963adf01a5",
        urls = ["https://gitlab-master.nvidia.com/api/v4/projects/61004/packages/generic/instance_segmentation_pretrained_models/0.1/instance_segmentation_pretrained_models.tar.xz"],
    )

    http_archive(
        name = "test-data-v3-shards",
        sha256 = "fce25e3cc81a019e46855c492951e34dd906330d17daad256e6d0a3a5551c425",
        urls = ["https://gitlab-master.nvidia.com/api/v4/projects/61004/packages/generic/test-data-v3-shards/0.5/test-data-v3-shards.tar.gz"],
    )

    http_archive(
        name = "eigen",
        build_file_content =
            """
# TODO(janickm): Replace this with a better version, like from TensorFlow.
# See https://github.com/tensorflow/tensorflow/tree/master/third_party/eigen3
cc_library(
    name = 'eigen',
    includes = ['.'],
    hdrs = glob(['Eigen/**']),
    visibility = ['//visibility:public'],
)
    """,
        sha256 = "685adf14bd8e9c015b78097c1dc22f2f01343756f196acdc76a678e1ae352e11",
        strip_prefix = "eigen-3.3.7",
        urls = [
            "https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.bz2",
        ],
    )

    git_repository(
        name = "waymo-open-dataset",
        commit = "576f63526281cf403be2b6720a0d3acb4d18f41d",  # tag v1.5.1
        shallow_since = "1680923402 -0700",
        remote = "https://github.com/waymo-research/waymo-open-dataset.git",
        strip_prefix = "src",
        repo_mapping = {"@wod_deps": "@pip_deps"},
    )

    http_archive(
        name = "rules_license",
        sha256 = "6157e1e68378532d0241ecd15d3c45f6e5cfd98fc10846045509fb2a7cc9e381",
        urls = [
            "https://github.com/bazelbuild/rules_license/releases/download/0.0.4/rules_license-0.0.4.tar.gz",
            "https://mirror.bazel.build/github.com/bazelbuild/rules_license/releases/download/0.0.4/rules_license-0.0.4.tar.gz",
        ],
    )
