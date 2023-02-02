# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

load("@bazel_tools//tools/build_defs/repo:http.bzl", _http_archive = "http_archive",)
load("@bazel_tools//tools/build_defs/repo:git.bzl",  _new_git_repository = "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def http_archive(name, **kwargs):
    maybe(_http_archive, name = name, **kwargs)

def new_git_repository(name, **kwargs):
    maybe(_new_git_repository, name = name, **kwargs)

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
        build_file = "@dsai-repo//3rdparty/numpyeigen:numpyeigen.BUILD",
        commit = "4916d926aa2b939bd8f625c7537563a1575dafe9",
        remote = "https://github.com/fwilliams/numpyeigen",
        shallow_since = "1643644288 -0500",
    )

    new_git_repository(
        name = "numpyeigen_pybind11",
        build_file = "@dsai-repo//3rdparty/numpyeigen_pybind11:numpyeigen_pybind11.BUILD",
        commit = "d8c0a26b06b4d6901f6af4b1cbdc975bb160221b",
        remote = "https://github.com/fwilliams/pybind11.git",
        shallow_since = "1656512085 -0400",
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
        sha256 = "c881f52d0e8319bbb2fadc3433b5b0be41874741559051fc7614dd9720dacbe9",
        urls = ["https://gitlab-master.nvidia.com/api/v4/projects/61004/packages/generic/test-data-v3-shards/0.4/test-data-v3-shards.tar.gz"],
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
