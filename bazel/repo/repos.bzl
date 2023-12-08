# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

load("@bazel_tools//tools/build_defs/repo:http.bzl", _http_archive = "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def http_archive(name, **kwargs):
    maybe(_http_archive, name = name, **kwargs)

def register_repositories():
    ## 3rdparty
    http_archive(
        name = "PoissonRecon",
        sha256 = "7fa4176d913a632afcde38308f4e41bac67ffdc6c6e88f73f434c8083e3d780e",
        # source-repo: https://github.com/mkazhdan/PoissonRecon, commit f1c71fe
        urls = ["https://gitlab-master.nvidia.com/api/v4/projects/61004/packages/generic/repo-PoissonRecon/f1c71fe/repo-PoissonRecon-f1c71fe.tar.gz"],
        build_file = "//3rdparty/PoissonRecon:PoissonRecon.BUILD",
        strip_prefix = "PoissonRecon",
        patch_args = ["-p1"],
        patches = ["//3rdparty/PoissonRecon:PoissonRecon.patch"],
    )

    http_archive(
        name = "numpyeigen",
        sha256 = "1b77ab89d90d407f1c2af035df4a5b708e1ddac91882d488e59d82dc0da8ef60",
        # source-repo: https://github.com/fwilliams/numpyeigen, commit e2ac640
        urls = ["https://gitlab-master.nvidia.com/api/v4/projects/61004/packages/generic/repo-numpyeigen/e2ac640/repo-numpyeigen-e2ac640.tar.gz"],
        build_file = "@ncore_repo//3rdparty/numpyeigen:numpyeigen.BUILD",
        strip_prefix = "numpyeigen",
    )

    http_archive(
        name = "numpyeigen_pybind11",
        sha256 = "5a13e0e17621622e61f4ce706167b80d4844b31dfd01aa7ab5c9d1ae99d540a9",
        # source-repo: https://github.com/fwilliams/pybind11, commit c230777e
        urls = ["https://gitlab-master.nvidia.com/api/v4/projects/61004/packages/generic/repo-pybind11/c230777e/repo-pybind11-c230777e.tar.gz"],
        build_file = "@ncore_repo//3rdparty/numpyeigen_pybind11:numpyeigen_pybind11.BUILD",
        strip_prefix = "pybind11",
    )

    http_archive(
        name = "semantic-segmentation",
        build_file = "//3rdparty/semantic-segmentation:semantic-segmentation.BUILD",
        sha256 = "0826aff938dc7efa0a17b54f3f775e674d065d5f9f39b454f296add959be3d5d",
        # source-repo: https://github.com/NVIDIA/semantic-segmentation.git, commit 7726b14
        urls = ["https://gitlab-master.nvidia.com/api/v4/projects/61004/packages/generic/repo-semantic-segmentation/7726b14/repo-semantic-segmentation-7726b14.tar.gz"],
        strip_prefix = "semantic-segmentation",
        patch_args = ["-p1"],
        patches = ["//3rdparty/semantic-segmentation:semantic-segmentation.patch"],
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

    http_archive(
        name = "waymo-open-dataset",
        sha256 = "cb67a97d99c28b6c801d0fbe9889bcaf7fb0e9f94b102624d38f9ae106db0891",
        # source-repo: https://github.com/waymo-research/waymo-open-dataset.git, tag v1.5.1
        urls = ["https://gitlab-master.nvidia.com/api/v4/projects/61004/packages/generic/repo-waymo-open-dataset/576f635/repo-waymo-open-dataset-576f635.tar.gz"],
        strip_prefix = "waymo-open-dataset/src",
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
