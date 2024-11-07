# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

load("@bazel_tools//tools/build_defs/repo:http.bzl", _http_archive = "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def http_archive(name, **kwargs):
    maybe(_http_archive, name = name, **kwargs)

def register_repositories():
    ## 3rdparty
    http_archive(
        name = "poisson_recon",
        sha256 = "7fa4176d913a632afcde38308f4e41bac67ffdc6c6e88f73f434c8083e3d780e",
        # source-repo: https://github.com/mkazhdan/PoissonRecon, commit f1c71fe
        urls = ["https://gitlab-master.nvidia.com/api/v4/projects/61004/packages/generic/repo-PoissonRecon/f1c71fe/repo-PoissonRecon-f1c71fe.tar.gz"],
        build_file = "//3rdparty/poisson_recon:poisson_recon.BUILD",
        strip_prefix = "PoissonRecon",
        patch_args = ["-p1"],
        patches = ["//3rdparty/poisson_recon:poisson_recon.patch"],
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
        sha256 = "cdb766ba178548f9307b1458bb408f4f92ba1557dd34bb30dac2a303749ab783",
        urls = ["https://gitlab-master.nvidia.com/api/v4/projects/61004/packages/generic/test-data-v3-shards/0.7/test-data-v3-shards.tar.gz"],
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

    http_archive(
        name = "cgrindel_bazel_starlib",
        sha256 = "c95de004f346cbcb51ba1185e8861227cd9ab248b53046f662aeda1095601bc9",
        strip_prefix = "bazel-starlib-0.7.1",
        urls = [
            "http://github.com/cgrindel/bazel-starlib/archive/v0.7.1.tar.gz",
        ],
    )
