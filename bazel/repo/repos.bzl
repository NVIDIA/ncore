# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

load("@bazel_tools//tools/build_defs/repo:http.bzl", _http_archive = "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def http_archive(name, **kwargs):
    maybe(_http_archive, name = name, **kwargs)

def register_repositories():
    """
    Registers all the necessary repositories for the project.
    """

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
        name = "test-data-v3-shards",
        sha256 = "cdb766ba178548f9307b1458bb408f4f92ba1557dd34bb30dac2a303749ab783",
        urls = ["https://gitlab-master.nvidia.com/api/v4/projects/61004/packages/generic/test-data-v3-shards/0.7/test-data-v3-shards.tar.gz"],
    )

    http_archive(
        name = "io_bazel_rules_docker",
        sha256 = "b1e80761a8a8243d03ebca8845e9cc1ba6c82ce7c5179ce2b295cd36f7e394bf",
        urls = ["https://github.com/bazelbuild/rules_docker/releases/download/v0.25.0/rules_docker-v0.25.0.tar.gz"],
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
