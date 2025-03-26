# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

load("@bazel_tools//tools/build_defs/repo:http.bzl", _http_archive = "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def http_archive(name, **kwargs):
    maybe(_http_archive, name = name, **kwargs)

def register_repositories():
    """
    Registers all the necessary repositories for the project when included in a WORKSPACE file.
    """

    ## 3rdparty
    http_archive(
        name = "rules_python",
        sha256 = "4f7e2aa1eb9aa722d96498f5ef514f426c1f55161c3c9ae628c857a7128ceb07",
        strip_prefix = "rules_python-1.0.0",
        url = "https://github.com/bazelbuild/rules_python/releases/download/1.0.0/rules_python-1.0.0.tar.gz",
    )

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
        name = "cgrindel_bazel_starlib",
        sha256 = "43e375213dabe0c3928e65412ea7ec16850db93285c8c6f8b0eaa41cacd0f882",
        urls = [
            "https://github.com/cgrindel/bazel-starlib/releases/download/v0.21.0/bazel-starlib.v0.21.0.tar.gz",
            ],
    )

    http_archive(
        name = "rules_oci",
        patch_args = ["-p1"],
        patches = [
            "@ncore_repo//:3rdparty/oci/oci.patch",
        ],
        sha256 = "1bd16e455278d523f01326e0c3964cd64d7840a7e99cdd6e2617e59f698f3504",
        strip_prefix = "rules_oci-2.2.0",
        url = "https://github.com/bazel-contrib/rules_oci/releases/download/v2.2.0/rules_oci-v2.2.0.tar.gz",
    )

    http_archive(
        name = "aspect_bazel_lib",
        sha256 = "6d758a8f646ecee7a3e294fbe4386daafbe0e5966723009c290d493f227c390b",
        strip_prefix = "bazel-lib-2.7.7",
        url = "https://github.com/aspect-build/bazel-lib/releases/download/v2.7.7/bazel-lib-v2.7.7.tar.gz",
    )

    http_archive(
        name = "bazel_features",
        sha256 = "06f02b97b6badb3227df2141a4b4622272cdcd2951526f40a888ab5f43897f14",
        strip_prefix = "bazel_features-1.9.0",
        url = "https://github.com/bazel-contrib/bazel_features/releases/download/v1.9.0/bazel_features-v1.9.0.tar.gz",
    )
