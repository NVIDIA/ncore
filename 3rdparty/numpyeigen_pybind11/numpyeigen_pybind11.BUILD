# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "pybind11",
    hdrs = glob(["include/**"]),
    includes = ["include"],
    deps = [
        "@python3//:libpython",
        "@python3//:python_headers",
    ],
)
