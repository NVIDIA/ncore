# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "pybind11",
    hdrs = glob(["include/**"]),
    includes = ["include"],
    deps = [
        "@python_3_10//:libpython",
        "@python_3_10//:python_headers",
    ],
)

cc_library(
    name = "pybind11_3_8",
    hdrs = glob(["include/**"]),
    includes = ["include"],
    deps = [
        "@python_3_8//:libpython",
        "@python_3_8//:python_headers",
    ],
)
