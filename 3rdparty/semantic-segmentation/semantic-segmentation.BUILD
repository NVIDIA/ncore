# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("@pip_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "pylib",
    srcs = glob(
        ["**/*.py"],
        exclude = ["train.py"],
    ),
)

py_binary(
    name = "train",
    srcs = [
        "train.py",
    ],
    deps = [
        requirement("torch"),
        requirement("torchvision"),
        requirement("apex"),
        requirement("numpy"),
        requirement("runx"),
        requirement("opencv-python"),
        requirement("scikit-image"),
        requirement("tqdm"),
        ":pylib",
    ],
)
