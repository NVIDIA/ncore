# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

""" Defines the aspect for running mypy on Python targets. """

load("@rules_mypy//mypy:mypy.bzl", "mypy")

mypy_aspect = mypy(
    mypy_cli = "@@//bazel/typing:mypy",
    mypy_ini = "@@//bazel/typing:mypy.ini",
    suppression_tags = ["no-mypy"],
)
