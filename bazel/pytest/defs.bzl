# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

""" Wrap py_test with a common pytest wrapper """

load("@rules_python//python:defs.bzl", "py_test")
load("@pip_deps//:requirements.bzl", "requirement")
load("@pip_deps_3_8//:requirements.bzl", requirement_3_8 = "requirement")
load("@python//3.10:defs.bzl", py_test_3_8 = "py_test")

def pytest_test(name, srcs, deps = [], args = [], **kwargs):
    """
        Call pytest using a common wrapper script
    """
    py_test(
        name = name,
        srcs = [
            "//bazel/pytest:pytest_wrapper.py",
        ] + srcs,
        main = "//bazel/pytest:pytest_wrapper.py",
        args = [
            "--capture=no",
        ] + args + ["$(location :%s)" % x for x in srcs],
        python_version = "PY3",
        srcs_version = "PY3",
        deps = deps + [
            requirement("pytest"),
        ],
        **kwargs
    )

def pytest_test_3_8(name, srcs, deps = [], args = [], **kwargs):
    """
        Call pytest using a common wrapper script (python3.8)
    """
    py_test_3_8(
        name = name,
        srcs = [
            "@nre_repo//bazel/pytest:pytest_wrapper.py",
        ] + srcs,
        main = "@nre_repo//bazel/pytest:pytest_wrapper.py",
        args = [
            "--capture=no",
        ] + args + ["$(location :%s)" % x for x in srcs],
        srcs_version = "PY3",
        deps = deps + [
            requirement_3_8("pytest"),
        ],
        **kwargs
    )
