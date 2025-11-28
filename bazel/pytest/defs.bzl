# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

""" Wrap py_test with a common pytest wrapper """

load("@ncore_pip_deps//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_test")

def pytest_test(name, srcs, python_versions = ["3.11", "3.8"], deps = [], args = [], **kwargs):
    """
        Call pytest using a common wrapper script
    """
    for python_version in python_versions:
        kwargs["python_version"] = python_version
        py_test(
            name = name + "_%s" % python_version.replace(".", "_"),
            srcs = [
                "//bazel/pytest:pytest_wrapper.py",
            ] + srcs,
            main = "//bazel/pytest:pytest_wrapper.py",
            args = [
                "--capture=no",
            ] + args + ["$(location :%s)" % x for x in srcs],
            srcs_version = "PY3",
            deps = deps + [
                requirement("pytest"),
            ],
            **kwargs
        )
