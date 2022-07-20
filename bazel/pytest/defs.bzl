""" Wrap py_test with a common pytest wrapper """

load("@rules_python//python:defs.bzl", "py_test")
load("@pip_deps//:requirements.bzl", "requirement")

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
