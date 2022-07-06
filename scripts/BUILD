load("@rules_python//python:defs.bzl", "py_binary")
load("@pip_deps//:requirements.bzl", "requirement")

py_binary(
    name = "nvidia_to_ngp",
    srcs = ["nvidia_to_ngp.py"],
    deps = [
        requirement("numpy"),
    ],
)
