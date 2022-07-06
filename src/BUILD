load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "lib",
    srcs = [
        "__init__.py",
        "common.py",
        "transformations.py",
        "visualization.py",
        "waymo_utils.py",
        "nvidia_utils.py"
    ],
    visibility = ["//:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("pillow"),
        requirement("scipy"),
    ],
)

py_binary(
    name = "testbinary",
    srcs = ["testbinary.py"],
    deps = [
        ":lib",
    ],
)
