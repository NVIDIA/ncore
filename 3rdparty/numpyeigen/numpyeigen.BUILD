load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "npe",
    srcs = [
        "src/npe_typedefs.cpp",
    ],
    hdrs = [
        "src/npe.h",
        "src/npe_dense_array.h",
        "src/npe_dtype.h",
        "src/npe_sparse_array.h",
        "src/npe_typedefs.h",
        "src/npe_utils.h",
    ],
    includes = ["src"],
    deps = [
        "@eigen",
        "@numpyeigen_pybind11//:pybind11",
        "@pip_deps_numpy//:numpy",
    ],
)

py_binary(
    name = "codegen_function",
    srcs = ["src/codegen_function.py"],
)

py_binary(
    name = "codegen_module",
    srcs = ["src/codegen_module.py"],
)
