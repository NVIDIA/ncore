load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "pybind11",
    hdrs = glob(["include/**"]),
    includes = ["include"],
    deps = [
        "@python38//:libpython",
        "@python38//:python_headers",
    ],
)
