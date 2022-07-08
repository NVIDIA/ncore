load("@rules_python//python:defs.bzl", "py_binary")
load("@pip_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "train",
    srcs = ["train.py"],
    deps = [
        requirement("torch"),
        requirement("torchvision"),
        requirement("apex"),
        requirement("numpy"),
        requirement("runx"),
        requirement("opencv-python"),
        requirement("scikit-image"),
        requirement("tqdm"),
    ],
)
