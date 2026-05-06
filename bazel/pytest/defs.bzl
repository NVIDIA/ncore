# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Wrap py_test with a common pytest wrapper """

load("@ncore_pip_deps//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_test")

def pytest_test(name, srcs, python_versions = ["3.11", "3.8"], deps = [], args = [], **kwargs):
    """
        Call pytest using a common wrapper script.

        To skip GPU-dependent tests, use `bazel test --config=no-gpu`.
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
            legacy_create_init = 0,
            deps = deps + [
                requirement("pytest"),
            ],
            **kwargs
        )
