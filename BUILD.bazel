load("@rules_python//python:pip.bzl", "compile_pip_requirements")

# Check that our compiled requirements are up-to-date
compile_pip_requirements(
    name = "requirements",
    extra_args = ["--allow-unsafe"],
)