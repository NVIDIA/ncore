workspace(name = "drivesim-ai")

# Load python rules
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_python",
    sha256 = "5fa3c738d33acca3b97622a13a741129f67ef43f5fdfcec63b29374cc0574c29",
    strip_prefix = "rules_python-0.9.0",
    url = "https://github.com/bazelbuild/rules_python/archive/refs/tags/0.9.0.tar.gz",
)

# Register python toolchain
load("@rules_python//python:repositories.bzl", "python_register_toolchains")

python_register_toolchains(
    name = "python38",
    # Available versions are listed in @rules_python//python:versions.bzl.
    python_version = "3.8",
)

# Create a central repo that knows about the dependencies needed from requirements.txt.
load("@python38//:defs.bzl", "interpreter")
load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    name = "pip_deps",
    python_interpreter_target = interpreter,
    quiet = False,
    requirements_lock = "//:requirements.txt",
)

# Initialize repositories for all packages in requirements.txt.
load("@pip_deps//:requirements.bzl", "install_deps")

install_deps()
