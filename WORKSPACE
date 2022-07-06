workspace(name = "drivesim-ai")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

## Python rules
http_archive(
    name = "rules_python",
    sha256 = "56dc7569e5dd149e576941bdb67a57e19cd2a7a63cc352b62ac047732008d7e1",
    strip_prefix = "rules_python-0.10.0",
    url = "https://github.com/bazelbuild/rules_python/archive/refs/tags/0.10.0.tar.gz",
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
    annotations = {
        "numpy": package_annotation(
            additive_build_content = """
# Make numpy headers available to C++ rules
cc_library(
    name = 'numpy',
    includes = ['site-packages/numpy/core/include'],
    hdrs = glob(['site-packages/numpy/core/include/numpy/*.h']),
    visibility = ['//visibility:public'],
)
""",
        ),
    },
    python_interpreter_target = interpreter,
    quiet = False,
    requirements_lock = "//:requirements.txt",
)

# Initialize repositories for all packages in requirements.txt.
load("@pip_deps//:requirements.bzl", "install_deps")

install_deps()

## Docker rules
http_archive(
    name = "io_bazel_rules_docker",
    sha256 = "b1e80761a8a8243d03ebca8845e9cc1ba6c82ce7c5179ce2b295cd36f7e394bf",
    urls = ["https://github.com/bazelbuild/rules_docker/releases/download/v0.25.0/rules_docker-v0.25.0.tar.gz"],
)

load(
    "@io_bazel_rules_docker//repositories:repositories.bzl",
    container_repositories = "repositories",
)

container_repositories()

load("@io_bazel_rules_docker//repositories:deps.bzl", container_deps = "deps")

container_deps()

load(
    "@io_bazel_rules_docker//python3:image.bzl",
    _py_image_repos = "repositories",
)

_py_image_repos()

load(
    "@io_bazel_rules_docker//container:container.bzl",
    "container_pull",
)

container_pull(
    name = "pytorch_base",
    registry = "nvcr.io",
    repository = "nvidia/pytorch",
    # digest = "sha256:e3470579ea78",  #
    tag = "22.05-py3",
)

container_pull(
    name = "dsai_dev_container",
    digest = "sha256:bcd91ec75a4e71eb449cd2dcbf5b4207cb26a5fab4ce8cff634e13d0d5799386",
    registry = "gitlab-master.nvidia.com:5005",
    repository = "zgojcic/drivesim-ai",
)

## Protobuf rules
http_archive(
    name = "com_google_protobuf",
    sha256 = "33cba8b89be6c81b1461f1c438424f7a1aa4e31998dbe9ed6f8319583daac8c7",
    strip_prefix = "protobuf-3.10.0",
    urls = ["https://github.com/google/protobuf/archive/v3.10.0.zip"],
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

## 3rdparty
new_git_repository(
    name = "PoissonRecon",
    build_file = "//3rdparty/PoissonRecon:PoissonRecon.BUILD",
    commit = "f1c71fe491828d00d34f95de9c264d4bf3481d61",  # v13.74
    patch_args = ["-p1"],
    patches = ["//3rdparty/PoissonRecon:PoissonRecon.patch"],
    remote = "https://github.com/mkazhdan/PoissonRecon",
    shallow_since = "1650856848 -0400",
)
