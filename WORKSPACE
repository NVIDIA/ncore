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
load("@rules_python//python:pip.bzl", "package_annotation", "pip_parse")

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
    name = "dsai_dev_container",
    digest = "sha256:572192880eebe61ce5d570b755eeda3e820e1d66e392db66467733054212f2ed",
    registry = "gitlab-master.nvidia.com:5005",
    repository = "zgojcic/drivesim-ai",
)

## Protobuf rules
http_archive(
    name = "com_google_protobuf",
    sha256 = "25680843adf0c3302648d35f744e38cc3b6b05a6c77a927de5aea3e1c2e36106",
    strip_prefix = "protobuf-3.19.4",
    urls = ["https://github.com/google/protobuf/archive/v3.19.4.zip"],
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

new_git_repository(
    name = "numpyeigen",
    build_file = "//3rdparty/numpyeigen:numpyeigen.BUILD",
    commit = "4916d926aa2b939bd8f625c7537563a1575dafe9",
    remote = "https://github.com/fwilliams/numpyeigen",
    shallow_since = "1643644288 -0500",
)

new_git_repository(
    name = "numpyeigen_pybind11",
    build_file = "//3rdparty/numpyeigen_pybind11:numpyeigen_pybind11.BUILD",
    commit = "d8c0a26b06b4d6901f6af4b1cbdc975bb160221b",
    remote = "https://github.com/fwilliams/pybind11.git",
    shallow_since = "1656512085 -0400",
)

http_archive(
    name = "eigen",
    build_file_content =
        """
# TODO(janickm): Replace this with a better version, like from TensorFlow.
# See https://github.com/tensorflow/tensorflow/tree/master/third_party/eigen3
cc_library(
    name = 'eigen',
    includes = ['.'],
    hdrs = glob(['Eigen/**']),
    visibility = ['//visibility:public'],
)
""",
    sha256 = "685adf14bd8e9c015b78097c1dc22f2f01343756f196acdc76a678e1ae352e11",
    strip_prefix = "eigen-3.3.7",
    urls = [
        "https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.bz2",
    ],
)
