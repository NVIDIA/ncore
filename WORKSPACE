workspace(name = "dsai")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

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
    name = "python39",
    # Available versions are listed in @rules_python//python:versions.bzl
    python_version = "3.9",
)

# Create a central repo that knows about the dependencies needed from requirements.txt
load("@python39//:defs.bzl", "interpreter")
load("@rules_python//python:pip.bzl", "package_annotation", "pip_parse")

pip_parse(
    name = "pip_deps",
    timeout = 3600,
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
    requirements_lock = "//3rdparty/python:requirements.txt",
)

# Initialize repositories for all packages in 3rdparty/python/requirements.txt
load("@pip_deps//:requirements.bzl", "install_deps")

install_deps()

# mypy-integration
http_archive(
    name = "mypy_integration",
    sha256 = "cf94c102fbaccb587eea8de5cf1cb7f55c5c74396a2468932c3a2a4df989aa1d",
    strip_prefix = "bazel-mypy-integration-0.4.0",
    url = "https://github.com/thundergolfer/bazel-mypy-integration/archive/refs/tags/0.4.0.tar.gz",
)

load(
    "@mypy_integration//repositories:repositories.bzl",
    mypy_integration_repositories = "repositories",
)

mypy_integration_repositories()

load("@mypy_integration//repositories:deps.bzl", mypy_integration_deps = "deps")

mypy_integration_deps(
    mypy_requirements_file = "//bazel/typing:mypy_version.txt",
    python_interpreter_target = interpreter,
)

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
    timeout = 7200,
    digest = "sha256:6aac4280c52b8bae76e6aa57c67c3964006f005785393290db028fb9ed3d0225",
    registry = "gitlab-master.nvidia.com:5005",
    repository = "toronto_dl_lab/dsai",
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

new_git_repository(
    name = "semantic-segmentation",
    build_file = "//3rdparty/semantic-segmentation:semantic-segmentation.BUILD",
    commit = "7726b144c2cc0b8e09c67eabb78f027efdf3f0fa",
    patch_args = ["-p1"],
    patches = ["//3rdparty/semantic-segmentation:semantic-segmentation.patch"],
    remote = "https://github.com/NVIDIA/semantic-segmentation.git",
    shallow_since = "1614912803 -0800",
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

## bazel-starlib / buildifier
http_archive(
    name = "cgrindel_bazel_starlib",
    sha256 = "c95de004f346cbcb51ba1185e8861227cd9ab248b53046f662aeda1095601bc9",
    strip_prefix = "bazel-starlib-0.7.1",
    urls = [
        "http://github.com/cgrindel/bazel-starlib/archive/v0.7.1.tar.gz",
    ],
)

load("@cgrindel_bazel_starlib//:deps.bzl", "bazel_starlib_dependencies")

bazel_starlib_dependencies()

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()

load("@buildifier_prebuilt//:deps.bzl", "buildifier_prebuilt_deps")

buildifier_prebuilt_deps()

load("@buildifier_prebuilt//:defs.bzl", "buildifier_prebuilt_register_toolchains")

buildifier_prebuilt_register_toolchains()
