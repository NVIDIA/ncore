workspace(name = "dsai")

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
    name = "python3",
    # Available versions are listed in @rules_python//python:versions.bzl
    python_version = "3.10",
)

# Create a central repo that knows about the dependencies needed from requirements.txt
load("@python3//:defs.bzl", "interpreter")
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
    patch_args = ["-p1"],
    patches = [
        "@//:bazel/typing/mypy.patch",
    ],
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
    digest = "sha256:bd6a0d9ae9a9df131c372e5eb749556e14dc31fdcdf50365dd9548d29dd28ed0",
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

## Dependencies
load("//bazel/repo:repos.bzl", "register_repositories")

register_repositories()

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
