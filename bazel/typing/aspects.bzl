# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Mypy aspects for type checking Python targets.

This module provides:
- `mypy_aspect`: The default mypy aspect for the ncore repository
- `mypy_with_first_party_deps`: A factory for creating mypy aspects that can
  type-check against first-party external dependencies (e.g., local_path_override modules)

Example usage in a sub-module:

    load("@ncore//bazel/typing:aspects.bzl", "mypy_with_first_party_deps")

    mypy_aspect = mypy_with_first_party_deps(
        mypy_cli = "//bazel/typing:mypy",
        mypy_ini = "//bazel/typing:mypy.ini",
        first_party_repos = ["ncore+"],
    )
"""

load("@rules_mypy//mypy:mypy.bzl", "mypy")
load("@rules_python//python:py_info.bzl", RulesPythonPyInfo = "PyInfo")

# =============================================================================
# Default aspect for the ncore repository (no first-party external deps)
# =============================================================================

mypy_aspect = mypy(
    mypy_cli = "@@//bazel/typing:mypy",
    mypy_ini = "@@//bazel/typing:mypy.ini",
)

# =============================================================================
# Reusable aspect factory for sub-modules with first-party external deps
# =============================================================================

def _extract_import_dir(import_):
    """Extract the import directory from a full import path."""

    # _main/path/to/package -> path/to/package
    return import_.split("/", 1)[-1]

def _imports(target):
    """Get imports from a target's PyInfo provider."""
    if RulesPythonPyInfo in target:
        return target[RulesPythonPyInfo].imports.to_list()
    elif PyInfo in target:
        return target[PyInfo].imports.to_list()
    else:
        return []

def _extract_imports(target):
    """Extract cleaned import paths from a target."""
    return [_extract_import_dir(i) for i in _imports(target)]

def _opt_out(opt_out_tags, rule_tags):
    """Returns true iff at least one opt_out_tag appears in rule_tags."""
    for tag in opt_out_tags:
        if tag in rule_tags:
            return True
    return False

def _make_mypy_impl(first_party_repos):
    """Create a mypy aspect implementation with the given first-party repos."""

    def _is_first_party_external(workspace_root):
        """Check if the workspace root is a first-party external dependency."""
        for repo in first_party_repos:
            if workspace_root == "external/" + repo:
                return True
        return False

    def _mypy_impl(target, ctx):
        # Skip non-root targets, EXCEPT for first-party externals which we want
        # to include in the MYPYPATH but not actually type-check
        if target.label.workspace_root != "" and not _is_first_party_external(target.label.workspace_root):
            return []

        # For first-party externals, we only collect their sources for MYPYPATH
        # We don't run mypy on them (they're checked in their own repo)
        is_first_party_external = _is_first_party_external(target.label.workspace_root)

        if RulesPythonPyInfo not in target and PyInfo not in target:
            return []

        # Disable if a target is tagged with at least one suppression tag
        if _opt_out(ctx.attr._suppression_tags, ctx.rule.attr.tags):
            return []

        # Ignore rules that don't carry source files like py_proto_library
        if not hasattr(ctx.rule.files, "srcs"):
            return []

        # For first-party externals, just return empty (they contribute to deps but aren't checked)
        if is_first_party_external:
            return []

        # Collect paths for MYPYPATH
        external_deps = {}
        imports_dirs = {}
        generated_dirs = {}
        depsets = []

        # Add imports from the target itself
        for import_ in _extract_imports(target):
            imports_dirs[import_] = 1

        # Process all dependencies
        for dep in ctx.rule.attr.deps:
            depsets.append(dep.default_runfiles.files)

            # Handle first-party externals specially - add the repo root to MYPYPATH
            # This allows mypy to find modules like `ncore.*` from `external/ncore+/ncore/...`
            if _is_first_party_external(dep.label.workspace_root):
                external_deps[dep.label.workspace_root] = 1
            elif dep.label.workspace_root.startswith("external/"):
                # Standard external deps (pip packages)
                # Add site-packages paths for pip deps
                for imp in _imports(dep):
                    if "site-packages" in imp:
                        external_deps["external/" + imp] = 1
            elif dep.label.workspace_name == "":
                # Internal deps
                for import_ in _extract_imports(dep):
                    imports_dirs[import_] = 1

            # Collect generated file directories
            for file in dep.default_runfiles.files.to_list():
                if file.root.path:
                    generated_dirs[file.root.path] = 1

        # Build generated imports dirs
        generated_imports_dirs = []
        for generated_dir in generated_dirs.keys():
            for import_ in imports_dirs.keys():
                generated_imports_dirs.append("{}/{}".format(generated_dir, import_))

        # Build MYPYPATH
        mypy_path = ":".join(
            sorted(external_deps) +
            sorted(imports_dirs) +
            sorted(generated_dirs) +
            sorted(generated_imports_dirs),
        )

        # Declare output files
        output_file = ctx.actions.declare_file(ctx.rule.attr.name + ".mypy_stdout")
        cache_directory = ctx.actions.declare_directory(ctx.rule.attr.name + ".mypy_cache")
        outputs = [output_file, cache_directory]

        # Build arguments
        args = ctx.actions.args()
        args.add("--output", output_file)
        args.add("--cache-dir", cache_directory.path)
        args.add_all([s for s in ctx.rule.files.srcs if "/_virtual_imports/" not in s.short_path])

        # Add mypy.ini if configured
        config_files = []
        if hasattr(ctx.attr, "_mypy_ini") and ctx.file._mypy_ini:
            args.add("--mypy-ini", ctx.file._mypy_ini.path)
            config_files = [ctx.file._mypy_ini]

        # Environment
        extra_env = {
            "MYPYPATH": mypy_path,
            "MYPY_FORCE_COLOR": "1",
            "TERM": "xterm-256color",
        }

        # Run mypy
        ctx.actions.run(
            mnemonic = "mypy",
            progress_message = "mypy %{label}",
            inputs = depset(
                direct = ctx.rule.files.srcs + config_files,
                transitive = depsets,
            ),
            outputs = outputs,
            executable = ctx.executable._mypy_cli,
            arguments = [args],
            env = extra_env | ctx.configuration.default_shell_env,
        )

        return [OutputGroupInfo(mypy = depset(outputs))]

    return _mypy_impl

def mypy_with_first_party_deps(
        mypy_cli,
        mypy_ini = None,
        suppression_tags = None,
        first_party_repos = None):
    """
    Create a mypy aspect that includes first-party external dependencies.

    This is an extended version of rules_mypy's mypy() that properly handles
    local_path_override dependencies by including their sources in MYPYPATH.
    This allows mypy to resolve and type-check against APIs from first-party
    external modules.

    Args:
        mypy_cli: The mypy_cli target to use (create with rules_mypy's mypy_cli macro)
        mypy_ini: (optional) mypy.ini configuration file
        suppression_tags: (optional) tags that suppress running mypy (default: ["no-mypy"])
        first_party_repos: (optional) list of external repo names to treat as first-party
            dependencies. These repos will have their source roots added to MYPYPATH.
            Example: ["ncore+"] for a module that depends on @ncore via local_path_override

    Returns:
        A mypy aspect that can be used with --aspects flag

    Example:
        # In your sub-module's bazel/typing/aspects.bzl:
        load("@ncore//bazel/typing:aspects.bzl", "mypy_with_first_party_deps")

        mypy_aspect = mypy_with_first_party_deps(
            mypy_cli = "//bazel/typing:mypy",
            mypy_ini = "//bazel/typing:mypy.ini",
            first_party_repos = ["ncore+"],
        )
    """
    first_party_repos = first_party_repos or []
    suppression_tags = suppression_tags or ["no-mypy"]

    attrs = {
        "_mypy_cli": attr.label(
            default = mypy_cli,
            cfg = "exec",
            executable = True,
        ),
        "_suppression_tags": attr.string_list(default = suppression_tags),
    }

    if mypy_ini:
        attrs["_mypy_ini"] = attr.label(
            default = mypy_ini,
            allow_single_file = True,
        )

    return aspect(
        implementation = _make_mypy_impl(first_party_repos),
        attr_aspects = ["deps"],
        attrs = attrs,
    )
