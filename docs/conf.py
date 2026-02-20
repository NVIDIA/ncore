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


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "NCore"
copyright = "2026, NVIDIA Corporation & Affiliates"
author = "NVIDIA"

# Determine the Git version/tag from CI environment variables, fallback to 'main'.
GITHUB_VERSION = os.environ.get("GITHUB_REF_NAME") or "main"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.githubpages",
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx_copybutton",
]

todo_include_todos = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "PyTorch": ("https://pytorch.org/docs/main/", None),
    "zarr": ("https://zarr.readthedocs.io/en/stable/", None),
}

autodoc_default_options = {
    "show-inheritance": False,
    "undoc-members": False,
}

master_doc = "index"

templates_path = ["_templates"]
exclude_patterns = ["_build"]
html_static_path = ["_static"]

# -- Options for HTML output -------------------------------------------------

html_theme = "nvidia_sphinx_theme"
html_title = "NCore"
html_show_sphinx = False

html_theme_options = {
    "secondary_sidebar_items": ["page-toc"],
    "copyright_override": {"start": 2026},
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
    "footer_links": {},
    "github_url": "https://github.com/NVIDIA/ncore",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/nvidia-ncore",
            "icon": "fa-brands fa-python",
            "type": "fontawesome",
        },
    ],
    "navigation_depth": -1,
    "collapse_navigation": False,
}

html_context = {
    "github_user": "NVIDIA",
    "github_repo": "ncore",
    "github_version": GITHUB_VERSION,
    "doc_path": "docs",
    "default_mode": "light",
}

html_css_files = ["custom.css"]

# -- sphinx_copybutton -------------------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
