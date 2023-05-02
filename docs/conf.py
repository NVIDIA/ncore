# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NRECore'
copyright = '2022, NVIDIA'
author = 'NVIDIA - Toronto AI Lab'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
]

todo_include_todos = True

intersphinx_mapping = {
    'python': ("https://docs.python.org/3", None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'PyTorch': ('https://pytorch.org/docs/master/', None),
    'zarr': ('https://zarr.readthedocs.io/en/stable/', None),
}

autodoc_default_options = {
    'show-inheritance': False,
    'undoc-members': False,
}

master_doc = 'index'

templates_path = ['_templates']
exclude_patterns = ['_build']
html_static_path = ['_static']

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
import sphinx_rtd_theme

html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    'collapse_navigation': False,
    'navigation_depth': -1,
}
