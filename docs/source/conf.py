# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import renku_sphinx_theme

project = 'nonlinear-causal'
copyright = '2024, Ben Dai'
author = 'Ben Dai'

# -- Project information -----------------------------------------------------
import sys, os
# import numpydoc
sys.path.append('.')
sys.path.insert(0, os.path.abspath('../..'))
sys.path.append(os.path.abspath('../nl_causal'))
# sys.path.append(os.path.abspath('../../nonlinear-causal/nl_causal'))
# sys.path.append('../..')
# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
master_doc = 'index'
extensions = [
	'sphinx.ext.autodoc',
	'autoapi.extension',
	"sphinx_autodoc_typehints",
	'sphinx.ext.autosummary',
	'numpydoc',
	'nbsphinx',
	'myst_parser'
	]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

autoapi_dirs = ['../../nl_causal']

autosummary_generate = True
numpydoc_show_class_members = False
nbsphinx_execute = 'never'
nbsphinx_allow_errors = True
# autodoc_mock_imports = ['numpy']
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

# html_theme = 'alabaster'
html_theme = 'renku'
# html_logo = "logo.png"
# html_theme_path = [hachibee_sphinx_theme.get_html_themes_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

html_static_path = ['_static']

# html_css_files = [
#     'css/custom.css',
# ]

def skip_submodules(app, what, name, obj, skip, options):
    if what == "module":
        skip = True
    return skip


def setup(sphinx):
    sphinx.connect("autoapi-skip-member", skip_submodules)

autoapi_template_dir = "_templates/autoapi"
# autoapi_add_toctree_entry = False