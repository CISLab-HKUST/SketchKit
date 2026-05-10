# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "SketchKit"
copyright = "2025, SketchKit Contributors"
author = "SketchKit Contributors"

# -- Determine current docs version ------------------------------------------
version = os.environ.get("SKETCHKIT_DOCS_VERSION", "main")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.apidoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # for Google/NumPy style docstrings
    "sphinx.ext.viewcode",
    "myst_parser",
]
autosummary_generate = True
primary_domain = 'py'

apidoc_modules = [
    {
        "path": "../../sketchkit",
        "destination": "api",
        "exclude_patterns": ["**/test*"],
        "max_depth": 4,
        "follow_links": False,
        "separate_modules": False,
        "include_private": True,
        "no_headings": False,
        "module_first": False,
        "implicit_namespaces": False,
        "automodule_options": {"members", "show-inheritance", "undoc-members"},
    }
]

autodoc_mock_imports = ["gdown", "svg", "ndjson"]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]

# Make the current version available in templates
html_context = {
    "version": version
}
