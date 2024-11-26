# docs/source/conf.py
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information
project = "nanofed"
copyright = "2024, Camille Dunning"
author = "Camille Dunning"
release = "0.1.0"

# -- General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_design",  # For grids, cards, tabs
    "sphinx_tabs.tabs",  # For tab sets
    "sphinx_copybutton",  # For copy buttons on code blocks
    "sphinx_togglebutton",  # For dropdowns/toggles
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output
html_theme = "sphinx_book_theme"
html_title = "NanoFed"

html_theme_options = {
    "repository_url": "https://github.com/camille-004/nanofed",
    "use_repository_button": True,
    "use_issues_button": True,
}

# -- Extension configuration
myst_enable_extensions = [
    "colon_fence",
]

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}
