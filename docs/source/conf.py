# docs/source/conf.py
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information
project = "nanofed"
copyright = "2024, Camille Dunning"
author = "Camille Dunning"
release = "0.1.1"

# -- General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_design",
    "sphinx_tabs.tabs",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "sphinx_immaterial",
]


templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output
html_theme = "sphinx_immaterial"
html_title = "NanoFed"


html_theme_options = {
    "icon": {"repo": "fontawesome/brands/github"},
    "repo_url": "https://github.com/camille-004/nanofed",
    "repo_name": "nanofed",
    "edit_uri": "blob/main/docs/",
    "globaltoc_collapse": False,
    "features": [],
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "blue",
            "accent": "blue",
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "blue",
            "accent": "blue",
        },
    ],
    "social": [
        {
            "icon": "fontawesome/brands/github",
            "link": "https://github.com/camille-004/nanofed",
            "name": "GitHub",
        },
    ],
}

# -- Extension configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "substitution",
    "linkify",
    "smartquotes",
]

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# Add mermaid CDN
html_js_files = [
    ('https://cdn.jsdelivr.net/npm/mermaid@latest/dist/mermaid.min.js', {'async': 'async'}),
]
