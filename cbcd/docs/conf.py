"""Sphinx configuration for the cbcd user docs.

The site is built with Sphinx + MyST + Furo + sphinx-autoapi following
the suite-wide doc style (see suite/docs/style_options/ for the
options that were considered).

Hand-authored pages live under ``cbcd/docs/`` (this file's directory);
the API reference is regenerated from docstrings on every build by
sphinx-autoapi. Internal-developer material (the design folder,
implementation journal, parity reports, the causal-learn audit) is
preserved next door but excluded from the rendered site via
``exclude_patterns`` below.
"""

from __future__ import annotations

import importlib.metadata as _md
import sys
from pathlib import Path

# Make the package importable so autoapi / autodoc can introspect it.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# -- Project information -----------------------------------------------------

project = "cbcd"
author = "Pavel Averin"

try:
    release = _md.version("cbcd")
except _md.PackageNotFoundError:  # pragma: no cover
    release = "0.0.0"
version = release.split("+", 1)[0]
copyright = f"2026, {author}"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "autoapi.extension",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",  # numpydoc / google docstrings
    "sphinx.ext.mathjax",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}

templates_path = ["_templates"]

# Anything matching these is hidden from the rendered site. Internal
# material that lives in cbcd/docs/ for developer reference but should
# not appear on the public site is listed explicitly.
exclude_patterns = [
    "_build",
    "_internal",
    "README.md",
    "journal.md",
    "parity_report.md",
    "audit_causal_learn.md",
    "design",
    ".ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
]

# -- MyST options ------------------------------------------------------------

myst_enable_extensions = [
    "amsmath",       # block-level LaTeX math: \begin{align*}...\end{align*}
    "dollarmath",    # inline $..$ and display $$..$$ math
    "deflist",       # definition lists
    "colon_fence",   # ::: admonitions
    "smartquotes",
    "html_image",
    "linkify",
]

myst_heading_anchors = 3
myst_dmath_double_inline = True

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = f"cbcd {version}"
html_static_path = ["_static"]
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_buttons": ["view", "edit"],
    "source_repository": "",  # filled in once cbcd has a public remote
    "footer_icons": [],
}

pygments_style = "tango"
pygments_dark_style = "monokai"

# -- sphinx-autoapi ---------------------------------------------------------

autoapi_type = "python"
autoapi_dirs = [str(_REPO_ROOT / "cbcd")]
autoapi_root = "reference/api"
autoapi_keep_files = False
autoapi_add_toctree_entry = False  # we wire it manually in reference/index.md
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_python_class_content = "both"
autoapi_member_order = "groupwise"
autoapi_ignore = ["*/_internal/*", "*/tests/*", "*/parity/*"]

# -- intersphinx -------------------------------------------------------------
# Cross-link into the standard library, NumPy, pandas, networkx, and
# eventually the sister packages once they have published doc sites.

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
}
intersphinx_disabled_reftypes = ["std:doc"]

# -- napoleon ----------------------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
