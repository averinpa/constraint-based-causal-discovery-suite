"""Sphinx configuration for the dagsampler user docs.

Same setup as the sister cbcd / bnm / citk doc sites:
Sphinx + MyST + Furo + sphinx-autoapi, hand-authored pages in a
Diátaxis layout, internal-developer material excluded from the
rendered build.
"""

from __future__ import annotations

import importlib.metadata as _md
import sys
from pathlib import Path

# docs/conf.py → docs → repo root.  dagsampler uses the src/ layout,
# so the package lives at <repo>/src/dagsampler/.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

# -- Project information -----------------------------------------------------

project = "dagsampler"
author = "Pavel Averin"

try:
    release = _md.version("dagsampler")
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
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}

templates_path = ["_templates"]

exclude_patterns = [
    "_build",
    "_internal",
    "journal.md",
    ".ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
]

# -- MyST options ------------------------------------------------------------

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
    "deflist",
    "colon_fence",
    "smartquotes",
    "html_image",
    "linkify",
]

myst_heading_anchors = 3
myst_dmath_double_inline = True

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = f"dagsampler {version}"
html_static_path = ["_static"]
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_buttons": ["view", "edit"],
    "source_repository": "",  # filled in once dagsampler has a public remote
    "footer_icons": [],
}

pygments_style = "tango"
pygments_dark_style = "monokai"

# -- sphinx-autoapi ----------------------------------------------------------

autoapi_type = "python"
autoapi_dirs = [str(_REPO_ROOT / "src" / "dagsampler")]
autoapi_root = "reference/api"
autoapi_keep_files = False
autoapi_add_toctree_entry = False
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_python_class_content = "both"
autoapi_member_order = "groupwise"
autoapi_ignore = ["*/_internal/*", "*/cli.py"]

# -- intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
intersphinx_disabled_reftypes = ["std:doc"]

# -- napoleon ----------------------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
