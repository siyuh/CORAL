# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath(".."))

# Ensure pandoc from pypandoc_binary is discoverable
_pandoc_dir = os.path.join(
    os.path.dirname(__import__("pypandoc").__file__), "files"
)
if os.path.isdir(_pandoc_dir):
    os.environ["PATH"] = _pandoc_dir + os.pathsep + os.environ.get("PATH", "")

project = "CORAL"
copyright = "2025, Siyu He"
author = "Siyu He"
release = "0.0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# Theme
html_theme = "furo"
html_static_path = ["_static"]
html_logo = "_static/coral_logo.png"
html_title = "CORAL"

# Napoleon settings (NumPy-style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_mock_imports = [
    "torch",
    "torch_geometric",
    "torchvision",
    "scanpy",
    "anndata",
    "scipy",
    "sklearn",
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "networkx",
    "umap",
    "cv2",
    "einops",
]

# nbsphinx settings
nbsphinx_execute = "never"  # Don't re-execute notebooks during build

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "anndata": ("https://anndata.readthedocs.io/en/latest/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
}
