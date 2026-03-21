from __future__ import annotations

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, os.path.abspath('../..'))

project = "PMLDL Assignment 1"
author = "Artyom Tuzov"
release = "main"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True

# These imports are not needed when Sphinx reads docstrings.
# Mocking them keeps the docs build lightweight and avoids UI/runtime setup.
autodoc_mock_imports = [
    'fastapi',
    'pydantic',
    'uvicorn',
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'streamlit',
    'streamlit_drawable_canvas',
    'PIL',
    'numpy',
    'tensorflow',
]


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
language = "en"
html_theme = "furo"
html_title = "PMLDL Assignment 1"
html_static_path = ["_static"]
