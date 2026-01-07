"""Configuration file for the Sphinx documentation builder."""

import os
import sys
import tomllib

from typing import Any

from sphinx.application import Sphinx
from sphinx.ext.autodoc import Options


# Get the absolute path of the directory containing this file (docs/)
CONF_DIR = os.path.abspath(os.path.dirname(__file__))

# Correctly point to src relative to this file
sys.path.insert(0, os.path.join(os.path.dirname(CONF_DIR), 'src'))

with open(os.path.join(os.path.dirname(CONF_DIR), 'pyproject.toml'), 'rb') as f:
    pyproject = tomllib.load(f)
    project = pyproject['project']['name']
    release = pyproject['project']['version']
    author = pyproject['project']['authors'][0]['name']

# Extensions
extensions = [
    'myst_nb',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinxcontrib.mermaid',
    'sphinx.ext.intersphinx',
]

# Autodoc configuration
add_module_names = False
autosummary_generate = True
autoclass_content = 'both'
autodoc_member_order = 'bysource'
autodoc_default_options = {
    'members': True,
    'special-members': '__call__',
    'show-inheritance': True,
    'private-members': False,
}

autodoc_typehints = 'both'
autodoc_typehints_description_target = 'all'
autodoc_typehints_format = 'short'

# Napoleon configuration
napoleon_attr_annotations = True
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# MyST configuration
myst_enable_extensions = [
    'colon_fence',
    'dollarmath',
    'deflist',
]
myst_fence_as_directive = ['mermaid']

# Notebook execution configuration
nb_execution_mode = 'cache'
nb_execution_cache_path = os.path.join(CONF_DIR, 'jupyter_cache')
nb_execution_timeout = 600

# General configuration
templates_path = ['_templates']
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    'tutorials/*.ipynb',
    'jupyter_execute',
    'jupyter_cache',
]

# HTML output configuration
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = '_static/drytorch_logo.png'
html_favicon = '_static/drytorch_icon.png'
html_theme_options = {'logo_only': True}

# Cross-references
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'optuna': ('https://optuna.readthedocs.io/en/stable/', None),
    'sqlalchemy': ('https://docs.sqlalchemy.org/en/20/', None),
    'plotly': ('https://plotly.com/python-api-reference/', None),
}


def customize_signature(
    app: Sphinx,
    what: str,
    name: str,
    obj: Any,
    options: Options,
    signature: str | None,
    return_annotation: str | None,
) -> tuple[str | None, str | None]:
    """Simplify Wandb default setting representation."""
    if signature is not None and name == 'drytorch.trackers.wandb.Wandb':
        pattern = ' Settings'
        index = signature.find(pattern)
        signature = signature[:index] + ' wandb.sdk.wandb_settings.Settings())'

    return signature, return_annotation


def setup(app: Sphinx) -> None:
    """Custom setup function."""
    app.connect('autodoc-process-signature', customize_signature)
