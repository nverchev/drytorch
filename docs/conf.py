"""Configuration file for the Sphinx documentation builder."""

import os
import sys
import tomllib


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
    'inherited-members': True,
    'show-inheritance': True,
}

autodoc_typehints = 'description'
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

# General configuration
templates_path = ['_templates']
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    'tutorials/*.ipynb',
    'jupyter_execute',
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
}
