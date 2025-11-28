"""Configuration file for the Sphinx documentation builder."""

import os
import sys


sys.path.insert(0, os.path.abspath('../src'))
project = 'drytorch'
author = 'Nicolas Vercheval'
release = '0.1.0rc'
extensions = [
    'myst_nb',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc.typehints',
    'sphinx.ext.viewcode',
    'sphinxcontrib.mermaid',
]
autosummary_generate = True  # Generate stub files automatically
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
myst_enable_extensions = [
    'colon_fence',
    'dollarmath',
    'deflist',
]
autoclass_content = 'both'
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'tutorials/*.ipynb']
nb_execution_mode = 'off'
myst_fence_as_directive = ['mermaid']
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = '_static/drytorch_logo.png'
html_favicon = '_static/drytorch_icon.png'
html_theme_options = {'logo_only': True}
