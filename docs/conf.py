"""Configuration file for the Sphinx documentation builder."""

import inspect
import os
import re
import sys
import tomllib

from typing import Any

from docutils import nodes
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.ext import autodoc, napoleon
from sphinx.util import typing as sphinx_typing


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
# the class docstring only: insert_init_parameters adds the __init__ one
autoclass_content = 'class'
autodoc_member_order = 'bysource'
autodoc_default_options = {
    'members': True,
    'special-members': '__call__',
    'show-inheritance': True,
    'private-members': False,
}

# types only in signatures: in descriptions, parameters took attribute types
autodoc_typehints = 'signature'
autodoc_typehints_format = 'short'
python_use_unqualified_type_names = True

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
myst_heading_anchors = 3

# Mermaid configuration
mermaid_dark_theme = 'default'

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
    'torch': ('https://docs.pytorch.org/docs/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'optuna': ('https://optuna.readthedocs.io/en/stable/', None),
    'sqlalchemy': ('https://docs.sqlalchemy.org/en/20/', None),
    'plotly': ('https://plotly.com/python-api-reference/', None),
    'torchmetrics': ('https://lightning.ai/docs/torchmetrics/stable/', None),
    'torcheval': ('https://meta-pytorch.org/torcheval/stable/', None),
}


# a type variable for a role, in any form Sphinx writes it
ROLE_TYPEVAR = re.compile(
    r'(?::py:obj:`)?(?<![\w])~?(?:[A-Za-z_]\w*\.)*'
    r'_?(Data|Input|Target|Output)(?:_contra|_co)?\b`?'
)
# the module path before a private type variable
PRIVATE_TYPEVAR_PATH = re.compile(r'(?<![\w])~?(?:[A-Za-z_]\w*\.)+(?=_[A-Z])')
# default values without a valid representation
FUNCTION_REPR = re.compile(r'<function (?:[\w.]+\.)?(\w+)(?: at 0x[0-9a-f]+)?>')
OBJECT_REPR = re.compile(r'<(?:[\w.]+\.)?(\w+) object(?: at 0x[0-9a-f]+)?>')


def customize_signature(
    app: Sphinx,
    what: str,
    name: str,
    obj: Any,
    options: autodoc.Options,
    signature: str | None,
    return_annotation: str | None,
) -> tuple[str | None, str | None]:
    """Make default values valid Python and simplify the Wandb settings.

    Sphinx shows a signature it cannot parse as plain text.
    """
    if signature is None:
        return signature, return_annotation

    if name == 'drytorch.trackers.wandb.Wandb':
        pattern = ' Settings'
        index = signature.find(pattern)
        signature = signature[:index] + ' wandb.sdk.wandb_settings.Settings())'

    signature = signature.replace('<factory>', '...')
    signature = FUNCTION_REPR.sub(r'\1', signature)
    signature = OBJECT_REPR.sub(r'\1()', signature)
    return signature, return_annotation


def highlight_cells_as_ipython(app: Sphinx, doctree: nodes.document) -> None:
    """Highlight notebook code cells with the IPython lexer.

    Without execution, cells use the Python lexer, which rejects shell
    commands such as ``! uv pip install``.
    """
    for block in doctree.findall(nodes.literal_block):
        parent = block.parent
        if not isinstance(parent, nodes.Element):
            continue

        if 'cell_input' not in parent['classes']:
            continue

        if block.get('language') == 'python':
            block['language'] = 'ipython3'

    return


def insert_init_parameters(
    app: Sphinx,
    what: str,
    name: str,
    obj: Any,
    options: autodoc.Options,
    lines: list[str],
) -> None:
    """Insert the ``__init__`` parameters before the class attributes.

    The ``Initialize.`` line of the ``__init__`` docstring is left out.
    """
    if what not in ('class', 'exception'):
        return

    init = getattr(obj, '__init__', None)
    doc = inspect.getdoc(init) if inspect.isfunction(init) else None
    if not doc or doc == inspect.getdoc(object.__init__):
        return

    parsed = napoleon.GoogleDocstring(
        doc, app.config, app, 'method', f'{name}.__init__', init
    ).lines()
    parameters = [line for line in parsed if line.strip() != 'Initialize.']
    if not any(line.strip() for line in parameters):
        return

    index = len(lines)
    for i, line in enumerate(lines):
        if line.startswith('.. attribute::'):
            index = i
            break

    lines[index:index] = [*parameters, '']
    return


def show_roles(text: str) -> str:
    """Show type variables by role and private ones without their module.

    Args:
        text: the text to change.

    Returns:
        The text with ``_Output_contra`` shown as ``Output``, for example.
    """
    text = ROLE_TYPEVAR.sub(r'\1', text)
    return PRIVATE_TYPEVAR_PATH.sub('', text)


def show_role_bases(
    app: Sphinx,
    name: str,
    obj: Any,
    options: autodoc.Options,
    bases: list[Any],
) -> None:
    """Show the type variables of the base classes by role."""
    for i, base in enumerate(bases):
        rendered = sphinx_typing.restify(base, mode='smart')
        shown = show_roles(rendered)
        if shown != rendered:
            bases[i] = shown

    return


def show_signature_roles(app: Sphinx, doctree: nodes.document) -> None:
    """Show the type variables of signatures and attribute types by role."""
    typed: list[nodes.Node] = list(doctree.findall(addnodes.desc_signature))
    for field in doctree.findall(nodes.field):
        if field[0].astext() == 'Type':
            typed.append(field[1])

    for element in typed:
        for text in list(element.findall(nodes.Text)):
            shown = show_roles(str(text))
            if shown != str(text):
                text.parent.replace(text, nodes.Text(shown))

    return


def skip_undefined_attributes(
    app: Sphinx,
    what: str,
    name: str,
    obj: Any,
    skip: bool,
    options: autodoc.Options,
) -> bool | None:
    """Skip attributes without a value in the class body.

    The class docstring documents them. Sphinx takes the first parameter
    after ``/`` for ``self``, so ``SWAModel.start_epoch`` would otherwise
    appear with the ``__init__`` docstring.
    """
    if obj is autodoc.INSTANCEATTR or obj is autodoc.UNINITIALIZED_ATTR:
        return True

    return None


def setup(app: Sphinx) -> None:
    """Custom setup function."""
    # the mermaid extension reads the reader's system preference and renders
    # dark diagrams and a dark fullscreen view on this light-only theme: the
    # page states it is light before the extension's module runs.
    app.add_js_file(
        None,
        body="document.documentElement.dataset.theme = 'light';",
        priority=400,
    )
    app.connect('autodoc-process-bases', show_role_bases)
    app.connect('autodoc-process-docstring', insert_init_parameters)
    app.connect('autodoc-process-signature', customize_signature)
    app.connect('autodoc-skip-member', skip_undefined_attributes)
    app.connect('doctree-read', highlight_cells_as_ipython)
    app.connect('doctree-read', show_signature_roles)
