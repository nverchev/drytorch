# Style guide

Recommendations for writing drytorch code: follow them unless there is a good
reason not to. Ruff (`ruff.toml`) and basedpyright (basic mode) enforce
formatting, linting and typing; their rules are not repeated here.

## Principles
- An experiment is correct if it is well documented: configuration that
  affects results is recorded and does not change.
- Type checkers are the guard: immutability and interfaces are enforced
  statically.
- `lib` depends on protocols; implementations carry the protocol's duties
  (for example, emitting events).

## Writing
- One line when a quick look at the code makes the rest clear.
- A paragraph after the first line only for important behaviour that the
  code does not make clear.
- Other libraries' behaviour (for example, `torch.compile`) is not
  documented; users read those libraries' docs.
- This applies to docstrings, changelog entries and commit messages.

## Typing
- Avoid inline suppressions (`# type: ignore`, `# pyright: ignore`,
  `# ty: ignore`).
- Public type variables are named by role (`Input`, `Target`, `Output`);
  private ones state their variance (`_T`, `_T_contra`, `_R_co`).
- Declare variance explicitly. A class that stores or merges values of a
  type variable is invariant.
- `Self` for return types; as a parameter only in merge-like methods.
- Mark overrides with `@override` from `typing_extensions`.
- Abstract classes inherit `abc.ABC`.
- Callable types used in several places: a callable protocol with
  positional-only parameters, not `TypeAlias`.

## Attributes
- Prefer immutable objects for experiment settings.
- Declare every instance attribute in the class body; `__init__` only
  assigns.
- Configuration is public and `Final` in the class body: `attr: Final[T]`.
- Attributes and module-level constants that are never reassigned are
  `Final`; in trackers, only where reassignment would cause errors.
- Private attributes (`_attr`) are internal: not documented, not recorded.
- A container the class changes is private, exposed by a read-only property
  (`Mapping`, `Sequence`).
- No properties for configuration.

## Naming
- Modules: `-ing` form for base logic, plural for a class and its
  subclasses, the dependency's name for bindings.
- Counts: `n_*`.
- Boolean flags are plain words (`record`, `distribute`), without `should_`
  or `is_`.
- Protocols end in `Protocol`.
- Default names come from `repr_utils.DefaultName`, exposed by a read-only
  `name` property; recorded objects add `repr_utils.CreatedAtMixin` for
  their timestamp.

## Layout
- Forward references: `from __future__ import annotations` instead of
  quoted names (`'MyClass'`), only in modules that need it.
- `from drytorch.core import protocols as p`.
- Import modules, not their members: `from torch import distributed as dist`,
  then `torch.nn.Module`. Exception: names from `typing`,
  `typing_extensions`, `collections.abc` and `_typeshed`.
- Empty collections are literals; the type comes from the class-body or
  variable annotation.
- A blank line after every indented block.
- A blank line before a full-line comment, except at the start of a block or
  right after a docstring.
- An explicit `return` at the end of every function.

## Docstrings
- Class docstrings list public attributes under `Attributes:`.
- Argument descriptions start in lowercase.
- `__init__`: `"""Initialize."""` plus `Args:`.
- Overrides get a docstring only when their behaviour differs.

## Sphinx
- Every module defines `__all__` after the imports: Sphinx documents the
  classes and functions it lists.
- A module docstring's first line is its summary in the API tables.
- Docstrings are reStructuredText, not Markdown.
- Code in docstrings uses double backticks (``` ``name`` ```); single
  backticks render as italic text.
- Link other objects with roles: `:class:`, `:func:`, `:meth:`, `:attr:`,
  `:mod:`, `:exc:`, without the `py:` prefix.
- `~` shows only the last name: ``:class:`~drytorch.lib.objectives.Metric` ``.
- External objects link through intersphinx: ``:class:`torch.nn.Module` ``.
- Types are not repeated in docstrings; Sphinx takes them from the
  annotations.

## Errors
- Arguments that break the signature or the documented rules: built-in
  errors (`TypeError`, `ValueError`).
- Actions that break drytorch's rules (for example, registering the same
  model twice): a `DryTorchError`.
- Library errors and warnings live in `core/exceptions.py`, subclass
  `DryTorchError` or `DryTorchWarning`, and use `_template`.
- Warnings: `warnings.warn(exceptions.SomeWarning(...), stacklevel=...)`.

## Events
- Events are frozen dataclasses in `core/log_events.py`, published on
  creation.
- Events are records; `IterateBatchEvent.push_updates` is the only
  exception.
- Trackers handle events with a `functools.singledispatchmethod` `notify`
  and handlers named `_`, each ending with `return super().notify(event)`.

## Tests
- The other sections apply loosely to tests.
- pytest only, no `unittest`: mocks come from `pytest-mock` (`mocker.Mock`,
  `mocker.patch`, `mocker.create_autospec`); scoped fixtures use
  `class_mocker` or `module_mocker`.
- A `TestClassName` class for each class under test; a module-level
  `test_*` function for each function.
- The object under test is a fixture inside its test class; an autouse
  `setup` fixture stores shared mocks on `self`.
- Mocks of drytorch objects come from their protocol
  (`mocker.create_autospec(SomeProtocol, instance=True)`) and are shared
  through `conftest.py`.
- `tests/units` tests one module with mocks; `tests/integration` combines
  real components; `tests/functional` runs whole workflows. Each level has
  its own `conftest.py`.
- Unit test files mirror the source: `tests/units/<package>/test_<module>.py`.
- Arrange, act, assert, with all assertions at the end.
- Each test checks one behaviour; several assertions are fine when together
  they describe it.
- Test docstrings start with "Test"; fixture docstrings say what they
  provide.
- `tmp_path` for files; `pytest.raises` and `pytest.warns` for errors and
  warnings; `pytest.mark.parametrize` for input variations.
- Skipped tests state their reason.

## Changelog and commits
- Changelog sections: `[BREAKING CHANGES]`, `Added`, `Changed`, `Fixed`;
  one lowercase line per entry, code names in backticks.
- Commit prefixes: `feat:`, `fix:`, `created:`, `refactor:`, `chore:`,
  `docs:`, `test:`, `project:` (project settings); one topic per commit.
