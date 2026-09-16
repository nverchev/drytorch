# Style guide

Recommendations for writing drytorch code: follow them unless there is a good
reason not to. Ruff (`ruff.toml`) and basedpyright (standard mode) enforce
formatting, linting and typing; their rules are not repeated here.
`.pre-commit-config.yaml` runs them, and jupytext, which keeps each
tutorial's paired `.md` and `.ipynb` in sync.

## Principles
- An experiment is correct if it is well documented: configuration that
  affects results is recorded and does not change.
- Type checkers are the guard: immutability and interfaces are enforced
  statically.
- `lib` depends on protocols; implementations carry the protocol's duties
  (for example, emitting events).
- Protocols are necessary conditions for the framework, abstract classes are
  sufficient conditions with recommended behavior.

## Typing
- Avoid inline suppressions (`# type: ignore`, `# pyright: ignore`,
  `# ty: ignore`).
- Public type variables are named by role (`Input`, `Target`, `Output`);
  private ones state their variance (`_T`, `_T_contra`, `_R_co`).
- Declare variance explicitly. A class that stores or merges values of a
  type variable is invariant.
- `Self` for return types; as a parameter only in merge-like methods.
- Mark overrides with `@override` from `typing_extensions`.
- Narrow an unknown type with a runtime-checkable protocol check, not with
  `getattr` or `hasattr`: the check narrows the type for the checker.
  `getattr` is for dynamic names and Python internals (`__dict__`,
  `__slots__`, `__len__`).
- Write simple types in public signatures in full, so a preview shows what
  to pass without opening the source: `Callable[[Output, Target], Tensor]`,
  not an alias or a callable protocol.

## Attributes
- Prefer immutable objects for experiment settings.
- Declare every instance attribute in the class body; `__init__` only
  assigns.
- Configuration is public and `Final` in the class body: `attr: Final[T]`.
  Other attributes and module-level constants are `Final` when never
  reassigned; in trackers, only where reassignment would cause errors.
- Private attributes (`_attr`) are internal: not documented, not recorded.
- A container the class changes is private, exposed by a read-only property
  (`Mapping`, `Sequence`).

## Protocols
- Protocols end in `Protocol` and are minimal: they declare only what the
  code using them needs.
- Abstract classes inherit `abc.ABC`.
- Protocol members carry a docstring and no body; `...` is only for private
  abstract methods and for `@overload` stubs, which the implementation
  documents.
- Optional behaviour of an object the library does not own is its own small
  protocol, checked with `isinstance`; a protocol used by one module only
  stays private to it.
- Protocols declare attributes as abstract read-only properties; a `Final`
  attribute satisfies them, so implementations keep plain names.
- Classes match protocols structurally and never inherit them: inheriting
  makes the protocol's properties unassignable and its members abstract.
  `tests/units/core/test_protocols.py`,
  `tests/integration/lib/test_protocol_implementations.py` and
  `tests/functional/contrib/test_metrics_compatibility.py` check that the
  implementations match; basedpyright checks all three.

## Naming
- Modules: `-ing` form for base logic, plural for a class and its
  subclasses, the dependency's name for bindings.
- Counts: `n_*`.
- Boolean flags are plain words (`record`, `distribute`), without `should_`
  or `is_`.
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
- No other blank lines in a function body.
- An explicit `return` at the end of every function.
- Parameters an override does not use are consumed with `_not_used = ...`.
- Avoid parentheses to break expressions across lines, reserving them
  exceptionally for long strings: rephrase with statements instead (for
  example an `if`/`else` clause). Parentheses that contain a single line
  are fine.

## Order
A preferred order for equivalent code. Break it when a name must be defined
first (a helper used in a class body, the bound of a type variable) or when
Python requires it (dataclass fields with defaults after those without).
- Modules: docstring, imports, `TYPE_CHECKING` block, `__all__`, type
  variables and aliases, constants, public classes, public functions,
  private classes and functions.
- Classes: docstring, class variables (public, then private, including
  descriptors), instance attributes (public, then private), `__init__`,
  `__call__`, properties (each setter after its property), other special
  methods, public methods, operators (`__add__`, `__or__`), private methods,
  static and class methods (public before private). In trackers, `notify` and
  its registered handlers are placed as the first method after properties.
- `@override` does not move a member out of its group.
- Arguments: positional first, then keyword; parameters inherited from a base
  class precede the ones the subclass adds.
- Alphabetical order is weakly preferred within a group, for members and for
  arguments alike. An order of use (`update`, `compute`, `reset`) is also
  fine, and a subclass keeps the order of its base class or protocol.
- Items that belong together form their own group, separated by a commented
  header banner:
  ```python
  # ---------------------------------------------------------------------------
  # Title
  # ---------------------------------------------------------------------------
  ```
  and ordered alphabetically within each group: in `core/exceptions.py`,
  warnings follow errors; in `core/protocols.py`, contract protocols follow
  component protocols; in `lib/runners.py`, private protocols follow runners.
- The class body, `__init__` and the docstring list attributes and arguments
  in the same order; docstrings list no private attributes.
- Test modules: docstring, imports, constants, module fixtures, private
  helper classes and functions, test classes, test functions.
- Test classes: fixtures first, starting with `setup`, then the tests.

## Writing
- One line when a quick look at the code makes the rest clear.
- A paragraph after the first line only for important behaviour that the
  code does not make clear.
- Other libraries' behaviour (for example, `torch.compile`) is not
  documented; users read those libraries' docs.
- This applies to docstrings and changelog entries.

## Docstrings
- Class docstrings list every public attribute under `Attributes:`.
  Properties are methods: their own docstring documents them.
- Public functions and methods document their arguments (`Args:`), return
  value (`Returns:`), errors (`Raises:`) and side effects
  (`Side Effects:`); the one-line rule in Writing applies to the
  description, not to these sections.
- Argument descriptions start in lowercase.
- `__init__`: `"""Initialize."""` plus `Args:`.
- Overrides get a docstring only when their behaviour differs.
- Private functions and methods (`_name`) get no docstring, except abstract
  ones that subclasses implement; a comment where the logic needs one.
- Types are not repeated in docstrings; Sphinx takes them from the
  annotations.
- A module docstring's first line is its summary in the API tables: it names
  the contents without a prefix such as "Module containing".
- Docstrings are reStructuredText, not Markdown: code uses double backticks
  (``` ``name`` ```), since single backticks render as italic text.
- Link other objects with roles (`:class:`, `:func:`, `:meth:`, `:attr:`,
  `:mod:`, `:exc:`), without the `py:` prefix; `~` shows only the last name
  (``:class:`~drytorch.lib.objectives.Metric` ``), and external objects link
  through intersphinx (``:class:`torch.nn.Module` ``).

## Documentation
- The API pages list what each module's `__all__` names, so every module
  defines it after the imports. Package `__init__.py` files, which hold only
  a docstring, do not.
- Tutorials are MyST `.md` files paired with notebooks: edit the `.md`, then
  run `jupytext --sync`; do not edit the `.ipynb`.
- Build as Read the Docs does, after `uv sync --all-extras`:
  `DRYTORCH_INIT_MODE=minimal uv run sphinx-build -b html docs docs/_build/html`.
  The notebooks run during the build; `-D nb_execution_mode=off` skips them.

## Errors
- Arguments that break the signature or the documented rules: built-in
  errors (`TypeError`, `ValueError`).
- Actions that break drytorch's rules (for example, registering the same
  model twice): a `DryTorchError`.
- Library errors and warnings live in `core/exceptions.py`, subclass
  `DryTorchError` or `DryTorchWarning`, and use `_template`.
- Warnings: `warnings.warn(exceptions.SomeWarning(...), stacklevel=...)`.
- Optional dependencies are imported inside `try`/`except (ImportError,
  ModuleNotFoundError)`, warning with `FailedOptionalImportWarning`.

## Events
- Events are frozen dataclasses in `core/log_events.py`, published on
  creation.
- Events are records; `IterateBatchEvent.push_updates` is the only
  exception.
- Trackers handle events with a `functools.singledispatchmethod` `notify`
  and handlers named `_`, each ending with `return super().notify(event)`.
  `notify` and its handlers are placed first after properties.
- Events follow the chronological workflow order:
  `StartExperimentEvent`, `ModelRegistrationEvent`, `LoadModelEvent`,
  `ActorRegistrationEvent`, `StartTrainingEvent`, `StartEpochEvent`,
  `IterateBatchEvent`, `MetricEvent`, `EndEpochEvent`,
  `IncrementEpochEvent`, `LearningRateEvent`, `SaveModelEvent`,
  `PauseExperimentEvent`, `ContinueExperimentEvent`,
  `TerminatedTrainingEvent`, `EndTrainingEvent`, `StartTestEvent`,
  `EndTestEvent`, `StopExperimentEvent`.
- This workflow order defines the order of event classes in
  `core/log_events.py` and the order of `@notify.register` handlers in
  trackers.

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
- Shared fixtures are named by what they provide: `example_*` for values, in
  the package-scoped `tests/conftest.py`, and `mock_*` for protocol mocks, in
  the `conftest.py` of their level.
- A test module that needs an optional dependency imports it at the top and
  skips itself: `pytest.skip('<name> not available', allow_module_level=True)`
  in the `except ImportError` branch.
- `tests/units` tests one module with mocks; `tests/integration` combines
  real components; `tests/functional` runs whole workflows. Each level has
  its own `conftest.py`.
- Unit test files mirror the source:
  `tests/units/<package>/test_<module>.py` (except `contrib`, tested under
  integration).
- Arrange, act, assert, with all assertions at the end: a blank line after
  arranging and a blank line after acting, besides the blank lines that
  Layout allows.
- Each test checks one behaviour; several assertions are fine when together
  they describe it.
- Fixture docstrings say what they provide; private fixtures (`_fixture`)
  need no docstring.
- `tmp_path` and `tmp_path_factory` for files; `pytest.raises` and
  `pytest.warns` for errors and warnings; `pytest.mark.parametrize` for input
  variations.
- Skipped tests state their reason.

## Changelog and commits
- Changelog sections: `[BREAKING CHANGES]`, `Added`, `Changed`, `Removed`,
  `Fixed`; one lowercase line per entry, code names in backticks. A removal
  goes under `Removed` only, even when it breaks the API.
- Commit prefixes: `feat:`, `fix:`, `created:`, `refactor:`, `chore:`,
  `docs:`, `test:`, `project:` (project settings); one topic per commit.
- Commit messages are a single line; the diff holds the details.
