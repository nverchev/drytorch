## [0.1.0] - 02-09-2026

## Added
- added experiment pause and continue feature to allow pausing training sessions cleanly
- added specific behaviour for trackers on pause to safely stash and restore resources
- added `OptimizerAlreadyBoundError` and `ModelAlreadyBoundError`
- added `ModuleFromAnotherRunError` to provide accurate messaging when a model is already registered to a different run
- added `__repr__` to `AbstractScheduler`

## Changed
- `MetricCollection` merging operations (`|`) now raise `RepeatedMetricsError` if the two collections share metric names
- `MetricTracker` emits a warning when the metric improvement direction is automatically guessed from initial measurements
- bumped minimum `torch` requirement to `>=2.10.0` which resolves compilation issues on Python 3.14, and consequently removed `pytest.skip` workarounds for Python 3.14 in test suites
- last epoch sorts checkpoint by epoch and not by time of creation
- unwrapped returns original model and not compiled one
- `CheckpointPathManager` evaluates `run_dir` and `model_dir` statically at initialization
- abstracted check for active experiment run ownership to `registering.check_current_run`
- re-architected global `ALL_MODULES` and `ALL_ACTORS` registries to use `WeakKeyDictionary` and `WeakSet` to prevent memory leaks while preserving cross-run safeguards
- removed the `Experiment.run` setter to prevent lifecycle state bypasses
- removed unused `Averager` and `MeanAccumulator` aggregators
- removed unused `min_delta` and `best_is` arguments from `optuna.TrialCallback`
- `optuna.suggest_overrides` now uses full parameter names for repeated values to prevent collisions

## Fixed
- terminated training no longer emits `EndTrainingEvent`
- `GradZScoreNormalizer` now properly skips single-element parameters and zero-variance gradients to prevent division by zero causing `nan` gradients
- `ZStatCriterion` now properly clips only the upper tail (abnormally large values) rather than improperly increasing abnormally small values
- `torcheval` wrapper stores and returns synchronized metric values
- `torcheval` wrapper returns metrics as a named dictionary
- `torchmetrics` wrapper raises error for duplicate metric names
- `HistClipper` and `ParamHistClipper` now construct `ZStatCriterion` and `GradNormClipper` on a per-instance basis to avoid leaking state across instances due to shared class-variable defaults
- `DataLoader.split()` now correctly propagates `pin_memory` and `n_workers` configurations to the resulting sub-loaders instead of dropping them
- `Permutation` now properly uses the modern numpy RNG instead of legacy `np.random.randint` when no seed is provided
- `validate_dataset_length` now correctly looks up `__len__` on the class type to respect Python's dunder resolution rules
- updated `__len__` docstrings for `DataLoader` and `LoaderProtocol` to clarify they return the global batch count
- fixed `reduce()` in aggregators mutating internal cache by returning a copy
- fixed `unregister_actor` raising an error when called outside of an active experiment run
- fixed `PruneCallback` now unconditionally prunes when evaluated against a `None` threshold instead of bypassing and recording the benchmark value
- fixed training loop failing to break immediately if a pre-epoch hook terminates training
- fixed `TqdmLogger` leaking epoch progress bars when training is abruptly terminated
- fixed `MetricMonitor` failing on name disagreements when cross-referencing metrics against objectives
- fixed callback docstrings to correctly attach the `'auto'` behavior explanation to the `best_is` parameter
- removed erroneous fallback to `_get_name` in `MetricExtractor` when resolving metric names
- val_hook is reassigned on bind
- fixed NestedScopeError message arguments
- fixed NoActiveExperimentError class name evaluation
- fixed unformatted string in _validate_batch_size error message
- fixed MetricCollection typechecker Self bound error
- removed previous_runs caching to prevent cross-experiment bleeding
- fixed typo in formula startswith check in objectives
- model.compile is now assigned correctly
- torch.is_initialized is called correctly
- added base lr in PolynomialScheduler
- fixed a run ID collision issue that occurred when starting multiple runs in the same second
- fixed `wandb` metrics logging degradation on subsequent runs by resetting `_defined_metrics`
- fixed a bug where closing an already-failed run threw a spurious warning
- fixed a bug where a run ID with multiple `@` symbols broke path resolution for checkpoints and trackers
- fixed noisy warnings in test suite
- fixed missing docstrings in newly added tracker tests
- fixed a static type checker error (`reportInvalidTypeForm`) in `experimenting.py` by casting to `Any` instead of `cls`
- fixed reference duplication bug in `runners.py` where list multiplication created identical list references for gathered outputs
- fixed `_remove_outer_parentheses` in `objectives.py` erroneously stripping metric name brackets from formulas
- fixed test suite cross-contamination caused by a leaky session-scoped experiment mock
- fixed a bug where short-lived actors reusing memory IDs would falsely appear as already registered
- fixed a bug where `check_current_run` would incorrectly report the current active run's metadata instead of the model's actual owning run
- fixed `ReduceLROnPlateau` so that learning rate reductions correctly accumulate via composition instead of resetting
- fixed order dependence in tests unit -> integration


## [0.1.0rc11] - 31-07-2026

### Fixed
- repr_utils.recursive_representation uses native printing for Torch tensors


## [0.1.0rc10] - 31-07-2026

### Added
- tests to check that memory leaks no longer occur in recursive_representation

### Changed
- recursive_representation now checks visited before dispatching

### Fixed
- memory leak in recursive_representation


## [0.1.0rc9] - 30-07-2026

### Added
- added fn positional argument to MetricCollection and DictLoss class

### Fixed
- fixed buggy check for DistributedParallel


## [0.1.0rc8] -29-05-2026

### [BREAKING CHANGES]
- removed LossBase.__or__ in favor of LossBase.watch with clearer intent.

### Added
- added LossBase.watch
- added JoinMetrics and JoinLossMetrics
- added AverageObjective

### Changed
- get_aggregator in Objective is now abstract.
- the Objective class does not define a default aggregator anymore.
- MetricCollection inherits from AverageObjective.


## [0.1.0rc7] - 2026-05-28

### Added
- added utils.local_ops
- added unit tests for utils.local_ops
- classes created with static_hook_class are now pickleable

### Changed:
- now schedulers and hooks implement monad logic correctly

### Removed:
- some internal classes in schedulers and hooks


## [0.1.0rc6] - 2026-02-23

### Added
- EMA and SWA models
- AbstractAggregator allows for other aggregation of metrics than mean
- AbstractAccumulator and subclasses for different accumulators

### Changed
- checkpoint can now bind with multiple modules
- refactor: module names strictly adhering to:
  1) -ing form for modules containing base logic
  2) plural for modules containing a class and its subclasses
  3) name of the external dependency for binder modules

### [BREAKING CHANGES]
- deleted contrib/swa_utils.py


## [0.1.0rc5] - 2026-01-20

### [BREAKING CHANGES]
- renamed TrackerNotActiveError -> TrackerNotUsedError

### Changed
- more informative and clean logging output and naming for optuna


## [0.1.0rc4] - 2026-01-12

### Added
- added an interval parameter to Trainer add_validation

### Changed
- get model's device index from global settings
- removed redundant get_dataset method

### Fixed
- optuna get_best_trial_value works also with parallelization
- explicitly closing pbar after the last epoch


## [0.1.0rc3] - 2026-01-07

### Changed
- improved documentation and sphinx build configuration.
- new release pipeline.

### Fixed
- corrected an error when calculating the actual number of batches for ddp.


## [0.1.0rc2] - 2025-12-11

### Changed
- simplified TensorBoard tracker.
- checkpointing automatically wraps/unwraps parallelized modules.

### Added
- support for multiprocessing for Experiment class
- support for metrics syncing in distributed settings
- support for distributed samplers
- from_torcheval added to allow syncing of torcheval metrics
- support for Python 3.14
- added optional compile and distributed functionalities to the Model class
- extended test coverage


## [0.1.0rc1] - 2025-11-25

### [BREAKING CHANGES]
- renamed EventDispatcher.register -> EventDispatcher.subscribe
- corrected typo in ModelCreationEvent's architecture_repr variable

### Changed
- now possible to change the maximum depth of the automatic documentation


## [0.1.0b5] - 2025-11-23

### Added
- CHANGELOG.md
- extended README.md
- support for notebooks when using TensorBoard
- support for readable parameter names for optuna
- added last git commit hash when available to run metadata
- architecture.md

### Changed
- README.md location
- repr_utils also provides array shape info
- default uses readable parameter names for optuna
