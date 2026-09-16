## [0.1.0] - 14-09-2026

## [BREAKING CHANGES]
- protocol attributes are read-only properties: `batch_size`, `dataset`, `sampler`, `mixed_precision`, `learning_schema`, `model` in `MonitorProtocol` and `TrainerProtocol`, and the attributes of `LearningProtocol`
- replaced `AbstractAggregator` and `TorchAverager` with `Aggregator`, a container whose `merge` combines accumulators by name; `reduce` and `all_reduce` are removed, and the accumulators are reduced by their owner
- `Objective._get_aggregator` is replaced by `Objective._get_accumulator_cls`; `Objective` creates the accumulators and reduces them
- `Objective.merge_state` accepts any objective and raises `AccumulatorMismatchError` if the two objectives use different accumulators
- `AbstractAccumulator.merge` takes `other: Any`; subclasses annotate `other` with their own class
- runners register as actors of their model when created instead of when first called
- `Hook.parameters` and `StaticHook.parameters` are read-only mappings copied at creation; `TrainerHook` subclasses implement `_get_name` and `_get_parameters`
- `Trainer` takes `val_loader`, `val_name`, `validate_every`, `pre_epoch_hooks` and `post_epoch_hooks` at creation, or collects them with `Trainer.builder`; `Trainer.add_validation` is removed
- `Trainer.pre_epoch_hooks` and `Trainer.post_epoch_hooks` are `HookChain`; `TrainerProtocol.validation` is a read-only property
- merged `CompositionalLoss` into `LossBase` and renamed it `LossWithMetrics`, which is not abstract
- renamed `JoinMetrics` to `JoinedMetrics` and `JoinLossMetrics` to `JoinedLoss`
- replaced `LossBase.watch` with the function `join`, which returns a new loss instead of changing the original
- `JoinedMetrics` takes any number of objectives following `ObjectiveProtocol`; its `update` returns `None` and `compute` raises `RepeatedMetricsError` on repeated names; `JoinedLoss` takes any `LossProtocol`
- `MetricCollection.fn` and `MetricCollection.named_fn` are read-only

- `MetricTracker.metric_name` and `history` are read-only, `best_is` is `Final` and `best_value` has no setter; `resolved_best_is` is the direction in use, and `resolve_metric_name` and `resolve_best_is` set the name and the direction while they are unknown
- library classes match protocols structurally instead of inheriting them
- `Model.epoch` and `ModelProtocol.epoch` are read-only; `CheckpointProtocol.load` returns the loaded epoch, which `Model.load_state` sets
- `ModelProtocol` requires `bind_optimizer`, `load_state` and `save_state`; `ModelOptimizer` and `Trainer` bind, load and save through the model
- `CheckpointPathManager.epoch_dir` is replaced by `get_epoch_dir`; `get_model_state_path` and `get_optimizer_state_path` take the epoch
## Added
- added experiment pause and continue feature to allow pausing training sessions cleanly
- added specific behaviour for trackers on pause to safely stash and restore resources
- added `AccumulatorMismatchError`
- added `OptimizerAlreadyBoundError` and `ModelAlreadyBoundError`
- added `ModuleFromAnotherRunError` to provide accurate messaging when a model is already registered to a different run
- added `__repr__` to `AbstractScheduler`
- added `PauseExperimentEvent` and `ContinueExperimentEvent` to test workflows
- added `torch_compile` and `distribute` to `EMAModel` and `SWAModel`
- added `HookChain`, `TrainerBuilder` and `hooks.evaluation_hook`
- added `registering.RegisteringMeta`, the metaclass registering runners as actors of their model once created
- added `objectives.join`, which joins objectives into one of the same kind as the first

- added `IncrementEpochEvent`, emitted by `Model.increment_epoch`
- added `DirectionNotResolvedError`, raised by `MetricTracker.resolved_best_is` before the direction is known
- moved `ModuleProtocol` from `lib.models` to `core.protocols`, where it is documented with the other protocols
## Changed
- ordered arguments across signatures and docstrings to weakly prefer alphabetical order within positional and named argument groups
- `AbstractCheckpoint` emits `SaveModelEvent` and `LoadModelEvent` after the save or load succeeds
- enforced explicit returns at the end of all functions and methods
- renamed `DistributedDatasetNotDivisibleWarning` attribute `num_processes` to `n_processes`
- normalized all docstring argument descriptions to start in lowercase and `__init__` docstrings to `"""Initialize."""`
- immutable `Experiment.tags`, `ModelOptimizer.base_lr`, `LearningSchema.base_lr`, `LearningSchema.optimizer_defaults`, `MetadataManager.named_metadata`, `EventDispatcher.named_trackers`, `checkpoints.SAFE_GLOBALS`, and transform scheduler parameters
- added `IterateBatchEvent.refresh` to update metrics without advancing progress and documented callback ordering and mutability
- widened parameter types: `extend_default_trackers` accepts `Iterable[Tracker]`, and `MissingParamError`/`RepeatedMetricsError` accept `Sequence[str]`
- `ModelRunner.outputs_list` and `Aggregator.accumulators` are exposed as read-only properties backed by private containers
- `LocalCheckpoint.par_dir` is public for metadata recording
- `MetadataDumper.dump` is public for dumping metadata
- declared `Final` attributes at class level across models, runners, schedulers, trainers, objectives, checkpoints, hooks, loaders, trackers, and exception classes
- unified abstract classes on `abc.ABC` and removed redundant `self: Self` method annotations
- `MetricCollection` merging operations (`|`) now raise `RepeatedMetricsError` if a metric name maps to different functions
- `MetricCollection` and its subclasses are invariant in `Output` and `Target`
- moved `SupportsSync` from `runners` to `protocols` as `SyncProtocol`
- loss formulas follow the written operations (`[A] - 3`, `[A] / [B]`, `[A]^-2`) and use parentheses only where needed
- `MetricTracker` emits a warning when the metric improvement direction is automatically guessed from initial measurements
- bumped minimum `torch` requirement to `>=2.10.0` which resolves compilation issues on Python 3.14, and consequently removed `pytest.skip` workarounds for Python 3.14 in test suites
- last epoch sorts checkpoint by epoch and not by time of creation
- unwrapped returns original model and not compiled one
- `Model` flags renamed: `should_compile` → `torch_compile`, `should_distribute` → `distribute`
- `CheckpointPathManager` evaluates `run_dir` and `model_dir` statically at initialization
- abstracted check for active experiment run ownership to `registering.check_current_run`
- re-architected global `ALL_MODULES` and `ALL_ACTORS` registries to use `WeakKeyDictionary` and `WeakSet` to prevent memory leaks while preserving cross-run safeguards
- `optuna.suggest_overrides` now uses full parameter names for repeated values to prevent collisions
- renamed `YamlDumper` to `MetadataDumper` in `drytorch.trackers.yaml`
- consolidated integration tracker tests into one module per tracker
- moved run id and recording logic from `Run` to `Experiment`; `Run` only stores them

- `TrainerProtocol.update_learning_rate` accepts learning rates by module name and defaults both arguments to `None`, like `Trainer`
- `MetricMonitor.metric_tracker` and `MetricExtractor.optional_monitor` are `Final`; `MetricMonitor.history` is a `Sequence`
- `MetricMonitor` takes an automatic `best_is` from the monitored metric when the metric states its direction, so the direction is recorded
- `RegisteringMeta` derives from `abc.ABCMeta`, as runners no longer inherit protocols
## Removed
- removed `ModelCaller`; `ModelRunner` is the base class of the runners
- removed `HookRegistry`; hooks are passed to `Trainer` or added to `TrainerBuilder`
- removed the unused `fun` attribute from `Metric` and `DictLoss`
- removed the `Experiment.run` setter to prevent lifecycle state bypasses
- removed unused `Averager` and `MeanAccumulator` aggregators
- removed unused `min_delta` and `best_is` arguments from `optuna.TrialCallback`
- removed preferential dataclass apply branch in `apply`
- removed unused `_removed_start` attribute and relative documentation in `BasePlotter`

- removed `TrainerProtocol.objective`, which no code used
- removed `ModelProtocol.checkpoint`; the model saves and loads its own state
## Fixed
- `Trainer.update_learning_rate` emits `LearningRateEvent` only after the update succeeds
- combining losses whose metrics share a name with different functions raises `RepeatedMetricsError` instead of dropping one function
- loss formulas keep the parentheses of grouped terms
- multiplying or dividing a loss by a negative number flips its direction
- a composed loss reports its own value when a metric has the same name
- `LossWithMetrics` without a formula records its name in brackets instead of `()`
- `JoinedLoss.forward` also updates the joined metrics, which training never updated
- typed actor metadata in `extract_metadata` and `ActorRegistrationEvent.metadata` as `Mapping[str, Any] | str`
- `recursive_apply` falls back to type name for callables that are not functions
- `MatPlotter` creates figures with `plt.figure()` for managed pyplot interactive display
- `MatPlotter` clears and rebuilds layout grid when new metrics appear mid-run
- `MatPlotter.close()` closes all managed figures and clears color mappings on tracker teardown
- `MatPlotter` uses `ax.plot` with marker styling for single points to autoscale axes properly
- `MatPlotter` disperses continuous colormap colors across sources via successive interval bisection
- `MatPlotter` raises `TrackerError` when palette has insufficient distinct colors
- `MatPlotter` assigns metric titles to each subplot
- `BasePlotter`, `MatPlotter`, and `PlotlyPlotter` synchronize figure display after all metrics are plotted via `_display_plot`
- `PlotlyPlotter` renders figures on an explicit `plot` call and does no work during training
- `HydraLink` raises `TrackerError` when `HydraConfig` is unset and hydra has not started
- `HydraLink` resolves symlink collisions when runs share timestamps or following clean-up
- `HydraLink.clean_up()` uses `mkdir=False` keyword argument
- `MetadataDumper` scopes sequence representers to `DryTorchDumper` to avoid mutating global PyYAML representers
- `MetadataDumper` reads sequence length limits dynamically from module attributes
- `MetadataDumper` sorts set and frozenset elements to ensure deterministic serialization across runs
- added `-> None` return type annotation to `MetadataDumper.__init__`
- `BasePlotter` raises `TrackerError` when epoch and value counts do not match
- `BuiltinLogger` keys metric format arguments by position to prevent collisions with `desc` and `_value`
- `set_formatter` validates style upfront regardless of registered handlers
- `CSVDumper` writes rows matching column headers and raises `TrackerError` on metric changes
- `CSVDumper.clean_up()` resets active sources so consecutive runs write headers
- `CSVDumper` and `SQLConnection` release stashed state on `close()`
- `Dumper.close()` releases stashed paused state
- `DryTorchFormatter` formats messages via `formatMessage()` instead of mutating `_style._fmt`
- `TqdmLogger` closes active progress bars on pause and replaces incomplete epoch bars cleanly
- `TqdmLogger.clean_up()` invokes `super().clean_up()`
- `TensorBoard` catches `OSError` on server launch failure to raise `TrackerError`
- `TensorBoard` reuses running server across runs for the same directory and terminates it on close or directory change
- `TensorBoard` avoids flushing on every metric event so `SummaryWriter` honors `max_queue_size` and `flush_secs`
- `Wandb` relies on base tracker cleanup on stop and removes redundant scope guard in `MetricEvent`
- `Wandb` stashes `_defined_metrics` across pause and continue to prevent redundant metric definitions
- `SQLConnection` disposes its engine on `close()` rather than between runs
- `SQLConnection` reuses existing experiment rows and avoids duplicating tags across runs
- `MetricTracker.filtered_value` raises `ResultNotAvailableError` when history is empty
- setting `MAX_REPR_SIZE` to 0 now records the element count without the elements, as it already did for dictionaries
- batches carrying strings or bytes beside their tensors no longer stop training
- `local_ops` validates targets upfront to prevent partial directory operations
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
- `DefaultName` returns descriptor on class access
- numpy arrays of four or more dimensions now display values in representations
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
- `torch_compile=True` now compiles the module (it had no effect before), and the default is now `False`
- `Model` skips DDP at inference, so one rank does not wait for the others
- `EMAModel` and `SWAModel` use mixed precision at inference
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
- a failed checkpoint load keeps the epoch of the model


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
