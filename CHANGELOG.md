# [0.1.0] - 01-10-2026

## [BREAKING CHANGES]
- the `device` argument of `Model` and its subclasses defaults to `'auto'` instead of `None`
- `take_from_dataset` returns the indices of the samples it takes, then the batch, so that random samples can be told apart
- `mixed_precision` takes the autocast precision (`torch.float16` or `torch.bfloat16`) or `None` instead of a boolean; the gradient scaler is enabled only for `torch.float16`
- renamed `drytorch.lib.aggregators` to `drytorch.lib.accumulators`
- settings, options, and hyperparameters across constructors and functions are keyword-only (after `*`); positional arguments are restricted to primary target objects
- standardized model argument to `module` across all `Model` constructors, replacing `torch_module`
- positional-only arguments (before `/`) are restricted to generic or obvious names (`module`, `fn`, `scheduler`, `obj`, `calculator`), while arguments with meaningful names (`dataset`, `device`, `config`) can be passed both positionally and by keyword
- renamed AbstractScheduler.bind to compose; a composed scheduler is the transformation wrapping the scheduler it modifies
- the trainer binds its hooks when it is created and calls them without arguments; a hook belongs to one trainer and `HookChain.execute` takes no argument
- hooks are objects implementing `HookProtocol` and never receive the trainer: a hook subclasses `TrainerHook` and reads `self.trainer`, so functions are no longer wrapped into hooks
- removed `Hook`, `StaticHook`, `static_hook_class` and `saving_hook`, replaced by the `SavingHook` and `EvaluationHook` classes and by writing a `TrainerHook` subclass
- removed `call_every` and `evaluation_hook`, replaced by `EvaluationHook` and condition classes (such as `Every` and `After`) passed to `TrainerHook`
- hooks subclass `TrainerHook` and are generic in their `when` condition; callbacks specialize the condition and expose typed condition properties
- monitoring is redesigned around `Watch`, `Monitor`, and condition classes (`Condition`, `Always`, `After`, `AtEpochs`, `Cooldown`, `Every`, `OnTermination`, `Plateau`, `WorseThan`, `AndCondition`, `OrCondition`, `NotCondition`) supporting `&`, `|`, `~` operators
- renamed `EarlyStoppingCallback` to `EarlyStopping`, `PruneCallback` to `Pruning` and `ChangeSchedulerOnPlateauCallback` to `ChangeScheduler`
- `patience` counts the runs of the followed source, not epochs
- renamed the actor of a model to runner: `ModelRunner` is `Runner`, `ActorProtocol` is `RunnerProtocol`, `register_actor` and `unregister_actor` are `register_runner` and `unregister_runner`, and `ActorRegistrationEvent` is `RunnerRegistrationEvent` with `runner_name` and `runner_ts`
- renamed `MonitorProtocol` to `SourceProtocol`, after the `source_name` of the events, and merged `ObjectiveMonitorProtocol` into it: a source states the objective computing its metrics and inherits `RunnerProtocol`
- merged `ModelRunnerWithObjective` and `ModelRunnerWithLogs` into `Source`, which computes an objective and emits `MetricEvent`; `ModelRunner` no longer reports metrics
- the monitoring hooks take `watch`, a `Watch` stating the source, the metric and its direction, instead of `metric`, `source`, `best_is` and `filter_fn`; without a source, they follow the trainer
- protocol attributes are read-only properties: `batch_size`, `dataset`, `sampler`, `mixed_precision`, `learning_schema`, `model` in `SourceProtocol` and `TrainerProtocol`, and the attributes of `LearningProtocol`
- replaced `AbstractAggregator` and `TorchAverager` with `Aggregator`, a container whose `merge` combines accumulators by name; `reduce` and `all_reduce` are removed, and the accumulators are reduced by their owner
- `Objective._get_aggregator` is replaced by `Objective._get_accumulator_cls`; `Objective` creates the accumulators and reduces them
- `Objective.merge_state` accepts any objective and raises `AccumulatorMismatchError` if the two objectives use different accumulators
- `AbstractAccumulator.merge` takes `other: Any`; subclasses annotate `other` with their own class
- runners register as actors of their model when created instead of when first called
- `TrainerHook` subclasses implement `_get_name` and `_get_parameters`
- `Trainer` takes `pre_epoch_hooks` and `post_epoch_hooks` at creation, or collects them with `Trainer.builder`; it no longer holds a validation: `Trainer.add_validation` is removed, and a validation runs as an `EvaluationHook`
- `Trainer.pre_epoch_hooks` and `Trainer.post_epoch_hooks` are `HookChain`
- `SourceProtocol.computed_metrics` is empty when the source has not run at the current epoch of the model
- merged `CompositionalLoss` into `LossBase` and renamed it `LossWithMetrics`, which is not abstract
- renamed `JoinMetrics` to `JoinedMetrics` and `JoinLossMetrics` to `JoinedLoss`
- replaced `LossBase.watch` with the function `join`, which returns a new loss instead of changing the original
- `JoinedMetrics` takes any number of objectives following `ObjectiveProtocol`; its `update` returns `None` and `compute` raises `RepeatedMetricsError` on repeated names; `JoinedLoss` takes any `LossProtocol`
- `MetricCollection.fn` and `MetricCollection.named_fn` are read-only
- library classes match protocols structurally instead of inheriting them
- `Model.epoch` and `ModelProtocol.epoch` are read-only; `CheckpointProtocol.load` returns the loaded epoch, which `Model.load_state` sets
- `ModelProtocol` requires `bind_optimizer`, `load_state` and `save_state`; `ModelOptimizer` and `Trainer` bind, load and save through the model
- `CheckpointPathManager.epoch_dir` is replaced by `get_epoch_dir`; `get_model_state_path` and `get_optimizer_state_path` take the epoch
- `ModelOptimizer.base_lr` and `StatsCollector.active` are read-only; `ModelOptimizer.update_learning_rate` sets the learning rates
- `Trainer` raises `ValueError` when the loss is better when higher
- removed `MetricTracker`, `MetricExtractor` and `MetricMonitor`, replaced by `Watch`, `Monitor` and conditions in the new `lib.monitoring`
- plotters no longer draw during training: `BasePlotter.plot` renders the figures of a model on request, and its arguments after `model_name` are keyword-only
- dropped the `Grad` prefix from gradient operations: renamed `GradNormClipper`, `GradValueClipper`, `GradParamNormalizer`, and `GradZScoreNormalizer` to `NormClipper`, `ValueClipper`, `ParamNormalizer`, and `ZScoreNormalizer`

## Added
- `BasePlotter.create_figures` returns the figure of each metric of a model without rendering it, to show or save
- plotters are dumpers: `BasePlotter.save_figures` saves the figure of each metric in the folder of the current run, as PNG (`MatPlotter`) or HTML (`PlotlyPlotter`)
- `MatPlotter.plot_grid` renders the metrics of a model as subplots of one figure
- `Wandb.log_at_epoch` logs the values of other events against the epoch of a model, on the axis of its metrics
- `TensorBoard` titles its server after the folder of the experiment and logs the title and address at the new level `SERVER_LOG_LEVEL` (26)
- added `SyncedObjectiveProtocol` and `SyncedLossProtocol` in `core.protocols`; `join` and the two wrappers state that their result synchronizes
- added the tutorials `data.md`, `models.md`, `training.md`, `learning.md`, `custom_events.md`, and `distributed_training.md`
- the `objectives.md`, `experiments_and_runs.md`, and `trackers.md` tutorials cover custom accumulators, run utilities and logging formatters
- the README and the documentation home page point to the project template `drytorch-template`
- added `TestProtocol`
- added `HookNotBoundError` and `HookAlreadyBoundError`, raised when a hook is called before a trainer binds it or is given to a second trainer
- added `SourceRunAfterHookError`, raised when a hook runs before the source it follows
- added `ValidationAlreadyAddedError`, raised when a builder is given a second validation
- added `NoValidationWarning`, emitted by `TrainerBuilder.watch` when it follows the training because no validation was added
- added `last_run_epoch` to `SourceProtocol` and `Source`
- added `TerminateTraining` and `monitoring.ALWAYS`, the default condition of the hooks
- added `TrainerBuilder.watch`, following the loss of the validation by default, and `training=True` to follow the trainer
- added `TrainerBuilder.add_early_stopping`, `add_pruning`, `add_reduce_lr_on_plateau`, `add_restart_schedule_on_plateau` and `add_saving`
- added `TrainerBuilder.validation`; `TrainerBuilder.add_validation` adds the validation to the post-epoch hooks in call order and raises when called twice
- added experiment pause and continue feature to allow pausing training sessions cleanly
- added specific behaviour for trackers on pause to safely stash and restore resources
- added `AccumulatorMismatchError`
- added `OptimizerAlreadyBoundError` and `ModelAlreadyBoundError`
- added `ModuleFromAnotherRunError` to provide accurate messaging when a model is already registered to a different run
- added `__repr__` to `AbstractScheduler`
- added `PauseExperimentEvent` and `ContinueExperimentEvent` to test workflows
- added `torch_compile` and `distribute` to `EMAModel` and `SWAModel`
- added `HookChain`, `TrainerBuilder` and `EvaluationHook`
- added `registering.RegisteringMeta`, the metaclass registering runners as actors of their model once created
- added `objectives.join`, which joins objectives into one of the same kind as the first
- added `IncrementEpochEvent`, emitted by `Model.increment_epoch`
- added `higher_is_better` to `from_torcheval`, stated by the wrapped metric
- the wrapper from `from_torchmetrics` keeps the direction of the composed metric
- the run registry records a session for each start and resumption of a run, with its timestamp, git commit and drytorch version
- `Evaluation`, the public evaluation source running the model in inference mode, base of `Validation` and `Test`
- added `n_parameters` and `n_trainable_parameters` to `ModelRegistrationEvent`, counted when the model is registered
- added `indices` and `collate_fn` to `take_from_dataset`, which take the given samples in the given order and combine them with the given function
- added `DataLoader.fold`, which splits a loader into the training and validation loaders of a k-fold cross-validation

## Changed
- `Source` is generic in any objective, not only in a loss
- `Test` and `TestProtocol` derive from `Evaluation` and `SourceProtocol` instead of from the validation role
- renamed `EventDispatcher.clean_up` to `close_all`, which closes its trackers, unlike `Tracker.clean_up`
- updated `STYLE.md` with the 3-tier argument convention: positional-only for generic/obvious target objects, positional-or-keyword for semantically named targets, and keyword-only for settings and options, requiring multiple positional-only arguments to have the same type
- regrouped `core/protocols.py` into components, running and control, and capabilities
- moved `ModuleProtocol` from `core.protocols` to `lib.models`, where the models that take a module document it
- `Plateau` and `WorseThan` evaluate only on fresh records computed for the current model epoch
- `TrialCallback` reports to optuna only on fresh records computed for the current model epoch
- each termination hook states its own reason when it terminates the training
- an object recorded more than once is represented once and referred to afterwards, so the metadata file names it with an anchor such as `&Model` and refers to it with `*Model`
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
- `RegisteringMeta` derives from `abc.ABCMeta`, as runners no longer inherit protocols
- `MetadataDumper` writes the parameter counts with the architecture, and `BuiltinLogger` logs how many parameters a registered model has and how many are trainable
- the `matplotlib` optional dependency requires version 3.11.0 or later

## Removed
- `Diagnostic` and `EvaluationMixin`; use `Evaluation`
- ComposedScheduler, and parameters and base_scheduler_name from the schedulers
- removed the public `CreatedAtProtocol`, replaced by the private `_CreatedAtProtocol` in `core/tracking`
- removed `CallEvery`, `call_every` and `evaluation_hook`
- removed `MetricTracker`, `MetricExtractor` and `MetricMonitor`
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
- removed the `start`, `metric_names`, `model_names` and `source_names` arguments of `BasePlotter`, `MatPlotter` and `PlotlyPlotter`, which only the updates during training used

## Fixed
- `Model` defaults to the CPU when torch is built for an accelerator but no device is available
- `FolderAlreadyExistsError` shows its message, which the empty template of the base class discarded
- `clone_experiment_data`, `rename_experiment_data` and `delete_experiment_data` raise `ExperimentNotFoundError` when no folder holds the experiment, instead of doing nothing
- in distributed runs, only the main process checks whether a new run id is already recorded; the other processes join its run instead of raising `RunAlreadyRecordedError` when the main process recorded it first
- removed the mention of a browser from the `TensorBoard` docstrings: the tracker no longer opens one
- trackers are closed when their experiment is released, and at the latest when the process ends, so the TensorBoard server stops, stashed wandb runs finish and the SQL engine is disposed
- the TensorBoard server no longer holds the output of the process that started it, which kept shells and CI jobs waiting
- the wrapper from `from_torchmetrics` synchronizes its metrics across processes, instead of leaving them unsynchronized in distributed training
- `from_torcheval` accepts a metric whatever type it computes, instead of only one computing both a tensor and a dictionary
- `LearningSchema.adam`, `adam_w`, `r_adam` and `sgd` accept learning rates by submodule, as the class and the trainer already did
- composed schedulers record each transformation with its own settings and in order
- `Trainer.update_learning_rate` emits `LearningRateEvent` only after the update succeeds
- the wrapper from `from_torchmetrics` reports its value under its name, `Loss`, so monitoring it without naming the metric works
- combining losses whose metrics share a name with different functions raises `RepeatedMetricsError` instead of dropping one function
- loss formulas keep the parentheses of grouped terms
- multiplying or dividing a loss by a negative number flips its direction
- a composed loss reports its own value when a metric has the same name
- `LossWithMetrics` without a formula records its name in brackets instead of `()`
- `JoinedLoss.forward` also updates the joined metrics, which training never updated
- typed actor metadata in `extract_metadata` and `ActorRegistrationEvent.metadata` as `Mapping[str, Any] | str`
- `recursive_apply` falls back to type name for callables that are not functions
- `MatPlotter` uses `ax.plot` with marker styling for single points to autoscale axes properly
- `MatPlotter` disperses continuous colormap colors across sources via successive interval bisection
- `MatPlotter` raises `TrackerError` when palette has insufficient distinct colors
- `MatPlotter` assigns metric titles to each subplot
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
- `Monitor.value` raises `ResultNotAvailableError` when history is empty
- setting `MAX_REPR_SIZE` to 0 now records the element count without the elements, as it already did for dictionaries
- batches carrying strings or bytes beside their tensors no longer stop training
- `local_ops` validates targets upfront to prevent partial directory operations
- terminated training no longer emits `EndTrainingEvent`
- `ZScoreNormalizer` now properly skips single-element parameters and zero-variance gradients to prevent division by zero causing `nan` gradients
- `ZStatCriterion` now properly clips only the upper tail (abnormally large values) rather than improperly increasing abnormally small values
- `torcheval` wrapper stores and returns synchronized metric values
- `torcheval` wrapper returns metrics as a named dictionary
- `torchmetrics` wrapper raises error for duplicate metric names
- `HistClipper` and `ParamHistClipper` now construct `ZStatCriterion` and `NormClipper` on a per-instance basis to avoid leaking state across instances due to shared class-variable defaults
- `DataLoader.split()` now correctly propagates `pin_memory` and `n_workers` configurations to the resulting sub-loaders instead of dropping them
- `Permutation` now properly uses the modern numpy RNG instead of legacy `np.random.randint` when no seed is provided
- `validate_dataset_length` now correctly looks up `__len__` on the class type to respect Python's dunder resolution rules
- updated `__len__` docstrings for `DataLoader` and `LoaderProtocol` to clarify they return the global batch count
- `DefaultName` returns descriptor on class access
- numpy arrays of four or more dimensions now display values in representations
- fixed `reduce()` in aggregators mutating internal cache by returning a copy
- fixed `unregister_actor` raising an error when called outside of an active experiment run
- fixed training loop failing to break immediately if a pre-epoch hook terminates training
- fixed `TqdmLogger` leaking epoch progress bars when training is abruptly terminated
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
