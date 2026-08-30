## [0.1.0] - 30-08-2026

## Changed
- last epoch sorts checkpoint by epoch and not by time of creation

## Fixed
- val_hook is reassigned on bind
- fixed NestedScopeError message arguments
- fixed typo in formula startswith check in objectives
- model.compile is now assigned correctly
- torch.is_initialized is called correctly
- added base lr in PolynomialScheduler


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
