## [Unreleased] -

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
