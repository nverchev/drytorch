## [0.1.0rc2] - 2025-12-11

### Changed
- Simplified TensorBoard tracker.

### Added
- support for multiprocessing for Experiment class
- support for metrics syncing in distributed settings
- support for distributed samplers
- from_torcheval added to allow syncing of torcheval metrics

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
- add last git commit hash when available to run metadata
- architecture.md

### Changed
- README.md location
- repr_utils also provides array shape info
- default uses readable parameter names for optuna
