## [0.1.0b6] - 2025-11-25

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
