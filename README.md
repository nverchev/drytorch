# dry_torch

This library provides abstraction for machine learning project that 
interfaces with PyTorch.

## Philosophy
In spirit with the Don't Repeat Yourself principle, this library implements 
common functionalities with minimal requirements to meet the project-specific
necessities.

### The functionalities include:
- adaptive data loading 
- metrics with automatic formatting
- composable schedulers classes
- learning schemes with structured learning rates
- automatic metadata extraction
- training cycle with hooks
- simplified checkpointing

### The flexibility is obtained through:
- Classes communicate through protocols expressing necessary conditions.
- Classes are build from abstract classes providing an initial implementation.
- Generic variables ensure type safety for user classes throughout the project.

### Interfaces to external loggers and trackers:
- Event system send notifications to external classes (hydra, wandb, ...)
- Optional dependencies for the interfaces
- Easy to maintain

### Principled:
- Experimenting and monitoring only possible within an experiment scope
- Discourage dependencies between experiments
- Prevent accidentally mixing experiment

