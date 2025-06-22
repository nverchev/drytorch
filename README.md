![image info](drytorch_logo.png)
Following the Don't Repeat Yourself principles, this library offers:
- Functionalities for a wide range of machine-learning applications.
- Modularity allows you to easily build project-specific classes and data type.
- Decoupling external trackers and loggers from the training cycle. 
- Design uses independent experiment scopes to actively encourage best practice.

### Functionalities:
- Adaptive data loading.
- Metrics with automatic formatting.
- Composable schedulers classes.
- Learning schemes with structured learning rates.
- Automatic metadata extraction.
- Training cycle with hooks.
- Simplified checkpointing.

### Modularity:
- Classes communicate through protocols expressing necessary conditions.
- Classes are build from abstract classes providing an initial implementation.
- Type safety and hints for user data classes thanks to generic annotations.

### Decoupling:
- Event system send notifications to optional external libraries.
- Already implemented trackers (hydra, wandb, tensorboard, ...).
- Only required dependency: PyTorch and NumPy.

### Design:
- Training and evaluating within an experiment scope.
- Discourage dependencies between experiments.
- Prevent accidentally mixing experiments by passing wrong configuration files.

