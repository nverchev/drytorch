# Types and Protocols

## Core Types
Throughout drytorch, generic variables must satisfy these constraints:

- `Input` and `Target`: `torch.Tensor` | `MutableSequence[torch.Tensor]` | `NamedTuple`
- `Output`: no constraints
- `Data`: 2-tuple where both elements follow `Input`/`Target` constraints

**Note**: The notation for the generic variables has been simplified to ignore
subtype relationships.

## Diagram

The following diagram maps the dependencies between the core interfaces using UML-style notation:

- **Refinement (<|–):** Indicates one protocol extends or refines another (e.g.,
  an inheritance relationship).

- **Structural Association (–>):** Represents a structural requirement and is often implemented using Dependency Injection.

- **Dependency (..>):** Represents a logical dependency that is not enforced by the protocol but often necessary for its implementation.

```mermaid

classDiagram
direction TB
    class LoaderProtocol["LoaderProtocol[Input, Target]"] {
        batch_size: int | None
        sampler: torch.utils.data.Sampler | Iterable
        dataset: torch.utils.data.Dataset
        +__iter__() Iterator[Input, Target]
        +__len__() int
    }

    class ModuleProtocol["ModuleProtocol[Input, Output]"] {
        +forward(inputs: Input) Output
    }

    class ModelProtocol["ModelProtocol[Input, Output]"] {
        module: torch.nn.Module
        epoch: int
        checkpoint: CheckpointProtocol
        mixed_precision: bool
        +name: str
        +__call__(inputs: Input) Output
        +increment_epoch()
        +update_parameters()
    }

    class CheckpointProtocol {
        +bind_model(model: ModelProtocol)
        +bind_optimizer(optimizer: Optimizer)
        +save()
        +load(epoch: int)
    }

    class SchedulerProtocol {
        +__call__(base_lr, epoch) float
    }

    class GradientOpProtocol {
        +__call__(params: Iterable[torch.nn.Parameter])
    }

    class LearningProtocol {
        optimizer_cls: type[torch.optim.Optimizer]
        base_lr: float | dict[str, float]
        scheduler: SchedulerProtocol
        optimizer_defaults: dict[str, Any]
        gradient_op: GradientOpProtocol
    }

    class ObjectiveProtocol["ObjectiveProtocol[Output, Target]"] {
        +update(outputs: Output, targets: Target)
        +compute() Mapping[str, torch.Tensor] | torch.Tensor | None
        +reset()
    }

    class LossProtocol["LossProtocol[Output, Target]"] {
        +forward(outputs: Output, targets: Target) torch.Tensor
    }

    class MonitorProtocol {
        model: ModelProtocol
        +name: str
        +computed_metrics: Mapping[str, float]
    }

    class TrainerProtocol["TrainerProtocol[Input, Target, Output]"] {
        model: ModelProtocol[Input, Output]
        learning_schema: LearningProtocol
        objective: LossProtocol[Output, Target]
        validation: MonitorProtocol | None
        +terminated: bool
        +save_checkpoint()
        +load_checkpoint(epoch: int)
        +terminate_training(reason: str)
        +train(num_epochs: int)
        +update_learning_rate(base_lr, scheduler)
    }

    CheckpointProtocol <--> ModelProtocol : saves / binds with
    ModelProtocol --> ModuleProtocol : wraps
    LearningProtocol --> GradientOpProtocol : contains
    LearningProtocol --> SchedulerProtocol : contains
    MonitorProtocol ..> LoaderProtocol : often gets data from
    MonitorProtocol --> ModelProtocol : evaluates
    MonitorProtocol ..> ObjectiveProtocol : typically according to
    TrainerProtocol --> LearningProtocol : follows
    TrainerProtocol ..> LoaderProtocol : often gets data from
    TrainerProtocol --> LossProtocol : optimizes
    TrainerProtocol --> ModelProtocol : trains
    TrainerProtocol --> MonitorProtocol : can be validated by
    MonitorProtocol <|-- TrainerProtocol : refines
    ObjectiveProtocol <|-- LossProtocol : refines

```
