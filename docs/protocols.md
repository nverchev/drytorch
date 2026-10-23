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

- **Generalization (<|–):** one protocol extends another and inherits its
  members.

- **Structural Association (–>):** Represents a structural requirement and is often implemented using Dependency Injection.

- **Dependency (..>):** Represents a logical dependency that is not enforced by the protocol but often necessary for its implementation.

- **Members:** those marked with `+` are methods or read-only properties; the
  others are attributes that implementations may reassign.

- **Events:** names in «guillemets» are events. Among the attributes, they
  are emitted when an instance is created; after a method, by the method.

```mermaid
%%{init: {'theme': 'default'}}%%

classDiagram
direction TB

    class RunnerProtocol {
        «RunnerRegistrationEvent»
        +model: ModelProtocol
    }


    class SourceProtocol {
        +name: str
        +objective: ObjectiveProtocol
        +computed_metrics: Mapping[str, float]
        +last_run_epoch: int
        +__call__(store_outputs: bool) «IterateBatchEvent» «MetricEvent»
    }

    class ObjectiveProtocol["ObjectiveProtocol[Output, Target]"] {
        +update(outputs: Output, targets: Target)
        +compute() Mapping[str, torch.Tensor] | torch.Tensor | None
        +reset()
    }

    class LossProtocol["LossProtocol[Output, Target]"] {
        +forward(outputs: Output, targets: Target) torch.Tensor
    }


    class TestProtocol {
        +__call__(store_outputs: bool) «StartTestEvent» «EndTestEvent»
    }

    class TrainerProtocol["TrainerProtocol[Input, Target, Output]"] {
        +model: ModelProtocol[Input, Output]
        +learning_schema: LearningProtocol
        +terminated: bool
        +save_checkpoint()
        +load_checkpoint(epoch: int)
        +terminate_training(reason: str) «TerminatedTrainingEvent»
        +train(n_epochs: int) «StartTrainingEvent» «StartEpochEvent» «EndEpochEvent» «EndTrainingEvent»
        +update_learning_rate(base_lr, scheduler) «LearningRateEvent»
    }

    class LearningProtocol {
        +base_lr: float | Mapping[str, float]
        +gradient_op: GradientOpProtocol
        +optimizer_cls: type[torch.optim.Optimizer]
        +optimizer_defaults: Mapping[str, Any]
        +scheduler: SchedulerProtocol
    }

    class SchedulerProtocol {
        +__call__(base_lr, epoch) float
    }

    class GradientOpProtocol {
        +__call__(params: Iterable[torch.nn.Parameter])
    }

    class ModelProtocol["ModelProtocol[Input, Output]"] {
        «ModelRegistrationEvent»
        +epoch: int
        +module: torch.nn.Module
        +device: torch.device
        +mixed_precision: torch.dtype | None
        +name: str
        +__call__(inputs: Input) Output
        +bind_optimizer(optimizer: Optimizer)
        +increment_epoch() «IncrementEpochEvent»
        +load_state(epoch: int)
        +post_batch_update()
        +post_epoch_update()
        +save_state()
    }

    class CheckpointProtocol {
        +bind_model(model: ModelProtocol)
        +bind_module(name: str, module: torch.nn.Module)
        +bind_optimizer(optimizer: Optimizer)
        +save() «SaveModelEvent»
        +load(epoch: int) int «LoadModelEvent»
    }

    class HookProtocol {
        +last_run_epoch: int
        +trainer: TrainerProtocol
        +__call__()
        +bind_trainer(trainer: TrainerProtocol)
    }

    class LoaderProtocol["LoaderProtocol[Input, Target]"] {
        +batch_size: int | None
        +sampler: torch.utils.data.Sampler | Iterable
        +dataset: torch.utils.data.Dataset
        +__iter__() Iterator[Input, Target]
        +__len__() int
    }

    RunnerProtocol ..> LoaderProtocol : supports
    LearningProtocol --> GradientOpProtocol : defines
    LearningProtocol --> SchedulerProtocol : follows
    CheckpointProtocol --> ModelProtocol : binds to
    RunnerProtocol --> ModelProtocol : runs
    SourceProtocol --> ObjectiveProtocol : computes
    TrainerProtocol --> LearningProtocol : implements
    TrainerProtocol --> LossProtocol : minimizes
    TrainerProtocol --> ModelProtocol : updates
    HookProtocol --> TrainerProtocol : binds to
    RunnerProtocol <|-- SourceProtocol : extends
    SourceProtocol <|-- TrainerProtocol : extends
    SourceProtocol <|-- TestProtocol : extends
    ObjectiveProtocol <|-- LossProtocol : extends

```
