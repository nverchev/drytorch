"""Support for optuna."""

from collections.abc import Callable, Sequence
from typing import Any, Final, Generic, Literal, TypeVar

import optuna

from omegaconf import DictConfig

from drytorch.core import exceptions
from drytorch.core import protocols as p
from drytorch.lib import hooks


_Target_contra = TypeVar(
    '_Target_contra', bound=p.TargetType, contravariant=True
)
_Output_contra = TypeVar(
    '_Output_contra', bound=p.OutputType, contravariant=True
)


class TrialCallback(Generic[_Output_contra, _Target_contra]):
    """Implements pruning logic for training models.

    Attributes:
        monitor: Monitor instance
        trial: Optuna trial.
    """

    def __init__(
        self,
        trial: optuna.Trial,
        filter_fn: Callable[[Sequence[float]], float] = hooks.get_last,
        metric: p.ObjectiveProtocol[_Output_contra, _Target_contra]
        | str
        | None = None,
        monitor: p.MonitorProtocol | None = None,
        min_delta: float = 1e-8,
        best_is: Literal['auto', 'higher', 'lower'] = 'auto',
    ) -> None:
        """Constructor.

        Args:
            trial: Optuna trial
            filter_fn: function to aggregate recent metric values.
            metric: Name of metric to monitor or metric calculator instance.
                    Defaults to the first metric found.
            monitor: Evaluation protocol to monitor. Defaults to validation
                if available, trainer instance otherwise.
            min_delta: Minimum change required to qualify as an improvement.
            best_is: Whether higher or lower metric values are better. Default
               'auto' will determine this from the first measurements.
        """
        self.monitor = hooks.MetricMonitor(
            metric=metric,
            monitor=monitor,
            min_delta=min_delta,
            best_is=best_is,
            filter_fn=filter_fn,
        )
        self.trial: Final = trial
        self.reported: Final = dict[int, float]()
        return

    def __call__(
        self,
        instance: p.TrainerProtocol[Any, _Target_contra, _Output_contra],
    ) -> None:
        """Evaluate whether training should be stopped early.

        Args:
            instance: Trainer instance to evaluate.
        """
        self.monitor.record_metric_value(instance)
        epoch = instance.model.epoch
        value = self.monitor.filtered_value
        self.trial.report(value, epoch)
        self.reported[epoch] = value
        if self.trial.should_prune():
            metric_name = self.monitor.metric_name
            msg = f'Optuna pruning while monitoring {metric_name}'
            instance.terminate_training(msg)
            raise optuna.TrialPruned()

        return


def suggest_overrides(tune_cfg: DictConfig, trial: optuna.Trial) -> list[str]:
    """Suggest values for a trial from structured configurations.

    This function helps integrate optuna into hydra by specifying trial
    parameters present in the hydra run configuration.

    The configuration file (loadable with hydra) should follow this structure:
    ```{code-block} yaml
    >>>tune:
    >>>  params:
    >>>    param_name:
    >>>      suggest: "suggest_float"  # or other optuna suggest method
    >>>      settings:
    >>>        low: 0.0
    >>>        high: 1.0
    >>>    list_param:
    >>>      suggest: "suggest_list"
    >>>      settings:
    >>>        min_length: 1
    >>>        max_length: 5
    >>>        suggest: "suggest_float"  # method for sampling list elements
    >>>        settings:
    >>>          low: 0.0
    >>>          high: 1.0
    >>>overrides: []  # additional static overrides
    ```

    For 'suggest_list' configurations, the settings must specify:
    - min_length and max_length: bounds for the size of the list.
    - nested suggest and settings: used to sample each list element.

    The resulting values can be used with hydra.initialize and hydra.compose.
    Example usage:
    ```{code-block} python
    >>>import hydra
    >>>
    >>>with hydra.initialize(version_base=None, config_path='path/to/config'):
    >>>    overrides = suggest_overrides(tune_cfg, trial)
    >>>    dict_cfg = hydra.compose(config_name='config', overrides=overrides)
    ```

    Here, "your_hydra_config" is the name of the overall configuration and must
    include the configuration parameters whose value will be overridden.

    Args:
        tune_cfg: a structure specifying how to sample new parameter values.
        trial: the optuna trial related to the sampled parameters.

    Returns:
        A list of strings for hydra configuration overrides.
    """
    all_overrides: list[str] = [*tune_cfg.overrides]
    for attr_name, param in tune_cfg.tune.params.items():
        if param.suggest == 'suggest_list':
            new_value = []
            for i in range(
                trial.suggest_int(
                    name='_'.join([attr_name, 'len']),
                    low=param.settings.min_length,
                    high=param.settings.max_length,
                )
            ):
                try:
                    bound_suggest = getattr(trial, param.settings.suggest)
                except AttributeError as ae:
                    msg = f'Invalid Optuna suggest configuration: {ae}.'
                    raise exceptions.DryTorchError(msg) from ae
                new_value.append(
                    bound_suggest(
                        '_'.join([attr_name, str(i)]), **param.settings.settings
                    )
                )
        else:
            try:
                bound_suggest = getattr(trial, param.suggest)
            except AttributeError as ae:
                msg = f'Invalid Optuna suggest configuration: {ae}.'
                raise exceptions.DryTorchError(msg) from ae
            new_value = bound_suggest(attr_name, **param.settings)
        all_overrides.append(f'{attr_name}={new_value}')

    return all_overrides


def get_final_value(
    trial: optuna.Trial,
    filter_fn: Callable[[Sequence[float]], float] | None = None,
) -> float:
    """Calculates a trial's final value from its intermediate reported values.

    This function aggregates the intermediate values reported during trial
    optimization using trial.report().

    Important: This function will not work with trials created using study.ask()
    as these don't populate the intermediate values in the corresponding
    FrozenTrial.

    Args:
        trial: the completed Optuna trial to evaluate.
        filter_fn: function to aggregate the trial's intermediate values.
            Defaults to min or max depending on the study direction.

    Returns:
        The aggregated final value for the trial.

    Raises:
        DryTorchException: if the trial has no reported values, or if there's
            a trial number mismatch.
    """
    current_study = trial.study
    if filter_fn is None:
        filter_fn = min if current_study.direction.name == 'MINIMIZE' else max
    frozen_trial = current_study.trials[-1]  # current trial as a FrozenTrial
    if frozen_trial.number != trial.number:
        msg = 'Trial number mismatch.'
        raise exceptions.DryTorchError(msg)
    reported_values = list(frozen_trial.intermediate_values.values())
    if not reported_values:
        msg = 'Optuna Trial has no reported values. Did you use study.optimize?'
        raise exceptions.DryTorchError(msg)

    return filter_fn(reported_values)
