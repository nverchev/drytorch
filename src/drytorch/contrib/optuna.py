"""Support for optuna."""

import collections
import string

from collections.abc import Callable, Sequence
from typing import Any, Final, Generic, TypeVar

import omegaconf
import optuna

from drytorch.core import exceptions
from drytorch.core import protocols as p
from drytorch.lib import monitoring


__all__ = [
    'OptunaError',
    'TrialCallback',
    'get_final_value',
    'suggest_overrides',
]

_Target = TypeVar('_Target', bound=p.TargetType)
_Output = TypeVar('_Output', bound=p.OutputType)


class OptunaError(exceptions.DryTorchError):
    """Base class for Optuna errors."""

    _template = 'Optuna: {}'


class TrialCallback(Generic[_Output, _Target]):
    """Implements pruning logic for training models.

    Attributes:
        monitor: monitor instance.
        reported: dictionary mapping epochs to reported values.
        trial: ``optuna`` trial.
    """

    monitor: Final[monitoring.MetricMonitor]
    reported: Final[dict[int, float]]
    trial: Final[optuna.Trial]

    def __init__(
        self,
        trial: optuna.Trial,
        filter_fn: Callable[[Sequence[float]], float] = monitoring.get_last,
        metric: p.ObjectiveProtocol[_Output, _Target] | str | None = None,
        source: p.SourceProtocol | None = None,
    ) -> None:
        """Initialize.

        Args:
            trial: ``optuna`` trial.
            filter_fn: function to aggregate recent metric values.
            metric: name of metric to monitor or metric calculator instance.
                    Defaults to the first metric found.
            source: the source computing the metric. Defaults to the
                validation if available, the trainer otherwise.
        """
        self.monitor = monitoring.MetricMonitor(
            metric=metric,
            source=source,
            filter_fn=filter_fn,
        )
        self.reported = {}
        self.trial = trial
        return

    def __call__(
        self,
        instance: p.TrainerProtocol[Any, _Target, _Output],
    ) -> None:
        """Evaluate whether training should be stopped early.

        Args:
            instance: trainer instance to evaluate.

        Raises:
            optuna.TrialPruned: if the trial should be pruned.
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


def suggest_overrides(
    tune_cfg: omegaconf.DictConfig,
    trial: optuna.Trial,
    use_full_name: bool = False,
) -> list[str]:
    """Suggest values for a trial from structured configurations.

    This function helps integrate optuna into hydra by specifying trial
    parameters present in the hydra run configuration.

    The configuration file (loadable with hydra) should follow this structure:

    .. code-block:: yaml

        tune:
          params:
            param_name:
              suggest: "suggest_float"  # or other optuna suggest method
              settings:
                low: 0.0
                high: 1.0
            list_param:
              suggest: "suggest_list"
              settings:
                min_length: 1
                max_length: 5
                suggest: "suggest_float"  # method for sampling list elements
                settings:
                  low: 0.0
                  high: 1.0
        overrides: []  # additional static overrides

    For 'suggest_list' configurations, the settings must specify:
    - min_length and max_length: bounds for the size of the list.
    - nested suggest and settings: used to sample each list element.

    The resulting values can be used with hydra.initialize and hydra.compose.
    Example usage:

    .. code-block:: python

        import hydra

        with hydra.initialize(version_base=None, config_path='path/to/config'):
            overrides = suggest_overrides(tune_cfg, trial)
            dict_cfg = hydra.compose(config_name='config', overrides=overrides)

    Here, "your_hydra_config" is the name of the configuration file that
    includes the configuration parameters to override.

    Args:
        tune_cfg: a structure specifying how to sample new parameter values.
        trial: the optuna trial related to the sampled parameters.
        use_full_name: use the fully qualified setting name. Default to a
            human-readable name.

    Returns:
        A list of strings for hydra configuration overrides.

    Raises:
        OptunaError: if the suggested configuration is invalid.
    """

    def _get_short_name(name: str) -> str:
        *prefix_parts, suffix = name.rsplit('.', maxsplit=2)
        if suffix.isdigit() and prefix_parts:
            suffix = f'{prefix_parts[-1]} {suffix}'

        return string.capwords(suffix.replace('_', ' '))

    short_names = {k: _get_short_name(k) for k in tune_cfg.tune.params}
    name_counts = collections.Counter(short_names.values())

    all_overrides: list[str] = [*tune_cfg.overrides]
    for setting_name, param_value in tune_cfg.tune.params.items():
        param_name = short_names[setting_name]
        if use_full_name or name_counts[param_name] > 1:
            param_name = setting_name

        if param_value.suggest == 'suggest_list':
            new_value: list[Any] = []
            for i in range(
                trial.suggest_int(
                    name=f'{param_name} #',
                    low=param_value.settings.min_length,
                    high=param_value.settings.max_length,
                )
            ):
                try:
                    bound_suggest = getattr(trial, param_value.settings.suggest)
                except AttributeError as ae:
                    msg = f'Invalid suggest configuration: {ae}.'
                    raise OptunaError(msg) from ae
                new_value.append(
                    bound_suggest(
                        f'{param_name} {i}', **param_value.settings.settings
                    )
                )
        else:
            try:
                bound_suggest = getattr(trial, param_value.suggest)
            except AttributeError as ae:
                msg = f'Invalid suggest configuration: {ae}.'
                raise OptunaError(msg) from ae

            new_value = bound_suggest(param_name, **param_value.settings)
        all_overrides.append(f'{setting_name}={new_value}')

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
        trial: the completed ``optuna`` trial to evaluate.
        filter_fn: function to aggregate the trial's intermediate values.
            Defaults to min or max depending on the study direction.

    Returns:
        The aggregated final value for the trial.

    Raises:
        OptunaError: if the trial has no reported values, or if there's
            a trial number mismatch.
    """
    current_study = trial.study
    if filter_fn is None:
        filter_fn = min if current_study.direction.name == 'MINIMIZE' else max

    for frozen_trial in reversed(current_study.trials):
        if frozen_trial.number == trial.number:
            break
    else:
        raise OptunaError('trial number mismatch.')

    reported_values = list(frozen_trial.intermediate_values.values())
    if not reported_values:
        msg = 'trial has no reported values. Did you use study.optimize?'
        raise OptunaError(msg)

    return filter_fn(reported_values)
