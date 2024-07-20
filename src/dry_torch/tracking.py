from __future__ import annotations

import logging
import pathlib
import datetime
from typing import Any, Optional, Final

import pandas as pd
import yaml  # type: ignore
from dry_torch import exceptions
from dry_torch import default_logging
from dry_torch import repr_utils
from dry_torch import protocols as p

logger = logging.getLogger('dry_torch')


class DefaultName:
    def __init__(self, prefix: str):
        self.prefix = prefix
        self.count_defaults = -1

    def __call__(self) -> str:
        self.count_defaults += 1
        return repr(self)

    def __repr__(self):
        if not self.count_defaults:
            return self.prefix
        return f"{self.prefix}_{self.count_defaults}"


class ModelTracker:

    def __init__(self, name: str, model_repr: str) -> None:
        self.name: Final = name
        self.epoch = 0
        model_literal = repr_utils.LiteralStr(model_repr)
        self.metadata: dict[str, Any] = {'Model': {name: model_literal}}
        self.bindings: dict[str, DefaultName] = {}
        self.log = {split: pd.DataFrame() for split in p.Split}


class ModelTrackerDict:

    def __init__(self, exp_name: str) -> None:
        self.exp_name: Final = exp_name
        self._models: dict[str, ModelTracker] = {}

    def __contains__(self, item) -> bool:
        return self._models.__contains__(item)

    def __getitem__(self, key: str) -> ModelTracker:
        if key not in self:
            raise exceptions.ModelNotExistingError(key, self.exp_name)
        return self._models.__getitem__(key)

    def __setitem__(self, key: str, value: ModelTracker):
        if key in self:
            raise exceptions.AlreadyRegisteredError(key, self.exp_name)
        self._models.__setitem__(key, value)

    def __delitem__(self, key: str):
        self._models.__delitem__(key)

    def __iter__(self):
        return self._models.__iter__()


class Experiment:
    past_experiments: set[Experiment] = set()
    _current: Optional[Experiment] = None
    default_exp_name = DefaultName('Experiment')
    """
    This class is used to describe the experiment.

    Args:
        exp_name: the model_name of the experiment.
        config: configuration for the experiment.
        exp_pardir: parent directory for the folders with the module checkpoints
        allow_extract_metadata: whether to extract metadata from classes that 
        implement the allow_extract_metadata decorator
        max_items_repr: limits the size of iterators and arrays.



    Attributes:
        metric_logger: contains the saved metric and a plotting function
        epoch: the current epoch, that is, the number of epochs the module has 
        been trainer plus one.


    """

    def __init__(self,
                 exp_name: str = '',
                 config: Optional[Any] = None,
                 exp_pardir: str | pathlib.Path = pathlib.Path('experiments'),
                 allow_extract_metadata: bool = True,
                 max_items_repr: int = 3) -> None:

        self.exp_name: Final = exp_name or datetime.date.today().isoformat()
        self.exp_pardir = pathlib.Path(exp_pardir)
        self.exp_dir = self.exp_pardir / self.exp_name
        self.exp_dir.mkdir(exist_ok=True)
        self.config_path = self.exp_dir / 'config.yml'
        if config is None:
            self.config = self.load_config()
        else:
            self.config = config
            self.save_config(config)
        self.allow_extract_metadata = allow_extract_metadata
        self.max_items_repr = max_items_repr
        self.tracker = ModelTrackerDict(exp_name=self.exp_name)
        self.__class__.past_experiments.add(self)
        self.activate()

    def save_config(self, config: Any) -> None:
        loaded_config = self.load_config()
        if loaded_config is not None and loaded_config != config:
            raise exceptions.ConfigNotMatchingError(config, loaded_config)
        else:
            with self.config_path.open('w') as config_file:
                yaml.dump(config, config_file)
        return

    def load_config(self) -> Any:
        # if self.config_path.exists():
        #     with self.config_path.open('r') as config_file:
        #         config = yaml.safe_load(config_file)
        #     return config
        return None

    def activate(self):
        if self._current is not None:
            self.stop()
        self.__class__._current = self
        logger.log(default_logging.INFO_LEVELS.experiment,
                   'Running experiment: %(exp_name)s.',
                   {'exp_name': self.exp_name})

    def stop(self):
        logger.log(default_logging.INFO_LEVELS.experiment,
                   f'Stopping experiment:  %(exp_name)s.',
                   {'exp_name': self._current.exp_name})
        self._current = None

    def __repr__(self):
        return self.__class__.__name__ + f'(exp_name={self.exp_name})'

    @classmethod
    def current(cls) -> Experiment:
        if cls._current is not None:
            return cls._current
        unnamed_experiment = cls(cls.default_exp_name())
        unnamed_experiment.activate()
        return unnamed_experiment

    @classmethod
    def current_cfg(cls) -> Any:
        cfg = cls.current().config
        if cfg is None:
            raise exceptions.NoConfigError()
        return cfg


def track(model: p.ModelProtocol) -> ModelTracker:
    exp = Experiment.current()
    return exp.tracker[model.name]
