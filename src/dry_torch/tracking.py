from __future__ import annotations

import logging
import pathlib
import warnings
from typing import Any, Optional, Type, Final, Callable

import pandas as pd
from dry_torch import exceptions
from dry_torch import default_logging
from dry_torch import repr_utils
from dry_torch import data_types

logger = logging.getLogger('dry_torch')


def default_name(prefix: str) -> Callable[[], str]:
    prefix = prefix
    count_defaults = 0

    def default() -> str:
        nonlocal count_defaults
        count_defaults += 1
        return prefix + str(count_defaults)

    return default


class ModelTracking:

    def __init__(self,
                 name: str,
                 model_repr: str,
                 model_settings: dict[str, Any]) -> None:
        self.name: Final = name
        self.metadata: dict[str, Any] = {
            name: repr_utils.LiteralStr(model_repr)
        }
        self.metadata |= model_settings
        self.epoch = 0
        self.bindings: dict[Type, Any] = {}
        self.log: dict[data_types.Split, pd.DataFrame] = {
            split: pd.DataFrame() for split in data_types.Split
        }


class ModelTrackingDict:

    def __init__(self, exp_name: str) -> None:
        self.exp_name: Final = exp_name
        self._models: dict[str, ModelTracking] = {}

    def __contains__(self, item) -> bool:
        return self._models.__contains__(item)

    def __getitem__(self, key: str):
        if key not in self:
            raise exceptions.ModelNotFoundError(key, self.exp_name)
        return self._models.__getitem__(key)

    def __setitem__(self, key: str, value: ModelTracking):
        if key in self:
            raise exceptions.ModelAlreadyRegisteredError(key, self.exp_name)
        self._models.__setitem__(key, value)

    def __delitem__(self, key: str):
        self._models.__delitem__(key)


class Experiment:
    past_experiments: set[Experiment] = set()
    _current: Optional[Experiment] = None
    default_exp_name = default_name('Experiment_')
    """
    This class is used to describe the experiment.

    Args:
        exp_name: the model_name of the experiment.
        config: configuration for the experiment.
        exp_pardir: parent directory for the folders with the module checkpoints
        allow_extract_metadata: whether to extract metadata from classes that 
        implement the allow_extract_metadata decorator
        max_item_repr: limits the size of iterators and arrays.



    Attributes:
        metric_logger: contains the saved metric and a plotting function
        epoch: the current epoch, that is, the number of epochs the module has 
        been trainer plus one.


    """

    def __init__(self,
                 exp_name: Optional[str] = None,
                 config: Optional[dict[str, Any]] = None,
                 exp_pardir: str | pathlib.Path = pathlib.Path('experiments'),
                 allow_extract_metadata: bool = True,
                 max_item_repr: int = 3) -> None:

        self.exp_name: Final = exp_name or self.__class__.default_exp_name()
        self.config = config
        self.exp_pardir = pathlib.Path(exp_pardir)
        self.allow_extract_metadata = allow_extract_metadata
        self.max_item_repr = max_item_repr
        self.model = ModelTrackingDict(exp_name=self.exp_name)
        self.__class__.past_experiments.add(self)
        self.activate()

    def register_model(self, model, name):
        self.model[name] = ModelTracking(
            name, model_repr=model.__repr__(),
            model_settings=getattr(model, 'settings', {}))

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


def extract_metadata(attr_dict: dict[str, Any],
                     max_size: int = 3) -> dict[str, Any]:
    # tries to get the most informative representation of the metadata.
    try:
        metadata = {k: repr_utils.struc_repr(v, max_size=max_size)
                    for k, v in attr_dict.items()}
    except RecursionError:
        msg = 'Could not extract metadata because of recursive objects.'
        warnings.warn(msg)
        metadata = {}
    return metadata


def add_metadata(exp: Experiment,
                 model_name: str,
                 object_name: str,
                 attr_dict: dict[str, Any]) -> None:
    if exp.allow_extract_metadata:
        # tries to get the most informative representation of the metadata.
        metadata = extract_metadata(attr_dict, exp.max_item_repr)
        exp.model[model_name].metadata[object_name] = metadata
