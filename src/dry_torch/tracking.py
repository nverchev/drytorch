from __future__ import annotations

import logging
import pathlib
import warnings
from typing import Any, Optional, Type, Final
import pandas as pd
from dry_torch import exceptions
from dry_torch import default_logging
from dry_torch import data_types
from dry_torch import repr_utils

logger = logging.getLogger('dry_torch')


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
        self.log: data_types.LogsDict = {
            split: pd.DataFrame() for split in data_types.Split
        }
        self.epoch = 0
        self.bindings: dict[Type, Any] = {}


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
    environment_set: set[Experiment] = set()
    active_environment: Optional[Experiment] = None
    default_prefix: str = 'Experiment_'
    number_default_environments: int = 0

    """
    This class is used to describe the experiment.

    Args:
        exp_name: the name of the experiment.
        config: configuration for the experiment.
        exp_pardir: parent directory for the folders with the model checkpoints
        allow_extract_metadata: whether to extract metadata from classes that 
        implement the allow_extract_metadata decorator
        max_string_repr: limits the size of iterators and arrays.



    Attributes:
        metric_logger: contains the saved metric and a plotting function
        epoch: the current epoch, that is, the number of epochs the model has 
        been trainer plus one.


    """

    def __init__(self,
                 exp_name: str,
                 config: Optional[dict[str, Any]] = None,
                 exp_pardir: str | pathlib.Path = pathlib.Path('experiments'),
                 allow_extract_metadata: bool = True,
                 max_string_repr: int = 9) -> None:
        self.exp_name: Final = exp_name
        self.config = config
        self.exp_pardir = pathlib.Path(exp_pardir)
        self.allow_extract_metadata = allow_extract_metadata
        self.max_string_repr = max_string_repr
        self.model = ModelTrackingDict(exp_name=exp_name)
        self.__class__.environment_set.add(self)

    def register_model(self, model, name):
        self.model[name] = ModelTracking(
            name, model_repr=model.__repr__(),
            model_settings=getattr(model, 'settings', {}))

    def run(self):
        if self.active_environment is not None:
            self.stop()
        self.__class__.active_environment = self
        logger.log(default_logging.INFO_LEVELS.experiment,
                   'Running experiment: %(exp_name)s.',
                   {'exp_name': self.exp_name})

    def stop(self):
        logger.log(default_logging.INFO_LEVELS.experiment,
                   f'Stopping experiment:  %(exp_name)s.',
                   {'exp_name': self.active_environment.exp_name})
        self.active_environment = None

    def __repr__(self):
        return self.__class__.__name__ + f'(exp_name={self.exp_name})'

    @classmethod
    def get_active_environment(cls) -> Experiment:
        if cls.active_environment is not None:
            return cls.active_environment
        unnamed_experiment = cls.new_default_environment()
        unnamed_experiment.run()
        return unnamed_experiment

    @classmethod
    def new_default_environment(cls) -> Experiment:
        cls.number_default_environments += 1
        return cls(cls.default_prefix + str(cls.number_default_environments))


def extract_metadata(attr_dict: dict[str, Any],
                     max_size: int = 10) -> dict[str, Any]:
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
                 name: str,
                 attr_dict: dict[str, Any]) -> None:
    if exp.allow_extract_metadata:
        # tries to get the most informative representation of the metadata.
        try:
            metadata = extract_metadata(attr_dict, exp.max_string_repr)
            exp.model[name].metadata |= metadata
        except RecursionError:
            msg = 'Could not extract metadata because of recursive objects.'
            warnings.warn(msg)
