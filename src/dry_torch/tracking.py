from __future__ import annotations

import logging
import pathlib
import datetime
from typing import Any, Optional, Final, TypeVar, Generic

import pandas as pd

from dry_torch import descriptors
from dry_torch import exceptions
from dry_torch import log_settings
from dry_torch import repr_utils
from dry_torch import protocols as p

logger = logging.getLogger('dry_torch')

_T = TypeVar('_T')


class DefaultName:
    def __init__(self, prefix: str, start: int = -1):
        self.prefix = prefix
        self.count_defaults = start

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
        self.binding: Optional[str] = None
        self.default_names: dict[str, DefaultName] = {}
        self.log = {split: pd.DataFrame() for split in descriptors.Split}


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
            raise exceptions.NameAlreadyExistsError(key, self.exp_name)
        self._models.__setitem__(key, value)

    def __delitem__(self, key: str):
        self._models.__delitem__(key)

    def __iter__(self):
        return self._models.__iter__()


class Experiment(Generic[_T]):
    past_experiments: set[Experiment] = set()
    _current: Optional[Experiment] = None
    _current_config: Optional[_T] = None
    _default_link_name = DefaultName('outputs')

    """
    This class is used to describe the experiment.

    Args:
        name: the model_name of the experiment.
        config: configuration for the experiment.
        pardir: parent directory for the folders with the module checkpoints
        allow_extract_metadata: whether to extract metadata from classes that 
        implement the allow_extract_metadata decorator
        max_items_repr: limits the size of iterators and arrays.



    Attributes:
        metric_logger: contains the saved metric_name and a plotting function
        epoch: the current epoch, that is, the number of epochs the module has 
        been trainer plus one.


    """

    def __init__(self,
                 name: str = '',
                 pardir: str | pathlib.Path = pathlib.Path(''),
                 config: Optional[Any] = None,
                 allow_extract_metadata: bool = True,
                 max_items_repr: int = 3,
                 link_to_hydra: bool = False) -> None:

        self.name: Final = name or datetime.datetime.now().isoformat()
        self.pardir = pathlib.Path(pardir)
        self.dir = self.pardir / name
        self.dir.mkdir(exist_ok=True, parents=True)
        self.config = config
        self.allow_extract_metadata = allow_extract_metadata
        self.max_items_repr = max_items_repr
        if link_to_hydra:
            self.link_to_hydra()
        self.tracker = ModelTrackerDict(exp_name=self.name)
        self.__class__.past_experiments.add(self)
        self.activate()

    def link_to_hydra(self) -> None:
        try:
            import hydra
        except ImportError:
            raise exceptions.LibraryNotAvailableError('hydra')

        # noinspection PyUnresolvedReferences
        str_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        hydra_dir = pathlib.Path(str_dir)
        hydra_link = self.dir / 'hydra'
        hydra_link.mkdir(exist_ok=True)
        while True:
            link_name = self.__class__._default_link_name()
            link_dir = hydra_link / link_name
            if link_dir.exists():
                continue
            if not hydra_dir.exists() and hydra_dir.is_dir():
                raise exceptions.LibraryNotSupportedError('hydra')
            else:
                link_dir.symlink_to(hydra_dir, target_is_directory=True)
                break

        return

    def activate(self) -> None:
        if Experiment._current is not None:
            self.stop()
        Experiment._current = self
        self.__class__._current_config = self.config
        logger.log(log_settings.INFO_LEVELS.experiment,
                   'Running experiment: %(name)s.',
                   {'name': self.name})
        return

    def stop(self) -> None:
        logger.log(log_settings.INFO_LEVELS.experiment,
                   f'Stopping experiment:  %(name)s.',
                   {'name': self.name})
        Experiment._current = None
        return

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(name={self.name})'

    @classmethod
    def current(cls) -> Experiment:
        if Experiment._current is not None:
            return Experiment._current
        unnamed_experiment = cls(datetime.datetime.now().isoformat())
        unnamed_experiment.activate()
        return unnamed_experiment

    @classmethod
    def get_config(cls) -> _T:
        cfg = cls._current_config
        if cfg is None:
            raise exceptions.NoConfigError()
        return cfg


def track(model: p.ModelProtocol) -> ModelTracker:
    exp = Experiment.current()
    return exp.tracker[model.name]
