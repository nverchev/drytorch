"""Module containing YAML options and dumper.

Attributes:
    MAX_LENGTH_PLAIN_REPR: Maximum length for sequences in plain style.
    MAX_LENGTH_SHORT_REPR: Maximum string length in Sequences in plain style.
    TS_FMT: format for representing a timestamp.
"""

import functools
import pathlib

from collections.abc import Sequence
from typing import Any

import yaml

from typing_extensions import override

from drytorch.core import log_events
from drytorch.trackers import base_classes
from drytorch.utils import repr_utils


MAX_LENGTH_PLAIN_REPR = 30
MAX_LENGTH_SHORT_REPR = 10

TS_FMT = repr_utils.CreatedAtMixin.ts_fmt


class YamlDumper(base_classes.Dumper):
    """Tracker that dumps metadata in a YAML file.

    Class Attributes:
        folder_name: name for the folder that contains metadata.
    """

    folder_name = 'metadata'

    def __init__(self, par_dir: pathlib.Path | None = None):
        """Constructor.

        Args:
            par_dir: directory where to dump metadata. Defaults to the current
                experiment's one.
        """
        super().__init__(par_dir)

    @functools.singledispatchmethod
    @override
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StartExperimentEvent) -> None:
        super().notify(event)
        run_dir = self._get_run_dir()
        repr_config = repr_utils.recursive_repr(event.config, depth=1000)
        model_with_ts = 'config_' + event.run_ts.strftime(TS_FMT) + '.yaml'
        self._dump(repr_config, run_dir / model_with_ts)
        return

    @notify.register
    def _(self, event: log_events.ModelRegistrationEvent) -> None:
        run_dir = self._get_run_dir()
        name_with_ts = event.model_name + '_' + event.model_ts.strftime(TS_FMT)
        file = self._file_path(run_dir, name_with_ts, 'architecture')
        self._dump(event.architecure_repr, file)
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.ActorRegistrationEvent) -> None:
        run_dir = self._get_run_dir()
        model_with_ts = event.model_name + '_' + event.model_ts.strftime(TS_FMT)
        actor_with_ts = event.actor_name + '_' + event.actor_ts.strftime(TS_FMT)
        file = self._file_path(run_dir, model_with_ts, actor_with_ts)
        self._dump(event.metadata, file)
        return super().notify(event)

    @staticmethod
    def _dump(metadata: dict[str, Any] | str, file_path: pathlib.Path) -> None:
        with file_path.open('w') as metadata_file:
            yaml.dump(metadata, metadata_file)

        return

    @staticmethod
    def _file_path(
        run_dir: pathlib.Path,
        model_name: str,
        obj_name: str,
    ) -> pathlib.Path:
        model_path = run_dir / model_name
        model_path.mkdir(exist_ok=True)
        return model_path / f'{obj_name}.yaml'


def has_short_repr(
    obj: object, max_length: int = MAX_LENGTH_SHORT_REPR
) -> bool:
    """Indicate whether an object has a short representation."""
    if isinstance(obj, repr_utils.LiteralStr):
        return False
    elif isinstance(obj, str):
        return len(obj) <= max_length
    elif hasattr(obj, '__len__'):
        return False

    return True


def represent_literal_str(
    dumper: yaml.Dumper, literal_str: repr_utils.LiteralStr
) -> yaml.ScalarNode:
    """YAML representer for literal strings."""
    return dumper.represent_scalar(
        'tag:yaml.org,2002:str', literal_str, style='|'
    )


def represent_sequence(
    dumper: yaml.Dumper,
    sequence: Sequence[Any] | set[Any],
    max_length_for_plain: int = MAX_LENGTH_PLAIN_REPR,
) -> yaml.SequenceNode:
    """YAML representer for sequences."""
    flow_style = False
    if len(sequence) <= max_length_for_plain:
        if all(has_short_repr(elem) for elem in sequence):
            flow_style = True

    return dumper.represent_sequence(
        tag='tag:yaml.org,2002:seq', sequence=sequence, flow_style=flow_style
    )


def represent_omitted(
    dumper: yaml.Dumper, data: repr_utils.Omitted
) -> yaml.MappingNode:
    """YAML representer for omitted values."""
    return dumper.represent_mapping(
        '!Omitted', {'omitted_elements': data.count}
    )


yaml.add_representer(repr_utils.LiteralStr, represent_literal_str)
yaml.add_representer(list, represent_sequence)
yaml.add_representer(tuple, represent_sequence)
yaml.add_representer(set, represent_sequence)
yaml.add_representer(repr_utils.Omitted, represent_omitted)
