import functools
import pathlib
from collections.abc import Sequence
from typing import Any

import yaml  # type: ignore

from dry_torch import log_events
from dry_torch import repr_utils
from dry_torch import tracking

MAX_LENGTH_PLAIN_REPR = 30
"""Sequences longer than this will be represented in flow style by yaml."""
MAX_LENGTH_SHORT_REPR = 10
"""Sequences with strings longer than this will be represented in flow style."""


def has_short_repr(obj: object, max_length: int = MAX_LENGTH_SHORT_REPR) -> bool:
    """Function that indicates whether an object has a short representation."""
    if isinstance(obj, repr_utils.LiteralStr):
        return False
    elif isinstance(obj, str):
        return obj.__len__() <= max_length
    elif hasattr(obj, '__len__'):
        return False
    return True


def represent_literal_str(dumper: yaml.Dumper,
                          literal_str: repr_utils.LiteralStr) -> yaml.Node:
    """YAML representer for literal strings."""
    return dumper.represent_scalar('tag:yaml.org,2002:str',
                                   literal_str,
                                   style='|')


def represent_str_with_ts(dumper: yaml.Dumper,
                          str_with_ts: repr_utils.StrWithTS) -> yaml.Node:
    """YAML representer for strings with timestamps."""
    return dumper.represent_scalar('tag:yaml.org,2002:str', str_with_ts)


def represent_sequence(
        dumper: yaml.Dumper,
        sequence: Sequence,
        max_length_for_plain: int = MAX_LENGTH_PLAIN_REPR,
) -> yaml.Node:
    """YAML representer for sequences."""
    flow_style = False
    if len(sequence) <= max_length_for_plain:
        if all(has_short_repr(elem) for elem in sequence):
            flow_style = True
    return dumper.represent_sequence(tag=u'tag:yaml.org,2002:seq',
                                     sequence=sequence,
                                     flow_style=flow_style)


def represent_omitted(dumper: yaml.Dumper,
                      data: repr_utils.Omitted) -> yaml.Node:
    """YAML representer for omitted values."""
    return dumper.represent_mapping(u'!Omitted',
                                    {'omitted_elements': data.count})


yaml.add_representer(repr_utils.LiteralStr, represent_literal_str)
yaml.add_representer(repr_utils.StrWithTS, represent_str_with_ts)
yaml.add_representer(list, represent_sequence)
yaml.add_representer(tuple, represent_sequence)
yaml.add_representer(set, represent_sequence)
yaml.add_representer(repr_utils.Omitted, represent_omitted)


class YamlDumper(tracking.Tracker):

    def __init__(self, par_dir: pathlib.Path = pathlib.Path('experiments')):
        super().__init__()
        self.par_dir = par_dir
        self.exp_name: str

    @property
    def exp_path(self) -> pathlib.Path:
        path = self.par_dir / self.exp_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @functools.singledispatchmethod
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.StartExperiment) -> None:
        self.exp_name = event.exp_name

    @notify.register
    def _(self, event: log_events.ModelCreation) -> None:
        self.dump(event.metadata, event.model_name)
        return

    @notify.register
    def _(self, event: log_events.CallModel) -> None:
        event.name = self.dump(event.metadata, event.name)
        return

    def dump(self,
             metadata: dict[str, Any],
             file_name: str) -> str:
        metadata_path = self.exp_path / 'metadata'
        metadata_path.mkdir(parents=True, exist_ok=True)
        file_path = metadata_path / file_name
        file_path = file_path.with_suffix('.yaml')

        yaml_str = yaml.dump(metadata, sort_keys=False)
        # if file_path.exists():
        #     with file_path.open('r') as metadata_file:
        #         file_out = metadata_file.read()
        #         if file_out != yaml_str:
        #             warnings.warn(
        #                 exceptions.MetadataNotMatchingWarning(file_name,
        #                                                       file_path)
        #             )
        #             now = datetime.datetime.now().isoformat(timespec='seconds')
        #             file_name = file_name + '.' + now
        #             file_path = file_path.with_stem(file_name)
        with file_path.open('w') as metadata_file:
            metadata_file.write(yaml_str)
        return file_name
