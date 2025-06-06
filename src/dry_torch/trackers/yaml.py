"""
Module containing YAML options and dumper.

Attributes:
    MAX_LENGTH_PLAIN_REPR: Maximum length for sequences in plain style.
    MAX_LENGTH_SHORT_REPR: Maximum string length in Sequences in plain style.
"""

from collections.abc import Sequence
import functools
import pathlib
from typing import Any, Optional
from typing_extensions import override

import yaml  # type: ignore

from dry_torch import log_events
from dry_torch.trackers import base_classes
from dry_torch.utils import repr_utils

MAX_LENGTH_PLAIN_REPR = 30
MAX_LENGTH_SHORT_REPR = 10


class YamlDumper(base_classes.Dumper):
    """Tracker that dumps metadata in a YAML file."""

    def __init__(self, par_dir: Optional[pathlib.Path] = None):
        """
        Args:
            par_dir: directory where to dump metadata. Defaults uses the current
                experiment's one.
        """
        super().__init__(par_dir)
        self.metadata_folder = 'metadata'
        self.archive_folder = 'archive'
        self._exp_dir: Optional[pathlib.Path] = None

    @override
    @functools.singledispatchmethod
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.ModelCreation) -> None:
        self._version(event.metadata,
                      event.model_name,
                      event.model_name,
                      event.model_version)
        return super().notify(event)

    @notify.register
    def _(self, event: log_events.CallModel) -> None:
        model_name = event.model_name
        self._version(event.metadata,
                      model_name,
                      event.source_name,
                      event.source_version)
        return super().notify(event)

    def _version(self,
                 metadata: dict[str, Any],
                 sub_folder: str,
                 file_name: str,
                 file_version: str) -> None:
        directory = self.par_dir / sub_folder / self.metadata_folder
        archive_directory = directory / self.archive_folder / file_name
        archive_directory.mkdir(exist_ok=True, parents=True)
        self._dump(metadata, directory / file_name)
        self._dump(metadata, archive_directory / file_version)
        return

    @staticmethod
    def _dump(metadata: dict[str, Any], file_path: pathlib.Path) -> None:
        file_with_suffix = file_path.with_suffix(file_path.suffix + '.yaml')
        with file_with_suffix.open('w') as metadata_file:
            yaml.dump(metadata, metadata_file)

        return


def has_short_repr(obj: object,
                   max_length: int = MAX_LENGTH_SHORT_REPR) -> bool:
    """Indicate whether an object has a short representation."""
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
yaml.add_representer(list, represent_sequence)
yaml.add_representer(tuple, represent_sequence)
yaml.add_representer(set, represent_sequence)
yaml.add_representer(repr_utils.Omitted, represent_omitted)
