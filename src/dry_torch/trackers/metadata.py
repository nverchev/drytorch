import functools
import warnings
from typing import Any

import yaml  # type: ignore

from src.dry_torch import log_events
from src.dry_torch import repr_utils
from src.dry_torch import exceptions
from src.dry_torch import tracking


class MetadataExtractor(tracking.Tracker):
    def __init__(self, max_items_repr: int = 10) -> None:
        super().__init__()
        self.max_items_repr = max_items_repr
        self.default_names: dict[str, dict[str, repr_utils.DefaultName]] = {}
        self.exp_name: str

    @functools.singledispatchmethod
    def notify(self, event: log_events.Event) -> None:
        return

    @notify.register
    def _(self, event: log_events.StartExperiment) -> None:
        self.exp_name = event.exp_name
        return

    @notify.register
    def _(self, event: log_events.ModelCreation) -> None:
        name = event.model.name
        if name in self.default_names:
            raise exceptions.ModelNameAlreadyExistsError(name, self.exp_name)
        self.default_names[name] = {}
        metadata = {'module': repr_utils.LiteralStr(repr(event.model.module))}
        event.metadata |= metadata
        return

    @notify.register
    def _(self, event: log_events.RecordMetadata) -> None:
        model_default_names = self.default_names.get(event.model_name)
        if model_default_names is None:
            raise exceptions.ModelNotExistingError(event.model_name,
                                                   self.exp_name)
        cls_count = model_default_names.setdefault(
            event.name,
            repr_utils.DefaultName(event.name)
        )
        name = cls_count()

        metadata = extract_metadata(event.kwargs, self.max_items_repr)
        event.kwargs['name'] = name
        event.metadata |= {event.class_name: metadata}
        return


def extract_metadata(to_document: dict[str, Any], max_size) -> dict[str, Any]:
    """
    Wrapper of recursive_repr that catches Recursion Errors

    Args:
        to_document: a dictionary of objects to document.
        max_size: maximum number of documented items in an obj.
    """
    try:
        metadata = {
            k: repr_utils.recursive_repr(v, max_size=max_size)
            for k, v in to_document.items()
        }
    except RecursionError:
        warnings.warn(exceptions.RecursionWarning())
        metadata = {}
    return metadata
