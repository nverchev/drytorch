import datetime
import functools
import pathlib
import warnings
from typing import Any

import yaml  # no typing
from hypothesis import event

from src.dry_torch import events
from src.dry_torch import tracking
from src.dry_torch import repr_utils

from src.dry_torch import exceptions


class ModelTracker:

    def __init__(self) -> None:
        self.metadata: dict[str, Any] = {}
        self.default_names: dict[str, repr_utils.DefaultName] = {}


class ModelTrackerDict:

    def __init__(self, exp_name: str) -> None:
        self.exp_name = exp_name
        self._models: dict[str, ModelTracker] = {}

    def __contains__(self, item) -> bool:
        return self._models.__contains__(item)

    def __getitem__(self, key: str) -> ModelTracker:
        if key not in self:
            raise exceptions.ModelNotExistingError(key, self.exp_name)
        return self._models.__getitem__(key)

    def __setitem__(self, key: str, value: ModelTracker):
        if key in self:
            raise exceptions.ModelNameAlreadyExistsError(key, self.exp_name)
        self._models.__setitem__(key, value)

    def __delitem__(self, key: str):
        self._models.__delitem__(key)

    def __iter__(self):
        return self._models.__iter__()


class LogMetadata(events.Subscriber):
    def __init__(self, par_dir: pathlib.Path, max_items_repr: int = 10) -> None:
        super().__init__()
        self.model_tracker_dict = ModelTrackerDict(self.exp_name)
        self.par_dir = par_dir
        self.max_items_repr = max_items_repr

    @property
    def path(self) -> pathlib.Path:
        return self.par_dir / self.exp_name

    @functools.singledispatchmethod
    def notify(self, event: events.Event) -> None:
        return

    @notify.register
    def _(self, event: events.ModelCreation) -> None:
        tracker = ModelTracker()
        name = event.model.name
        class_str = event.model.__class__.__name__
        metadata = {name: repr_utils.LiteralStr(repr(event.model.module))}
        tracker.metadata[class_str] = metadata
        self.model_tracker_dict[name] = tracker
        self.dump_metadata(tracker, class_str)
        return

    @notify.register
    def _(self, event: events.CreateEvaluation) -> None:
        if event.args:
            warnings.warn(exceptions.NotDocumentedArgs())
        model_tracker = self.model_tracker_dict[event.model.name]
        cls_count = model_tracker.default_names.setdefault(
            event.cls_str,
            repr_utils.DefaultName(event.cls_str)
        )
        name = cls_count()
        if name in model_tracker.metadata:
            raise exceptions.NameAlreadyExistsError(name, event.model.name)
        tracker = ModelTracker()
        name = event.model.name
        class_str = event.model.__class__.__name__
        metadata = extract_metadata(event.kwargs)
        self.dump_metadata(tracker, name)

        tracker.metadata[class_str] = metadata
        self.model_tracker_dict[name] = tracker
        self.dump_metadata(tracker, class_str)
        return

    def dump_metadata(self, tracker: ModelTracker, class_str: str) -> str:
        metadata = tracker.metadata[class_str]
        metadata_path = (self.path / class_str).with_suffix('.yaml')
        yaml_str = yaml.dump(metadata,
                             default_flow_style=False,
                             sort_keys=False)
        if metadata_path.exists():
            with metadata_path.open('r') as metadata_file:
                file_out = metadata_file.read()
                if file_out != yaml_str:
                    warnings.warn(
                        exceptions.MetadataNotMatchingWarning(class_str,
                                                              metadata_path)
                    )
                    now = datetime.datetime.now().isoformat(timespec='seconds')
                    class_str_now = class_str + '.' + now
                    metadata_path = metadata_path.with_stem(class_str_now)
        with metadata_path.open('w') as metadata_file:
            metadata_file.write(yaml_str)
        return metadata_path.stem




def extract_metadata(to_document: dict[str, Any],
                     max_size: int = 3) -> dict[str, Any]:
    """
    Wrapper of recursive_repr that catches Recursion Errors

    Args:
        to_document: a dictionary of objects to document.
        max_size: maximum number of documented items in an obj.
    """
    # get the recursive representation of the objects.
    try:
        metadata = {k: repr_utils.recursive_repr(v, max_size=max_size)
                    for k, v in to_document.items()}
    except RecursionError:
        warnings.warn(exceptions.RecursionWarning())
        metadata = {}
    return metadata