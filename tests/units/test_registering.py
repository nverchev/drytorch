import pathlib
from typing import Protocol, TypeVar, Generic

import pytest
import torch

from src.dry_torch import exceptions
from src.dry_torch.registering import record_metadata
from src.dry_torch import tracking
from src.dry_torch import Model


class _SimpleTrainer:
    def __init__(self):
        self.name = 'simple_trainer'
        self.model = Model(torch.nn.Linear(1, 1), name='SimpleModel')
        record_metadata(self)


class TestMetadataRecorder:
    """Test the MetadataRecorder metaclass."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up the SimpleTrainer."""
        self.trainer = _SimpleTrainer()
        return

    def test_register_model(self, mock_experiment) -> None:
        manager = mock_experiment.metadata_manager
        manager.record_metadata.assert_called_once_with(self.trainer.name,
                                                        self.trainer)
        manager.register_model.assert_called_once_with(self.trainer.model)
        assert True

#
# def test_register_model_already_registered(mock_model, mock_experiment) -> None:
#     """Test that an error is raised if the model is already registered."""
#     register_model(mock_model)
#
#     with pytest.raises(exceptions.ModuleAlreadyRegisteredError):
#         register_model(mock_model)
#
#     return
#
#
# def test_register_kwargs_decorator_calls_event(mocker, mock_model) -> None:
#     """Test that register_kwargs collects metadata."""
#
#     class _TestEvent:
#         name = 'TestEvent'
#
#     mock_record_event = mocker.patch('src.dry_torch.log_events.RecordMetadata',
#                                      return_value=_TestEvent())
#
#     @register_kwargs
#     def _sample_init(instance, model, /, **kwargs):
#         _not_used = instance, model, kwargs
#
#     mock_instance = mocker.MagicMock()
#     _sample_init(mock_instance, mock_model, arg1=10, name='TestName')
#
#     # name is not in the dictionary when the even is called but added later
#     mock_record_event.assert_called_once_with(
#         'mock_model',
#         mock_instance.__class__.__name__,
#         'TestName',
#         {'arg1': 10, 'name': 'TestEvent'}  #
#     )
#
#     return
#
#
# def test_register_kwargs_raises_model_first_error(mocker) -> None:
#     """
#     Test register_kwargs raises an error if model does not follow instance.
#     """
#
#     @register_kwargs
#     def _sample_init(instance, model, /, **kwargs):
#         _not_used = instance, model, kwargs
#
#     mock_instance = mocker.MagicMock()
#     with pytest.raises(exceptions.ModelFirstError):
#         _sample_init(mock_instance, 'not_a_model', arg1=10)  # type: ignore
#
#     return
#
#
# def test_register_kwargs_warns_for_non_keyword_args(mocker, mock_model) -> None:
#     """Test that register_kwargs issues a warning for positional arguments."""
#
#     @register_kwargs
#     def _sample_init(instance, model, arg, **kwargs):
#         _not_used = instance, model, arg, kwargs
#
#     mock_instance = mocker.MagicMock()
#     with pytest.warns(exceptions.NotDocumentedArgs):
#         _sample_init(mock_instance, mock_model, 'pos_arg')
#
#     return
#
#
# def test_register_kwargs_sets_default_name(mocker, mock_model) -> None:
#     """Test that register_kwargs assigns the class name by default."""
#
#     @register_kwargs
#     def _sample_init(instance, model, /, name='', **kwargs):
#         _not_used = instance, model, kwargs
#         return name
#
#     mock_instance = mocker.MagicMock()
#     default_name = _sample_init(mock_instance, mock_model, arg1=10)
#     assert default_name == mock_instance.__class__.__name__
#
#     return
