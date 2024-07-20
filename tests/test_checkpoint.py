import pathlib
import pytest
import torch
from dry_torch import Model
from dry_torch import ModelStateIO
from dry_torch import CheckpointIO
from dry_torch import Experiment
from dry_torch import LearningScheme
from dry_torch import register_model
from dry_torch.learning import ModelOptimizer
from dry_torch.exceptions import ConfigNotMatchingError
from dry_torch.exceptions import AlreadyRegisteredError


def test_checkpoint():
    exp_pardir = pathlib.Path(__file__).parent / 'experiments'
    model = torch.nn.Linear(1, 1)
    Experiment('test_checkpoint',
               exp_pardir=exp_pardir,
               config={'test': 'test'})
    model = Model(model, name='first_model')
    register_model(model)
    with pytest.raises(AlreadyRegisteredError):
        register_model(model)
    model_state_io = ModelStateIO(model)
    model_state_io.save()
    model_state_io.load()
    first_loaded_parameter = model_state_io.model.module.parameters().__next__()
    first_saved_parameter = model.module.parameters().__next__()
    assert first_loaded_parameter == first_saved_parameter
    second_model = model.clone('second_model')
    register_model(second_model)
    model_optimizer = ModelOptimizer(second_model, LearningScheme())
    checkpoint_io = CheckpointIO(second_model, model_optimizer.optimizer)
    checkpoint_io.save()
    # with pytest.raises(ConfigNotMatchingError):
    #     Experiment('test_checkpoint',
    #                exp_pardir=exp_pardir,
    #                config={'test': 'test2'})
