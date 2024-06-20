import pytest
import torch
from dry_torch import Model
from dry_torch import CheckpointIO
from dry_torch import Experiment
from dry_torch.model_utils import ModelOptimizer, LearningScheme


def test_checkpoint():
    exp_pardir = 'test_experiments'
    model = torch.nn.Linear(1, 1)
    Experiment('test_checkpoint', exp_pardir=exp_pardir).activate()
    model = Model(model, name='first_model')
    checkpoint_io = CheckpointIO(model)
    checkpoint_io.save()
    checkpoint_io.load()
    first_loaded_parameter = checkpoint_io.model.module.parameters().__next__()
    first_saved_parameter = model.module.parameters().__next__()
    assert first_loaded_parameter == first_saved_parameter
    model_optimizer = ModelOptimizer(model.clone('second_model'),
                                     LearningScheme())
    checkpoint_io = CheckpointIO(model, model_optimizer.optimizer)
    checkpoint_io.save()
