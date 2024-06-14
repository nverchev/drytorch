import pytest
import torch
from dry_torch import ModelOptimizer
from dry_torch import CheckpointIO
from dry_torch import Experiment


def test_checkpoint():
    exp_pardir = 'test_experiments'
    model = torch.nn.Linear(1, 1)
    Experiment('test_checkpoint', exp_pardir=exp_pardir).run()
    model_optimizer = ModelOptimizer(model,
                                     optimizer_cls=torch.optim.SGD,
                                     name='first_model')
    checkpoint_io = CheckpointIO(model_optimizer)
    checkpoint_io.save()
    checkpoint_io.load()
    first_loaded_parameter = checkpoint_io.model.model.parameters().__next__()
    first_saved_parameter = model.parameters().__next__()
    assert first_loaded_parameter == first_saved_parameter
    model_optimizer = ModelOptimizer(model, name='second_model')
    checkpoint_io = CheckpointIO(model_optimizer)
    checkpoint_io.save()
