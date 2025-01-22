import torch

from src.dry_torch import Trainer
# from src.dry_torch import Test as _Test  # pytest interprets Test as a test
from src.dry_torch import Experiment
from src.dry_torch import LearningScheme
from src.dry_torch import hooks


# from tests.example_classes import TorchTuple

def test_convergence(exp_pardir,
                     identity_trainer) -> None:
    """ Trainer works if the model weight converges to 1."""
    identity_trainer.train(6)
    linear_weight = next(identity_trainer.model.module.parameters())
    assert torch.isclose(linear_weight, torch.tensor(1.))


def test_early_stopping_on_val(exp_pardir,
                               identity_loader,
                               linear_model,
                               square_loss_calc,
                               zero_metrics_calc) -> None:
    Experiment('test_early_stop_val', par_dir=exp_pardir).start()

    trainer = Trainer(linear_model,
                      name='MyTrainer',
                      learning_scheme=LearningScheme(torch.optim.Adam, lr=0.01),
                      loss_calc=square_loss_calc,
                      loader=identity_loader)

    trainer.add_validation(val_loader=identity_loader)
    trainer._post_epoch_hooks.register(
        hooks.EarlyStoppingCallback(patience=1)
    )
    # trainer.post_epoch_hooks.register(
    #     hooks.call_every(5, hooks.saving_hook())
    # )

    # tuple_in = TorchTuple(input=torch.FloatTensor([.2]).to(linear_model.device))
    # out = linear_model(tuple_in)
    # assert torch.isclose(out.output, torch.tensor(.2), atol=0.01)
    # _Test(linear_model, metrics_calc=metrics_calc, loader=identity_loader)
