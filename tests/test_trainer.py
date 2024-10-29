import torch

from src.dry_torch import Trainer
from src.dry_torch import Test as _Test  # pytest interprets Test as a test
from src.dry_torch import Experiment
from src.dry_torch import LearningScheme
from src.dry_torch import hooks

from tests.example_classes import TorchTuple


def test_all(exp_pardir,
             identity_loader,
             linear_model,
             square_calc,
             metrics_calc) -> None:
    Experiment('test_simple_training', pardir=exp_pardir).start()

    trainer = Trainer(linear_model,
                      name='MyTrainer',
                      learning_scheme=LearningScheme(torch.optim.Adam, lr=0.01),
                      loss_calc=square_calc,
                      loader=identity_loader)

    trainer.add_validation(val_loader=identity_loader)
    trainer.post_epoch_hooks.register(
        hooks.early_stopping_callback(
            monitor_validation=False,
            monitor_external=trainer.validation,
            patience=1))
    trainer.post_epoch_hooks.register(
        hooks.call_every(5, hooks.saving_hook())
    )

    trainer.train(6)
    tuple_in = TorchTuple(input=torch.FloatTensor([.2]).to(linear_model.device))
    out = linear_model(tuple_in)
    assert torch.isclose(out.output, torch.tensor(.2), atol=0.01)
    test = _Test(linear_model,
                 metrics_calc=metrics_calc,
                 loader=identity_loader,
                 )
    test(store_outputs=True)
