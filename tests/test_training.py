# from dry_torch import Trainer
# from torch.utils.data import TensorDataset
# import torch
# from torch.optim import Adam
# from torch.nn import MSELoss
#
#
# uniform = torch.linspace(0, 1, 1000).unsqueeze(1)
# dset = TensorDataset(uniform, uniform)  # x == y
# trainer = Trainer(torch.nn.Linear(1, 1), exp_name='test_trainer', device=torch.device('cpu'),
#                   batch_size=64, optimizer_cls=Adam, optim_args={'lr': 0.1}, loss_fun=MSELoss(reduction='none'),
#                   train_dataset=dset, val_dataset=dset, test_dataset=dset, _amp=False)
#
# with trainer.quiet:
#     trainer.train(10, val_after_train=True)
# trainer.save()
# with trainer.quiet:
#     trainer.train(10, val_after_train=True)
# trainer.test('test')
# trainer.plot_learning_curves(lib='auto')
# trainer.load(10)
