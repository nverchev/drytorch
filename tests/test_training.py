from custom_trainer import Trainer, ConstantScheduler
from torch.utils.data import Dataset, DataLoader
import torch


class TestDataset(Dataset):

    def __len__(self):
        return 4 * 4096

    def __getitem__(self, item):
        return torch.FloatTensor([1]), torch.randn(1), item


class TestTrainer(Trainer):

    def loss(self, outputs, inputs, targets):
        return {'Criterion': torch.nn.functional.mse_loss(outputs, targets, reduction='none')}


class MyNet(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


dset = TestDataset()
loader = DataLoader(dset, batch_size=32)
model = MyNet()
device = torch.device('cpu')
optimizer_cls = torch.optim.Adam
optimizer_args = {'lr': 0.00001}
trainer = TestTrainer(model, device=device, exp_name='test_trainer', model_pardir='tests', train_loader=loader,
                      optimizer_cls=optimizer_cls, optim_args=optimizer_args, scheduler=ConstantScheduler(),
                      val_loader=loader)
trainer.quiet_mode= False
trainer.train(10, val_after_train=True)
