import pytorch_lightning as pl
import netharn as nh
import torch


#
# TODO: expose as a toydata module
class LightningToyNet2d(pl.LightningModule):
    def __init__(self, num_train=100, num_val=10, batch_size=4):
        super().__init__()
        self.num_train = num_train
        self.num_val = num_val
        self.batch_size = batch_size
        self.model = nh.models.ToyNet2d()

    def forward(self, x):
        return self.model(x)

    def forward_step(self, batch, batch_idx):
        if self.trainer is None:
            stage = 'disconnected'
        else:
            stage = self.trainer.state.stage.lower()
        inputs, targets = batch
        logits = self.forward(inputs)
        loss = torch.nn.functional.nll_loss(logits.log_softmax(dim=1), targets)
        # https://pytorch-lightning.readthedocs.io/en/latest/extensions/logging.html
        self.log(f'{stage}_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        return self.forward_step(batch, batch_idx)

    def training_step(self, batch, batch_idx):
        return self.forward_step(batch, batch_idx)

    def train_dataloader(self):
        dataset = nh.data.toydata.ToyData2d(n=self.num_train)
        loader = dataset.make_loader(batch_size=self.batch_size)
        return loader

    def val_dataloader(self):
        dataset = nh.data.toydata.ToyData2d(n=self.num_val)
        loader = dataset.make_loader(batch_size=self.batch_size)
        return loader

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]
