import pytorch_lightning as pl
import netharn as nh
import torch


class LightningToyNet2d(pl.LightningModule):
    """
    Toydata lightning module
    """

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
        loader = dataset.make_loader(batch_size=self.batch_size, num_workers=0)
        return loader

    def val_dataloader(self):
        dataset = nh.data.toydata.ToyData2d(n=self.num_val)
        loader = dataset.make_loader(batch_size=self.batch_size, num_workers=0)
        return loader

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]


def demo_trainer():
    """
    Example:
        >>> from watch.utils.lightning_ext.demo import *  # NOQA
        >>> trainer = demo_trainer()
        >>> print('trainer.train_dpath = {!r}'.format(trainer.train_dpath))
        >>> print('trainer.log_dir = {!r}'.format(trainer.log_dir))
        >>> trainer.fit(trainer.model)
        >>> train_dpath = trainer.logger.log_dir
        >>> print('trainer.log_dir = {!r}'.format(trainer.log_dir))

    """
    import ubelt as ub
    from watch.utils.lightning_ext.trainer import kitware_trainer
    default_root_dir = ub.ensure_app_cache_dir('lightning_ext/demo_trainer')
    model = LightningToyNet2d(num_train=55)
    trainer = kitware_trainer(default_root_dir=default_root_dir,
                              max_epochs=100)
    trainer.model = model
    return trainer
