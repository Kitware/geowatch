import pytorch_lightning as pl
import netharn as nh
import torch


class LightningToyNet2d(pl.LightningModule):
    """
    Toydata lightning module
    """

    def __init__(self, num_train=100, num_val=10, batch_size=4):
        super().__init__()
        self.save_hyperparameters()
        self.num_train = num_train
        self.num_val = num_val
        self.batch_size = batch_size
        self.model = nh.models.ToyNet2d()

    def forward(self, x):
        return self.model(x)

    def get_cfgstr(self):
        return 'This is for BatchPlotter'

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

    Notes wrt to the trainer:

    ~/.pyenv/versions/3.8.6/envs/pyenv3.8.6/lib/python3.8/site-packages/pytorch_lightning/__init__.py

    ~/.pyenv/versions/3.8.6/envs/pyenv3.8.6/lib/python3.8/site-packages/pytorch_lightning/trainer/trainer.py

    ~/.pyenv/versions/3.8.6/envs/pyenv3.8.6/lib/python3.8/site-packages/pytorch_lightning/trainer/connectors/logger_connector/logger_connector.py

    ~/.pyenv/versions/3.8.6/envs/pyenv3.8.6/lib/python3.8/site-packages/pytorch_lightning/callbacks/model_checkpoint.py

    ~/.pyenv/versions/3.8.6/envs/pyenv3.8.6/lib/python3.8/site-packages/pytorch_lightning/trainer/connectors/checkpoint_connector.py


    Example:
        >>> # xdoctest: +SKIP
        >>> from geowatch.utils.lightning_ext.demo import *  # NOQA
        >>> trainer = demo_trainer()
        >>> print('trainer.log_dir = {!r}'.format(trainer.log_dir))
        >>> trainer.fit(trainer.model)
        >>> print('trainer.log_dir = {!r}'.format(trainer.log_dir))

    """
    import ubelt as ub
    default_root_dir = ub.Path.appdir('lightning_ext/demo_trainer').ensuredir()
    model = LightningToyNet2d(num_train=55)

    from geowatch.utils import lightning_ext as pl_ext
    kwargs = {}

    callbacks = [
        pl_ext.callbacks.AutoResumer(),
        pl_ext.callbacks.StateLogger(),
        pl_ext.callbacks.BatchPlotter(
            num_draw=kwargs.get('num_draw', 4),
            draw_interval=kwargs.get('draw_interval', '10m'),
        ),
        pl_ext.callbacks.TensorboardPlotter(),  # draw tensorboard
        pl.callbacks.LearningRateMonitor(logging_interval='epoch', log_momentum=True),
        pl.callbacks.LearningRateMonitor(logging_interval='step', log_momentum=True),

        pl.callbacks.ModelCheckpoint(monitor='train_loss', mode='min', save_top_k=2),
        # pl.callbacks.GPUStatsMonitor(),
    ]
    # # if args.vali_dataset is not None:
    # if kwargs.get('vali_dataset', None) is not None:
    #     callbacks += [
    #         pl.callbacks.EarlyStopping(
    #             monitor='val_loss', mode='min',
    #             patience=kwargs.get('patience', 10),
    #             verbose=True),
    #         pl.callbacks.ModelCheckpoint(
    #             monitor='val_loss', mode='min', save_top_k=2),
    #     ]
    # kwargs['callbacks'] = callbacks

    trainer = pl.Trainer(default_root_dir=default_root_dir, max_epochs=100,
                         callbacks=callbacks, accelerator='cpu', devices=1)
    trainer.model = model
    return trainer
