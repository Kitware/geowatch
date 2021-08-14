import pytorch_lightning as pl
import netharn as nh
import torch
from typing import Dict, Any, Optional


#
# TODO: expose as a toydata module
class LightningToyNet2d(pl.LightningModule):
    """
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


class KitwareCallbacks(pl.callbacks.Callback):
    """
    Available:
        on_configure_sharded_model(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        on_before_accelerator_backend_setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        on_init_start(self, trainer: "pl.Trainer") -> None:
        on_init_end(self, trainer: "pl.Trainer") -> None:
        on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        on_sanity_check_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        on_sanity_check_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int,) -> None:
        on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        on_train_epoch_end( self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", unused: Optional = None) -> None:

        on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        on_predict_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        on_predict_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: List[Any]) -> None:

        on_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        on_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        on_validation_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        on_validation_batch_end( self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int,) -> None:
        on_test_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        on_test_batch_end( self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int,) -> None:
        on_predict_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        on_predict_batchend(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int,) -> None:
        on_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        on_before_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", loss: torch.Tensor) -> None:
        on_after_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        on_before_optimizer_step(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer: Optimizer, opt_idx: int) -> None:
        on_before_zero_grad(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer: Optimizer) -> None:

        on_pretrain_routine_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        on_pretrain_routine_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        on_predict_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        on_keyboard_interrupt(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        on_save_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]) -> dict:
        on_load_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", callback_state: Dict[str, Any]) -> None:

        setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        teardown(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
    """
    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        print('setup kitware callbacks')
        print('trainer.train_dpath = {!r}'.format(trainer.train_dpath))

    def teardown(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        print('teardown kitware callbacks')

    def on_init_start(self, trainer: "pl.Trainer") -> None:
        print('on_init_start')

    def on_init_end(self, trainer: "pl.Trainer") -> None:
        print('on_init_start')

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print('on_train_start')

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print('on_train_end')

    def on_load_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", callback_state: Dict[str, Any]) -> None:
        print('on_load_checkpoint')

    def on_sanity_check_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print('on_sanity_check_start')

    def on_sanity_check_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print('on_sanity_check_end')

    def on_keyboard_interrupt(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print('on_keyboard_interrupt')
        print('KEYBOARD INTERUPT')
        print('trainer.train_dpath = {!r}'.format(trainer.train_dpath))
        print('trainer.log_dir = {!r}'.format(trainer.log_dir))


def kitware_trainer(*args, **kwargs):
    """
    Example:
        >>> from watch.tasks.fusion.lightning_extensions.demo import *  # NOQA
        >>> import ubelt as ub
        >>> default_root_dir = ub.ensure_app_cache_dir('lightning_ext/test/kwtrainer')
        >>> ub.delete(default_root_dir)
        >>> model = LightningToyNet2d(num_train=55)
        >>> trainer = kitware_trainer(default_root_dir=default_root_dir)
        >>> print('trainer.train_dpath = {!r}'.format(trainer.train_dpath))
        >>> print('trainer.log_dir = {!r}'.format(trainer.log_dir))
        >>> trainer.fit(model)
        >>> train_dpath = trainer.logger.log_dir
        >>> print('trainer.log_dir = {!r}'.format(trainer.log_dir))
    """

    # It seems we have to override the init, not sure
    resume_from_checkpoint = kwargs.get('resume_from_checkpoint', 'auto')
    train_dpath = kwargs.get('default_root_dir', None)
    assert train_dpath is not None, 'must specify'

    if resume_from_checkpoint == 'auto':
        resume_from_checkpoint = find_most_recent_checkpoint(train_dpath)
        kwargs['resume_from_checkpoint'] = resume_from_checkpoint

    callbacks = kwargs.get('callbacks', 'auto')
    if callbacks == 'auto':
        print('callbacks = {!r}'.format(callbacks))
        from watch.tasks.fusion.lightning_extensions.tensorboard_plotter import TensorboardPlotter  # NOQA
        from watch.tasks.fusion.lightning_extensions.draw_batch import DrawBatchCallback
        from pytorch_lightning.callbacks.early_stopping import EarlyStopping
        # callbacks = []
        # callbacks += [TensorboardPlotter()]
        # kwargs['callbacks'] = callbacks

        callbacks = [
            KitwareCallbacks(),
            DrawBatchCallback(
                num_draw=kwargs.get('num_draw', 4),
                draw_interval=kwargs.get('draw_interval', '10m'),
            ),
            TensorboardPlotter(),  # draw tensorboard
            pl.callbacks.LearningRateMonitor(logging_interval='epoch', log_momentum=True),
            pl.callbacks.LearningRateMonitor(logging_interval='step', log_momentum=True),

            pl.callbacks.ModelCheckpoint(monitor='train_loss', mode='min', save_top_k=2),
            # pl.callbacks.GPUStatsMonitor(),
        ]
        # if args.vali_dataset is not None:
        if kwargs.get('vali_dataset', None) is not None:
            callbacks += [
                EarlyStopping(
                    monitor='val_loss', mode='min',
                    patience=kwargs.get('patience', 10),
                    verbose=True),
                pl.callbacks.ModelCheckpoint(
                    monitor='val_loss', mode='min', save_top_k=2),
            ]

        kwargs['callbacks'] = callbacks

    # TODO: explititly initialize the tensorboard logger
    # logger = [
    #     pl.loggers.TensorBoardLogger(
    #         save_dir=args.default_root_dir, version=self.trainer.slurm_job_id, name="lightning_logs"
    #     )
    # ]

    trainer = pl.Trainer(*args, **kwargs)
    trainer.train_dpath = train_dpath

    return trainer


def demo_trainer():
    """
    Example:
        >>> from watch.tasks.fusion.lightning_extensions.demo import *  # NOQA
        >>> trainer = demo_trainer()
        >>> print('trainer.train_dpath = {!r}'.format(trainer.train_dpath))
        >>> print('trainer.log_dir = {!r}'.format(trainer.log_dir))
        >>> trainer.fit(trainer.model)
        >>> train_dpath = trainer.logger.log_dir
        >>> print('trainer.log_dir = {!r}'.format(trainer.log_dir))

    """
    import ubelt as ub
    default_root_dir = ub.ensure_app_cache_dir('lightning_ext/demo_trainer')
    model = LightningToyNet2d(num_train=55)
    trainer = kitware_trainer(default_root_dir=default_root_dir,
                              max_epochs=100)
    trainer.model = model
    return trainer


def ensurepath(path_like):
    import pathlib
    if isinstance(path_like, pathlib.Path):
        return path_like
    else:
        return pathlib.Path(path_like)


def find_most_recent_checkpoint(train_dpath):
    train_dpath = ensurepath(train_dpath)
    candidates = list(train_dpath.glob('*/*/checkpoints/*.ckpt'))
    if len(candidates):
        chosen = sorted(candidates)[-1]
    else:
        chosen = None
    return chosen
