import pytorch_lightning as pl
import ubelt as ub
from typing import Dict, Any, Optional


class StateLogger(pl.callbacks.Callback):
    """
    Prints out what callbacks are being called
    """

    def __init__(self):
        pass

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        print('setup state logger')
        print('trainer.default_root_dir = {!r}'.format(trainer.default_root_dir))

    def teardown(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        print('teardown state logger')

    def on_init_start(self, trainer: "pl.Trainer") -> None:
        print('on_init_start')

    def on_init_end(self, trainer: "pl.Trainer") -> None:
        print('on_init_start')

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print('on_fit_start')

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print('on_fit_end')

    # def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    #     print('on_train_start')

    # def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    #     print('on_train_end')

    def on_save_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]) -> dict:
        print('on_save_checkpoint - checkpoint = {}'.format(ub.repr2(checkpoint.keys(), nl=1)))

    def on_load_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", callback_state: Dict[str, Any]) -> None:
        print('on_load_checkpoint - callback_state = {}'.format(ub.repr2(callback_state.keys(), nl=1)))

    def on_sanity_check_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print('on_sanity_check_start')

    def on_sanity_check_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print('on_sanity_check_end')

    def on_keyboard_interrupt(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print('on_keyboard_interrupt')
        print('KEYBOARD INTERUPT')
        print('trainer.default_root_dir = {!r}'.format(trainer.default_root_dir))
        print('trainer.log_dir = {!r}'.format(trainer.log_dir))
