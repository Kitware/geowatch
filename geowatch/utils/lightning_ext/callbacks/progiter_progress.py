"""
ProgIter progress callback for lighting

Modified from the TQDM Progress Class, with this license:
# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
import math
import os
import sys
from typing import Any, Dict, Optional, Union

from pytorch_lightning.utilities.types import STEP_OUTPUT

# check if ipywidgets is installed before importing tqdm.auto
# to ensure it won't fail and a progress bar is displayed
from progiter import ProgIter

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.progress_bar import ProgressBar
from pytorch_lightning.utilities.rank_zero import rank_zero_debug

_PAD_SIZE = 5

_DEBUG = 0


class _ProgIter(ProgIter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Custom progiter progressbar in case we need to customize"""
        # this just to make the make docs happy, otherwise it pulls docs which has some issues...
        kwargs.pop('smoothing', 0)  # API compatibility, unhandled, could implement
        super().__init__(*args, **kwargs)

    def reset(self, total=None):
        """
        Resets the progress to the start optionally with a new length
        """
        self.total = total
        self._reset_internals()

    @property
    def disable(self):
        return not self.enabled

    @property
    def n(self):
        """
        Alias for `self._iter_idx`
        """
        return self._iter_idx

    @n.setter
    def n(self, value):
        """
        TQDM allows the user to set 'n' to control number of iterations
        """
        self._iter_idx = value


class ProgIterProgressBar(ProgressBar):
    r"""
    This is the default progress bar used by Lightning. It prints to ``stdout`` using the
    :mod:`progiter` package and shows up to four different bars:

        - **sanity check progress:** the progress during the sanity check run
        - **train progress:** shows the training progress. It will pause if validation starts and will resume
          when it ends, and also accounts for multiple validation runs during training when
          :paramref:`~pytorch_lightning.trainer.trainer.Trainer.val_check_interval` is used.
        - **validation progress:** only visible during validation;
          shows total progress over all validation datasets.
        - **test progress:** only active when testing; shows total progress over all test datasets.

    For infinite datasets, the progress bar never ends.

    If you want to customize the default ``progiter`` progress bars used by
    Lightning, you can override specific methods of the callback class and pass
    your custom implementation to the
    :class:`~pytorch_lightning.trainer.trainer.Trainer`.

    CommandLine:
        xdoctest -m geowatch.utils.lightning_ext.callbacks.progiter_progress ProgIterProgressBar

    Example:
        >>> from geowatch.utils.lightning_ext.callbacks.progiter_progress import *  # NOQA
        >>> from geowatch.utils.lightning_ext.demo import LightningToyNet2d
        >>> from geowatch.utils.lightning_ext.callbacks import StateLogger
        >>> import ubelt as ub
        >>> from geowatch.utils.lightning_ext.monkeypatches import disable_lightning_hardware_warnings
        >>> disable_lightning_hardware_warnings()
        >>> default_root_dir = ub.Path.appdir('geowatch/lightning_ext/test/progiter_progress')
        >>> default_root_dir.delete().ensuredir()
        >>> # Test starting a model without any existing checkpoints
        >>> self = ProgIterProgressBar(time_thresh=1.0)
        >>> trainer = pl.Trainer(default_root_dir=default_root_dir, callbacks=[
        >>>     self,
        >>>     StateLogger()
        >>> ], max_epochs=5, accelerator='cpu', devices=1)
        >>> #model = LightningToyNet2d(num_train=5_000, num_val=5_000)
        >>> model = LightningToyNet2d()
        >>> trainer.fit(model)

    Args:
        refresh_rate: Determines at which rate (in number of batches) the progress bars get updated.
            Set it to ``0`` to disable the display.
        process_position: Set this to a value greater than ``0`` to offset the progress bars by this many lines.
            This is useful when you have progress bars defined elsewhere and want to show all of them
            together. This corresponds to
            :paramref:`~pytorch_lightning.trainer.trainer.Trainer.process_position` in the
            :class:`~pytorch_lightning.trainer.trainer.Trainer`.
    """

    def __init__(self, refresh_rate: int = 1, process_position: int = 0, time_thresh: float = 1.0):
        super().__init__()
        self._refresh_rate = self._resolve_refresh_rate(refresh_rate)
        self._process_position = process_position
        self._enabled = True
        self._train_progress_bar: Optional[ProgIter] = None
        self._val_progress_bar: Optional[ProgIter] = None
        self._test_progress_bar: Optional[ProgIter] = None
        self._predict_progress_bar: Optional[ProgIter] = None
        self._time_thresh = time_thresh

    def __getstate__(self) -> Dict:
        # can't pickle the ProgIter objects
        return {k: v if not isinstance(v, ProgIter) else None for k, v in vars(self).items()}

    @property
    def train_progress_bar(self) -> ProgIter:
        if self._train_progress_bar is None:
            raise TypeError(f"The `{self.__class__.__name__}._train_progress_bar` reference has not been set yet.")
        return self._train_progress_bar

    @train_progress_bar.setter
    def train_progress_bar(self, bar: ProgIter) -> None:
        self._train_progress_bar = bar

    @property
    def val_progress_bar(self) -> ProgIter:
        if self._val_progress_bar is None:
            raise TypeError(f"The `{self.__class__.__name__}._val_progress_bar` reference has not been set yet.")
        return self._val_progress_bar

    @val_progress_bar.setter
    def val_progress_bar(self, bar: ProgIter) -> None:
        self._val_progress_bar = bar

    @property
    def test_progress_bar(self) -> ProgIter:
        if self._test_progress_bar is None:
            raise TypeError(f"The `{self.__class__.__name__}._test_progress_bar` reference has not been set yet.")
        return self._test_progress_bar

    @test_progress_bar.setter
    def test_progress_bar(self, bar: ProgIter) -> None:
        self._test_progress_bar = bar

    @property
    def predict_progress_bar(self) -> ProgIter:
        if self._predict_progress_bar is None:
            raise TypeError(f"The `{self.__class__.__name__}._predict_progress_bar` reference has not been set yet.")
        return self._predict_progress_bar

    @predict_progress_bar.setter
    def predict_progress_bar(self, bar: ProgIter) -> None:
        self._predict_progress_bar = bar

    @property
    def refresh_rate(self) -> int:
        return self._refresh_rate

    @property
    def process_position(self) -> int:
        return self._process_position

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self.refresh_rate > 0

    @property
    def is_disabled(self) -> bool:
        return not self.is_enabled

    def disable(self) -> None:
        self._enabled = False

    def enable(self) -> None:
        self._enabled = True

    def init_sanity_progiter(self) -> _ProgIter:
        """Override this to customize the ProgIter bar for the validation sanity run."""
        return _ProgIter(
            desc=self.sanity_check_description,
            # position=(2 * self.process_position),
            disable=self.is_disabled,
            # leave=False,
            # dynamic_ncols=True,
            file=sys.stdout,
            # homogeneous=False,
            time_thresh=self._time_thresh,
        )

    def init_train_progiter(self) -> _ProgIter:
        """Override this to customize the ProgIter bar for training."""
        return _ProgIter(
            desc=self.train_description,
            # position=(2 * self.process_position),
            disable=self.is_disabled,
            # leave=True,
            # dynamic_ncols=True,
            file=sys.stdout,
            # smoothing=0,
            homogeneous=False,
            time_thresh=self._time_thresh,
        )

    def init_predict_progiter(self) -> _ProgIter:
        """Override this to customize the ProgIter bar for predicting."""
        return _ProgIter(
            desc=self.predict_description,
            # position=(2 * self.process_position),
            disable=self.is_disabled,
            # leave=True,
            # dynamic_ncols=True,
            file=sys.stdout,
            # smoothing=0,
            homogeneous=False,
            time_thresh=self._time_thresh,
        )

    def init_validation_progiter(self) -> _ProgIter:
        """Override this to customize the ProgIter bar for validation."""
        # The train progress bar doesn't exist in `trainer.validate()`
        # has_main_bar = self.trainer.state.fn != "validate"
        return _ProgIter(
            desc=self.validation_description,
            # position=(2 * self.process_position + has_main_bar),
            disable=self.is_disabled,
            # leave=not has_main_bar,
            # dynamic_ncols=True,
            file=sys.stdout,
            homogeneous=False,
            time_thresh=self._time_thresh,
        )

    def init_test_progiter(self) -> _ProgIter:
        """Override this to customize the ProgIter bar for testing."""
        return _ProgIter(
            desc="Testing",
            # position=(2 * self.process_position),
            disable=self.is_disabled,
            # leave=True,
            # dynamic_ncols=True,
            file=sys.stdout,
            homogeneous=False,
            time_thresh=self._time_thresh,
        )

    def on_sanity_check_start(self, *_: Any) -> None:
        if _DEBUG:
            print('\n on_sanity_check_start \n')
        self.val_progress_bar = self.init_sanity_progiter()
        self.train_progress_bar = _ProgIter(disable=True)  # dummy progress bar

    def on_sanity_check_end(self, *_: Any) -> None:
        if _DEBUG:
            print('\n on_sanity_check_end \n')
        self.train_progress_bar.close()
        self.val_progress_bar.close()

    def on_train_start(self, *_: Any) -> None:
        if _DEBUG:
            print('\n on_train_start \n')
        self.train_progress_bar = self.init_train_progiter()

    def on_train_epoch_start(self, trainer: "pl.Trainer", *_: Any) -> None:
        if _DEBUG:
            print('\n on_train_epoch_start \n')
        self.train_progress_bar.reset(convert_inf(self.total_train_batches))
        self.train_progress_bar.begin()
        self.train_progress_bar.initial = 0
        self.train_progress_bar.set_description(f"Epoch {trainer.current_epoch}")

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        n = batch_idx + 1
        if self._should_update(n, self.train_progress_bar.total):
            _update_n(self.train_progress_bar, n)
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module), refresh=False)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if _DEBUG:
            print('\n on_train_epoch_end \n')
        if not self.train_progress_bar.disable:
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))
            self.train_progress_bar.close()

    def on_train_end(self, *_: Any) -> None:
        # self.train_progress_bar.close()
        if _DEBUG:
            print('\n on_train_end \n')

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if _DEBUG:
            print('\n on_validation_start \n')
        # Hack: on_train_epoch_end happens after on_validation_epoch_end, so
        # we close the train progress bar here if there is a validation set.
        if not self.train_progress_bar.disable:
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))
            self.train_progress_bar.close()

        if not trainer.sanity_checking:
            self.val_progress_bar = self.init_validation_progiter()

    def on_validation_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self.has_dataloader_changed(dataloader_idx):
            return

        self.val_progress_bar.reset(convert_inf(self.total_val_batches_current_dataloader))
        self.val_progress_bar.begin()
        self.val_progress_bar.initial = 0
        desc = self.sanity_check_description if trainer.sanity_checking else self.validation_description
        self.val_progress_bar.set_description(f"{desc} DataLoader {dataloader_idx}")

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        n = batch_idx + 1
        if self._should_update(n, self.val_progress_bar.total):
            _update_n(self.val_progress_bar, n)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if _DEBUG:
            print('\n on_validation_end \n')
        if self._train_progress_bar is not None and trainer.state.fn == "fit":
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))
        self.val_progress_bar.close()
        self.reset_dataloader_idx_tracker()

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.test_progress_bar = self.init_test_progiter()

    def on_test_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self.has_dataloader_changed(dataloader_idx):
            return

        self.test_progress_bar.reset(convert_inf(self.total_test_batches_current_dataloader))
        self.test_progress_bar.begin()
        self.test_progress_bar.initial = 0
        self.test_progress_bar.set_description(f"{self.test_description} DataLoader {dataloader_idx}")

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        n = batch_idx + 1
        if self._should_update(n, self.test_progress_bar.total):
            _update_n(self.test_progress_bar, n)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.test_progress_bar.close()
        self.reset_dataloader_idx_tracker()

    def on_predict_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.predict_progress_bar = self.init_predict_progiter()

    def on_predict_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self.has_dataloader_changed(dataloader_idx):
            return

        self.predict_progress_bar.reset(convert_inf(self.total_predict_batches_current_dataloader))
        self.predict_progress_bar.begin()
        self.predict_progress_bar.initial = 0
        self.predict_progress_bar.set_description(f"{self.predict_description} DataLoader {dataloader_idx}")

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        n = batch_idx + 1
        if self._should_update(n, self.predict_progress_bar.total):
            _update_n(self.predict_progress_bar, n)

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.predict_progress_bar.close()
        self.reset_dataloader_idx_tracker()

    def print(self, *args: Any, sep: str = " ", **kwargs: Any) -> None:
        active_progress_bar = None
        raise AssertionError

        if self._train_progress_bar is not None and not self.train_progress_bar.disable:
            active_progress_bar = self.train_progress_bar
        elif self._val_progress_bar is not None and not self.val_progress_bar.disable:
            active_progress_bar = self.val_progress_bar
        elif self._test_progress_bar is not None and not self.test_progress_bar.disable:
            active_progress_bar = self.test_progress_bar
        elif self._predict_progress_bar is not None and not self.predict_progress_bar.disable:
            active_progress_bar = self.predict_progress_bar

        if active_progress_bar is not None:
            s = sep.join(map(str, args))
            active_progress_bar.write(s, **kwargs)

    def _should_update(self, current: int, total: int) -> bool:
        # return self.is_enabled and (current % self.refresh_rate == 0 or current == total)
        return True

    @staticmethod
    def _resolve_refresh_rate(refresh_rate: int) -> int:
        if os.getenv("COLAB_GPU") and refresh_rate == 1:
            # smaller refresh rate on colab causes crashes, choose a higher value
            rank_zero_debug("Using a higher refresh rate on Colab. Setting it to `20`")
            return 20
        return refresh_rate


def convert_inf(x: Optional[Union[int, float]]) -> Optional[Union[int, float]]:
    """The tqdm doesn't support inf/nan values.

    We have to convert it to None.

    """
    if x is None or math.isinf(x) or math.isnan(x):
        return None
    return x


def _update_n(bar: ProgIter, value: int) -> None:
    if not bar.disable:
        bar.n = value
        bar._slow_path_step_body()
        # bar.refresh()
