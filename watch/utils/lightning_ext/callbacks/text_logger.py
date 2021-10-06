import sys
import pytorch_lightning as pl
import ubelt as ub
from typing import Dict, Any, Optional
import logging


class TextLogger(pl.callbacks.Callback):
    """
    Writes logging information to text files.

    Example:
        >>> #
        >>> from watch.utils.lightning_ext.callbacks.text_logger import *  # NOQA
        >>> from watch.utils.lightning_ext import demo
        >>> self = demo.LightningToyNet2d(num_train=55)
        >>> default_root_dir = ub.ensure_app_cache_dir('lightning_ext/tests/TextLogger')
        >>> #
        >>> trainer = pl.Trainer(callbacks=[TextLogger()],
        >>>                      default_root_dir=default_root_dir,
        >>>                      max_epochs=3)
        >>> trainer.fit(self)
        >>> text_logs = ub.readfrom(trainer.text_logger.log_fpath)
        >>> print(text_logs)
    """

    def __init__(self, args=None):
        self._log = None
        # Hack to log all args
        self.args = args

    # def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
    #     self._log('setup state _log')
    #     self._log('trainer.default_root_dir = {!r}'.format(trainer.default_root_dir))

    def teardown(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        self._log('teardown state _log')

    # def on_init_start(self, trainer: "pl.Trainer") -> None:
    #     # self._log('on_init_start')
    #     pass

    def on_init_end(self, trainer: "pl.Trainer") -> None:
        import pathlib
        self.log_dir = pathlib.Path(trainer.log_dir)
        self.log_fpath = self.log_dir / 'text_logs.log'
        self._log = _InstanceLogger.from_instance(trainer, self.log_fpath)
        self._log('on_init_end')
        self._log('sys.argv = {!r}'.format(sys.argv))
        trainer.text_logger = self
        if self.args is not None:
            self._log('args_dict = {}'.format(ub.repr2(self.args.__dict__, nl=1, sort=0)))

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._log('on_fit_start')

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._log('on_fit_end')

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._log('on_train_start')

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._log('on_train_end')

    def on_save_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]) -> dict:
        self._log('on_save_checkpoint - checkpoint = {}'.format(ub.repr2(checkpoint.keys(), nl=1)))

    def on_load_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", callback_state: Dict[str, Any]) -> None:
        self._log('on_load_checkpoint - callback_state = {}'.format(ub.repr2(callback_state.keys(), nl=1)))

    def on_sanity_check_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._log('on_sanity_check_start')

    def on_sanity_check_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._log('on_sanity_check_end')

    def on_keyboard_interrupt(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._log('on_keyboard_interrupt')
        self._log('KEYBOARD INTERUPT')
        self._log('trainer.default_root_dir = {!r}'.format(trainer.default_root_dir))
        self._log('trainer.log_dir = {!r}'.format(trainer.log_dir))

    def on_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._log.debug('on_epoch_start')

    def on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._log.debug('on_epoch_end')

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._log.debug('on_train_epoch_start')

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._log.debug('on_train_epoch_end')

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._log.debug('on_validation_epoch_end')

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._log.debug('on_validation_epoch_start')


class _InstanceLogger():
    """
    This wraps a Python logger and handlers and is targeted to a specific
    instance of an object.

    Example:
        >>> import pathlib
        >>> dpath = ub.ensure_app_cache_dir('watch/test/logger')
        >>> fpath = pathlib.Path(dpath) / 'mylog.log'
        >>> self = _InstanceLogger(fpath=fpath)
    """

    def __init__(self, name=None, fpath=None, verbose=1):
        from os.path import join
        import pathlib

        if name is None:
            name = self._instance_name(self)

        self.fpath = fpath
        self.verbose = verbose
        self.name = name

        _log = logging.getLogger(name)
        _log.propagate = False
        _log.setLevel(logging.DEBUG)

        f_formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
        s_formatter = logging.Formatter('%(levelname)s: %(message)s')

        if fpath is not None:
            # File handlers
            a_flog_fpath = pathlib.Path(fpath)

            history_dname = ('_' + a_flog_fpath.stem + '_history')

            flog_dpath = a_flog_fpath.parent / history_dname
            flog_dpath.mkdir(exist_ok=True, parents=True)

            # Add timestamped fpath write handler:
            # This file will be specific to this instance of the harness, which
            # means different intances of the harness wont clobber value here.
            flog_fname = '{}_{}{}'.format(
                a_flog_fpath.stem, ub.timestamp(), a_flog_fpath.suffix)
            w_flog_fpath = join(flog_dpath, flog_fname)
            w_handler = logging.FileHandler(w_flog_fpath, mode='w')
            w_handler.setFormatter(f_formatter)
            w_handler.setLevel(logging.DEBUG)

            # Add a simple root append handler:
            # This file is shared by all instances of the harness, so logs over
            # multiple starts and stops can be viewed in a consolidated file.
            a_flog_fpath = fpath
            a_handler = logging.FileHandler(a_flog_fpath, mode='a')
            a_handler.setFormatter(f_formatter)
            a_handler.setLevel(logging.DEBUG)
            _log.addHandler(w_handler)
            _log.addHandler(a_handler)

        # Add a stdout handler:
        # this allows us to print logging calls to the terminal
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(s_formatter)

        if verbose > 1:
            stdout_handler.setLevel(logging.DEBUG)
        else:
            stdout_handler.setLevel(logging.INFO)

        _log.addHandler(stdout_handler)

        # hack in attribute for internal use
        _log._stdout_handler = stdout_handler

        self._log = _log
        self.debug('Initialized logging')

    def _ensure_prog_newline(self):
        # TODO: this class should be able to see instance of
        # progress bars and update them so the stdout logger
        # doesnt clobber them
        pass
        # # Try and make sure the progress bar does not clobber log outputs.
        # # Only available with progiter. Not sure how to do with tqdm.
        # try:
        #     if self.epoch_prog is not None:
        #         self.epoch_prog.ensure_newline()
        #     if self.main_prog is not None:
        #         self.main_prog.ensure_newline()
        # except AttributeError:
        #     pass

    def log(self, msg, level='info'):
        """
        Logs a message with a specified verbosity level.

        Args:
            msg (str): an info message to log
            level (str): either info, debug, error, or warn
        """
        if level == 'info':
            self.info(msg)
        elif level == 'debug':
            self.debug(msg)
        elif level == 'error':
            self.error(msg)
        elif level == 'warn':
            self.warn(msg)
        else:
            raise KeyError(level)

    def __call__(self, msg):
        self.info(msg)

    def info(self, msg):
        """
        Writes an info message to the logs

        Args:
            msg (str): an info message to log
        """
        # if not self.preferences['colored']:
        #     msg = _strip_ansi(msg)
        self._ensure_prog_newline()
        if self._log:
            try:
                self._log.info(msg)
            except Exception:
                pass
        else:
            print(msg)

    def error(self, msg):
        """
        Writes an error message to the logs

        Args:
            msg (str): an error message to log
        """
        self._ensure_prog_newline()
        if self._log:
            msg = _strip_ansi(msg)
            self._log.error(msg)
        else:
            # if not self.preferences['colored']:
            #     msg = _strip_ansi(msg)
            print(msg)

    def warn(self, msg):
        """
        Writes a warning message to the logs

        Args:
            msg (str): a warning message to log
        """
        self._ensure_prog_newline()
        if self._log:
            msg = _strip_ansi(msg)
            self._log.warning(msg)
        else:
            # if not self.preferences['colored']:
            #     msg = _strip_ansi(msg)
            print(msg)

    def debug(self, msg):
        """
        Writes a debug message to the logs

        Args:
            msg (str): a debug message to log
        """
        if self._log:

            if self._log._stdout_handler.level <= logging.DEBUG:
                # Use our hacked attribute to ensure newlines if we are
                # writting debug info to stdout
                self._ensure_prog_newline()

            msg = _strip_ansi(str(msg))
            # Encode to prevent errors on windows terminals
            # On windows there is a sometimes a UnicodeEncodeError:
            # For more details see: https://wiki.python.org/moin/PrintFails
            if sys.platform.startswith('win32'):
                self._log.debug(msg.encode('utf8'))
            else:
                self._log.debug(msg)

    @classmethod
    def _instance_name(cls, instance):
        return instance.__class__.__name__ + ':' + str(id(instance))

    @classmethod
    def from_instance(cls, instance, fpath):
        """
        Construct a name from the instance
        """
        name = cls._instance_name(instance)
        self = cls(name, fpath=fpath)
        return self


def _strip_ansi(text):
    r"""
    Removes all ansi directives from the string.

    References:
        http://stackoverflow.com/questions/14693701/remove-ansi
        https://stackoverflow.com/questions/13506033/filtering-out-ansi-escape-sequences

    Examples:
        >>> line = '\t\u001b[0;35mBlabla\u001b[0m     \u001b[0;36m172.18.0.2\u001b[0m'
        >>> escaped_line = _strip_ansi(line)
        >>> assert escaped_line == '\tBlabla     172.18.0.2'
    """
    # ansi_escape1 = re.compile(r'\x1b[^m]*m')
    # text = ansi_escape1.sub('', text)
    # ansi_escape2 = re.compile(r'\x1b\[([0-9,A-Z]{1,2}(;[0-9]{1,2})?(;[0-9]{3})?)?[m|K]?')
    import re
    ansi_escape3 = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -/]*[@-~]', flags=re.IGNORECASE)
    text = ansi_escape3.sub('', text)
    return text