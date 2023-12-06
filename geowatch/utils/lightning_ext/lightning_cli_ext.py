"""
This module is an exension of jsonargparse and lightning CLI that will respect
scriptconfig style arguments

References:
    https://github.com/Lightning-AI/lightning/issues/15038
"""
import jsonargparse
try:
    from pytorch_lightning.cli import ActionConfigFile
except Exception:
    from jsonargparse import ActionConfigFile  # NOQA
from pytorch_lightning.cli import LightningArgumentParser
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.cli import Namespace
from packaging.version import parse as Version
JSONARGPARSE_VERSION = Version(jsonargparse.__version__)

if JSONARGPARSE_VERSION < Version('4.21.0'):
    from geowatch.utils.lightning_ext import _jsonargparse_ext_ge_4_xx_and_lt_4_21 as _jsonargparse_ext
elif JSONARGPARSE_VERSION < Version('4.22.0'):
    from geowatch.utils.lightning_ext import _jsonargparse_ext_ge_4_21_and_lt_4_22 as _jsonargparse_ext
elif JSONARGPARSE_VERSION < Version('4.24.0'):
    from geowatch.utils.lightning_ext import _jsonargparse_ext_ge_4_22_and_lt_4_24 as _jsonargparse_ext
else:
    from geowatch.utils.lightning_ext import _jsonargparse_ext_ge_4_24_and_lt_4_xx as _jsonargparse_ext


class LightningArgumentParser_Extension(_jsonargparse_ext.ArgumentParserPatches, LightningArgumentParser):
    """
    CommandLine:
        xdoctest -m geowatch.utils.lightning_ext.lightning_cli_ext LightningArgumentParser_Extension

    Example:
        >>> from geowatch.utils.lightning_ext.lightning_cli_ext import *  # NOQA
        >>> LightningArgumentParser_Extension()

    Refactor references:
        ~/.pyenv/versions/3.10.5/envs/pyenv3.10.5/lib/python3.10/site-packages/pytorch_lightning/cli.py
        ~/.pyenv/versions/3.10.5/envs/pyenv3.10.5/lib/python3.10/site-packages/jsonargparse/core.py
        ~/.pyenv/versions/3.10.5/envs/pyenv3.10.5/lib/python3.10/site-packages/jsonargparse/signatures.py
    """
    ...


# Monkey patch jsonargparse so its subcommands use our extended functionality
jsonargparse.ArgumentParser = LightningArgumentParser_Extension


if JSONARGPARSE_VERSION < Version('4.22.0'):
    jsonargparse.core.ArgumentParser = LightningArgumentParser_Extension
    jsonargparse.core._find_action_and_subcommand = _jsonargparse_ext._find_action_and_subcommand
    jsonargparse.actions._find_action_and_subcommand = _jsonargparse_ext._find_action_and_subcommand
else:
    jsonargparse._core.ArgumentParser = LightningArgumentParser_Extension
    jsonargparse._core._find_action_and_subcommand = _jsonargparse_ext._find_action_and_subcommand
    jsonargparse._actions._find_action_and_subcommand = _jsonargparse_ext._find_action_and_subcommand


# Should try to patch into upstream
class LightningCLI_Extension(LightningCLI):
    """
    Our customized :class:`LightningCLI` extension.
    """
    ...

    def init_parser(self, **kwargs):
        # Hack in our modified parser
        DEBUG = 0
        if DEBUG:
            kwargs['error_handler'] = None
        import pytorch_lightning as pl
        kwargs.setdefault("dump_header", [f"pytorch_lightning=={pl.__version__}"])
        parser = LightningArgumentParser_Extension(**kwargs)
        parser.add_argument(
            "-c", "--config", action=ActionConfigFile,
            help="Path to a configuration file in json or yaml format."
        )
        return parser

    def parse_arguments(self, parser: LightningArgumentParser, args) -> None:
        """Parses command line arguments and stores it in ``self.config``."""
        import sys
        if args is not None and len(sys.argv) > 1:
            # Please let us shoot ourselves in the foot.
            from pytorch_lightning.utilities.rank_zero import rank_zero_warn
            # import warnings
            rank_zero_warn(
                "LightningCLI's args parameter is intended to run from within Python like if it were from the command "
                "line. To prevent mistakes it is not recommended to provide both args and command line arguments, got: "
                f"sys.argv[1:]={sys.argv[1:]}, args={args}."
            )
        if isinstance(args, (dict, Namespace)):
            self.config = parser.parse_object(args)
        else:
            self.config = parser.parse_args(args)
