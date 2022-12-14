"""
Notes about lightning CLI.


* I want to add something that prints where the training information is being
written. I often run multiple training runs in multiple tmux sessions and it
helps to be able to copy / paste a directory to go look at logs. Currently,
when you create a LightningCLI object it just starts to execute. It would be
nice if it separated itself into a setup step and then an execute step. It
seems like it might sort of do that, but it's unclear how to actually run the
execute step if "run=False"

* Using `fast_dev_run` doesn't seem to output anything to disk. That's a
problem for me. I want disk output.

* While jsonargparse is cool, it would be nice if it could be combined with a
scriptconfig way of handling kwargs. This would mean the docs could be more
tightly coupled with the parameters themselves, and we could get scriptconfig
style booleans, which I am a fan of. I don't like having to specify docs in the
docstring and list out the argument names in the signature and then in the
docstring too (at least in the case where there can be a ton of args).
We might workaround this by having scriptconfig be able to generate part of the
docstring for the class that uses it.

References:
    https://pytorch-lightning.readthedocs.io/en/latest/cli/lightning_cli_advanced_3.html

    https://pytorch-lightning.readthedocs.io/en/latest/advanced/model_parallel.html?highlight=deepspeed#deepspeed
"""

from pytorch_lightning.cli import LightningCLI


class LightningCLI_Extension(LightningCLI):
    ...


def main():
    pass
