"""
This file demonstrates how to write a new CLI script using scriptconfig

First copy this template to the file name of your choice in this folder and
manually register the name of the file in `watch.cli.__main__.py`. Then
fill in the content of the file.
"""
import ubelt as ub
import scriptconfig as scfg


class YourConfigName(scfg.Config):
    """
    Add some documentation about your program
    """
    default = {
        'arg1': scfg.Value('default_value', help='some documentation', position=1),
        'arg2': scfg.Value('default_value', help='some documentation'),
        'arg3': scfg.Value('default_value', help='some documentation'),
    }

    epilog = """
    Example Usage:
        watch-cli scriptconfig_cli_template --arg1=foobar
    """


def main(cmdline=False, **kwargs):
    """
    Write your program main function

    Example:
        >>> # Dont forget to add a doctest
        >>> kwargs = {
        ...     'arg1': 1,
        ...     'arg2': 3,
        ... }
        >>> main(**kwargs)
    """
    config = YourConfigName(data=kwargs, cmdline=cmdline)
    print('config = {}'.format(ub.repr2(dict(config), nl=1)))


_SubConfig = YourConfigName


if __name__ == '__main__':
    """
    python -m watch.cli.scriptconfig_cli_template arg1-is-positional
    python -m watch.cli.scriptconfig_cli_template --arg1=but-can-be-keyworded
    python -m watch.cli.scriptconfig_cli_template --arg1=not-a-priority last_spec_takes_priority
    python -m watch.cli.scriptconfig_cli_template last_spec_takes_priority --arg1=now-a-priority
    """
    main(cmdline=True)
