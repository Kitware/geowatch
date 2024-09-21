#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""
Exposes the mlops tools in the CLI
"""
from scriptconfig.modal import ModalCLI
import ubelt as ub
from geowatch.mlops import manager
from geowatch.mlops import aggregate
from geowatch.mlops import schedule_evaluation


modal = ModalCLI(description=ub.codeblock(
    '''
    MLOPs CLI
    '''))
modal.__command__ = 'mlops'
# modal.__group__ = 'learning'


modal.register(manager.__cli__)
modal.register(aggregate.__cli__)
modal.register(schedule_evaluation.__cli__)


def main(cmdline=None, **kwargs):
    return modal.run(strict=True)


__cli__ = modal
__cli__.main = main


if __name__ == '__main__':
    main()
