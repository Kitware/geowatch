"""
Exposes the mlops tools in the CLI
"""
from scriptconfig.modal import ModalCLI
import ubelt as ub
from watch.mlops import manager
from watch.mlops import aggregate
from watch.mlops import schedule_evaluation


modal = ModalCLI(description=ub.codeblock(
    '''
    MLOPs CLI
    '''))
modal.__command__ = 'mlops'
modal.__group__ = 'learning'


modal.register(manager.__config__)
modal.register(aggregate.__config__)
modal.register(schedule_evaluation.__config__)


def main(cmdline=None, **kwargs):
    return modal.run(strict=True)


__config__ = modal
__config__.main = main


if __name__ == '__main__':
    main()
