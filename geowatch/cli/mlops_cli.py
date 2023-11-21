"""
Exposes the mlops tools in the CLI
"""
# from scriptconfig.modal import ModalCLI
# import ubelt as ub
from geowatch.mlops.__main__ import main, modal
__config__ = modal

if __name__ == '__main__':
    main()
