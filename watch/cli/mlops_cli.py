"""
Exposes the mlops tools in the CLI

BROKEN DO NOT USE
"""
from watch.mlops import expt_manager


__config__ = expt_manager.ExptManagerConfig
main = expt_manager.main


if __name__ == '__main__':
    main(cmdline=True)
