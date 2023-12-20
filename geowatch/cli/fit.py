#!/usr/bin/env python3
# FIXME: broken
import scriptconfig as scfg


class FitCLI(scfg.DataConfig):
    """
    Does not work from geowatch CLI yet. See help.
    Use ``python -m geowatch.tasks.fusion.fit_lightning fit`` instead for now. See

    ..code:: bash

        python -m geowatch.tasks.fusion.fit_lightning fit --help

    Execute fusion fit
    """
    __command__ = 'fit'
    __alias__ = ['fusion_fit']

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        """
        """
        from geowatch.tasks.fusion import fit_lightning
        return fit_lightning.main(cmdline=cmdline, **kwargs)


__cli__ = FitCLI
main = __cli__.main

if __name__ == '__main__':
    main()
