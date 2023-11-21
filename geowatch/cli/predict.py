#!/usr/bin/env python3
import scriptconfig as scfg


class PredictCLI(scfg.DataConfig):
    """
    Does not work from geowatch CLI yet. See help.
    Use ``python -m geowatch.tasks.fusion.predict`` instead for now.

    ..code:: bash

        python -m geowatch.tasks.fusion.predict --help

    Execute fusion predict
    """
    __command__ = 'predict'
    __alias__ = ['fusion_predict']

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        """
        """
        from geowatch.tasks.fusion import predict
        return predict.main(cmdline=cmdline, **kwargs)


__cli__ = PredictCLI
main = __cli__.main

if __name__ == '__main__':
    main()
