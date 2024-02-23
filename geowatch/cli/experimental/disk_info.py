#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class DiskInfoCLI(scfg.DataConfig):
    """
    Get information about the underlying storage medium used by a path.
    """
    dpath = scfg.Value('.', help='The path to inspect')
    key = scfg.Value(None, help=ub.paragraph(
        '''
        If specified print only the value of this key of interest, otherwise
        print all info. Useful keys are "hwtype", "filesystem".
        '''))
    verbose = scfg.Value(0, help='verbosity level')

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> from geowatch.cli.experimental.disk_info import *  # NOQA
            >>> cmdline = 0
            >>> kwargs = dict()
            >>> cls = DiskInfoCLI
            >>> cls.main(cmdline=cmdline, **kwargs)
        """
        import rich
        config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
        if config.verbose:
            rich.print('config = ' + ub.urepr(config, nl=1))

        from geowatch.utils.util_hardware import disk_info_of_path
        info = disk_info_of_path('.')

        if config.key is None:
            rich.print(f'info = {ub.urepr(info, nl=1)}')
        else:
            print(info[config.key])

__cli__ = DiskInfoCLI
main = __cli__.main

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/geowatch/geowatch/cli/experimental/disk_info.py
        python -m geowatch.cli.experimental.disk_info --key hwtype
    """
    main()
