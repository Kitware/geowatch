#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class FinishInstallCLI(scfg.DataConfig):
    """
    This script finishes the install of geowatch
    """
    __command__ = 'finish_install'

    strict = scfg.Value(False, isflag=True, help='if True, use strict versions')

    @classmethod
    def main(FinishInstallCLI, cmdline=1, **kwargs):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> cmdline = 0
            >>> kwargs = dict()
            >>> main(cmdline=cmdline, **kwargs)
        """
        import rich
        config = FinishInstallCLI.cli(cmdline=cmdline, data=kwargs, strict=True)
        rich.print('config = ' + ub.urepr(config, nl=1))

        # TODO: provide internal reference to requirements.txt
        command = 'pip install --prefer-binary GDAL>=3.4.1 --find-links https://girder.github.io/large_image_wheels'
        if config.strict:
            command = command.replace('>=', '==')
        ub.cmd(command, system=True, verbose=3)

        want_system_exes = ['ffmpeg']
        missing = []
        for exe in want_system_exes:
            found = ub.find_path(exe)
            if found is None:
                missing.append(exe)

        if missing:
            print('Warning: missing system packages: {missing}')


__cli__ = FinishInstallCLI
main = __cli__.main
__config__ = __cli__


if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/watch/watch/cli/special/finish_install.py
        python -m watch.cli.special.finish_install
    """
    main()
