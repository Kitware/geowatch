#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class MyNewConfig(scfg.DataConfig):
    repo_dpath = 'special:watch'


def main(cmdline=1, **kwargs):
    """
    Example:
        >>> # xdoctest: +SKIP
        >>> cmdline = 0
        >>> kwargs = dict(
        >>> )
        >>> main(cmdline=cmdline, **kwargs)
    """
    config = MyNewConfig.cli(cmdline=cmdline, data=kwargs, strict=True)
    print('config = ' + ub.urepr(dict(config), nl=1))

    if config.repo_dpath == 'special:watch':
        import watch
        repo_dpath = ub.Path(watch.__file__).parent.parent
    else:
        repo_dpath = ub.Path(config.repo_dpath)

    import xdev
    with xdev.ChDir(repo_dpath):
        setup = ub.import_module_from_path(repo_dpath / 'setup.py', index=0)

    reqs = setup.parse_requirements(repo_dpath / 'requirements.txt', versions=False)
    reqs = list(ub.oset([req.split(';')[0].split('[')[0].split('<')[0] for req in reqs]))

    versions = {}
    for pkgname in reqs:
        import pkg_resources
        try:
            version = pkg_resources.get_distribution(pkgname).version
        except pkg_resources.DistributionNotFound:
            versions[pkgname] = '<NotFound>'
        else:
            versions[pkgname] = version

    import rich
    rich.print('versions = {}'.format(ub.urepr(versions, nl=1, align=':')))

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/watch/dev/debug/debug_versions.py
    """
    main()
