import sys
import ubelt as ub

__devnotes__ = """

# We may want to delay actual imports, gdal import time can be excessive

python -X importtime -c "import watch"
WATCH_HACK_IMPORT_ORDER="" python  -X importtime -m watch.cli find_dvc

"""


def main(cmdline=True, **kw):
    """
    The watch command line interface
    """
    import argparse

    modnames = [
        'coco_add_watch_fields',
        'coco_align_geotiffs',
        'watch_coco_stats',
        'project_annotations',
        'coco_visualize_videos',
        'coco_spectra',
        'find_dvc',
        'kwcoco_to_geojson',
        'run_metrics_framework',
        'torch_model_stats',
        # 'mlops_cli',
        'gifify',
    ]
    module_lut = {}
    for name in modnames:
        mod = ub.import_module_from_name('watch.cli.{}'.format(name))
        module_lut[name] = mod

    # Create a list of all submodules with CLI interfaces
    cli_modules = list(module_lut.values())

    # Create a subparser that uses the first positional argument to run one of
    # the previous CLI interfaces.
    class RawDescriptionDefaultsHelpFormatter(
            argparse.RawDescriptionHelpFormatter,
            argparse.ArgumentDefaultsHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        description='The SMART WATCH CLI',
        formatter_class=RawDescriptionDefaultsHelpFormatter,
    )
    parser.add_argument('--version', action='store_true',
                        help='show version number and exit')
    subparsers = parser.add_subparsers(help='specify a command to run')

    cmd_alias = {
        'watch.cli.torch_model_stats': ['model_stats', 'model_info'],
        'watch.cli.watch_coco_stats': ['stats'],
        'watch.cli.coco_visualize_videos': ['visualize'],
        'watch.cli.coco_align_geotiffs': ['align'],
        'watch.cli.project_annotations': ['project'],
        'watch.cli.coco_add_watch_fields': ['add_fields'],
        'watch.cli.coco_spectra': ['spectra', 'intensity_histograms'],
        # 'watch.cli.mlops_cli': ['mlops'],
        'watch.cli.run_metrics_framework': ['iarpa_eval'],
        'watch.cli.kwcoco_to_geojson': [],
        'watch.cli.find_dvc': ['dvcdir'],
        'watch.cli.gifify': ['animate'],
    }

    for cli_module in cli_modules:

        if hasattr(cli_module, '_SubConfig'):
            # scriptconfig cli pattern
            cli_subconfig = cli_module._SubConfig
        else:
            cli_subconfig = None
            if hasattr(cli_module, '__config__'):
                # scriptconfig cli pattern
                cli_subconfig = cli_module.__config__
            elif hasattr(cli_module, '_CLI'):
                # scriptconfig cli pattern
                cli_subconfig = cli_module._CLI
            else:
                cli_subconfig = None
        if hasattr(cli_subconfig, 'main'):
            main_func = cli_subconfig.main
        elif hasattr(cli_module, 'main'):
            main_func = cli_module.main
        else:
            main_func = None

        if main_func is None:
            raise AssertionError(f'No main function for {cli_module}')

        cli_modname = cli_module.__name__
        cli_rel_modname = cli_modname.split('.')[-1]

        cmdname_aliases = cmd_alias.get(cli_modname, []) + [cli_rel_modname]

        parserkw = {}
        primary_cmdname = cmdname_aliases[0]
        secondary_cmdnames = cmdname_aliases[1:]
        if secondary_cmdnames:
            parserkw['aliases']  = secondary_cmdnames

        if cli_subconfig is not None:
            # TODO: make subparser.add_parser args consistent with what
            # scriptconfig generates when parser=None
            subconfig = cli_subconfig()
            parserkw.update(subconfig._parserkw())
            parserkw['help'] = parserkw['description'].split('\n')[0]
            subparser = subparsers.add_parser(primary_cmdname, **parserkw)
            subparser = subconfig.argparse(subparser)
        else:
            subparser = subparsers.add_parser(primary_cmdname, help='opaque sub command')

        subparser.set_defaults(main=main_func)

    import os
    WATCH_LOOSE_CLI = os.environ.get('WATCH_LOOSE_CLI', '')
    if WATCH_LOOSE_CLI:
        ns = parser.parse_known_args()[0]
    else:
        ns = parser.parse_args()

    # TODO: need to make a nicer pattern for new CLI integration, but this
    # works for now
    kw = ns.__dict__
    # print('kw = {}'.format(ub.repr2(kw, nl=1)))

    if kw.pop('version'):
        import watch
        print(watch.__version__)
        return 0

    main = kw.pop('main', None)
    if main is None:
        parser.print_help()
        raise ValueError('no command given')
        return 1

    try:
        ret = main(cmdline=0, **kw)
    except Exception as ex:
        print('ERROR ex = {!r}'.format(ex))
        raise
        return 1
    else:
        if ret is None:
            ret = 0
        return ret


if __name__ == '__main__':
    """
    CommandLine:
        python -m watch.cli --help
        python -m watch.cli.watch_coco_stats
    """
    sys.exit(main())
