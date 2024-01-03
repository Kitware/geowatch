#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import ubelt as ub

__devnotes__ = """
# We may want to delay actual imports, gdal import time can be excessive

python -X importtime -c "import geowatch"
WATCH_HACK_IMPORT_ORDER="" python  -X importtime -m geowatch.cli find_dvc
"""


def main(cmdline=True, **kw):
    """
    The geowatch command line interface
    """
    modnames = [
        'watch_coco_stats',
        'geojson_site_stats',
        'validate_annotation_schemas',
        'torch_model_stats',
        'coco_spectra',

        # 'geotiffs_to_kwcoco'  # TODO: cleanup and add

        'coco_visualize_videos',

        'find_dvc',

        'coco_add_watch_fields',
        'coco_align',
        'coco_clean_geotiffs',
        'coco_average_features',
        'coco_time_combine',

        'reproject_annotations',

        'run_tracker',
        'run_metrics_framework',

        'crop_sites_to_regions',
        'coco_remove_bad_images',
        'coco_average_features',

        # Due to LightningCLI fit does not play nicely here.
        'fit',
        # Predict also has issues because of its heavy imports.
        'predict',

        # 'mlops_cli',
        'special.finish_install',
    ]

    cmd_alias = {
        'geowatch.cli.torch_model_stats': ['model_stats', 'model_info'],
        # 'geowatch.cli.geojson_site_stats': ['site_stats', 'geojson_stats', 'geomodel_stats'],
        'geowatch.cli.watch_coco_stats': ['stats'],
        'geowatch.cli.coco_visualize_videos': ['visualize'],
        'geowatch.cli.coco_align': ['align', 'coco_align_geotiffs'],
        'geowatch.cli.reproject_annotations': ['reproject', 'project'],
        'geowatch.cli.coco_add_watch_fields': ['add_fields'],
        'geowatch.cli.coco_spectra': ['spectra', 'intensity_histograms'],
        'geowatch.cli.run_metrics_framework': ['iarpa_eval'],
        'geowatch.cli.coco_clean_geotiffs': ['clean_geotiffs'],
        'geowatch.cli.run_tracker': ['run_tracker'],
        'geowatch.cli.find_dvc': ['dvc', 'dvcdir'],
        'geowatch.cli.coco_average_features': ['average_features', 'ensemble'],
        'geowatch.cli.coco_time_combine': ['time_combine'],
        'geowatch.cli.crop_sites_to_regions': ['crop_sitemodels'],
        'geowatch.cli.coco_remove_bad_images': ['remove_bad_images'],
        'geowatch.cli.mlops_cli': ['mlops'],
        'geowatch.cli.special.finish_install': [],
    }

    module_lut = {}
    for name in modnames:
        mod = ub.import_module_from_name('geowatch.cli.{}'.format(name))
        module_lut[name] = mod

    module_lut['schedule'] = ub.import_module_from_name('geowatch.mlops.schedule_evaluation')
    module_lut['manager'] = ub.import_module_from_name('geowatch.mlops.manager')
    module_lut['aggregate'] = ub.import_module_from_name('geowatch.mlops.aggregate')
    module_lut['repackage'] = ub.import_module_from_name('geowatch.mlops.repackager')
    # module_lut['fit'] = ub.import_module_from_name('geowatch.tasks.fusion.fit_lightning')
    # module_lut['predict'] = ub.import_module_from_name('geowatch.tasks.fusion.predict')

    # Create a list of all submodules with CLI interfaces
    cli_modules = list(module_lut.values())

    import os
    from scriptconfig.modal import ModalCLI
    import geowatch
    WATCH_LOOSE_CLI = os.environ.get('WATCH_LOOSE_CLI', '')

    # https://emojiterra.com/time/
    # Not sure how to make this consistent on different terminals
    fun_header = ub.codeblock(
        '''
        🛰️⌚🛰️           🧠       🛰️⌚🛰️
        🌍🌎🌏        👁️   👁️      🌏🌎🌍
        ''')

    boring_description =  ub.codeblock(
        f'''
        🌐🌐🌐 The GEO-WATCH CLI 🌐🌐🌐

        An open source research and production environment for image and video
        segmentation and detection with geospatial awareness.

        Developed by [link=https://www.kitware.com/]Kitware[/link]. Funded by the [link=https://www.iarpa.gov/research-programs/smart]IARPA SMART[/link] challenge.

        Version: {geowatch.__version__}
        ''')

    FUN = os.getenv('FUN', '') and not os.getenv('NOFUN', '')
    if FUN:
        description = fun_header + '\n' + boring_description
    else:
        description = boring_description

    modal = ModalCLI(description)

    # scriptconfig bug made this not work...
    # modal.__class__.version = geowatch.__version__

    def get_version(self):
        import geowatch
        return geowatch.__version__
    modal.__class__.version = property(get_version)

    for cli_module in cli_modules:

        cli_config = None

        if hasattr(cli_module, '__config__'):
            # New way
            cli_config = cli_module.__config__
        elif hasattr(cli_module, '__cli__'):
            # New way
            cli_config = cli_module.__cli__
        else:
            if hasattr(cli_module, 'modal'):
                continue
            raise AssertionError(f'We are only supporting scriptconfig CLIs. {cli_module} does not have __config__ attr')

        if not hasattr(cli_config, 'main'):
            if hasattr(cli_module, 'main'):
                main_func = cli_module.main
                # Hack the main function into the config
                cli_config.main = main_func
            else:
                raise AssertionError(f'No main function for {cli_module}')

        # Update configs to have aliases / commands attributes
        cli_modname = cli_module.__name__
        cli_rel_modname = cli_modname.split('.')[-1]

        cmdname_aliases = ub.oset()
        alias = getattr(cli_config, '__alias__', [])
        if isinstance(alias, str):
            alias = [alias]
        command = getattr(cli_module, '__command__', None)
        command = getattr(cli_config, '__command__', command)
        if command is not None:
            cmdname_aliases.add(command)
        cmdname_aliases.update(alias)
        cmdname_aliases.update(cmd_alias.get(cli_modname, []) )
        cmdname_aliases.add(cli_rel_modname)
        # parserkw = {}
        primary_cmdname = cmdname_aliases[0]
        secondary_cmdnames = cmdname_aliases[1:]
        if not isinstance(primary_cmdname, str):
            raise AssertionError(primary_cmdname)
        cli_config.__command__ = primary_cmdname
        cli_config.__alias__ = list(secondary_cmdnames)
        modal.register(cli_config)

    ret = modal.main(strict=not WATCH_LOOSE_CLI)
    return ret


# def modal_main(self, argv=None, strict=True, autocomplete='auto'):
#     """
#     Overwrite modal main to support Lightning CLI
#     """
#     if isinstance(self, type):
#         self = self()

#     parser = self.argparse()

#     if autocomplete:
#         try:
#             import argcomplete
#             # Need to run: "$(register-python-argcomplete xdev)"
#             # or activate-global-python-argcomplete --dest=-
#             # activate-global-python-argcomplete --dest ~/.bash_completion.d
#             # To enable this.
#         except ImportError:
#             argcomplete = None
#             if autocomplete != 'auto':
#                 raise
#     else:
#         argcomplete = None

#     if argcomplete is not None:
#         argcomplete.autocomplete(parser)

#     if strict:
#         ns = parser.parse_args(args=argv)
#     else:
#         ns, _ = parser.parse_known_args(args=argv)

#     kw = ns.__dict__

#     if kw.pop('version', None):
#         print(self.version)
#         return 0

#     sub_main = kw.pop('main', None)
#     if sub_main is None:
#         parser.print_help()
#         raise ValueError('no command given')
#         return 1

#     try:
#         ret = sub_main(cmdline=False, **kw)
#     except Exception as ex:
#         print('ERROR ex = {!r}'.format(ex))
#         raise
#         return 1
#     else:
#         if ret is None:
#             ret = 0
#         return ret


if __name__ == '__main__':
    """
    CommandLine:
        python -m geowatch.cli --help
        python -m geowatch.cli.watch_coco_stats
    """
    main()
    # sys.exit(main())
