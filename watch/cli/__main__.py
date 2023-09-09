#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
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
    modnames = [
        'watch_coco_stats',
        'geojson_site_stats',
        'torch_model_stats',
        'coco_spectra',

        # 'geotiffs_to_kwcoco'  # TODO: cleanup and add

        'coco_visualize_videos',
        'gifify',

        'find_dvc',

        'coco_add_watch_fields',
        'coco_align',
        'coco_clean_geotiffs',
        'coco_average_features',
        'coco_time_combine',

        'reproject_annotations',

        'kwcoco_to_geojson',
        'run_metrics_framework',

        'crop_sites_to_regions',
        'coco_remove_bad_images',
        'coco_average_features',

        # 'mlops_cli',
        'special.finish_install',
    ]

    cmd_alias = {
        'watch.cli.torch_model_stats': ['model_stats', 'model_info'],
        'watch.cli.geojson_site_stats': ['site_stats', 'geojson_stats', 'geomodel_stats'],
        'watch.cli.watch_coco_stats': ['stats'],
        'watch.cli.coco_visualize_videos': ['visualize'],
        'watch.cli.coco_align': ['align', 'coco_align_geotiffs'],
        'watch.cli.reproject_annotations': ['reproject', 'project'],
        'watch.cli.coco_add_watch_fields': ['add_fields'],
        'watch.cli.coco_spectra': ['spectra', 'intensity_histograms'],
        'watch.cli.run_metrics_framework': ['iarpa_eval'],
        'watch.cli.coco_clean_geotiffs': ['clean_geotiffs'],
        'watch.cli.kwcoco_to_geojson': ['run_tracker'],
        'watch.cli.find_dvc': ['dvc', 'dvcdir'],
        'watch.cli.gifify': ['animate'],
        'watch.cli.coco_average_features': ['average_features', 'ensemble'],
        'watch.cli.coco_time_combine': ['time_combine'],
        'watch.cli.crop_sites_to_regions': ['crop_sitemodels'],
        'watch.cli.coco_remove_bad_images': ['remove_bad_images'],
        'watch.cli.mlops_cli': ['mlops'],
        'watch.cli.special.finish_install': [],
    }

    module_lut = {}
    for name in modnames:
        mod = ub.import_module_from_name('watch.cli.{}'.format(name))
        module_lut[name] = mod

    module_lut['schedule'] = ub.import_module_from_name('watch.mlops.schedule_evaluation')
    module_lut['manager'] = ub.import_module_from_name('watch.mlops.manager')
    module_lut['aggregate'] = ub.import_module_from_name('watch.mlops.aggregate')
    module_lut['repackage'] = ub.import_module_from_name('watch.mlops.repackager')
    # module_lut['fit'] = ub.import_module_from_name('watch.tasks.fusion.fit_lightning')
    # module_lut['predict'] = ub.import_module_from_name('watch.tasks.fusion.predict')

    # Create a list of all submodules with CLI interfaces
    cli_modules = list(module_lut.values())

    import os
    from scriptconfig.modal import ModalCLI
    import watch
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

        Version: {watch.__version__}
        ''')

    FUN = os.getenv('FUN', '') and not os.getenv('NOFUN', '')
    if FUN:
        description = fun_header + '\n' + boring_description
    else:
        description = boring_description

    modal = ModalCLI(description)

    # scriptconfig bug made this not work...
    # modal.__class__.version = watch.__version__

    def get_version(self):
        import watch
        return watch.__version__
    modal.__class__.version = property(get_version)

    for cli_module in cli_modules:

        cli_subconfig = None
        if not hasattr(cli_module, '__config__'):
            if hasattr(cli_module, 'modal'):
                continue
            raise AssertionError('We are only supporting scriptconfig CLIs')
        # scriptconfig cli pattern
        cli_subconfig = cli_module.__config__

        if not hasattr(cli_subconfig, 'main'):
            if hasattr(cli_module, 'main'):
                main_func = cli_module.main
                # Hack the main function into the config
                cli_subconfig.main = main_func
            else:
                raise AssertionError(f'No main function for {cli_module}')

        # Update configs to have aliases / commands attributes
        cli_modname = cli_module.__name__
        cli_rel_modname = cli_modname.split('.')[-1]

        cmdname_aliases = ub.oset()
        alias = getattr(cli_subconfig, '__alias__', [])
        if isinstance(alias, str):
            alias = [alias]
        command = getattr(cli_module, '__command__', None)
        command = getattr(cli_subconfig, '__command__', command)
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
        cli_subconfig.__command__ = primary_cmdname
        cli_subconfig.__alias__ = list(secondary_cmdnames)
        modal.register(cli_subconfig)

    ret = modal.run(strict=not WATCH_LOOSE_CLI)
    return ret


if __name__ == '__main__':
    """
    CommandLine:
        python -m watch.cli --help
        python -m watch.cli.watch_coco_stats
    """
    main()
    # sys.exit(main())
