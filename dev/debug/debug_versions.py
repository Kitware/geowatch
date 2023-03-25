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

    versions = ub.udict(versions)
    remain = versions.copy()

    library_categories = {
        'kitware': {
            'scriptconfig',
            'kwarray',
            'kwimage',
            'kwimage_ext',
            'kwcoco',
            'kwplot',
            'delayed_image',
            'ndsampler',
            'cmd_queue',
            'torch_liberator',
            'netharn',
            'ubelt',
        },

        'numeric': {
            'scipy',
            'numpy',
            'dask',
            'pandas',
            'scikit_learn',
            'filterpy',
            'einops',
            'xarray',
            'numexpr',
        },

        'imaging': {
            'Pillow',
            'scikit_image',
            'tifffile',
            'opencv-python-headless',
        },

        'plotting': {
            'seaborn',
            'matplotlib',
            'dataframe_image',
            'PyQt5',
            'distinctipy',
        },

        'utils': {
            'jq',
            'rich',
            'textual',
            'parse',
            'pint',
        },

        'algorithms': {
            'networkx',
            'pygtrie',
            'xxhash',
            'blakce3',
        },

        'gis': {
            'rasterio',
            'geojson',
            'geopandas',
            'shapely',
            'mgrs',
            'pyproj',
            'fiona',
            'rtree',
            'affine',
            'rgd_client',
            'rgd_imagery_client',
            'utm',
        },

        'env': {
            'psutil',
            'py-cpuinfo',
            'codecarbon',
        },

        'testing': {
            'coverage',
            'xdoctest',
            'pytest',
            'pytest_cov',
        },

        'learning': {
            'torch',
            'torchvision',
            'torchmetrics',
            'pytorch_lightning',
            'torch_optimizer',
            'perceiver-pytorch',
            'reformer_pytorch',
            'performer_pytorch',
        },

        'development': {
            'xdev',
            'autopep8',
            'flake8',
            'timerit',
        }

    }
    grouped_libraries = ub.udict()
    for key, val in library_categories.items():
        remain = remain - val
        grouped_libraries[key] = versions & val

    grouped_libraries['other'] = remain

    group_hashes = grouped_libraries.map_values(lambda x: ub.hash_data(x)[0:8])
    full_hash = ub.hash_data(group_hashes)
    import rich
    rich.print('grouped_libraries = {}'.format(ub.urepr(grouped_libraries, nl=2, align=':')))
    rich.print('group_hashes = {}'.format(ub.urepr(group_hashes, nl=2, align=':')))
    rich.print(f'full_hash={full_hash}')

    from torch.utils import collect_env
    collect_env.main()

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/watch/dev/debug/debug_versions.py
    """
    main()
