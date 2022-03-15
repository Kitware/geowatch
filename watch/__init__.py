"""
The SMART WATCH module
"""
import os
import ubelt as ub
import warnings


__devnotes__ = """

# Command to autogenerate lazy imports for this file
mkinit -m watch --lazy --diff
mkinit -m watch --lazy -w

# Debug import time
python -X importtime -c "import watch"
WATCH_HACK_IMPORT_ORDER=variant3 python -X importtime -c "import watch"
WATCH_HACK_IMPORT_ORDER=variant1 python -X importtime -c "import watch"
"""

if 1:
    # hack for sanity
    os.environ['KWIMAGE_DISABLE_TRANSFORM_WARNINGS'] = 'True'
    # os.environ['PROJ_DEBUG'] = '3'

WATCH_AUTOHACK_IMPORT_VARIANTS = {
    'variant1': ['geopandas', 'pyproj', 'gdal'],  # align-crs on horologic
    'variant2': ['pyproj', 'gdal'],   # CI machine
    'variant3': ['geopandas', 'pyproj'],   # delay gdal import
}

if ub.argflag('--warntb'):
    import xdev
    xdev.make_warnings_print_tracebacks()

WATCH_HACK_IMPORT_ORDER = os.environ.get('WATCH_HACK_IMPORT_ORDER', 'auto')


def _imoprt_hack(modname):
    if modname == 'gdal':
        from osgeo import gdal as module
    elif modname == 'pyproj':
        import pyproj as module
        from pyproj import CRS  # NOQA
    elif modname == 'geopandas':
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', (
                '.*is incompatible with the GEOS version '
                'PyGEOS was compiled with.*'))
            import geopandas as module
    elif modname == 'rasterio':
        import rasterio as module
    elif modname == 'fiona':
        import fiona as module
    elif modname == 'pygeos':
        import pygeos as module
    elif modname == 'torch':
        import torch as module
    elif modname == 'numpy':
        import numpy as module
    else:
        raise KeyError(modname)
    return module


def _execute_import_order_hacks(WATCH_HACK_IMPORT_ORDER):
    if WATCH_HACK_IMPORT_ORDER == 'auto':
        # Some imports need to happen in a specific order, otherwise we get crashes
        # This is very annoying
        # This is the "known" best order for importing
        watch_hack_import_order = WATCH_AUTOHACK_IMPORT_VARIANTS['variant1']
    elif WATCH_HACK_IMPORT_ORDER in WATCH_AUTOHACK_IMPORT_VARIANTS:
        watch_hack_import_order = WATCH_AUTOHACK_IMPORT_VARIANTS[WATCH_HACK_IMPORT_ORDER]
    elif WATCH_HACK_IMPORT_ORDER.lower() in {'0', 'false', 'no', ''}:
        watch_hack_import_order = None
    else:
        watch_hack_import_order = WATCH_HACK_IMPORT_ORDER.split(',')

    if watch_hack_import_order is not None:
        for modname in watch_hack_import_order:
            _imoprt_hack(modname)


if WATCH_HACK_IMPORT_ORDER:
    _execute_import_order_hacks(WATCH_HACK_IMPORT_ORDER)


__version__ = '0.2.1'

# Choose which submodules (and which submodule attributes) to expose
__submodules__ = {
    '*': [],  # include all modules, but don't expose attributes
    'demo': ['coerce_kwcoco'],
    'utils': ['find_smart_dvc_dpath']
}


def lazy_import(module_name, submodules, submod_attrs):
    import importlib
    import os
    name_to_submod = {
        func: mod for mod, funcs in submod_attrs.items()
        for func in funcs
    }

    def __getattr__(name):
        if name in submodules:
            attr = importlib.import_module(
                '{module_name}.{name}'.format(
                    module_name=module_name, name=name)
            )
        elif name in name_to_submod:
            submodname = name_to_submod[name]
            module = importlib.import_module(
                '{module_name}.{submodname}'.format(
                    module_name=module_name, submodname=submodname)
            )
            attr = getattr(module, name)
        else:
            raise AttributeError(
                'No {module_name} attribute {name}'.format(
                    module_name=module_name, name=name))
        globals()[name] = attr
        return attr

    if os.environ.get('EAGER_IMPORT', ''):
        for name in name_to_submod.values():
            __getattr__(name)

        for attrs in submod_attrs.values():
            for attr in attrs:
                __getattr__(attr)
    return __getattr__


__getattr__ = lazy_import(
    __name__,
    submodules={
        'cli',
        'datacube',
        'datasets',
        'demo',
        'gis',
        'heuristics',
        'models',
        'rc',
        'sequencing',
        'tasks',
        'utils',
        'validation',
    },
    submod_attrs={
        'demo': [
            'coerce_kwcoco',
        ],
        'utils': [
            'find_smart_dvc_dpath',
        ],
    },
)


def __dir__():
    return __all__

__all__ = ['cli', 'coerce_kwcoco', 'datacube', 'datasets', 'demo',
           'find_smart_dvc_dpath', 'gis', 'heuristics', 'models', 'rc',
           'sequencing', 'tasks', 'utils', 'validation']
