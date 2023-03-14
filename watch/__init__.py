"""
The SMART WATCH module

Useful environs:
    DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
    DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)
"""
import os
import ubelt as ub
import warnings


__version__ = '0.4.0'
# git shortlog -e --summary --numbered
# ~/code/watch/dev/maintain/generate_authors.py
__author__ = 'WATCH Developers, Kitware Inc., Jon Crall, David Joy, Matthew Bernstein, Benjamin Brodie, Usman Rafique, Jacob DeRosa, Connor Greenwell, Peri Akiva, Matthew Purri, Ajay Upadhyaya'
__author_email__ = 'kitware@kitware.com, jon.crall@kitware.com'
__url__ = 'https://gitlab.kitware.com/watch/watch'


os.environ['USE_PYGEOS'] = '0'


WATCH_PREIMPORT_VARIANTS = {
    'variant1': ['geopandas', 'pyproj', 'gdal'],  # align-crs on horologic
    'variant2': ['pyproj', 'gdal'],               # CI machine
    'variant3': ['geopandas', 'pyproj'],          # delay gdal import
    'none': [],                                   # no pre-imports
    '0': [],                                      # no pre-imports
}


def _handle_hidden_commands():
    """
    Hidden developer features
    """
    if ub.argflag('--warntb') or os.environ.get('WARN_WITH_TRACEBACK', ''):
        import xdev
        xdev.make_warnings_print_tracebacks()


def _import_troublesome_module(modname):
    """
    Defines exactly how to import each troublesome (binary) module
    """
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
    # elif modname == 'pygeos':
    #     import pygeos as module
    elif modname == 'torch':
        import torch as module
    elif modname == 'numpy':
        import numpy as module
    else:
        raise KeyError(modname)
    return module


def _is_running_a_fast_cli_tool():
    """
    Determine if we are running a fast-cli tool.

    This is used to short circuit the pre-imports for certain command line
    tools so we don't incur the import time, which can be multiple seconds.

    TODO:
        Is there a better way to prevent certain entry points from executing
        the pre-imports?

    Notes:
        There is crazy sys.argv behavior with -m [SO42076706]_.

    References:
        .. [SO42076706] https://stackoverflow.com/questions/42076706/sys-argv-behavior-with-python-m
    """
    import sys
    if sys.argv and 'smartwatch_dvc' in sys.argv[0]:
        return True
    return False


def _execute_ordered_preimports():
    """
    The order in which certain modules with binary libraries are imported can
    impact runtime stability and can even cause crashes if a specific order
    isnt used.

    There are several known good configurations registered in the
    ``WATCH_PREIMPORT_VARIANTS`` dictionary and the setting is
    controlled by the ``WATCH_PREIMPORT`` environment variable.
    """

    # Shorter alias because we are using it now
    WATCH_PREIMPORT = os.environ.get('WATCH_PREIMPORT', 'auto')
    WATCH_PREIMPORT = os.environ.get('WATCH_HACK_IMPORT_ORDER', WATCH_PREIMPORT)

    if not WATCH_PREIMPORT:
        return

    if WATCH_PREIMPORT == 'auto':
        if _is_running_a_fast_cli_tool():
            watch_preimport = None
        else:
            # This is the "known" best order for importing
            watch_preimport = WATCH_PREIMPORT_VARIANTS['variant1']
    elif WATCH_PREIMPORT in WATCH_PREIMPORT_VARIANTS:
        watch_preimport = WATCH_PREIMPORT_VARIANTS[WATCH_PREIMPORT]
    elif WATCH_PREIMPORT.lower() in {'0', 'false', 'no', ''}:
        watch_preimport = None
    else:
        watch_preimport = WATCH_PREIMPORT.split(',')

    if watch_preimport is not None:
        for modname in watch_preimport:
            _import_troublesome_module(modname)


_handle_hidden_commands()
_execute_ordered_preimports()

from watch.monkey import monkey_numpy  # NOQA
monkey_numpy.patch_numpy_dtypes()


if 'hard-to-inspect-key' in vars():
    # Defined to hack jon's editor into autocompleting this.
    find_dvc_dpath = None


__devnotes__ = """

# Command to autogenerate lazy imports for this file
mkinit -m watch --lazy --diff
mkinit -m watch --lazy -w

# Debug import time
python -X importtime -c "import watch"
WATCH_HACK_IMPORT_ORDER=variant3 python -X importtime -c "import watch"
WATCH_HACK_IMPORT_ORDER=variant1 python -X importtime -c "import watch"
"""


# Choose which submodules (and which submodule attributes) to expose
__submodules__ = {
    '*': [],  # include all modules, but don't expose attributes
    'demo': ['coerce_kwcoco'],
    'utils': ['find_smart_dvc_dpath', 'find_dvc_dpath']
}


# --- AUTOGENERATED LAZY IMPORTS ---


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
        for name in submodules:
            __getattr__(name)

        for attrs in submod_attrs.values():
            for attr in attrs:
                __getattr__(attr)
    return __getattr__


__getattr__ = lazy_import(
    __name__,
    submodules={
        'cli',
        'demo',
        'exceptions',
        'gis',
        'heuristics',
        'rc',
        'stac',
        'tasks',
        'utils',
    },
    submod_attrs={
        'demo': [
            'coerce_kwcoco',
        ],
        'utils': [
            'find_smart_dvc_dpath',
            'find_dvc_dpath',
        ],
    },
)


def __dir__():
    return __all__


__all__ = ['WATCH_PREIMPORT_VARIANTS', 'cli', 'coerce_kwcoco', 'demo',
           'exceptions', 'find_smart_dvc_dpath', 'find_dvc_dpath',
           'gis', 'heuristics', 'rc', 'stac', 'tasks', 'utils']
