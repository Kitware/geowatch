"""
The GeoWATCH module

+----------------------------------------------------------+----------------------------------------------------------------+
| The GeoWATCH Gitlab Repo                                 | https://gitlab.kitware.com/computer-vision/geowatch/           |
+----------------------------------------------------------+----------------------------------------------------------------+
| The GeoWATCH Github Repo (Mirror)                        | https://github.com/Kitware/geowatch/                           |
+----------------------------------------------------------+----------------------------------------------------------------+
| Pypi                                                     | https://pypi.org/project/geowatch/                             |
+----------------------------------------------------------+----------------------------------------------------------------+
| Read the docs                                            | https://geowatch.readthedocs.io                                |
+----------------------------------------------------------+----------------------------------------------------------------+
| Slides                                                   | `Software Overview Slides`_  and `KHQ Demo Slides`_            |
+----------------------------------------------------------+----------------------------------------------------------------+

.. _Software Overview Slides: https://docs.google.com/presentation/d/125kMWZIwfS85lm7bvvCwGAlYZ2BevCfBLot7A72cDk8/

.. _KHQ Demo Slides: https://docs.google.com/presentation/d/1HKH_sGJX4wH60j8t4iDrZN8nH71jGX1vbCXFRIDVI7c/


Main modules of interest are:

    * :mod:`geowatch.cli`

    * :mod:`geowatch.mlops`

    * :mod:`geowatch.tasks`

Main Tasks:

    * :mod:`geowatch.tasks.fusion`

    * :mod:`geowatch.tasks.tracking`

Supported Feature Tasks:

    * :mod:`geowatch.tasks.cold`

    * :mod:`geowatch.tasks.depth`

    * :mod:`geowatch.tasks.depth_pcd`

    * :mod:`geowatch.tasks.dino_detector`

    * :mod:`geowatch.tasks.invariants`

    * :mod:`geowatch.tasks.landcover`

    * :mod:`geowatch.tasks.mae`

    * :mod:`geowatch.tasks.rutgers_material_seg_v2`

    * :mod:`geowatch.tasks.sam`

Also see:

    * :mod:`geowatch.gis`

    * :mod:`geowatch.geoannots`

    * :mod:`geowatch.demo`

    * :mod:`geowatch.stac`

    * :mod:`geowatch.utils`

    * :mod:`geowatch.utils.lightning_ext`

    * :mod:`geowatch.utils.lightning_ext.callbacks`

You probably wont need:

    * :mod:`geowatch.rc`

    * :mod:`geowatch.monkey`


.. code::

    # Useful environs

    # Phase 3
    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware=auto)
    DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware=auto)
    echo "$DVC_DATA_DPATH"
    echo "$DVC_EXPT_DPATH"

    # Phase 2
    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
    DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
    HIGHRES_DVC_EXPT_DPATH=$(geowatch_dvc --tags='smart_drop7' --hardware=auto)
    DATA_DVC_DPATH=$DVC_DATA_DPATH
    EXPT_DVC_DPATH=$DVC_EXPT_DPATH

    # To get the above make sure you have run:
    geowatch_dvc add my_phase2_data_repo --path=<path-to-your-phase2-data-dvc-repo> --hardware=hdd --priority=100 --tags=phase2_data
    geowatch_dvc add my_phase2_expt_repo --path=<path-to-your-phase2-expt-dvc-repo> --hardware=hdd --priority=100 --tags=phase2_expt


.. code:: python

    # Useful Python Snippets
    import geowatch
    data_dvc_dpath = geowatch.find_dvc_dpath(tags='phase3_data', hardware='auto')
    expt_dvc_dpath = geowatch.find_dvc_dpath(tags='phase3_expt', hardware='auto')

"""
import os
import sys
import ubelt as ub
import warnings


__version__ = '0.19.0'


# ../dev/maintain/generate_authors.py
__author__ = 'GeoWATCH Developers, Kitware Inc., Jon Crall, David Joy, Matthew Bernstein, Connor Greenwell, Benjamin Brodie, Peri Akiva, Usman Rafique, Jacob DeRosa, Matthew Purri, Ajay Upadhyaya, Ji Won Suh, Jacob Birge, Ryan LaClair, Scott Workman, Dexter Lau, Sergii Skakun, Aram Ansary Ogholbake, Cohen Archbold, Bane Sullivan, Srikumar Sastry, Armin Hadzic'

__author_email__ = 'kitware@kitware.com, jon.crall@kitware.com'
__url__ = 'https://gitlab.kitware.com/computer-vision/geowatch'


os.environ['USE_PYGEOS'] = '0'


GEOWATCH_PREIMPORT_VARIANTS = {
    'variant1': ['geopandas', 'pyproj', 'gdal'],  # align-crs on horologic
    'variant2': ['pyproj', 'gdal'],               # CI machine
    'variant3': ['geopandas', 'pyproj'],          # delay gdal import
    'none': [],                                   # no pre-imports
    '0': [],                                      # no pre-imports
}


# Use this to debug import-time effects
_GEOWATCH_DEBUG = False


def _handle_hidden_commands():
    """
    Hidden developer features based on CLI flags or enviornment variables.
    The following table summarizes these options and their effect.

    +-------------------+------------------------+-----------------------------------------------+
    | CLI Flag          | Environment Variable   | Effect                                        |
    +-------------------+------------------------+-----------------------------------------------+
    | --warntb          | WARN_WITH_TRACEBACK    | Any warnings will emit a traceback to stderr  |
    +-------------------+------------------------+-----------------------------------------------+
    | --geowatch-debug  | _GEOWATCH_DEBUG        | Will print internal module setup info         |
    +-------------------+------------------------+-----------------------------------------------+

    """
    if ub.argflag('--geowatch-debug') or os.environ.get('_GEOWATCH_DEBUG', ''):
        global _GEOWATCH_DEBUG
        _GEOWATCH_DEBUG = 1
        if _GEOWATCH_DEBUG:
            print('Enabled geowatch debugging')

    if ub.argflag('--warntb') or os.environ.get('WARN_WITH_TRACEBACK', ''):
        import xdev
        xdev.make_warnings_print_tracebacks()

        if _GEOWATCH_DEBUG:
            print('Enable warning tracebacks')

    if _GEOWATCH_DEBUG:
        print('Finished handling hidden commands')


def _import_troublesome_module(modname):
    """
    Defines exactly how to import each troublesome (binary) module
    """
    if _GEOWATCH_DEBUG:
        print(f'Import troublesome module: {modname}')
    if modname == 'gdal':
        from osgeo import gdal as module
        gdal = module
        if not getattr(gdal, '_UserHasSpecifiedIfUsingExceptions', lambda: False)():
            if _GEOWATCH_DEBUG:
                print('Configuring GDAL to use exceptions')
            gdal.UseExceptions()
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
    Determine if we are running a fast cli tool.

    This is used to short circuit the pre-imports for certain command line
    tools so we don't incur the import time, which can be multiple seconds.

    TODO:
        Is there a better way to prevent certain entry points from executing
        the pre-imports?

    Notes:
        There is crazy sys.argv behavior with -m [SO42076706]_.

    References:
        .. [SO42076706] https://stackoverflow.com/questions/42076706/sys-argv-behavior-with-python-m

    Returns:
        bool: True if we should minimize startup time
    """
    if os.environ.get('_ARGCOMPLETE', ''):
        flag = True
    elif sys.argv and 'geowatch_dvc' in sys.argv[0] or 'geowatch' in sys.argv[0]:
        flag = True
    # elif sys.argv and len(sys.argv) == 1 and sys.argv[0] != '-c':
    #     # No args given case
    #     flag = True
    # elif sys.argv and sys.argv == ['-m']:
    #     flag = True
    elif sys.argv and ('--help' in sys.argv or '-h' in sys.argv):
        flag = True
    elif 'finish_install' in sys.argv:
        flag = True
    else:
        flag = False

    if _GEOWATCH_DEBUG:
        print(f'sys.argv={sys.argv}')
        print(f'Is running a fast CLI tool?: {flag}')

    return flag


def _execute_ordered_preimports():
    """
    The order in which certain modules with binary libraries are imported can
    impact runtime stability and can even cause crashes if a specific order
    isnt used.

    There are several known good configurations registered in the
    ``GEOWATCH_PREIMPORT_VARIANTS`` dictionary and the setting is
    controlled by the ``GEOWATCH_PREIMPORT`` environment variable.
    """
    if _GEOWATCH_DEBUG:
        print('Execute preordered imports')
    # Shorter alias because we are using it now
    GEOWATCH_PREIMPORT = os.environ.get('GEOWATCH_PREIMPORT', os.environ.get('WATCH_PREIMPORT', 'auto'))
    GEOWATCH_PREIMPORT = os.environ.get('GEOWATCH_HACK_IMPORT_ORDER', os.environ.get('WATCH_HACK_IMPORT_ORDER', GEOWATCH_PREIMPORT))

    if _GEOWATCH_DEBUG:
        print(f'GEOWATCH_PREIMPORT = {ub.urepr(GEOWATCH_PREIMPORT, nl=1)}')

    if not GEOWATCH_PREIMPORT:
        if _GEOWATCH_DEBUG:
            print('Fast return')
        return

    if GEOWATCH_PREIMPORT == 'auto':
        if RUNNING_FAST_CLI_TOOL:
            watch_preimport = None
        else:
            # This is the "known" best order for importing
            watch_preimport = GEOWATCH_PREIMPORT_VARIANTS['variant1']
    elif GEOWATCH_PREIMPORT in GEOWATCH_PREIMPORT_VARIANTS:
        watch_preimport = GEOWATCH_PREIMPORT_VARIANTS[GEOWATCH_PREIMPORT]
    elif GEOWATCH_PREIMPORT.lower() in {'0', 'false', 'no', ''}:
        watch_preimport = None
    else:
        watch_preimport = GEOWATCH_PREIMPORT.split(',')

    if watch_preimport is not None:
        for modname in watch_preimport:
            _import_troublesome_module(modname)


RUNNING_FAST_CLI_TOOL = _is_running_a_fast_cli_tool()

_handle_hidden_commands()
_execute_ordered_preimports()


# This can be removed if we bump the min scriptconfig version to 0.7.14
# from geowatch.monkey import monkey_scriptconfig   # NOQA
# monkey_scriptconfig.patch_0_7_14()


if 1:
    try:
        from geowatch.monkey import monkey_numpy  # NOQA
        # monkey_numpy.patch_numpy_dtypes()
        monkey_numpy.patch_numpy_2x()
    except Exception as ex:
        warnings.warn(f'issue with applying monkey patch: ex={ex}')


if 'hard-to-inspect-key' in vars():
    # Defined to hack jon's editor into autocompleting this.
    find_dvc_dpath = None


__devnotes__ = """

# Command to autogenerate lazy imports for this file
mkinit geowatch --lazy_loader --diff
mkinit geowatch --lazy_loader -w

# Debug import time
python -X importtime -c "import geowatch"
GEOWATCH_HACK_IMPORT_ORDER=variant3 python -X importtime -c "import geowatch"
GEOWATCH_HACK_IMPORT_ORDER=variant1 python -X importtime -c "import geowatch"
"""


# Choose which submodules (and which submodule attributes) to expose
__submodules__ = {
    '*': [],  # include all modules, but don't expose attributes
    'demo': ['coerce_kwcoco'],
    'utils': ['find_smart_dvc_dpath', 'find_dvc_dpath']
}


# --- AUTOGENERATED LAZY IMPORTS ---
# import lazy_loader  # NOQA


def lazy_import(module_name, submodules, submod_attrs, eager='auto'):
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

    eager_import_flag = False
    if eager == 'auto':
        eager_import_text = os.environ.get('EAGER_IMPORT', '')
        if eager_import_text:
            eager_import_text_ = eager_import_text.lower()
            if eager_import_text_ in {'true', '1', 'on', 'yes'}:
                eager_import_flag = True

        eager_import_module_text = os.environ.get('EAGER_IMPORT_MODULES', '')
        if eager_import_module_text:
            if __name__.lower() in eager_import_module_text.lower():
                eager_import_flag = True
    else:
        eager_import_flag = eager

    if eager_import_flag:
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
        'geoannots',
        'heuristics',
        'mlops',
        'monkey',
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

__all__ = ['GEOWATCH_PREIMPORT_VARIANTS', 'cli', 'coerce_kwcoco', 'demo',
           'exceptions', 'find_dvc_dpath', 'find_smart_dvc_dpath', 'gis',
           'heuristics', 'mlops', 'monkey', 'rc', 'stac', 'tasks', 'utils']
