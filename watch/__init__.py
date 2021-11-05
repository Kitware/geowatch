"""
The SMART WATCH module
"""
import os


__devnotes__ = """

# Command to autogenerate lazy imports for this file
mkinit -m watch --lazy --noattr
mkinit -m watch --lazy --noattr -w
"""


DISABLE_IMPORT_ORDER_HACK = os.environ.get('DISABLE_IMPORT_ORDER_HACK', '0')

if DISABLE_IMPORT_ORDER_HACK != '1':
    # Some imports need to happen in a specific order, otherwise we get crashes
    # This is very annoying
    from pyproj import CRS  # NOQA
    from osgeo import gdal  # NOQA

__version__ = '0.1.5'


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
        'rc',
        'sequencing',
        'tasks',
        'utils',
        'validation',
    },
    submod_attrs={},
)


def __dir__():
    return __all__


__all__ = ['cli', 'datacube', 'datasets', 'demo', 'gis', 'rc', 'sequencing',
           'tasks', 'utils', 'validation']
