"""
Module for access to simple data for demo and testing purposes.
"""

__devnotes__ = """
mkinit -m watch.demo --lazy -w
"""


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
        'dummy_demodata',
        'landsat_demodata',
        'sentinel2_demodata',
        'nitf_demodata',
    },
    submod_attrs={
        'dummy_demodata': [
            'dummy_rpc_geotiff_fpath',
        ],
        'landsat_demodata': [
            'grab_landsat_product',
        ],
        'sentinel2_demodata': [
            'grab_sentinel2_product',
        ],
        'nitf_demodata': [
            'DEFAULT_KEY',
            'grab_nitf_fpath',
        ],
    },
)


def __dir__():
    return __all__

__all__ = ['DEFAULT_KEY', 'dummy_demodata', 'dummy_rpc_geotiff_fpath',
           'grab_landsat_product', 'grab_nitf_fpath', 'landsat_demodata',
           'nitf_demodata']
