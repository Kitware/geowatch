"""
mkinit ~/code/watch/watch/utils/__init__.py --lazy -w
"""

__submodules__ = {
    '*': [],
    'util_data': ['find_smart_dvc_dpath'],
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
        'configargparse_ext',
        'ext_monai',
        'kwcoco_extensions',
        'lightning_ext',
        'slugify_ext',
        'util_bands',
        'util_data',
        'util_gdal',
        'util_girder',
        'util_gis',
        'util_iter',
        'util_kwarray',
        'util_kwimage',
        'util_kwplot',
        'util_norm',
        'util_parallel',
        'util_path',
        'util_raster',
        'util_regex',
        'util_rgdc',
        'util_stac',
        'util_time',
        'util_framework',
    },
    submod_attrs={
        'util_data': [
            'find_smart_dvc_dpath',
        ],
    },
)


def __dir__():
    return __all__

__all__ = ['configargparse_ext', 'ext_monai', 'find_smart_dvc_dpath',
           'kwcoco_extensions', 'lightning_ext', 'slugify_ext', 'util_bands',
           'util_data', 'util_gdal', 'util_girder', 'util_gis', 'util_iter',
           'util_kwarray', 'util_kwimage', 'util_kwplot', 'util_norm',
           'util_parallel', 'util_path', 'util_raster', 'util_regex',
           'util_rgdc', 'util_stac', 'util_time', 'util_framework']
