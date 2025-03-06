__mkinit__ = """
mkinit ~/code/watch/geowatch/utils/__init__.py --lazy --diff
mkinit ~/code/watch/geowatch/utils/__init__.py --lazy -w

TEST:
    python -c "from geowatch import utils"
    EAGER_IMPORT_MODULES=geowatch python -c "from geowatch import utils"
"""

__submodules__ = {
    '*': [],
    'util_data': ['find_smart_dvc_dpath', 'find_dvc_dpath'],
    # 'util_yaml': ['Yaml'],
}


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
            if eager_import_module_text.lower() in __name__.lower():
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
        'configargparse_ext',
        'ext_monai',
        'ijson_ext',
        'kwcoco_extensions',
        'lightning_ext',
        'process_context',
        'remedian',
        'result_analysis',
        'reverse_hashid',
        'simple_dvc',
        'util_bands',
        'util_codes',
        'util_data',
        'util_dotdict',
        'util_exception',
        'util_framework',
        'util_fsspec',
        'util_gdal',
        'util_girder',
        'util_gis',
        'util_globals',
        'util_hardware',
        'util_iter',
        'util_kwarray',
        'util_kwimage',
        'util_kwplot',
        'util_logging',
        'util_nesting',
        'util_netharn',
        'util_nvidia',
        'util_pandas',
        'util_param_grid',
        'util_raster',
        'util_regex',
        'util_resolution',
        'util_retry',
        'util_s3',
        'util_stringalgo',
        'util_torchmetrics',
        'util_units',
    },
    submod_attrs={
        'util_data': [
            'find_smart_dvc_dpath',
            'find_dvc_dpath',
        ],
    },
)


def __dir__():
    return __all__

__all__ = ['configargparse_ext', 'ext_monai', 'find_dvc_dpath',
           'find_smart_dvc_dpath', 'ijson_ext', 'kwcoco_extensions',
           'lightning_ext', 'process_context', 'remedian', 'result_analysis',
           'reverse_hashid', 'simple_dvc', 'util_bands',
           'util_codes', 'util_data', 'util_dotdict', 'util_exception',
           'util_framework', 'util_fsspec', 'util_gdal', 'util_girder',
           'util_gis', 'util_globals', 'util_hardware', 'util_iter',
           'util_kwarray', 'util_kwimage', 'util_kwplot', 'util_logging',
           'util_nesting', 'util_netharn', 'util_nvidia', 'util_pandas',
           'util_param_grid', 'util_raster', 'util_regex', 'util_resolution',
           'util_retry', 'util_s3', 'util_stringalgo',
           'util_torchmetrics', 'util_units']
