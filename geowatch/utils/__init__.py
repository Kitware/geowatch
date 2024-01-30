__autogen__ = """
mkinit ~/code/watch/geowatch/utils/__init__.py --lazy_loader --diff
mkinit ~/code/watch/geowatch/utils/__init__.py --lazy_loader -w

TEST:
    python -c "from geowatch import utils"
    EAGER_IMPORT=1 python -c "from geowatch import utils"
"""

__submodules__ = {
    '*': [],
    'util_data': ['find_smart_dvc_dpath', 'find_dvc_dpath'],
    # 'util_yaml': ['Yaml'],
}


import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'configargparse_ext',
        'ext_monai',
        'ijson_ext',
        'kwcoco_extensions',
        'lightning_ext',
        'process_context',
        'result_analysis',
        'reverse_hashid',
        'simple_dvc',
        'util_bands',
        'util_chmod',
        'util_codes',
        'util_data',
        'util_dotdict',
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
        'util_rgdc',
        'util_s3',
        'util_stringalgo',
        'util_torchmetrics',
    },
    submod_attrs={
        'util_data': [
            'find_smart_dvc_dpath',
            'find_dvc_dpath',
        ],
    },
)

__all__ = ['configargparse_ext', 'ext_monai', 'find_dvc_dpath',
           'find_smart_dvc_dpath', 'ijson_ext', 'kwcoco_extensions',
           'lightning_ext', 'process_context', 'result_analysis',
           'reverse_hashid', 'simple_dvc', 'util_bands', 'util_chmod',
           'util_codes', 'util_data', 'util_dotdict', 'util_framework',
           'util_fsspec', 'util_gdal', 'util_girder', 'util_gis',
           'util_globals', 'util_hardware', 'util_iter', 'util_kwarray',
           'util_kwimage', 'util_kwplot', 'util_logging', 'util_nesting',
           'util_netharn', 'util_nvidia', 'util_pandas', 'util_param_grid',
           'util_raster', 'util_regex', 'util_resolution', 'util_rgdc',
           'util_s3', 'util_stringalgo', 'util_torchmetrics']
