"""
This submodule contains tools relating to:

    * the IARPA region and site model geojson formats.
    * geographic kwcoco extensions

At the time of creation we have many various tools scattered across the repo to
deal with these, but we should work to consolidate the ones pertaining strictly
to these specific types of geojsons into this submodule.


Location of existing tools that should likely be moved:

    * ../demo/metrics_demo/__init__.py :: various
    * ../cli/run_tracker.py :: various
    * ../cli/reproject_annotations.py :: various
    * ../cli/validate_annotation_schemas.py :: various
    * ../cli/extend_sc_sites.py :: various
    * ../cli/crop_sites_to_regions.py :: various
    * ../cli/cluster_sites.py :: various
    * ../utils/util_framework.py :: determine_region_id

    * ~/code/watch/dev/poc/make_region_from_sitemodel.py


Related tool that should NOT be moved are related to general geojson:

    * ../utils/util_gis.py :: coerce_geojson_datas
"""


__mkinit__ = """
mkinit ~/code/geowatch/geowatch/geoannots/__init__.py --lazy --diff

TEST:
    python -c "from geowatch import geoannots"
    EAGER_IMPORT_MODULES=geowatch python -c "from geowatch import geoannots"
"""
# import lazy_loader

__submodules__ = {
    'geococo_objects': [],
    'geomodels': [],
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
        'geococo_objects',
        'geomodels',
    },
    submod_attrs={},
)


def __dir__():
    return __all__

__all__ = ['geococo_objects', 'geomodels']
