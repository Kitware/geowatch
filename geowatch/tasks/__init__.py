__mkinit__ = """
mkinit ~/code/watch/geowatch/tasks/__init__.py --lazy --diff
mkinit ~/code/watch/geowatch/tasks/__init__.py --lazy --noattrs -w

TEST:
    python -c "from geowatch import tasks"
    EAGER_IMPORT_MODULES=geowatch python -c "from geowatch import tasks"
"""

###


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
        'cold',
        'depth',
        'depth_pcd',
        'dino_detector',
        'fusion',
        'invariants',
        'landcover',
        'mae',
        'metrics',
        'poly_from_point',
        'rutgers_material_change_detection',
        'rutgers_material_seg',
        'sam',
        'tracking',
        'uky_temporal_prediction',
    },
    submod_attrs={},
)


def __dir__():
    return __all__

__all__ = ['cold', 'depth', 'depth_pcd', 'dino_detector', 'fusion',
           'invariants', 'landcover', 'mae', 'metrics', 'poly_from_point',
           'rutgers_material_change_detection', 'rutgers_material_seg', 'sam',
           'tracking', 'uky_temporal_prediction']
