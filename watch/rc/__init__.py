__dev__ = """
mkinit ~/code/watch/watch/rc/__init__.py --lazy -w
"""


def lazy_import(module_name, submodules, submod_attrs):
    import importlib
    import os
    name_to_submod = {
        func: mod
        for mod, funcs in submod_attrs.items() for func in funcs
    }

    def __getattr__(name):
        if name in submodules:
            attr = importlib.import_module('{module_name}.{name}'.format(
                module_name=module_name, name=name))
        elif name in name_to_submod:
            submodname = name_to_submod[name]
            module = importlib.import_module(
                '{module_name}.{submodname}'.format(module_name=module_name,
                                                    submodname=submodname))
            attr = getattr(module, name)
        else:
            raise AttributeError('No {module_name} attribute {name}'.format(
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
        'registry',
    },
    submod_attrs={
        'registry': [
            'load_site_model_schema',
            'load_region_model_schema',
            'load_job_schema',
            'geoidgrid_path',
            'dem_path',
        ],
    },
)


def __dir__():
    return __all__


__all__ = [
    'load_site_model_schema', 'load_region_model_schema', 'load_job_schema',
    'geoidgrid_path', 'dem_path', 'registry'
]
