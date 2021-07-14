"""
The SMART WATCH module
"""


__devnotes__ = """

# Command to autogenerate lazy imports for this file
mkinit -m watch --lazy --noattr
mkinit -m watch --lazy --noattr -w
"""

__version__ = '0.1.0'


def _hello_world():
    """
    Demonstrate how to write a doctest for any function in this library.

    Example:
        >>> ans = _hello_world()
        >>> assert ans == 42
    """
    print('hello world')
    return 42


def lazy_import(module_name, submodules, submod_attrs):
    import importlib
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
    return __getattr__


__getattr__ = lazy_import(
    __name__,
    submodules={
        'datacube',
        'demo',
        'gis',
        'io',
        'sequencing',
        'tasks',
        'tools',
        'utils',
        'validation',
    },
    submod_attrs={},
)


def __dir__():
    return __all__

__all__ = ['datacube', 'demo', 'gis', 'io', 'sequencing', 'tasks', 'tools',
           'utils', 'validation']
