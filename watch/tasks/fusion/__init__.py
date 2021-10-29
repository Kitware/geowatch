"""
mkinit -m watch.tasks.fusion --lazy --noattrs
-w
mkinit -m watch.tasks.fusion --noattrs -w
"""

# Hack to supress pytorch-lightning warning
import setuptools  # NOQA

from watch.tasks.fusion import utils
from watch.tasks.fusion import datamodules
from watch.tasks.fusion import architectures
from watch.tasks.fusion import methods


# Not including these removes the runtime warning:
# 'watch.tasks.fusion.fit' found in sys.modules after import of package
# 'watch.tasks.fusion', but prior to execution of 'watch.tasks.fusion.fit';
# this may result in unpredictable behaviour

# from watch.tasks.fusion import fit
# from watch.tasks.fusion import predict
# from watch.tasks.fusion import evaluate

# We can include them via lazy imports. The other packages are not given lazy
# imports so torch.package can introspect them.


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
        'architectures',
        'datamodules',
        'evaluate',
        'fit',
        'methods',
        'organize',
        'postprocess',
        'predict',
        'repackage',
        'utils',
    },
    submod_attrs={},
)


def __dir__():
    return __all__


__all__ = ['architectures', 'datamodules', 'evaluate', 'fit', 'methods',
           'organize', 'postprocess', 'predict', 'repackage', 'utils']
