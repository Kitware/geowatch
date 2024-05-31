"""
mkinit geowatch.tasks.fusion --lazy --noattrs -w
mkinit -m geowatch.tasks.fusion --noattrs -w
"""

# Hack to supress pytorch-lightning warning
import setuptools  # NOQA

from geowatch.tasks.fusion import utils
from geowatch.tasks.fusion import datamodules
from geowatch.tasks.fusion import architectures
from geowatch.tasks.fusion import methods


# Not including these removes the runtime warning:
# 'geowatch.tasks.fusion.fit' found in sys.modules after import of package
# 'geowatch.tasks.fusion', but prior to execution of 'geowatch.tasks.fusion.fit';
# this may result in unpredictable behaviour

# from geowatch.tasks.fusion import fit
# from geowatch.tasks.fusion import predict
# from geowatch.tasks.fusion import evaluate

# We can include them via lazy imports. The other packages are not given lazy
# imports so torch.package can introspect them.


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
        'architectures',
        'coco_stitcher',
        'datamodules',
        'evaluate',
        'fit',
        'fit_lightning',
        'methods',
        'predict',
        'production',
        'utils',
    },
    submod_attrs={},
)


def __dir__():
    return __all__

__all__ = ['architectures', 'coco_stitcher', 'datamodules', 'evaluate', 'fit',
           'fit_lightning', 'methods', 'predict', 'production', 'utils']
