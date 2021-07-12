"""
mkinit -m watch.tasks.fusion --lazy --noattrs -w
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
        'datasets',
        'eval',
        'fit',
        'fit_bigvoter',
        'fit_voter',
        'fusion_transformer_v1',
        'methods',
        'models',
        'onera_channelwisetransformer_train',
        'onera_experiment_predict',
        'onera_transformer_train',
        'onera_unet_train',
        'predict',
        'predict_baselines',
        'predict_bigvoter',
        'predict_ctf',
        'predict_voter',
        'utils',
    },
    submod_attrs={},
)


def __dir__():
    return __all__

__all__ = ['datasets', 'eval', 'fit', 'fit_bigvoter', 'fit_voter',
           'fusion_transformer_v1', 'methods', 'models',
           'onera_channelwisetransformer_train', 'onera_experiment_predict',
           'onera_transformer_train', 'onera_unet_train', 'predict',
           'predict_baselines', 'predict_bigvoter', 'predict_ctf',
           'predict_voter', 'utils']
