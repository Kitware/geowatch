"""
Can we make scriptconfig handle what we really need?
"""
#!/usr/bin/env python3
import scriptconfig as scfg
from geowatch.tasks.fusion._lightning_components import SmartTrainer
from geowatch.tasks.fusion.methods import MultimodalTransformer  # NOQA
from geowatch.tasks.fusion.datamodules.kwcoco_datamodule import KWCocoVideoDataModule
# import ubelt as ub


class PocAltFitCLI(scfg.DataConfig):
    """
    import sys, ubelt
    sys.path.append(ubelt.expandpath('~/code/geowatch'))
    from geowatch.tasks.fusion._poc_alt_fit import *  # NOQA
    """
    trainer = scfg.Value(None, help='A YAML dictionary of updates to the defaults.')
    data = scfg.Value(None, help='A YAML dictionary of updates to the defaults.')
    model = scfg.Value(None, help='A YAML dictionary of updates to the defaults.')
    optimizer = scfg.Value(None, help='A YAML dictionary of updates to the defaults.')
    lr_scheduler = scfg.Value(None, help='A YAML dictionary of updates to the defaults.')
    torch_globals = scfg.Value(None, help='A YAML dictionary of updates to the defaults.')
    initializer = scfg.Value(None, help='A YAML dictionary of updates to the defaults.')


def main():
    """

    config = ub.codeblock(
        '''
        data:
            class_path: KWCocoVideoDataModule
            init_args:
                train_dataset: special:vidshapes8-frames9-speed0.5
                window_dims: 64
                num_workers: 4
                batch_size: 4
                normalize_inputs:
                    input_stats:
                        - sensor: '*'
                          channels: r|g|b
                          video: video1
                          mean: [87.572401, 87.572402, 87.572403]
                          std: [99.449997, 99.449998, 99.449999]
        model:
            class_path: MultimodalTransformer
        optimizer:
            class_path: torch.optim.Adam
        trainer:
            accelerator: gpu
            devices: 1
            default_root_dir: ./demo_train
        ''')
    """
    import ubelt as ub

    defaults = dict(
        trainer={'class_path': SmartTrainer, 'init_args': {}},
        model={'class_path': MultimodalTransformer, 'init_args': {}},
        data={'class_path': KWCocoVideoDataModule, 'init_args': {}},
        optimizer={'class_path': 'torch.optim.AdamW', 'init_args': {}},
        lr_scheduler={'class_path': 'torch.optim.lr_scheduler.ConstantLR', 'init_args': {}},
    )

    __path_defaults__ = {}

    for key, data in defaults.items():
        new_data = coerce_instance_defaults(key, data)

        for param in new_data:
            path = f'{key}.{param.name}'
            __path_defaults__[path] = scfg.Value(param.default, help=None if param.doc is None else ub.paragraph(param.doc))

    print(f'__path_defaults__ = {ub.urepr(__path_defaults__, nl=1)}')


def coerce_instance_defaults(key, data):
    import torch  # NOQA
    import kwutil
    from geowatch.tasks.fusion import _jsonargparse_introspect
    data = kwutil.Yaml.coerce(data)
    if isinstance(data, dict):
        if 'class_path' in data:
            cls = data['class_path']
            if isinstance(cls, str):
                cls = eval(cls, globals(), locals())

            default_params = _jsonargparse_introspect.get_signature_parameters(cls)
            return default_params

            if 'init_args' in data:
                init_kw = data['init_args']
            else:
                init_kw = {}
        else:
            raise NotImplementedError
        new_data = {'cls': cls, 'init_kw': init_kw}
        return
    else:
        raise NotImplementedError

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/geowatch/geowatch/tasks/fusion/_poc_alt_fit.py
        python -m geowatch.tasks.fusion._poc_alt_fit
    """
    main()
