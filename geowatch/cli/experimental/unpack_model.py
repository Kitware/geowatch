#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class UnpackModelCLI(scfg.DataConfig):
    """
    Unpack the core components of a torch package to make them suitable for
    environment-agnostic use or repackaging.
    """
    fpath = scfg.Value(None, help='the path to a torch package')

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> from geowatch.cli.experimental.unpack_model import *  # NOQA
            >>> cmdline = 0
            >>> kwargs = dict()
            >>> kwargs['fpath'] = '/data/joncrall/dvc-repos/smart_phase3_expt/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_V001/Drop8-ARA-Cropped2GSD-V1_allsensors_V001_epoch0_step21021.pt'
            >>> cls = UnpackModelCLI
            >>> cls.main(cmdline=cmdline, **kwargs)


        Ignore:
            ...
        """
        import rich
        from rich.markup import escape
        config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
        rich.print('config = ' + escape(ub.urepr(config, nl=1)))
        package_fpath = config.fpath
        unpack_model(package_fpath)


def unpack_model(package_fpath):
    """
    Ignore:
        package_fpath = '/data/joncrall/dvc-repos/smart_phase3_expt/models/fusion/Drop8-ARA-Cropped2GSD-V1/packages/Drop8-ARA-Cropped2GSD-V1_allsensors_V001/Drop8-ARA-Cropped2GSD-V1_allsensors_V001_epoch0_step21021.pt'
        result = unpack_model(package_fpath)
        checkpoint_fpath = result['ckpt_fpath']
        from geowatch.mlops.repackager import repackage
        new_package_fpath = repackage(checkpoint_fpath=checkpoint_fpath)[0]
        round2_result = unpack_model(new_package_fpath)

        import kwutil
        config1 = kwutil.Yaml.load(result['config_fpath'])
        config2 = kwutil.Yaml.load(round2_result['config_fpath'])
        config1 == config2

        from kwcoco.util.util_json import indexable_diff
        info = indexable_diff(config1, config2)
        assert info['similarity'] == 1.0
    """
    from geowatch.tasks.fusion import utils
    # from geowatch.utils import util_netharn
    from geowatch.monkey import monkey_torchmetrics
    from kwutil.util_yaml import Yaml
    import torch
    monkey_torchmetrics.fix_torchmetrics_compatability()

    package_fpath = ub.Path(package_fpath)

    if not package_fpath.exists():
        if package_fpath.augment(tail='.dvc').exists():
            raise Exception('model does not exist, but its dvc file does')
        else:
            raise Exception('model does not exist')

    # file_stat = package_fpath.stat()
    # TODO: generalize the load-package
    raw_module = utils.load_model_from_package(package_fpath)
    if hasattr(raw_module, 'module'):
        module = raw_module.module
    else:
        module = raw_module

    # TODO: get the category freq

    module.__class__
    module.hparams

    # Construct a checkpoint the repackager will accept.
    checkpoint = {}
    checkpoint['state_dict'] = raw_module.state_dict()
    checkpoint['hyper_parameters'] = raw_module.hparams

    package_header = utils.load_model_header(package_fpath)
    config_content = package_header['config']

    dst_dpath = package_fpath.parent / package_fpath.stem
    dst_dpath.ensuredir()

    config_fpath = dst_dpath / 'config.yaml'
    config_fpath.write_text(Yaml.dumps(config_content))

    ckpt_name = package_fpath.stem + '.ckpt'
    ckpt_dpath = (dst_dpath / 'checkpoints').ensuredir()
    checkpoint_fpath = ckpt_dpath / ckpt_name

    with open(checkpoint_fpath, 'wb') as file:
        torch.save(checkpoint, file)

    result = {}
    result['dpath'] = dst_dpath
    result['ckpt_fpath'] = checkpoint_fpath
    result['config_fpath'] = config_fpath
    return result

__cli__ = UnpackModelCLI

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/geowatch/geowatch/cli/experimental/unpack_model.py
        python -m geowatch.cli.experimental.unpack_model
    """
    __cli__.main()
