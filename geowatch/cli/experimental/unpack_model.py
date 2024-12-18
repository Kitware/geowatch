#!/usr/bin/env python3
"""
TODO:
    - [ ] Add relevant documentation to point at
          ~/code/geowatch/geowatch/mlops/repackager.py so it is easy for the user to
          unpack / repack models.

"""
import scriptconfig as scfg
import ubelt as ub


class UnpackModelCLI(scfg.DataConfig):
    """
    Unpack the core components of a torch package to make them suitable for
    environment-agnostic use or repackaging.

    SeeAlso
    -------
    python -m geowatch/mlops/repackager
    """
    fpath = scfg.Value(None, help='the path to a torch package', position=1)
    dst_dpath = scfg.Value(None, help='Path to a destination directory to write to. If unspecfied chooses one.')

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
        """
        import rich
        from rich.markup import escape
        config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
        rich.print('config = ' + escape(ub.urepr(config, nl=1)))
        package_fpath = config.fpath
        unpack_model(package_fpath, dst_dpath=config.dst_dpath)


def unpack_model(package_fpath, dst_dpath=None):
    """
    Extracts and writes extracted files to disk.

    Returns:
        Dict[str, ub.Path]: mapping from keys to written paths on disk

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

    Example:
        >>> import ubelt as ub
        >>> from geowatch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
        >>> dpath = ub.Path.appdir('geowatch/tests/package').ensuredir()
        >>> package_fpath = dpath / 'my_package.pt'
        >>> dst_dpath = dpath / 'unpacked_packages'
        >>> if not package_fpath.exists():
        >>>     # Use one of our fusion.architectures in a test
        >>>     from geowatch.tasks.fusion import methods
        >>>     from geowatch.tasks.fusion import datamodules
        >>>     model = self = methods.MultimodalTransformer(
        >>>         arch_name="smt_it_joint_p2", input_sensorchan=5,
        >>>         change_head_hidden=0, saliency_head_hidden=0,
        >>>         class_head_hidden=0)
        >>>     # Save the model (TODO: need to save datamodule as well)
        >>>     model.save_package(package_fpath)
        >>> results = unpack_model(package_fpath)
        >>> # Test repackage
        >>> from geowatch.mlops.repackager import repackage_single_checkpoint
        >>> repackage_single_checkpoint
    """
    from kwutil.util_yaml import Yaml
    import torch

    package_fpath = ub.Path(package_fpath)
    package_content = extract_package_contents(package_fpath)

    if dst_dpath is None:
        dst_dpath = package_fpath.parent / package_fpath.stem
    else:
        dst_dpath = ub.Path(dst_dpath)
    dst_dpath.ensuredir()

    config_fpath = dst_dpath / 'config.yaml'
    config_fpath.write_text(Yaml.dumps(package_content['config']))

    ckpt_name = package_fpath.stem + '.ckpt'
    ckpt_dpath = (dst_dpath / 'checkpoints').ensuredir()
    checkpoint_fpath = ckpt_dpath / ckpt_name

    with open(checkpoint_fpath, 'wb') as file:
        torch.save(package_content['checkpoint'], file)

    result = {}
    result['dpath'] = dst_dpath
    result['ckpt_fpath'] = checkpoint_fpath
    result['config_fpath'] = config_fpath
    return result


def extract_package_contents(package_fpath):
    """
    Returns:
        dict
    """
    from geowatch.tasks.fusion import utils
    # from geowatch.utils import util_netharn
    from geowatch.monkey import monkey_torchmetrics
    monkey_torchmetrics.fix_torchmetrics_compatability()

    package_fpath = ub.Path(package_fpath)

    if not package_fpath.exists():
        if package_fpath.augment(tail='.dvc').exists():
            raise Exception('model does not exist, but its dvc file does')
        else:
            raise Exception('model does not exist')

    # file_stat = package_fpath.stat()
    # TODO: generalize the load-package

    # TODO: is it possible to extract only the weights or only the torch /
    # numpy objects from a pickle file and ignore any part that references an
    # unavailable package?
    raw_module = utils.load_model_from_package(package_fpath)
    if hasattr(raw_module, 'module'):
        module = raw_module.module
    else:
        module = raw_module

    # TODO: get the category freq

    module.__class__
    module.hparams

    HACK_REMOVE_PROBLEMATIC_INFO = 1
    if HACK_REMOVE_PROBLEMATIC_INFO:
        try:
            raw_module.hparams['dataset_stats'].pop('modality_input_stats', None)
        except AttributeError:
            ...

    # Construct a checkpoint the repackager will accept.
    checkpoint = {}
    checkpoint['state_dict'] = raw_module.state_dict()
    try:
        # TODO: remove numpy arrays if possible
        # They can cause read errors if there are binary incompatabilities
        # todo: make configurable?
        import kwutil
        checkpoint['hyper_parameters'] = kwutil.Json.ensure_serializable(raw_module.hparams)
    except Exception:
        checkpoint['hyper_parameters'] = raw_module.hparams

    package_header = utils.load_model_header(package_fpath)
    config_content = package_header.get('config', None)

    if HACK_REMOVE_PROBLEMATIC_INFO:
        if config_content and 'model' in config_content:
            if 'class_path' in config_content['model']:
                if config_content['model']['class_path'].startswith('watch.'):
                    config_content['model']['class_path'] = 'geo' + config_content['model']['class_path']

    package_content = {}
    package_content['config'] = config_content
    package_content['checkpoint'] = checkpoint
    return package_content


# def unpack_model_backup(package_fpath):
#     import zipfile
#     zfile = zipfile.ZipFile(package_fpath, 'r')
#     names = zfile.namelist()

#     cand_model_pkl = [n for n in names if n.endswith('model.pkl')]
#     assert len(cand_model_pkl) == 1
#     model_pkl = cand_model_pkl[0]
#     file = zfile.open(model_pkl)

#     data = file.read()
#     import pickle
#     loaded = pickle.loads(data)


__cli__ = UnpackModelCLI

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/geowatch/geowatch/cli/experimental/unpack_model.py
        python -m geowatch.cli.experimental.unpack_model
    """
    __cli__.main()
