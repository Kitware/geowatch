"""
Detectron2 is less polished than I thought it would be. It has two different
incompatible config systems. This file helps with the new LazyConfig system.
"""
from geowatch.tasks.detectron2._common import Detectron2WrapperBase
import ubelt as ub
import os


class Detectron2WrapperNewStyle(Detectron2WrapperBase):
    """
    Wrapper for the new-style detectron configs.

    CommandLine:
        xdoctest -m geowatch.tasks.detectron2._new_config_backend Detectron2WrapperNewStyle

    Example:
        >>> # xdoctest: +REQUIRES(module:fvcore)
        >>> from geowatch.tasks.detectron2.fit import DetectronFitCLI
        >>> from geowatch.tasks.detectron2._new_config_backend import *  # NOQA
        >>> import geowatch_tpl
        >>> geowatch_tpl.import_submodule('detectron2')  # NOQA
        >>> import kwcoco
        >>> import os
        >>> dset = kwcoco.CocoDataset.demo('vidshapes8')
        >>> dpath = ub.Path.appdir('geowatch/tests/detectron/fit_regnet')
        >>> config = DetectronFitCLI()
        >>> config['train_fpath'] = os.fspath(dset.fpath)
        >>> config['default_root_dir'] = os.fspath(dpath)
        >>> config['base'] = 'new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py'
        >>> config['init'] = 'new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py'
        >>> config['cfg'] = ub.codeblock(
                '''
                dataloader:
                    train:
                        total_batch_size: 4
                        num_workers: 1
                optimizer:
                    lr: 0.001
                train:
                    amp:
                        enabled: true
                    max_iter: 10
                    eval_period: 1
                    log_period: 1
                    checkpointer:
                        period: 1
                        max_to_keep: 100
                    device: cuda
                ''')
        >>> self = Detectron2WrapperNewStyle(config)
        >>> self.register_datasets()
        >>> self.resolve_config()
        >>> self.build_trainer()
        >>> self.train()
    """

    def __init__(self, config):
        super().__init__(config)

    def resolve_config(self):
        """
        I really dont like these "new" lazy Python configs.
        There isnt a clear way to create an instance of the model.

        References:
            https://github.com/facebookresearch/detectron2/blob/main/docs/tutorials/lazyconfigs.md

        Ignore:
            from detectron2 import model_zoo
            cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py')
            instantiate(cfg.model)
        """

        import omegaconf
        import rich
        from rich.markup import escape
        from detectron2 import model_zoo
        # from omegaconf import OmegaConf
        import kwutil
        config = self.config
        dataset_infos = self.dataset_infos

        @ub.urepr.extensions.register(omegaconf.dictconfig.DictConfig)
        def _format_omegaconfg(d, **kw):
            # Hack for printing omegaconfig with some reasonablness
            return ub.util_repr._format_dict(d, **kw)[0]

        # # Hack for hashing
        # found = None
        # for k, v in ub.hash_data.extensions._hash_dispatch.registry.items():
        #     if k.__name__ == 'dict':
        #         found = v
        # assert found is not None
        # ub.hash_data.extensions._hash_dispatch.register(omegaconf.dictconfig.DictConfig)(found)

        cfg = model_zoo.get_config(config.base)
        cfg.dataloader.train.dataset.names = dataset_infos['train']['name']

        if dataset_infos['vali'] is None:
            cfg.dataloader.test.dataset.names = None
        else:
            cfg.dataloader.test.dataset.names = dataset_infos['vali']['name']

        base_cfg_text = omegaconf_dumps(cfg)
        rich.print(f'base cfg = {escape(base_cfg_text)}')

        cfg.model.roi_heads.num_classes = len(dataset_infos['train']['categories'])

        # Override config values
        cfg_final_layer = kwutil.Yaml.coerce(config.cfg, backend='pyyaml')
        walker = ub.IndexableWalker(cfg)
        to_set_walker = ub.IndexableWalker(cfg_final_layer)
        for p, v in to_set_walker:
            if not isinstance(v, dict):
                walker[p] = v

        if self.config.init == 'noop':
            cfg.train.init_checkpoint = ''
        else:
            cfg.train.init_checkpoint = model_zoo.get_checkpoint_url(self.config.init)  # Let training initialize from model zoo

        cfg.train.output_dir = None  # hack: null out for the initial hashing
        text = omegaconf_dumps(cfg)
        hashid = ub.hash_data(text)[0:8]

        output_dpath = ub.Path(config.default_root_dir) / f'v_{hashid}'
        output_dpath.ensuredir()
        self.output_dpath = output_dpath
        cfg.train.output_dir = os.fspath(output_dpath)
        overlaid_cfg_text = omegaconf_dumps(cfg)
        rich.print(f'overlaid cfg =  {escape(overlaid_cfg_text)}')
        self.cfg = cfg

    def dump_model_config(self):
        from detectron2.config import LazyConfig
        output_dpath = self.output_dpath.ensuredir()
        LazyConfig.save(self.cfg, os.fspath(output_dpath / 'detectron_config.yaml'))

    def build_predictor(self):
        ...

    def build_trainer(self):
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.config import instantiate
        from detectron2.engine import (
            AMPTrainer,
            SimpleTrainer,
            default_writers,
            hooks,
        )
        from detectron2.engine.defaults import create_ddp_model
        from detectron2.utils import comm
        """
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
        """
        cfg = self.cfg.copy()
        model = instantiate(cfg.model)
        model.to(cfg.train.device)

        cfg.optimizer.params.model = model
        optim = instantiate(cfg.optimizer)

        train_loader = instantiate(cfg.dataloader.train)

        model = create_ddp_model(model, **cfg.train.ddp)
        trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, train_loader, optim)
        checkpointer = DetectionCheckpointer(
            model,
            cfg.train.output_dir,
            trainer=trainer,
        )
        trainer.register_hooks([
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            (
                hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
                if comm.is_main_process()
                else None
            ),
        ])
        trainer.register_hooks([
            (None if self.dataset_infos['vali'] is None else hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model))),
        ])
        trainer.register_hooks([
            (
                hooks.PeriodicWriter(
                    default_writers(cfg.train.output_dir, cfg.train.max_iter),
                    period=cfg.train.log_period,
                )
                if comm.is_main_process()
                else None
            ),
        ]
        )

        checkpointer.resume_or_load(cfg.train.init_checkpoint)
        if checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            self.start_iter = trainer.iter + 1
        else:
            self.start_iter = 0

        self.trainer = trainer

    def train(self):
        self.dump_model_config()
        if self.trainer is None:
            self.build_trainer()
        import rich
        rich.print(f'Output Dpath: [link={self.output_dpath}]{self.output_dpath}[/link]')
        self.trainer.train(self.start_iter, self.cfg.train.max_iter)
        rich.print(f'Output Dpath: [link={self.output_dpath}]{self.output_dpath}[/link]')


def do_test(cfg, model):
    from detectron2.config import instantiate
    from detectron2.evaluation import inference_on_dataset, print_csv_format
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model,
            instantiate(cfg.dataloader.test),
            instantiate(cfg.dataloader.evaluator),
        )
        print_csv_format(ret)
        return ret


def omegaconf_dumps(cfg):
    """
    Get some semi-serializable representation of the new-style configuration
    """
    from omegaconf import OmegaConf, SCMode
    dict = OmegaConf.to_container(
        cfg,
        # Do not resolve interpolation when saving, i.e. do not turn ${a} into
        # actual values when saving.
        resolve=False,
        # Save structures (dataclasses) in a format that can be instantiated later.
        # Without this option, the type information of the dataclass will be erased.
        structured_config_mode=SCMode.INSTANTIATE,
    )
    import yaml
    dumped = yaml.dump(dict, default_flow_style=None, allow_unicode=True, width=9999)
    import io
    f = io.StringIO()
    f.write(dumped)
    text = f.getvalue()
    return text
