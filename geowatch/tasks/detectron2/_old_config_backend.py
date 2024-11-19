import os
import ubelt as ub
from geowatch.tasks.detectron2._common import Detectron2WrapperBase


class Detectron2WrapperOldStyle(Detectron2WrapperBase):
    """
    Wrapper for the old-style detectron configs.

    References:
        https://github.com/facebookresearch/detectron2/blob/main/docs/tutorials/configs.md

    CommandLine:
        xdoctest -m geowatch.tasks.detectron2._old_config_backend Detectron2WrapperOldStyle

    Example:
        >>> # xdoctest: +REQUIRES(module:fvcore)
        >>> from geowatch.tasks.detectron2._old_config_backend import *  # NOQA
        >>> from geowatch.tasks.detectron2.fit import DetectronFitCLI
        >>> import kwcoco
        >>> import os
        >>> import geowatch_tpl
        >>> detectron2 = geowatch_tpl.import_submodule('detectron2')  # NOQA
        >>> dset = kwcoco.CocoDataset.demo('vidshapes8')
        >>> dset.conform()
        >>> dset.dump()
        >>> dpath = ub.Path.appdir('geowatch/tests/detectron/fit_frcnn')
        >>> config = DetectronFitCLI()
        >>> config['train_fpath'] = dset.fpath
        >>> config['default_root_dir'] = dpath
        >>> config['base'] = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
        >>> config['init'] = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
        >>> config['cfg'] = ub.codeblock(
                '''
                DATALOADER:
                    NUM_WORKERS: 1
                SOLVER:
                    IMS_PER_BATCH: 1
                    BASE_LR: 0.00025
                    CHECKPOINT_PERIOD: 1
                    WARMUP_ITERS: 3
                    MAX_ITER: 10
                ''')
        >>> self = Detectron2WrapperOldStyle(config)
        >>> self.register_datasets()
        >>> self.resolve_config()
        >>> self.build_trainer()
        >>> self.train()
    """

    def resolve_config(self):
        from detectron2.config import CfgNode
        from detectron2 import model_zoo
        import kwutil

        # cfg = get_cfg()
        # base_cfg = model_zoo.get_config_file(config.base)
        # cfg.merge_from_file(base_cfg)
        cfg = model_zoo.get_config(self.config.base)
        cfg.DATASETS.TRAIN = (self.dataset_infos['train']['name'],)
        if self.dataset_infos['vali'] is None:
            cfg.DATASETS.TEST = ()
        else:
            cfg.DATASETS.TEST = (self.dataset_infos['vali']['name'],)

        if self.config.init == 'noop':
            cfg.MODEL.WEIGHTS = ""
        else:
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.config.init)  # Let training initialize from model zoo

        if 0:
            # Old hacky stuff
            cfg.DATALOADER.NUM_WORKERS = 2
            # cfg.SOLVER.IMS_PER_BATCH = 2   # This is the real 'batch size' commonly known to deep learning people
            cfg.SOLVER.IMS_PER_BATCH = 16   # This is the real 'batch size' commonly known to deep learning people
            cfg.SOLVER.BASE_LR = 0.00025   # pick a good LR
            cfg.SOLVER.MAX_ITER = 520_000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
            cfg.SOLVER.STEPS = []          # do not decay learning rate
            cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The 'RoIHead batch size'. 128 is faster, and good enough for this toy dataset (default: 512)

        # see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets
        # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
        # We can introspect the training file to determine the number of classes.
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.dataset_infos['train']['categories'])

        print(ub.urepr(cfg, nl=-1))

        cfg_final_layer = kwutil.Yaml.coerce(self.config.cfg, backend='pyyaml')
        cfg2 = CfgNode(cfg_final_layer)
        print(ub.urepr(cfg2, nl=-1))
        cfg.merge_from_other_cfg(cfg2)
        print(ub.urepr(cfg, nl=-1))

        cfg.OUTPUT_DIR = None  # hack: null out for the initial hashing
        hashid = ub.hash_data(cfg)[0:8]

        output_dpath = ub.Path(self.config.default_root_dir) / f'v_{hashid}'
        output_dpath.ensuredir()
        cfg.OUTPUT_DIR = os.fspath(output_dpath)
        print(ub.urepr(cfg, nl=-1))
        self.cfg = cfg
        self.output_dpath = output_dpath

    def dump_model_config(self):
        """
        Write something to disk that someone can use to reconstruct the model
        to load weights into. WHY DONT THESE LIBRARIES TO THIS?!
        """
        detectron_config_fpath = (self.output_dpath / 'detectron_config.yaml')
        config_text = self.cfg.dump()
        detectron_config_fpath.write_text(config_text)

    def build_trainer(self):
        from detectron2.engine import DefaultTrainer
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        model = trainer.model
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params}")
        self.trainer = trainer

    def train(self):
        self.dump_model_config()
        if self.trainer is None:
            self.build_trainer()
        import rich
        rich.print(f'Output Dpath: [link={self.output_dpath}]{self.output_dpath}[/link]')
        self.trainer.train()
        rich.print(f'Output Dpath: [link={self.output_dpath}]{self.output_dpath}[/link]')
