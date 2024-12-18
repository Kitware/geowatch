#!/usr/bin/env python3
"""
TODO:
    - [ ] Custom dataloader [Detectron2CustomDataloader]_

    - [ ] Requirements:
        pip install fairscale timm yacs "pycocotools>=2.0.2" cloudpickle "fvcore>=0.1.5,<0.1.6" "iopath>=0.1.7,<0.1.10" "omegaconf>=2.1,<2.4"

References:
    .. [Detectron2CustomDataloader] https://detectron2.readthedocs.io/en/latest/tutorials/data_loading.html#use-a-custom-dataloader
"""
import scriptconfig as scfg
import ubelt as ub


class DetectronFitCLI(scfg.DataConfig):
    """
    Wrapper around detectron2 trainers
    """
    train_fpath = scfg.Value(None, help='param1')
    vali_fpath = scfg.Value(None, help='param1')
    expt_name = scfg.Value(None, help='param1')
    default_root_dir = scfg.Value('./out')

    # base = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
    # init = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'

    base = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
    init = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'

    cfg = scfg.Value(ub.codeblock(
        '''
        # DATALOADER:
        #     NUM_WORKERS: 2
        # SOLVER:
        #     IMS_PER_BATCH: 2   # This is the real 'batch size' commonly known to deep learning people
        #     BASE_LR: 0.00025   # pick a good LR
        #     MAX_ITER: 120_000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
        #     STEPS: []          # do not decay learning rate
        '''),
        help=ub.paragraph(
            '''
            Overlaid config on top of whatever base config path is specified.

            This is the neat thing about how scriptconfig handles the nested
            config necessary for full detectron control: it doesn't. It just
            handles something that can be coerced into YAML and merged.

            Could be something like:
                * COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml
                * COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
                * new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py
            '''))

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> cmdline = 0
            >>> kwargs = dict()
            >>> cls = DetectronFitCLI
            >>> cls.main(cmdline=cmdline, **kwargs)
        """
        import rich
        from rich.markup import escape
        config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
        rich.print('config = ' + escape(ub.urepr(config, nl=1)))
        detectron_fit(config)


def detectron_fit(config):
    """
    References:
        https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5
    """
    import ubelt as ub
    import geowatch_tpl
    detectron2 = geowatch_tpl.import_submodule('detectron2')  # NOQA
    print(f'detectron2={detectron2}')

    import kwutil
    proc_context = kwutil.ProcessContext(
        name='geowatch.tasks.detectron2.fit',
        config=kwutil.Json.ensure_serializable(dict(config)),
        track_emissions=True,
    )
    proc_context.start()
    print(f'proc_context.obj = {ub.urepr(proc_context.obj, nl=3)}')

    # Handle both types of detectron configs as best as possible.
    if config.base.endswith('.py'):
        from geowatch.tasks.detectron2._new_config_backend import Detectron2WrapperNewStyle
        DetectronWrapper = Detectron2WrapperNewStyle
    else:
        from geowatch.tasks.detectron2._old_config_backend import Detectron2WrapperOldStyle
        DetectronWrapper = Detectron2WrapperOldStyle

    self = DetectronWrapper(config)
    self.register_datasets()
    self.resolve_config()
    self.build_trainer()

    telemetry_fpath1 = self.output_dpath / 'initial_telemetry.json'
    telemetry_fpath1.write_text(kwutil.Json.dumps(proc_context.obj))

    self.train()

    proc_context.stop()
    print(f'proc_context.obj = {ub.urepr(proc_context.obj, nl=3)}')

    telemetry_fpath2 = self.output_dpath / 'final_telemetry.json'
    telemetry_fpath2.write_text(kwutil.Json.dumps(proc_context.obj))

__cli__ = DetectronFitCLI

if __name__ == '__main__':
    """

    CommandLine:
        python -m geowatch.tasks.detectron2.fit
    """
    __cli__.main()
