"""
Defines a lightning DataModule for kwcoco video data.

The parameters to each are handled by scriptconfig objects, which prevents us
from needing to specify what the available options are in multiple places.
"""
import os
import kwcoco
import kwimage
import ndsampler
import pathlib
import pytorch_lightning as pl
import ubelt as ub
from typing import Dict, List  # NOQA
import scriptconfig as scfg


from watch.utils.lightning_ext import util_globals
from watch.tasks.fusion import utils
from watch.tasks.fusion.datamodules.kwcoco_dataset import KWCocoVideoDatasetConfig, KWCocoVideoDataset

try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


class KWCocoVideoDataModuleConfig(scfg.Config):
    """
    These are the argument accepted by the KWCocoDataModule.

    The scriptconfig class is not used directly as it normally would be here.
    Instead we use it as a convinience to minimize lightning boilerplate later
    when it constructs its own argparse object, and for handling arguments
    passed directly to the KWCocoDataModule

    In the future this might be convertable to, or handled by omegaconfig
    """
    default = ub.udict({
        'train_dataset': scfg.Value(None, help='path to the train kwcoco file'),
        'vali_dataset': scfg.Value(None, help='path to the validation kwcoco file'),
        'test_dataset': scfg.Value(None, help='path to the test kwcoco file'),

        'batch_size': scfg.Value(4, type=int),
        'normalize_inputs': scfg.Value(True, help=ub.paragraph(
            '''
            if True, computes the mean/std for this dataset on each mode
            so this can be passed to the model.
            ''')),

        'num_workers': scfg.Value(4, type=str, help=ub.paragraph(
            '''
            number of background workers. Can be auto or an avail
            expression. TODO: rename to data_workers?
            ''')),

        'torch_sharing_strategy': scfg.Value('default', help=ub.paragraph(
            '''
            Torch multiprocessing sharing strategy. Can be 'default',
            "file_descriptor", "file_system". On linux, the default is
            "file_descriptor".

            See https://pytorch.org/docs/stable/multiprocessing.html#sharing-strategies
            for descriptions of options.

            When using sqlview=True, using "file_system" can help prevent the
            "received 0 items of ancdata" Error. It is unclear why using
            "file_descriptor" fails in this case for some datasets.
            ''')),

        'torch_start_method': scfg.Value('default', help=ub.paragraph(
            '''
            Torch multiprocessing sharing strategy. Can be "default", "fork",
            "spawn", "forkserver". The default method on Linux is "spawn".
            ''')),

        'sqlview': scfg.Value(False, help=ub.paragraph(
            '''
            If True, use SQL views when reading COCO datasets.
            ''')),
        # Mixin the dataset config
    }) | KWCocoVideoDatasetConfig.default

    def normalize(self):
        # hack because we dont have proper inheritence
        KWCocoVideoDatasetConfig.normalize(self)


class KWCocoVideoDataModule(pl.LightningDataModule):
    """
    Prepare the kwcoco dataset as torch video datamodules

    Example:
        >>> # Demo of the data module on auto-generated toy data
        >>> from watch.tasks.fusion.datamodules.kwcoco_datamodule import *  # NOQA
        >>> import watch
        >>> import kwcoco
        >>> coco_dset = watch.coerce_kwcoco('vidshapes8-watch')
        >>> channels = None
        >>> batch_size = 1
        >>> time_steps = 3
        >>> chip_size = 416
        >>> self = KWCocoVideoDataModule(
        >>>     train_dataset=coco_dset,
        >>>     test_dataset=None,
        >>>     batch_size=batch_size,
        >>>     normalize_inputs=8,
        >>>     channels=channels,
        >>>     num_workers=0,
        >>>     time_steps=time_steps,
        >>>     chip_size=chip_size,
        >>>     neg_to_pos_ratio=0,
        >>> )
        >>> self.setup('fit')
        >>> dl = self.train_dataloader()
        >>> dataset = dl.dataset
        >>> batch = next(iter(dl))
        >>> batch = [dl.dataset[0]]
        >>> # Visualize
        >>> canvas = self.draw_batch(batch)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(canvas)
        >>> kwplot.show_if_requested()

    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> # Run the following tests on real watch data if DVC is available
        >>> from watch.tasks.fusion.datamodules.kwcoco_datamodule import *  # NOQA
        >>> import watch
        >>> import kwcoco
        >>> dvc_dpath = watch.find_dvc_dpath()
        >>> coco_fpath = dvc_dpath / 'Drop2-Aligned-TA1-2022-02-15/combo_ILM.kwcoco.json'
        >>> #coco_fpath = dvc_dpath / 'Aligned-Drop2-TA1-2022-03-07/combo_DILM.kwcoco.json'
        >>> #coco_fpath = dvc_dpath / 'Drop2-Aligned-TA1-2022-02-15/combo_DILM.kwcoco.json'
        >>> dset = kwcoco.CocoDataset(coco_fpath)
        >>> images = dset.images()
        >>> train_dataset = dset
        >>> #sub_images = dset.videos(names=['KR_R002']).images[0]
        >>> #train_dataset = dset.subset(sub_images.lookup('id'))
        >>> test_dataset = None
        >>> img = ub.peek(train_dataset.imgs.values())
        >>> chan_info = kwcoco_extensions.coco_channel_stats(dset)
        >>> #channels = chan_info['common_channels']
        >>> channels = 'blue|green|red|nir|swir16|swir22,forest|bare_ground,matseg_0|matseg_1|matseg_2,invariants.0:3,cloudmask'
        >>> #channels = 'blue|green|red|depth'
        >>> #chan_spec = kwcoco.channel_spec.FusedChannelSpec.coerce(channels)
        >>> #channels = None
        >>> #
        >>> batch_size = 1
        >>> time_steps = 8
        >>> chip_size = 512
        >>> datamodule = KWCocoVideoDataModule(
        >>>     train_dataset=train_dataset,
        >>>     test_dataset=test_dataset,
        >>>     batch_size=batch_size,
        >>>     channels=channels,
        >>>     num_workers=0,
        >>>     normalize_inputs=8,
        >>>     time_steps=time_steps,
        >>>     chip_size=chip_size,
        >>>     neg_to_pos_ratio=0,
        >>>     min_spacetime_weight=0.5,
        >>> )
        >>> datamodule.setup('fit')
        >>> dl = datamodule.train_dataloader()
        >>> dataset = dl.dataset
        >>> dataset.requested_tasks['change'] = False
        >>> dataset.disable_augmenter = True
        >>> target = 0
        >>> item, *_ = batch = [dataset[target]]
        >>> #item, *_ = batch = next(iter(dl))
        >>> # Visualize
        >>> canvas = datamodule.draw_batch(batch)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(canvas, doclf=1)
        >>> kwplot.show_if_requested()

    Example:
        >>> # xdoctest: +SKIP
        >>> # NOTE: I DONT KNOW WHY THIS IS FAILING ON CI AT THE MOMENT. FIXME!
        >>> # Run the data module on coco demo datamodules for the CI
        >>> from watch.tasks.fusion.datamodules.kwcoco_datamodule import *  # NOQA
        >>> import kwcoco
        >>> train_dataset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=5)
        >>> test_dataset = kwcoco.CocoDataset.demo('vidshapes1-multispectral', num_frames=5)
        >>> channels = '|'.join([aux['channels'] for aux in train_dataset.imgs[1]['auxiliary']])
        >>> chan_spec = kwcoco.channel_spec.FusedChannelSpec.coerce(channels)
        >>> #
        >>> batch_size = 2
        >>> time_steps = 3
        >>> chip_size = 128
        >>> channels = channels
        >>> self = KWCocoVideoDataModule(
        >>>     train_dataset=train_dataset,
        >>>     test_dataset=test_dataset,
        >>>     batch_size=batch_size,
        >>>     channels=channels,
        >>>     num_workers=0,
        >>>     time_steps=time_steps,
        >>>     chip_size=chip_size,
        >>>     normalize_inputs=True,
        >>> )
        >>> self.setup('fit')
        >>> dl = self.train_dataloader()
        >>> item, *_ = batch = next(iter(dl))
        >>> expect_shape = (batch_size, time_steps, len(chan_spec), chip_size, chip_size)
        >>> assert len(batch) == batch_size
        >>> for item in batch:
        ...     assert len(item['frames']) == time_steps
        ...     for mode_key, mode_val in item['frames'][0]['modes'].items():
        ...         assert mode_val.shape[1:3] == (chip_size, chip_size)
    """

    __scriptconfig__ = KWCocoVideoDataModuleConfig

    def __init__(self, verbose=1, **kwargs):
        """
        For details on accepted arguments see KWCocoVideoDataModuleConfig
        """
        super().__init__()
        self.verbose = verbose
        self.config = KWCocoVideoDataModuleConfig(cmdline=0, data=kwargs)
        cfgdict = self.config.to_dict()
        self.save_hyperparameters(cfgdict)
        # Backwards compatibility. Previous iterations had the
        # config saved directly as datamodule arguments
        # print('cfgdict = {}'.format(ub.repr2(cfgdict, nl=1)))
        self.__dict__.update(cfgdict)
        self.train_kwcoco = self.config['train_dataset']
        self.vali_kwcoco = self.config['vali_dataset']
        self.test_kwcoco = self.config['test_dataset']

        common_keys = set(KWCocoVideoDatasetConfig.default.keys())
        # Pass the relevant parts of the config to the underlying datasets
        self.train_dataset_config = ub.dict_subset(cfgdict, common_keys)
        # with small changes made for validation and test datasets.
        self.vali_dataset_config = self.train_dataset_config.copy()
        self.vali_dataset_config['chip_overlap'] = 0.0
        self.vali_dataset_config['neg_to_pos_ratio'] = 0.0
        self.vali_dataset_config['use_grid_positives'] = True
        self.vali_dataset_config['use_centered_positives'] = False

        self.test_dataset_config = self.vali_dataset_config.copy()

        self.num_workers = util_globals.coerce_num_workers(cfgdict['num_workers'])
        self.dataset_stats = None

        # will only correspond to train
        self.classes = None
        # self.input_channels = None
        self.input_sensorchan = None

        # Store train / test / vali
        self.torch_datasets: Dict[str, KWCocoVideoDataset] = {}
        self.coco_datasets: Dict[str, kwcoco.CocoDataset] = {}

        self.requested_tasks = None
        self.did_setup = False

        if self.verbose:
            print('Init KWCocoVideoDataModule')
            print('self.train_kwcoco = {!r}'.format(self.train_kwcoco))
            print('self.vali_kwcoco = {!r}'.format(self.vali_kwcoco))
            print('self.test_kwcoco = {!r}'.format(self.test_kwcoco))
            print('self.time_steps = {!r}'.format(self.time_steps))
            print('self.chip_dims = {!r}'.format(self.chip_dims))
            print('self.input_sensorchan = {!r}'.format(self.input_sensorchan))

    def setup(self, stage):
        import watch
        if self.verbose:
            print('Setup DataModule: stage = {!r}'.format(stage))

        util_globals.configure_global_attributes(**{
            'num_workers': self.num_workers,
            'torch_sharing_strategy': self.torch_sharing_strategy,
            'torch_start_method': self.torch_start_method,
        })
        sqlview = self.config['sqlview']

        if stage == 'fit' or stage is None:
            train_data = self.train_kwcoco
            if isinstance(train_data, pathlib.Path):
                train_data = os.fspath(train_data.expanduser())

            if self.verbose:
                print('Build train kwcoco dataset')
            train_coco_dset = watch.demo.coerce_kwcoco(train_data,
                                                       sqlview=sqlview)
            self.coco_datasets['train'] = train_coco_dset

            print('self.exclude_sensors', self.exclude_sensors)
            coco_train_sampler = ndsampler.CocoSampler(train_coco_dset)
            train_dataset = KWCocoVideoDataset(
                coco_train_sampler, mode='fit', **self.train_dataset_config,
            )

            # Unfortunately lightning seems to only enable / disables
            # validation depending on the methods that are defined, so we are
            # not able to statically define them.
            self.classes = train_dataset.classes
            self.torch_datasets['train'] = train_dataset
            ub.inject_method(self, lambda self: self._make_dataloader('train', shuffle=True), 'train_dataloader')

            # if self.input_channels is None:
            #     self.input_channels = train_dataset.input_channels

            if self.input_sensorchan is None:
                self.input_sensorchan = train_dataset.input_sensorchan

            stats_params = {
                'num': None,
                'with_intensity': False,
                'with_class': True,
                'num_workers': self.num_workers,
                'batch_size': self.batch_size,
            }
            if self.normalize_inputs:
                if isinstance(self.normalize_inputs, str):
                    if self.normalize_inputs == 'transfer':
                        # THIS MEANS WE EXPECT THAT WE CAN TRANSFER FROM AN
                        # EXISTING MODEL. THE FIT METHOD MUST HANDLE THIS
                        stats_params = None
                    else:
                        raise NotImplementedError(
                            'TODO: handle special normalization keys, '
                            'e.g. imagenet')
                else:
                    if isinstance(self.normalize_inputs, int):
                        stats_params['num'] = self.normalize_inputs
                    else:
                        stats_params['num'] = None
            else:
                stats_params['with_intensity'] = False

            # Hack for now:
            # TODO: Note: also need for class weights
            if stats_params is not None:
                print(f'stats_params={stats_params}')
                self.dataset_stats = train_dataset.cached_dataset_stats(**stats_params)

            if self.vali_kwcoco is not None:
                # Explicit validation dataset should be prefered
                vali_data = self.vali_kwcoco
                if isinstance(vali_data, pathlib.Path):
                    vali_data = os.fspath(vali_data.expanduser())
                if self.verbose:
                    print('Build validation kwcoco dataset')
                kwcoco_ds = watch.demo.coerce_kwcoco(vali_data,
                                                     sqlview=sqlview)
                vali_coco_sampler = ndsampler.CocoSampler(kwcoco_ds)
                vali_dataset = KWCocoVideoDataset(
                    vali_coco_sampler, mode='vali', **self.vali_dataset_config)
                self.torch_datasets['vali'] = vali_dataset
                ub.inject_method(self, lambda self: self._make_dataloader('vali', shuffle=False), 'val_dataloader')

        if stage == 'test' or stage is None:
            test_data = self.test_kwcoco
            if isinstance(test_data, pathlib.Path):
                test_data = os.fspath(test_data.expanduser())
            if self.verbose:
                print('Build test kwcoco dataset')
            test_coco_dset = watch.demo.coerce_kwcoco(test_data,
                                                      sqlview=sqlview)
            test_coco_sampler = ndsampler.CocoSampler(test_coco_dset)
            self.coco_datasets['test'] = test_coco_dset
            self.torch_datasets['test'] = KWCocoVideoDataset(
                test_coco_sampler, mode='test', **self.test_dataset_config,
            )
            ub.inject_method(self, lambda self: self._make_dataloader('test', shuffle=False), 'test_dataloader')

        print('self.torch_datasets = {}'.format(ub.repr2(self.torch_datasets, nl=1)))
        self._notify_about_tasks(self.requested_tasks)
        self.did_setup = True

    @property
    def train_dataset(self):
        return self.torch_datasets.get('train', None)

    @property
    def test_dataset(self):
        return self.torch_datasets.get('test', None)

    @property
    def vali_dataset(self):
        return self.torch_datasets.get('vali', None)

    def _make_dataloader(self, stage, shuffle=False):
        # import nonechucks
        # nonechucks.SafeDataset
        loader = self.torch_datasets[stage].make_loader(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=True,
        )
        return loader

    def _notify_about_tasks(self, requested_tasks=None, model=None):
        """
        Hacky method. Given the multimodal model, tell all the datasets which
        tasks they will need to generate data for. (This helps make the
        visualizations cleaner).
        """
        if model is not None:
            assert requested_tasks is None
            requested_tasks = {k: w > 0 for k, w in model.global_head_weights.items()}
        print(f'datamodule notified: requested_tasks={requested_tasks}')
        if requested_tasks is not None:
            self.requested_tasks = requested_tasks
            for dataset in self.torch_datasets.values():
                dataset._notify_about_tasks(requested_tasks)

    @classmethod
    def add_argparse_args(cls, parent_parser):
        """
        Previously the arguments were in multiple places including here.  This
        has been updated to use the :class:`KWCocoVideoDataModuleConfig` as the
        single point where arguments are defined. The functionality of this
        method is roughly the same as it used to be given that scriptconfig
        objects can be transformed into argparse objects.

        CommandLine:
            xdoctest -m /home/joncrall/code/watch/watch/tasks/fusion/datamodules/kwcoco_datamodule.py add_argparse_args

        Example:
            >>> from watch.tasks.fusion.datamodules.kwcoco_datamodule import *  # NOQA
            >>> cls = KWCocoVideoDataModule
            >>> # TODO: make use of watch.utils.lightning_ext import argparse_ext
            >>> import argparse
            >>> parent_parser = argparse.ArgumentParser()
            >>> cls.add_argparse_args(parent_parser)
            >>> parent_parser.print_help()
            >>> args, _ = parent_parser.parse_known_args(['--use_grid_positives=True'])
            >>> assert args.use_grid_positives
            >>> args, _ = parent_parser.parse_known_args(['--use_grid_positives=False'])
            >>> assert not args.use_grid_positives
            >>> args, _ = parent_parser.parse_known_args(['--exclude_sensors=l8,f3'])
            >>> assert args.exclude_sensors == 'l8,f3'
            >>> args, _ = parent_parser.parse_known_args(['--exclude_sensors=l8'])
            >>> assert args.exclude_sensors == 'l8'
        """
        # from functools import partial
        parser = parent_parser.add_argument_group('kwcoco_datamodule')
        config = KWCocoVideoDataModuleConfig(cmdline=0)
        config.argparse(parser)
        return parent_parser

    @classmethod
    def compatible(cls, cfgdict):
        """
        Given keyword arguments, find the subset that is compatible with this
        constructor. This is somewhat hacked because of usage of scriptconfig,
        but could be made nicer by future updates.
        """
        # init_kwargs = ub.compatible(config, cls.__init__)
        import inspect
        nameable_kinds = {inspect.Parameter.POSITIONAL_OR_KEYWORD,
                          inspect.Parameter.KEYWORD_ONLY}
        cls_sig = inspect.signature(cls)
        explicit_argnames = [
            argname for argname, argtype in cls_sig.parameters.items()
            if argtype.kind in nameable_kinds
        ]
        valid_argnames = explicit_argnames + list(KWCocoVideoDataModuleConfig.default.keys())
        datamodule_vars = ub.dict_isect(cfgdict, valid_argnames)
        return datamodule_vars

    def draw_batch(self, batch, stage='train', outputs=None, max_items=2,
                   overlay_on_image=False, **kwargs):
        """
        Visualize a batch produced by this DataSet.

        Args:
            batch (Dict[str, List[Tensor]]): dictionary of uncollated lists of Dataset Items
                change: [ [T-1, H, W] \in [0, 1] \forall examples ]
                saliency: [ [T, H, W, 2] \in [0, 1] \forall examples ]
                class: [ [T, H, W, 10] \in [0, 1] \forall examples ]

            outputs (Dict[str, Tensor]):
                maybe-collated list of network outputs?

        Example:
            >>> from watch.tasks.fusion.datamodules.kwcoco_datamodule import *  # NOQA
            >>> from watch.tasks.fusion import datamodules
            >>> self = datamodules.KWCocoVideoDataModule(
            >>>     train_dataset='special:vidshapes8-multispectral', num_workers=0)
            >>> self.setup('fit')
            >>> loader = self.train_dataloader()
            >>> batch = next(iter(loader))
            >>> item = batch[0]
            >>> # Visualize
            >>> B = len(batch)
            >>> C, H, W = ub.peek(item['frames'][0]['modes'].values()).shape
            >>> T = len(item['frames'])
            >>> import torch
            >>> outputs = {'change_probs': [torch.rand(T - 1, H, W) for _ in range(B)]}
            >>> outputs.update({'class_probs': [torch.rand(T, H, W, 10) for _ in range(B)]})
            >>> stage = 'train'
            >>> canvas = self.draw_batch(batch, stage=stage, outputs=outputs)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas)
            >>> kwplot.show_if_requested()


        Example:
            >>> # xdoctest: +REQUIRES(--slow)
            >>> from watch.tasks.fusion.datamodules.kwcoco_datamodule import *  # NOQA
            >>> from watch.tasks.fusion import datamodules
            >>> import watch
            >>> train_dataset = watch.demo.demo_kwcoco_multisensor()
            >>> self = datamodules.KWCocoVideoDataModule(
            >>>     train_dataset=train_dataset, chip_size=256, time_steps=5, num_workers=0, batch_size=3)
            >>> self.setup('fit')
            >>> loader = self.train_dataloader()
            >>> batch_iter = iter(loader)
            >>> batch = next(batch_iter)
            >>> item = batch[0]
            >>> # Visualize
            >>> B = len(batch)
            >>> outputs = {'change_probs': [], 'class_probs': [], 'saliency_probs': []}
            >>> # Add dummy outputs
            >>> import torch
            >>> for item in batch:
            >>>     [v.append([]) for v in outputs.values()]
            >>>     for frame_idx, frame in enumerate(item['frames']):
            >>>         H, W = frame['class_idxs'].shape
            >>>         if frame_idx > 0:
            >>>             outputs['change_probs'][-1].append(torch.rand(H, W))
            >>>         outputs['class_probs'][-1].append(torch.rand(H, W, 10))
            >>>         outputs['saliency_probs'][-1].append(torch.rand(H, W, 2))
            >>> stage = 'train'
            >>> canvas = self.draw_batch(batch, stage=stage, outputs=outputs, max_items=4)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas)
            >>> kwplot.show_if_requested()
        """
        dataset = self.torch_datasets[stage]
        # Get the raw dataset class
        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset

        # assume collation is disabled
        batch_items = batch

        canvas_list = []
        for item_idx, item in zip(range(max_items), batch_items):
            # HACK: I'm not sure how general accepting outputs is
            # TODO: more generic handling of outputs.
            # Should be able to accept
            # - [ ] binary probability of change
            # - [ ] fine-grained probability of change
            # - [ ] per-frame semenatic segmentation
            item_output = {}
            if outputs is not None:
                item_output = ub.AutoDict()
                for head_key in ['change_probs', 'class_probs', 'saliency_probs']:
                    if head_key in outputs:
                        item_output[head_key] = [f.data.cpu().numpy() for f in outputs[head_key][item_idx]]

            part = dataset.draw_item(item, item_output=item_output, overlay_on_image=overlay_on_image, **kwargs)
            canvas_list.append(part)
        canvas = kwimage.stack_images_grid(
            canvas_list, axis=1, overlap=-12, bg_value=[64, 60, 60])

        with_legend = True
        if with_legend:
            label_to_color = {
                node: data['color']
                for node, data in dataset.classes.graph.nodes.items()}
            label_to_color = ub.sorted_keys(label_to_color)
            legend_img = utils._memo_legend(label_to_color)
            canvas = kwimage.stack_images([canvas, legend_img], axis=1)

        return canvas