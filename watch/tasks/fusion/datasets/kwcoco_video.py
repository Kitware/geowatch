import kwcoco
import ndsampler
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils import data
import pathlib
from watch.tasks.fusion.datasets import common
import ubelt as ub
from watch.tasks.fusion import utils
from functools import partial


class WatchDataModule(pl.LightningDataModule):
    """
    Prepare the kwcoco dataset as torch video datasets


    Ignore:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> # Run the following tests on real watch data if DVC is available
        >>> from watch.tasks.fusion.datasets.kwcoco_video import *  # NOQA
        >>> from os.path import join
        >>> import os
        >>> _default = ub.expandpath('$HOME/data/dvc-repos/smart_watch_dvc')
        >>> dvc_dpath = os.environ.get('DVC_DPATH', _default)
        >>> coco_fpath = join(dvc_dpath, 'drop1_S2_aligned_c1/data.kwcoco.json')
        >>> import kwcoco
        >>> train_dataset = kwcoco.CocoDataset(coco_fpath)
        >>> test_dataset = None
        >>> img = ub.peek(train_dataset.imgs.values())
        >>> channels = '|'.join([aux['channels'] for aux in img['auxiliary']])
        >>> chan_spec = kwcoco.channel_spec.FusedChannelSpec.coerce(channels)
        >>> #
        >>> batch_size = 2
        >>> time_steps = 3
        >>> chip_size = 330
        >>> channels = channels
        >>> self = WatchDataModule(
        >>>     train_dataset=train_dataset,
        >>>     test_dataset=test_dataset,
        >>>     batch_size=batch_size,
        >>>     channels=channels,
        >>>     num_workers=0,
        >>>     time_steps=time_steps,
        >>>     chip_size=chip_size,
        >>> )
        >>> self.setup("fit")
        >>> dl = self.train_dataloader()
        >>> batch = next(iter(dl))
        >>> expect_shape = (batch_size, time_steps, len(chan_spec), chip_size, chip_size)
        >>> got_shape = tuple(batch['images'].shape)
        >>> assert got_shape == expect_shape
        >>> # Visualize
        >>> canvas = self.draw_batch(batch)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(canvas)
        >>> kwplot.show_if_requested()

    Example:
        >>> # Run the data module on coco demo datasets for the CI
        >>> from watch.tasks.fusion.datasets.kwcoco_video import *  # NOQA
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
        >>> self = WatchDataModule(
        >>>     train_dataset=train_dataset,
        >>>     test_dataset=test_dataset,
        >>>     batch_size=batch_size,
        >>>     channels=channels,
        >>>     num_workers=0,
        >>>     time_steps=time_steps,
        >>>     chip_size=chip_size,
        >>> )
        >>> self.setup("fit")
        >>> dl = self.train_dataloader()
        >>> batch = next(iter(dl))
        >>> expect_shape = (batch_size, time_steps, len(chan_spec), chip_size, chip_size)
        >>> got_shape = tuple(batch['images'].shape)
        >>> assert got_shape == expect_shape

    Example:
        >>> from watch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
        >>> from watch.tasks.fusion import datasets
        >>> datamodule = datasets.WatchDataModule(
        >>>     train_dataset='special:vidshapes8-multispectral', num_workers=0)
        >>> datamodule.setup('fit')
        >>> loader = datamodule.train_dataloader()
        >>> batch = next(iter(loader))

    """
    def __init__(
        self,
        train_dataset=None,
        vali_dataset=None,
        test_dataset=None,
        time_steps=2,
        chip_size=128,
        time_overlap=0,
        chip_overlap=0.1,
        channels=None,
        valid_pct=0.1,
        batch_size=4,
        num_workers=4,
        preprocessing_step=None,
        tfms_channel_subset=None,
    ):
        super().__init__()
        self.train_kwcoco = train_dataset
        self.vali_kwcoco = vali_dataset
        self.test_kwcoco = test_dataset
        self.time_steps = time_steps
        self.chip_size = chip_size
        self.time_overlap = time_overlap
        self.chip_overlap = chip_overlap
        self.channels = channels
        self.valid_pct = valid_pct
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preprocessing_step = preprocessing_step

        # TODO: there is no need for tfms_channel_subset,
        # Remove that parameter and just send ``channels`` in as the requested
        # subset. To handle the case where you want to test how well the model
        # works when channels are missing pass some channel dropout parameter
        # to the dataset.

        if 0:
            tfms_channel_subset = channels if (tfms_channel_subset is None) else tfms_channel_subset
            channel_split = channels.split("|")
            tfms_channel_subset = [
                idx
                for idx, channel in enumerate(tfms_channel_subset.split("|"))
                if channel in channel_split
            ]
            self.tfms_channel_subset = tfms_channel_subset

            self.train_tfms = self.preprocessing_step
            self.test_tfms = transforms.Compose([
                self.preprocessing_step,
                utils.Lambda(lambda x: x[:, tfms_channel_subset]),
            ])

        # Store train / test / vali
        self.torch_datasets = {}
        self.coco_datasets = {}

    def draw_batch(self, batch, stage='train', max_items=2):
        """
        Helper method to draw a batch of data.
        """
        dataset = self.torch_datasets[stage]
        # Get the raw dataset class
        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        import kwimage
        batch_items = _decollate_batch(batch)
        canvas_list = []
        for item_idx, item in zip(range(max_items), batch_items):
            part = dataset.draw_item(item)
            canvas_list.append(part)
        canvas = kwimage.stack_images_grid(canvas_list, axis=1, overlap=-12)
        return canvas

    def setup(self, stage):
        if stage == "fit" or stage is None:
            train_data = self.train_kwcoco
            if isinstance(train_data, pathlib.Path):
                train_data = str(train_data.expanduser())
            train_coco_dset = kwcoco.CocoDataset.coerce(train_data)
            self.coco_datasets['train'] = train_coco_dset

            coco_train_sampler = ndsampler.CocoSampler(train_coco_dset)
            train_dataset = common.VideoDataset(
                coco_train_sampler,
                sample_shape=(self.time_steps, self.chip_size, self.chip_size),
                window_overlap=(self.time_overlap, self.chip_overlap, self.chip_overlap),
                channels=self.channels,
                # transform=self.train_tfms,
            )

            # Unfortunately lightning seems to only enable / disables
            # validation depending on the methods that are defined, so we are
            # not able to statically define them.
            self.torch_datasets['train'] = train_dataset
            ub.inject_method(self, lambda self: self._make_dataloader('train', shuffle=True), 'train_dataloader')

            if self.vali_kwcoco is not None:
                # Explicit validation dataset should be prefered
                vali_data = self.vali_kwcoco
                if isinstance(vali_data, pathlib.Path):
                    vali_data = str(vali_data.expanduser())
                kwcoco_ds = kwcoco.CocoDataset.coerce(vali_data)
                vali_coco_sampler = ndsampler.CocoSampler(kwcoco_ds)
                vali_dataset = common.VideoDataset(
                    vali_coco_sampler,
                    sample_shape=(self.time_steps, self.chip_size, self.chip_size),
                    window_overlap=(self.time_overlap, self.chip_overlap, self.chip_overlap),
                    channels=self.channels,
                    # transform=self.vali_tfms,
                )
                self.torch_datasets['vali'] = vali_dataset
                ub.inject_method(self, lambda self: self._make_dataloader('vali', shuffle=True), 'val_dataloader')
            else:
                if 0:
                    # TODO:
                    raise NotImplementedError(
                        "TODO: Can take different video subsets at the coco level"
                        " but dont do it at the torch dataset level."
                        " too much leakage"
                    )
                    num_examples = len(train_dataset)
                    num_valid = int(self.valid_pct * num_examples)
                    num_train = num_examples - num_valid

                    # FIXME Probably not the right way to do this, too much leakage
                    train_dataset, vali_dataset = data.random_split(
                        train_dataset,
                        [num_train, num_valid],
                    )
                    self.torch_datasets['train'] = train_dataset
                    self.torch_datasets['vali'] = vali_dataset

        if stage == "test" or stage is None:
            test_data = self.test_kwcoco
            if isinstance(test_data, pathlib.Path):
                test_data = str(test_data.expanduser())
            test_coco_dset = kwcoco.CocoDataset.coerce(test_data)
            self.coco_datasets['train'] = test_coco_dset
            test_coco_sampler = ndsampler.CocoSampler(test_coco_dset)
            self.torch_datasets['test'] = common.VideoDataset(
                test_coco_sampler,
                sample_shape=(self.time_steps, self.chip_size, self.chip_size),
                window_overlap=(self.time_overlap, self.chip_overlap, self.chip_overlap),
                channels=self.channels,
                # transform=self.test_tfms,
            )

            ub.inject_method(self, lambda self: self._make_dataloader('train', shuffle=True), 'test_dataloader')

    def _make_dataloader(self, stage, shuffle=False):
        return data.DataLoader(
            self.torch_datasets[stage],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=True,
        )

    # def train_dataloader(self):
    #     return data.DataLoader(
    #         self.torch_datasets['train'],
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         shuffle=True,
    #         pin_memory=True,
    #     )

    # def val_dataloader(self):
    #     return data.DataLoader(
    #         self.torch_datasets['vali'],
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         shuffle=False,
    #         pin_memory=True,
    #     )

    # def test_dataloader(self):
    #     return data.DataLoader(
    #         self.torch_datasets['test'],
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         shuffle=False,
    #         pin_memory=True,
    #     )

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("kwcoco_video")
        parser.add_argument("--train_dataset", default=None, type=pathlib.Path)
        # parser.add_argument("--vali_dataset", default=None, type=pathlib.Path)
        parser.add_argument("--test_dataset", default=None, type=pathlib.Path)
        parser.add_argument("--time_steps", default=2, type=int)
        parser.add_argument("--chip_size", default=128, type=int)
        parser.add_argument("--time_overlap", default=0, type=int)
        parser.add_argument("--chip_overlap", default=0.1, type=float)
        parser.add_argument("--channels", default='<TODO-AUTO>', type=str)
        parser.add_argument("--valid_pct", default=0.1, type=float)
        parser.add_argument("--batch_size", default=4, type=int)
        parser.add_argument("--num_workers", default=4, type=int)
        parser.add_argument("--transform_key", default="none", type=str)
        parser.add_argument("--tfms_scale", default=2000., type=float)
        parser.add_argument("--tfms_window_size", default=8, type=int)
        return parent_parser


def _decollate_batch(batch):
    """
    Breakup a collated batch of BatchContainers back into ItemContainers
    """
    import ubelt as ub
    from kwcoco.util.util_json import IndexableWalker
    import torch
    from netharn.data.data_containers import ItemContainer
    from netharn.data.data_containers import BatchContainer
    walker = IndexableWalker(batch)
    decollated_dict = ub.AutoDict()
    decollated_walker = IndexableWalker(decollated_dict)
    for path, batch_val in walker:
        if isinstance(batch_val, BatchContainer):
            for bx, item_val in enumerate(ub.flatten(batch_val.data)):
                decollated_walker[[bx] + path] = ItemContainer(item_val)
        elif isinstance(batch_val, torch.Tensor):
            for bx, item_val in enumerate(batch_val):
                decollated_walker[[bx] + path] = item_val
    decollated = list(decollated_dict.to_dict().values())
    return decollated
