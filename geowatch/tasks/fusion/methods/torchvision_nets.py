
"""
Example:
    from geowatch.tasks.fusion.datamodules.network_io import RGBImageBatchItem
    item0 = RGBImageBatchItem.demo(index=0)
    item1 = RGBImageBatchItem.demo(index=1)
    batch_items = [item0, item1]
    self.imdata_chw.shape
    self.channels

    from geowatch.tasks.fusion.methods.torchvision_efficientnet import *  # NOQA
    self = FCNResNet50()
    batch = self.collate(batch_items)
    out = self.forward(batch)

    self = EfficientNetB7()
    out = self.forward(batch)

"""
import torch
import pytorch_lightning as pl
# from torch.nn import functional as F
from geowatch.tasks.fusion.methods import heads as heads_module
from geowatch.tasks.fusion.methods.watch_module_mixins import WatchModuleMixins
from geowatch.utils.util_netharn import InputNorm, Identity
import ubelt as ub  # NOQA
from geowatch.tasks.fusion.datamodules import network_io
from typing import Iterator  # NOQA
# from geowatch.tasks.fusion.methods.loss import coerce_criterion
# from geowatch.tasks.fusion.methods.heads import TaskHeads


try:
    from line_profiler import profile
except Exception:
    profile = ub.identity


class TorchvisionWrapper(pl.LightningModule):

    # def __init__(self, heads):
    #     super().__init__()
    #     self.ots_model = self.define_ots_model()
    #     self.heads = heads_module.TaskHeads(heads)

    def define_ots_model(self):
        """
        This should define the backbone model.
        """
        raise NotImplementedError('Child class must define this')

    def forward(self, batch):
        imdata_bchw = batch['imdata_bchw']
        # self.ots_model.features.forward(imdata_bchw).shape
        out = self.ots_model.forward(imdata_bchw)
        return out

    def collate(self, batch_items):
        imdatas = [batch_item.imdata_chw for batch_item in batch_items]
        imdata_bchw = torch.stack(imdatas)
        nonlocal_class_ohes = [batch_item.nonlocal_class_ohe for batch_item in batch_items]
        nonlocal_class_ohe = torch.stack(nonlocal_class_ohes)
        batch = {
            'imdata_bchw': imdata_bchw,
            'nonlocal_class_ohe': nonlocal_class_ohe,
        }
        return batch


class TorchvisionSegmentationWrapper(TorchvisionWrapper):
    ...


class TorchvisionClassificationWrapper(TorchvisionWrapper):
    ...


class TorchvisionDetectionWrapper(TorchvisionWrapper):
    ...


class EfficientNetB7(TorchvisionClassificationWrapper):
    def define_ots_model(self):
        import torchvision
        ots_model = torchvision.models.efficientnet_b7(weights='EfficientNet_B7_Weights.IMAGENET1K_V1')
        # adaptpool = ots_model.avgpool  # adaptive
        # head = ots_model.classifier  # 0x1000 classifier
        # out_feat = ots_model.features[-1]  # 640x2560 output features.
        # backbone = ots_model.features[0:]  # 640x2560 output features.
        # stem = ots_model.features[0]  # 3x64 input stem
        return ots_model


class FCNResNet50(TorchvisionSegmentationWrapper, WatchModuleMixins):
    """
    Ignore:
        from geowatch.tasks.fusion.datamodules.network_io import RGBImageBatchItem
        item1 = RGBImageBatchItem.demo()
        item2 = RGBImageBatchItem.demo()
        batch_items = [item1, item2]
        from geowatch.tasks.fusion.methods.torchvision_nets import *  # NOQA
        heads = ub.codeblock(
            '''
            feat_dim: 2048
            tasks:
                # Mirrors the simple FCNHead in torchvision
                - name: saliency
                  type: mlp
                  hidden_channels: [256]
                  out_channels: 2
                  dropout: 0.1
                  norm: batch
                  loss:
                      type: focal
                      gamma: 2.0
                  global_weight: 1.0
            ''')
        self = FCNResNet50(
            heads=head_text,
        )
        batch = self.collate(batch_items)

    Ignore:
        >>> # Test with datamodule
        >>> import ubelt as ub
        >>> from geowatch.tasks.fusion import datamodules
        >>> datamodule = datamodules.kwcoco_video_data.KWCocoVideoDataModule(
        >>>     train_dataset='special:vidshapes8', chip_size=32,
        >>>     batch_size=1, time_steps=1, num_workers=2, normalize_inputs=10)
        >>> datamodule.setup('fit')
        >>> dataset_stats = datamodule.torch_datasets['train'].cached_dataset_stats(num=3)
        >>> classes = datamodule.torch_datasets['train'].classes
        >>> # Use one of our fusion.architectures in a test
        >>> self = FCNResNet50(
        >>>     classes=classes,
        >>>     dataset_stats=dataset_stats, input_sensorchan=datamodule.input_sensorchan)

    """
    def __init__(self, heads, classes=10, dataset_stats=None):
        super().__init__()
        self.ots_model = self.define_ots_model()
        feat_dim = self.ots_model.backbone.layer4[2].conv3.out_channels
        self.automatic_optimization = True
        assert feat_dim == 2048, 'hard coded sanity check'

        # import kwcoco
        # self.classes = kwcoco.CategoryTree.coerce(classes)
        self.num_classes = len(self.classes)
        self.heads = heads_module.TaskHeads(heads)

    def define_ots_model(self):
        import torchvision
        ots_model = torchvision.models.segmentation.fcn_resnet50(weights='FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1')
        # num_classes = len(self.classes)
        # torchvision.models.segmentation.FCNHead(2048, num_classes)
        return ots_model

    def forward(self, batch):
        # imdata_bchw = batch['imdata_bchw']
        # input_hw = imdata_bchw.shape[-2:]
        # self.ots_model.features.forward(imdata_bchw).shape
        # downscaled_feats = self.ots_model.backbone.forward(imdata_bchw)['out']
        # downscaled_task_outs = self.heads(downscaled_feats)
        raise NotImplementedError

        # x = F.interpolate(downscaled_feats, size=input_hw, mode="bilinear", align_corners=False)
        # out = self.ots_model.forward(imdata_bchw)['out']
        # return out

    def save_package(self, package_path, verbose=1):
        self._save_package(package_path, verbose=verbose)

    def forward_step(self, batch, batch_idx=None, with_loss=True):
        raise NotImplementedError
        # outputs = {
        #     "change_probs": [
        #         [
        #             0.5 * torch.ones(*frame["output_dims"])
        #             for frame in example["frames"]
        #             if frame["change"] is not None
        #         ]
        #         for example in batch
        #     ],
        #     "saliency_probs": [
        #         [
        #             torch.ones(*frame["output_dims"], 2).sigmoid()
        #             for frame in example["frames"]
        #         ]
        #         for example in batch
        #     ],
        #     "class_probs": [
        #         [
        #             torch.ones(*frame["output_dims"], self.num_classes).softmax(dim=-1)
        #             for frame in example["frames"]
        #         ]
        #         for example in batch
        #     ],
        # }

        # if with_loss:
        #     outputs["loss"] = self.dummy_param
        # return outputs


class Resnet50(TorchvisionClassificationWrapper, WatchModuleMixins):
    """
    Ignore:
        >>> import ubelt as ub
        >>> from geowatch.tasks.fusion import datamodules
        >>> datamodule = datamodules.kwcoco_video_data.KWCocoVideoDataModule(
        >>>     train_dataset='special:vidshapes8', chip_size=32,
        >>>     batch_size=1, time_steps=1, num_workers=2, normalize_inputs=10)
        >>> datamodule.setup('fit')
        >>> dataset_stats = datamodule.torch_datasets['train'].cached_dataset_stats(num=3)
        >>> dataset = datamodule.torch_datasets['train']
        >>> classes = dataset.predictable_classes
        >>> dataset.requested_tasks['nonlocal_class'] = True
        >>> item1 = dataset[0]
        >>> item2 = dataset[1]
        >>> batch_items = [item1, item2]
        >>> from geowatch.tasks.fusion.methods.torchvision_nets import *  # NOQA
        >>> # Use one of our fusion.architectures in a test
        >>> heads = ub.codeblock(
        >>>     '''
        >>>         # Mirrors the simple FCNHead in torchvision
        >>>         - name: nonlocal_class
        >>>           type: mlp
        >>>           hidden_channels: [256]
        >>>           out_channels: 4
        >>>           dropout: 0.1
        >>>           norm: batch
        >>>           loss:
        >>>               type: focal
        >>>               gamma: 2.0
        >>>           head_weight: 1.0
        >>>     ''')
        >>> self = Resnet50(
        >>>     heads=heads,
        >>>     classes=classes,
        >>>     dataset_stats=dataset_stats)
        >>> outputs = self.forward_step(batch_items, with_loss=True)
        >>> canvas = datamodule.draw_batch(batch_items, outputs=outputs)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(canvas)
        >>> kwplot.show_if_requested()

    """

    def __init__(self, heads, classes=None, dataset_stats=None):
        super().__init__()

        input_stats = self.set_dataset_specific_attributes(None, dataset_stats)
        assert len(input_stats) == 1
        stats = list(input_stats.values())[0]
        norm_kw = {'mean': stats['mean'][None, ...],
                   'std': stats['std'][None, ...]}
        self.input_norms = InputNorm(**norm_kw)
        self.ots_model = ots_model = self.define_ots_model()
        feat_dim = ots_model.layer4[-1].conv3.out_channels

        import kwcoco
        self.classes = kwcoco.CategoryTree.coerce(classes)
        self.num_classes = len(self.classes)

        assert feat_dim == 2048, 'hard coded sanity check'
        self.heads = heads_module.TaskHeads(heads, feat_dim=feat_dim)
        self.automatic_optimization = True

    def forward(self, batch):
        imdata_bchw = batch['imdata_bchw']

        # Mean/Std Normalize the Input Batch
        imdata_bchw = self.input_norms(imdata_bchw)
        imdata_bchw.nan_to_num_()

        # Compute backbone features
        feats = self.ots_model(imdata_bchw)

        outputs = self.heads(feats)
        return outputs

    def forward_step(self, batch, with_loss=False, stage='unspecified'):

        if stage == 'train':
            if not self.automatic_optimization:
                # Do we have to do this ourselves?
                # https://lightning.ai/docs/pytorch/stable/common/optimization.html
                opt = self.optimizers()
                opt.zero_grad()

        batch_size = len(batch['imdata_bchw'])
        outputs = self.forward(batch)

        if with_loss:
            losses = self.heads.compute_loss(outputs, batch)
            outputs.update(losses)
            total_loss = losses['loss']
            self.log(f'{stage}_loss', total_loss, prog_bar=True, batch_size=batch_size)

        outputs = network_io.CollatedNetworkOutputs(outputs)

        if stage == 'train':
            if not self.automatic_optimization:
                loss = outputs['loss']
                self.manual_backwards(loss)
        return outputs

    def define_ots_model(self):
        import torchvision
        ots_model = torchvision.models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2')
        # Hack off the head
        ots_model.fc = Identity()
        return ots_model

    def _to_collated(self, batch_items):
        from geowatch.tasks.fusion.datamodules import network_io
        self._cpu_batch_items = batch_items
        batch_items = network_io.UncollatedRGBImageBatch.from_items(batch_items)
        batch = batch_items.collate()
        batch = batch.to(self.device)
        return batch

    def _grab_batch_from_dataloader(self, dataloader_iter):
        raw_item = next(dataloader_iter)
        batch_items = raw_item
        return self._to_collated(batch_items)

    # These train / vali / test specific methods should be moved to a mixin

    # def training_step(self, dataloader_iter: Iterator) -> None:
    #     self._DataLoaderIterDataFetcher_training_step(dataloader_iter)

    # def training_step(self, batch, batch_idx=None):
    #     return self._PrefetchDataFetcher_training_step(batch, batch_idx)

    # def _DataLoaderIterDataFetcher_training_step(self, dataloader_iter) -> None:
    #     # it is the user responsibility to fetch and move the batch to the right device.
    #     # batch, batch_idx, dataloader_idx
    #     self._PrefetchDataFetcher_training_step(batch)

    # def training_step(self, batch):
    def training_step(self, dataloader_iter: Iterator) -> None:
        # self._grab_batch_from_dataloader()
        batch = self._to_collated(next(dataloader_iter))
        outputs = self.forward_step(batch, with_loss=True, stage='train')
        return outputs

    # # def validation_step(self, batch, batch_idx=None):
    # def validation_step(self, dataloader_iter: Iterator) -> None:
    #     batch = self._to_collated(next(dataloader_iter))
    #     outputs = self.forward_step(batch, with_loss=True, stage='val')
    #     return outputs

    # # def test_step(self, batch, batch_idx=None):
    # def test_step(self, dataloader_iter: Iterator) -> None:
    #     batch = self._to_collated(next(dataloader_iter))
    #     outputs = self.forward_step(batch, with_loss=True, stage='test')
    #     return outputs

    # @profile
    # def on_before_batch_transfer(self, batch_items, dataloader_idx):
    #     from geowatch.tasks.fusion.datamodules import network_io
    #     self._cpu_batch_items = batch_items
    #     batch_items = network_io.UncollatedRGBImageBatch.from_items(batch_items)
    #     batch = batch_items.collate()
    #     return batch

    # def on_after_batch_transfer(batch, dataloader_idx):
    #     ...

    # def transfer_batch_to_device(batch, device, dataloader_idx):
    #     ...
