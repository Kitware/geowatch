
"""
Example:
    from geowatch.tasks.fusion.datamodules.batch_item import RGBImageBatchItem
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
import pytorch_lightning as pl
# from torch.nn import functional as F
from geowatch.tasks.fusion.methods import heads as heads_module
from geowatch.tasks.fusion.methods.watch_module_mixins import WatchModuleMixins
# from geowatch.tasks.fusion.methods.loss import coerce_criterion
# from geowatch.tasks.fusion.methods.heads import TaskHeads


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
        import torch
        imdatas = []
        for batch_item in batch_items:
            imdatas.append(batch_item.imdata_chw)
        imdata_bchw = torch.stack(imdatas)
        batch = {
            'imdata_bchw': imdata_bchw,
        }
        return batch


class TorchvisionSegmentationWrapper(TorchvisionWrapper):
    ...


class TorchvisionClassificationWrapper(TorchvisionWrapper):
    ...


class TorchvisionDetectionWrapper(TorchvisionWrapper):
    ...


# class EfficientNetB7(TorchvisionClassificationWrapper):
#     def define_ots_model(self):
#         import torchvision
#         ots_model = torchvision.models.efficientnet_b7(weights='EfficientNet_B7_Weights.IMAGENET1K_V1')
#         # adaptpool = ots_model.avgpool  # adaptive
#         # head = ots_model.classifier  # 0x1000 classifier
#         # out_feat = ots_model.features[-1]  # 640x2560 output features.
#         # backbone = ots_model.features[0:]  # 640x2560 output features.
#         # stem = ots_model.features[0]  # 3x64 input stem
#         return ots_model


class FCNResNet50(TorchvisionSegmentationWrapper, WatchModuleMixins):
    """
    Ignore:
        from geowatch.tasks.fusion.datamodules.batch_item import RGBImageBatchItem
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
    def __init__(self, heads, dataset_stats=None):
        super().__init__()
        self.ots_model = self.define_ots_model()
        feat_dim = self.ots_model.backbone.layer4[2].conv3.out_channels
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
