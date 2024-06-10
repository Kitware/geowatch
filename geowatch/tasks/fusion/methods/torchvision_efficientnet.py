
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
# from geowatch.tasks.fusion.methods.loss import coerce_criterion
# from geowatch.tasks.fusion.methods.heads import TaskHeads


class TorchvisionWrapper(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.ots_model = self.define_ots_model()

    def define_ots_model(self):
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


class FCNResNet50(TorchvisionSegmentationWrapper):

    def define_ots_model(self):
        import torchvision
        ots_model = torchvision.models.segmentation.fcn_resnet50(weights='FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1')
        return ots_model


class EfficientNetB7(TorchvisionWrapper):

    def define_ots_model(self):
        import torchvision
        ots_model = torchvision.models.efficientnet_b7(weights='EfficientNet_B7_Weights.IMAGENET1K_V1')
        # adaptpool = ots_model.avgpool  # adaptive
        # head = ots_model.classifier  # 0x1000 classifier
        # out_feat = ots_model.features[-1]  # 640x2560 output features.
        # backbone = ots_model.features[0:]  # 640x2560 output features.
        # stem = ots_model.features[0]  # 3x64 input stem
        return ots_model
