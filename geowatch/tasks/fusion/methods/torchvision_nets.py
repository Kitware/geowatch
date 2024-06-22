
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
import torch
import pytorch_lightning as pl
# from torch.nn import functional as F
from geowatch.tasks.fusion.methods import heads as heads_module
from geowatch.tasks.fusion.methods.watch_module_mixins import WatchModuleMixins
from geowatch.utils.util_netharn import InputNorm, Identity
import ubelt as ub  # NOQA
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
        >>>     feat_dim: 2048
        >>>     tasks:
        >>>         # Mirrors the simple FCNHead in torchvision
        >>>         - name: class
        >>>           type: mlp
        >>>           hidden_channels: [256]
        >>>           out_channels: 4
        >>>           dropout: 0.1
        >>>           norm: batch
        >>>           loss:
        >>>               type: focal
        >>>               gamma: 2.0
        >>>           global_weight: 1.0
        >>>     ''')
        >>> self = Resnet50(
        >>>     heads=heads,
        >>>     classes=classes,
        >>>     dataset_stats=dataset_stats)
        >>> self.forward_step(batch_items, with_loss=True)
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
        self.heads = heads_module.TaskHeads(heads)

    def forward(self, batch):
        imdata_bchw = batch['imdata_bchw']
        imdata_bchw = self.input_norms(imdata_bchw)
        feats = self.ots_model(imdata_bchw)

        # Pretend there is a space part
        # spacetime_features = feats[:, None, None, :]
        logits = {}
        if 'class' in self.heads:
            logits['class'] = self.heads['class'](feats)
        if 'saliency' in self.heads:
            logits['saliency'] = self.heads['saliency'](feats)

        resampled_logits = logits  # no resample
        # Convert logits into probabilities for output
        # Remove batch index in both cases
        probs = {}
        if 'class' in resampled_logits:
            criterion_encoding = self.heads["class"].criterion.target_encoding
            _logits = resampled_logits['class'].detach()
            if criterion_encoding == "onehot":
                probs['class'] = _logits.sigmoid()
            elif criterion_encoding == "index":
                probs['class'] = _logits.softmax(dim=-1)
            else:
                raise NotImplementedError

        outputs = {
            'probs': probs,
            'logits': logits,
        }
        return outputs

    def forward_step(self, batch_items, with_loss=False, stage='unspecified'):

        from geowatch.tasks.fusion.datamodules.batch_item import RGBImageBatchItem
        batch_items = [RGBImageBatchItem(item) for item in batch_items]
        batch = self.collate(batch_items)
        batch_outputs = self.forward(batch)

        outputs = {}
        outputs['batch_outputs'] = batch_outputs
        if with_loss:
            item_loss_parts = {}
            # Compute criterion loss for each head
            resampled_logits = batch_outputs['logits']
            import einops
            for head_key, head_logits in resampled_logits.items():
                assert head_key == 'class'
                head_truth = batch['nonlocal_class_ohe']
                truth_encoding = 'ohe'
                head = self.heads[head_key]
                criterion = head.criterion
                head_pred_input = einops.rearrange(head_logits, '(b t h w) c -> ' + criterion.logit_shape, t=1, h=1, w=1).contiguous()

                if criterion.target_encoding == 'index':
                    head_true_idxs = head_truth.long()
                    head_true_input = einops.rearrange(head_true_idxs, '(b t h w) -> ' + criterion.target_shape, t=1, h=1, w=1).contiguous()
                elif criterion.target_encoding == 'onehot':
                    # Note: 1HE is much easier to work with
                    if truth_encoding == 'index':
                        import kwarray
                        head_true_ohe = kwarray.one_hot_embedding(head_truth.long(), criterion.in_channels, dim=-1)
                    elif truth_encoding == 'ohe':
                        head_true_ohe = head_truth
                    else:
                        raise KeyError(truth_encoding)
                    head_true_input = einops.rearrange(head_true_ohe, '(b t h w) c -> ' + criterion.target_shape, t=1, h=1, w=1).contiguous()
                else:
                    raise KeyError(criterion.target_encoding)

                unreduced_head_loss = criterion(head_pred_input, head_true_input)
                # full_head_weight = torch.broadcast_to(head_weights_input, unreduced_head_loss.shape)
                # Weighted reduction
                # EPS_F32 = 1.1920929e-07
                # weighted_head_loss = (full_head_weight * unreduced_head_loss).sum() / (full_head_weight.sum() + EPS_F32)
                weighted_head_loss = unreduced_head_loss
                global_head_weight = 1
                head_loss = global_head_weight * weighted_head_loss
                item_loss_parts[head_key] = head_loss

            total_loss = sum(t.sum() for t in item_loss_parts.values())
            outputs['loss'] = total_loss
            # outputs['item_losses'] = item_losses
        return outputs

    def define_ots_model(self):
        import torchvision
        ots_model = torchvision.models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2')
        # Hack off the head
        ots_model.fc = Identity()
        # adaptpool = ots_model.avgpool  # adaptive
        # head = ots_model.classifier  # 0x1000 classifier
        # out_feat = ots_model.features[-1]  # 640x2560 output features.
        # backbone = ots_model.features[0:]  # 640x2560 output features.
        # stem = ots_model.features[0]  # 3x64 input stem
        return ots_model


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
    def __init__(self, heads, classes=10, dataset_stats=None):
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
