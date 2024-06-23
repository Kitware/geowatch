"""
# TODO:

* Generalize Task Heads, and hook up to off-the-shelf backbones. We almost
  never will be able to use those models as-is.

* Port _build_item_loss_parts
* Port forward_item
* Port forward_step
* Port forward_foot
"""
import torch
import kwutil
import einops
import kwarray
from geowatch.utils.util_netharn import MultiLayerPerceptronNd


class TaskHeads(torch.nn.ModuleDict):
    """
    Experimental feature. Not finished.

    Sends features to task specific heads.
    """

    def __init__(heads, heads_config, feat_dim=None, classes=None):
        """
        Args:
            heads_config (str | List):
                yaml coercable config containing the user-level configuration
                for the heads.

        Example:
            >>> from geowatch.tasks.fusion.methods.heads import *  # NOQA
            >>> import ubelt as ub
            >>> heads_config = ub.codeblock(
            >>>     '''
            >>>     - name: class
            >>>       type: MultiLayerPerceptron
            >>>       hidden_channels: 3
            >>>       classes: auto
            >>>       loss:
            >>>           type: dicefocal
            >>>           gamma: 2.0
            >>>       head_weight: 1.0
            >>>     - name: nonlocal_class
            >>>       type: MultiLayerPerceptron
            >>>       hidden_channels: 3
            >>>       classes: auto
            >>>       loss:
            >>>           type: dicefocal
            >>>           gamma: 2.0
            >>>       head_weight: 1.0
            >>>     #
            >>>     # Mirrors the simple FCNHead in torchvision
            >>>     - name: saliency
            >>>       type: mlp
            >>>       hidden_channels: [256]
            >>>       out_channels: 2
            >>>       dropout: 0.1
            >>>       norm: batch
            >>>       loss:
            >>>           type: focal
            >>>           gamma: 2.0
            >>>       head_weight: 1.0
            >>>     ''')
            >>> classes = ['a', 'b', 'c']
            >>> feat_dim = 1024
            >>> heads = TaskHeads(heads_config, feat_dim, classes=classes)
            >>> print(heads)
        """
        super().__init__()

        heads_config = kwutil.util_yaml.Yaml.coerce(heads_config)
        if isinstance(heads_config, dict):
            raise NotImplementedError

        task_configs = heads_config

        for task_config in task_configs:
            task_config = task_config.copy()
            head_name = task_config.pop('name')
            head_type = task_config.pop('type')
            head_weight = task_config.pop('head_weight', 1.0)
            if head_weight > 0:
                if head_type == 'box':
                    from geowatch.tasks.fusion.methods.object_head import DetrDecoderForObjectDetection
                    from transformers import DetrConfig
                    detr_config = DetrConfig(
                        d_model=feat_dim,
                        num_labels=1,
                        dropout=0.0,
                        eos_coefficient=1.0,
                        num_queries=20
                    )
                    heads[head_name] = head = DetrDecoderForObjectDetection(
                        config=detr_config,
                        d_model=feat_dim,
                        d_hidden=feat_dim
                    )
                    head.head_weight = head_weight
                    raise NotImplementedError
                    if 'loss' in task_config:
                        raise ValueError('DETR-HEAD handles its own loss')
                elif head_type == 'nonlocal_clf':
                    raise NotImplementedError
                elif head_type in {'mlp', 'MultiLayerPerceptron'}:
                    loss_config = task_config.pop('loss', {})
                    if 'norm' not in task_config:
                        task_config['norm'] = None
                    if 'classes' in task_config:
                        task_classes = task_config.pop('classes')
                        if task_classes == 'auto':
                            task_classes = classes
                            assert task_classes is not None
                        task_config['out_channels'] = len(task_classes)
                    heads[head_name] = head = MultiLayerPerceptronNd(
                        dim=0,
                        in_channels=feat_dim,
                        **task_config,
                    )
                    head.head_weight = head_weight
                    head.criterion = _coerce_loss(head.out_channels, loss_config)
                elif head_type == 'segmenter':
                    from geowatch.tasks.fusion.architectures import segmenter_decoder
                    heads[head_name] = segmenter_decoder.MaskTransformerDecoder(
                        d_model=feat_dim,
                        n_layers=task_config['hidden_channels'],
                        n_cls=task_config['out_channels'],
                    )
                    head.head_weight = head_weight
                    raise NotImplementedError
                else:
                    raise KeyError(head_type)

    def forward(heads, feats):
        logits = {}
        probs = {}
        with_probs = True
        for head_key, head in heads.items():
            logits[head_key] = head(feats)

            # Convert logits into probabilities for output
            if with_probs:
                # not a huge fan of modifying the key, but doing it to agree
                # with existing code. Might want to refactor later.
                probs_key = head_key + '_probs'
                criterion_encoding = head.criterion.target_encoding
                _logits = logits[head_key].detach()
                if criterion_encoding == "onehot":
                    probs[probs_key] = _logits.sigmoid()
                elif criterion_encoding == "index":
                    probs[probs_key] = _logits.softmax(dim=-1)
                else:
                    raise NotImplementedError

        head_outputs = {}
        head_outputs['logits'] = logits
        if with_probs:
            head_outputs['probs'] = probs
        return head_outputs

    def compute_loss(heads, outputs, batch):
        head_loss_parts = {}
        # Compute criterion loss for each head
        resampled_logits = outputs['logits']
        for head_key, head_logits in resampled_logits.items():
            assert head_key == 'nonlocal_class'
            head_truth = batch['nonlocal_class_ohe']
            truth_encoding = 'ohe'
            head = heads[head_key]
            criterion = head.criterion
            head_pred_input = einops.rearrange(head_logits, '(b t h w) c -> ' + criterion.logit_shape, t=1, h=1, w=1).contiguous()

            if criterion.target_encoding == 'index':
                head_true_idxs = head_truth.long()
                head_true_input = einops.rearrange(head_true_idxs, '(b t h w) -> ' + criterion.target_shape, t=1, h=1, w=1).contiguous()
            elif criterion.target_encoding == 'onehot':
                # Note: 1HE is much easier to work with
                if truth_encoding == 'index':
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
            head_loss_parts[head_key] = head_loss

        total_loss = sum(t.sum() for t in head_loss_parts.values())

        losses = {}
        losses['head_loss_parts'] = head_loss_parts
        losses['loss'] = total_loss
        return losses


def _coerce_loss(out_channels, loss_config):
    from geowatch.tasks.fusion.methods.loss import coerce_criterion
    # hack
    loss_config['type']
    loss_config['loss_code'] = loss_config.pop('type')
    if 'gamma' in loss_config:
        loss_config['focal_gamma'] = loss_config.pop('gamma')
    if 'weights' not in loss_config:
        # TODO: determine a good way for the heads to be
        # notified about changes in class / saliency weighting.
        loss_config['weights'] = torch.ones(out_channels)
    criterion = coerce_criterion(**loss_config)
    return criterion


# class MLPHead(MultiLayerPerceptronNd):
#     """
#     Unfinished

#     Example:
#         >>> from geowatch.tasks.fusion.methods.heads import *  # NOQA
#         >>> head = MLPHead(0, 3, [], 5)
#     """
#     def __init__(self, dim, in_channels, hidden_channels, out_channels,
#                  bias=True, dropout=None, norm=None, noli='relu',
#                  residual=False, loss=None, **kwargs):

#         super().__init__(dim=dim, in_channels=in_channels,
#                          hidden_channels=hidden_channels,
#                          out_channels=out_channels, bias=bias,
#                          dropout=dropout, noli=noli, residual=residual,
#                          **kwargs)
#         if loss is None:
#             loss = {
#                 'type': 'dicefocal',
#                 'gamma': 2.0,
#             }
#         self.criterion = _coerce_loss(self.out_channels, loss)

#     @classmethod
#     def coerce(cls, config):
#         out_channels = config.get('out_channels', None)
#         if out_channels is None:
#             _config_classes = config.get('classes', None)
#             if _config_classes is not None:
#                 if _config_classes == 'auto':
#                     raise NotImplementedError
#                 else:
#                     out_channels = len(_config_classes)

#         ...

#     def forward(self, inputs):
#         ...
