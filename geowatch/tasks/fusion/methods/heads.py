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


class TaskHeads(torch.nn.ModuleDict):
    """
    Sends features to task specific heads.
    """

    def __init__(self, heads_config):
        """
        Args:
            heads_config (str | Dict): yaml coercable config

        Example:
            >>> from geowatch.tasks.fusion.methods.heads import *  # NOQA
            >>> heads_config = ub.codeblock(
            >>>     '''
            >>>     feat_dim: 1024
            >>>     tasks:
            >>>         - name: class
            >>>           type: mlp
            >>>           hidden_channels: 3
            >>>           out_channels: 4
            >>>           loss:
            >>>               type: dicefocal
            >>>               gamma: 2.0
            >>>           global_weight: 1.0
            >>>         #
            >>>         # Mirrors the simple FCNHead in torchvision
            >>>         - name: saliency
            >>>           type: mlp
            >>>           hidden_channels: [256]
            >>>           out_channels: 2
            >>>           dropout: 0.1
            >>>           norm: batch
            >>>           loss:
            >>>               type: focal
            >>>               gamma: 2.0
            >>>           global_weight: 1.0
            >>>     ''')
            >>> heads = TaskHeads(heads_config)
            >>> print(heads)
        """
        import kwutil
        heads_config = kwutil.util_yaml.Yaml.coerce(heads_config)
        super().__init__()
        feat_dim = heads_config['feat_dim']
        task_configs = heads_config['tasks']

        for task_config in task_configs:
            task_name = task_config.pop('name')
            task_type = task_config.pop('type')
            global_weight = task_config.pop('global_weight', 1.0)
            if global_weight > 0:
                if task_type == 'box':
                    from geowatch.tasks.fusion.methods.object_head import DetrDecoderForObjectDetection
                    from transformers import DetrConfig
                    detr_config = DetrConfig(
                        d_model=feat_dim,
                        num_labels=1,
                        dropout=0.0,
                        eos_coefficient=1.0,
                        num_queries=20
                    )
                    self[task_name] = DetrDecoderForObjectDetection(
                        config=detr_config,
                        d_model=feat_dim,
                        d_hidden=feat_dim
                    )
                    if 'loss' in task_config:
                        raise ValueError('DETR-HEAD handles its own loss')
                elif task_type == 'mlp':
                    from geowatch.utils.util_netharn import MultiLayerPerceptronNd
                    loss_config = task_config.pop('loss', {})
                    if 'norm' not in task_config:
                        task_config['norm'] = None
                    self[task_name] = head = MultiLayerPerceptronNd(
                        dim=0,
                        in_channels=feat_dim,
                        **task_config,
                    )
                    head.criterion = _coerce_loss(head.out_channels, loss_config)
                elif task_type == 'segmenter':
                    from geowatch.tasks.fusion.architectures import segmenter_decoder
                    self[task_name] = segmenter_decoder.MaskTransformerDecoder(
                        d_model=feat_dim,
                        n_layers=task_config['hidden_channels'],
                        n_cls=task_config['out_channels'],
                    )
                else:
                    raise KeyError(task_type)


def _coerce_loss(out_channels, loss_config):
    from geowatch.tasks.fusion.methods.loss import coerce_criterion
    # hack
    loss_config['loss_code'] = loss_config.pop('type')
    if 'gamma' in loss_config:
        loss_config['focal_gamma'] = loss_config.pop('gamma')
    if 'weights' not in loss_config:
        # TODO: determine a good way for the heads to be
        # notified about changes in class / saliency weighting.
        loss_config['weights'] = torch.ones(out_channels)
    criterion = coerce_criterion(**loss_config)
    return criterion
