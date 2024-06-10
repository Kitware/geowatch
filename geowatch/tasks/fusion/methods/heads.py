"""
# TODO:

* Generalize Task Heads, and hook up to off-the-shelf backbones. We almost
  never will be able to use those models as-is.

* Port _build_item_loss_parts
* Port forward_item
* Port forward_step
* Port forward_foot
"""


class TaskHeads:

    def demo(cls):
        import torch
        self = cls()
        self.criterions = torch.nn.ModuleDict()
        self.heads = torch.nn.ModuleDict()
        import ubelt as ub
        import kwutil
        kwutil.util_yaml.Yaml.coerce(ub.codeblock(
            '''
            feat_dim: 2048
            heads:
                - name: class
                  hidden: 3
                  channels: 4
                  loss:
                      type: dicefocal
                      gamma: 2.0
                  weights: null

                - name: saliency
                  hidden: 3
                  channels: 2
                  loss:
                      type: focal
                      gamma: 2.0
                  weights: null
            '''))
        feat_dim = 2048  # FIXME

        head_properties = [
            {
                'name': 'change',
                'hidden': self.change_head_hidden,
                'channels': 2,
                'loss': self.hparams.change_loss,
                'weights': self.change_weights,
            },
            {
                'name': 'saliency',
                'hidden': self.saliency_head_hidden,
                'channels': self.saliency_num_classes,
                'loss': self.hparams.saliency_loss,
                'weights': self.saliency_weights,
            },
            {
                'name': 'class',
                'hidden': self.class_head_hidden,
                'channels': self.num_classes,
                'loss': self.hparams.class_loss,
                'weights': self.class_weights,
            },
            {
                'name': 'box',
                'weights': self.global_head_weights['box'],
            },
        ]

        for prop in head_properties:
            head_name = prop['name']
            global_weight = self.global_head_weights[head_name]
            if global_weight > 0:
                if head_name != 'box':
                    from geowatch.tasks.fusion.methods.loss import coerce_criterion
                    self.criterions[head_name] = coerce_criterion(prop['loss'],
                                                                  prop['weights'],
                                                                  # FIXME
                                                                  # ohem_ratio=_config.ohem_ratio,
                                                                  # focal_gamma=_config.focal_gamma
                                                                  )
                if head_name == 'box':
                    from geowatch.tasks.fusion.methods.object_head import DetrDecoderForObjectDetection
                    from transformers import DetrConfig

                    self.heads[head_name] = DetrDecoderForObjectDetection(
                        config=DetrConfig(d_model=feat_dim, num_labels=1, dropout=0.0, eos_coefficient=1.0, num_queries=20),
                        d_model=feat_dim,
                        d_hidden=feat_dim
                    ).to(self.device)

                elif self.hparams.decoder == 'mlp':
                    from geowatch.utils.util_netharn import MultiLayerPerceptronNd
                    self.heads[head_name] = MultiLayerPerceptronNd(
                        dim=0,
                        in_channels=feat_dim,
                        hidden_channels=prop['hidden'],
                        out_channels=prop['channels'],
                        norm=None
                    )
                elif self.hparams.decoder == 'segmenter':
                    from geowatch.tasks.fusion.architectures import segmenter_decoder
                    self.heads[head_name] = segmenter_decoder.MaskTransformerDecoder(
                        d_model=feat_dim,
                        n_layers=prop['hidden'],
                        n_cls=prop['channels'],
                    )
                else:
                    raise KeyError(self.hparams.decoder)

    ...
