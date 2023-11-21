import torch.nn as nn

from geowatch.tasks.rutgers_material_change_detection.models.apnb import AFNB


class Attention(nn.Module):
    def __init__(self, framework_name, feat_sizes, active_layers, n_heads):
        super(Attention, self).__init__()
        self.n_heads = n_heads
        self.feat_sizes = feat_sizes
        self.active_layers = active_layers
        self.framework_name = framework_name

        self.build()

    def build(self):
        raise NotImplementedError("Build method for attention not implemented.")

    def forward(self):
        pass


class SelfAttention(Attention):
    def __init__(self, framework_name, feat_sizes, active_layers, n_heads):
        super(SelfAttention, self).__init__(framework_name, feat_sizes, active_layers, n_heads)

    def build(self):
        # Generate Attention per frame.
        self.attention_layers = nn.ModuleDict()
        for i, feat_size in enumerate(self.feat_sizes):
            self.attention_layers["layer" + str(i + 1)] = nn.TransformerEncoderLayer(
                feat_size[0], self.n_heads, dim_feedforward=feat_size[0] * 2
            )

    def forward(self, feats):
        refined_feats = {}
        for layer_name, feat in feats.items():
            batch_size, feat_size, height, width = feat.shape
            feat = feat.flatten(2)  # [batch_size, feat_size, seq]
            feat = feat.permute(2, 0, 1)  # [seq, batch_size, feat_size]
            feat = self.attention_layers[layer_name](feat)  # [seq, batch_size, feat_size]
            feat = feat.permute(1, 2, 0)  # [batch_size, feat_size, seq]
            feat = feat.reshape(batch_size, feat_size, height, width)
            refined_feats[layer_name] = feat
        return refined_feats


class AsymmetricPyramidSelfAttention(Attention):
    def __init__(self, framework_name, feat_sizes, active_layers, n_heads):
        super(AsymmetricPyramidSelfAttention, self).__init__(framework_name, feat_sizes, active_layers, n_heads)

    def build(self):
        self.attention_layers = nn.ModuleDict()
        for i, feat_size in enumerate(self.feat_sizes):
            if i == 0:
                self.attention_layers["layer" + str(i + 1)] = nn.TransformerEncoderLayer(
                    feat_size[0], self.n_heads, dim_feedforward=feat_size[0] * 2
                )
            else:
                prev_channels = self.feat_sizes[i - 1][0]
                curr_channels = self.feat_sizes[i][0]
                self.attention_layers["layer" + str(i + 1)] = AFNB(
                    low_in_channels=prev_channels,
                    high_in_channels=curr_channels,
                    out_channels=curr_channels,
                    key_channels=prev_channels,
                    value_channels=prev_channels,
                    dropout=0.0,
                    norm_type="batchnorm",
                )

    def forward(self, feats):
        refined_feats = {}
        for i, (layer_name, feat) in enumerate(feats.items()):
            if i == 0:
                batch_size, feat_size, height, width = feat.shape
                feat = feat.flatten(2)  # [batch_size, feat_size, seq]
                feat = feat.permute(2, 0, 1)  # [seq, batch_size, feat_size]
                feat = self.attention_layers[layer_name](feat)  # [seq, batch_size, feat_size]
                feat = feat.permute(1, 2, 0)  # [batch_size, feat_size, seq]
                feat = feat.reshape(batch_size, feat_size, height, width)
                refined_feats[layer_name] = feat
            else:
                prev_feat = feats["layer" + str(i)]
                curr_feat = feat
                refined_feats[layer_name] = self.attention_layers[layer_name](prev_feat, curr_feat)
        return refined_feats
