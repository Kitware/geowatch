import torch
import torch.nn as nn
from torch.nn import functional as F


class AFNB(nn.Module):
    """
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """

    def __init__(
        self,
        low_in_channels,
        high_in_channels,
        out_channels,
        key_channels,
        value_channels,
        dropout,
        sizes=([1]),
        norm_type=None,
        psp_size=(1, 3, 6, 8),
    ):
        super(AFNB, self).__init__()
        self.stages = []
        self.norm_type = norm_type
        self.psp_size = psp_size
        self.stages = nn.ModuleList(
            [
                self._make_stage([low_in_channels, high_in_channels], out_channels, key_channels, value_channels, size)
                for size in sizes
            ]
        )
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(out_channels + high_in_channels, out_channels, kernel_size=1, padding=0),
            ModuleHelper.BatchNorm2d(norm_type=self.norm_type)(out_channels),
            nn.Dropout2d(dropout),
        )

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return SelfAttentionBlock2D(
            in_channels[0],
            in_channels[1],
            key_channels,
            value_channels,
            output_channels,
            size,
            self.norm_type,
            psp_size=self.psp_size,
        )

    def forward(self, low_feats, high_feats):
        priors = [stage(low_feats, high_feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(torch.cat([context, high_feats], 1))
        return output


class _SelfAttentionBlock(nn.Module):
    """
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    """

    def __init__(
        self,
        low_in_channels,
        high_in_channels,
        key_channels,
        value_channels,
        out_channels=None,
        scale=1,
        norm_type=None,
        psp_size=(1, 3, 6, 8),
    ):
        super(_SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = low_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels is None:
            self.out_channels = high_in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, norm_type=norm_type),
        )
        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels=high_in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, norm_type=norm_type),
        )
        self.f_value = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.value_channels, kernel_size=1, stride=1, padding=0
        )
        self.W = nn.Conv2d(
            in_channels=self.value_channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0
        )

        self.psp = PSPModule(psp_size)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, low_feats, high_feats):
        batch_size = high_feats.size(0)
        # if self.scale > 1:
        #     x = self.pool(x)

        value = self.psp(self.f_value(low_feats))

        query = self.f_query(high_feats).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(low_feats)
        # value=self.psp(value)#.view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        key = self.psp(key)  # .view(batch_size, self.key_channels, -1)
        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *high_feats.size()[2:])
        context = self.W(context)
        return context


class SelfAttentionBlock2D(_SelfAttentionBlock):
    def __init__(
        self,
        low_in_channels,
        high_in_channels,
        key_channels,
        value_channels,
        out_channels=None,
        scale=1,
        norm_type=None,
        psp_size=(1, 3, 6, 8),
    ):
        super(SelfAttentionBlock2D, self).__init__(
            low_in_channels,
            high_in_channels,
            key_channels,
            value_channels,
            out_channels,
            scale,
            norm_type,
            psp_size=psp_size,
        )


class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, sizes=(1, 3, 6, 8), dimension=2):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center


class ModuleHelper(object):
    @staticmethod
    def BNReLU(num_features, norm_type=None, **kwargs):
        if norm_type == "batchnorm":
            return nn.Sequential(nn.BatchNorm2d(num_features, **kwargs), nn.ReLU())
        elif norm_type == "instancenorm":
            return nn.Sequential(nn.InstanceNorm2d(num_features, **kwargs), nn.ReLU())
        else:
            exit(1)

    @staticmethod
    def BatchNorm3d(norm_type=None, ret_cls=False):
        if norm_type == "batchnorm":
            return nn.BatchNorm3d
        elif norm_type == "instancenorm":
            return nn.InstanceNorm3d

        else:
            exit(1)

    @staticmethod
    def BatchNorm2d(norm_type=None, ret_cls=False):
        if norm_type == "batchnorm":
            return nn.BatchNorm2d

        elif norm_type == "instancenorm":
            return nn.InstanceNorm2d
        else:
            exit(1)

    @staticmethod
    def BatchNorm1d(norm_type=None, ret_cls=False):
        if norm_type == "batchnorm":
            return nn.BatchNorm1d
        elif norm_type == "instancenorm":
            return nn.InstanceNorm1d
        else:
            exit(1)

    @staticmethod
    def constant_init(module, val, bias=0):
        nn.init.constant_(module.weight, val)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def xavier_init(module, gain=1, bias=0, distribution="normal"):
        assert distribution in ["uniform", "normal"]
        if distribution == "uniform":
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def normal_init(module, mean=0, std=1, bias=0):
        nn.init.normal_(module.weight, mean, std)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def uniform_init(module, a=0, b=1, bias=0):
        nn.init.uniform_(module.weight, a, b)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def kaiming_init(module, mode="fan_in", nonlinearity="leaky_relu", bias=0, distribution="normal"):
        assert distribution in ["uniform", "normal"]
        if distribution == "uniform":
            nn.init.kaiming_uniform_(module.weight, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity=nonlinearity)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)
