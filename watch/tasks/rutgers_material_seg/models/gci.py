import torch
import torch.nn as nn
import torch.nn.functional as F


class Pre_Norm_Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Pre_Norm_Conv2d, self).__init__(in_channels, out_channels,
                                              kernel_size, stride, padding,
                                              dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=(1, 2, 3), keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class GCI(nn.Module):
    """Global Cue Injection
    Takes shallow features with low receptive
    field and augments it with global info via
    adaptive instance normalisation"""

    def __init__(self, NormLayer=nn.BatchNorm2d):
        super(GCI, self).__init__()

        self.NormLayer = NormLayer
        self.from_scratch_layers = []

        self._init_params()

    def _conv2d(self, *args, **kwargs):
        conv = nn.Conv2d(*args, **kwargs)
        self.from_scratch_layers.append(conv)
        torch.nn.init.kaiming_normal_(conv.weight)
        return conv

    def _bnorm(self, *args, **kwargs):
        bn = self.NormLayer(*args, **kwargs)
        # self.bn_learn.append(bn)
        self.from_scratch_layers.append(bn)
        if bn.weight is not None:
            bn.weight.data.fill_(1)
            bn.bias.data.zero_()
        return bn

    def _init_params(self):

        self.fc_deep = nn.Sequential(Pre_Norm_Conv2d(256, 512, 1, bias=False),
                                     #  Pre_Norm_Conv2d(64, 512, 1, bias=False),
                                     #  self._bnorm(512),
                                     #  nn.BatchNorm2d(512, track_running_stats = False),
                                     nn.GroupNorm(32, 512),
                                     nn.ReLU())

        self.fc_skip = nn.Sequential(Pre_Norm_Conv2d(256, 256, 1, bias=False),
                                     #  Pre_Norm_Conv2d(64, 256, 1, bias=False),
                                     #  nn.BatchNorm2d(256, track_running_stats = False, affine=False),
                                     #  self._bnorm(256, affine=False)
                                     nn.GroupNorm(32, 256))

        self.fc_cls = nn.Sequential(Pre_Norm_Conv2d(256, 256, 1, bias=False),
                                    # Pre_Norm_Conv2d(256, 64, 1, bias=False),
                                    # nn.BatchNorm2d(256, track_running_stats = False),
                                    nn.GroupNorm(32, 256),
                                    # nn.GroupNorm(32,64),
                                    # self._bnorm(64),
                                    nn.ReLU())

    def forward(self, x, y):
        """Forward pass

        Args:
            x: shalow features
            y: deep features
        """

        # extract global attributes
        y = self.fc_deep(y)
        attrs, _ = y.view(y.size(0), y.size(1), -1).max(-1)

        # pre-process shallow features
        x = self.fc_skip(x)
        x = F.relu(self._adin_conv(x, attrs))

        return self.fc_cls(x)

    def _adin_conv(self, x, y):

        bs, num_c, _, _ = x.size()
        assert 2 * num_c == y.size(1), "AdIN: dimension mismatch"

        y = y.view(bs, 2, num_c)
        gamma, beta = y[:, 0], y[:, 1]

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        return x * (gamma + 1) + beta
