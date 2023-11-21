import torch.nn as nn
import torch
from torch.nn import functional as F


class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)

    def forward(self, x):
        # return super(Conv2d, self).forward(x)
        weight = self.weight
        weight_mean = weight.mean(dim=(1, 2, 3), keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of
        # channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)
        # Given transposed=1, weight of size [48, 48, 2, 2], 48 -> 32+64//2, instead,
        # expected input[4, 64, 128, 128] to have 48 channels, but got 64
        # channels instead

    def forward(self, x1, x2):
        # print(x1.shape)
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # self.conv = nn.Sequential(
        #                             nn.Conv2d(in_channels,out_channels,kernel_size=1),
        #                             nn.Sigmoid()
        #                             )

    def forward(self, x):
        return self.conv(x)


class ShallowSeg(nn.Module):
    def __init__(self, num_channels=3, num_classes=3,
                 bilinear=True, pretrained=False,
                 beta=False, weight_std=False,
                 num_groups=32, out_dim=128, feats=[64, 64, 128, 256, 512]):
        super(ShallowSeg, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.out_dim = out_dim
        # feats = [64, 64, 128, 256, 512]
        # feats = [32, 32, 64, 64, 128]

        self.inc = DoubleConv(num_channels, feats[0])
        self.down1 = Down(feats[0], feats[1])
        self.down2 = Down(feats[1], feats[2])
        self.down3 = Down(feats[2], feats[3])
        self.down4 = Down(feats[3], feats[4])

        self.up1 = Up(feats[4] + feats[3], feats[3], bilinear)
        self.up2 = Up(feats[2] + feats[3], feats[2], bilinear)
        self.up3 = Up(feats[1] + feats[2], feats[1], bilinear)
        self.up4 = Up(feats[0] + feats[1], feats[0], bilinear)

        self.outc = OutConv(feats[0], num_classes)
        self.features_outc = OutConv(feats[0], out_dim)

    def forward(self, x):
        # b, c, h, w = x.shape
        # print(x.shape)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # features = self.features_outc(x)
        # print(dictionary.shape)
        logits = self.outc(x)
        return logits, x5
