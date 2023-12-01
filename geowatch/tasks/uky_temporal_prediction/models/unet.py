# Code for different variants of U-Net
# Some parts taken from https://github.com/milesial/Pytorch-UNet
# Implements light (half feature channels) and lighter (quarter number of
# feature maps) U-Net


import torch
import torch.nn as nn
import torch.nn.functional as F


def count_trainable_parameters(model):  # to count trainable parameters
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.up = nn.Upsample(
            scale_factor=2,
            mode='bilinear',
            align_corners=True)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        #x1 = self.up(x1)
        x1 = nn.functional.interpolate(x1, scale_factor=2, mode='nearest')

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            (diffX // 2,
             diffX - diffX // 2,
             diffY // 2,
             diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x


# U-Net
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.inc = inconv(in_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, out_channels)

    def forward(self, x):
        # x1 = self.inc(x.permute(0, 3, 1, 2))
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class UNetEncoder(nn.Module):
    def __init__(self, in_channels):
        super(UNetEncoder, self).__init__()

        self.inc = inconv(in_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)

    def forward(self, x):
        # x1 = self.inc(x.permute(0, 3, 1, 2))
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5


class UNetDecoder(nn.Module):
    def __init__(self, out_channels):
        super(UNetDecoder, self).__init__()

        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, out_channels)

    def forward(self, X):
        x1, x2, x3, x4, x5 = X
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class UNetHalfsizeEncoder(nn.Module):
    def __init__(self, in_channels):
        super(UNetEncoder, self).__init__()

        self.inc = inconv(in_channels, 32)
        self.down1 = down(32, 64)  # down(64, 128)
        self.down2 = down(64, 128)  # down(128, 256)
        self.down3 = down(128, 256)  # down(256, 512)
        self.down4 = down(256, 256)  # down(512, 512)

    def forward(self, x):
        # x1 = self.inc(x.permute(0, 3, 1, 2))
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5


class UNetHalfsizeDecoder(nn.Module):
    def __init__(self, out_channels):
        super(UNetDecoder, self).__init__()

        self.up1 = up(512, 128)  # up(1024, 256)
        self.up2 = up(256, 64)  # up(512, 128)
        self.up3 = up(128, 32)  # up(256, 64)
        self.up4 = up(64, 32)  # up(128, 64)
        self.outc = outconv(32, out_channels)

    def forward(self, X):
        x1, x2, x3, x4, x5 = X
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
