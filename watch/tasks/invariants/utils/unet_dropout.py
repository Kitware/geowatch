# Code for different variants of U-Net
# Some parts taken from https://github.com/milesial/Pytorch-UNet
# Implements light (half feature channels) and lighter (quarter number of feature maps) U-Net


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.parallel


class BlurPool(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=4, stride=2, pad_off=0):
        super(BlurPool, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        if (self.filt_size == 1):
            a = np.array([1., ])
        elif (self.filt_size == 2):
            a = np.array([1., 1.])
        elif (self.filt_size == 3):
            a = np.array([1., 2., 1.])
        elif (self.filt_size == 4):
            a = np.array([1., 3., 3., 1.])
        elif (self.filt_size == 5):
            a = np.array([1., 4., 6., 4., 1.])
        elif (self.filt_size == 6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif (self.filt_size == 7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if (self.filt_size == 1):
            if (self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


def get_pad_layer(pad_type):
    if (pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif (pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif (pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


class BlurPool1D(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(BlurPool1D, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        # print('Filter size [%i]' % filt_size)
        if (self.filt_size == 1):
            a = np.array([1., ])
        elif (self.filt_size == 2):
            a = np.array([1., 1.])
        elif (self.filt_size == 3):
            a = np.array([1., 2., 1.])
        elif (self.filt_size == 4):
            a = np.array([1., 3., 3., 1.])
        elif (self.filt_size == 5):
            a = np.array([1., 4., 6., 4., 1.])
        elif (self.filt_size == 6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif (self.filt_size == 7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a)
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :].repeat((self.channels, 1, 1)))

        self.pad = get_pad_layer_1d(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if (self.filt_size == 1):
            if (self.pad_off == 0):
                return inp[:, :, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride]
        else:
            return F.conv1d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


def get_pad_layer_1d(pad_type):
    if (pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad1d
    elif (pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad1d
    elif (pad_type == 'zero'):
        PadLayer = nn.ZeroPad1d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


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


class down_blur(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_blur, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2, stride=1),
            BlurPool(in_ch, filt_size=4, stride=2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        #x1 = self.up(x1)
        x1 = nn.functional.interpolate(x1, scale_factor=2, mode='nearest')

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

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


## U-Net with dropout. Dropout applied at innermost 3 up and down layers as suggested in https://arxiv.org/pdf/1806.05034.pdf. The argument dropout_at_eval allows us to use dropout layers in evaluation mode.
class UNet_Dropout(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=.5, dropout_at_eval=True):
        super(UNet_Dropout, self).__init__()

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
        self.drop = nn.Dropout(p=dropout_rate)
        self.dropout_at_eval = dropout_at_eval

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.drop(self.down2(x2))
        x4 = self.drop(self.down3(x3))
        x5 = self.drop(self.down4(x4))

        x = self.drop(self.up1(x5, x4))
        x = self.drop(self.up2(x, x3))
        x = self.drop(self.up3(x, x2))
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

    def eval(self, ):
        if not self.dropout_at_eval:
            return self.train(False)
        else:
            self.train(False)
            for m in self.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()


## U-Net Blur with dropout. Dropout applied at innermost 3 up and down layers as suggested in https://arxiv.org/pdf/1806.05034.pdf. The argument dropout_at_eval allows us to use dropout layers in evaluation mode.
class UNet_Blur_Dropout(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=.5, dropout_at_eval=True):
        super(UNet_Blur_Dropout, self).__init__()

        self.encoder = UNet_Blur_Dropout_Encoder(in_channels, dropout_rate)

        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, out_channels)
        self.drop = nn.Dropout(p=dropout_rate)
        self.dropout_at_eval = dropout_at_eval

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)

        x = self.drop(self.up1(x5, x4))
        x = self.drop(self.up2(x, x3))
        x = self.drop(self.up3(x, x2))
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

    def eval(self, ):
        if not self.dropout_at_eval:
            return self.train(False)
        else:
            self.train(False)
            for m in self.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()


class UNet_Blur_Dropout_Encoder(nn.Module):
    def __init__(self, in_channels, dropout_rate=0):
        super(UNet_Blur_Dropout_Encoder, self).__init__()
        self.inc = inconv(in_channels, 64)
        self.down1 = down_blur(64, 128)
        self.down2 = down_blur(128, 256)
        self.down3 = down_blur(256, 512)
        self.down4 = down_blur(512, 512)

        self.drop = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.drop(self.down2(x2))
        x4 = self.drop(self.down3(x3))
        x5 = self.drop(self.down4(x4))
        return x1, x2, x3, x4, x5
