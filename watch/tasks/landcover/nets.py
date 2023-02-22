import torch
import torch.nn as nn
from torchvision import models


def double_conv(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                         nn.ReLU(inplace=True),
                         nn.Conv2d(out_channels, out_channels, 3, padding=1),
                         nn.ReLU(inplace=True))


class UNetR(nn.Module):

    def __init__(self, num_outputs, num_channels=3):
        super().__init__()

        self.in_channels = num_channels
        self.out_channels = num_outputs

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        if num_channels == 3:
            self.firstconv = resnet.conv1
        else:
            self.firstconv = nn.Conv2d(num_channels,
                                       64,
                                       kernel_size=(7, 7),
                                       stride=(2, 2),
                                       padding=(3, 3))

        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.upsample = nn.Upsample(scale_factor=2,
                                    mode='bilinear',
                                    align_corners=True)

        self.decoder4 = double_conv(256 + 512, 256)
        self.decoder3 = double_conv(128 + 256, 128)
        self.decoder2 = double_conv(64 + 128, 64)
        self.decoder1 = double_conv(64 + 64, 64)

        self.conv_last = nn.Conv2d(64, num_outputs, 3, padding=1)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        e0 = self.firstrelu(x)

        e1 = self.encoder1(self.firstmaxpool(e0))
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        x = self.upsample(e4)
        d3 = self.decoder4(torch.cat([x, e3], dim=1))

        x = self.upsample(d3)
        d2 = self.decoder3(torch.cat([x, e2], dim=1))

        x = self.upsample(d2)
        d1 = self.decoder2(torch.cat([x, e1], dim=1))

        x = self.upsample(d1)
        d0 = self.decoder1(torch.cat([x, e0], dim=1))

        x = self.upsample(d0)
        out = self.conv_last(x)

        return out
