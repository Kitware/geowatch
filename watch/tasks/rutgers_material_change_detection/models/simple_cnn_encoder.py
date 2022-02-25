import torch.nn as nn


class SimpleCNNEncoder(nn.Module):
    def __init__(self, in_channels, bilinear=True):
        super(SimpleCNNEncoder, self).__init__()
        self.bilinear = bilinear
        self.in_channels = in_channels

        self.build()

    def build(self):
        self.inc = DoubleConv(self.in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)

    def forward(self, input):
        output = {}

        enc_input = self.inc(input)
        output["layer1"] = self.down1(enc_input)
        output["layer2"] = self.down2(output["layer1"])
        output["layer3"] = self.down3(output["layer2"])
        output["layer4"] = self.down4(output["layer3"])

        return output


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)
