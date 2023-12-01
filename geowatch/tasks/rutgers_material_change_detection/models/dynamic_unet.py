import torch
import torch.nn as nn
import torch.nn.functional as F

from geowatch.tasks.rutgers_material_change_detection.models.base_model import BaseDecoder


class DynamicUNet(BaseDecoder):
    def __init__(self, feat_sizes, in_channels, out_channels, base_feat_channels=64, name="unet"):
        super(DynamicUNet, self).__init__(feat_sizes, out_channels, name)
        self.bfc = base_feat_channels
        self.in_channels = in_channels

        assert type(feat_sizes) is list
        assert len(feat_sizes) == 4

        self.build()

    def build(self):
        # Build input encoder layer.
        self.in_encode_layer = DoubleConv(self.in_channels, self.bfc)

        # Build upsampling layers.

        # last of encoder feats and lowest travel feats
        #  512 + 256 --> 512
        self.up1 = Up(self.feat_sizes[3][0] + self.feat_sizes[2][0], self.feat_sizes[3][0])

        # 1st upsampled feats and 2nd travel feats
        # 256 + 128 --> 256
        self.up2 = Up(self.feat_sizes[3][0] + self.feat_sizes[1][0], self.feat_sizes[2][0])

        # 128 + 64 --> 128
        self.up3 = Up(self.feat_sizes[2][0] + self.feat_sizes[0][0], self.feat_sizes[1][0])

        # 64 + bfc --> 64
        self.up4 = Up(self.feat_sizes[1][0] + self.bfc, self.bfc)

        # Build output convolution layer.
        # bfc --> n_classes
        self.out_conv_layer = OutConv(self.bfc, self.out_channels)

    def forward(self, feats, input):
        """Foward method of Dynamic UNet model.

        Args:
            feats (dict(torch.tensor)): A dict containing keys of form "layerX" and values of tensors of shape [batch_size, channels, height, width].

        Returns:
            (torch.tensor): A tensor of shape [batch_size, out_channels, height, width]
        """
        l0_feats = self.in_encode_layer(input)
        l1_feats = feats["layer1"]
        l2_feats = feats["layer2"]
        l3_feats = feats["layer3"]
        l4_feats = feats["layer4"]

        x = self.up1(l4_feats, l3_feats)
        x = self.up2(x, l2_feats)
        x = self.up3(x, l1_feats)
        x = self.up4(x, l0_feats)

        output = self.out_conv_layer(x)

        return output


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


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


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
