# flake8: noqa

import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from geowatch.tasks.rutgers_material_seg.models.sg import StochasticGate
from geowatch.tasks.rutgers_material_seg.models.gci import GCI

__all__ = ['ResNet', 'resnet50', 'resnet101', 'resnet152']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def rescale_as(x, y, mode="bilinear", align_corners=True):
    h, w = y.size()[2:]
    x = F.interpolate(x, size=[h, w], mode=mode, align_corners=align_corners)
    return x


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


class ASPP(nn.Module):

    def __init__(self, C, depth, num_classes, conv=nn.Conv2d,
                 norm=nn.BatchNorm2d, momentum=0.0003, mult=1):
        super(ASPP, self).__init__()
        self._C = C
        self._depth = depth
        self._num_classes = num_classes

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.aspp1 = conv(C, depth, kernel_size=1, stride=1, bias=False)
        self.aspp2 = conv(C, depth, kernel_size=3, stride=1,
                          dilation=int(6 * mult), padding=int(6 * mult),
                          bias=False)
        self.aspp3 = conv(C, depth, kernel_size=3, stride=1,
                          dilation=int(12 * mult), padding=int(12 * mult),
                          bias=False)
        self.aspp4 = conv(C, depth, kernel_size=3, stride=1,
                          dilation=int(18 * mult), padding=int(18 * mult),
                          bias=False)
        self.aspp5 = conv(C, depth, kernel_size=1, stride=1, bias=False)
        self.aspp1_bn = norm(depth, momentum)
        self.aspp2_bn = norm(depth, momentum)
        self.aspp3_bn = norm(depth, momentum)
        self.aspp4_bn = norm(depth, momentum)
        self.aspp5_bn = norm(depth, momentum)
        self.conv2 = conv(depth * 5, depth, kernel_size=1, stride=1,
                          bias=False)
        self.bn2 = norm(depth, momentum)
        self.conv3 = nn.Conv2d(depth, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.aspp1(x)
        x1 = self.aspp1_bn(x1)
        x1 = self.relu(x1)
        x2 = self.aspp2(x)
        x2 = self.aspp2_bn(x2)
        x2 = self.relu(x2)
        x3 = self.aspp3(x)
        x3 = self.aspp3_bn(x3)
        x3 = self.relu(x3)
        x4 = self.aspp4(x)
        x4 = self.aspp4_bn(x4)
        x4 = self.relu(x4)
        x5 = self.global_pooling(x)
        x5 = self.aspp5(x5)
        x5 = self.aspp5_bn(x5)
        x5 = self.relu(x5)
        x5 = nn.Upsample((x.shape[2], x.shape[3]), mode='bilinear',
                         align_corners=True)(x5)
        x = torch.cat((x1, x2, x3, x4, x5), 1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, conv=None, norm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm(planes)
        self.conv2 = conv(planes, planes, kernel_size=3, stride=stride,
                          dilation=dilation, padding=dilation, bias=False)
        self.bn2 = norm(planes)
        self.conv3 = conv(planes, planes * self.expansion, kernel_size=1,
                          bias=False)
        self.bn3 = norm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, num_groups=None,
                 weight_std=True, beta=False, num_channels=3 , feats=None, out_dim=None):
        self.inplanes = 64
        feats = [256, 512, 1024, 2048, 256]

        def _norm(planes, momentum=0.05):
            if num_groups is None:
                return nn.BatchNorm2d(planes, momentum=momentum)
            else:
                return nn.GroupNorm(num_groups, planes)
        self.norm = _norm
        self.conv = Conv2d if weight_std else nn.Conv2d

        super(ResNet, self).__init__()
        if not beta:
            self.conv1_ = self.conv(num_channels, 64, kernel_size=3, stride=1,
                                   padding=1, bias=False)
        else:
            self.conv1_ = nn.Sequential(
                self.conv(num_channels, 64, 3, stride=1, padding=1, bias=False),
                # self.conv(64, 64, 3, stride=1, padding=1, bias=False),
                # self.conv(64, 64, 3, stride=1, padding=1, bias=False)
                )
        self.bn1 = self.norm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(block, feats[0] // block.expansion, layers[0])
        self.layer2 = self._make_layer(block, feats[1] // block.expansion, layers[1], stride=2)
        self.layer3 = self._make_layer(block, feats[2] // block.expansion, layers[2], stride=2)
        self.layer4 = self._make_layer(block, feats[3] // block.expansion, layers[3], stride=2, dilation=2)
        # self.aspp = ASPP(512 * block.expansion, 256, num_classes, conv=self.conv, norm=self.norm)
        self.aspp = ASPP(feats[3], 256, 256, conv=self.conv, norm=self.norm)

        for m in self.modules():
            if isinstance(m, self.conv):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # self.shallow_mask = GCI()
        # self.from_scratch_layers += self.shallow_mask.from_scratch_layers

        # Stochastic Gate
        # self.sg = StochasticGate()
        self.up1 = Up(feats[4] + feats[3], feats[3], bilinear=True)
        self.up2 = Up(feats[2] + feats[3], feats[2], bilinear=True)
        self.up3 = Up(feats[1] + feats[2], feats[1], bilinear=True)
        # self.up4 = Up(feats[0] + feats[1], feats[0], bilinear=True)

        # self.fc8_skip = nn.Sequential(Conv2d(256, 48, 1, bias=False),
        #                               # nn.BatchNorm2d(48, track_running_stats = False),
        #                               nn.GroupNorm(24, 48),
        #                               nn.LeakyReLU(0.2))
        # self.fc8_x = nn.Sequential(Conv2d(560, 256, kernel_size=3, stride=1, padding=1, bias=False),
        #                            #    nn.BatchNorm2d(256, track_running_stats = False),
        #                            nn.GroupNorm(32, 256),
        #                            nn.LeakyReLU(0.2))

        # decoder
        # self.last_conv = nn.Sequential(Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
        #                                # nn.BatchNorm2d(256, track_running_stats = False),
        #                                nn.GroupNorm(32, 256),
        #                                nn.LeakyReLU(0.2),
        #                                nn.Dropout(0.5),
        #                                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
        #                                # nn.BatchNorm2d(256, track_running_stats = False),
        #                                nn.GroupNorm(32, 256),
        #                                nn.LeakyReLU(0.2),
        #                                nn.Dropout(0.1),
        #                                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #                                nn.LeakyReLU(0.2),
        #                                nn.Conv2d(256, 256, kernel_size=1, stride=1),
        #                                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #                                nn.LeakyReLU(0.2),
        #                                nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.last_conv = Conv2d(feats[0], num_classes, kernel_size=1, stride=1, bias=False)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or dilation != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                self.conv(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, dilation=max(1, dilation / 2), bias=False),
                self.norm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            dilation=max(1, dilation / 2), conv=self.conv,
                            norm=self.norm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                                conv=self.conv, norm=self.norm))

        return nn.Sequential(*layers)

    def forward(self, x):
        # size = (x.shape[2], x.shape[3])
        x1 = self.conv1_(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)
        x1 = self.layer1(x1)
        # conv3 = x1
        # print(conv3.shape)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x_feats = self.aspp(x4)

        # x2_x = self.fc8_skip(conv3)
        # x_up = rescale_as(x_feats, x2_x)
        # x = self.fc8_x(torch.cat([x_up, x2_x], 1))
        x = self.up1(x_feats, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        # x = self.up4(x, x1)
        # 3.2 deep feature context for shallow features
        # x2 = self.shallow_mask(conv3, x)
        # 3.3 stochastically merging the masks
        # x = self.sg(x, x2, alpha_rate=0.3)
        # x = self.last_conv(x)
        # x = nn.Upsample(size, mode='bilinear', align_corners=True)(x)
        return x, x4


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    if pretrained:
        model_dict = model.state_dict()
        pretrained_path = "/home/native/projects/data/smart_watch/models/experiments_onera/tasks_experiments_onera_2021-10-07-10:23/experiments_epoch_34_loss_2151.7745061910377_valmIoU_0.5357620684181676_time_2021-10-09-07:21:41.pth"
        pretrained_dict = torch.load(pretrained_path)
        # pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        overlap_dict = {k: v for k, v in pretrained_dict.items()
                        if k in model_dict}
        model_dict.update(overlap_dict)
        model.load_state_dict(model_dict)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    if pretrained:
        model_dict = model.state_dict()
        pretrained_path = "/home/native/projects/data/smart_watch/models/experiments_onera/tasks_experiments_onera_2021-10-07-10:23/experiments_epoch_34_loss_2151.7745061910377_valmIoU_0.5357620684181676_time_2021-10-09-07:21:41.pth"
        pretrained_dict = torch.load(pretrained_path)['model']
        print(pretrained_dict.keys())
        # print(pretrained_dict.values())
        print(model_dict.keys())
        # pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        overlap_dict = {k[7:]: v for k, v in pretrained_dict.items()
                        if k[7:] in model_dict}
        print(f"loaded {len(overlap_dict)} layers")
        model_dict.update(overlap_dict)
        model.load_state_dict(model_dict)
    exit()
    return model


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [2, 2, 2, 2], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        overlap_dict = {k: v for k, v in pretrained_dict.items()
                        if k in model_dict}
        model_dict.update(overlap_dict)
        model.load_state_dict(model_dict)
    return model


def resnet101(pretrained=False, num_groups=None, weight_std=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                   num_groups=num_groups, weight_std=weight_std, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        if num_groups and weight_std:
            pretrained_path = '/home/native/projects/data/smart_watch/models/R-101-GN-WS.pth.tar'
            # pretrained_path = "/home/native/projects/data/smart_watch/models/experiments_onera/tasks_experiments_onera_2021-10-07-10:23/experiments_epoch_34_loss_2151.7745061910377_valmIoU_0.5357620684181676_time_2021-10-09-07:21:41.pth"
            pretrained_dict = torch.load(pretrained_path)
            # print(pretrained_dict['conv1'])
            overlap_dict = {k[7:]: v for k, v in pretrained_dict.items()
                            if k[7:] in model_dict}
            # assert len(overlap_dict) == 312, len(overlap_dict)
            print(f"loaded {len(overlap_dict)} layers")
        elif not num_groups and not weight_std:
            pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
            overlap_dict = {k: v for k, v in pretrained_dict.items()
                            if k in model_dict}
        else:
            raise ValueError('Currently only support BN or GN+WS')
        model_dict.update(overlap_dict)
        model.load_state_dict(model_dict, strict=False)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
