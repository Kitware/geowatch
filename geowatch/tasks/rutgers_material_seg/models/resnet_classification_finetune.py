# flake8: noqa

"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from geowatch.tasks.rutgers_material_seg.models.encoding import Encoding
from geowatch.tasks.rutgers_material_seg.models.tex_refine import TeRN


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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_channels=3, zero_init_residual=False,
                pretrained=False, num_classes=None, beta=False, weight_std=False,
                num_groups=32, out_dim=128, feats=[64, 64, 128, 256, 512]):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.num_codewords = 64
        self.conv1_ft = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1,
        #                        padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.out = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, bias=False)
        self.aspp_ft = ASPP(512, 256, num_classes)
        self.fc = nn.Linear(256, num_classes)

        self._aff = TeRN(num_iter=10, dilations=[1, 1, 2, 4, 6, 8])

        self.encoding = nn.Sequential(
            Encoding(channels=num_classes, num_codes=self.num_codewords),
            # nn.BatchNorm2d(self.num_codewords),
            nn.LeakyReLU(-0.2))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1_ft(x)))
        refined_out = self._aff(x, out)
        out = self.layer1(refined_out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.aspp_ft(out)
        out = self.encoding(out).mean(dim=1)
        # classifer = self.out(out)
        # classifer = torch.flatten(classifer, 1)
        out = torch.flatten(out, 1)
        # out = self.fc(out)
        return out  # , classifer


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(pretrained=False, **kwargs):
    # return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        # /home/native/projects/data/smart_watch/models/experiments_onera/tasks_experiments_onera_2021-10-18-13:27/experiments_epoch_0_loss_11.28138166103723_valmF1_0.6866047574166068_valChangeF1_0.49019877611815305_time_2021-10-18-14:15:27.pth
        # pretrained_path = "/home/native/projects/data/smart_watch/models/experiments_onera/tasks_experiments_onera_2021-10-07-10:23/experiments_epoch_8_loss_3394.9326448260613_valmIoU_0.5388350590429163_time_2021-10-07-22:05:00.pth"
        # pretrained_path = "/home/native/projects/data/smart_watch/models/experiments_onera/tasks_experiments_onera_2021-10-18-13:27/experiments_epoch_0_loss_11.28138166103723_valmF1_0.6866047574166068_valChangeF1_0.49019877611815305_time_2021-10-18-14:15:27.pth"
        pretrained_dict = torch.load(pretrained)['model']
        # pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        overlap_dict = {k[7:]: v for k, v in pretrained_dict.items()
                        if k[7:] in model_dict}
        for k, v in overlap_dict.items():
            v.requires_grad = False
        model_dict.update(overlap_dict)
        model.load_state_dict(model_dict)
        print(f"loaded {len(overlap_dict)}/{len(pretrained_dict)} layers")
        non_loaded_layers = pretrained_dict.keys() - overlap_dict.keys()
        print(f"Layers not loaded: {non_loaded_layers}")
    return model


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
}


class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""

    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x


class SupConResNet(nn.Module):
    """backbone + projection head"""

    def __init__(self, name='resnet50', head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat


class SupCEResNet(nn.Module):
    """encoder + classifier"""

    def __init__(self, name='resnet50', num_classes=10):
        super(SupCEResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))


class LinearClassifier(nn.Module):
    """Linear classifier"""

    def __init__(self, name='resnet50', num_classes=10):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)
