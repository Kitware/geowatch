import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Resnet


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, in_channels, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            in_channels, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.linear = nn.Linear(512, 2*embed_dim) ###change num_classes->2
        # self.linear2 = nn.Linear(embed_dim,2*embed_dim)
        # self.classifier = nn.Linear(2*embed_dim,num_classes) ###new
        # self.embed_dim = embed_dim

        self.act1 = nn.ReLU()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, view=None):

        x = F.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        out = self.layer2(x)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)

        return out


def ResNet18(in_channels):
    return ResNet(in_channels, BasicBlock, [2, 2, 2, 2])


class Head(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.layer1 = nn.Linear(in_size, in_size)
        self.layer2 = nn.Linear(in_size, out_size)
        self.bn = nn.BatchNorm1d(in_size)

    def forward(self, x):
        x.unsqueeze(1)
        x = F.relu(self.bn(self.layer1(x)))
        x = self.layer2(x)
        return x


class Prob_ResNet18(nn.Module):
    def __init__(self, in_channels=3, in_size=512, encoded_size=2):
        super(Prob_ResNet18, self).__init__()

        self.act1 = nn.ELU()
        self.fc1a = nn.Linear(in_size // 2, 32)
        self.fc1b = nn.Linear(in_size // 2, 32)
        #self.fc2a = nn.Linear(13, 16)
        #self.fc2b = nn.Linear(13, 16)
        self.fc2a = nn.Linear(32, encoded_size)
        self.fc2b = nn.Linear(32, encoded_size)
        self.fc4 = nn.Linear(encoded_size, 20)
        #self.act2 = nn.Softmax(dim=0)
        self.fc5 = nn.Linear(20, 10)

        self.resnet = ResNet18(in_channels)

        self.encoded_size = encoded_size

    def forward(self, x):

        x = self.resnet(x)
        means = self.act1(self.fc1a(x[:, :256]))
        variances = self.act1(self.fc1b(x[:, 256:]))

        means = self.fc2a(means)  # B x 35
        variances = F.softplus(self.fc2b(variances))

        dist = Normal(means, variances)  # build distributions
        # B x 20 features for classifier
        tmp = self.act1(self.fc4(dist.rsample()))
        out = self.fc5(tmp)  # B x 10 Classification layer
        return dist, out


if __name__ == '__main__':
    x = torch.randn((4, 3, 64, 64))
    model = ResNet18(in_channels=3)
    print(model(x).shape)
