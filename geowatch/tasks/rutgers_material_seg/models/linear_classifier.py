import torch.nn as nn
# import torch
# from torch.nn import functional as F


class VSNet(nn.Module):
    def __init__(self, num_channels=3, num_classes=3,
                 bilinear=True, pretrained=False,
                 beta=False, weight_std=False,
                 num_groups=32):
        super(VSNet, self).__init__()

        self.layers = [
                        nn.Conv2d(num_channels, num_channels, kernel_size=1),
                        # nn.ReLU()
                        ]
        self.model = nn.Sequential(*self.layers)
        # self.model = nn.Linear(3,1)

    def forward(self, image):
        out = self.model(image)
        return out
