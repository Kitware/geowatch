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
from geowatch.tasks.rutgers_material_seg.models.quantizer import Quantizer
from geowatch.tasks.rutgers_material_seg.models.tex_refine import TeRN
from torchvision import transforms


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

        self.pre_conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.pre_bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64 + num_channels, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.aspp = ASPP(512, 256, 256)
        self.aspp = ASPP(2048, 256, 256)
        # self.quantizer = Quantizer(n_clusters=self.num_codewords, mode='euclidean', verbose=0, minibatch=None)

        self.encoding = nn.Sequential(
            Encoding(channels=256, num_codes=self.num_codewords),
            # nn.BatchNorm2d(self.num_codewords),
            nn.LeakyReLU(-0.2))
        # self.fc = nn.Sequential(nn.Linear(256, 256), nn.Sigmoid())

        self._aff = TeRN(num_iter=10, dilations=[1, 1, 2, 4, 6, 8])

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

    def uncrop(self, cropped_image, params, H, W):
        bs, n_crops, c, h, w = cropped_image.shape
        uncrop = torch.zeros((bs, c, H, W), device=torch.device('cuda'))
        # uncrop_test = cropped_image.clone()
        # uncrop_test = uncrop_test.expand(-1,10,-1,-1,-1)
        # uncrop_test = torch.cat([uncrop_test, torch.zeros((bs,H*W-n_crops,c,h,w), device=torch.device('cuda'))], dim=1)
        # print(uncrop_test.shape)
        # with torch.no_grad():
        for b in range(bs):
            for crop in range(n_crops):
                top, left, height, width = params[crop]
                right = left + width
                bottom = top + height
                f_top, f_left = 0, 0
                f_right = f_left + width
                f_bottom = f_top + height

                if left < 0:
                    f_left = left - W
                    left = 0
                    # continue
                if top < 0:
                    f_top = top - H
                    top = 0
                    # continue
                if right > W:
                    f_right = W - left
                    right = W
                    # continue
                if bottom > H:
                    f_bottom = H - top
                    bottom = H
                    # continue

                # print(f"height: {height}, width:{width}")
                # print(f"left: {left}, top:{top}, right: {right}, bottom: {bottom} ")
                # print(f"f_left: {f_left}, f_top:{f_top}, f_right: {f_right}, f_bottom: {f_bottom} ")
                # features = cropped_image[b,crop,:,f_top:f_bottom, f_left:f_right]
                # print(features.shape)
                uncrop[b, :, top:bottom, left:right] += cropped_image[b, crop, :, f_top:f_bottom, f_left:f_right]
        # print(f"uncrop: {uncrop.dtype}")
        return uncrop

    def forward(self, x, original_image, sampled_crops):
        N, C, H, W = x.shape
        bs, c, h, w = original_image.shape

        # cropped_image = torch.stack([transforms.functional.crop(original_image, *params) for params in sampled_crops],dim=1)
        # bs, ps, pc, ph, pw = cropped_image.shape
        # cropped_image = cropped_image.view(bs*ps,pc,ph,pw)
        # x = x.view(bs*ps,c,ph,pw)
        # print(f"out: {out.dtype}")
        # print(out.requires_grad)
        # print(f"original_image: {original_image.shape}")
        out = F.relu(self.pre_bn1(self.pre_conv1(original_image)))

        refined_out = self._aff(original_image, out)
        # import matplotlib.pyplot as plt
        # import numpy as np
        # # out1_show = uncropped_out.sum(dim=1).cpu().detach().numpy()[0,:,:]
        # out1_show = refined_out.sum(dim=1).cpu().detach().numpy()[0,:,:]
        # # out1_show = uncropped_out.sum(dim=1).cpu().detach().numpy()[0,:,:]
        # out2_show = out.sum(dim=1).cpu().detach().numpy()[0,:,:]
        # x_show = np.transpose(original_image.cpu().detach().numpy()[0,:3,:,:], (1,2,0))
        # x_show = (x_show - x_show.min()) / (x_show.max() - x_show.min())
        # cmap_gradients = plt.cm.get_cmap('jet')
        # fig = plt.figure()
        # ax1 = fig.add_subplot(1,3,1)
        # ax2 = fig.add_subplot(1,3,2)
        # ax3 = fig.add_subplot(1,3,3)

        # ax1.imshow(x_show)
        # ax2.imshow(out1_show, cmap=cmap_gradients)
        # ax3.imshow(out2_show, cmap=cmap_gradients)
        # ax1.axis('off')
        # ax2.axis('off')
        # ax3.axis('off')
        # # plt.show()
        # image_name = np.random.randint(100000000000, size=1)[0]
        # plots_path_save = f"/home/native/projects/data/smart_watch/visualization/TextureRefinementNet"
        # fig_save_image_root = (f"{plots_path_save}/image_root/", ax1)
        # fig_save_out_root = (f"{plots_path_save}/out/", ax2)
        # fig_save_out_refined_root = (f"{plots_path_save}/out_refined/", ax3)

        # roots = [
        #     fig_save_image_root,
        #     fig_save_out_root,
        #     fig_save_out_refined_root,
        # ]

        # fig.savefig(
        #     f"{plots_path_save}/figs/{image_name}.png", bbox_inches='tight')
        # for root, ax in roots:
        #     file_path = f"{root}/{image_name}.png"
        #     # extent = ax.get_window_extent().transformed(figure.dpi_scale_trans.inverted())
        #     extent = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        #     fig.savefig(file_path, bbox_inches=extent)

        # fig.clear()
        # plt.cla()
        # plt.clf()
        # plt.close('all')
        # plt.close(fig)

        out = torch.stack([transforms.functional.crop(refined_out, *params) for params in sampled_crops], dim=1)
        bs, ps, pc, ph, pw = out.shape
        out = out.view(bs * ps, pc, ph, pw)

        # print(f"out: {out.shape}")
        # print(f"x: {x.shape}")
        out = torch.cat([out, x], dim=1)
        # print(f"out: {out.shape}")
        out = F.relu(self.bn1(self.conv1(out)))
        # print(f"out: {out.shape}")

        # out_clone = out.clone().detach()
        # print(f"out leaf: {out.is_leaf}")
        # with torch.no_grad():
        # with torch.no_grad():
        # chuncked_out = torch.stack(torch.chunk(out_clone, chunks=bs, dim=0), dim=0)
        # uncropped_out = self.uncrop(chuncked_out, params=sampled_crops, H=h, W=w)
        # # print(f"uncropped_out: {uncropped_out.shape}")
        # cropped_out = torch.stack([transforms.functional.crop(uncropped_out, *params) for params in sampled_crops], dim=1)
        # # print(f"cropped_out: {cropped_out.shape}")
        # bs, ps, pc, ph, pw = cropped_out.shape
        # cropped_out = cropped_out.view(bs*ps,pc,ph,pw)

        # print(cropped_out.requires_grad)
            # print(f"out: {out.shape}")
        # cropped_out.grad = out.grad
        # print(out.requires_grad)

        # print(f"out: {out.shape}")
        # print(f"uncropped_out: {uncropped_out.shape}")
        # print(f"uncropped_out: {uncropped_out.shape}")
        # print(f"cropped_image: {cropped_image.shape}")

        # print(f"out: {out.shape}")

        # out[:,:,:,:] = cropped_out.data

        # print(f"out: {out.shape}")
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.aspp(out)
        # print(f"out: {out.shape}")
        out_enc = self.encoding(out)
        out_enc = out_enc.mean(dim=1)
        out = torch.flatten(out_enc, 1)
        # print(f"encoding_feat: {encoding_feat.shape}")
        # gamma = self.fc(encoding_feat)
        # print(f"gamma: {gamma.shape}")
        # y = gamma.unsqueeze().unsqueeze()
        # out = F.relu_(out + out * gamma)
        # out = torch.flatten(out, 1)
        # print(f"output: {output.shape}")

        # dictionary = self.quantizer.fit_predict(out)
        # centroids = self.quantizer.centroids.T
        # print(f"centroids: {centroids.shape}")
        # residuals = torch.cdist(out, centroids.T)
        # print(f"residuals: {residuals.shape}")
        # quant =  torch.tensor([torch.histc(residuals[i], bins=self.num_codewords).tolist() for i in range(N)], device=torch.device('cuda'), requires_grad=True)

        return out  # , out

    # def forward(self, x, layer=100):
    #     N, C, H, W = x.shape
    #     recon_img = torch.stack(torch.chunk(x, chunks=4, dim=0), dim=0).view(4,1024,9,-1)#.mean(dim=-1)
    #     print(recon_img.shape)
    #     print(f"x: {x.shape}")
    #     out = F.relu(self.bn1(self.conv1(x)))
    #     print(f"out: {out.shape}")
    #     out = self._aff(x, out)
    #     print(f"out: {out.shape}")
    #     out = self.layer1(out)
    #     out = self.layer2(out)
    #     out = self.layer3(out)
    #     out = self.layer4(out)
    #     out = self.avgpool(out)
    #     out = self.aspp(out)
    #     # print(f"out: {out.shape}")
    #     encoding_feat = self.encoding(out).mean(dim=1)
    #     out = torch.flatten(out, 1)
    #     # print(f"encoding_feat: {encoding_feat.shape}")
    #     # gamma = self.fc(encoding_feat)
    #     # print(f"gamma: {gamma.shape}")
    #     # y = gamma.unsqueeze().unsqueeze()
    #     # out = F.relu_(out + out * gamma)
    #     # out = torch.flatten(out, 1)
    #     # print(f"output: {output.shape}")

    #     # dictionary = self.quantizer.fit_predict(out)
    #     # centroids = self.quantizer.centroids.T
    #     # print(f"centroids: {centroids.shape}")
    #     # residuals = torch.cdist(out, centroids.T)
    #     # print(f"residuals: {residuals.shape}")
    #     # quant =  torch.tensor([torch.histc(residuals[i], bins=self.num_codewords).tolist() for i in range(N)], device=torch.device('cuda'), requires_grad=True)

    #     return out, encoding_feat


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


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
