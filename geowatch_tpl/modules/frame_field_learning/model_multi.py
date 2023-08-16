import torch
from torch import nn
# import torch.nn.functional as F
# from segmentation_models_pytorch.base.modules import Flatten, Activation

from torchvision.models.segmentation._utils import _SimpleSegmentationModel
from frame_field_learning import tta_utils


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class Activation(nn.Module):

    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == 'identity':
            self.activation = nn.Identity(**params)
        elif name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif name == 'softmax2d':
            self.activation = nn.Softmax(dim=1, **params)
        elif name == 'softmax':
            self.activation = nn.Softmax(**params)
        elif name == 'logsoftmax':
            self.activation = nn.LogSoftmax(**params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError('Activation should be callable/sigmoid/softmax/logsoftmax/None; got {}'.format(name))

    def forward(self, x):
        return self.activation(x)


def get_out_channels(module):
    if hasattr(module, "out_channels"):
        return module.out_channels
    children = list(module.children())
    i = 1
    out_channels = None
    while out_channels is None and i <= len(children):
        last_child = children[-i]
        out_channels = get_out_channels(last_child)
        i += 1
    # If we get out of the loop but out_channels is None,
    # then the prev child of the parent module will be checked, etc.
    return out_channels


# Need to change this later
def get_encoder(module, x):
    x1 = module.conv1(x)
    x2 = module.conv2(x1)
    x3 = module.conv3(x2)
    x4 = module.conv4(x3)
    x5 = module.conv5(x4)

    return [x, x1, x2, x3, x4, x5]


def get_encoder_channels(module):
    c1 = get_out_channels(module.conv1)
    c2 = get_out_channels(module.conv2)
    c3 = get_out_channels(module.conv3)
    c4 = get_out_channels(module.conv4)
    c5 = get_out_channels(module.conv5)

    return [3, c1, c2, c3, c4, c5]


class Multi_FrameFieldModel(torch.nn.Module):
    def __init__(self, config: dict, backbone, train_transform=None, eval_transform=None):
        """

        :param config:
        :param backbone: A _SimpleSegmentationModel network,
                         its output features will be used to compute seg and framefield.
        :param train_transform: transform applied to the inputs when self.training is True
        :param eval_transform: transform applied to the inputs when self.training is False
        """
        super(Multi_FrameFieldModel, self).__init__()
        assert config["compute_seg"] or config["compute_crossfield"], \
            "Model has to compute at least one of those:\n" \
            "\t- segmentation\n" \
            "\t- cross-field"
        assert isinstance(backbone, _SimpleSegmentationModel), \
            "backbone should be an instance of _SimpleSegmentationModel"
        self.config = config
        self.backbone = backbone
        self.train_transform = train_transform
        self.eval_transform = eval_transform

        bn_momentum = 0.1

        backbone_out_features = get_out_channels(self.backbone)

        # For geo-centric pose
        encoder_out_channels = get_encoder_channels(self.backbone.backbone)

        self.xydir_head = EncoderRegressionHead(
            in_channels=encoder_out_channels[-1],
            out_channels=2,
        )

        self.height_head = RegressionHead(
            in_channels=backbone_out_features,
            out_channels=1,
            kernel_size=3,
        )

        self.mag_head = RegressionHead(
            in_channels=backbone_out_features,
            out_channels=1,
            kernel_size=3,
        )

        self.scale_head = ScaleHead()

        # self.initialize_geopose()

        # --- Add other modules if activated in config:
        seg_channels = 0
        if self.config["compute_seg"]:
            seg_channels = self.config["seg_params"]["compute_vertex"]\
                           + self.config["seg_params"]["compute_edge"]\
                           + self.config["seg_params"]["compute_interior"]
            self.seg_module = torch.nn.Sequential(
                torch.nn.Conv2d(backbone_out_features, backbone_out_features, 3, padding=1),
                torch.nn.BatchNorm2d(backbone_out_features, momentum=bn_momentum),
                torch.nn.ELU(),
                torch.nn.Conv2d(backbone_out_features, seg_channels, 1),
                torch.nn.Sigmoid(),)

        if self.config["compute_crossfield"]:
            crossfield_channels = 4
            self.crossfield_module = torch.nn.Sequential(
                torch.nn.Conv2d(backbone_out_features + seg_channels, backbone_out_features, 3, padding=1),
                torch.nn.BatchNorm2d(backbone_out_features, momentum=bn_momentum),
                torch.nn.ELU(),
                torch.nn.Conv2d(backbone_out_features, crossfield_channels, 1),
                torch.nn.Tanh(),
            )

        # Add the entry in the config file later
        if True:
            depth_channels = 1

            self.depth_module = torch.nn.Sequential(

                    torch.nn.Conv2d(backbone_out_features, backbone_out_features * 2, 3, padding=1, bias=False),
                    torch.nn.ReLU(),

                    torch.nn.Conv2d(backbone_out_features * 2, backbone_out_features * 2, 3, padding=1, bias=False),
                    torch.nn.BatchNorm2d(backbone_out_features * 2, momentum=bn_momentum),
                    torch.nn.ReLU(),

                    torch.nn.Conv2d(backbone_out_features * 2, backbone_out_features * 2, 3, padding=1, bias=False),
                    torch.nn.BatchNorm2d(backbone_out_features * 2, momentum=bn_momentum),
                    torch.nn.ReLU(),

                    # This is original
                    torch.nn.Conv2d(backbone_out_features * 2, depth_channels, 1, bias=True),
                    # torch.nn.ReLU(),

                    # torch.nn.Conv2d(backbone_out_features*2, backbone_out_features*2, 3, padding=1, bias=False),

                    # torch.nn.BatchNorm2d(backbone_out_features*2),
                    # torch.nn.ReLU(),

                    # torch.nn.Conv2d(backbone_out_features*2, depth_channels, 1, bias=True),
                    # torch.nn.ReLU(),
            )

            # shadow_channels = 1

            self.shadow_module = torch.nn.Sequential(
                    torch.nn.Conv2d(backbone_out_features, backbone_out_features * 2, 3, padding=1, bias=True),
                    torch.nn.ReLU(),

                    torch.nn.Conv2d(backbone_out_features * 2, backbone_out_features * 2, 3, padding=1, bias=False),
                    torch.nn.BatchNorm2d(backbone_out_features * 2, momentum=bn_momentum),
                    torch.nn.ReLU(),

                    torch.nn.Conv2d(backbone_out_features * 2, backbone_out_features * 2, 3, padding=1, bias=False),
                    torch.nn.BatchNorm2d(backbone_out_features * 2, momentum=bn_momentum),
                    torch.nn.ReLU(),

                    torch.nn.Conv2d(backbone_out_features * 2, depth_channels, 1, bias=True),
                    torch.nn.Sigmoid(),
                    # torch.nn.Tanh(),
            )

            # facade_channels = 1

            self.facade_module = torch.nn.Sequential(
                    torch.nn.Conv2d(backbone_out_features, backbone_out_features * 2, 3, padding=1, bias=True),
                    torch.nn.ReLU(),

                    torch.nn.Conv2d(backbone_out_features * 2, backbone_out_features * 2, 3, padding=1, bias=False),
                    torch.nn.BatchNorm2d(backbone_out_features * 2, momentum=bn_momentum),
                    torch.nn.ReLU(),

                    torch.nn.Conv2d(backbone_out_features * 2, backbone_out_features * 2, 3, padding=1, bias=False),
                    torch.nn.BatchNorm2d(backbone_out_features * 2, momentum=bn_momentum),
                    torch.nn.ReLU(),

                    torch.nn.Conv2d(backbone_out_features * 2, depth_channels, 1, bias=True),
                    torch.nn.Sigmoid(),
                    # torch.nn.Tanh(),
            )

    '''
    def initialize_geopose(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.xydir_head)
        init.initialize_head(self.height_head)
        init.initialize_head(self.mag_head)
        init.initialize_head(self.scale_head)
    '''

    def inference(self, image):
        outputs = {}

        encoder_features = get_encoder(self.backbone.backbone, image)

        # --- Extract features for every pixel of the image with a U-Net --- #
        backbone_features0 = self.backbone(image)["out"]

        if self.config["compute_seg"]:
            # --- Output a segmentation of the image --- #
            seg = self.seg_module(backbone_features0)
            seg_to_cat = seg.clone().detach()
            backbone_features = torch.cat([backbone_features0, seg_to_cat], dim=1)  # Add seg to image features
            outputs["seg"] = seg

        if self.config["compute_crossfield"]:
            # --- Output a cross-field of the image --- #
            crossfield = 2 * self.crossfield_module(backbone_features)  # Outputs c_0, c_2 values in [-2, 2]
            outputs["crossfield"] = crossfield

        # Add the entry in the config file later
        if True:
            # --- Output a depth map of the image --- #
            depth = self.depth_module(backbone_features0)  # Outputs values in [0, ???]
            outputs["depth"] = depth

            # --- Output a shadow response map of the image --- #
            shadow = self.shadow_module(backbone_features0)  # Outputs values in [0, 1]
            outputs["shadow"] = shadow

            # --- Output a facade response map of the image --- #
            facade = self.facade_module(backbone_features0)  # Outputs values in [0, 1]
            outputs["facade"] = facade

        if True:

            # For Geo-Centric Pose
            xydir = self.xydir_head(encoder_features[-1])
            height = self.height_head(backbone_features0)
            # height = depth
            mag = self.mag_head(backbone_features0)
            scale = self.scale_head(mag, height)

            if scale.ndim == 0:
                scale = torch.unsqueeze(scale, axis=0)

            outputs['xydir'] = xydir
            outputs['height'] = height
            outputs['mag'] = mag
            outputs['scale'] = scale

        return outputs

    # @profile
    def forward(self, xb, tta=False):

        # pdb.set_trace()

        # print("\n### --- PolyRefine.forward(xb) --- ####")
        if self.training:
            if self.train_transform is not None:
                xb = self.train_transform(xb)
        else:
            if self.eval_transform is not None:
                xb = self.eval_transform(xb)

        if not tta:
            final_outputs = self.inference(xb["image"])
        else:
            final_outputs = tta_utils.tta_inference(self, xb, self.config["eval_params"]["seg_threshold"])

        return final_outputs, xb


class RegressionHead(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        identity = torch.nn.Identity()
        activation = Activation(None)
        super().__init__(conv2d, identity, activation)


class EncoderRegressionHead(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        pool = torch.nn.AdaptiveAvgPool2d(1)
        flatten = Flatten()
        dropout = torch.nn.Dropout(p=0.5, inplace=True)
        linear = torch.nn.Linear(in_channels, 2, bias=True)
        super().__init__(pool, flatten, dropout, linear)


class ScaleHead(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.flatten = torch.flatten
        self.dot = torch.dot

    def forward(self, mag, height):
        curr_mag = self.flatten(mag, start_dim=1)
        curr_height = self.flatten(height, start_dim=1)
        batch_size = curr_mag.shape[0]
        length = curr_mag.shape[1]
        denom = (
            torch.squeeze(
                torch.bmm(
                    curr_height.view(batch_size, 1, length),
                    curr_height.view(batch_size, length, 1),
                )
            )
            + 0.01
        )
        pinv = curr_height / denom.view(batch_size, 1)
        scale = torch.squeeze(
            torch.bmm(
                pinv.view(batch_size, 1, length), curr_mag.view(batch_size, length, 1)
            )
        )
        return scale
