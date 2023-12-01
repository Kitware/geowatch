import torch.nn as nn
import torchvision.models as models

import geowatch.tasks.rutgers_material_change_detection.models.resnet as resnet


class BaseFramework(nn.Module):
    def __init__(self, task_mode, encoder, decoder, attention):
        super(BaseFramework, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention
        self.task_mode = task_mode


class BaseDecoder(nn.Module):
    def __init__(self, feat_sizes, out_channels, name):
        super(BaseDecoder, self).__init__()
        self.name = name
        self.feat_sizes = feat_sizes
        self.out_channels = out_channels


class BaseModel(nn.Module):
    def __init__(self, input_size):
        """Constructor.

        Args:
            input_size (list): A 2 length list containing integer values for the height and width of the input images.
        """
        super(BaseModel, self).__init__()
        assert len(input_size) == 2
        self._input_size = input_size

    def load_resnet_model(self, backbone_name, n_input_channels, pretrained=False, freeze=False):
        # Get backbone model with forward_feat method.
        if backbone_name[:6] == "resnet":
            if backbone_name == "resnet18":
                model = resnet.resnet18()
                feat_dim = 128
            elif backbone_name == "resnet34":
                model = resnet.resnet34()
                feat_dim = 128
            elif backbone_name == "resnet50":
                model = resnet.resnet50()
                feat_dim = 256
        else:
            raise NotImplementedError(f"Have not implemented {self._backbone_name} model.")

        if pretrained is not None:
            # Get pretrained model weights.
            if pretrained == "imagenet":
                print("Loading ImageNet weights.")
                if backbone_name[:6] == "resnet":
                    if backbone_name == "resnet18":
                        pt_model_weights = models.resnet18(self._pretrain)
                    elif backbone_name == "resnet34":
                        pt_model_weights = models.resnet34(self._pretrain)
                    elif backbone_name == "resnet50":
                        pt_model_weights = models.resnet50(self._pretrain)
                else:
                    raise NotImplementedError(f"Have not implemented {self._backbone_name} pretrained weights.")
            else:
                raise NotImplementedError(f"Pretrained weights not implemented for {pretrained}.")

            # Load pretrained weights into backbone model
            model.load_state_dict(pt_model_weights.state_dict())

        if n_input_channels != 3:
            # Update the number of input channels.
            model.conv1 = nn.Conv2d(
                n_input_channels,
                model.conv1.out_channels,
                kernel_size=model.conv1.kernel_size,
                stride=model.conv1.stride,
                padding=model.conv1.padding,
                bias=model.conv1.bias,
            )

            # Initialize with Xavier weights.
            nn.init.xavier_uniform_(model.conv1.weight)

        # Freeze model weights.
        if freeze is True:
            print("Freezing encoder weights.")
            for param in model.parameters():
                param.requires_grad = False

        return model, feat_dim

    def build_upsample_conv_decoder(
        self, height, width, feat_channels, out_channels, pad_size=2, upsample_mode="nearest"
    ):
        decoder = nn.Sequential(
            nn.ReflectionPad2d(pad_size),
            nn.Upsample(size=(height, width), mode=upsample_mode),
            nn.Conv2d(feat_channels, out_channels, 3, stride=1, padding=1),
        )
        return decoder

    def build_transpose_conv_decoder(self, height, width, feat_channels, out_channels, upsample_mode="nearest"):
        decoder = nn.Sequential(
            nn.ConvTranspose2d(feat_channels, feat_channels // 2, 4, stride=2, padding=4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feat_channels // 2, feat_channels // 4, 4, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feat_channels // 4, feat_channels // 4, 4, stride=2),
            nn.Upsample(size=(height, width), mode=upsample_mode),
            nn.Conv2d(feat_channels // 4, out_channels, 3, stride=1, padding=1),
        )
        return decoder

    def forward(self, index):
        raise NotImplementedError

    def video2frames(self, video):
        """Separate frames of video object.

        Args:
            video (torch.Tensor): A float tensor of shape [batch_size, n_frames, n_channels, height, width].

        Returns:
            list: A list containing each frame of video. Each frame is shape [n_channels, height, width].
        """

        assert len(video.shape) == 5

        n_frames = video.shape[1]

        frames = []
        for i in range(n_frames):
            frames.append(video[:, i])

        return frames
