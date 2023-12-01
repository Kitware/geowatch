import torch
import torch.nn as nn

from geowatch.tasks.rutgers_material_change_detection.models.base_model import BaseModel
from geowatch.tasks.rutgers_material_change_detection.models.unet import UNetDecoder, UNetEncoder


class SiameseDifference(BaseModel):
    def __init__(
        self,
        input_size=256,
        num_channels=13,
        threshold=1.3,
        feat_layer=2,
        backbone_name="resnet18",
        pretrain=False,
        mode="bilinear",
        decoder_type="transpose_conv",
    ):

        input_size = [input_size, input_size]
        super(SiameseDifference, self).__init__(input_size)

        self._mode = mode
        self._pretrain = pretrain
        self._threshold = threshold
        self.decoder_type = decoder_type
        self._num_channels = num_channels
        self._feat_layer = "layer" + str(feat_layer)
        self._backbone_name = backbone_name.lower()

        self.build()

    def build(self):

        # Build encoder network to get features.
        if self._backbone_name[:6] == "resnet":
            # Get feature extractor network (resnet).
            self.encoder, feat_dim = self.load_resnet_model(
                self._backbone_name, n_input_channels=self._num_channels, pretrained=self._pretrain
            )
        elif self._backbone_name == "unet":
            assert self._backbone_name == "unet"
            self.encoder = UNetEncoder(n_channels=self._num_channels)
            feat_dim = 512

        # Build decoder network to upsample features.
        if self.decoder_type == "upsample":
            self.decoder = self.build_upsample_conv_decoder(
                height=self._input_size[0],
                width=self._input_size[1],
                feat_channels=feat_dim,  # Features are concatenated
                out_channels=2,
            )
        elif self.decoder_type == "transpose_conv":
            self.decoder = self.build_transpose_conv_decoder(
                height=self._input_size[0],
                width=self._input_size[1],
                feat_channels=feat_dim,  # Features are concatenated
                out_channels=2,
            )
        elif self.decoder_type == "unet":
            assert self._backbone_name == "unet"
            self.decoder = UNetDecoder(n_classes=2, channel_factor=2)

    def forward(self, data):
        """Computes a forward pass of model.

        Compare the difference in feature activation between two images and then resize to the input image shape.

        Args:
            data (dict): A dictionary that must contain an 'images' key. The 'images' key corresponds to a tensor of
                shape [batch_size, num_images, num_channels, height, width].

        Returns:
            output (dict): A dictionary containing model predictions. Return dictionary with 'change_pred' key containing
                a int32 tensor of shape [batch_size, 1, height, width].
        """
        images = data["images"]

        # Expecting a number of images dimension
        assert len(images.shape) == 5

        B, N, C, H, W = images.shape

        # The number of images should be equal or more than two
        assert N >= 2

        # Get first and last image in input data
        ref_image = images[:, 0, ...]
        query_image = images[:, -1, ...]

        # Pass images through pretrained network
        ref_feats = self.model.forward_feat(ref_image)[self._feat_layer]
        query_feats = self.model.forward_feat(query_image)[self._feat_layer]

        # Compute absolute difference between backbone features
        diff_feats = abs(ref_feats - query_feats)  # [B, c, h, w];  h != H & w != W, c != C
        diff_feats, _ = torch.max(diff_feats, dim=1)  # [B, h, w]
        diff_feats = diff_feats.unsqueeze(1)  # [B, 1, h, w]

        # Resize features to original input size -> [B, 1, H, W]
        diff_feats = nn.functional.interpolate(diff_feats, size=(H, W), mode=self._mode, align_corners=True)

        # Compute change prediction
        # Compare difference activation to threshold
        change_pred = diff_feats > self._threshold

        # Convert from bool to integer (int32)
        change_pred = change_pred.int()

        return change_pred

    # def encode(self, ref_frame, query_frame):
    #     """[summary]

    #     Args:
    #         ref_frame ([type]): [description]
    #         query_frame ([type]): [description]

    #     Returns:
    #         (): TODO
    #     """
    #     # Pass images through pretrained network
    #     ref_feats = self.model.forward_feat(ref_frame)[self._feat_layer]
    #     query_feats = self.model.forward_feat(query_frame)[self._feat_layer]

    #     # Compute absolute difference between backbone features
    #     diff_feats = ref_feats - query_feats  # [c, h, w];  h != H & w != W, c != C
    #     change_feats = self.final_change_conv(diff_feats)

    #     return change_feats

    def encode(self, ref_frame, query_frame):
        # Pass images through pretrained network
        # TODO: Clean logic up.
        if self._backbone_name[:6] == "resnet":
            ref_feats = self.encoder.forward_feat(ref_frame)[self._feat_layer]
            query_feats = self.encoder.forward_feat(query_frame)[self._feat_layer]
        else:
            ref_feats = self.encoder(ref_frame)
            query_feats = self.encoder(query_frame)

        if self._backbone_name[:6] == "resnet":
            # Concatenate features along channel dimension.
            # Features shape: [batch_size, n_channels, height, width]
            feats = query_feats - ref_feats
        else:
            feats = []
            for i in range(len(ref_feats)):
                feats.append(query_feats[i] - ref_feats[i])

        return feats

    # def decode(self, feats, height, width):
    #     """[summary]

    #     Args:
    #         feats ([type]): [description]

    #     Returns:
    #         (): TODO
    #     """
    #     # Resize features to original input size -> [1, H, W]
    #     output = nn.functional.interpolate(feats, size=(height, width), mode=self._mode, align_corners=True)
    #     output = output.squeeze()

    #     return output

    def decode(self, feats, height, width):
        output = self.decoder(feats)
        output = output.squeeze()
        return output

    def decode_out_features(self, feats, height, width):
        output_feats = self.decoder.get_output_feats(feats)
        return output_feats


if __name__ == "__main__":
    # Create fake input data (cpu)
    data = {"images": torch.zeros(5, 2, 3, 256, 256)}

    # Build model object
    model = SiameseDifference()

    # Pass data through model
    pred = model(data)
    assert "change_pred" in pred.keys()
    assert len(pred["change_pred"].shape) == 4
