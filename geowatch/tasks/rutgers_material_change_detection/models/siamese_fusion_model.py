import torch

from geowatch.tasks.rutgers_material_change_detection.models.base_model import BaseModel
from geowatch.tasks.rutgers_material_change_detection.models.unet import UNetDecoder, UNetEncoder


class SiameseFusion(BaseModel):
    def __init__(
        self,
        input_size=256,
        num_channels=13,
        feat_layer=2,
        backbone_name="resnet18",
        pretrain=False,
        mode="bilinear",
        decoder_type="transpose_conv",
    ):
        input_size = [input_size, input_size]
        super().__init__(input_size)

        self._mode = mode
        self._pretrain = pretrain
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
                feat_channels=feat_dim * 2,  # Features are concatenated
                out_channels=2,
            )
        elif self.decoder_type == "transpose_conv":
            self.decoder = self.build_transpose_conv_decoder(
                height=self._input_size[0],
                width=self._input_size[1],
                feat_channels=feat_dim * 2,  # Features are concatenated
                out_channels=2,
            )
        elif self.decoder_type == "unet":
            assert self._backbone_name == "unet"
            self.decoder = UNetDecoder(n_classes=2, channel_factor=2)

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
            feats = torch.concat((ref_feats, query_feats), dim=1)
        else:
            feats = []
            for i in range(len(ref_feats)):
                feats.append(torch.concat((ref_feats[i], query_feats[i]), dim=1))

        return feats

    def decode(self, feats, height, width):
        output = self.decoder(feats)
        output = output.squeeze()
        return output

    def decode_out_features(self, feats, height, width):
        output_feats = self.decoder.get_output_feats(feats)
        return output_feats


if __name__ == "__main__":
    # Pass dummy data through model.
    F, C, H, W = 2, 4, 250, 250
    data = torch.zeros([F, C, H, W])
    model = SiameseFusion(input_size=H, num_channels=C, decoder_type="transpose_conv")

    feats = model.encode(data[0].unsqueeze(0), data[-1].unsqueeze(0))
    output = model.decode(feats, H, W)

    print("Input shape: ", [C, H, W])
    print("Output shape: ", output.shape)
