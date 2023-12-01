from geowatch.tasks.rutgers_material_change_detection.models.base_model import BaseFramework


class EarlyFusionFramework(BaseFramework):
    def __init__(self, task_mode, encoder, decoder, attention, **kwargs):
        super(EarlyFusionFramework, self).__init__(task_mode, encoder, decoder, attention)

    def forward(self, data):
        """Forward pass.

        Args:
            data (dict): [description]

        Returns:
            [type]: [description]
        """

        video = data["video"]  # [batch_size, frames, channels, height, width]

        # Stack frames along the channel dimension.
        n_channels = video.shape[1] * video.shape[2]
        video = video.view(video.shape[0], n_channels, video.shape[3], video.shape[4])

        # Pass input into the encoder and decoder.
        feats = self.encoder(video)

        # Compute attention over features.
        if self.attention:
            # TODO: Concatenate relative positional features here.
            feats = self.attention(feats)

        # Pass features through decoder.
        if self.decoder.name == "unet":
            pred = self.decoder(feats, video)
        else:
            pred = self.decoder(feats)

        output = {}
        output[self.task_mode] = pred
        return output
