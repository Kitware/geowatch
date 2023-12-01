import torch
import torch.nn as nn
import torch.nn.functional as F

from geowatch.tasks.rutgers_material_change_detection.models.base_model import BaseFramework


class EarlyFusionMatFramework(BaseFramework):
    def __init__(self, task_mode, encoder, decoder, attention, mat_encoder, mat_embed_dim, mat_integration, **kwargs):
        super(EarlyFusionMatFramework, self).__init__(task_mode, encoder, decoder, attention)

        self.mat_encoder = mat_encoder
        self.mat_integration = mat_integration

        self.mat_feat_proj = nn.Sequential(
            nn.Conv2d(mat_encoder.discretizer.out_feat_dim, mat_embed_dim, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(mat_embed_dim, mat_embed_dim, kernel_size=1),
        )

    def forward(self, data):
        """Forward pass.

        Args:
            data (dict): [description]

        Returns:
            [type]: [description]
        """

        output = {}

        video = data["video"]  # [batch_size, frames, channels, height, width]

        # Generate material feature masks from frames.
        n_frames = video.shape[1]
        if self.mat_integration == "features":
            comb_frames = []
            for i in range(n_frames):
                mat_feat, _ = self.mat_encoder.encode_mat_feats(video[:, i])

                mat_feat = self.mat_feat_proj(mat_feat)

                # Resize features.
                mat_feat = F.interpolate(mat_feat, size=(video.shape[3], video.shape[4]), mode="nearest")

                # Stack video frames and material features.
                frame = video[:, i]
                comb_frames.append(torch.concat((frame, mat_feat), dim=1))
            video = torch.concat(comb_frames, dim=1)

        elif self.mat_integration == "change_conf":
            mat_confs = []
            for i in range(n_frames):
                _, mat_conf = self.mat_encoder.encode_mat_feats(video[:, i])
                mat_confs.append(mat_conf)

            # mat_change_conf = 1 - torch.max(torch.prod(torch.stack(mat_confs, dim=0), dim=0), dim=1)[0]
            mat_change_conf = 1 - torch.max(torch.mean(torch.stack(mat_confs, dim=0), dim=0), dim=1)[0]
            mat_change_conf = mat_change_conf**4
            # Resize features.
            mat_change_conf = F.interpolate(
                mat_change_conf.unsqueeze(1), size=(video.shape[3], video.shape[4]), mode="bilinear", align_corners=True
            )
            output["mat_change_conf"] = mat_change_conf

            # Stack video frames and material features.
            n_channels = video.shape[2]
            mat_change_frame = mat_change_conf.unsqueeze(1).repeat(1, 1, n_channels, 1, 1)
            video = torch.concat([video, mat_change_frame], dim=1)

            # Stack frames along the channel dimension.
            n_channels = video.shape[1] * video.shape[2]
            video = video.view(video.shape[0], n_channels, video.shape[3], video.shape[4])
        else:
            raise NotImplementedError

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

        output[self.task_mode] = pred
        return output
