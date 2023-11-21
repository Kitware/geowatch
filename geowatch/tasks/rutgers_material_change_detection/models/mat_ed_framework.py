import torch
import torch.nn as nn


class MatED(nn.Module):
    def __init__(self, task_mode, encoder, discretizer, decoder, attention_block=None, encoder_out_layer="layer1"):
        super(MatED, self).__init__()

        self.task_mode = task_mode
        self.encoder = encoder
        self.decoder = decoder
        self.discretizer = discretizer
        self.attention_block = attention_block

    def encode_mat_feats(self, frame):
        # frame: [batch_size, n_channels, height, width]

        # Get continuous features
        if self.attention_block:
            feats = self.encoder(frame)  # [batch_size, n_channels, height, width]
            feats = self.attention_block(feats)["layer1"]
        else:
            feats = self.encoder(frame)["layer1"]  # [batch_size, n_channels, height, width]

        # Use learned dictionary to get discrete features.
        d_feats, codewords, kld_loss, mat_confs = self.discretizer(feats)

        return d_feats, mat_confs

    def mat_id_to_image(self, mat_id):
        # mat_id: [height, width]

        # Get material features from mat IDs.
        mat_feat = self.discretizer.mat_id_to_mat_feat(mat_id)

        # Resize material features.
        # from: [height, width, feat_dim]
        # to: [1, feat_dim, height, width]
        mat_feat = mat_feat.permute(2, 0, 1).unsqueeze(0)

        # Generate image from material features.
        image = self.decoder(mat_feat)

        return image

    def forward(self, data):
        output = {}

        # Get image.
        video = data["video"]  # [batch_size, 1, n_channels, height, width]
        ## Get only frame from video.
        frame = video[:, 0]

        # Get continuous features
        if self.attention_block:
            feats = self.encoder(frame)  # [batch_size, n_channels, height, width]
            feats = self.attention_block(feats)["layer1"]
        else:
            feats = self.encoder(frame)["layer1"]  # [batch_size, n_channels, height, width]

        if self.discretizer:
            # Use learned dictionary to get discrete features.
            d_feats, codewords, kld_loss, mat_confs = self.discretizer(feats)
            output["kld_loss"] = kld_loss
            output["mat_ids"] = codewords
            output["mat_conf"] = mat_confs
        else:
            d_feats = feats

        # Generate prediction with decoder.
        pred = self.decoder(d_feats)

        output[self.task_mode] = pred.unsqueeze(1)

        return output


class MTMatED(nn.Module):
    def __init__(self, task_mode, encoder, discretizer, decoder, attention_block=None, encoder_out_layer="layer1"):
        super(MTMatED, self).__init__()

        self.task_mode = task_mode
        self.encoder = encoder
        self.decoder = decoder
        self.discretizer = discretizer
        self.attention_block = attention_block

    def encode_mat_feats(self, frame):
        # frame: [batch_size, n_channels, height, width]

        # Get continuous features
        if self.attention_block:
            feats = self.encoder(frame)  # [batch_size, n_channels, height, width]
            feats = self.attention_block(feats)["layer1"]
        else:
            feats = self.encoder(frame)["layer1"]  # [batch_size, n_channels, height, width]

        # Use learned dictionary to get discrete features.
        d_feats, codewords, kld_loss, mat_confs = self.discretizer(feats)

        return d_feats, mat_confs

    def encode(self, frame):
        # Get continuous features
        if self.attention_block:
            feats = self.encoder(frame)  # [batch_size, n_channels, height, width]
            feats = self.attention_block(feats)["layer1"]
        else:
            feats = self.encoder(frame)["layer1"]  # [batch_size, n_channels, height, width]
        return feats

    def mat_id_to_image(self, mat_id):
        # mat_id: [height, width]

        # Get material features from mat IDs.
        mat_feat = self.discretizer.mat_id_to_mat_feat(mat_id)

        # Resize material features.
        # from: [height, width, feat_dim]
        # to: [1, feat_dim, height, width]
        mat_feat = mat_feat.permute(2, 0, 1).unsqueeze(0)

        # Generate image from material features.
        image = self.decoder(mat_feat)

        return image

    def forward(self, data):
        output = {}

        # Get image.
        video = data["video"]  # [batch_size, n_frames, n_channels, height, width]
        n_frames = video.shape[1]

        preds, mat_masks, mat_confs = [], [], []
        kld_losses = 0
        for i in range(n_frames):
            frame = video[:, i]
            feats = self.encode(frame)

            if self.discretizer:
                # Use learned dictionary to get discrete features.
                d_feats, codewords, kld_loss, mat_conf = self.discretizer(feats)
                kld_losses += kld_loss
                mat_masks.append(codewords)
                mat_confs.append(mat_conf)
            else:
                d_feats = feats

            # Generate prediction with decoder.
            pred = self.decoder(d_feats)
            preds.append(pred)

        output = {}
        if self.discretizer:
            output["kld_loss"] = kld_losses / n_frames
            output["mat_ids"] = torch.stack(mat_masks, dim=1)
            output["mat_conf"] = torch.stack(mat_confs, dim=1)

            # Compute material consistency loss.
            output["mat_consistency_loss"] = (
                (output["mat_conf"].mean(dim=2, keepdim=True) - output["mat_conf"]).std(dim=1).sum(1).mean()
            )

        output[self.task_mode] = torch.stack(preds, dim=1)
        return output
