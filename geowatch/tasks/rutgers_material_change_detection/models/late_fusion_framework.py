from collections import defaultdict

import torch
import numpy as np

from geowatch.tasks.rutgers_material_change_detection.models.base_model import BaseFramework


class LateFusionFramework(BaseFramework):
    def __init__(self, task_mode, encoder, decoder, attention, sequence_model, **kwargs):
        super(LateFusionFramework, self).__init__(task_mode, encoder, decoder, attention)

        self.sequence_model = sequence_model
        cfg = kwargs["cfg"]

        # Handle material reconstruction discritzation.
        if task_mode == "ss_mat_recon":
            if cfg.loss.discretize:
                k_classes = cfg.loss.k_classes
                self.nn_totems, self.color_codes = {}, {}
                for i, feat_size in enumerate(kwargs["feat_sizes"]):
                    feat_dim = feat_size[0]
                    self.nn_totems["layer" + str(i + 1)] = torch.randn(size=[feat_dim, k_classes])
                    self.nn_totems["layer" + str(i + 1)] = (
                        self.nn_totems["layer" + str(i + 1)]
                        / torch.norm(self.nn_totems["layer" + str(i + 1)], p=2, dim=0)
                    ).to(kwargs["device"])
                    self.color_codes["layer" + str(i + 1)] = np.random.rand(k_classes, 3)
            else:
                self.nn_totems = None
        else:
            self.nn_totems = None

        # Feature differencing option.
        self.feature_difference = cfg.framework.feat_difference

    def forward(self, data):
        """Forward pass.

        Args:
            data (dict): [description]

        Returns:
            [type]: [description]
        """
        output = {}

        video = data["video"]  # [batch_size, frames, channels, height, width]

        # Separate frames of video.
        batch_size, n_frames, n_channels, height, width = video.shape
        frames = [video[:, i] for i in range(n_frames)]

        # Pass frames through encoder separately.
        feats = []
        for frame in frames:
            feat = self.encoder(frame)
            feats.append(feat)

        # Stack features per layer.
        stacked_feats = defaultdict(list)
        for feat_dict in feats:
            for key, value in feat_dict.items():
                stacked_feats[key].append(value)

        # If TRUE, compute feature difference.
        if self.feature_difference:
            for layer_name, feat_list in stacked_feats.items():
                compare_feat = feat_list[0]
                diff_feats = []
                for feat in feat_list:
                    diff_feat = feat - compare_feat
                    diff_feats.append(diff_feat)
                stacked_feats[layer_name] = diff_feats

        # TODO: Add option to add relative positional information here.

        # Pass encoder features through sequence model.
        if self.sequence_model == "mean_feats":
            ## Concatenate and compute mean features.
            for key, value in stacked_feats.items():
                stacked_feats[key] = torch.mean(torch.stack(stacked_feats[key], dim=0), dim=0)

            if self.attention:
                stacked_feats = self.attention(stacked_feats)

        elif self.sequence_model == "max_feats":
            # Concatenate and compute max features.
            for key, value in stacked_feats.items():
                stacked_feats[key] = torch.max(torch.stack(stacked_feats[key], dim=0), dim=0)[0]

            if self.attention:
                stacked_feats = self.attention(stacked_feats)

        elif self.sequence_model is None:
            pass

        elif type(self.sequence_model) is not str:
            # Concatenate and compute features.
            for layer_name, seq_name in zip(stacked_feats.keys(), self.sequence_model.keys()):
                # Pass fetures into sequence model.
                if type(self.sequence_model[seq_name]) is str:
                    stacked_feats[layer_name] = torch.max(torch.stack(stacked_feats[layer_name], dim=0), dim=0)[0]
                else:
                    seq_output = self.sequence_model[seq_name](torch.stack(stacked_feats[layer_name], dim=1))

                    # Reassemble the output tokens into OG feature shape.
                    device = stacked_feats[layer_name][0].device
                    n_out_channels = stacked_feats[layer_name][0].shape[1]
                    canvas = torch.zeros([batch_size, n_out_channels, height, width]).to(device)

                    for feat_token in seq_output:
                        # Fill prediction into output canvas.
                        H0, W0, dH, dW = feat_token["crop_slice"]
                        canvas[:, :, H0 : H0 + dH, W0 : W0 + dW] = feat_token["prediction"][:, -1].reshape(
                            [batch_size, n_out_channels, dH, dW]
                        )
                        del feat_token

                    stacked_feats[layer_name] = canvas
        else:
            raise NotImplementedError(
                f'Sequence model name "{self.sequence_model}" is not implemented for Late Fusion.'
            )

        # Discretize features.
        if self.nn_totems is not None:
            material_cluster_ids = {}
            for layer_name, layer_feats in stacked_feats.items():
                if type(layer_feats) is list:
                    totem_layer_feats, layer_cluster_ids = [], []
                    for layer_feat in layer_feats:
                        norm_values = torch.norm(layer_feat, p=2, dim=1)
                        layer_feat = layer_feat / norm_values.unsqueeze(1)
                        batch_size, n_channels, height, width = layer_feat.shape

                        layer_feat = layer_feat.permute(0, 2, 3, 1)
                        nn_totem = self.nn_totems[layer_name]

                        feat_class_sim = torch.matmul(layer_feat, nn_totem)
                        _, cluster_ids = torch.max(feat_class_sim, dim=-1)

                        layer_cluster_ids.append(feat_class_sim)

                        totem_feats = (
                            torch.index_select(nn_totem, 1, cluster_ids.flatten())
                            .reshape([n_channels, batch_size, height, width])
                            .permute(1, 0, 2, 3)
                        )
                        totem_layer_feats.append(totem_feats)
                    material_cluster_ids[layer_name] = torch.stack(layer_cluster_ids, dim=0)
                else:
                    breakpoint()
                    pass
                pass
            output["mat_cluster_ids"] = material_cluster_ids

        if self.task_mode == "ss_mat_recon":
            # Reorganize features.
            n_frames = len(stacked_feats["layer1"])
            all_frame_feats = []
            for i in range(n_frames):
                frame_feats = {}
                for layer_name in stacked_feats.keys():
                    frame_feats[layer_name] = stacked_feats[layer_name][i]
                all_frame_feats.append(frame_feats)

            # Separately recompute each frame of the video.
            if self.decoder.name == "unet":
                frame_preds = []
                for i, frame_feat in enumerate(all_frame_feats):
                    frame_pred = self.decoder(frame_feat, video[:, i])
                    frame_preds.append(frame_pred)
                pred = torch.stack(frame_preds, dim=1)
            else:
                frame_preds = []
                for i, frame_feat in enumerate(all_frame_feats):
                    frame_pred = self.decoder(frame_feat)
                    frame_preds.append(frame_pred)
                pred = torch.stack(frame_preds, dim=1)
        else:
            # Use fused features to generate prediction.
            if self.decoder.name == "unet":
                pred = self.decoder(stacked_feats, video[:, -1])
            else:
                pred = self.decoder(stacked_feats)

        output[self.task_mode] = pred
        return output
