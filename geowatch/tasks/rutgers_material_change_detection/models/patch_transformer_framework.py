import torch

from geowatch.tasks.rutgers_material_change_detection.models.base_model import BaseFramework


class PatchTransformerFramework(BaseFramework):
    def __init__(self, task_mode, encoder, decoder, **kwargs):
        super(PatchTransformerFramework, self).__init__(task_mode, encoder, decoder)

        self.first_last_feat_aux_loss = kwargs["first_last_feat_aux_loss"]

    def forward(self, data):
        video = data["video"]  # [batch_size, frames, channels, height, width]

        # Pass input into the encoder.
        feat_tokens = self.encoder(video)  # TODO: Add datetimes here.

        # Pass input into the decoder.
        if self.task_mode in ["total_bin_change", "total_sem_change"]:
            # Generate a prediction image from patches.
            height = feat_tokens[0]["og_height"]
            width = feat_tokens[0]["og_width"]
            batch_size, n_out_channels = self.decoder(feat_tokens[0]["prediction"]).shape[:2]
            device = self.decoder(feat_tokens[0]["prediction"])
            canvas = torch.zeros([batch_size, n_out_channels, height, width]).to(device)

            first_token_feats, last_token_feats = [], []
            for feat_token in feat_tokens:
                # Get prediciton.
                final_preds = self.decoder(feat_token["prediction"])

                # Fill prediction into output canvas.
                H0, W0, dH, dW = feat_token["crop_slice"]
                canvas[:, :, H0 : H0 + dH, W0 : W0 + dW] = final_preds

                if self.first_last_feat_aux_loss:
                    first_token_feats.append(feat_token["prediction"][:, 0].flatten(1))
                    last_token_feats.append(feat_token["prediction"][:, -2].flatten(1))

        elif self.task_mode in ["ss_splice_change", "ss_splice_change_index", "ss_arrow_of_time"]:
            # Generate a final prediction from all guesses.
            batch_size, n_out_channels = self.decoder(feat_tokens[0]["prediction"]).shape[:2]
            device = self.decoder(feat_tokens[0]["prediction"])
            canvas = torch.zeros([batch_size, n_out_channels]).to(device)

            first_token_feats, last_token_feats = [], []
            for feat_token in feat_tokens:
                canvas += feat_token["prediction"]

                if self.first_last_feat_aux_loss:
                    first_token_feats.append(feat_token["prediction"][:, 0].flatten(1))
                    last_token_feats.append(feat_token["prediction"][:, -2].flatten(1))
        else:
            raise NotImplementedError(
                f'Task mode "{self.task_mode}" not implemented for Patch Tranformer Framework forward method.'
            )

        output = {}
        output[self.task_mode] = canvas
        output["first_token_feat"] = first_token_feats
        output["first_token_feat"] = last_token_feats
        return output
