import torch
import torch.nn as nn

from geowatch.tasks.rutgers_material_change_detection.models.vit import ViT
from geowatch.tasks.rutgers_material_change_detection.models.pos_embedding import PositionEncoder

from geowatch.tasks.rutgers_material_change_detection.utils.util_misc import get_crop_slices


class PatchTransformerEncoder(nn.Module):
    def __init__(
        self,
        task_mode,
        in_channels,
        frame_shape,
        patch_shape,
        max_frames,
        dim=1024,
        n_heads=4,
        n_blocks=6,
        dim_linear_block=2048,
        p_dropout=0.0,
    ):
        super(PatchTransformerEncoder, self).__init__()

        # Check inputs.
        assert type(frame_shape) is int
        assert type(patch_shape) is int

        # Check that patch size matches frame size perfectly.
        assert frame_shape % patch_shape == 0, "Image dimensions must be divisible by the patch size."

        # Generate crop slices.
        self.crop_slices = get_crop_slices(frame_shape, frame_shape, patch_shape, patch_shape, mode="exact")

        # Build model.
        self.dim = dim
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.p_dropout = p_dropout
        self.task_mode = task_mode
        self.max_frames = max_frames
        self.in_channels = in_channels
        self.frame_shape = frame_shape
        self.patch_shape = patch_shape
        self.dim_linear_block = dim_linear_block
        self.build()

    def build(self):
        self.encoder = ViT(
            image_size=self.frame_shape,
            patch_size=self.patch_shape,
            num_classes=1,  # Not used so just need a real int here.
            dim=self.dim,
            depth=self.n_blocks,
            heads=self.n_heads,
            mlp_dim=self.dim_linear_block,
            pool="cls",  # just set this to generate an extra token.
            channels=self.in_channels,
            dim_head=64,  # TODO: Not sure what to set for this param.
            dropout=self.p_dropout,
            emb_dropout=self.p_dropout,
        )

    def forward(self, video, datetimes=None):
        """Forward call for PatchTransformer feature encoder.

        Args:
            video (torch.tensor): A tensor of shape [batch_size, n_frames, n_channels, height, width]
            datetimes (list(datetimes.datetimes), optional): A list of datetimes. Used for positional encoding.

        Returns:
            (torch.tensor): TODO
        """

        # Split input video into crops.
        outputs = []
        for crop_slice in self.crop_slices:
            H0, W0, dH, dW = crop_slice

            crop_video = video[:, :, :, H0 : H0 + dH, W0 : W0 + dW]
            # Rearrange crop video to shape [batch_size, n_frames, n_channels * height * width]
            # token_dim = n_channels * height * width
            crop_video = crop_video.flatten(2)

            # Refine patches into tokens of shape [n_frames+1, token_dim]
            crop_tokens = self.encoder.patch_forward(crop_video)

            crop_output = {}
            crop_output["og_height"] = self.frame_shape
            crop_output["og_width"] = self.frame_shape
            crop_output["crop_slice"] = crop_slice
            crop_output["prediction"] = crop_tokens

            outputs.append(crop_output)
        return outputs


class PatchTransformerDecoder(nn.Module):
    def __init__(self, feat_size, task_mode, n_out_channels, decoder_agg_mode, patch_size):
        super(PatchTransformerDecoder, self).__init__()
        self.feat_size = feat_size
        self.task_mode = task_mode
        self.patch_size = patch_size
        self.n_out_channels = n_out_channels
        self.decoder_agg_mode = decoder_agg_mode

        self.build()

    def build(self):
        token_dim = self.feat_size[-1]

        if self.task_mode in ["total_bin_change", "total_sem_change"]:
            out_dim = self.patch_size**2 * self.n_out_channels
            self.decoder = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, out_dim))
        elif self.task_mode in ["ss_splice_change", "ss_splice_change_index", "ss_arrow_of_time"]:
            self.decoder = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, self.n_out_channels))
        else:
            raise NotImplementedError(
                f'Task mode "{self.task_mode}" not implemented for Patch Tranformer Decoder build method.'
            )

    def forward(self, patch_output):
        """[summary]

        Args:
            patch_output (torch.tensor): A tensor of shape [batch_size, n_tokens+1, token_dim].
        Returns:
            (torch.tensor): A torch.tensor of shape [batch_size, n_channels, patch_height, patch_width]
        """

        # Get final token representation.
        if self.decoder_agg_mode == "mean_tokens":
            patch_agg = torch.max(patch_output[:, :-1], dim=1)[0].squeeze(1)  # [batch_size, token_dim]
        elif self.decoder_agg_mode == "max_tokens":
            patch_agg = torch.mean(patch_output[:, :-1], dim=1).squeeze(1)  # [batch_size, token_dim]
        elif self.decoder_agg_mode == "cls_token":
            patch_agg = patch_output[:, -1]
        else:
            raise NotImplementedError(f'Decoder input mode "{self.decoder_agg_mode}" not implemented.')

        pred = self.decoder(patch_agg)

        if self.task_mode in ["total_bin_change", "total_sem_change"]:
            pred = torch.reshape(pred, [-1, self.n_out_channels, self.patch_size, self.patch_size])
        elif self.task_mode in ["ss_splice_change", "ss_splice_change_index", "ss_arrow_of_time"]:
            pass
        else:
            raise NotImplementedError(
                f'Task mode "{self.task_mode}" not implemented for Patch Tranformer Decoder forward method.'
            )

        return pred


class PatchTransformer(nn.Module):
    def __init__(
        self,
        task_mode,
        height,
        width,
        n_channels,
        patch_length,
        max_n_frames,
        out_mode="features",
        token_dim_reduce_factor=2,
        n_heads=4,
        hidden_dim_token_factor=1,
        p_dropout=0.0,
        n_enc_layers=3,
        positional_type="positional",
        patch_seq_normalize=True,
    ):
        super(PatchTransformer, self).__init__()
        self.width = width
        self.height = height
        self.out_mode = out_mode
        self.task_mode = task_mode
        self.n_channels = n_channels
        self.patch_seq_normalize = patch_seq_normalize

        # Check that patches divde equally into image size.
        assert (
            height % patch_length == 0 and width % patch_length == 0
        ), "Image dimensions must be divisible by the patch size."

        # Create crop slices based on image height and width.
        self.crop_slices = get_crop_slices(height, width, patch_length, patch_length, mode="exact")

        # Create linear projection for reducing the dimensionality of input tokens.
        self.patch_feat_length = patch_length**2 * n_channels
        self.token_length = self.patch_feat_length // token_dim_reduce_factor
        # self.project_tokens_layer = nn.Linear(self.patch_feat_length, self.token_length)

        # Build transformer encoder.
        hidden_dim = int(hidden_dim_token_factor * self.token_length)
        encoder_layer = nn.TransformerEncoderLayer(self.patch_feat_length, n_heads, hidden_dim, p_dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_enc_layers)

        # Create positional encoding.
        self.pos_encoder = PositionEncoder(max_n_frames, self.patch_feat_length, pos_type=positional_type)

        # Create output_token
        if self.out_mode == "token_output":
            if task_mode == "total_bin_change":
                self.out_channels = 2
            else:
                raise NotImplementedError(f'Output channels for Patch Transformers task mode "{task_mode}".')

            self.output_token = nn.Embedding(1, self.patch_feat_length)

            # self.channel_comp_layer = nn.Conv2d(n_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
            self.output_resize_layer = nn.Linear(n_channels * patch_length**2, 2 * patch_length**2, bias=False)

    def feature_refinement(self, image):
        # image: [B, F, C, H, W]

        # Create canvas to paste output features.
        B, F, C, H, W = image.shape
        canvas = torch.zeros([B, F, C, H, W], dtype=image.dtype).to(image.device)

        for crop_slice in self.crop_slices:
            # Compute crop of image sequence based on crop parameters.
            h0, w0, h, w = crop_slice
            crop = image[:, :, :, h0 : (h0 + h), w0 : (w0 + w)]

            # Flatten height, width, and channel dimension.
            tokens = crop.flatten(2)

            if self.patch_seq_normalize:
                # Normalize patch.
                tokens = self.normalize_patch(tokens)

            # Linearly project features to lower dimension.
            # tokens = tokens.view(-1, self.patch_feat_length)
            # tokens = self.project_tokens_layer(tokens)
            # tokens = tokens.view(B, -1, self.token_length)

            # Add relative positinal information to tokens.
            # TODO:

            # Add output token if out_mode == 'token'
            if self.out_mode == "token":
                # TODO: Figure this out
                tokens = torch.concat((tokens, self.out_token), dim=0)

            # Pass tokens into transformer encoder.
            tokens = tokens.permute(1, 0, 2)  # [B, F, f] --> [F, B, f]
            output_tokens = self.transformer(tokens)
            output_tokens = output_tokens.permute(1, 0, 2)  # [F, B, f] --> [B, F, f]

            # # TODO: Get output prediction representation.
            # if self.out_mode == 'token':
            #     pass

            # Reshape output prediction.
            out_crop = output_tokens.view(B, F, C, h, w)

            # Put output prediction into canvas
            canvas[:, :, :, h0 : (h0 + h), w0 : (w0 + w)] = out_crop

        return canvas

    def token_output(self, video, active_frames=None):
        # image: [B, F, C, H, W]

        # Create canvas to paste output features.
        B, _, C, H, W = video.shape
        canvas = torch.zeros([B, 2, H, W], dtype=video.dtype).to(video.device)

        for crop_slice in self.crop_slices:
            # Compute crop of video sequence based on crop parameters.
            h0, w0, h, w = crop_slice
            crop = video[:, :, :, h0 : (h0 + h), w0 : (w0 + w)]

            # Flatten height, width, and channel dimension.
            tokens = crop.flatten(2)

            if self.patch_seq_normalize:
                # Normalize patch.
                tokens = self.normalize_patch(tokens, active_frames)

            # Add relative positinal information to tokens.
            tokens = self.pos_encoder(tokens)

            # Add output token if out_mode == 'token'
            output_token = self.output_token(torch.LongTensor([0] * B).to(video.device)).unsqueeze(1)
            tokens = torch.concat((tokens, output_token), dim=1)

            # Pass tokens into transformer encoder.
            tokens = tokens.permute(1, 0, 2)  # [B, F, f] --> [F, B, f]
            output_tokens = self.transformer(tokens)
            output_tokens = output_tokens.permute(1, 0, 2)  # [F, B, f] --> [B, F, f]

            # Reshape output prediction of last ouput token.
            out_crop = output_tokens[:, -1].view(B, C, h, w)

            out_crop = self.output_resize_layer(out_crop.flatten(1)).view(B, 2, h, w)

            # Put output prediction into canvas
            canvas[:, :, h0 : (h0 + h), w0 : (w0 + w)] = out_crop

        # if self.channel_comp_layer:
        #     canvas = self.channel_comp_layer(canvas)

        return canvas

    def encode_image(self, image):
        if self.out_mode == "features":
            return self.image_feature_refinement(image)
        elif self.out_mode == "token_output":
            return self.image_token_output(image)
        else:
            raise NotImplementedError(
                f'Output mode for Patch Transformer encode image not implemented for "{self.out_mode}".'
            )

    def forward(self, data):
        if self.out_mode == "features":
            raise NotImplementedError("Patch Transformer forward: Output mode featues not implemnted.")
            # return self.feature_refinement(data)
        elif self.out_mode == "token_output":
            pred = self.token_output(data["video"], data["active_frames"])
            return {self.task_mode: pred}
        else:
            raise NotImplementedError(
                f'Output mode for Patch Transformer forward not implemented for "{self.out_mode}".'
            )

    def normalize_patch(self, patch, active_frames, eta=1e-3):
        """[summary]

        Note: Only active frames are normalized.

        Args:
            patch (tensor): A tensor of shape [batch_size, n_frames, channels*height*width]

        Returns:
            TODO
        """

        # Get the number of active frames for sequence.
        assert active_frames is not None
        max_frames = (active_frames.sum(dim=1) - 1).int()

        # Compute the mean and std along the number of frames.
        B = patch.shape[0]
        means, stds = [], []
        for b in range(B):
            mean = patch[b, : max_frames[b].item()].mean(dim=1).unsqueeze(1)
            std = patch[b, : max_frames[b].item()].std(dim=1).unsqueeze(1)

            means.append(mean)
            stds.append(std)

        # Convert Means and STDs are on correct device.
        means = torch.tensor(means).unsqueeze(1).unsqueeze(1).to(patch.device)
        stds = torch.tensor(stds).unsqueeze(1).unsqueeze(1).to(patch.device)

        # Normalize the patch.
        patch = (patch - means) / (stds + eta)

        return patch


class PatchTransformer_0(nn.Module):
    def __init__(self, cfg, token_length):
        super().__init__()

        self.mode = cfg.db.mode
        self.token_length = token_length

        n_heads = cfg.db.m.n_heads
        n_enc_layers = cfg.db.m.n_enc_layers
        hidden_dim = cfg.db.m.hidden_dim
        p_dropout = cfg.db.m.p_dropout

        encoder_layer = nn.TransformerEncoderLayer(token_length, n_heads, hidden_dim, p_dropout)
        self.seq_enc = nn.TransformerEncoder(encoder_layer, n_enc_layers)

        # get positional encoder
        if cfg.db.pos_type:
            self.pos_encoder = PositionEncoder(cfg.db.max_seq_len, token_length, cfg.db.pos_type)
        else:
            self.pos_encoder = None

        if self.mode == "change":
            # add change token
            self.change_token = nn.Embedding(1, token_length)

            # add change head
            self.change_head = nn.Sequential(nn.Linear(token_length, 2))
        elif self.mode == "stage":
            self.stage_head = nn.Sequential(nn.Linear(token_length, 14))
        else:
            raise NotImplementedError(self.mode)

    def forward(self, data):
        output = {}

        seq = data["seq"].permute((1, 0, 2))

        # add positional encoding
        if self.pos_encoder:
            seq = self.pos_encoder(seq, data["date_delta"])

        if self.mode == "change":
            # add change token to sequence
            ct = self.change_token(torch.tensor(0).to(seq.device))
            ct = ct.view(1, 1, -1).repeat(1, seq.shape[1], 1)
            seq = torch.cat((seq, ct), dim=0)

            # pass sequence through model
            enc_seq = self.seq_enc(seq)

            output["enc_seq"] = enc_seq[:-1]  # dont need the classification head

            # get change prediction
            output["change_est"] = self.change_head(enc_seq[-1, :]).squeeze()
        elif self.mode == "stage":
            # pass sequence through model
            enc_seq = self.seq_enc(seq)  # [S, B, T]
            S, B, T = enc_seq.shape
            output["enc_seq"] = enc_seq

            stage_est = self.stage_head(enc_seq.permute((2, 0, 1)).flatten(1).T)  # [S*B, C]
            output["stage_est"] = stage_est.view((S, B, 14)).permute((1, 0, 2))

        else:
            raise NotImplementedError(self.mode)

        return output


if __name__ == "__main__":
    frame_size = 40
    n_channels = 4
    patch_size = 20
    n_frames = 5
    encoder = PatchTransformerEncoder("total_bin_mode", n_channels, frame_size, patch_size, n_frames)
    test_input = torch.zeros([1, n_frames, n_channels, frame_size, frame_size])
    output = encoder(test_input)
