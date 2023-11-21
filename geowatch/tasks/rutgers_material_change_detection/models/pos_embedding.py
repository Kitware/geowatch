import numpy as np

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, pos_emb_mode, seq_length, feat_dim, freq=10000):
        super(PositionalEncoding, self).__init__()
        self.freq = freq
        self.feat_dim = feat_dim
        self.seq_length = seq_length
        self.pos_emb_mode = pos_emb_mode

        if self.pos_emb_mode == "absolute":
            self.pos_emb = self._abs_positional_embedding()
        elif self.pos_emb_mode == "relative":
            pass
        elif self.pos_emb_mode == "learned":
            meta_feat_dim = 11  # Datetime, geo-angles (2), season (4), sun angles (2),
            self.pos_emb_layers = torch.nn.Sequential(
                nn.Linear(meta_feat_dim, feat_dim),
                nn.LayerNorm(feat_dim),
                nn.PReLU(),
                nn.Linear(feat_dim, feat_dim),
            )
            pass
        else:
            raise NotImplementedError(f'Positional encoding type "{self.pos_emb_mode}" not implemented.')

    def _abs_positional_embedding(self):
        """Generate an absolute positional embedding.

        Returns:
            torch.tensor: A tensor of shape [feat_dim, seq_length]
        """
        pos_embed = torch.zeros((self.feat_dim, self.seq_length))
        dim_t = torch.div(
            1,
            torch.pow(self.freq, 2 * torch.div(torch.arange(self.feat_dim), 2, rounding_mode="trunc") / self.feat_dim),
            rounding_mode="trunc",
        )
        dim_t = torch.arange(self.seq_length).unsqueeze(0) * dim_t.unsqueeze(1)

        pos_embed[:, 0::2] = torch.sin(dim_t[:, 0::2])
        pos_embed[:, 1::2] = torch.cos(dim_t[:, 1::2])
        return pos_embed

    def _rel_positional_embedding(self, datetimes):
        breakpoint()
        pass

    def _learned_embedding(self, datetimes, metadata):
        # Format metadata.
        breakpoint()
        pass
        meta_feats = None

        # Pass metadata tensor into embedding layer.
        pos_emb = self.pos_emb_layers(meta_feats)

        return pos_emb

    def __call__(self, datetimes=None, metadata=None):
        """Generate positional features given datetime sequence.

        Args:
            datetimes (list(datetime.datetime)): A list of datetime objects describing the times when the frames were captured.

        Returns:
            torch.tensor: A tensor of shape [feat_dim, height, width]
        """

        if self.pos_emb_mode == "absolute":
            pos_emb = self.pos_emb
        elif self.pos_emb_mode == "relative":
            pos_emb = self._rel_positional_embedding(datetimes)
        elif self.pos_emb_mode == "learned":
            assert datetimes is not None
            assert metadata is not None
            pos_emb = self._learned_embedding(datetimes, metadata)
        else:
            raise NotImplementedError(f'Positional encoding type "{self.pos_emb_mode}" not implemented.')

        return pos_emb


def get_position_encoding(seq_length, token_length, freq, date_delta=None):
    # B = date_delta.shape[0]

    pos_embed = torch.zeros((token_length, seq_length))
    dim_t = torch.div(
        1,
        torch.pow(freq, 2 * torch.div(torch.arange(token_length), 2, rounding_mode="trunc") / token_length),
        rounding_mode="trunc",
    )
    # dim_t = torch.div(1, torch.pow(freq, 2 * (torch.arange(token_length) // 2) / token_length), rounding_mode='trunc')
    dim_t = torch.arange(seq_length).unsqueeze(0) * dim_t.unsqueeze(1)

    pos_embed[:, 0::2] = torch.sin(dim_t[:, 0::2])
    pos_embed[:, 1::2] = torch.cos(dim_t[:, 1::2])

    # import pdb
    # pdb.set_trace()

    # format to [S, B, T] shape
    pos_embed = pos_embed.transpose(1, 0)  # [S, T]
    # pos_embed = pos_embed.unsqueeze(1).repeat(1, B, 1)
    # pos_embed = pos_embed.to(date_delta.device)

    return pos_embed


def get_temporal_encoding(seq_length, token_length, period, date_delta):
    # B = date_delta.shape[0]
    # pos_embed = torch.zeros((seq_length, B, token_length))  # [S, B, T]

    # generate sinusoid
    # dim_t = 1 / torch.pow(period, 2 * (torch.arange(token_length) // 2) / token_length)
    # dim_t = dim_t.unsqueeze(1).unsqueeze(1)
    # dim_t = dim_t.to(date_delta.device)

    # dim_t = date_delta.permute(1, 0).unsqueeze(0) * dim_t  # [T, S, B]
    # dim_t = dim_t.permute(1, 2, 0)  # [S, B, T]

    device = date_delta.device

    B = date_delta.shape[0]

    # get constants
    two_pi = 2 * np.pi
    seq_freq = 1 / period
    token_freq = 1 / token_length
    y = torch.arange(token_length).int()

    # generate line
    slope = seq_freq * 10
    bias = -slope * seq_length / 2
    line = slope * torch.arange(seq_length) + bias  # [S]
    line = line.unsqueeze(1).repeat(1, token_length)  # [S, T]

    # send to device
    y = y.to(device)
    line = line.to(device)

    pos_embeds = []
    for b in range(B):
        x = date_delta[b].int()
        xx, yy = torch.meshgrid(x, y)

        # generate 2d sinusoid
        sinusoid = torch.sin((two_pi * seq_freq * xx) + (two_pi * token_freq * yy))  # [S, T]

        # add line
        pos_embed = sinusoid + line

        pos_embeds.append(pos_embed)

    pos_embeds = torch.stack(pos_embeds).permute(1, 0, 2)  # [S, B, T]

    return pos_embeds


class PositionEncoder(nn.Module):
    def __init__(self, seq_length, token_length, pos_type="positional"):
        super().__init__()

        self.seq_length = seq_length
        self.token_length = token_length

        if pos_type == "positional":
            self.encoding_func = get_position_encoding
            self.freq = 1e5
        elif pos_type == "temporal":
            self.freq = 365
            self.encoding_func = get_temporal_encoding
        else:
            raise NotImplementedError(pos_type)

    def forward(self, x, date_delta=None):
        # x: [B, S, T] (sequence)
        B = x.shape[0]

        # create sinusoid embedding
        pos_embed = self.encoding_func(self.seq_length, self.token_length, self.freq, date_delta=date_delta)  # [S, T]

        # Copy position embedding for as many examples are in the batch.
        pos_embed = pos_embed.unsqueeze(0).repeat(B, 1, 1).to(x.device)  # [B, S, T]

        # apply mask
        mask = x.ne(0).float()
        pos_embed *= mask

        # combine token and position encoding
        x = pos_embed + x

        return x

    def visualize(self, date_delta=None, index=0):
        pos_embed = self.encoding_func(date_delta, self.seq_length, self.token_length, self.freq)  # [S, B, T]
        pos_embed = pos_embed.detach().cpu().numpy()

        import matplotlib.pyplot as plt

        plt.imshow(pos_embed[:, index, :], cmap="viridis")
        plt.xlabel("Feature Position")
        plt.ylabel("Sequence Position")
        plt.xlim((0, self.token_length))
        plt.ylim((self.seq_length, 0))
        plt.colorbar()
        plt.show()
