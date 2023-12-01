"""

import sys
sys.path.append('/home/joncrall/code/SITS-Former/code')
from model import classification_model as clf

import liberator
lib = liberator.Liberator()
lib.add_dynamic(clf.BERTClassification)
lib.expand(['model'])
print(lib.current_sourcecode())

"""
from torch.nn.modules import LayerNorm
from torch.nn.modules.transformer import TransformerEncoder
from torch.nn.modules.transformer import TransformerEncoderLayer
import math
import torch.nn as nn
import torch


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=366):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len + 1, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float(
        ).unsqueeze(1)         # [max_len, 1]
        div_term = (torch.arange(0, d_model, 2).float() * -
                    (math.log(10000.0) / d_model)).exp()  # [d_model/2,]

        # keep pe[0,:] to zeros
        # broadcasting to [max_len, d_model/2]
        pe[1:, 0::2] = torch.sin(position * div_term)
        # broadcasting to [max_len, d_model/2]
        pe[1:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, time):
        output = torch.stack([torch.index_select(self.pe, 0, time[i, :])
                             for i in range(time.shape[0])], dim=0)
        return output       # [batch_size, seq_length, embed_dim]


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. InputEmbedding : project the input to embedding size through a lightweight 3D-CNN
        2. PositionalEncoding : adding positional information using sin/cos functions

        sum of both features are output of BERTEmbedding
    """

    def __init__(self, num_features, dropout=0.1):
        """
        :param num_features: number of input features
        :param dropout: dropout rate
        """
        super().__init__()
        channel_size = (32, 64, 256)
        kernel_size = (5, 3, 5, 3)

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1,
                      out_channels=channel_size[0],
                      kernel_size=(kernel_size[0], kernel_size[1], kernel_size[1])),
            nn.ReLU(),
            nn.BatchNorm3d(channel_size[0]),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=channel_size[0],
                      out_channels=channel_size[1],
                      kernel_size=(kernel_size[2], kernel_size[3], kernel_size[3])),
            nn.ReLU(),
            nn.BatchNorm3d(channel_size[1]),
        )

        self.linear = nn.Linear(in_features=channel_size[1] * 2,
                                out_features=channel_size[2])

        self.embed_size = channel_size[-1]
        self.position = PositionalEncoding(
            d_model=self.embed_size, max_len=366)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_sequence, doy_sequence):
        batch_size = input_sequence.size(0)
        seq_length = input_sequence.size(1)
        band_num = input_sequence.size(2)
        patch_size = input_sequence.size(3)
        first_dim = batch_size * seq_length

        obs_embed = input_sequence.view(
            first_dim, band_num, patch_size, patch_size).unsqueeze(1)
        obs_embed = self.conv1(obs_embed)
        obs_embed = self.conv2(obs_embed)
        # [batch_size*seq_length, embed_size]
        obs_embed = self.linear(obs_embed.view(first_dim, -1))
        obs_embed = obs_embed.view(batch_size, seq_length, -1)

        position_embed = self.position(doy_sequence)
        x = obs_embed + position_embed   # [batch_size, seq_length, embed_size]

        return self.dropout(x)


class BERT(nn.Module):

    def __init__(self, num_features, hidden,
                 n_layers, attn_heads, dropout=0.1):
        """
        :param num_features: number of input features
        :param hidden: hidden size of the SITS-Former model
        :param n_layers: numbers of Transformer blocks (layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        feed_forward_hidden = hidden * 4

        self.embedding = BERTEmbedding(num_features)

        encoder_layer = TransformerEncoderLayer(
            hidden, attn_heads, feed_forward_hidden, dropout)
        encoder_norm = LayerNorm(hidden)

        self.transformer_encoder = TransformerEncoder(
            encoder_layer, n_layers, encoder_norm)

    def forward(self, x, doy, mask):
        mask = mask == 0

        x = self.embedding(input_sequence=x, doy_sequence=doy)

        x = x.transpose(0, 1)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = x.transpose(0, 1)

        return x


class MulticlassClassification(nn.Module):

    def __init__(self, hidden, num_classes):
        super().__init__()
        self.linear = nn.Linear(hidden, num_classes)

    def forward(self, x, mask):
        mask = (1 - mask.unsqueeze(-1)) * 1e6
        x = x - mask        # mask invalid timesteps
        x, _ = torch.max(x, dim=1)      # max-pooling
        x = self.linear(x)
        return x


class BERTClassification(nn.Module):
    """
    Downstream task: Satellite Time Series Classification
    """

    def __init__(self, bert: BERT, num_classes):
        """
        :param bert: the BERT-Former model
        :param num_classes: number of classes to be classified
        """

        super().__init__()
        self.bert = bert
        self.classification = MulticlassClassification(
            self.bert.hidden, num_classes)

    def forward(self, x, doy, mask):
        x = self.bert(x, doy, mask)     # [batch_size, seq_length, embed_size]
        return self.classification(x, mask)
