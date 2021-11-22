import torch
import torch.nn as nn
from .unet_dropout import up, outconv, inconv, down_blur


class positional_encoding(nn.Module):
    def __init__(self, dimensions=64):
        super(positional_encoding, self).__init__()
        self.dimensions = dimensions

    def __call__(self, x):
        embedding = []
        for t in range(self.dimensions // 2):
            embedding.append(torch.sin(x / 2.**t))
            embedding.append(torch.cos(x / 2.**t))
        return torch.stack(embedding, dim=-1)


class attention_unet(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=.5, pos_encode=False, num_attention_layers=1):
        super(attention_unet, self).__init__()

        assert(num_attention_layers <= 4)
        self.num_attention_layers = num_attention_layers
        self.feature_dimensions = [64, 128, 256, 512, 1024]
        self.inc = inconv(in_channels, self.feature_dimensions[0])
        self.down1 = down_blur(self.feature_dimensions[0], self.feature_dimensions[1])
        self.down2 = down_blur(self.feature_dimensions[1], self.feature_dimensions[2])
        self.down3 = down_blur(self.feature_dimensions[2], self.feature_dimensions[3])
        self.down4 = down_blur(self.feature_dimensions[3], self.feature_dimensions[3])

        self.up1 = up(self.feature_dimensions[4], self.feature_dimensions[2])
        self.up2 = up(self.feature_dimensions[3], self.feature_dimensions[1])
        self.up3 = up(self.feature_dimensions[2], self.feature_dimensions[0])
        self.up4 = up(self.feature_dimensions[1], self.feature_dimensions[0])
        self.outc = outconv(self.feature_dimensions[0], out_channels)
        self.drop = nn.Dropout(p=dropout_rate)

        self.query = nn.ModuleList()
        self.key = nn.ModuleList()
        self.value = nn.ModuleList()
        self.positional_encoding = nn.ModuleList()
        for k in range(num_attention_layers):
            self.query.append(nn.Conv2d(self.feature_dimensions[k], self.feature_dimensions[k], kernel_size=1))
            self.key.append(nn.Conv2d(self.feature_dimensions[k], self.feature_dimensions[k], kernel_size=1))
            self.value.append(nn.Conv2d(self.feature_dimensions[k], self.feature_dimensions[k], kernel_size=1))
            if pos_encode:
                self.positional_encoding.append(positional_encoding(dimensions=self.feature_dimensions[k]))
        self.pos_encode = pos_encode

    def forward(self, images, timestamps=None):
        attention_layers = range(1, 1 + self.num_attention_layers)

        if timestamps is None:
            timestamps = torch.tensor(range(images.shape[1]))

        # images # B x T x C x H x W

        B, T, c, h, w = images.shape

        images = images.view(B * T, c, h, w)
        x1 = self.inc(images)

        ### Attention
        if 1 in attention_layers:
            #B x T x 64 x H x W
            _, C, H, W = x1.shape
            Q = self.query[0](x1).view(B, T, -1, H, W)
            K = self.key[0](x1).view(B, T, -1, H, W)
            V = self.value[0](x1).view(B, T, -1, H, W)
            att_linear = torch.einsum('bpchw,bqchw->bpqhw', Q, K)
            att = torch.softmax(att_linear, dim=1)
            x1 = torch.einsum('bpqhw, bpchw -> bqchw', att, V)
            if self.pos_encode:
                positional_encode = self.positional_encoding[0](timestamps)
                positional_encode = positional_encode.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, H, W)
                x1 = x1 + positional_encode
            x1 = x1.contiguous().view(B * T, -1, H, W)
        ###

        x2 = self.drop(self.down1(x1))
        ### Attention
        if 2 in attention_layers:
            _, C, H, W = x2.shape
            Q = self.query[1](x2).view(B, T, -1, H, W)
            K = self.key[1](x2).view(B, T, -1, H, W)
            V = self.value[1](x2).view(B, T, -1, H, W)
            att_linear = torch.einsum('bpchw,bqchw->bpqhw', Q, K)
            att = torch.softmax(att_linear, dim=1)
            x2 = torch.einsum('bpqhw, bpchw -> bqchw', att, V)
            if self.pos_encode:
                positional_encode = self.positional_encoding[1](timestamps)
                positional_encode = positional_encode.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, H, W)
                x2 = x2 + positional_encode
            x2 = x2.contiguous().view(B * T, -1, H, W)
        ###

        x3 = self.drop(self.down2(x2))

        ### Attention
        if 3 in attention_layers:
            _, C, H, W = x3.shape
            Q = self.query[2](x3).view(B, T, -1, H, W)
            K = self.key[2](x3).view(B, T, -1, H, W)
            V = self.value[2](x3).view(B, T, -1, H, W)
            att_linear = torch.einsum('bpchw,bqchw->bpqhw', Q, K)
            att = torch.softmax(att_linear, dim=1)
            x3 = torch.einsum('bpqhw, bpchw -> bqchw', att, V)
            if self.pos_encode:
                positional_encode = self.positional_encoding[2](timestamps)
                positional_encode = positional_encode.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, H, W)
                x3 = x3 + positional_encode
            x3 = x3.contiguous().view(B * T, -1, H, W)
        ###

        x4 = self.drop(self.down3(x3))

        ### Attention
        if 4 in attention_layers:
            _, C, H, W = x4.shape
            Q = self.query[3](x4).view(B, T, -1, H, W)
            K = self.key[3](x4).view(B, T, -1, H, W)
            V = self.value[3](x4).view(B, T, -1, H, W)
            att_linear = torch.einsum('bpchw,bqchw->bpqhw', Q, K)
            att = torch.softmax(att_linear, dim=1)
            x4 = torch.einsum('bpqhw, bpchw -> bqchw', att, V)
            if self.pos_encode:
                positional_encode = self.positional_encoding[3](timestamps)
                positional_encode = positional_encode.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, H, W)
                x4 = x4 + positional_encode
            x4 = x4.contiguous().view(B * T, -1, H, W)
        ###

        x5 = self.drop(self.down4(x4))
        x = self.drop(self.up1(x5, x4)) #### make sure the right residual connections connect
        x = self.drop(self.up2(x, x3))
        x = self.drop(self.up3(x, x2))
        x = self.up4(x, x1)
        x = self.outc(x)

        return x.view(B, T, -1, h, w)
