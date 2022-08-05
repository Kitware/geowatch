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
    def __init__(self, in_channels, out_channels, dropout_rate=.2, pos_encode=False, positional_layers=None, attention_layers=1, mode='addition'):
        super(attention_unet, self).__init__()

        self.mode = mode
        if isinstance(attention_layers, int):
            self.attention_layers = list(range(1, attention_layers + 1))
        else:
            self.attention_layers = list(attention_layers)

        num_attention_layers = len(self.attention_layers)

        if not pos_encode:
            self.positional_layers = []
        elif not positional_layers:
            self.positional_layers = self.attention_layers
        else:
            self.positional_layers = positional_layers

        assert (num_attention_layers <= 8)

        if self.mode == 'concatenation':
            self.pos_embed_dim = 16
            dimension_adjustment = self.pos_embed_dim
            # self.positional_encoding = positional_encoding(dimensions=self.pos_embed_dim)
        else:
            # self.positional_encoding = positional_encoding(dimensions=self.feature_dimensions[0])
            dimension_adjustment = 0

        self.feature_dimensions = [32, 64, 128, 256, 512, 256, 128, 64, 32]
        self.attention_dimensions = [32, 64, 128, 256, 256, 128, 64, 32]

        self.dimension_adjustment = [0, 0, 0, 0, 0, 0, 0, 0, 0]

        for x in self.positional_layers:
            self.dimension_adjustment[x - 1] += dimension_adjustment
            if x < 5:
                self.dimension_adjustment[8 - x] += dimension_adjustment
        self.pos_encode = pos_encode

        self.inc = inconv(in_channels, self.feature_dimensions[0])
        #x1
        self.down1 = down_blur(self.feature_dimensions[0] + self.dimension_adjustment[0], self.feature_dimensions[1])
        #x2
        self.down2 = down_blur(self.feature_dimensions[1] + self.dimension_adjustment[1], self.feature_dimensions[2])
        #x3
        self.down3 = down_blur(self.feature_dimensions[2] + self.dimension_adjustment[2], self.feature_dimensions[3])
        #x4
        self.down4 = down_blur(self.feature_dimensions[3] + self.dimension_adjustment[3], self.feature_dimensions[3])
        #x5
        self.up1 = up(self.feature_dimensions[4] + self.dimension_adjustment[4], self.feature_dimensions[2])
        #x6
        self.up2 = up(self.feature_dimensions[3] + self.dimension_adjustment[5], self.feature_dimensions[1])
        #x7
        self.up3 = up(self.feature_dimensions[2] + self.dimension_adjustment[6], self.feature_dimensions[0])
        #x8
        self.up4 = up(self.feature_dimensions[1] + self.dimension_adjustment[7], self.feature_dimensions[0])

        self.outc = outconv(self.feature_dimensions[0], out_channels)

        self.drop = nn.Dropout(p=dropout_rate)

        self.query = nn.ModuleList()
        self.key = nn.ModuleList()
        self.value = nn.ModuleList()

        self.positional_encoding = nn.ModuleList()
        for k in range(self.attention_layers[-1]):
            self.query.append(nn.Conv2d(self.attention_dimensions[k], self.attention_dimensions[k], kernel_size=1))
            self.key.append(nn.Conv2d(self.attention_dimensions[k], self.attention_dimensions[k], kernel_size=1))
            self.value.append(nn.Conv2d(self.attention_dimensions[k], self.attention_dimensions[k], kernel_size=1))
            if pos_encode:
                if self.mode == 'concatenation':
                    self.positional_encoding.append(positional_encoding(dimensions=self.pos_embed_dim))
                else:
                    self.positional_encoding.append(positional_encoding(dimensions=self.attention_dimensions[k]))

    def forward(self, images, timestamps=None):
        # images # B x T x C x H x W

        B, T, c, h, w = images.shape
        images = images.view(B * T, c, h, w)
        x1 = self.inc(images)

        ### Attention
        if 1 in self.attention_layers:
            #B x T x 64 x H x W
            _, _, H, W = x1.shape
            Q = self.query[0](x1).view(B, T, -1, H, W)
            K = self.key[0](x1).view(B, T, -1, H, W)
            V = self.value[0](x1).view(B, T, -1, H, W)
            att_linear = torch.einsum('bpchw,bqchw->bpqhw', Q, K)
            att = torch.softmax(att_linear, dim=1)
            x1 = torch.einsum('bpqhw, bpchw -> bqchw', att, V)
            x1 = x1.contiguous().view(B * T, -1, H, W)
            if 1 in self.positional_layers:
                if timestamps is None:
                    if self.mode == 'concatenation':
                        positional_encode = torch.zeros(B * T, -1).to(x1.device)
                    else:
                        positional_encode = torch.zeros(B * T, self.attention_dimensions[0]).to(x1.device)
                else:
                    positional_encode = self.positional_encoding[0](timestamps).view(B * T, -1)

                positional_encode = positional_encode.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, w)

                if self.mode == 'concatenation':
                    x1 = torch.cat([x1, positional_encode], dim=1)
                else:
                    x1 = x1 + positional_encode

        ###
        x2 = self.drop(self.down1(x1))
        ### Attention
        if 2 in self.attention_layers:
            _, C, H, W = x2.shape
            Q = self.query[1](x2).view(B, T, -1, H, W)
            K = self.key[1](x2).view(B, T, -1, H, W)
            V = self.value[1](x2).view(B, T, -1, H, W)
            att_linear = torch.einsum('bpchw,bqchw->bpqhw', Q, K)
            att = torch.softmax(att_linear, dim=1)
            x2 = torch.einsum('bpqhw, bpchw -> bqchw', att, V)
            x2 = x2.contiguous().view(B * T, -1, H, W)

            if 2 in self.positional_layers:
                if timestamps is None:
                    if self.mode == 'concatenation':
                        positional_encode = torch.zeros(B * T, -1).to(x1.device)
                    else:
                        positional_encode = torch.zeros(B * T, self.attention_dimensions[1]).to(x1.device)
                else:
                    positional_encode = self.positional_encoding[1](timestamps).view(B * T, -1)

                positional_encode = positional_encode.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)

                if self.mode == 'concatenation':
                    x2 = torch.cat([x2, positional_encode], dim=1)
                else:
                    x2 = x2 + positional_encode
            ###

        x3 = self.drop(self.down2(x2))

        ### Attention
        if 3 in self.attention_layers:
            _, C, H, W = x3.shape
            Q = self.query[2](x3).view(B, T, -1, H, W)
            K = self.key[2](x3).view(B, T, -1, H, W)
            V = self.value[2](x3).view(B, T, -1, H, W)
            att_linear = torch.einsum('bpchw,bqchw->bpqhw', Q, K)
            att = torch.softmax(att_linear, dim=1)
            x3 = torch.einsum('bpqhw, bpchw -> bqchw', att, V)
            x3 = x3.contiguous().view(B * T, -1, H, W)
            if 3 in self.positional_layers:
                if timestamps is None:
                    if self.mode == 'concatenation':
                        positional_encode = torch.zeros(B * T, -1).to(x1.device)
                    else:
                        positional_encode = torch.zeros(B * T, self.attention_dimensions[2]).to(x1.device)
                else:
                    positional_encode = self.positional_encoding[2](timestamps).view(B * T, -1)

                positional_encode = positional_encode.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)

                if self.mode == 'concatenation':
                    x3 = torch.cat([x3, positional_encode], dim=1)
                else:
                    x3 = x3 + positional_encode
            ###

        x4 = self.drop(self.down3(x3))

        ### Attention
        if 4 in self.attention_layers:
            _, C, H, W = x4.shape
            Q = self.query[3](x4).view(B, T, -1, H, W)
            K = self.key[3](x4).view(B, T, -1, H, W)
            V = self.value[3](x4).view(B, T, -1, H, W)
            att_linear = torch.einsum('bpchw,bqchw->bpqhw', Q, K)
            att = torch.softmax(att_linear, dim=1)
            x4 = torch.einsum('bpqhw, bpchw -> bqchw', att, V)
            x4 = x4.contiguous().view(B * T, -1, H, W)
            if 4 in self.positional_layers:
                if timestamps is None:
                    if self.mode == 'concatenation':
                        positional_encode = torch.zeros(B * T, -1).to(x1.device)
                    else:
                        positional_encode = torch.zeros(B * T, self.attention_dimensions[3]).to(x1.device)
                else:
                    positional_encode = self.positional_encoding[3](timestamps).view(B * T, -1)

                positional_encode = positional_encode.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)

                if self.mode == 'concatenation':
                    x4 = torch.cat([x4, positional_encode], dim=1)
                else:
                    x4 = x4 + positional_encode

        ###
        x5 = self.drop(self.down4(x4))

        ## 5
        if 5 in self.attention_layers:
            _, C, H, W = x5.shape
            Q = self.query[4](x5).view(B, T, -1, H, W)
            K = self.key[4](x5).view(B, T, -1, H, W)
            V = self.value[4](x5).view(B, T, -1, H, W)
            att_linear = torch.einsum('bpchw,bqchw->bpqhw', Q, K)
            att = torch.softmax(att_linear, dim=1)
            x5 = torch.einsum('bpqhw, bpchw -> bqchw', att, V)
            x5 = x5.contiguous().view(B * T, -1, H, W)
            if 5 in self.positional_layers:
                if timestamps is None:
                    if self.mode == 'concatenation':
                        positional_encode = torch.zeros(B * T, -1).to(x1.device)
                    else:
                        positional_encode = torch.zeros(B * T, self.attention_dimensions[4]).to(x1.device)
                else:
                    positional_encode = self.positional_encoding[4](timestamps).view(B * T, -1)

                positional_encode = positional_encode.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)

                if self.mode == 'concatenation':
                    x5 = torch.cat([x5, positional_encode], dim=1)
                else:
                    x5 = x5 + positional_encode
            ###

        x6 = self.drop(self.up1(x5, x4))  # make sure the right residual connections connect

        ## 6
        if 6 in self.attention_layers:
            _, C, H, W = x6.shape
            Q = self.query[5](x6).view(B, T, -1, H, W)
            K = self.key[5](x6).view(B, T, -1, H, W)
            V = self.value[5](x6).view(B, T, -1, H, W)
            att_linear = torch.einsum('bpchw,bqchw->bpqhw', Q, K)
            att = torch.softmax(att_linear, dim=1)
            x6 = torch.einsum('bpqhw, bpchw -> bqchw', att, V)
            x6 = x6.contiguous().view(B * T, -1, H, W)
            if 6 in self.positional_layers:
                if timestamps is None:
                    if self.mode == 'concatenation':
                        positional_encode = torch.zeros(B * T, -1).to(x1.device)
                    else:
                        positional_encode = torch.zeros(B * T, self.attention_dimensions[5]).to(x1.device)
                else:
                    positional_encode = self.positional_encoding[5](timestamps).view(B * T, -1)

                positional_encode = positional_encode.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)

                if self.mode == 'concatenation':
                    x6 = torch.cat([x6, positional_encode], dim=1)
                else:
                    x6 = x6 + positional_encode
            ###

        x7 = self.drop(self.up2(x6, x3))

        ## 7
        if 7 in self.attention_layers:
            _, C, H, W = x7.shape
            Q = self.query[6](x7).view(B, T, -1, H, W)
            K = self.key[6](x7).view(B, T, -1, H, W)
            V = self.value[6](x7).view(B, T, -1, H, W)
            att_linear = torch.einsum('bpchw,bqchw->bpqhw', Q, K)
            att = torch.softmax(att_linear, dim=1)
            x7 = torch.einsum('bpqhw, bpchw -> bqchw', att, V)
            x7 = x7.contiguous().view(B * T, -1, H, W)
            if 7 in self.positional_layers:
                if timestamps is None:
                    if self.mode == 'concatenation':
                        positional_encode = torch.zeros(B * T, -1).to(x1.device)
                    else:
                        positional_encode = torch.zeros(B * T, self.attention_dimensions[6]).to(x1.device)
                else:
                    positional_encode = self.positional_encoding[6](timestamps).view(B * T, -1)

                positional_encode = positional_encode.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)

                if self.mode == 'concatenation':
                    x7 = torch.cat([x7, positional_encode], dim=1)
                else:
                    x7 = x7 + positional_encode
            ###

        x8 = self.drop(self.up3(x7, x2))

        ## 8
        if 8 in self.attention_layers:
            _, C, H, W = x8.shape
            Q = self.query[7](x8).view(B, T, -1, H, W)
            K = self.key[7](x8).view(B, T, -1, H, W)
            V = self.value[7](x8).view(B, T, -1, H, W)
            att_linear = torch.einsum('bpchw,bqchw->bpqhw', Q, K)
            att = torch.softmax(att_linear, dim=1)
            x8 = torch.einsum('bpqhw, bpchw -> bqchw', att, V)
            x8 = x8.contiguous().view(B * T, -1, H, W)
            if 8 in self.positional_layers:
                if timestamps is None:
                    if self.mode == 'concatenation':
                        positional_encode = torch.zeros(B * T, -1).to(x1.device)
                    else:
                        positional_encode = torch.zeros(B * T, self.attention_dimensions[7]).to(x1.device)
                else:
                    positional_encode = self.positional_encoding[7](timestamps).view(B * T, -1)

                positional_encode = positional_encode.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)

                if self.mode == 'concatenation':
                    x8 = torch.cat([x8, positional_encode], dim=1)
                else:
                    x8 = x8 + positional_encode
            ###

        x = self.up4(x8, x1)

        x = self.outc(x)

        return x.view(B, T, -1, h, w)
