from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDiscritizer(nn.Module):
    def __init__(self, n_classes, feat_dim, norm='l2'):
        super(ResidualDiscritizer, self).__init__()
        self.norm = norm
        self.feat_dim = feat_dim
        self.n_classes = n_classes
        self.totems = nn.parameter.Parameter(torch.randn(feat_dim, n_classes), requires_grad=True)

    def forward(self, feature):
        """Computes the residual representation between features and a set of learnable classes.

        Args:
            feature (torch.tensor): A float tensor of shape [batch_size, n_frames, n_tokens, token_dim].

        Returns:
            torch.tensor: A float tensor of shape [batch_size, n_frames, n_tokens, n_classes].
        """
        if self.norm == 'l2':
            norm_vector = torch.linalg.vector_norm(feature, ord=2, dim=3, keepdim=True)
            feature = feature / norm_vector
        elif self.norm == 'l1':
            norm_vector = torch.linalg.vector_norm(feature, ord=1, dim=3, keepdim=True)
            feature = feature / norm_vector
        elif self.norm is None:
            pass
        else:
            raise NotImplementedError(f'Norm of type "{self.norm}" not implemented.')

        totems = self.totems.reshape(1, 1, 1, self.feat_dim, self.n_classes)
        residual_feats = (feature.unsqueeze(-1) - totems) / self.n_classes

        return residual_feats


class Gumbel_Softmax(nn.Module):
    def __init__(self, n_codewords, in_feat_dim, out_feat_dim=None, init_temperature=0.01, norm=None, **kwargs):
        super(Gumbel_Softmax, self).__init__()
        self.norm = norm
        self.n_codewords = n_codewords
        self.init_temperature = init_temperature

        if out_feat_dim is not None:
            self.proj = nn.Conv2d(in_feat_dim, out_feat_dim, kernel_size=1)
            self.codebook = nn.Embedding(n_codewords, out_feat_dim)
        else:
            self.proj = None
            self.codebook = nn.Embedding(n_codewords, in_feat_dim)
            out_feat_dim = in_feat_dim
        self.out_feat_dim = out_feat_dim

        self.codebook.weight.data.normal_()

        self.id_to_rgb = torch.rand(n_codewords, 3)

    def mat_id_to_rgb(self, codework_mask):
        # codework_mask: [height, width]
        return self.id_to_rgb[codework_mask]

    def forward(self, feats, temp=None):
        # feats: [batch_size, n_channels, height, width]

        if temp is None:
            temp = self.init_temperature

        # Condense number of channels
        if self.proj:
            feats = self.proj(feats)
        batch_size, n_channels, height, width = feats.shape

        # Compute difference between features and codebook words.
        # feats: [batch_size, n_channels, height, width]
        feats = feats.flatten(2).permute(0, 2, 1).unsqueeze(2)  # [batch_size, height*width, 1, n_channels]
        codebook_weights = self.codebook.weight.unsqueeze(0).unsqueeze(0)  # [1, 1, n_codewords, feat_dim]

        ## Compute L2 norm of features and codebook.
        if self.norm == 'l2':
            feats = F.normalize(feats, p=2.0, dim=3)
            codebook_weights = F.normalize(codebook_weights, p=2.0, dim=3)
        elif self.norm is None:
            pass
        else:
            raise NotImplementedError(f'Norm mode "{self.norm}" is not implemented for Gumbel Softmax class.')

        affinity = ((feats - codebook_weights).sum(dim=-1))**2

        # Compute logits.
        # [batch_size, n_feats, n_classes, feat_dim]
        logits = torch.exp(-affinity)

        # Compute Gumbel-Softmax.
        # [batch_size, n_feats, n_classes, feat_dim]
        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=2, hard=False)

        # Compute KL divergence.
        qy = F.softmax(logits, dim=2)

        # V1: Taming Transformers Gumbel; https://github.com/CompVis/taming-transformers/blob/24268930bf1dce879235a7fddd0b2355b84d7ea6/taming/modules/vqvae/quantize.py#L191
        kld_loss = torch.sum(qy * torch.log(qy * self.n_codewords + 1e-10), dim=2).mean()

        # V2: Dalle; https://github.com/lucidrains/DALLE-pytorch/blob/289fddbef7e5a9b4c6f3f223a7505c34afe4ac67/dalle_pytorch/dalle_pytorch.py#L245
        # NOTE: Pushes optimization to uniform distribution and model does not learn.
        # log_uniform = torch.ones(self.n_codewords, device=qy.device) / self.n_codewords
        # kld_loss = F.kl_div(log_uniform.log(), qy.log(), None, None, 'batchmean', log_target=True)

        # V3: My idea to encourage confident predictions.
        # kld_loss = F.kl_div(qy, soft_one_hot, size_average=None, reduce=None, reduction='batchmean', log_target=True)

        # breakpoint()

        # Sample codebook for codewords.
        # [batch_size, n_feats, feat_dim]
        codewords = soft_one_hot.argmax(dim=2)
        sel_codewords = self.codebook(codewords)

        # Resize output features.
        sel_codewords = torch.matmul(soft_one_hot,
                                     self.codebook.weight).permute(0, 2, 1).reshape(batch_size, n_channels, height,
                                                                                    width)
        codewords = codewords.reshape([batch_size, height, width])
        mat_id_probs = soft_one_hot.permute(0, 2, 1).reshape(batch_size, self.n_codewords, height, width)

        return sel_codewords, codewords, kld_loss, mat_id_probs


class VectorQuantizer2(nn.Module):
    def __init__(self, n_codewords, in_feat_dim, out_feat_dim=None, init_temperature=0.01, norm=None, **kwargs):
        super(VectorQuantizer2, self).__init__()
        self.beta = kwargs['beta']
        self.norm = norm
        self.n_codewords = n_codewords
        self.init_temperature = init_temperature

        if out_feat_dim is not None:
            self.proj = nn.Conv2d(in_feat_dim, out_feat_dim, kernel_size=1)
            self.codebook = nn.Embedding(n_codewords, out_feat_dim)
            codeword_dim = out_feat_dim
        else:
            self.proj = None
            self.codebook = nn.Embedding(n_codewords, in_feat_dim)
            codeword_dim = in_feat_dim
            out_feat_dim = in_feat_dim
        self.out_feat_dim = out_feat_dim

        self.codebook.weight.data.uniform_(-1.0 / n_codewords, 1.0 / codeword_dim)

        self.id_to_rgb = torch.rand(n_codewords, 3)

    def mat_id_to_rgb(self, codework_mask):
        # codework_mask: [height, width]
        return self.id_to_rgb[codework_mask]

    def mat_id_to_mat_feat(self, mat_id_mask):
        # mat_id_mask: [height, width]
        return self.codebook(mat_id_mask)

    def forward(self, feats, temp=None):
        # feats: [batch_size, n_channels, height, width]

        if temp is None:
            temp = self.init_temperature

        # Condense number of channels
        if self.proj:
            feats = self.proj(feats)

        batch_size, n_channels, height, width = feats.shape
        z = feats.permute(0, 2, 3, 1).contiguous()  # [batch_size, height, width, feat_dim]
        zf = z.view(-1, n_channels)

        d = torch.sum(zf ** 2, dim=1, keepdim=True) + \
            torch.sum(self.codebook.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', zf, rearrange(self.codebook.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        mat_id_probs = F.softmax(-torch.abs(d), dim=1)
        z_q = self.codebook(min_encoding_indices).view(z.shape)

        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach() - z)**2) + torch.mean((z_q - z.detach())**2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        min_encoding_indices = min_encoding_indices.reshape(batch_size, height, width)
        mat_id_probs = mat_id_probs.reshape(batch_size, height, width, self.n_codewords).permute(0, 3, 1, 2)

        return z_q, min_encoding_indices, loss, mat_id_probs


if __name__ == '__main__':
    batch_size, n_frames, n_tokens, token_dim = 4, 5, 25, 100
    test_feats = torch.randn([batch_size, n_frames, n_tokens, token_dim])

    n_classes = 30
    norm = 'l2'
    discritizer = ResidualDiscritizer(n_classes, token_dim, norm=norm)

    output_feats = discritizer(test_feats)
