from performer_pytorch import FastAttention
import math
import einops
from torch import nn
import ubelt as ub

try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


class FastMultiheadSelfAttention(FastAttention):
    """
    This seems like a good idea, but either I'm using it wrong or the
    C-bindings in normal attention make this lose all of its benefit.

    Ignore:
        D = 9  # embedding dimension
        H = 3   # number of heads
        B = 5   # batch size
        S = 7   # sequence length
        x = torch.rand(S, B, D)
        MultiheadSelfAttention(D, H)(x).shape
        FastMultiheadSelfAttention(D, H)(x)
        from performer_pytorch import FastAttention
        q = einops.rearrange(x, 's b (h e) -> b h s e', h=H)
        FastAttention(dim_heads=D // H, nb_features=None)(q, q, q).shape
    """

    def __init__(self, embed_dim, num_heads):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0
        dim_heads = embed_dim // num_heads
        nb_features = int(dim_heads * math.log(dim_heads))
        # nb_features = int(dim_heads * 2)
        super().__init__(
            dim_heads, nb_features=nb_features, ortho_scaling=0,
            causal=False, generalized_attention=False, kernel_fn=nn.ReLU(),
            no_projection=False)

    @profile
    def forward(self, x, key_padding_mask=None):
        # import xdev
        # xdev.embed()
        # make compatible with nn.MultiheadAttention
        # s, b, he = x.shape
        # e = self.dim_heads
        # h = self.num_heads
        # Much faster than einops
        if key_padding_mask is not None:
            raise NotImplementedError
        # q = x.contiguous().view(s, b, h, e).permute(1, 2, 0, 3)
        q = einops.rearrange(x, 's b (h e) -> b h s e', e=self.dim_heads)
        # a = FastAttention.forward(self, q, q, q)
        a = super().forward(q, q, q)
        # out = a.permute(2, 1, 0, 3).contiguous().view(s, b, he)
        out = einops.rearrange(a, 'b h s e -> s b (h e)', e=self.dim_heads)
        return out
