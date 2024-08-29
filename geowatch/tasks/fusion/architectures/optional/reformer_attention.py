from reformer_pytorch import LSHSelfAttention
import ubelt as ub

try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


class ReformerMultiheadedSelfAttention(LSHSelfAttention):
    """
    This seems like a good idea, but either I'm using it wrong or the
    C-bindings in normal attention make this lose all of its benefit.

    Ignore:
        from geowatch.tasks.fusion.architectures.transformer import *  # NOQA
        D = 9  # embedding dimension
        H = 3   # number of heads
        B = 5   # batch size
        S = 7   # sequence length
        x = torch.rand(S, B, D)

        self = ReformerMultiheadedSelfAttention(D, H)

        MultiheadSelfAttention(D, H)(x).shape
        ReformerMultiheadedSelfAttention(D, H)(x)
        from reformer_pytorch import LSHAttention
        q = einops.rearrange(x, 's b (h e) -> b h s e', h=H)
        FastAttention(dim_heads=D // H, nb_features=None)(q, q, q).shape
    """

    def __init__(self, embed_dim, num_heads):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0
        dim_heads = embed_dim // num_heads
        self.dim_heads = dim_heads
        # nb_features = int(dim_heads * math.log(dim_heads))
        # nb_features = int(dim_heads * 2)
        super().__init__(
            dim=embed_dim, heads=num_heads, dim_head=dim_heads,
            bucket_size=64, n_hashes=8, causal=False)

    @profile
    def forward(self, x, key_padding_mask=None):
        if key_padding_mask is not None:
            raise NotImplementedError
        s, b, he = x.shape
        bsd = x.permute(1, 0, 2)
        # a = LSHSelfAttention.forward(self, bsd)
        a = super().forward(bsd)
        out = a.permute(1, 0, 2)
        return out
