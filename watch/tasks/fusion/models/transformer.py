"""
Notes:
    pip install reformer_pytorch
    pip install performer-pytorch  <- this one
"""
from torch import nn
import einops
import ubelt as ub  # NOQA


class ResidualLayer(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


class MultiheadSelfAttention(nn.MultiheadAttention):
    def forward(self, x):
        # attention returns a tuple of output and weights, so just take the
        # output
        attn_out, attn_weights = super().forward(x, x, x)
        return attn_out


try:
    from performer_pytorch import FastAttention
    class FastMultiheadSelfAttention(FastAttention):
        def __init__(self, embed_dim, num_heads):
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            assert embed_dim % num_heads == 0
            dim_heads = embed_dim // num_heads
            import math
            # nb_features = int(dim_heads * math.log(dim_heads))
            nb_features = int(dim_heads * 2)
            super().__init__(
                dim_heads, nb_features=nb_features, ortho_scaling=0,
                causal=False, generalized_attention=False, kernel_fn=nn.ReLU(),
                no_projection=False)

        def forward(self, x):
            # import xdev
            # xdev.embed()
            # make compatible with nn.MultiheadAttention
            q = einops.rearrange(x, 's b (h e) -> b h s e', e=self.dim_heads)
            a = super().forward(q, q, q)
            out = einops.rearrange(a, 'b h s e -> s b (h e)', e=self.dim_heads)
            return out
except ImportError:
    pass


def new_attention_layer(embedding_size, n_heads, attention_impl='exact'):
    """
    Example:
        >>> from watch.tasks.fusion.models.transformer import *  # NOQA
        >>> import torch
        >>> batch_size = 1
        >>> embedding_size = 4
        >>> n_heads = 2
        >>> num_tokens = 3
        >>> input_shape = (batch_size, num_tokens, embedding_size)
        >>> inputs = torch.rand(*input_shape)
        >>> layer1 = new_attention_layer(embedding_size, n_heads, 'exact')
        >>> outputs1 = layer1(inputs)
        >>> assert outputs1.shape == input_shape
        >>> # xdoctest: +REQUIRES(module:performer_pytorch)
        >>> layer2 = new_attention_layer(embedding_size, n_heads, 'performer')
        >>> outputs2 = layer2(inputs)
        >>> assert outputs2.shape == input_shape

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
    if attention_impl == 'exact':
        attention = MultiheadSelfAttention(embedding_size, n_heads)
    elif attention_impl == 'performer':
        import performer_pytorch  # NOQA
        # from performer_pytorch import SelfAttention
        # from performer_pytorch import FastAttention
        # from performer_pytorch import SelfAttention
        # attention = SelfAttention(dim=embedding_size, heads=n_heads)
        attention = FastMultiheadSelfAttention(embedding_size, n_heads)
    else:
        raise KeyError(attention_impl)

    layer = ResidualLayer(
        nn.Sequential(
            nn.LayerNorm(embedding_size),
            attention,
        ))
    return layer


def new_mlp_layer(embedding_size, dropout, **kwargs):
    """
    Example:
        >>> import torch
        >>> embedding_size = 3
        >>> batch_size = 1
        >>> layer = new_mlp_layer(embedding_size, dropout=0)
        >>> input_shape = (batch_size, embedding_size)
        >>> inputs = torch.rand(*input_shape)
        >>> outputs = layer(inputs)
        >>> assert outputs.shape == (batch_size, embedding_size)
    """
    return ResidualLayer(
        nn.Sequential(
            nn.Linear(embedding_size, embedding_size, **kwargs),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(embedding_size, embedding_size, **kwargs),
        ))


class ChannelwiseTransformerEncoderLayer(nn.Module):
    """

    Notes:
        * Currently "mode" might indicate something like a sensor or special
          computation. Each "mode" might have a differet number of "features".
          In the future this might be better specified as a dictionary that
          maps "mode"-codes to a tensor containing only the "features" for that
          mode. E.g.:

              inputs = {
                  'S2':        Tensor([B, T, H, W, 13]),
                  'WV':        Tensor([B, T, H, W, 8]),
                  'Latent':    Tensor([B, T, H, W, 512]),
                  'Materials': Tensor([B, T, H, W, 16]),
              }

        Currently these are all stacked into a B x T x M x H x W x max(F)
        and padded with zeros.

        Correction: the last statement is not correct.
        Curently F is hard coded to be F = 1 * ws * ws (where ws is the window
        size), so features are really spatial positions in a window. And
        the "width" and "height" here refer to the "number of windows"
        in the area.

    Example:
        >>> from watch.tasks.fusion.models.transformer import *  # NOQA
        >>> import torch
        >>> image_size = 128
        >>> #
        >>> ws = window_size = 32
        >>> W = H = image_size // ws
        >>> B = batch_size = 2
        >>> T = num_times = 3
        >>> M = num_modes = 13  # hack for number of features in S2
        >>> F = 1 * ws * ws # hack to use spatial positions in a windows as features
        >>> input_shape = (B, T, M, H, W, F)
        >>> x = torch.rand(*input_shape)
        >>> #
        >>> # Embedding size must be equal to F
        >>> embedding_size = F
        >>> self = ChannelwiseTransformerEncoderLayer(
        >>>     axes=[("time", "mode", "height", "width")],
        >>>     default_shape=["batch", "time", "mode", "height", "width", "feature"],
        >>>     feature_axis="feature",
        >>>     batch_axis="batch",
        >>>     embedding_size=embedding_size,
        >>>     n_heads=4
        >>> )
        >>> outputs = self(x)
        >>> assert tuple(outputs.shape) == (2, 3, 13, 4, 4, 1024)
    """
    def __init__(
        self,
        axes,
        embedding_size,
        n_heads,
        dropout=0.,
        default_shape=["batch", "time", "mode", "height", "width", "feature"],
        feature_axis="feature",
        batch_axis="batch",
        attention_impl='exact'
    ):
        super().__init__()
        self.axes = axes
        self.default_shape = default_shape
        self.feature_axis = feature_axis
        self.batch_axis = batch_axis
        self.default_shape_str = " ".join(default_shape)

        self.attention_modules = nn.ModuleDict({
            " ".join(axis): new_attention_layer(embedding_size, n_heads,
                                                attention_impl=attention_impl)
            for axis in axes
        })
        self.mlp = new_mlp_layer(embedding_size, dropout)

    def forward(self, x):
        shape_dict = dict(zip(self.default_shape, x.shape))

        previous_axial_shape = self.default_shape_str
        for axis in self.axes:
            if not isinstance(axis, (list, tuple)):
                axis = [axis]

            sequence_axes = " ".join(axis)
            batch_axes = " ".join([
                a for a in self.default_shape
                if (a == self.batch_axis or a not in axis) and a != self.feature_axis
            ])
            axial_shape = f"({sequence_axes}) ({batch_axes}) {self.feature_axis}"

            x = einops.rearrange(x, f"{previous_axial_shape} -> {axial_shape}", **shape_dict)
            x = self.attention_modules[" ".join(axis)](x)

            previous_axial_shape = axial_shape

        sequence_axes = " ".join([
            a for a in self.default_shape
            if a not in (self.batch_axis, self.feature_axis)
        ])
        axial_shape = f"({sequence_axes}) {self.batch_axis} {self.feature_axis}"

        x = einops.rearrange(x, f"{previous_axial_shape} -> {axial_shape}", **shape_dict)
        x = self.mlp(x)
        x = einops.rearrange(x, f"{axial_shape} -> {self.default_shape_str}", **shape_dict)
        return x


class FusionEncoder(nn.Module):
    """
    Primary entry point to create a feature transformer

    Example:
        >>> from watch.tasks.fusion.models.transformer import *  # NOQA
        >>> import torch
        >>> in_features = 7
        >>> model = FusionEncoder(
        >>>     in_features=in_features,
        >>>     axes=[("time", "mode", "height", "width")],
        >>>     default_shape=["batch", "time", "mode", "height", "width", "feature"],
        >>>     feature_axis="feature",
        >>>     batch_axis="batch",
        >>>     n_layers=8,
        >>>     embedding_size=256,
        >>>     n_heads=4
        >>> )
        >>> input_shape = B, T, M, H, W, F = (2, 3, 5, 2, 2, in_features)
        >>> inputs = torch.rand(*input_shape)
        >>> model(inputs)
        >>> output = model(inputs)
        >>> assert output.shape == (2, 3, 5, 2, 2, 256)
        >>> #
        >>> # Test Lazy variant
        >>> model = FusionEncoder(
        >>>     in_features=None,
        >>>     axes=[("time", "mode", "height", "width")],
        >>>     default_shape=["batch", "time", "mode", "height", "width", "feature"],
        >>>     feature_axis="feature",
        >>>     batch_axis="batch",
        >>>     n_layers=8,
        >>>     embedding_size=256,
        >>>     n_heads=4
        >>> )
        >>> inputs = torch.rand(*input_shape)
        >>> output = model(inputs)
        >>> assert output.shape == (2, 3, 5, 2, 2, 256)
    """
    def __init__(self, axes,
                 default_shape=["batch", "sequence", "feature"],
                 feature_axis="feature",
                 batch_axis="batch",
                 embedding_size=128,
                 n_layers=4,
                 n_heads=8,
                 dropout=0.0,
                 attention_impl='exact',
                 in_features=None):
        super().__init__()
        if in_features is None:
            # Use lazy linear to allow data to specify the channel dims
            first = nn.LazyLinear(embedding_size)
        else:
            first = nn.Linear(in_features=in_features, out_features=embedding_size)

        _layers = [
            ChannelwiseTransformerEncoderLayer(
                axes,
                embedding_size=embedding_size,
                n_heads=n_heads,
                dropout=dropout,
                default_shape=default_shape,
                feature_axis=feature_axis,
                batch_axis=batch_axis,
                attention_impl=attention_impl,
            )
            for _ in range(n_layers)
        ]

        self.first = first
        self.layers = nn.Sequential(*_layers)

    def forward(self, x):
        x = self.first(x)
        x = self.layers(x)
        return x


# TODO: dont define tons of functions, use a configuration dictionary
_smt_axes_basis = dict(
    joint=[("time", "mode", "height", "width")],
    stm=[("height", "width"), ("time",), ("mode",)],
    sm=[("height", "width"), ("mode",)],
    st=[("height", "width"), ("time",)],
    tm=[("time",), ("mode",)],
    s=[("height", "width")],
    t=[("time",)],
    hwtm=[("height",), ("width",), ("time",), ("mode",)],
    m=[("mode",)],
)

_encoder_size_basis = {
    'p8': dict(n_layers=8, embedding_size=128, n_heads=4),
    'n12': dict(n_layers=12, embedding_size=128, n_heads=4),
    't12': dict(n_layers=12, embedding_size=192, n_heads=4),
    't24': dict(n_layers=24, embedding_size=192, n_heads=4),
    's12': dict(n_layers=12, embedding_size=384, n_heads=8),
    's24': dict(n_layers=24, embedding_size=384, n_heads=8),
    'm24': dict(n_layers=24, embedding_size=512, n_heads=8),
    'l24': dict(n_layers=24, embedding_size=768, n_heads=8),
}


# space-mode-time transformer params
_smt_value = dict(
    default_shape=["batch", "time", "mode", "height", "width", "feature"],
    feature_axis="feature",
    batch_axis="batch",
)

encoder_configs = {}
for axes_code, axes_value in _smt_axes_basis.items():
    for size_code, size_value in _encoder_size_basis.items():
        code = f'smt_it_{axes_code}_{size_code}'
        encoder_configs[code] = ub.dict_union(
            size_value, _smt_value, dict(axes=axes_value))


# space-mode transformer params
_sm_value = dict(
    default_shape=["batch", "mode", "height", "width", "feature"],
    feature_axis="feature",
    batch_axis="batch",
)

_sm_axes_basis = {
    'joint': [("mode", "height", "width")],
    'sm': [("height", "width"), ("mode",)],
}

for axes_code, axes_value in _sm_axes_basis.items():
    for size_code, size_value in _encoder_size_basis.items():
        code = f'sm_it_{axes_code}_{size_code}'
        encoder_configs[code] = ub.dict_union(
            size_value, _sm_value, dict(axes=axes_value))

# Names of current encoder configs are:
# ['smt_it_joint_p8',
# 'smt_it_joint_n12',
# 'smt_it_joint_t12',
# 'smt_it_joint_t24',
# 'smt_it_joint_s12',
# 'smt_it_joint_s24',
# 'smt_it_joint_m24',
# 'smt_it_joint_l24',
# 'smt_it_stm_p8',
# 'smt_it_stm_n12',
# 'smt_it_stm_t12',
# 'smt_it_stm_t24',
# 'smt_it_stm_s12',
# 'smt_it_stm_s24',
# 'smt_it_stm_m24',
# 'smt_it_stm_l24',
# 'smt_it_sm_p8',
# 'smt_it_sm_n12',
# 'smt_it_sm_t12',
# 'smt_it_sm_t24',
# 'smt_it_sm_s12',
# 'smt_it_sm_s24',
# 'smt_it_sm_m24',
# 'smt_it_sm_l24',
# 'smt_it_st_p8',
# 'smt_it_st_n12',
# 'smt_it_st_t12',
# 'smt_it_st_t24',
# 'smt_it_st_s12',
# 'smt_it_st_s24',
# 'smt_it_st_m24',
# 'smt_it_st_l24',
# 'smt_it_tm_p8',
# 'smt_it_tm_n12',
# 'smt_it_tm_t12',
# 'smt_it_tm_t24',
# 'smt_it_tm_s12',
# 'smt_it_tm_s24',
# 'smt_it_tm_m24',
# 'smt_it_tm_l24',
# 'smt_it_s_p8',
# 'smt_it_s_n12',
# 'smt_it_s_t12',
# 'smt_it_s_t24',
# 'smt_it_s_s12',
# 'smt_it_s_s24',
# 'smt_it_s_m24',
# 'smt_it_s_l24',
# 'smt_it_t_p8',
# 'smt_it_t_n12',
# 'smt_it_t_t12',
# 'smt_it_t_t24',
# 'smt_it_t_s12',
# 'smt_it_t_s24',
# 'smt_it_t_m24',
# 'smt_it_t_l24',
# 'smt_it_hwtm_p8',
# 'smt_it_hwtm_n12',
# 'smt_it_hwtm_t12',
# 'smt_it_hwtm_t24',
# 'smt_it_hwtm_s12',
# 'smt_it_hwtm_s24',
# 'smt_it_hwtm_m24',
# 'smt_it_hwtm_l24',
# 'smt_it_m_p8',
# 'smt_it_m_n12',
# 'smt_it_m_t12',
# 'smt_it_m_t24',
# 'smt_it_m_s12',
# 'smt_it_m_s24',
# 'smt_it_m_m24',
# 'smt_it_m_l24',
# 'sm_it_joint_p8',
# 'sm_it_joint_n12',
# 'sm_it_joint_t12',
# 'sm_it_joint_t24',
# 'sm_it_joint_s12',
# 'sm_it_joint_s24',
# 'sm_it_joint_m24',
# 'sm_it_joint_l24',
# 'sm_it_sm_p8',
# 'sm_it_sm_n12',
# 'sm_it_sm_t12',
# 'sm_it_sm_t24',
# 'sm_it_sm_s12',
# 'sm_it_sm_s24',
# 'sm_it_sm_m24',
# 'sm_it_sm_l24']

# print('encoder_configs = {}'.format(ub.repr2(encoder_configs, nl=1)))
