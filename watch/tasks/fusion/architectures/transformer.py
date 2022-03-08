"""

Current encoder config names are:

    'smt_it_joint_p8', 'smt_it_joint_n12', 'smt_it_joint_t12', 'smt_it_joint_t24',
    'smt_it_joint_s12', 'smt_it_joint_s24', 'smt_it_joint_m24', 'smt_it_joint_l24',

    'smt_it_stm_p8', 'smt_it_stm_n12', 'smt_it_stm_t12', 'smt_it_stm_t24',
    'smt_it_stm_s12', 'smt_it_stm_s24', 'smt_it_stm_m24', 'smt_it_stm_l24',

    'smt_it_sm_p8', 'smt_it_sm_n12', 'smt_it_sm_t12', 'smt_it_sm_t24',
    'smt_it_sm_s12', 'smt_it_sm_s24', 'smt_it_sm_m24', 'smt_it_sm_l24',

    'smt_it_st_p8', 'smt_it_st_n12', 'smt_it_st_t12', 'smt_it_st_t24',
    'smt_it_st_s12', 'smt_it_st_s24', 'smt_it_st_m24', 'smt_it_st_l24',

    'smt_it_tm_p8', 'smt_it_tm_n12', 'smt_it_tm_t12', 'smt_it_tm_t24',
    'smt_it_tm_s12', 'smt_it_tm_s24', 'smt_it_tm_m24', 'smt_it_tm_l24',

    'smt_it_s_p8', 'smt_it_s_n12', 'smt_it_s_t12', 'smt_it_s_t24',
    'smt_it_s_s12', 'smt_it_s_s24', 'smt_it_s_m24', 'smt_it_s_l24',

    'smt_it_t_p8', 'smt_it_t_n12', 'smt_it_t_t12', 'smt_it_t_t24',
    'smt_it_t_s12', 'smt_it_t_s24', 'smt_it_t_m24', 'smt_it_t_l24',

    'smt_it_hwtm_p8', 'smt_it_hwtm_n12', 'smt_it_hwtm_t12', 'smt_it_hwtm_t24',
    'smt_it_hwtm_s12', 'smt_it_hwtm_s24', 'smt_it_hwtm_m24', 'smt_it_hwtm_l24',

    'smt_it_m_p8', 'smt_it_m_n12', 'smt_it_m_t12', 'smt_it_m_t24',
    'smt_it_m_s12', 'smt_it_m_s24', 'smt_it_m_m24', 'smt_it_m_l24',

    'sm_it_joint_p8', 'sm_it_joint_n12', 'sm_it_joint_t12', 'sm_it_joint_t24',
    'sm_it_joint_s12', 'sm_it_joint_s24', 'sm_it_joint_m24', 'sm_it_joint_l24',

    'sm_it_sm_p8', 'sm_it_sm_n12', 'sm_it_sm_t12', 'sm_it_sm_t24',
    'sm_it_sm_s12', 'sm_it_sm_s24', 'sm_it_sm_m24', 'sm_it_sm_l24'

Notes:
    pip install reformer_pytorch
    pip install performer-pytorch  <- this one
"""
import torch
import einops
import ubelt as ub  # NOQA
import math
from torch import nn


try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


class ResidualSequential(nn.Sequential):
    """
    A Sequential layer with a residual operation at the end
    """

    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x):
        return x + super().forward(x)


class MultiheadSelfAttention(ub.NiceRepr, torch.nn.MultiheadAttention):
    """
    Inherits from :class:`torch.nn.MultiheadAttention`

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
    """

    def __init__(self, embed_dim, num_heads, *args, **kwargs):
        super().__init__(embed_dim, num_heads, *args, **kwargs)

    def __nice__(self):
        return (
            f'embed_dim={self.embed_dim} '
            f'num_heads={self.num_heads} '
        )

    def __repr__(self):
        return super().__str__()

    def forward(self, x):
        """
        Args:
            x : of shape (seq, batch, feature)

        Returns:
            attn_out : of shape (seq, batch, feature)
        """
        # attention returns a tuple of output and weights, so just take the
        # output
        attn_out, attn_weights = super().forward(x, x, x)
        return attn_out


try:
    from performer_pytorch import FastAttention

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
        def forward(self, x):
            # import xdev
            # xdev.embed()
            # make compatible with nn.MultiheadAttention
            # s, b, he = x.shape
            # e = self.dim_heads
            # h = self.num_heads
            # Much faster than einops
            # q = x.contiguous().view(s, b, h, e).permute(1, 2, 0, 3)
            q = einops.rearrange(x, 's b (h e) -> b h s e', e=self.dim_heads)
            # a = FastAttention.forward(self, q, q, q)
            a = super().forward(q, q, q)
            # out = a.permute(2, 1, 0, 3).contiguous().view(s, b, he)
            out = einops.rearrange(a, 'b h s e -> s b (h e)', e=self.dim_heads)
            return out
except ImportError:
    pass


try:
    from reformer_pytorch import LSHSelfAttention

    class ReformerMultiheadedSelfAttention(LSHSelfAttention):
        """
        This seems like a good idea, but either I'm using it wrong or the
        C-bindings in normal attention make this lose all of its benefit.

        Ignore:
            from watch.tasks.fusion.architectures.transformer import *  # NOQA
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
        def forward(self, x):
            s, b, he = x.shape
            bsd = x.permute(1, 0, 2)
            # a = LSHSelfAttention.forward(self, bsd)
            a = super().forward(bsd)
            out = a.permute(1, 0, 2)
            return out
except ImportError:
    pass


def new_attention_layer(embedding_size, n_heads, attention_impl='exact'):
    """
    Example:
        >>> from watch.tasks.fusion.architectures.transformer import *  # NOQA
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
    """
    if attention_impl == 'exact':
        attention = MultiheadSelfAttention(embedding_size, n_heads)
    elif attention_impl == 'performer':
        import performer_pytorch  # NOQA
        # from performer_pytorch import SelfAttention
        # attention = SelfAttention(dim=embedding_size, heads=n_heads)
        attention = FastMultiheadSelfAttention(embedding_size, n_heads)
    elif attention_impl == 'reformer':
        attention = ReformerMultiheadedSelfAttention(embedding_size, n_heads)
    else:
        raise KeyError(attention_impl)

    # num_groups = num_groups_hueristic(embedding_size)
    # norm = nn.GroupNorm(num_groups=num_groups, num_channels=embedding_size)
    norm = nn.LayerNorm(embedding_size)
    layer = ResidualSequential(
        norm,
        attention,
    )
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
    return ResidualSequential(
        nn.Linear(embedding_size, embedding_size, **kwargs),
        nn.Dropout(dropout),
        nn.GELU(),
        nn.Linear(embedding_size, embedding_size, **kwargs),
    )


class ChannelwiseTransformerEncoderLayer(nn.Module):
    """

    TODO:
        - [ ] Can we resitrict how far the spatial window looks, so it only
              sees neighboring spatial regions?

    Notes:
        * Currently 'mode' might indicate something like a sensor or special
          computation. Each 'mode' might have a differet number of 'features'.
          In the future this might be better specified as a dictionary that
          maps 'mode'-codes to a tensor containing only the 'features' for that
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
        the 'width' and 'height' here refer to the 'number of windows'
        in the area.

    Example:
        >>> from watch.tasks.fusion.architectures.transformer import *  # NOQA
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
        >>> embedding_size = F   # Embedding size must be equal to F
        >>> #
        >>> # ================================
        >>> # Joint Attentions Across all Axes
        >>> self = ChannelwiseTransformerEncoderLayer(
        >>>     axes=[('time', 'mode', 'height', 'width')],
        >>>     default_shape=['batch', 'time', 'mode', 'height', 'width', 'feature'],
        >>>     feature_axis='feature',
        >>>     batch_axis='batch',
        >>>     embedding_size=embedding_size,
        >>>     n_heads=4
        >>> )
        >>> print(self)
        >>> outputs = self(x)
        >>> assert tuple(outputs.shape) == (2, 3, 13, 4, 4, 1024)
        >>> #
        >>> # ================================
        >>> # Separable Attentions Across Time, Mode, and then Space
        >>> self = ChannelwiseTransformerEncoderLayer(
        >>>     axes=[('time', 'mode'), ('height', 'width')],
        >>>     default_shape=['batch', 'time', 'mode', 'height', 'width', 'feature'],
        >>>     feature_axis='feature',
        >>>     batch_axis='batch',
        >>>     embedding_size=embedding_size,
        >>>     n_heads=4
        >>> )
        >>> print(self)
        >>> outputs = self(x)
        >>> assert tuple(outputs.shape) == (2, 3, 13, 4, 4, 1024)
        >>> #
        >>> # ================================
        >>> # Space Only Attention
        >>> self = ChannelwiseTransformerEncoderLayer(
        >>>     axes=[('height', 'width')],
        >>>     default_shape=['batch', 'time', 'mode', 'height', 'width', 'feature'],
        >>>     feature_axis='feature',
        >>>     batch_axis='batch',
        >>>     embedding_size=embedding_size,
        >>>     n_heads=4
        >>> )
        >>> print(self)
        >>> outputs = self(x)
        >>> assert tuple(outputs.shape) == (2, 3, 13, 4, 4, 1024)
    """

    def __init__(
        self,
        axes,
        embedding_size,
        n_heads,
        dropout=0.,
        default_shape=('batch', 'time', 'mode', 'height', 'width', 'feature'),
        feature_axis='feature',
        batch_axis='batch',
        attention_impl='exact'
    ):
        super().__init__()
        self.axes = axes
        self.default_shape = default_shape
        self.feature_axis = feature_axis
        self.batch_axis = batch_axis

        self.axsep = ' '

        self.default_shape_str = self.axsep.join(default_shape)

        self.attention_modules = nn.ModuleDict({
            self.axsep.join(axis): new_attention_layer(
                embedding_size, n_heads, attention_impl=attention_impl)
            for axis in axes
        })
        self.mlp = new_mlp_layer(embedding_size, dropout)

    @profile
    def forward(self, x):
        """
        Args:
            x (Tensor): of shape B, T, M, H, W, F
        """
        shape_dict = dict(zip(self.default_shape, x.shape))

        previous_axial_shape = self.default_shape_str
        for axis in self.axes:
            if not isinstance(axis, (list, tuple)):
                axis = [axis]
            sequence_axes = self.axsep.join(axis)
            attention_layer = self.attention_modules[sequence_axes]
            batch_axes = self.axsep.join([
                a for a in self.default_shape
                if (a == self.batch_axis or a not in axis) and a != self.feature_axis
            ])

            # Reshape Input to Sequence-Batch-Feature wrt to the specified axis
            # at this layer.
            axial_shape = f"({sequence_axes}) ({batch_axes}) {self.feature_axis}"
            rearrange_op = f"{previous_axial_shape} -> {axial_shape}"
            x = einops.rearrange(x, rearrange_op, **shape_dict)
            x = attention_layer(x)
            previous_axial_shape = axial_shape

        sequence_axes = self.axsep.join([
            a for a in self.default_shape
            if a not in (self.batch_axis, self.feature_axis)
        ])
        axial_shape = f"({sequence_axes}) {self.batch_axis} {self.feature_axis}"
        rearrange_op = f"{previous_axial_shape} -> {axial_shape}"
        x = einops.rearrange(x, rearrange_op, **shape_dict)
        x = self.mlp(x)

        rearrange_op = f"{axial_shape} -> {self.default_shape_str}"
        x = einops.rearrange(x, rearrange_op, **shape_dict)
        return x


class TimmEncoder:
    """

    Example:
        >>> from watch.tasks.fusion.architectures.transformer import *  # NOQA
        >>> import torch
        >>> in_features = 7
        >>> input_shape = B, T, M, H, W, F = (2, 3, 5, 2, 2, in_features)
        >>> inputs = torch.rand(*input_shape)
        >>> arch_name = 'vit_base_patch16_224'
        >>> self = TimmEncoder(arch_name)
    """

    def __init__(self, arch_name='vit_base_patch16_224', pretrained=True,
                 dropout=0.0, attention_impl='exact', in_features=None):
        import timm
        self.timm_model = timm.create_model(arch_name, pretrained=True)
        # embedding_size=128,
        # n_layers=4,
        # n_heads=8,
        self.timm_model
        timm.create_model('mobilenetv3_large_100_miil_in21k')
        import netharn as nh
        nh.OutputShapeFor(self.timm_model.patch_embed.proj)
        nh.OutputShapeFor(self.timm_model.blocks)
        nh.OutputShapeFor(self.timm_model.head)


class DeiTEncoder(nn.Module):
    """
    https://github.com/rishikksh20/ViViT-pytorch
    https://pytorch.org/tutorials/beginner/vt_tutorial.html

    Example:
        >>> from watch.tasks.fusion.architectures.transformer import *  # NOQA
        >>> import torch
        >>> in_features = 7
        >>> input_shape = B, T, M, H, W, F = (2, 3, 5, 2, 2, in_features)
        >>> inputs = torch.rand(*input_shape)
        >>> self = DeiTEncoder(in_features)
        >>> outputs = self.forward(inputs)
    """

    def __init__(self, in_features):
        super().__init__()
        deit = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
        blocks = deit.blocks
        block_in_features = blocks[0].norm1.weight.shape[0]
        self.first = nn.Linear(in_features, out_features=block_in_features)
        self.blocks = blocks
        self.in_features = in_features
        self.out_features = blocks[-1].mlp.fc2.out_features

    def forward(self, inputs):
        B, T, M, H, W, F = inputs.shape
        x = einops.rearrange(inputs, 'b t m h w f -> b (t m h w) f')
        x = self.first(x)
        x = self.blocks(x)
        outputs = einops.rearrange(x, 'b (t m h w) f -> b t m h w f', t=T, m=M, h=H, w=W)
        return outputs


class FusionEncoder(nn.Module):
    """
    Primary entry point to create a feature transformer

    Performs multiple "channelwise" (maybe rename to axil?) attention
    encodings in a row

    Example:
        >>> from watch.tasks.fusion.architectures.transformer import *  # NOQA
        >>> import torch
        >>> in_features = 7
        >>> input_shape = B, T, M, H, W, F = (2, 3, 5, 2, 2, in_features)
        >>> inputs = torch.rand(*input_shape)
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
        >>> print(model)
        >>> inputs = torch.rand(*input_shape)
        >>> output = model(inputs)
        >>> assert output.shape == (2, 3, 5, 2, 2, 256)

    Ignore:
        traced = torch.jit.trace(model, inputs)
        import timerit
        ti = timerit.Timerit(5, bestof=1, verbose=2)
        for timer in ti.reset('time'):
            model(inputs)
        for timer in ti.reset('time'):
            traced(inputs)

    Ignore:
        >>> # Get a sense of the arch size
        >>> from watch.tasks.fusion.architectures.transformer import *  # NOQA
        >>> rows = []
        >>> import netharn as nh  # NOQA
        >>> for key, config in ub.ProgIter(list(encoder_configs.items())):
        >>>     self = FusionEncoder(in_features=256, **config)
        >>>     num_params = nh.util.number_of_parameters(self)
        >>>     row = {'arch': key, 'num_params': num_params}
        >>>     row.update(config)
        >>>     print('row = {}'.format(ub.repr2(row, nl=0, sort=0)))
        >>>     rows.append(row)
        >>> import pandas as pd
        >>> data = pd.DataFrame(rows).sort_values('num_params')
        >>> print(data.to_string())

        >>> # Look at only smt configs
        >>> flags = data['axes'].apply(lambda x: x == [("height", "width"), ("time",), ("mode",)])
        >>> print(data[flags].to_string())
    """

    def __init__(self, axes,
                 default_shape=('batch', 'sequence', 'feature'),
                 feature_axis='feature',
                 batch_axis='batch',
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

        self.in_features = in_features
        self.out_features = embedding_size

        self.first = first
        self.layers = nn.Sequential(*_layers)

    def forward(self, x):
        x = self.first(x)  # TODO: make this optionally convolutional
        x = self.layers(x)
        return x


def _build_global_configs():
    """
    Previously we manually defined a bunch of functions, now it
    is defined programatically
    """
    # dont define tons of functions, use a configuration dictionary
    _smt_axes_basis = dict(
        joint=[('time', 'mode', 'height', 'width')],
        stm=[('height', 'width'), ('time',), ('mode',)],
        sm=[('height', 'width'), ('mode',)],
        st=[('height', 'width'), ('time',)],
        tm=[('time',), ('mode',)],
        s=[('height', 'width')],
        t=[('time',)],
        hwtm=[('height',), ('width',), ('time',), ('mode',)],
        m=[('mode',)],
    )

    _encoder_size_basis = {
        'p1': dict(n_layers=1, embedding_size=64, n_heads=4),
        'p2': dict(n_layers=2, embedding_size=64, n_heads=4),
        'p2w': dict(n_layers=2, embedding_size=128, n_heads=8),

        'p3': dict(n_layers=4, embedding_size=128, n_heads=4),
        'p4': dict(n_layers=4, embedding_size=128, n_heads=4),

        'p8': dict(n_layers=8, embedding_size=128, n_heads=4),
        'b8': dict(n_layers=8, embedding_size=384, n_heads=4),
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
        default_shape=['batch', 'time', 'mode', 'height', 'width', 'feature'],
        feature_axis='feature',
        batch_axis='batch',
    )

    encoder_configs = {}
    for axes_code, axes_value in _smt_axes_basis.items():
        for size_code, size_value in _encoder_size_basis.items():
            code = f'smt_it_{axes_code}_{size_code}'
            encoder_configs[code] = ub.dict_union(
                size_value, _smt_value, dict(axes=axes_value))

    # space-mode transformer params
    _sm_value = dict(
        default_shape=['batch', 'mode', 'height', 'width', 'feature'],
        feature_axis='feature',
        batch_axis='batch',
    )

    _sm_axes_basis = {
        'joint': [('mode', 'height', 'width')],
        'sm': [('height', 'width'), ('mode',)],
    }

    for axes_code, axes_value in _sm_axes_basis.items():
        for size_code, size_value in _encoder_size_basis.items():
            code = f'sm_it_{axes_code}_{size_code}'
            encoder_configs[code] = ub.dict_union(
                size_value, _sm_value, dict(axes=axes_value))
    return encoder_configs


encoder_configs = _build_global_configs()

# print('encoder_configs = {}'.format(ub.repr2(list(encoder_configs.keys(), nl=1)))
