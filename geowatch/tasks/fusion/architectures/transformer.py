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

from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

import einops
from einops import rearrange, repeat

import ubelt as ub  # NOQA
import math

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


class ResidualAttentionSequential(ResidualSequential):
    """
    Special case of ResidualSequential to support masking
    """

    def __init__(self, norm, attention):
        super().__init__(norm, attention)

    def forward(self, x, key_padding_mask=None):
        h = x
        h = self[0](h)
        h = self[1](h, key_padding_mask=key_padding_mask)
        return x + h


def assert_allclose(a, b, rtol=1e-05, atol=1e-08):
    """
    TODO: integrate with :func:`kwcoco.coco_sql_dataset.assert_dsets_allclose`.

    Add to kwarray

    Ignore:
        >>> from geowatch.tasks.fusion.architectures.transformer import *  # NOQA
        >>> import pytest
        >>> a = np.random.rand(1, 2, 3)
        >>> b = a + 0
        >>> assert_allclose(a, b)
        >>> b = np.random.rand(1, 2, 3)
        >>> with pytest.raises(AssertionError):
        >>>     assert_allclose(a, b)
        >>> b = a.copy()
        >>> b.ravel()[0] += 1
        >>> with pytest.raises(AssertionError):
        >>>     assert_allclose(a, b)
    """
    a_shape = a.shape
    b_shape = b.shape
    if len(b_shape) != len(a_shape):
        raise AssertionError(f'len(a.shape:={a_shape}) != len(b.shape:={b.shape})')
    if b_shape != a_shape:
        raise AssertionError(f'a.shape:={a_shape} != b.shape:={b.shape}')
    import kwarray
    import numpy as np
    a = kwarray.ArrayAPI.numpy(a)
    b = kwarray.ArrayAPI.numpy(b)
    flag = np.allclose(a, b, rtol=rtol, atol=atol)
    if flag:
        ...
    else:
        impl = kwarray.ArrayAPI.coerce(a)
        flags = np.isclose(a, b)
        num_close = flags.sum()
        num_total = impl.numel(flags)
        num_not_close = num_total - num_close
        a_stats = kwarray.stats_dict(a)
        b_stats = kwarray.stats_dict(b)
        msg = ub.codeblock(
            f'''
            Failed closeness check

            Found not close entries: {num_not_close} / {num_total}
            a_stats = {ub.urepr(a_stats, nl=0, precision=4)}
            b_stats = {ub.urepr(b_stats, nl=0, precision=4)}
            ''')
        raise AssertionError(msg)


class MultiheadSelfAttention(torch.nn.MultiheadAttention):
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

    CommandLine:
        xdoctest -m geowatch.tasks.fusion.architectures.transformer MultiheadSelfAttention

    Example:
        >>> from geowatch.tasks.fusion.architectures.transformer import *  # NOQA
        >>> self = MultiheadSelfAttention(4, 1).eval()
        >>> S, B, F = (7, 3, 4)
        >>> x  = (torch.rand(S, B, F) * 10).round()
        >>> # Results should be independent of the batch dim
        >>> y  = self.forward(x)
        >>> y0 = self.forward(x[:, 0:1, :])
        >>> y1 = self.forward(x[:, 1:2, :])
        >>> y2 = self.forward(x[:, 2:3, :])
        >>> assert_allclose(y[:, 0:1, :], y0, rtol=1e-3, atol=1e-6)
        >>> assert_allclose(y[:, 1:2, :], y1, rtol=1e-3, atol=1e-6)
        >>> assert_allclose(y[:, 2:3, :], y2, rtol=1e-3, atol=1e-6)

        >>> key_padding_mask = torch.rand(B, S) > 0.5
        >>> masked_result = self.forward(x, key_padding_mask=key_padding_mask)
    """

    def __init__(self, embed_dim, num_heads, *args, **kwargs):
        super().__init__(embed_dim, num_heads, *args, **kwargs)

    def forward(self, x, key_padding_mask=None):
        """
        Args:
            x (Tensor) : of shape (seq, batch, feature)

            key_padding_mask (Tensor) : of shape (batch, seq).
                A value of True means we will **ignore** the token.

        Returns:
            attn_out : of shape (seq, batch, feature)
        """
        # attention returns a tuple of output and weights, so just take the
        # output
        outs = super().forward(
            query=x, key=x, value=x, key_padding_mask=key_padding_mask)
        attn_out, attn_weights = outs
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
except ImportError:
    pass


try:
    from reformer_pytorch import LSHSelfAttention

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
except ImportError:
    pass


def new_attention_layer(embedding_size, n_heads, attention_impl='exact', **kwargs):
    """
    Example:
        >>> from geowatch.tasks.fusion.architectures.transformer import *  # NOQA
        >>> import torch
        >>> batch_size = 1
        >>> embedding_size = 4
        >>> n_heads = 2
        >>> num_tokens = 3
        >>> input_shape = (num_tokens, batch_size, embedding_size)
        >>> inputs = torch.rand(*input_shape)
        >>> layer1 = new_attention_layer(embedding_size, n_heads, 'exact')
        >>> outputs1 = layer1(inputs)
        >>> assert outputs1.shape == input_shape
        >>> # xdoctest: +REQUIRES(module:performer_pytorch)
        >>> layer2 = new_attention_layer(embedding_size, n_heads, 'performer')
        >>> outputs2 = layer2(inputs)
        >>> assert outputs2.shape == input_shape

    Example:
        >>> # Test with a mask
        >>> from geowatch.tasks.fusion.architectures.transformer import *  # NOQA
        >>> import torch
        >>> batch_size = 1
        >>> embedding_size = 4
        >>> n_heads = 2
        >>> num_tokens = 3
        >>> input_shape = (num_tokens, batch_size, embedding_size)
        >>> inputs = torch.rand(*input_shape)
        >>> key_padding_mask = torch.rand(batch_size, num_tokens) > 0.5
        >>> layer1 = new_attention_layer(embedding_size, n_heads, 'exact')
        >>> outputs1 = layer1(inputs, key_padding_mask=key_padding_mask)
    """
    if attention_impl == 'exact':
        attention = MultiheadSelfAttention(embedding_size, n_heads, **kwargs)
    elif attention_impl == 'performer':
        import performer_pytorch  # NOQA
        # from performer_pytorch import SelfAttention
        # attention = SelfAttention(dim=embedding_size, heads=n_heads)
        attention = FastMultiheadSelfAttention(embedding_size, n_heads, **kwargs)
    elif attention_impl == 'reformer':
        attention = ReformerMultiheadedSelfAttention(embedding_size, n_heads, **kwargs)
    else:
        raise KeyError(attention_impl)

    # num_groups = num_groups_hueristic(embedding_size)
    # norm = nn.GroupNorm(num_groups=num_groups, num_channels=embedding_size)
    norm = nn.LayerNorm(embedding_size)
    layer = ResidualAttentionSequential(
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

        - [ ] Flatten tokens completely and have a mask that indicates
              what tokens are allowed to see each other in each step

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
        >>> from geowatch.tasks.fusion.architectures.transformer import *  # NOQA
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
        attention_impl='exact',
        attention_kwargs=None,
    ):

        if attention_kwargs is None:
            attention_kwargs = {}

        super().__init__()

        self.axes = axes
        self.default_shape = default_shape
        self.feature_axis = feature_axis
        self.batch_axis = batch_axis

        self.axsep = ' '

        self.default_mask_shape = [s for s in self.default_shape
                                   if s != self.feature_axis]

        self.default_shape_str = self.axsep.join(default_shape)
        self.default_mask_shape_str = self.axsep.join(self.default_mask_shape)

        self.attention_modules = nn.ModuleDict({
            self.axsep.join(axis): new_attention_layer(
                embedding_size, n_heads,
                attention_impl=attention_impl,
                **attention_kwargs,
            )
            for axis in axes
        })
        self.mlp = new_mlp_layer(embedding_size, dropout)

    @profile
    def forward(self, inputs, flat_coordinates=None, key_padding_mask=None):
        r"""
        Args:
            x (Tensor):
                of shape B, T, M, H, W, F if flat_coordinates is unspecified
                otherwise it should be of shape N, F where N is the total
                number of tokens

            flat_coordinates (Dict[str, Tensor]):
                the time, mode, height, and width coordinate of each token
                if specified batches are unsupported

            key_padding_mask (Tensor):
                of shape B, T, M, H, W if flat_coordinates is unspecified
                otherwise should be of shape N. A True value means ignore the
                token.

        CommandLine:
            xdoctest -m geowatch.tasks.fusion.architectures.transformer ChannelwiseTransformerEncoderLayer.forward

        Example:
            >>> # Test that coordinate aware implementation exactly reproduces aligned variant
            >>> from geowatch.tasks.fusion.architectures.transformer import *  # NOQA
            >>> import numpy as np
            >>> F = embedding_size = 4
            >>> B, T, M, H, W = 1, 3, 5, 7, 11
            >>> aligned_inputs = aligned_x = (torch.rand(B, T, M, H, W, F) * 100).round() / 10
            >>> key_padding_mask = torch.rand(B, T, M, H, W) > 0.9
            >>> flat_inputs = flat_x = aligned_inputs.view(-1, embedding_size)
            >>> flat_kpm = key_padding_mask.view(-1)
            >>> #inputs = flat_inputs
            >>> flat_coordinates = None
            >>> inputs = aligned_inputs
            >>> # Test that coordinate-aware flat attention works
            >>> t_coords, m_coords, h_coords, w_coords = np.meshgrid(np.arange(T), np.arange(M), np.arange(H), np.arange(W), indexing='ij')
            >>> flat_coordinates = {
            >>>     'time':   t_coords.ravel(),
            >>>     'mode':   m_coords.ravel(),
            >>>     'height': h_coords.ravel(),
            >>>     'width':  w_coords.ravel(),
            >>> }
            >>> flat_coordinates = ub.map_vals(torch.from_numpy, flat_coordinates)
            >>> self = ChannelwiseTransformerEncoderLayer(
            >>>     #axes=[('height', 'width'), ('time',)],
            >>>     axes=[('time',)],
            >>>     default_shape=['batch', 'time', 'mode', 'height', 'width', 'feature'],
            >>>     feature_axis='feature',
            >>>     batch_axis='batch',
            >>>     embedding_size=embedding_size,
            >>>     n_heads=1
            >>> )
            >>> self = self.eval()
            >>> with torch.set_grad_enabled(False):
            >>>     print('----')
            >>>     flat_y = self.forward(flat_inputs, flat_coordinates)
            >>>     print('----')
            >>>     aligned_y = self.forward(aligned_inputs)
            >>>     print('----')
            >>>     aligned_y_mask = self.forward(aligned_inputs, key_padding_mask=key_padding_mask)
            >>>     print('----')
            >>>     flat_y_mask = self.forward(flat_inputs, flat_coordinates, key_padding_mask=flat_kpm)
            >>> print('----====-')
            >>> recon_y1 = aligned_y.view(-1, embedding_size)
            >>> recon_y1_mask = aligned_y_mask.view(-1, embedding_size)
            >>> print('flat_y=\n{!r}'.format(flat_y))
            >>> print('recon_y1=\n{!r}'.format(recon_y1))
            >>> abs_diff = (flat_y - recon_y1).abs().max()
            >>> print('abs_diff = {!r}'.format(abs_diff))
            >>> assert abs_diff < 1e-5
            >>> #
            >>> flat_y_mask.nan_to_num_()
            >>> recon_y1_mask.nan_to_num_()
            >>> abs_diff_mask = (flat_y_mask - recon_y1_mask).abs().max()
            >>> print('abs_diff_mask = {!r}'.format(abs_diff_mask))
            >>> assert abs_diff_mask < 1e-5
            >>> #flags = torch.isclose(flat_y, recon_y1)
            >>> #assert flags.all()
        """
        shape_dict = dict(zip(self.default_shape, inputs.shape))
        mask_shape_dict = ub.udict(shape_dict) - {self.feature_axis}

        if flat_coordinates:
            flat_x = inputs
        else:
            x = inputs

        previous_axial_shape = self.default_shape_str
        # previous_mask_axial_shape = self.axsep.join(self.default_shape[:-1])

        for axis in self.axes:
            if not isinstance(axis, (list, tuple)):
                axis = [axis]
            sequence_axes = self.axsep.join(axis)
            attention_layer = self.attention_modules[sequence_axes]

            if flat_coordinates is None:
                # Fast axis aligned method
                batch_axes = self.axsep.join([
                    a for a in self.default_shape
                    if (a == self.batch_axis or a not in axis) and a != self.feature_axis
                ])

                # Reshape Input to Sequence-Batch-Feature wrt to the specified axis
                # at this layer.
                axial_shape = f"({sequence_axes}) ({batch_axes}) {self.feature_axis}"
                rearrange_op = f"{previous_axial_shape} -> {axial_shape}"
                x = einops.rearrange(x, rearrange_op, **shape_dict)

                if key_padding_mask is not None:
                    mask_axial_shape = f"({batch_axes}) ({sequence_axes})"
                    mask_rearrange_op = f"{self.default_mask_shape_str} -> {mask_axial_shape}"
                    kpm = einops.rearrange(key_padding_mask, mask_rearrange_op, **mask_shape_dict)
                    # previous_mask_axial_shape = mask_axial_shape
                    y = attention_layer(x, key_padding_mask=kpm)
                else:
                    kpm = None
                    y = attention_layer(x)
                x = y
                previous_axial_shape = axial_shape
            else:
                # Batch size must be 1 in this mode.
                import numpy as np
                # New generalized coordinate method
                batch_coords = ub.dict_diff(flat_coordinates, axis)
                # axial_coords = ub.dict_subset(flat_coordinates, axis)

                # For each dimension, build groups of indices where only
                # one coordinate is varied in that dimension (e.g.
                # the same mode,x/y position over different times.
                groupids = torch.stack(list(batch_coords.values())).T.contiguous()
                groupids_numpy = groupids.numpy()
                arr = groupids_numpy
                dtype_view = np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))
                arr_view = arr.view(dtype_view)
                groupids_bytes = [r.tobytes() for r in arr_view]
                group_to_idxs = ub.group_items(range(len(groupids_bytes)), groupids_bytes)

                if key_padding_mask is None:
                    flat_kpm = None
                else:
                    flat_kpm = key_padding_mask.view(-1)

                # TODO: can we build a mask and do masked attention?
                flat_y = torch.empty_like(flat_x)
                for groupid, idxs in group_to_idxs.items():
                    x_part = flat_x[idxs]
                    x_attn = x_part[:, None, :]
                    if flat_kpm is None:
                        y_attn = attention_layer(x_attn)
                    else:
                        kpm_part = flat_kpm[idxs]
                        kpm = kpm_part[None, :]
                        y_attn = attention_layer(x_attn, key_padding_mask=kpm)
                    y_part = y_attn[:, 0, :]
                    flat_y[idxs] = y_part
                flat_x = flat_y

        if flat_coordinates is None:
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
            outputs = x
        else:
            flat_x = self.mlp(flat_x)
            outputs = flat_x
        return outputs


class TimmEncoder:
    """

    Example:
        >>> # xdoctest: +REQUIRES(module:timm)
        >>> from geowatch.tasks.fusion.architectures.transformer import *  # NOQA
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


class MM_VITEncoder(nn.Module):
    """
    mmsegmentation variant of VIT

    Needs 768 features.

    Notes:
        https://github.com/open-mmlab/mmsegmentation/tree/master/configs/vit

    Results:
        # 1
        https://github.com/open-mmlab/mmsegmentation/tree/master/configs/vit#ade20k
        https://github.com/open-mmlab/mmsegmentation/blob/master/configs/vit/upernet_vit-b16_mln_512x512_80k_ade20k.py
        https://github.com/open-mmlab/mmsegmentation/blob/master/configs/_base_/models/upernet_vit-b16_ln_mln.py


    Ignore:
        >>> from mmseg.models.backbones import vit
        >>> from geowatch.tasks.fusion.architectures.transformer import *  # NOQA
        >>> self = MM_VITEncoder()
        >>> x = torch.rand(2, 3, 768)
        >>> self.forward(x)
    """

    def __init__(self):
        super().__init__()
        from mmseg.models.backbones.vit import VisionTransformer
        kwargs = dict(
            img_size=(512, 512),
            patch_size=16,
            in_channels=3,
            embed_dims=768,
            num_layers=12,
            num_heads=12,
            mlp_ratio=4,
            out_indices=(2, 5, 8, 11),
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            with_cls_token=True,
            norm_cfg=dict(type='LN', eps=1e-6),
            act_cfg=dict(type='GELU'),
            norm_eval=False,
            interpolate_mode='bicubic')
        vit_model = VisionTransformer(**kwargs)
        # We only need the encoder
        self.layers = vit_model.layers
        self.initialize_from_pretrained()
        self.in_features = self.layers[0].ln1.weight.shape[0]
        self.out_features = self.layers[-1].ffn.layers[1].out_features

    def initialize_from_pretrained(self):
        # pretrained_fpath = ub.grabdata('https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_vit-b16_mln_512x512_80k_ade20k/upernet_vit-b16_mln_512x512_80k_ade20k_20210624_130547-0403cee1.pth')
        # FIXME: Having this import here breaks torch.package
        # not exactly sure why
        # from geowatch.tasks.fusion.fit import coerce_initializer
        # initializer = coerce_initializer(pretrained_fpath)
        # info = initializer.forward(self, verbose=0)  # NOQA
        ...

    def forward(self, x):
        orig_shape = x.shape
        x = x.view(x.shape[0], -1, x.shape[1])
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # if i == len(self.layers) - 1:
            #     if self.final_norm:
            #         x = self.norm1(x)
            # if i in self.out_indices:
            #     if self.with_cls_token:
            #         # Remove class token and reshape token for decoder head
            #         out = x[:, 1:]
            #     else:
            #         out = x
            #     B, _, C = out.shape
            #     out = out.reshape(B, hw_shape[0], hw_shape[1],
            #                       C).permute(0, 3, 1, 2).contiguous()
            #     if self.output_cls_token:
            #         out = [out, x[:, 0]]
            #     outs.append(out)
        x = x.view(*orig_shape[0], x.shape[-1])
        return x


class DeiTEncoder(nn.Module):
    """
    https://github.com/rishikksh20/ViViT-pytorch
    https://pytorch.org/tutorials/beginner/vt_tutorial.html

    Example:
        >>> # xdoctest: +REQUIRES(module:timm)
        >>> from geowatch.tasks.fusion.architectures.transformer import *  # NOQA
        >>> import torch
        >>> in_features = 7
        >>> input_shape = B, T, M, H, W, F = (2, 3, 5, 2, 2, in_features)
        >>> inputs = torch.rand(*input_shape)
        >>> self = DeiTEncoder(in_features, pretrained=False)
        >>> outputs = self.forward(inputs)
    """

    def __init__(self, in_features, pretrained=True):
        super().__init__()
        deit = torch.hub.load('facebookresearch/deit:main',
                              'deit_base_patch16_224', pretrained=pretrained)
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


class PerceiverEncoder(nn.Module):
    """
    https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_io.py

    Example:
        >>> # xdoctest: +REQUIRES(module:perceiver_pytorch)
        >>> from geowatch.tasks.fusion.architectures.transformer import PerceiverEncoder  # NOQA
        >>> import torch
        >>> B, T, M, H, W, F = 1, 2, 3, 5, 8, 13
        >>> self = PerceiverEncoder(F, dropout=0.1)
        >>> inputs = torch.rand(B, T, M, H, W, F)
        >>> outputs = self(inputs)
        >>> assert outputs.shape == (B, T, M, H, W, F)
    """

    def __init__(self, in_features, depth=4, dropout=0.0):
        super().__init__()
        import perceiver_pytorch as perceiver
        # No dropout in perceiver? Perform it on input tokens.
        self.dropout = nn.Dropout(dropout)
        self.perceiver = perceiver.PerceiverIO(
            depth=depth,
            dim=in_features,
            queries_dim=in_features,
            num_latents=512,
            latent_dim=256,
            cross_heads=1,
            latent_heads=8,
            cross_dim_head=64,
            latent_dim_head=64,
            weight_tie_layers=False,
            decoder_ff=False,
            logits_dim=None,
        )
        self.in_features = in_features
        self.out_features = in_features

    def forward(self, inputs):
        B, T, M, H, W, F = inputs.shape
        x = einops.rearrange(inputs, 'b t m h w f -> b (t m h w) f')
        x = self.dropout(x)
        x = self.perceiver(x, queries=x)
        outputs = einops.rearrange(x, 'b (t m h w) f -> b t m h w f', t=T, m=M, h=H, w=W)
        return outputs


class FusionEncoder(nn.Module):
    """
    Primary entry point to create a feature transformer

    Performs multiple "channelwise" (maybe rename to axil?) attention
    encodings in a row

    Example:
        >>> from geowatch.tasks.fusion.architectures.transformer import *  # NOQA
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
        >>> from geowatch.tasks.fusion.architectures.transformer import *  # NOQA
        >>> rows = []
        >>> import netharn as nh  # NOQA
        >>> for key, config in ub.ProgIter(list(encoder_configs.items())):
        >>>     self = FusionEncoder(in_features=256, **config)
        >>>     num_params = nh.util.number_of_parameters(self)
        >>>     row = {'arch': key, 'num_params': num_params}
        >>>     row.update(config)
        >>>     print('row = {}'.format(ub.urepr(row, nl=0, sort=0)))
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
                 attention_kwargs=dict(),
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
                attention_kwargs=attention_kwargs,
            )
            for _ in range(n_layers)
        ]

        self.in_features = in_features
        self.out_features = embedding_size

        self.first = first
        self.layers = nn.Sequential(*_layers)

    def forward(self, x, flat_coordinates=None, key_padding_mask=None, mask=None):
        if mask is not None:
            key_padding_mask = mask
        x = self.first(x)
        for layer in self.layers:
            # Can't use sequentail because we need extra args
            x = layer(x, flat_coordinates=flat_coordinates,
                      key_padding_mask=key_padding_mask)
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

        'p3': dict(n_layers=3, embedding_size=128, n_heads=4),
        'p4': dict(n_layers=4, embedding_size=128, n_heads=4),

        'p8': dict(n_layers=8, embedding_size=128, n_heads=4),
        'p16': dict(n_layers=16, embedding_size=128, n_heads=4),
        'p24': dict(n_layers=24, embedding_size=128, n_heads=4),
        'p32': dict(n_layers=32, embedding_size=128, n_heads=4),

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

# print('encoder_configs = {}'.format(ub.urepr(list(encoder_configs.keys(), nl=1)))

# ========================================
# Below is an implementation of the transformer architecture that uses the
# same base components and follows the same patterns as the perceiver. This
# is a 1-to-1 drop-in replacement for perceiver models (minus latent_dim/num_latents options).
# ========================================


def default(val, d):
    return val if val is not None else d


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

# helper classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if context_dim is not None else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if self.norm_context is not None:
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, output_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        output_dim = default(output_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, output_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class BackboneEncoderDecoder:
    pass


class TransformerEncoderDecoder(nn.Module, BackboneEncoderDecoder):
    def __init__(
        self,
        encoder_depth: int = 2,
        decoder_depth: int = 1,
        dim: int = 128,
        queries_dim: int = 96,
        logits_dim: int = 32,
        decode_cross_every: int = 1,
        cross_heads: int = 1,
        latent_heads: int = 8,
        cross_dim_head: int = 64,
        latent_dim_head: int = 64,
        weight_tie_layers: bool = False,
    ):
        super().__init__()

        self.has_decoder = (decoder_depth > 0) and (queries_dim is not None) and (queries_dim > 0)

        def latent_cross_attn():
            return PreNorm(
                queries_dim,
                Attention(
                    queries_dim, dim, dim,
                    heads=cross_heads,
                    dim_head=cross_dim_head),
                context_dim=dim,
            )

        def latent_attn():
            return PreNorm(
                dim,
                Attention(
                    dim,
                    heads=latent_heads,
                    dim_head=latent_dim_head),
            )

        def latent_ff():
            return PreNorm(
                dim,
                FeedForward(dim),
            )

        cache_args = {'_cache': weight_tie_layers}

        get_latent_attn, get_latent_ff = map(cache_fn, (latent_attn, latent_ff))
        self.encoder_layers = nn.ModuleList([])
        for i in range(encoder_depth):
            self.encoder_layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        if self.has_decoder:
            get_latent_cross_attn, get_latent_attn, get_latent_ff = map(cache_fn, (latent_cross_attn, latent_attn, latent_ff))
            self.decoder_layers = nn.ModuleList([])
            for i in range(decoder_depth):
                if (i % decode_cross_every) == 0:
                    layers = [
                        get_latent_cross_attn(**cache_args),
                        get_latent_attn(**cache_args),
                        get_latent_ff(**cache_args)
                    ]
                else:
                    layers = [
                        get_latent_attn(**cache_args),
                        get_latent_ff(**cache_args)
                    ]
                self.decoder_layers.append(nn.ModuleList(layers))

        self.to_logits = nn.Linear(dim, logits_dim) if logits_dim is not None else nn.Identity()

    def forward(
        self,
        x,
        mask=None,
        queries=None
    ):
        b = x.shape[0]

        # layers
        for self_attn, self_ff in self.encoder_layers:
            x = self_attn(x, mask=mask) + x
            x = self_ff(x) + x

        if queries is None or not self.has_decoder:
            return self.to_logits(x)

        # make sure queries contains batch dimension
        if queries.ndim == 2:
            queries = repeat(queries, 'n d -> b n d', b=b)

        # cross attend from decoder queries to latents
        for layers in self.decoder_layers:
            if len(layers) == 3:
                cross_attn, self_attn, self_ff = layers
                x = cross_attn(queries, context=x)
            else:
                self_attn, self_ff = layers

            x = self_attn(x) + x
            x = self_ff(x) + x

        # final linear out
        return self.to_logits(x)


class TransformerEncoderLayerExtended(nn.TransformerEncoderLayer):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu",
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 mha_kwargs=None,
                 device=None, dtype=None) -> None:
        # factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            device=device,
            dtype=dtype,
        )
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            **mha_kwargs,
        )


class VanillaTransformerEncoder(nn.Module, BackboneEncoderDecoder):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = True,
        mha_kwargs=None,
    ):
        if mha_kwargs is None:
            mha_kwargs = dict()

        super().__init__()
        self.encoder = nn.TransformerEncoder(
            TransformerEncoderLayerExtended(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                batch_first=batch_first,
                norm_first=norm_first,
                mha_kwargs=mha_kwargs,
            ),
            num_layers=num_layers,
        )

    def forward(
        self,
        x,
        mask=None,
        queries=None
    ):
        assert queries is None
        return self.encoder(x, src_key_padding_mask=(mask == 0))


class MM_VITEncoderDecoder(nn.Module, BackboneEncoderDecoder):
    """
    mmsegmentation variant of VIT

    Needs 768 features.

    Notes:
        https://github.com/open-mmlab/mmsegmentation/tree/master/configs/vit

    Results:
        # 1
        https://github.com/open-mmlab/mmsegmentation/tree/master/configs/vit#ade20k
        https://github.com/open-mmlab/mmsegmentation/blob/master/configs/vit/upernet_vit-b16_mln_512x512_80k_ade20k.py
        https://github.com/open-mmlab/mmsegmentation/blob/master/configs/_base_/models/upernet_vit-b16_ln_mln.py


    Example:
        >>> # xdoctest: +REQUIRES(module:mmseg)
        >>> from geowatch.tasks.fusion.architectures.transformer import *  # NOQA
        >>> self = MM_VITEncoderDecoder(16, 16, 16)
        >>> x = torch.rand(2, 3, 16)
        >>> self.forward(x)

    Ignore:
        >>> # xdoctest: +REQUIRES(module:mmseg)
        >>> # This tests downloading weights from the MM repo
        >>> from geowatch.tasks.fusion.architectures.transformer import *  # NOQA
        >>> self = MM_VITEncoderDecoder(16, 16, 16, pretrained="upernet_vit-b16_mln_512x512_80k_ade20k")
        >>> x = torch.rand(2, 3, 16)
        >>> self.forward(x)
    """

    pretrained_fpath_shortnames = {
        "upernet_vit-b16_mln_512x512_80k_ade20k":
            'https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_vit-b16_mln_512x512_80k_ade20k/upernet_vit-b16_mln_512x512_80k_ade20k_20210624_130547-0403cee1.pth',
    }

    def __init__(
        self,
        dim,
        logits_dim,
        queries_dim=None,
        pretrained=None,
    ):
        from mmseg.models.backbones.vit import VisionTransformer
        super().__init__()

        # if a short name is used, replace it with the appropriate full path
        if pretrained in MM_VITEncoderDecoder.pretrained_fpath_shortnames.keys():
            pretrained = MM_VITEncoderDecoder.pretrained_fpath_shortnames[pretrained]
            pretrained = ub.grabdata(pretrained)

        kwargs = dict(
            pretrained=pretrained,
            img_size=(512, 512),
            patch_size=16,
            in_channels=3,
            embed_dims=768,
            num_layers=12,
            num_heads=12,
            mlp_ratio=4,
            out_indices=(2, 5, 8, 11),
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            with_cls_token=True,
            norm_cfg=dict(type='LN', eps=1e-6),
            act_cfg=dict(type='GELU'),
            norm_eval=False,
            interpolate_mode='bicubic')
        vit_model = VisionTransformer(**kwargs)
        # We only need the encoder
        self.layers = vit_model.layers

        # if a pretrained path is provided, try to use it
        # if isinstance(pretrained, str):
        #     self.initialize_from_pretrained(pretrained)

        self.encoder_in_features = self.layers[0].ln1.weight.shape[0]
        self.encoder_out_features = self.layers[-1].ffn.layers[1].out_features

        self.input_projector = nn.Linear(dim, self.encoder_in_features)
        self.output_projector = nn.Linear(self.encoder_out_features, logits_dim)

        self.has_decoder = (queries_dim is not None) and (queries_dim > 0)

        if self.has_decoder:
            self.query_projector = nn.Linear(queries_dim, self.encoder_out_features)
            self.decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(d_model=self.encoder_out_features, nhead=8, dim_feedforward=512, batch_first=True),
                num_layers=1,
            )

    def initialize_from_pretrained(self, fpath):
        # FIXME: Having this import here breaks torch.package
        # initializer = coerce_initializer(fpath)
        # info = initializer.forward(self, verbose=0)  # NOQA
        pass

    def forward(self, x, mask=None, queries=None):
        # orig_shape = x.shape
        # x = x.view(x.shape[0], -1, x.shape[1])
        x = self.input_projector(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # if i == len(self.layers) - 1:
            #     if self.final_norm:
            #         x = self.norm1(x)
            # if i in self.out_indices:
            #     if self.with_cls_token:
            #         # Remove class token and reshape token for decoder head
            #         out = x[:, 1:]
            #     else:
            #         out = x
            #     B, _, C = out.shape
            #     out = out.reshape(B, hw_shape[0], hw_shape[1],
            #                       C).permute(0, 3, 1, 2).contiguous()
            #     if self.output_cls_token:
            #         out = [out, x[:, 0]]
            #     outs.append(out)
        # x = x.view(*orig_shape[0], x.shape[-1])

        if (queries is None) or (not self.has_decoder):
            return x

        queries = self.query_projector(queries)
        x = self.decoder(queries, x)
        x = self.output_projector(x)

        return x
