import torch
from torch import nn
from einops.layers.torch import Rearrange, Reduce
import einops


class ResidualLayer(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


class KthOutput(nn.Module):
    def __init__(self, module: nn.Module, k: int):
        super().__init__()
        self.module = module
        self.k = k

    def forward(self, x):
        return self.module(x)[self.k]


class MultiheadSelfAttention(nn.MultiheadAttention):
    def forward(self, x):
        return super().forward(x, x, x)


def new_attention_layer(embedding_size, n_heads, **kwargs):
    return ResidualLayer(
        nn.Sequential(
            nn.LayerNorm(embedding_size),
            KthOutput(
                MultiheadSelfAttention(embedding_size, n_heads, **kwargs),
                k=0),
        ))


def new_mlp_layer(embedding_size, dropout, **kwargs):
    return ResidualLayer(
        nn.Sequential(
            nn.Linear(embedding_size, embedding_size, **kwargs),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(embedding_size, embedding_size, **kwargs),
        ))


class ChannelwiseTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        axes,
        embedding_size,
        n_heads,
        dropout=0.,
        default_shape=["batch", "time", "mode", "height", "width", "feature"],
        feature_axis="feature",
        batch_axis="batch",
    ):
        super().__init__()
        self.axes = axes
        self.default_shape = default_shape
        self.feature_axis = feature_axis
        self.batch_axis = batch_axis
        self.default_shape_str = " ".join(default_shape)

        self.attention_modules = nn.ModuleDict({
            " ".join(axis): new_attention_layer(embedding_size, n_heads)
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
            batch_axes = " ".join([a for a in self.default_shape if (a == self.batch_axis or a not in axis) and a != self.feature_axis])
            axial_shape = f"({sequence_axes}) ({batch_axes}) {self.feature_axis}"

            x = einops.rearrange(x, f"{previous_axial_shape} -> {axial_shape}", **shape_dict)
            x = self.attention_modules[" ".join(axis)](x)

            previous_axial_shape = axial_shape

        sequence_axes = " ".join([a for a in self.default_shape if a not in (self.batch_axis, self.feature_axis)])
        axial_shape = f"({sequence_axes}) {self.batch_axis} {self.feature_axis}"

        x = einops.rearrange(x, f"{previous_axial_shape} -> {axial_shape}", **shape_dict)
        x = self.mlp(x)
        x = einops.rearrange(x, f"{axial_shape} -> {self.default_shape_str}", **shape_dict)
        return x

#   default_shape=["batch", "time", "mode", "height", "width", "feature"],


def transformer_encoder(
        axes,
        default_shape=["batch", "sequence", "feature"],
        feature_axis="feature",
        batch_axis="batch",
        embedding_size=128,
        n_layers=4,
        n_heads=8,
        dropout=0.0,
    ):

    layers = [
        nn.LazyLinear(embedding_size),
    ] + [
        ChannelwiseTransformerEncoderLayer(
            axes,
            embedding_size=embedding_size,
            n_heads=n_heads,
            dropout=dropout,
            default_shape=default_shape,
            feature_axis=feature_axis,
            batch_axis=batch_axis,
        )
        for _ in range(n_layers)
    ]
    return nn.Sequential(*layers)


def space_mode_time_transformer_encoder(axes, **kwargs):
    return transformer_encoder(
        axes,
        default_shape=["batch", "time", "mode", "height", "width", "feature"],
        feature_axis="feature",
        batch_axis="batch",
        **kwargs,
    )


def smt_it_joint(**kwargs):
    return space_mode_time_transformer_encoder(
        axes=[
            ("time", "mode", "height", "width"),
        ],
        **kwargs,
    )


def smt_it_joint_p8(**kwargs): return smt_it_joint(n_layers=8, embedding_size=128, n_heads=4, **kwargs)
def smt_it_joint_n12(**kwargs): return smt_it_joint(n_layers=12, embedding_size=128, n_heads=4, **kwargs)
def smt_it_joint_t12(**kwargs): return smt_it_joint(n_layers=12, embedding_size=192, n_heads=4, **kwargs)
def smt_it_joint_t24(**kwargs): return smt_it_joint(n_layers=24, embedding_size=192, n_heads=4, **kwargs)
def smt_it_joint_s12(**kwargs): return smt_it_joint(n_layers=12, embedding_size=384, n_heads=8, **kwargs)
def smt_it_joint_s24(**kwargs): return smt_it_joint(n_layers=24, embedding_size=384, n_heads=8, **kwargs)
def smt_it_joint_m24(**kwargs): return smt_it_joint(n_layers=24, embedding_size=512, n_heads=8, **kwargs)
def smt_it_joint_l24(**kwargs): return smt_it_joint(n_layers=24, embedding_size=768, n_heads=8, **kwargs)


def smt_it_stm(**kwargs):
    return space_mode_time_transformer_encoder(
        axes=[
            ("height", "width"),
            ("time",),
            ("mode",),
        ],
        **kwargs,
    )


def smt_it_stm_p8(**kwargs): return smt_it_stm(n_layers=8, embedding_size=128, n_heads=4, **kwargs)
def smt_it_stm_n12(**kwargs): return smt_it_stm(n_layers=12, embedding_size=128, n_heads=4, **kwargs)
def smt_it_stm_t12(**kwargs): return smt_it_stm(n_layers=12, embedding_size=192, n_heads=4, **kwargs)
def smt_it_stm_t24(**kwargs): return smt_it_stm(n_layers=24, embedding_size=192, n_heads=4, **kwargs)
def smt_it_stm_s12(**kwargs): return smt_it_stm(n_layers=12, embedding_size=384, n_heads=8, **kwargs)
def smt_it_stm_s24(**kwargs): return smt_it_stm(n_layers=24, embedding_size=384, n_heads=8, **kwargs)
def smt_it_stm_m24(**kwargs): return smt_it_stm(n_layers=24, embedding_size=512, n_heads=8, **kwargs)
def smt_it_stm_l24(**kwargs): return smt_it_stm(n_layers=24, embedding_size=768, n_heads=8, **kwargs)


def smt_it_sm(**kwargs):
    return space_mode_time_transformer_encoder(
        axes=[
            ("height", "width"),
            ("mode",),
        ],
        **kwargs,
    )


def smt_it_sm_p8(**kwargs): return smt_it_sm(n_layers=8, embedding_size=128, n_heads=4, **kwargs)
def smt_it_sm_n12(**kwargs): return smt_it_sm(n_layers=12, embedding_size=128, n_heads=4, **kwargs)
def smt_it_sm_t12(**kwargs): return smt_it_sm(n_layers=12, embedding_size=192, n_heads=4, **kwargs)
def smt_it_sm_t24(**kwargs): return smt_it_sm(n_layers=24, embedding_size=192, n_heads=4, **kwargs)
def smt_it_sm_s12(**kwargs): return smt_it_sm(n_layers=12, embedding_size=384, n_heads=8, **kwargs)
def smt_it_sm_s24(**kwargs): return smt_it_sm(n_layers=24, embedding_size=384, n_heads=8, **kwargs)
def smt_it_sm_m24(**kwargs): return smt_it_sm(n_layers=24, embedding_size=512, n_heads=8, **kwargs)
def smt_it_sm_l24(**kwargs): return smt_it_sm(n_layers=24, embedding_size=768, n_heads=8, **kwargs)


def smt_it_st(**kwargs):
    return space_mode_time_transformer_encoder(
        axes=[
            ("height", "width"),
            ("time",),
        ],
        **kwargs,
    )


def smt_it_st_p8(**kwargs): return smt_it_st(n_layers=8, embedding_size=128, n_heads=4, **kwargs)
def smt_it_st_n12(**kwargs): return smt_it_st(n_layers=12, embedding_size=128, n_heads=4, **kwargs)
def smt_it_st_t12(**kwargs): return smt_it_st(n_layers=12, embedding_size=192, n_heads=4, **kwargs)
def smt_it_st_t24(**kwargs): return smt_it_st(n_layers=24, embedding_size=192, n_heads=4, **kwargs)
def smt_it_st_s12(**kwargs): return smt_it_st(n_layers=12, embedding_size=384, n_heads=8, **kwargs)
def smt_it_st_s24(**kwargs): return smt_it_st(n_layers=24, embedding_size=384, n_heads=8, **kwargs)
def smt_it_st_m24(**kwargs): return smt_it_st(n_layers=24, embedding_size=512, n_heads=8, **kwargs)
def smt_it_st_l24(**kwargs): return smt_it_st(n_layers=24, embedding_size=768, n_heads=8, **kwargs)


def smt_it_tm(**kwargs):
    return space_mode_time_transformer_encoder(
        axes=[
            ("time",),
            ("mode",),
        ],
        **kwargs,
    )


def smt_it_tm_p8(**kwargs): return smt_it_tm(n_layers=8, embedding_size=128, n_heads=4, **kwargs)
def smt_it_tm_n12(**kwargs): return smt_it_tm(n_layers=12, embedding_size=128, n_heads=4, **kwargs)
def smt_it_tm_t12(**kwargs): return smt_it_tm(n_layers=12, embedding_size=192, n_heads=4, **kwargs)
def smt_it_tm_t24(**kwargs): return smt_it_tm(n_layers=24, embedding_size=192, n_heads=4, **kwargs)
def smt_it_tm_s12(**kwargs): return smt_it_tm(n_layers=12, embedding_size=384, n_heads=8, **kwargs)
def smt_it_tm_s24(**kwargs): return smt_it_tm(n_layers=24, embedding_size=384, n_heads=8, **kwargs)
def smt_it_tm_m24(**kwargs): return smt_it_tm(n_layers=24, embedding_size=512, n_heads=8, **kwargs)
def smt_it_tm_l24(**kwargs): return smt_it_tm(n_layers=24, embedding_size=768, n_heads=8, **kwargs)


def smt_it_s(**kwargs):
    return space_mode_time_transformer_encoder(
        axes=[
            ("height", "width"),
        ],
        **kwargs,
    )


def smt_it_s_p8(**kwargs): return smt_it_s(n_layers=8, embedding_size=128, n_heads=4, **kwargs)
def smt_it_s_n12(**kwargs): return smt_it_s(n_layers=12, embedding_size=128, n_heads=4, **kwargs)
def smt_it_s_t12(**kwargs): return smt_it_s(n_layers=12, embedding_size=192, n_heads=4, **kwargs)
def smt_it_s_t24(**kwargs): return smt_it_s(n_layers=24, embedding_size=192, n_heads=4, **kwargs)
def smt_it_s_s12(**kwargs): return smt_it_s(n_layers=12, embedding_size=384, n_heads=8, **kwargs)
def smt_it_s_s24(**kwargs): return smt_it_s(n_layers=24, embedding_size=384, n_heads=8, **kwargs)
def smt_it_s_m24(**kwargs): return smt_it_s(n_layers=24, embedding_size=512, n_heads=8, **kwargs)
def smt_it_s_l24(**kwargs): return smt_it_s(n_layers=24, embedding_size=768, n_heads=8, **kwargs)


def smt_it_t(**kwargs):
    return space_mode_time_transformer_encoder(
        axes=[
            ("time",),
        ],
        **kwargs,
    )


def smt_it_t_p8(**kwargs): return smt_it_t(n_layers=8, embedding_size=128, n_heads=4, **kwargs)
def smt_it_t_n12(**kwargs): return smt_it_t(n_layers=12, embedding_size=128, n_heads=4, **kwargs)
def smt_it_t_t12(**kwargs): return smt_it_t(n_layers=12, embedding_size=192, n_heads=4, **kwargs)
def smt_it_t_t24(**kwargs): return smt_it_t(n_layers=24, embedding_size=192, n_heads=4, **kwargs)
def smt_it_t_s12(**kwargs): return smt_it_t(n_layers=12, embedding_size=384, n_heads=8, **kwargs)
def smt_it_t_s24(**kwargs): return smt_it_t(n_layers=24, embedding_size=384, n_heads=8, **kwargs)
def smt_it_t_m24(**kwargs): return smt_it_t(n_layers=24, embedding_size=512, n_heads=8, **kwargs)
def smt_it_t_l24(**kwargs): return smt_it_t(n_layers=24, embedding_size=768, n_heads=8, **kwargs)


def smt_it_m(**kwargs):
    return space_mode_time_transformer_encoder(
        axes=[
            ("mode",),
        ],
        **kwargs,
    )


def smt_it_m_p8(**kwargs): return smt_it_m(n_layers=8, embedding_size=128, n_heads=4, **kwargs)
def smt_it_m_n12(**kwargs): return smt_it_m(n_layers=12, embedding_size=128, n_heads=4, **kwargs)
def smt_it_m_t12(**kwargs): return smt_it_m(n_layers=12, embedding_size=192, n_heads=4, **kwargs)
def smt_it_m_t24(**kwargs): return smt_it_m(n_layers=24, embedding_size=192, n_heads=4, **kwargs)
def smt_it_m_s12(**kwargs): return smt_it_m(n_layers=12, embedding_size=384, n_heads=8, **kwargs)
def smt_it_m_s24(**kwargs): return smt_it_m(n_layers=24, embedding_size=384, n_heads=8, **kwargs)
def smt_it_m_m24(**kwargs): return smt_it_m(n_layers=24, embedding_size=512, n_heads=8, **kwargs)
def smt_it_m_l24(**kwargs): return smt_it_m(n_layers=24, embedding_size=768, n_heads=8, **kwargs)


def smt_it_hwtm(**kwargs):
    return space_mode_time_transformer_encoder(
        axes=[
            ("height",),
            ("width",),
            ("time",),
            ("mode",),
        ],
        **kwargs,
    )


def smt_it_hwtm_p8(**kwargs): return smt_it_hwtm(n_layers=8, embedding_size=128, n_heads=4, **kwargs)
def smt_it_hwtm_n12(**kwargs): return smt_it_hwtm(n_layers=12, embedding_size=128, n_heads=4, **kwargs)
def smt_it_hwtm_t12(**kwargs): return smt_it_hwtm(n_layers=12, embedding_size=192, n_heads=4, **kwargs)
def smt_it_hwtm_t24(**kwargs): return smt_it_hwtm(n_layers=24, embedding_size=192, n_heads=4, **kwargs)
def smt_it_hwtm_s12(**kwargs): return smt_it_hwtm(n_layers=12, embedding_size=384, n_heads=8, **kwargs)
def smt_it_hwtm_s24(**kwargs): return smt_it_hwtm(n_layers=24, embedding_size=384, n_heads=8, **kwargs)
def smt_it_hwtm_m24(**kwargs): return smt_it_hwtm(n_layers=24, embedding_size=512, n_heads=8, **kwargs)
def smt_it_hwtm_l24(**kwargs): return smt_it_hwtm(n_layers=24, embedding_size=768, n_heads=8, **kwargs)
