"""
This module should be reorganized into architectures as it consists of smaller
modular network components

Ignore:
    import liberator
    lib = liberator.Liberator()
    from timm.models.layers import drop_path
    lib.add_dynamic(drop_path)
    lib.expand(['timm'])
    print(lib.current_sourcecode())
"""
import torch
from torch.nn.modules.container import Module
from torch._jit_internal import _copy_to_script_wrapper
import einops
import numpy as np
import netharn as nh
from torch import nn


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    From: from timm.models.layers import drop_path
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class RobustModuleDict(torch.nn.ModuleDict):
    """
    Regular torch.nn.ModuleDict doesnt allow empty str. Hack around this.

    Example:
        >>> from geowatch.tasks.fusion.methods.network_modules import *  # NOQA
        >>> import string
        >>> torch_dict = RobustModuleDict()
        >>> # All printable characters should be usable as keys
        >>> # If they are not, hack it.
        >>> failed = []
        >>> for c in list(string.printable) + ['']:
        >>>     try:
        >>>         torch_dict[c] = torch.nn.Linear(1, 1)
        >>>     except KeyError:
        >>>         failed.append(c)
        >>> assert len(failed) == 0
    """
    repl_dot = '#D#'
    repl_empty = '__EMPTY'

    def _normalize_key(self, key):
        key = self.repl_empty if key == '' else key.replace('.', self.repl_dot)
        return key

    @classmethod
    def _unnormalize_key(self, key):
        if key == self.repl_empty:
            return ''
        else:
            return key.replace(self.repl_dot, '.')

    @_copy_to_script_wrapper
    def __getitem__(self, key: str) -> Module:
        key = self._normalize_key(key)
        return self._modules[key]

    def __setitem__(self, key: str, module: Module) -> None:
        key = self._normalize_key(key)
        self.add_module(key, module)

    def __delitem__(self, key: str) -> None:
        key = self._normalize_key(key)
        del self._modules[key]

    @_copy_to_script_wrapper
    def __contains__(self, key: str) -> bool:
        key = self._normalize_key(key)
        return key in self._modules

    def pop(self, key: str) -> Module:
        r"""Remove key from the ModuleDict and return its module.

        Args:
            key (string): key to pop from the ModuleDict
        """
        key = self._normalize_key(key)
        v = self[key]
        del self[key]
        return v


class RobustParameterDict(torch.nn.ParameterDict):
    """
    Regular torch.nn.ParameterDict doesnt allow empty str. Hack around this.

    Example:
        >>> from geowatch.tasks.fusion.methods.network_modules import *  # NOQA
        >>> import string
        >>> torch_dict = RobustParameterDict()
        >>> # All printable characters should be usable as keys
        >>> # If they are not, hack it.
        >>> failed = []
        >>> for c in list(string.printable) + ['']:
        >>>     try:
        >>>         torch_dict[c] = torch.nn.Parameter(torch.ones((1, 1)))
        >>>     except KeyError:
        >>>         failed.append(c)
        >>> assert len(failed) == 0
        >>> for v in torch_dict.values():
        >>>     assert list(v.shape) == [1, 1]
    """
    repl_dot = '#D#'
    repl_empty = '__EMPTY'

    def _normalize_key(self, key):
        key = self.repl_empty if key == '' else key.replace('.', self.repl_dot)
        return key

    @classmethod
    def _unnormalize_key(self, key):
        if key == self.repl_empty:
            return ''
        else:
            return key.replace(self.repl_dot, '.')

    def __getitem__(self, key: str) -> Module:
        key = self._normalize_key(key)
        return super().__getitem__(key)

    def __setitem__(self, key: str, value) -> None:
        key = self._normalize_key(key)
        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        key = self._normalize_key(key)
        super().__delitem__(key, key)

    def __contains__(self, key: str) -> bool:
        key = self._normalize_key(key)
        return super().__contains__(key, key)

    def pop(self, key: str) -> Module:
        key = self._normalize_key(key)
        return super().pop(key)


class OurDepthwiseSeparableConv(nn.Module):
    """ DepthwiseSeparable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.

    From timm

    Example:
        from geowatch.tasks.fusion.methods.network_modules import *  # NOQA

        norm = nh.layers.rectify_normalizer(in_channels=3, key={'type': 'group', 'num_groups': 1})
        norm(torch.rand(2, 1))

        self = OurDepthwiseSeparableConv(11, 13, kernel_size=3, padding=1, residual=1)
        x = torch.rand(2, 11, 3, 3)
        y = self.forward(x)

        z = nh.OutputShapeFor(self.conv_dw)((2, 11, 1, 1))
        print('z = {!r}'.format(z))
        nh.OutputShapeFor(self.conv_pw)(z)

        in_modes = 13
        self =

        tokenizer = nn.Sequential(*[
            OurDepthwiseSeparableConv(in_modes, in_modes, kernel_size=3, stride=1, padding=1, residual=1, norm=None, noli=None),
            OurDepthwiseSeparableConv(in_modes, in_modes * 2, kernel_size=3, stride=2, padding=1, residual=0, norm=None),
            OurDepthwiseSeparableConv(in_modes * 2, in_modes * 4, kernel_size=3, stride=2, padding=1, residual=0),
            OurDepthwiseSeparableConv(in_modes * 4, in_modes * 8, kernel_size=3, stride=2, padding=1, residual=0),
        ])

        tokenizer = nn.Sequential(*[
            OurDepthwiseSeparableConv(in_modes, in_modes, kernel_size=3, stride=1, padding=1, residual=1),
            OurDepthwiseSeparableConv(in_modes, in_modes * 2, kernel_size=3, stride=2, padding=1, residual=0),
            OurDepthwiseSeparableConv(in_modes * 2, in_modes * 4, kernel_size=3, stride=2, padding=1, residual=0),
            OurDepthwiseSeparableConv(in_modes * 4, in_modes * 8, kernel_size=3, stride=2, padding=1, residual=0),
        ])
    """

    def __init__(
            self, in_chs, out_chs, kernel_size=3, stride=1, dilation=1,
            padding=0, residual=False, pw_kernel_size=1, norm='group',
            noli='swish', drop_path_rate=0.):

        super().__init__()
        if norm == 'auto':
            norm = {'type': 'group', 'num_groups': 'auto'}

        self.has_residual = (stride == 1 and in_chs == out_chs) and residual
        self.drop_path_rate = drop_path_rate

        conv_cls = nh.layers.rectify_conv(dim=2)
        # self.conv_dw = create_conv2d(
        #     in_chs, in_chs, kernel_size, stride=stride, dilation=dilation, padding=pad_type, depthwise=True)
        self.conv_dw = conv_cls(
            in_chs, in_chs, kernel_size, stride=stride, dilation=dilation,
            padding=padding, groups=in_chs)  # depthwise

        self.bn1 = nh.layers.rectify_normalizer(in_channels=in_chs, key=norm)
        if self.bn1 is None:
            self.bn1 = nh.layers.Identity()
        self.act1 = nh.layers.rectify_nonlinearity(noli)
        if self.act1 is None:
            self.act1 = nh.layers.Identity()

        self.conv_pw = conv_cls(in_chs, out_chs, pw_kernel_size, padding=0)
        # self.bn2 = norm_layer(out_chs)
        self.bn2 = nh.layers.rectify_normalizer(in_channels=out_chs, key=norm)
        if self.bn2 is None:
            self.bn2 = nh.layers.Identity()

    def feature_info(self, location):
        if location == 'expansion':  # after SE, input to PW
            info = dict(module='conv_pw', hook_type='forward_pre', num_chs=self.conv_pw.in_channels)
        else:  # location == 'bottleneck', block output
            info = dict(module='', hook_type='', num_chs=self.conv_pw.out_channels)
        return info

    def forward(self, x):
        shortcut = x

        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv_pw(x)
        x = self.bn2(x)

        if self.has_residual:
            if self.drop_path_rate > 0.:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut
        return x


class DWCNNTokenizer(nh.layers.Sequential):
    """
    self = DWCNNTokenizer(13, 2)
    inputs = torch.rand(2, 13, 16, 16)
    self(inputs)
    """

    def __init__(self, in_chn, out_chn, norm='auto'):
        super().__init__()
        if norm == 'none':
            norm = None
        self.norm = norm
        super().__init__(*[
            OurDepthwiseSeparableConv(in_chn, in_chn, kernel_size=3, stride=1, padding=1, residual=1, norm=None, noli=None),
            OurDepthwiseSeparableConv(in_chn, in_chn * 4, kernel_size=3, stride=2, padding=1, residual=0, norm=norm),
            OurDepthwiseSeparableConv(in_chn * 4, in_chn * 8, kernel_size=3, stride=2, padding=1, residual=0, norm=norm),
            OurDepthwiseSeparableConv(in_chn * 8, out_chn, kernel_size=3, stride=2, padding=1, residual=0, norm=norm),
        ])
        self.in_channels = in_chn
        self.out_channels = out_chn


class LinearConvTokenizer(nh.layers.Sequential):
    """
    Example:
        >>> from geowatch.tasks.fusion.methods.network_modules import *  # NOQA
        >>> LinearConvTokenizer(1, 512)
    """

    def __init__(self, in_channels, out_channels):
        # import math
        c1 = in_channels * 1
        c2 = in_channels * 4
        c3 = in_channels * 16
        c4 = in_channels * 8
        # final_groups = math.gcd(104, out_channels)
        final_groups = 1

        super().__init__(
            nh.layers.ConvNormNd(
                dim=2, in_channels=c1, out_channels=c2, groups=c1, norm=None,
                noli=None, kernel_size=3, stride=2, padding=1,
            ).conv,
            nh.layers.ConvNormNd(
                dim=2, in_channels=c2, out_channels=c3, groups=c2, norm=None,
                noli=None, kernel_size=3, stride=2, padding=1,
            ).conv,
            nh.layers.ConvNormNd(
                dim=2, in_channels=c3, out_channels=c4, groups=min(c3, c4), norm=None,
                noli=None, kernel_size=3, stride=2, padding=1,
            ).conv,
            nh.layers.ConvNormNd(
                dim=2, in_channels=c4, out_channels=out_channels,
                groups=final_groups, norm=None, noli=None, kernel_size=1,
                stride=1, padding=0,
            ).conv,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels


class ConvTokenizer(nn.Module):
    """
    Example:

        from geowatch.tasks.fusion.methods.network_modules import *  # NOQA
        self = ConvTokenizer(13, 64)
        print('self = {!r}'.format(self))
        inputs = torch.rand(2, 13, 128, 128)
        tokens = self(inputs)
        print('inputs.shape = {!r}'.format(inputs.shape))
        print('tokens.shape = {!r}'.format(tokens.shape))


     Benchmark:

        in_channels = 13
        tokenizer1 = ConvTokenizer(in_channels, 512)
        tokenizer2 = RearrangeTokenizer(in_channels, 8, 8)
        tokenizer3 = DWCNNTokenizer(in_channels, 512)
        tokenizer4 = LinearConvTokenizer(in_channels, 512)
        print(nh.util.number_of_parameters(tokenizer1))
        print(nh.util.number_of_parameters(tokenizer2))
        print(nh.util.number_of_parameters(tokenizer3))
        print(nh.util.number_of_parameters(tokenizer4))

        print(nh.util.number_of_parameters(tokenizer4[0]))
        print(nh.util.number_of_parameters(tokenizer4[1]))
        print(nh.util.number_of_parameters(tokenizer4[2]))
        print(nh.util.number_of_parameters(tokenizer4[3]))

        inputs = torch.rand(1, in_channels, 128, 128)

        import timerit
        ti = timerit.Timerit(100, bestof=1, verbose=2)

        tokenizer1(inputs).shape
        tokenizer2(inputs).shape

        for timer in ti.reset('tokenizer1'):
            with timer:
                tokenizer1(inputs)

        for timer in ti.reset('tokenizer2'):
            with timer:
                tokenizer2(inputs)

        for timer in ti.reset('tokenizer3'):
            with timer:
                tokenizer3(inputs)

        for timer in ti.reset('tokenizer4'):
            with timer:
                tokenizer4(inputs)

        input_shape = (1, in_channels, 64, 64)

        print(tokenizer2(torch.rand(*input_shape)).shape)
        downsampler1 = nh.layers.Sequential(*[
            nh.layers.ConvNormNd(
                dim=2, in_channels=in_channels, out_channels=in_channels,
                groups=in_channels, norm=None, noli=None, kernel_size=3,
                stride=2, padding=1,
            ),
            nh.layers.ConvNormNd(
                dim=2, in_channels=in_channels, out_channels=in_channels,
                groups=in_channels, norm=None, noli=None, kernel_size=3,
                stride=2, padding=1,
            ),
            nh.layers.ConvNormNd(
                dim=2, in_channels=in_channels, out_channels=in_channels,
                groups=in_channels, norm=None, noli=None, kernel_size=3,
                stride=2, padding=1,
            ),
        ])

        downsampler2 = nh.layers.Sequential(*[
            nh.layers.ConvNormNd(
                dim=2, in_channels=in_channels, out_channels=in_channels,
                groups=in_channels, norm=None, noli=None, kernel_size=7,
                stride=5, padding=3,
            ),
        ])
        print(ub.urepr(downsampler1.output_shape_for(input_shape).hidden.shallow(30), nl=1))
        print(ub.urepr(downsampler2.output_shape_for(input_shape).hidden.shallow(30), nl=1))


    """

    def __init__(self, in_chn, out_chn, norm=None):
        super().__init__()
        self.down = nh.layers.ConvNormNd(
            dim=2, in_channels=in_chn, out_channels=in_chn, groups=in_chn,
            norm=norm, noli=None, kernel_size=7, stride=5, padding=3,
        )
        self.one_by_one = nh.layers.ConvNormNd(
            dim=2, in_channels=in_chn, out_channels=out_chn, groups=1,
            norm=norm, noli=None, kernel_size=1, stride=1, padding=0,
        )
        self.out_channels = out_chn

    def forward(self, inputs):
        # b, t, c, h, w = inputs.shape
        b, c, h, w = inputs.shape
        # inputs2d = einops.rearrange(inputs, 'b t c h w -> (b t) c h w')
        inputs2d = inputs
        tokens2d = self.down(inputs2d)
        tokens2d = self.one_by_one(tokens2d)
        tokens = tokens2d
        # tokens = einops.rearrange(tokens2d, '(b t) c h w -> b t c h w 1', b=b, t=t)
        return tokens


class RearrangeTokenizer(nn.Module):
    """
    A mapping to a common number of channels and then rearrange

    Not quite a pure rearrange, but is this way for backwards compat
    """

    def __init__(self, in_channels, agree, window_size):
        super().__init__()
        self.window_size = window_size
        self.foot = nh.layers.MultiLayerPerceptronNd(
            dim=2, in_channels=in_channels, hidden_channels=3,
            out_channels=agree, residual=True, norm=None)
        self.out_channels = agree * window_size * window_size

    def forward(self, x):
        mixed_mode = self.foot(x)
        ws = self.window_size
        # HACK: reorganize and fix
        mode_vals_tokens = einops.rearrange(
            mixed_mode, 'b c (h hs) (w ws) -> b (ws hs c) h w', hs=ws, ws=ws)
        return mode_vals_tokens


def _torch_meshgrid(*basis_dims):
    """
    References:
        https://zhaoyu.li/post/how-to-implement-meshgrid-in-pytorch/
    """
    basis_lens = list(map(len, basis_dims))
    new_dims = []
    for i, basis in enumerate(basis_dims):
        # Probably a more efficent way to do this, but its right
        newshape = [1] * len(basis_dims)
        reps = list(basis_lens)
        newshape[i] = -1
        reps[i] = 1
        dd = basis.view(*newshape).repeat(*reps)
        new_dims.append(dd)
    return new_dims


def _class_weights_from_freq(total_freq, mode='median-idf'):
    """
    Example:
        >>> from geowatch.tasks.fusion.methods.network_modules import _class_weights_from_freq
        >>> total_freq = np.array([19503736, 92885, 883379, 0, 0])
        >>> print(_class_weights_from_freq(total_freq, mode='idf'))
        >>> print(_class_weights_from_freq(total_freq, mode='median-idf'))
        >>> print(_class_weights_from_freq(total_freq, mode='log-median-idf'))

        >>> total_freq = np.array([19503736, 92885, 883379, 0, 0, 0, 0, 0, 0, 0, 0])
        >>> print(_class_weights_from_freq(total_freq, mode='idf'))
        >>> print(_class_weights_from_freq(total_freq, mode='median-idf'))
        >>> print(_class_weights_from_freq(total_freq, mode='log-median-idf'))
    """

    def logb(arr, base):
        if base == 'e':
            return np.log(arr)
        elif base == 2:
            return np.log2(arr)
        elif base == 10:
            return np.log10(arr)
        else:
            out = np.log(arr)
            out /= np.log(base)
            return out

    freq = total_freq.copy()
    is_natural = total_freq > 0 & np.isfinite(total_freq)
    natural_freq = freq[is_natural]
    mask = is_natural.copy()

    if len(natural_freq):
        _min, _max = np.quantile(natural_freq, [0.05, 0.95])
        is_robust = (_max >= freq) & (freq >= _min)
        if np.any(is_robust):
            middle_value = np.median(freq[is_robust])
        else:
            middle_value = np.median(natural_freq)
        freq[~is_natural] = natural_freq.min() / 2
    else:
        middle_value = 2

    # variant of median-inverse-frequency
    if mode == 'idf':
        # There is no difference and this and median after reweighting
        weights = (1 / freq)
        mask &= np.isfinite(weights)
    elif mode == 'name-me':
        z = freq[mask]
        a = ((1 - np.eye(len(z))) * z[:, None]).sum(axis=0)
        b = a / z
        c = b / b.max()
        weights = np.zeros(len(freq))
        weights[mask] = c
    elif mode == 'median-idf':
        weights = (middle_value / freq)
        mask &= np.isfinite(weights)
    elif mode == 'log-median-idf':
        weights = (middle_value / freq)
        mask &= np.isfinite(weights)
        weights[~np.isfinite(weights)] = 1.0
        base = 2
        base = np.exp(1)
        weights = logb(weights + (base - 1), base)
        weights = np.maximum(weights, .1)
        weights = np.minimum(weights, 10)
    else:
        raise KeyError('mode = {!r}'.format(mode))

    # unseen classes should probably get a reasonably high weight in case we do
    # see them and need to learn them, but my intuition is to give them
    # less weight than things we have a shot of learning well
    # so they dont mess up the main categories
    natural_weights = weights[mask]
    if len(natural_weights):
        denom = natural_weights.max()
    else:
        denom = 1
    weights[mask] = weights[mask] / denom
    if np.any(mask):
        weights[~mask] = weights[mask].max() / 7
    else:
        weights[~mask] = 1e-1
    weights = np.round(weights, 6)
    return weights


def coerce_criterion(loss_code, weights, ohem_ratio, focal_gamma):
    """
    Helps build a loss function and returns information about the shapes needed
    by the specific loss.

    Args:
        loss_code (str): The code that corresponds to loss function call.
            One of ['cce', 'focal', 'dicefocal'].
        weights (torch.Tensor): Per class weights.
            Note: Only used for 'cce' and 'focal' losses.
        ohem_ratio (float): Ratio of hard examples to sample to compute loss.
            Note: Only applies to focal losses.
        focal_gamma (float): Focal loss gamma parameter.

    Raises:
        KeyError: if loss_code is not recognized.

    Returns:
        torch.nn.modules.loss._Loss: The loss function.
    """
    # import monai
    if loss_code == 'cce':
        criterion = torch.nn.CrossEntropyLoss(
            weight=weights, reduction='none')
        target_encoding = 'index'
        logit_shape = '(b t h w) c'
        target_shape = '(b t h w)'

    elif loss_code == 'focal_multiclass':
        from watch.utils.ext_monai import FocalLoss
        # from monai.losses import FocalLoss
        criterion = FocalLoss(
            reduction='none',
            to_onehot_y=True,
            weight=weights,
            ohem_ratio=ohem_ratio,
            gamma=focal_gamma)

        target_encoding = 'index'
        logit_shape = '(b t h w) c'
        target_shape = '(b t h w)'

    elif loss_code == 'focal':
        from geowatch.utils.ext_monai import FocalLoss
        # from monai.losses import FocalLoss
        criterion = FocalLoss(
            reduction='none',
            to_onehot_y=False,
            weight=weights,
            ohem_ratio=ohem_ratio,
            gamma=focal_gamma)

        target_encoding = 'onehot'
        logit_shape = 'b c h w t'
        target_shape = 'b c h w t'

    elif loss_code == 'dicefocal':
        from geowatch.utils.ext_monai import DiceFocalLoss
        # from monai.losses import DiceFocalLoss
        criterion = DiceFocalLoss(
            sigmoid=True,
            to_onehot_y=False,
            focal_weight=weights,
            reduction='none',
            ohem_ratio_focal=ohem_ratio,
            gamma=focal_gamma)
        target_encoding = 'onehot'
        logit_shape = 'b c h w t'
        target_shape = 'b c h w t'
    else:
        # self.class_criterion = nn.CrossEntropyLoss()
        # self.class_criterion = nn.BCEWithLogitsLoss()
        raise NotImplementedError(loss_code)

    # Augment the criterion with extra information about what it expects
    criterion.target_encoding = target_encoding
    criterion.logit_shape = logit_shape
    criterion.target_shape = target_shape
    criterion.in_channels = len(weights)
    # criterion_info = {
    #     'criterion': criterion,
    #     'target_encoding': target_encoding,
    #     'logit_shape': logit_shape,
    #     'target_shape': target_shape,
    # }
    return criterion


def torch_safe_stack(tensors, dim=0, *, out=None, item_shape=None, dtype=None, device=None):
    """
    Behaves like torch.stack, but does not error when tensors is empty.

    When tensors are not empty this is exactly :func:`torch.stack`.

    When tensors are empty, it constructs an empty output tensor based on
    explicit expected item shape if available, otherwise it assumes items would
    have had a shape of ``[0]``. Likewise dtype and device should be specified
    otherwise they use :func:`torch.empty` defaults.

    Args:
        tensors (List[Tensor]): sequence of tensors to concatenate.
            Passed to :func:`torch.stack`.

        dim (int): dimension to insert. Has to be between 0 and the number of
            dimensions of concatenated tensors (inclusive). Passed to
            :func:`torch.stack`.

        out (Tensor): passed to :func:`torch.stack`.

        item_shape (Tuple[int, ...]): what the shape of an item should be.
            used to construct a default output.

        dtype (torch.dtype): the expected output datatype when tensors is empty.

        device (torch.device | str | int | None) :
            the expected output device when tensors is empty.

    Example:
        >>> from geowatch.tasks.fusion.methods.network_modules import *  # NOQA
        >>> import ubelt as ub
        >>> grid = list(ub.named_product({
        >>>     # 'num': [0, 1, 2, 3],
        >>>     'num': [0, 7],
        >>>     'item_shape': ['auto', None],
        >>>     'shape': [[], [0], [2], [2, 3], [2, 0, 3]],
        >>>     'dim': [0, 1],
        >>> }))
        >>> results = []
        >>> for item in grid:
        >>>     print(f'item={item}')
        >>>     dim = item['dim']
        >>>     shape = item['shape']
        >>>     item['shape'] = tuple(item['shape'])
        >>>     if item['item_shape'] == 'auto':
        >>>         item['item_shape'] = item['shape']
        >>>     num = item['num']
        >>>     tensors = [torch.empty(shape)] * num
        >>>     if dim >= len(shape):
        >>>         continue
        >>>     out = torch_safe_stack(tensors, dim=dim,
        >>>         item_shape=item['item_shape'])
        >>>     row = {
        >>>         **item,
        >>>         'out.shape': out.shape,
        >>>     }
        >>>     print(f'row={row}')
        >>>     results.append(row)
        >>> import pandas as pd
        >>> import rich
        >>> df = pd.DataFrame(results)
        >>> for _, subdf in df.groupby('shape'):
        >>>     subdf = subdf.sort_values(['shape', 'dim', 'item_shape', 'num'])
        >>>     print('')
        >>>     rich.print(subdf.to_string())
    """
    if len(tensors) == 0:
        if item_shape is None:
            # TODO: WARN HERE, THE USER SHOULD PROVIDE A DEFAULT SHAPE
            # OTHERWISE THE FUNCTION MAY NOT PRODUCE COMPATIBLE RESULTS WITH
            # POPULATED VARIANTS
            item_shape = [0]

        out_shape = list(item_shape)
        if dim > len(out_shape):
            raise IndexError(
                f'Dimension out of range (expected to be in range of '
                f'[-1, {len(out_shape)}], but got {dim})'
            )
        out_shape.insert(dim, 0)
        return torch.empty(out_shape, dtype=dtype, device=device)
    else:
        return torch.stack(tensors, dim=dim, out=out)
