"""
Ported bits of netharn. This does not include any of the analytic
output-shape-for methods.

Ignore:
    import liberator
    lib = liberator.Liberator()

    import netharn as nh
    lib.add_dynamic(nh.api.Initializer)

    lib.expand(['netharn'])
    print(lib.current_sourcecode())

    lib.add_dynamic(nh.initializers.KaimingNormal)
    print(lib.current_sourcecode())


    import liberator
    lib = liberator.Liberator()
    lib.add_dynamic(nh.util.number_of_parameters)
    print(lib.current_sourcecode())


    import liberator
    lib = liberator.Liberator()
    import netharn as nh
    lib.add_dynamic(nh.data.collate.padded_collate)
    lib.expand(['netharn'])
    print(lib.current_sourcecode())
"""
import torch
import ubelt as ub
from math import gcd
import torch.nn.functional as F
from torch.utils import data as torch_data
import numpy as np  # NOQA

import collections.abc as container_abcs
from six import string_types as string_classes
from six import integer_types as int_classes
import re

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor
}


def _update_defaults(config, kw):
    config = dict(config)
    for k, v in kw.items():
        if k not in config:
            config[k] = v
    return config


class Optimizer:
    """
    Old netharn.api.Optimizer class. Ideally this is deprecated.
    """

    @staticmethod
    def coerce(config={}, **kw):
        """
        Accepts keywords:
            optimizer / optim :
                can be sgd, adam, adamw, rmsprop

            learning_rate / lr :
                a float

            weight_decay / decay :
                a float

            momentum:
                a float, only used if the optimizer accepts it

            params:
                This is a SPECIAL keyword that is handled differently.  It is
                interpreted by `netharn.hyper.Hyperparams.make_optimizer`.

                In this simplest case you can pass "params" as a list of torch
                parameter objects or a list of dictionaries containing param
                groups and special group options (just as you would when
                constructing an optimizer from scratch). We don't recommend
                this while using netharn unless you know what you are doing
                (Note, that params will correctly change device if the model is
                mounted).

                In the case where you do not want to group parameters with
                different options, it is best practice to simply not specify
                params.

                In the case where you want to group parameters set params to
                either a List[Dict] or a Dict[str, Dict].

                The items / values of this collection should be a dictionary.
                The keys / values of this dictionary should be the per-group
                optimizer options. Additionally, there should be a key "params"
                (note this is a nested per-group params not to be confused with
                the top-level "params").

                Each per-group "params" should be either (1) a list of
                parameter names (preferred), (2) a string that specifies a
                regular expression (matching layer names will be included in
                this group), or (3) a list of parameter objects.

                For example, the top-level params might look like:

                    params={
                        'head': {'lr': 0.003, 'params': '.*head.*'},
                        'backbone': {'lr': 0.001, 'params': '.*backbone.*'},
                        'preproc': {'lr': 0.0, 'params': [
                            'model.conv1', 'model.norm1', , 'model.relu1']}
                    }

                Note that head and backbone specify membership via regular
                expression whereas preproc explicitly specifies a list of
                parameter names.

        Notes:
            pip install torch-optimizer

        Returns:
            Tuple[type, dict]: a type and arguments to construct it

        References:
            https://datascience.stackexchange.com/questions/26792/difference-between-rmsprop-with-momentum-and-adam-optimizers
            https://github.com/jettify/pytorch-optimizer

        CommandLine:
            xdoctest -m /home/joncrall/code/netharn/netharn/api.py Optimizer.coerce

        Example:
            >>> config = {'optimizer': 'sgd', 'params': [
            >>>     {'lr': 3e-3, 'params': '.*head.*'},
            >>>     {'lr': 1e-3, 'params': '.*backbone.*'},
            >>> ]}
            >>> optim_ = Optimizer.coerce(config)

            >>> # xdoctest: +REQUIRES(module:torch_optimizer)
            >>> config = {'optimizer': 'DiffGrad'}
            >>> optim_ = Optimizer.coerce(config, lr=1e-5)
            >>> print('optim_ = {!r}'.format(optim_))
            >>> assert optim_[1]['lr'] == 1e-5

            >>> config = {'optimizer': 'Yogi'}
            >>> optim_ = Optimizer.coerce(config)
            >>> print('optim_ = {!r}'.format(optim_))

            >>> Optimizer.coerce({'optimizer': 'ASGD'})

        TODO:
            - [ ] https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/
        """
        config = _update_defaults(config, kw)
        key = config.get('optimizer', config.get('optim', 'sgd')).lower()
        lr = config.get('learning_rate', config.get('lr', 3e-3))
        decay = config.get('weight_decay', config.get('decay', 0))
        momentum = config.get('momentum', 0.9)
        params = config.get('params', None)
        # TODO: allow for "discriminative fine-tuning"
        if key == 'sgd':
            cls = torch.optim.SGD
            kw = {
                'lr': lr,
                'weight_decay': decay,
                'momentum': momentum,
                'nesterov': True,
            }
        elif key == 'adam':
            cls = torch.optim.Adam
            kw = {
                'lr': lr,
                'weight_decay': decay,
                # 'betas': (0.9, 0.999),
                # 'eps': 1e-8,
                # 'amsgrad': False
            }
        elif key == 'adamw':
            from torch.optim import AdamW
            cls = AdamW
            kw = {
                'lr': lr,
                'weight_decay': decay,
                # 'betas': (0.9, 0.999),
                # 'eps': 1e-8,
                # 'amsgrad': False
            }
        elif key == 'rmsprop':
            cls = torch.optim.RMSprop
            kw = {
                'lr': lr,
                'weight_decay': decay,
                'momentum': momentum,
                'alpha': 0.9,
            }
        else:
            _lut = {}

            optim_modules = [
                torch.optim,
            ]

            try:
                # Allow coerce to use torch_optimizer package if available
                import torch_optimizer
            except Exception:
                torch_optimizer = None
            else:
                optim_modules.append(torch_optimizer)
                _lut.update({
                    k.lower(): c.__name__
                    for k, c in torch_optimizer._NAME_OPTIM_MAP.items()})

            _lut.update({
                k.lower(): k for k in dir(torch.optim)
                if not k.startswith('_')})

            key = _lut[key]

            cls = None
            for module in optim_modules:
                cls = getattr(module, key, None)
                if cls is not None:
                    defaultkw = default_kwargs(cls)
                    kw = defaultkw.copy()
                    kw.update(ub.dict_isect(config, kw))
                    # Hacks for common cases, otherwise if learning_rate is
                    # given, but only lr exists in the signature, it will be
                    # incorrectly ignored.
                    if 'lr' in kw:
                        kw['lr'] = lr
                    if 'weight_decay' in kw:
                        kw['weight_decay'] = decay
                    break

        if cls is None:
            raise KeyError(key)

        kw['params'] = params
        optim_ = (cls, kw)
        return optim_


class Initializer(object):
    """
    Base class for initializers
    """

    def __call__(self, model, *args, **kwargs):
        return self.forward(model, *args, **kwargs)

    def forward(self, model):
        """
        Abstract function that does the initailization
        """
        raise NotImplementedError('implement me')

    def history(self):
        """
        Initializer methods have histories which are short for algorithms and
        can be quite long for pretrained models
        """
        return None

    def get_initkw(self):
        """
        Initializer methods have histories which are short for algorithms and
        can be quite long for pretrained models
        """
        initkw = self.__dict__.copy()
        # info = {}
        # info['__name__'] = self.__class__.__name__
        # info['__module__'] = self.__class__.__module__
        # info['__initkw__'] = initkw
        return initkw

    @staticmethod
    def coerce(config={}, **kw):
        """
        Accepts 'init', 'pretrained', 'pretrained_fpath', 'leftover', and
        'noli'.

        Args:
            config (dict | str): coercable configuration dictionary.
                if config is a string it is taken as the value for "init".

        Returns:
            Tuple[Initializer, dict]: initializer_ = initializer_cls, kw

        Examples:
            >>> from geowatch.utils.util_netharn import *  # NOQA
            >>> print(ub.urepr(Initializer.coerce({'init': 'noop'})))
            >>> config = {
            ...     'init': 'pretrained',
            ...     'pretrained_fpath': '/fit/nice/untitled'
            ... }
            >>> print(ub.urepr(Initializer.coerce(config)))
            >>> print(ub.urepr(Initializer.coerce({'init': 'kaiming_normal'})))
        """
        if isinstance(config, str):
            config = {
                'init': config,
            }

        config = _update_defaults(config, kw)

        pretrained_fpath = config.get('pretrained_fpath', config.get('pretrained', None))
        init = config.get('initializer', config.get('init', None))

        # Allow init to specify a pretrained fpath
        if isinstance(init, str) and pretrained_fpath is None:
            from os.path import exists
            pretrained_cand = ub.expandpath(init)
            if exists(pretrained_cand):
                pretrained_fpath = pretrained_cand

        config['init'] = init
        config['pretrained_fpath'] = pretrained_fpath
        config['pretrained'] = pretrained_fpath

        if pretrained_fpath is not None:
            config['init'] = 'pretrained'

        import os
        init_verbose = int(os.environ.get('WATCH_INIT_VERBOSE', 4))

        # ---
        initializer_ = None
        if config['init'].lower() in ['kaiming_normal']:
            initializer_ = (KaimingNormal, {
                # initialization params should depend on your choice of
                # nonlinearity in your model. See the Kaiming Paper for details.
                'param': 1e-2 if config.get('noli', 'relu') == 'leaky_relu' else 0,
            })
        elif config['init'] == 'noop':
            initializer_ = (NoOp, {})
        elif config['init'] == 'pretrained':
            from torch_liberator.initializer import Pretrained
            initializer_ = (Pretrained, {
                'fpath': ub.expandpath(config['pretrained_fpath']),
                'leftover': kw.get('leftover', None),
                'mangle': kw.get('mangle', False),
                'association': kw.get('association', None),
                'verbose': init_verbose,
            })
        elif config['init'] == 'cls':
            # Indicate that the model will initialize itself
            # We have to trust that the user does the right thing here.
            pass
        else:
            raise KeyError('Unknown coercable init: {!r}'.format(config['init']))
        return initializer_


class NoOp(Initializer):
    """
    An initializer that does nothing, which is useful when you have initialized
    the weights yourself.

    Example:
        >>> import copy
        >>> self = NoOp()
        >>> model = ToyNet2d()
        >>> old_state = sum(v.sum() for v in model.state_dict().values())
        >>> self(model)
        >>> new_state = sum(v.sum() for v in model.state_dict().values())
        >>> assert old_state == new_state
        >>> assert self.history() is None
    """

    def forward(self, model):
        return


class Orthogonal(Initializer):
    """
    Same as Orthogonal, but uses pytorch implementation

    Example:
        >>> self = Orthogonal()
        >>> model = ToyNet2d()
        >>> try:
        >>>     self(model)
        >>> except RuntimeError:
        >>>     import pytest
        >>>     pytest.skip('geqrf: Lapack probably not availble')
        >>> layer = torch.nn.modules.Conv2d(3, 3, 3)
        >>> self(layer)
    """

    def __init__(self, gain=1):
        self.gain = gain

    def forward(self, model):
        try:
            func = torch.nn.init.orthogonal_
        except AttributeError:
            func = torch.nn.init.orthogonal

        apply_initializer(model, func, self.__dict__)


class KaimingUniform(Initializer):
    """
    Same as HeUniform, but uses pytorch implementation

    Example:
        >>> from geowatch.utils.util_netharn import *  # NOQA
        >>> self = KaimingUniform()
        >>> model = ToyNet2d()
        >>> self(model)
        >>> layer = torch.nn.modules.Conv2d(3, 3, 3)
        >>> self(layer)
    """

    def __init__(self, param=0, mode='fan_in'):
        self.a = param
        self.mode = mode

    def forward(self, model):
        try:
            func = torch.nn.init.kaiming_uniform_
        except AttributeError:
            func = torch.nn.init.kaiming_uniform
        apply_initializer(model, func, self.__dict__)


class KaimingNormal(Initializer):
    """
    Same as HeNormal, but uses pytorch implementation

    Example:
        >>> from geowatch.utils.util_netharn import *  # NOQA
        >>> self = KaimingNormal()
        >>> model = ToyNet2d()
        >>> self(model)
        >>> layer = torch.nn.modules.Conv2d(3, 3, 3)
        >>> self(layer)
    """

    def __init__(self, param=0, mode='fan_in'):
        self.a = param
        self.mode = mode

    def forward(self, model):
        try:
            func = torch.nn.init.kaiming_normal_
        except AttributeError:
            func = torch.nn.init.kaiming_normal
        apply_initializer(model, func, self.__dict__)


def apply_initializer(input, func, funckw):
    """
    Recursively initializes the input using a torch.nn.init function.

    If the input is a model, then only known layer types are initialized.

    Args:
        input (Tensor | Module): can be a model, layer, or tensor
        func (callable): initialization function
        funckw (dict):

    Example:
        >>> from geowatch.utils.util_netharn import *  # NOQA
        >>> from torch import nn
        >>> import torch
        >>> class DummyNet(nn.Module):
        >>>     def __init__(self, n_channels=1, n_classes=10):
        >>>         super(DummyNet, self).__init__()
        >>>         self.conv = nn.Conv2d(n_channels, 10, kernel_size=5)
        >>>         self.norm = nn.BatchNorm2d(10)
        >>>         self.param = torch.nn.Parameter(torch.rand(3))
        >>> self = DummyNet()
        >>> func = nn.init.kaiming_normal_
        >>> apply_initializer(self, func, {})
        >>> func = nn.init.constant_
        >>> apply_initializer(self, func, {'val': 42})
        >>> assert np.all(self.conv.weight.detach().numpy() == 42)
        >>> assert np.all(self.conv.bias.detach().numpy() == 0), 'bias is always init to zero'
        >>> assert np.all(self.norm.bias.detach().numpy() == 0), 'bias is always init to zero'
        >>> assert np.all(self.norm.weight.detach().numpy() == 1)
        >>> assert np.all(self.norm.running_mean.detach().numpy() == 0.0)
        >>> assert np.all(self.norm.running_var.detach().numpy() == 1.0)
    """
    if getattr(input, 'bias', None) is not None:
        # print('zero input bias')
        # zero all biases
        input.bias.data.zero_()

    if isinstance(input, (torch.Tensor)):
        # assert False, ('input is tensor? does this make sense?')
        # print('input is tensor')
        func(input, **funckw)
        # data = input
    elif isinstance(input, (torch.nn.modules.conv._ConvNd)):
        # print('input is convnd')
        func(input.weight, **funckw)
    # elif isinstance(input, (torch.nn.modules.linear.Linear)):
    #     func(input.weight, **funckw)
    elif isinstance(input, torch.nn.modules.batchnorm._BatchNorm):
        # Use default batch norm
        input.reset_parameters()
    # elif isinstance(input, torch.nn.modules.Linear):
    #     input.reset_parameters()
    elif hasattr(input, 'reset_parameters'):
        # print('unknown input type fallback on reset_params')
        input.reset_parameters()
    else:
        # input is a torch module
        model = input
        # print('recurse input')
        layers = list(trainable_layers(model))
        # print('layers = {!r}'.format(layers))
        for item in layers:
            apply_initializer(item, func, funckw)


def trainable_layers(model, names=False):
    """
    Returns all layers containing trainable parameters

    Notes:
        It may be better to simply use model.named_parameters() instead in most
        situation. This is useful when you need the classes that contains the
        parameters instead of the parameters themselves.

    Example:
        >>> from geowatch.utils.util_netharn import *  # NOQA
        >>> import torchvision
        >>> model = torchvision.models.AlexNet()
        >>> list(trainable_layers(model, names=True))
    """
    if names:
        stack = [('', '', model)]
        while stack:
            prefix, basename, item = stack.pop()
            name = '.'.join([p for p in [prefix, basename] if p])
            if isinstance(item, torch.nn.modules.conv._ConvNd):
                yield name, item
            elif isinstance(item, torch.nn.modules.batchnorm._BatchNorm):
                yield name, item
            elif hasattr(item, 'reset_parameters'):
                yield name, item

            child_prefix = name
            for child_basename, child_item in list(item.named_children())[::-1]:
                stack.append((child_prefix, child_basename, child_item))
    else:
        queue = [model]
        while queue:
            item = queue.pop(0)
            # TODO: need to put all trainable layer types here
            # (I think this is just everything with reset_parameters)
            if isinstance(item, torch.nn.modules.conv._ConvNd):
                yield item
            elif isinstance(item, torch.nn.modules.batchnorm._BatchNorm):
                yield item
            elif hasattr(item, 'reset_parameters'):
                yield item
            # if isinstance(input, torch.nn.modules.Linear):
            #     yield item
            # if isinstance(input, torch.nn.modules.Bilinear):
            #     yield item
            # if isinstance(input, torch.nn.modules.Embedding):
            #     yield item
            # if isinstance(input, torch.nn.modules.EmbeddingBag):
            #     yield item
            for child in item.children():
                queue.append(child)


def number_of_parameters(model, trainable=True):
    """
    Returns number of trainable parameters in a torch module

    Example:
        >>> from geowatch.utils.util_netharn import *  # NOQA
        >>> model = torch.nn.Conv1d(2, 3, 5)
        >>> print(number_of_parameters(model))
        33
    """
    if trainable:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    else:
        model_parameters = model.parameters()
    n_params = sum([np.prod(p.size()) for p in model_parameters])
    return n_params


class ToyNet2d(torch.nn.Module):
    """
    Demo model for a simple 2 class learning problem
    """
    def __init__(self, input_channels=1, num_classes=2):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.layers = torch.nn.Sequential(*[
            torch.nn.Conv2d(input_channels, 8, kernel_size=3, padding=1, bias=False),

            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False),

            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(8, num_classes, kernel_size=3, padding=1, bias=False),
        ])

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, inputs):
        spatial_out = self.layers(inputs)
        num = float(np.prod(spatial_out.shape[-2:]))
        averaged = spatial_out.sum(dim=2).sum(dim=2) / num
        probs = self.softmax(averaged)
        return probs


class ToyData2d(torch_data.Dataset):
    """
    Simple black-on-white and white-on-black images.

    Args:
        n (int, default=100): dataset size
        size (int, default=4): width / height
        border (int, default=1): border mode
        rng (RandomCoercable, default=None): seed or random state

    CommandLine:
        python -m netharn.data.toydata ToyData2d --show

    Example:
        >>> self = ToyData2d()
        >>> data1, label1 = self[0]
        >>> data2, label2 = self[-1]
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> plt = kwplot.autoplt()
        >>> kwplot.figure(fnum=1, doclf=True)
        >>> kwplot.imshow(data1.numpy().squeeze(), pnum=(1, 2, 1))
        >>> kwplot.imshow(data2.numpy().squeeze(), pnum=(1, 2, 2))
        >>> kwplot.show_if_requested()
    """
    def __init__(self, size=4, border=1, n=100, rng=None):
        import kwarray
        import itertools as it
        rng = kwarray.ensure_rng(rng)

        h = w = size

        whiteish = 1 - (np.abs(rng.randn(n, 1, h, w) / 4) % 1)
        blackish = (np.abs(rng.randn(n, 1, h, w) / 4) % 1)

        fw = border
        slices = [slice(None, fw), slice(-fw, None)]

        # class 0 is white block inside a black frame
        data1 = whiteish.copy()
        for sl1, sl2 in it.product(slices, slices):
            data1[..., sl1, :] = blackish[..., sl1, :]
            data1[..., :, sl2] = blackish[..., :, sl2]

        # class 1 is black block inside a white frame
        data2 = blackish.copy()
        for sl1, sl2 in it.product(slices, slices):
            data2[..., sl1, :] = whiteish[..., sl1, :]
            data2[..., :, sl2] = whiteish[..., :, sl2]

        self.data = np.concatenate([data1, data2], axis=0)
        self.labels = np.array(([0] * n) + ([1] * n))

        suffix = ub.hash_data([
            size, border, n, rng
        ], base='abc', hasher='sha1')[0:16]
        self.input_id = 'TD2D_{}_'.format(n) + suffix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.FloatTensor(self.data[index])
        label = int(self.labels[index])
        return data, label

    def make_loader(self, *args, **kwargs):
        loader = torch_data.DataLoader(self, *args, **kwargs)
        return loader


class InputNorm(torch.nn.Module):
    """
    Normalizes the input by shifting and dividing by a scale factor.

    This allows for the network to take care of 0-mean 1-std normalization.
    The developer explicitly specifies what these shift and scale values are.
    By specifying this as a layer (instead of a data preprocessing step), the
    exporter will remember and associated this information with any
    deployed model. This means that a user does not need to remember what these
    shit/scale arguments were before passing inputs to a network.

    If the mean and std arguments are unspecified, this layer becomes a noop.

    References:
        .. [WhyNormalize] https://towardsdatascience.com/why-data-should-be-normalized-before-training-a-neural-network-c626b7f66c7d

    Example:
        >>> self = InputNorm(mean=50.0, std=29.0)
        >>> inputs = torch.rand(2, 3, 5, 7) * 100
        >>> outputs = self(inputs)
        >>> # If mean and std are unspecified, this becomes a noop.
        >>> assert torch.all(InputNorm()(inputs) == inputs)
        >>> # Specifying either the mean or the std is ok.
        >>> partial1 = InputNorm(mean=50)(inputs)
        >>> partial2 = InputNorm(std=29)(inputs)

    Ignore:
        import torch

        model = torch.nn.Sequential(*[
            InputNorm(mean=10, std=0.2),
            torch.nn.Conv2d(3, 3, 3),
        ])
        inputs = torch.rand(2, 3, 5, 7) * 100
        optim = torch.optim.SGD(model.parameters(), lr=1e-3)

        for i in range(100):
            optim.zero_grad()
            x = model(inputs).sum()
            x.backward()
            optim.step()

            std = model[0].mean
            mean = model[0].std
            print('std = {!r}'.format(std))
            print('mean = {!r}'.format(mean))
    """

    def __init__(self, mean=None, std=None):
        super(InputNorm, self).__init__()
        if mean is not None:
            mean = mean if ub.iterable(mean) else [mean]
            mean = torch.FloatTensor(mean)
        if std is not None:
            std = std if ub.iterable(std) else [std]
            std = torch.FloatTensor(std)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, inputs):
        outputs = inputs
        if self.mean is not None:
            outputs = outputs - self.mean
        if self.std is not None:
            outputs = outputs / self.std
        return outputs


class MultiLayerPerceptronNd(torch.nn.Module):
    """
    A multi-layer perceptron network for n dimensional data

    Choose the number and size of the hidden layers, number of output channels,
    wheather to user residual connections or not, nonlinearity, normalization,
    dropout, and more.

    Args:
        dim (int): specify if the data is 0, 1, 2, 3, or 4 dimensional.

        in_channels (int): number of input channels

        hidden_channels (List[int]): or an int specifying the number of hidden
            layers (we choose the channel size to linearly interpolate between
            input and output channels)

        out_channels (int): number of output channels

        dropout (float, default=None): amount of dropout to use between 0 and 1

        norm (str, default='batch'): type of normalization layer
            (e.g. batch or group), set to None for no normalization.

        noli (str, default='relu'): type of nonlinearity

        residual (bool, default=False):
            if true includes a resitual skip connection between inputs and
            outputs.

        norm_output (bool, default=True):
            if True, applies a final normalization layer to the output.

        noli_output (bool, default=True):
            if True, applies a final nonlineary to the output.

        standardize_weights (bool, default=False):
            Use weight standardization

    Example:
        >>> kw = {'dim': 0, 'in_channels': 2, 'out_channels': 1}
        >>> model0 = MultiLayerPerceptronNd(hidden_channels=0, **kw)
        >>> model1 = MultiLayerPerceptronNd(hidden_channels=1, **kw)
        >>> model2 = MultiLayerPerceptronNd(hidden_channels=2, **kw)
        >>> print('model0 = {!r}'.format(model0))
        >>> print('model1 = {!r}'.format(model1))
        >>> print('model2 = {!r}'.format(model2))

        >>> kw = {'dim': 0, 'in_channels': 2, 'out_channels': 1, 'residual': True}
        >>> model0 = MultiLayerPerceptronNd(hidden_channels=0, **kw)
        >>> model1 = MultiLayerPerceptronNd(hidden_channels=1, **kw)
        >>> model2 = MultiLayerPerceptronNd(hidden_channels=2, **kw)
        >>> print('model0 = {!r}'.format(model0))
        >>> print('model1 = {!r}'.format(model1))
        >>> print('model2 = {!r}'.format(model2))

    Example:
        >>> import ubelt as ub
        >>> self = MultiLayerPerceptronNd(dim=1, in_channels=128, hidden_channels=3, out_channels=2)
        >>> print(self)
        MultiLayerPerceptronNd...
    """
    def __init__(self, dim, in_channels, hidden_channels, out_channels,
                 bias=True, dropout=None, noli='relu', norm='batch',
                 residual=False, noli_output=False, norm_output=False,
                 standardize_weights=False):

        super(MultiLayerPerceptronNd, self).__init__()
        dropout_cls = rectify_dropout(dim)
        conv_cls = rectify_conv(dim=dim)
        curr_in = in_channels

        if isinstance(hidden_channels, int):
            n = hidden_channels
            hidden_channels = np.linspace(in_channels, out_channels, n + 1,
                                          endpoint=False)[1:]
            hidden_channels = hidden_channels.round().astype(int).tolist()
        self._hidden_channels = hidden_channels

        hidden = self.hidden = torch.nn.Sequential()
        for i, curr_out in enumerate(hidden_channels):
            layer = ConvNormNd(
                dim, curr_in, curr_out, kernel_size=1, bias=bias, noli=noli,
                norm=norm, standardize_weights=standardize_weights)
            hidden.add_module('hidden{}'.format(i), layer)
            if dropout is not None:
                hidden.add_module('dropout{}'.format(i), dropout_cls(p=dropout))
            curr_in = curr_out

        outkw = {'bias': bias, 'kernel_size': 1}
        self.hidden.add_module(
            'output', conv_cls(curr_in, out_channels, **outkw))

        if residual:
            if in_channels == out_channels:
                self.skip = Identity()
            else:
                self.skip = conv_cls(in_channels, out_channels, **outkw)
        else:
            self.skip = None

        if norm_output:
            self.final_norm = rectify_normalizer(out_channels, norm, dim=dim)
        else:
            self.final_norm = None

        if noli_output:
            self.final_noli = rectify_nonlinearity(noli, dim=dim)
        else:
            self.final_noli = None

        self.norm_output = norm_output
        self.noli_output = noli_output
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, inputs):
        outputs = self.hidden(inputs)

        if self.skip is not None:
            projected = self.skip(inputs)
            outputs = projected + outputs

        if self.final_norm is not None:
            outputs = self.final_norm(outputs)

        if self.final_noli is not None:
            outputs = self.final_noli(outputs)

        return outputs


def rectify_dropout(dim=2):
    conv_cls = {
        0: torch.nn.Dropout,
        1: torch.nn.Dropout,
        2: torch.nn.Dropout2d,
        3: torch.nn.Dropout3d,
    }[dim]
    return conv_cls


def rectify_nonlinearity(key=ub.NoParam, dim=2):
    """
    Allows dictionary based specification of a nonlinearity

    Example:
        >>> rectify_nonlinearity('relu')
        ReLU(...)
        >>> rectify_nonlinearity('leaky_relu')
        LeakyReLU(negative_slope=0.01...)
        >>> rectify_nonlinearity(None)
        None
        >>> rectify_nonlinearity('swish')
    """
    if key is None:
        return None

    if key is ub.NoParam:
        key = 'relu'

    if isinstance(key, str):
        key = {'type': key}
    elif isinstance(key, dict):
        key = key.copy()
    else:
        raise TypeError(type(key))
    kw = key
    noli_type = kw.pop('type')
    if 'inplace' not in kw:
        kw['inplace'] = True

    if noli_type == 'leaky_relu':
        cls = torch.nn.LeakyReLU
    elif noli_type == 'relu':
        cls = torch.nn.ReLU
    elif noli_type == 'elu':
        cls = torch.nn.ELU
    elif noli_type == 'celu':
        cls = torch.nn.CELU
    elif noli_type == 'selu':
        cls = torch.nn.SELU
    elif noli_type == 'relu6':
        cls = torch.nn.ReLU6
    elif noli_type == 'swish':
        kw.pop('inplace', None)
        cls = Swish
    elif noli_type == 'mish':
        kw.pop('inplace', None)
        cls = Mish
    else:
        raise KeyError('unknown type: {}'.format(kw))
    return cls(**kw)


def rectify_normalizer(in_channels, key=ub.NoParam, dim=2, **kwargs):
    """
    Allows dictionary based specification of a normalizing layer

    Args:
        in_channels (int): number of input channels
        dim (int): dimensionality
        **kwargs: extra args

    Example:
        >>> rectify_normalizer(8)
        BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        >>> rectify_normalizer(8, 'batch')
        BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        >>> rectify_normalizer(8, {'type': 'batch'})
        BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        >>> rectify_normalizer(8, 'group')
        GroupNorm(4, 8, eps=1e-05, affine=True)
        >>> rectify_normalizer(8, {'type': 'group', 'num_groups': 2})
        GroupNorm(2, 8, eps=1e-05, affine=True)
        >>> rectify_normalizer(8, dim=3)
        BatchNorm3d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        >>> rectify_normalizer(8, None)
        None
        >>> rectify_normalizer(8, key={'type': 'syncbatch'})
        >>> rectify_normalizer(8, {'type': 'group', 'num_groups': 'auto'})
        >>> rectify_normalizer(1, {'type': 'group', 'num_groups': 'auto'})
        >>> rectify_normalizer(16, {'type': 'group', 'num_groups': 'auto'})
        >>> rectify_normalizer(32, {'type': 'group', 'num_groups': 'auto'})
        >>> rectify_normalizer(64, {'type': 'group', 'num_groups': 'auto'})
        >>> rectify_normalizer(1024, {'type': 'group', 'num_groups': 'auto'})
    """
    if key is None:
        return None

    if key is ub.NoParam:
        key = 'batch'

    if isinstance(key, str):
        key = {'type': key}
    elif isinstance(key, dict):
        key = key.copy()
    else:
        raise TypeError(type(key))

    norm_type = key.pop('type')
    if norm_type == 'batch':
        in_channels_key = 'num_features'

        if dim == 0:
            cls = torch.nn.BatchNorm1d
        elif dim == 1:
            cls = torch.nn.BatchNorm1d
        elif dim == 2:
            cls = torch.nn.BatchNorm2d
        elif dim == 3:
            cls = torch.nn.BatchNorm3d
        else:
            raise ValueError(dim)
    elif norm_type == 'syncbatch':
        in_channels_key = 'num_features'
        cls = torch.nn.SyncBatchNorm
    elif norm_type == 'group':
        in_channels_key = 'num_channels'
        if key.get('num_groups') is None:
            key['num_groups'] = 'auto'
            # key['num_groups'] = ('gcd', min(in_channels, 32))

        if key.get('num_groups') == 'auto':
            if in_channels == 1:
                # Warning: cant group norm this
                return Identity()
            else:
                valid_num_groups = [
                    factor for factor in range(1, in_channels)
                    if in_channels % factor == 0
                ]
                if len(valid_num_groups) == 0:
                    raise Exception
                infos = [
                    {'ng': ng, 'nc': in_channels / ng}
                    for ng in valid_num_groups
                ]
                ideal = in_channels ** (0.5)
                for item in infos:
                    item['heuristic'] = abs(ideal - item['ng']) * abs(ideal - item['nc'])
                chosen = sorted(infos, key=lambda x: (x['heuristic'], 1 - x['ng']))[0]
                key['num_groups'] = chosen['ng']
                if key['num_groups'] == in_channels:
                    key['num_groups'] = 1

            if isinstance(key['num_groups'], tuple):
                if key['num_groups'][0] == 'gcd':
                    key['num_groups'] = gcd(
                        key['num_groups'][1], in_channels)

            if in_channels % key['num_groups'] != 0:
                raise AssertionError(
                    'Cannot divide n_inputs {} by num groups {}'.format(
                        in_channels, key['num_groups']))
        cls = torch.nn.GroupNorm

    elif norm_type == 'batch+group':
        return torch.nn.Sequential(
            rectify_normalizer(in_channels, 'batch', dim=dim),
            rectify_normalizer(in_channels, ub.dict_union({'type': 'group'}, key), dim=dim),
        )
    else:
        raise KeyError('unknown type: {}'.format(key))
    assert in_channels_key not in key
    key[in_channels_key] = in_channels

    try:
        import copy
        kw = copy.copy(key)
        kw.update(kwargs)
        return cls(**kw)
    except Exception:
        raise
        # Ignore kwargs
        import warnings
        warnings.warn('kwargs ignored in rectify normalizer')
        return cls(**key)


def _ws_extra_repr(self):
    s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
         ', stride={stride}')
    if self.padding != (0,) * len(self.padding):
        s += ', padding={padding}'
    if self.dilation != (1,) * len(self.dilation):
        s += ', dilation={dilation}'
    if self.output_padding != (0,) * len(self.output_padding):
        s += ', output_padding={output_padding}'
    if self.groups != 1:
        s += ', groups={groups}'
    if self.bias is None:
        s += ', bias=False'
    if self.padding_mode != 'zeros':
        s += ', padding_mode={padding_mode}'
    if self.standardize_weights:
        s += ', standardize_weights={standardize_weights}'
    return s.format(**self.__dict__)


class Identity(torch.nn.Sequential):
    """
    A identity-function layer.

    Example:
        >>> import torch
        >>> self = Identity()
        >>> a = torch.rand(3, 3)
        >>> b = self(a)
        >>> assert torch.all(a == b)
    """
    def __init__(self):
        super(Identity, self).__init__()


class Conv0d(torch.nn.Linear):
    """
    Ignore:
        self = Conv0d(2, 3, 1, standardize_weights=True)
        print('self = {!r}'.format(self))
        x = torch.rand(1, 2)
        y = self.forward(x)
        print('y = {!r}'.format(y))
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', standardize_weights=False):
        assert kernel_size == 1, 'Conv0D must have a kernel_size=1'
        assert padding == 0, 'Conv0D must have padding=1'
        assert stride == 1, 'Conv0D must have stride=1'
        assert groups == 1, 'Conv0D must have groups=1'
        assert dilation == 1, 'Conv0D must have a dilation=1'
        # assert padding_mode == 'zeros'
        super().__init__(in_features=in_channels, out_features=out_channels,
                         bias=bias)
        self.standardize_weights = standardize_weights
        if standardize_weights:
            assert in_channels > 1, 'must be greater than 1 to prevent nan'
        self.dim = 0
        self.eps = 1e-5

    def forward(self, x):
        if self.standardize_weights:
            weight = weight_standardization_nd(self.dim, self.weight, self.eps)
            return torch.nn.functional.linear(x, weight, self.bias)
        else:
            return super().forward(x)

    def extra_repr(self) -> str:
        s = 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
        if self.standardize_weights:
            s += ', standardize_weights={standardize_weights}'
        return s


class Conv1d(torch.nn.Conv1d):
    """
    Ignore:
        self = Conv1d(2, 3, 1, standardize_weights=True)
        print('self = {!r}'.format(self))
        x = torch.rand(1, 2, 1)
        y = self.forward(x)
        print('y = {!r}'.format(y))
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 standardize_weights=False):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)
        self.standardize_weights = standardize_weights
        if standardize_weights:
            assert in_channels > 1, 'must be greater than 1 to prevent nan'
        self.eps = 1e-5
        self.dim = 1

    def forward(self, x):
        if self.standardize_weights:
            weight = weight_standardization_nd(self.dim, self.weight, self.eps)
            return torch.nn.functional.conv1d(
                x, weight, self.bias, self.stride, self.padding, self.dilation,
                self.groups)
        else:
            return super().forward(x)

    extra_repr = _ws_extra_repr


class Conv2d(torch.nn.Conv2d):
    """
    Ignore:
        self = Conv2d(2, 3, 1, standardize_weights=True)
        print('self = {!r}'.format(self))
        x = torch.rand(1, 2, 3, 3)
        y = self.forward(x)
        print('y = {!r}'.format(y))
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 standardize_weights=False):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)
        self.standardize_weights = standardize_weights
        if standardize_weights:
            assert in_channels > 1, 'must be greater than 1 to prevent nan'
        self.eps = 1e-5
        self.dim = 2

    def forward(self, x):
        if self.standardize_weights:
            weight = weight_standardization_nd(self.dim, self.weight, self.eps)
            return torch.nn.functional.conv2d(
                x, weight, self.bias, self.stride, self.padding, self.dilation,
                self.groups)
        else:
            return super().forward(x)

    extra_repr = _ws_extra_repr


class Conv3d(torch.nn.Conv3d):
    """
    Ignore:
        self = Conv3d(2, 3, 1, standardize_weights=True)
        print('self = {!r}'.format(self))
        x = torch.rand(1, 2, 1, 1, 1)
        y = self.forward(x)
        print('y = {!r}'.format(y))
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 standardize_weights=False):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)
        self.standardize_weights = standardize_weights
        if standardize_weights:
            assert in_channels > 1, 'must be greater than 1 to prevent nan'
        self.eps = 1e-5
        self.dim = 3

    def forward(self, x):
        if self.standardize_weights:
            weight = weight_standardization_nd(self.dim, self.weight, self.eps)
            return torch.nn.functional.conv3d(
                x, weight, self.bias, self.stride, self.padding, self.dilation,
                self.groups)
        else:
            return super().forward(x)

    extra_repr = _ws_extra_repr


def rectify_conv(dim=2):
    conv_cls = {
        0: Conv0d,
        # 1: torch.nn.Conv1d,
        # 2: torch.nn.Conv2d,
        # 3: torch.nn.Conv3d,
        1: Conv1d,
        2: Conv2d,
        3: Conv3d,
    }[dim]
    return conv_cls


def weight_standardization_nd(dim, weight, eps):
    """
    Note: input channels must be greater than 1!

    Ignore:
        weight = torch.rand(3, 2, 1, 1)
        dim = 2
        eps = 1e-5
        weight_normed = weight_standardization_nd(dim, weight, eps)
        print('weight = {!r}'.format(weight))
        print('weight_normed = {!r}'.format(weight_normed))

        weight = torch.rand(1, 2)
        dim = 0
        eps = 1e-5
        weight_normed = weight_standardization_nd(dim, weight, eps)
        print('weight = {!r}'.format(weight))
        print('weight_normed = {!r}'.format(weight_normed))
    """
    # Note: In 2D Weight dimensions are [C_out, C_in, H, W]
    mean_dims = tuple(list(range(1, dim + 2)))
    weight_mean = weight.mean(dim=mean_dims, keepdim=True)
    weight = weight - weight_mean
    trailing = [1] * (dim + 1)
    std = weight.view(weight.shape[0], -1).std(dim=1).view(-1, *trailing) + eps
    weight = weight / std.expand_as(weight)
    return weight


class ConvNormNd(torch.nn.Sequential):
    """
    Backbone convolution component. The convolution hapens first, normalization
    and nonlinearity happen after the convolution.

    CONV[->NORM][->NOLI]

    Args:
        dim (int):
            dimensionality of the convolutional kernel (can be 0, 1, 2, or 3).

        in_channels (int):

        out_channels (int):

        kernel_size (int | Tuple):

        stride (int | Tuple):

        padding (int | Tuple):

        dilation (int | Tuple):

        groups (int):

        bias (bool):

        norm (str, dict, nn.Module): Type of normalizer,
            if None, then normalization is disabled.

        noli (str, dict, nn.Module): Type of nonlinearity,
            if None, then normalization is disabled.

        standardize_weights (bool, default=False):
            Implements weight standardization as described in Qiao 2020 -
            "Micro-Batch Training with Batch-Channel Normalization and Weight
            Standardization"- https://arxiv.org/pdf/1903.10520.pdf

    Example:
        >>> self = ConvNormNd(dim=2, in_channels=16, out_channels=64,
        >>>                    kernel_size=3)
        >>> print(self)
        ConvNormNd(
          (conv): Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1))
          (norm): BatchNorm2d(64, ...)
          (noli): ReLU(...)
        )

    Example:
        >>> self = ConvNormNd(dim=0, in_channels=16, out_channels=64)
        >>> print(self)
        ConvNormNd(
          (conv): Conv0d(in_features=16, out_features=64, bias=True)
          (norm): BatchNorm1d(64, ...)
          (noli): ReLU(...)
        )
        >>> input_shape = (None, 16)
    """
    def __init__(self, dim, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, noli='relu',
                 norm='batch', standardize_weights=False):
        super(ConvNormNd, self).__init__()

        conv_cls = rectify_conv(dim)
        conv = conv_cls(in_channels, out_channels, kernel_size=kernel_size,
                        padding=padding, stride=stride, groups=groups,
                        bias=bias, dilation=dilation,
                        standardize_weights=standardize_weights)

        norm = rectify_normalizer(out_channels, norm, dim=dim)
        noli = rectify_nonlinearity(noli, dim=dim)

        self.add_module('conv', conv)
        if norm:
            self.add_module('norm', norm)
        if noli:
            self.add_module('noli', noli)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.standardize_weights = standardize_weights
        self._dim = dim


class ConvNorm1d(ConvNormNd):
    """
    Backbone convolution component. The convolution hapens first, normalization
    and nonlinearity happen after the convolution.

    CONV[->NORM][->NOLI]

    Args:
        norm (str, dict, nn.Module): Type of normalizer,
            if None, then normalization is disabled.
        noli (str, dict, nn.Module): Type of nonlinearity,
            if None, then normalization is disabled.

    Example:
        >>> input_shape = [2, 3, 5]
        >>> self = ConvNorm1d(input_shape[1], 7, kernel_size=3)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, noli='relu',
                 norm='batch', standardize_weights=False):
        super(ConvNorm1d, self).__init__(dim=1, in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride, bias=bias,
                                         padding=padding, noli=noli, norm=norm,
                                         dilation=dilation, groups=groups,
                                         standardize_weights=standardize_weights)


class ConvNorm2d(ConvNormNd):
    """
    Backbone convolution component. The convolution hapens first, normalization
    and nonlinearity happen after the convolution.

    CONV[->NORM][->NOLI]

    Args:
        norm (str, dict, nn.Module): Type of normalizer,
            if None, then normalization is disabled.
        noli (str, dict, nn.Module): Type of nonlinearity,
            if None, then normalization is disabled.

    Example:
        >>> input_shape = [2, 3, 5, 7]
        >>> self = ConvNorm2d(input_shape[1], 11, kernel_size=3)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, noli='relu',
                 norm='batch', standardize_weights=False):
        super(ConvNorm2d, self).__init__(dim=2, in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride, bias=bias,
                                         padding=padding, noli=noli, norm=norm,
                                         dilation=dilation, groups=groups,
                                         standardize_weights=standardize_weights)


class ConvNorm3d(ConvNormNd):
    """
    Backbone convolution component. The convolution hapens first, normalization
    and nonlinearity happen after the convolution.

    CONV[->NORM][->NOLI]

    Args:
        norm (str, dict, nn.Module): Type of normalizer,
            if None, then normalization is disabled.
        noli (str, dict, nn.Module): Type of nonlinearity,
            if None, then normalization is disabled.

    Example:
        >>> input_shape = [2, 3, 5, 7, 11]
        >>> self = ConvNorm3d(input_shape[1], 13, kernel_size=3)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 bias=True, padding=0, noli='relu', norm='batch',
                 groups=1, standardize_weights=False):
        super(ConvNorm3d, self).__init__(dim=3, in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride, bias=bias,
                                         padding=padding, noli=noli, norm=norm,
                                         groups=groups,
                                         standardize_weights=standardize_weights)


class _SwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(torch.nn.Module):
    """
    When beta=1 this is Sigmoid-weighted Linear Unit (SiL)

    ``x * torch.sigmoid(x)``

    References:
        https://arxiv.org/pdf/1710.05941.pdf

    Example:
        >>> from geowatch.utils.util_netharn import *  # NOQA
        >>> x = torch.linspace(-20, 20, 100, requires_grad=True)
        >>> self = Swish()
        >>> y = self(x)
        >>> y.sum().backward()
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.multi_plot(xydata={'beta=1': (x.data, y.data)}, fnum=1, pnum=(1, 2, 1),
        >>>         ylabel='swish(x)', xlabel='x', title='activation')
        >>> kwplot.multi_plot(xydata={'beta=1': (x.data, x.grad)}, fnum=1, pnum=(1, 2, 2),
        >>>         ylabel='𝛿swish(x) / 𝛿(x)', xlabel='x', title='gradient')
        >>> kwplot.show_if_requested()

    """
    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        """
        Equivalent to ``x * torch.sigmoid(x)``
        """
        if self.beta == 1:
            return _SwishFunction.apply(x)
        else:
            return x * torch.sigmoid(x * self.beta)


@torch.jit.script
def mish(input):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    """
    return input * torch.tanh(F.softplus(input))


def beta_mish(input, beta=1.5):
    """
    Applies the β mish function element-wise:
        .. math::
            \\beta mish(x) = x * tanh(ln((1 + e^{x})^{\\beta}))
    See additional documentation for :mod:`echoAI.Activation.Torch.beta_mish`.

    References:
        https://github.com/digantamisra98/Echo/blob/master/echoAI/Activation/Torch/functional.py
    """
    return input * torch.tanh(torch.log(torch.pow((1 + torch.exp(input)), beta)))


class Mish_Function(torch.autograd.Function):
    """
    Applies the mish function element-wise:

    Math:
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    References:
        https://github.com/digantamisra98/Echo/blob/master/echoAI/Activation/Torch/mish.py

    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y = x * torch.tanh(F.softplus(x))  # x * tanh(ln(1 + exp(x)))
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid = torch.sigmoid(x)
        tanh_sp = torch.tanh(F.softplus(x))
        return grad_output * (tanh_sp + x * sigmoid * (1 - tanh_sp * tanh_sp))

    # else:
    #     @torch.jit.script
    #     def mish(input):
    #         delta = torch.exp(-input)
    #         alpha = 1 + 2 * delta
    #         return input * alpha / (alpha + 2 * delta * delta)


class Mish(torch.nn.Module):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    References:
        https://github.com/digantamisra98/Mish/blob/master/Mish/Torch/mish.py
        https://github.com/thomasbrandon/mish-cuda
        https://arxiv.org/pdf/1908.08681v2.pdf

    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)

    Example:
        >>> from geowatch.utils.util_netharn import *  # NOQA
        >>> x = torch.linspace(-20, 20, 100, requires_grad=True)
        >>> self = Mish()
        >>> y = self(x)
        >>> y.sum().backward()
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.multi_plot(xydata={'beta=1': (x.data, y.data)}, fnum=1, pnum=(1, 2, 1))
        >>> kwplot.multi_plot(xydata={'beta=1': (x.data, x.grad)}, fnum=1, pnum=(1, 2, 2))
        >>> kwplot.show_if_requested()
    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return Mish_Function.apply(input)
        # return mish(input)


def _debug_inbatch_shapes(inbatch):
    import ubelt as ub
    print('len(inbatch) = {}'.format(len(inbatch)))
    extensions = ub.util_format.FormatterExtensions()
    #
    @extensions.register((torch.Tensor, np.ndarray))
    def format_shape(data, **kwargs):
        return ub.repr2(dict(type=str(type(data)), shape=data.shape), nl=1, sv=1)
    print('inbatch = ' + ub.repr2(inbatch, extensions=extensions, nl=True))


def default_kwargs(cls):
    """
    Grab initkw defaults from the constructor

    Args:
        cls (type | callable): a class or function

    Example:
        >>> from geowatch.utils.util_netharn import *  # NOQA
        >>> import torch
        >>> import ubelt as ub
        >>> cls = torch.optim.Adam
        >>> default_kwargs(cls)
        >>> cls = KaimingNormal
        >>> print(ub.repr2(default_kwargs(cls), nl=0))
        {'mode': 'fan_in', 'param': 0}
        >>> cls = NoOp
        >>> default_kwargs(cls)
        {}

    SeeAlso:
        xinspect.get_func_kwargs(cls)
    """
    import inspect
    sig = inspect.signature(cls)
    default_kwargs = {
        k: p.default
        for k, p in sig.parameters.items()
        if p.default is not p.empty
    }
    return default_kwargs


default_collate = torch_data.dataloader.default_collate


def _collate_else(batch, collate_func):
    """
    Handles recursion in the else case for these special collate functions

    This is duplicates all non-tensor cases from `torch_data.dataloader.default_collate`
    This also contains support for collating slices.
    """
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], slice):
        batch = default_collate([{
            'start': sl.start,
            'stop': sl.stop,
            'step': 1 if sl.step is None else sl.step
        } for sl in batch])
        return batch
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], container_abcs.Mapping):
        if 0:
            # Hack the mapping collation implementation to print error info
            collated = {}
            try:
                for key in batch[0]:
                    collated[key] = collate_func([d[key] for d in batch])
            except Exception:
                print('\n!!Error collating key = {!r}\n'.format(key))
                raise
            return collated
        else:
            return {key: collate_func([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], tuple) and hasattr(batch[0], '_fields'):  # namedtuple
        return type(batch[0])(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(batch[0], container_abcs.Sequence):
        transposed = zip(*batch)
        return [collate_func(samples) for samples in transposed]
    else:
        raise TypeError((error_msg.format(type(batch[0]))))


class CollateException(Exception):
    pass


def padded_collate(inbatch, fill_value=-1):
    """
    Used for detection datasets with boxes.

    Example:
        >>> from geowatch.utils.util_netharn import *  # NOQA
        >>> import torch
        >>> rng = np.random.RandomState(0)
        >>> inbatch = []
        >>> bsize = 7
        >>> for i in range(bsize):
        >>>     # add an image and some dummy bboxes to the batch
        >>>     img = torch.rand(3, 8, 8)  # dummy 8x8 image
        >>>     n = 11 if i == 3 else rng.randint(0, 11)
        >>>     boxes = torch.rand(n, 4)
        >>>     item = (img, boxes)
        >>>     inbatch.append(item)
        >>> out_batch = padded_collate(inbatch)
        >>> assert len(out_batch) == 2
        >>> assert list(out_batch[0].shape) == [bsize, 3, 8, 8]
        >>> assert list(out_batch[1].shape) == [bsize, 11, 4]

    Example:
        >>> import torch
        >>> rng = np.random.RandomState(0)
        >>> inbatch = []
        >>> bsize = 4
        >>> for _ in range(bsize):
        >>>     # add an image and some dummy bboxes to the batch
        >>>     img = torch.rand(3, 8, 8)  # dummy 8x8 image
        >>>     #boxes = torch.empty(0, 4)
        >>>     boxes = torch.FloatTensor()
        >>>     item = (img, [boxes])
        >>>     inbatch.append(item)
        >>> out_batch = padded_collate(inbatch)
        >>> assert len(out_batch) == 2
        >>> assert list(out_batch[0].shape) == [bsize, 3, 8, 8]
        >>> #assert list(out_batch[1][0].shape) == [bsize, 0, 4]
        >>> assert list(out_batch[1][0].shape) in [[0], []]  # torch .3 a .4

    Example:
        >>> inbatch = [torch.rand(4, 4), torch.rand(8, 4),
        >>>            torch.rand(0, 4), torch.rand(3, 4),
        >>>            torch.rand(0, 4), torch.rand(1, 4)]
        >>> out_batch = padded_collate(inbatch)
        >>> assert list(out_batch.shape) == [6, 8, 4]
    """
    try:
        if torch.is_tensor(inbatch[0]):
            num_items = [len(item) for item in inbatch]
            if ub.allsame(num_items):
                if len(num_items) == 0:
                    batch = torch.FloatTensor()
                elif num_items[0] == 0:
                    batch = torch.FloatTensor()
                else:
                    batch = default_collate(inbatch)
            else:
                max_size = max(num_items)
                real_tail_shape = None
                for item in inbatch:
                    if item.numel():
                        tail_shape = item.shape[1:]
                        if real_tail_shape is not None:
                            assert real_tail_shape == tail_shape
                        real_tail_shape = tail_shape

                padded_inbatch = []
                for item in inbatch:
                    n_extra = max_size - len(item)
                    if n_extra > 0:
                        shape = (n_extra,) + tuple(real_tail_shape)
                        if torch.__version__.startswith('0.3'):
                            extra = torch.Tensor(np.full(shape, fill_value=fill_value))
                        else:
                            extra = torch.full(shape, fill_value=fill_value,
                                               dtype=item.dtype)
                        padded_item = torch.cat([item, extra], dim=0)
                        padded_inbatch.append(padded_item)
                    else:
                        padded_inbatch.append(item)
                batch = inbatch
                batch = default_collate(padded_inbatch)
        else:
            batch = _collate_else(inbatch, padded_collate)
    except Exception as ex:
        if not isinstance(ex, CollateException):
            try:
                _debug_inbatch_shapes(inbatch)
            except Exception:
                pass
            raise CollateException(
                'Failed to collate inbatch={}. Reason: {!r}'.format(inbatch, ex))
        else:
            raise
    return batch
