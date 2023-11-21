"""
Ported bits of netharn

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

"""
import torch
import ubelt as ub
import numpy as np  # NOQA


def _update_defaults(config, kw):
    config = dict(config)
    for k, v in kw.items():
        if k not in config:
            config[k] = v
    return config


class Initializer(object):
    """
    Base class for all netharn initializers
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
        >>> # xdoctest: +SKIP
        >>> import copy
        >>> self = NoOp()
        >>> model = toynet.ToyNet2d()
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
        >>> # xdoctest: +SKIP
        >>> self = Orthogonal()
        >>> model = toynet.ToyNet2d()
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
        >>> # xdoctest: +SKIP
        >>> from netharn.models import toynet
        >>> self = KaimingUniform()
        >>> model = toynet.ToyNet2d()
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
        >>> # xdoctest: +SKIP
        >>> from netharn.models import toynet
        >>> self = KaimingNormal()
        >>> model = toynet.ToyNet2d()
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
        >>> number_of_parameters(model)
        33
    """
    if trainable:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    else:
        model_parameters = model.parameters()
    n_params = sum([np.prod(p.size()) for p in model_parameters])
    return n_params
