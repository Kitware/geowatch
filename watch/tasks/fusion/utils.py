from torch import nn
import torch
import os
import numpy as np
import math
import ubelt as ub
import kwimage

millnames = ['', ' K', ' M', ' B', ' T']


try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


def millify(n):
    n = float(n)
    millidx = max(0, min(len(millnames) - 1,
                         int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))))

    return '{:.2f}{}'.format(n / 10**(3 * millidx), millnames[millidx])


def load_model_from_package(package_path):
    """
    Loads a kitware-flavor torch package (requires a package_header exists)

    Notes:
        * I don't like that we need to know module_name and arch_name a-priori
          given a path to a package, I just want to be able to construct
          the model instance. The package header solves this.

    Ignore:
        >>> from watch.tasks.fusion.utils import *  # NOQA
        >>> import watch
        >>> dvc_dpath = watch.utils.util_data.find_smart_dvc_dpath()
        >>> package_path = dvc_dpath / 'models/fusion/SC-20201117/SC_smt_it_stm_p8_newanns_weighted_rgb_v26/SC_smt_it_stm_p8_newanns_weighted_rgb_v26_epoch=101-step=4366925.pt'
        >>> model = load_model_from_package(package_path)
    """
    from watch.monkey import monkey_torchmetrics
    from watch.monkey import monkey_kwcoco
    monkey_torchmetrics.fix_torchmetrics_compatability()
    monkey_kwcoco.fix_sorted_set()
    from torch import package
    import json
    # imp = package.PackageImporter(package_path)
    import pathlib
    if not isinstance(package_path, (str, pathlib.Path)):
        raise TypeError(type(package_path))

    package_path = os.fspath(package_path)
    imp = package.PackageImporter(package_path)
    # Assume this standardized header information exists that tells us the
    # name of the resource corresponding to the model
    try:
        package_header = json.loads(imp.load_text(
            'package_header', 'package_header.json'))
    except Exception:
        print('warning: no standard package header')
        try:
            package_header = json.loads(imp.load_text(
                'kitware_package_header', 'kitware_package_header.json'))
        except Exception:
            package_header = imp.load_pickle(
                'kitware_package_header', 'kitware_package_header.pkl')
        print('warning: old package header?')
    arch_name = package_header['arch_name']
    module_name = package_header['module_name']

    model = imp.load_pickle(module_name, arch_name)

    if 0:
        imp.file_structure()['package_header']

    # Add extra metadata to the model
    raise Exception("foo")
    config_candidates = {
        "config_cli_yaml": "config.yaml",
        "fit_config": "fit_config.yaml",
    }
    for candidate_dest, candidate_fpath in config_candidates.items():
        try:
            fit_config_text = imp.load_text('package_header', candidate_fpath)
        except Exception:
            pass
        else:
            import io
            import yaml
            file = io.StringIO(fit_config_text)
            # Note: types might be wrong here
            fit_config = yaml.safe_load(file)
            # model.fit_config = fit_config
            setattr(model, candidate_dest, fit_config)

    model.package_path = package_path
    return model


class Lambda(nn.Module):
    def __init__(self, lambda_):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return self.lambda_(x)


class DimensionDropout(nn.Module):
    def __init__(self, dim, n_keep):
        super().__init__()
        self.dim = dim
        self.n_keep = n_keep

    def forward(self, x):
        shape = x.shape
        dim_size = shape[self.dim]

        index = [slice(0, None)] * len(shape)
        index[self.dim] = torch.randperm(dim_size)[:self.n_keep]

        return x[index]


def ordinal_position_encoding(num_items, feat_size, method='sin', device='cpu'):
    """
    A positional encoding that represents ordinal

    Args:
        num_items (int): number of dimensions to be encoded (
            e.g. this is a spatial or temporal index)
        feat_size (int): this is the number of dimensions in the positional
             encoding generated for each dimension / item

    Example:
        >>> # Use 5 feature dimensions to encode 3 timesteps
        >>> import sys, ubelt
        >>> sys.path.append(ubelt.expandpath('~/code/watch'))
        >>> from watch.tasks.fusion.utils import *  # NOQA
        >>> from watch.tasks.fusion.utils import _memo_legend
        >>> num_timesteps = num_items = 3
        >>> feat_size = 5
        >>> encoding = ordinal_position_encoding(num_items, feat_size)
    """
    assert method == 'sin'
    sf = 10000
    parts = []
    base = torch.arange(num_items, device=device)
    for idx in range(feat_size):
        exponent = (idx / feat_size)
        modulator = (1 / (sf ** exponent))
        theta = base * modulator
        if idx % 2 == 0:
            part = torch.sin(theta)
        else:
            part = torch.cos(theta)
        parts.append(part)
    encoding = torch.stack(parts, dim=1)
    return encoding


class SinePositionalEncoding(nn.Module):
    """
    Args:
        dest_dim (int): feature dimension to concat to
        dim_to_encode (int): dimension encoding is supposed to represent
        size (int): number of different encodings for the dim_to_encode

    Example:
        >>> from watch.tasks.fusion.utils import *  # NOQA
        >>> dest_dim = 3
        >>> dim_to_encode = 2
        >>> size = 8
        >>> self = SinePositionalEncoding(dest_dim, dim_to_encode, size=size)
        >>> x = torch.rand(3, 5, 7, 11, 13)
        >>> y = self(x)

    Ignore:
        >>> from watch.tasks.fusion.utils import *  # NOQA
        >>> self = SinePositionalEncoding(1, 0, size=8)
        >>> encoding = self._encoding_part(10)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> import pandas as pd
        >>> sns = kwplot.autosns()
        >>> df = pd.concat([pd.DataFrame({'x': np.arange(len(part)), 'y': part, 'part': [idx] * len(part)}) for idx, part in enumerate(encoding.T)]).reset_index()
        >>> fig = kwplot.figure(pnum=(1, 2, 1))
        >>> ax = fig.gca()
        >>> sns.lineplot(data=df, x='x', y='y', hue='part')
        >>> kwplot.imshow(kwarray.normalize(encoding.numpy()).T, pnum=(1, 2, 2), cmap='magma')
    """

    def __init__(self, dest_dim, dim_to_encode, size=4):
        super().__init__()
        self.dest_dim = dest_dim
        self.dim_to_encode = dim_to_encode
        self.size = size
        assert self.dest_dim != self.dim_to_encode

    def _encoding_part(self, num, device='cpu'):
        sf = 10000
        parts = []
        base = torch.arange(num, device=device)
        for idx in range(self.size):
            exponent = (idx / self.size)
            modulator = (1 / (sf ** exponent))
            theta = base * modulator
            if idx % 2 == 0:
                part = torch.sin(theta)
            else:
                part = torch.cos(theta)
            parts.append(part)
        encoding = torch.stack(parts, dim=1)
        return encoding

    @profile
    def forward(self, x):
        device = x.device
        expanded_shape = list(x.shape)
        expanded_shape[self.dest_dim] = -1

        expand_dims = [None] * len(expanded_shape)
        expand_dims[self.dim_to_encode] = slice(0, None)
        expand_dims[self.dest_dim] = slice(0, None)

        num = expanded_shape[self.dim_to_encode]

        encoding = self._encoding_part(num, device)
        encoding = encoding[expand_dims].expand(expanded_shape)

        x = torch.cat([x, encoding.type_as(x)], dim=self.dest_dim)
        return x


def add_auxiliary(dset, gid, fname, channels, aux_height, aux_width, warp_aux_to_img=None, extra_info=None):
    """
    Snippet for adding an auxiliary image

    Args:
        dset (CocoDataset)
        gid (int): image id to add auxiliary data to
        channels (str): name of the new auxiliary channels
        fname (str): path to save the new auxiliary channels (absolute or
            relative to dset.bundle_dpath)
        data (ndarray): actual auxiliary data
        warp_aux_to_img (kwimage.Affine): spatial relationship between
            auxiliary channel and the base image. If unspecified
            it is assumed that a simple scaling will suffice.

    Ignore:
        import kwcoco
        dset = kwcoco.CocoDataset.demo('shapes8')
        gid = 1
        data = np.random.rand(32, 55, 5)
        fname = 'myaux1.png'
        channels = 'hidden_logits'
        warp_aux_to_img = None
        add_auxiliary(dset, gid, fname, channels, data, warp_aux_to_img)

    """
    # from os.path import join
    import kwimage
    # fpath = join(dset.bundle_dpath, fname)
    # aux_height, aux_width = data.shape[0:2]
    img = dset.index.imgs[gid]

    if warp_aux_to_img is None:
        # Assume we can just scale up the auxiliary data to match the image
        # space unless the user says otherwise
        warp_aux_to_img = kwimage.Affine.scale((
            img['width'] / aux_width, img['height'] / aux_height))

    # Make the aux info dict
    aux = {
        'file_name': fname,
        'height': aux_height,
        'width': aux_width,
        'channels': channels,
        'warp_aux_to_img': kwimage.Affine.coerce(warp_aux_to_img).concise(),
    }

    if extra_info is not None:
        assert isinstance(extra_info, dict)
        aux = ub.dict_union(extra_info, aux)

    # Save the data to disk
    # kwimage.imwrite(fpath, data)

    auxiliary = img.setdefault('auxiliary', [])
    auxiliary.append(aux)
    dset._invalidate_hashid()


def confusion_image(pred, target):
    canvas = np.zeros_like(pred)
    np.putmask(canvas, (target == 0) & (pred == 0), 0)  # true-neg
    np.putmask(canvas, (target == 1) & (pred == 1), 1)  # true-pos
    np.putmask(canvas, (target == 1) & (pred == 0), 2)  # false-neg
    np.putmask(canvas, (target == 0) & (pred == 1), 3)  # false-pos
    return canvas


def model_json(model, max_depth=float('inf'), depth=0):
    """
    import torchvision
    model = torchvision.models.resnet50()
    info = model_json(model, max_depth=1)
    print(ub.repr2(info, sort=0, nl=-1))
    """
    info = {
        'type': model._get_name(),
    }
    params = model.extra_repr()
    if params:
        info['params'] = params

    if model._modules:
        if depth >= max_depth:
            info['children'] = ...
        else:
            children = {
                key: model_json(child, max_depth, depth=depth + 1)
                for key, child in model._modules.items()
            }
            info['children'] = children
    return info


@ub.memoize
def _memo_legend(label_to_color):
    import kwplot
    legend_img = kwplot.make_legend_img(label_to_color)
    return legend_img


def category_tree_ensure_color(classes):
    """
    Ensures that each category in a CategoryTree has a color

    TODO:
        - [ ] Add to CategoryTree
        - [ ] TODO: better function

    Example:
        >>> import kwcoco
        >>> classes = kwcoco.CategoryTree.demo()
        >>> assert not any('color' in data for data in classes.graph.nodes.values())
        >>> category_tree_ensure_color(classes)
        >>> assert all('color' in data for data in classes.graph.nodes.values())
    """
    backup_colors = iter(kwimage.Color.distinct(len(classes)))
    for node in classes.graph.nodes:
        color = classes.graph.nodes[node].get('color', None)
        if color is None:
            color = next(backup_colors)
            classes.graph.nodes[node]['color'] = kwimage.Color(color).as01()
