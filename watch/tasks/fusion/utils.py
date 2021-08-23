from torch import nn
import inspect
import torch
import numpy as np
import math
from torch import package

millnames = ['', ' K', ' M', ' B', ' T']


def millify(n):
    n = float(n)
    millidx = max(0, min(len(millnames) - 1,
                         int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))))

    return '{:.2f}{}'.format(n / 10**(3 * millidx), millnames[millidx])


# def create_package(model, package_path, module_name="watch_tasks_fusion", arch_name="model.pkl", verbose=False):
#     """
#     DEPRECATE IN FAVOR OF A MODEL METHOD?
#
#     CommandLine:
#         xdoctest watch.tasks.fusion.utils create_package
#
#     Example:
#         >>> import ubelt as ub
#         >>> from os.path import join
#         >>> from watch.tasks.fusion.utils import *  # NOQA
#         >>> dpath = ub.ensure_app_cache_dir('watch/tests/package')
#         >>> package_path = join(dpath, 'my_package.pt')
#
#         >>> # Use one of our fusion.architectures in a test
#         >>> from watch.tasks.fusion import methods
#         >>> model = methods.MultimodalTransformerDirectCD("smt_it_stm_p8")
#         >>> # We have to run an input through the module because it is lazy
#         >>> inputs = torch.rand(1, 2, 13, 128, 128)
#         >>> model(inputs)
#
#         >>> # Save the model
#         >>> create_package(model, package_path)
#
#         >>> # Test that the package can be reloaded
#         >>> recon = load_model_from_package(package_path)
#         >>> # Check consistency and data is actually different
#         >>> recon_state = recon.state_dict()
#         >>> model_state = model.state_dict()
#         >>> assert recon is not model
#         >>> assert set(recon_state) == set(recon_state)
#         >>> for key in recon_state.keys():
#         >>>     assert (model_state[key] == recon_state[key]).all()
#         >>>     assert model_state[key] is not recon_state[key]
#     """
#     with package.PackageExporter(package_path, verbose=verbose) as exp:
#         # TODO: this is not a problem yet, but some package types (mainly binaries) will need to be excluded and added as mocks
#         exp.extern("**", exclude=["watch.tasks.fusion.**"])
#         exp.intern("watch.tasks.fusion.**")
#         exp.save_pickle(module_name, arch_name, model)


def load_model_from_package(package_path, module_name="watch_tasks_fusion", arch_name="model.pkl"):
    """
    DEPRECATE IN FAVOR OF A MODEL METHOD?

    Notes:
        * I don't like that we need to know module_name and arch_name a-priori
          given a path to a package, I just want to be able to construct
          the model instance.
    """
    # imp = package.PackageImporter(package_path)
    import pathlib
    if not isinstance(package_path, (str, pathlib.Path)):
        raise TypeError(type(package_path))

    imp = package.PackageImporter(package_path)
    # Assume this standardized header information exists that tells us the
    # name of the resource corresponding to the model
    package_header = imp.load_pickle(
        'kitware_package_header', 'kitware_package_header.pkl')
    arch_name = package_header['arch_name']
    module_name = package_header['module_name']

    return imp.load_pickle(module_name, arch_name)


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


class AddPositionalEncoding(nn.Module):
    def __init__(self, dest_dim, dims_to_encode):
        super().__init__()
        self.dest_dim = dest_dim
        self.dims_to_encode = dims_to_encode
        assert self.dest_dim not in self.dims_to_encode

    def forward(self, x):

        inds = [
            slice(0, size) if (dim in self.dims_to_encode) else slice(0, 1)
            for dim, size in enumerate(x.shape)
        ]
        inds[self.dest_dim] = self.dims_to_encode

        encoding = torch.cat(torch.meshgrid([
            torch.linspace(0, 1, x.shape[dim]) if (dim in self.dims_to_encode) else torch.tensor(-1.)
            for dim in range(len(x.shape))
        ]), dim=self.dest_dim)[inds]

        expanded_shape = list(x.shape)
        expanded_shape[self.dest_dim] = -1
        x = torch.cat([x, encoding.expand(expanded_shape).type_as(x)], dim=self.dest_dim)
        return x


class SinePositionalEncoding(nn.Module):
    """
    Example:
        >>> from watch.tasks.fusion.utils import *  # NOQA
        >>> dest_dim = 3
        >>> dim_to_encode = 2
        >>> sine_pairs = 4
        >>> self = SinePositionalEncoding(dest_dim, dim_to_encode, sine_pairs=sine_pairs)
        >>> x = torch.rand(3, 5, 7, 11, 13)
        >>> y = self(x)
    """

    def __init__(self, dest_dim, dim_to_encode, sine_pairs=2):
        super().__init__()
        self.dest_dim = dest_dim
        self.dim_to_encode = dim_to_encode
        self.sine_pairs = sine_pairs
        assert self.dest_dim != self.dim_to_encode

    def forward(self, x):
        expanded_shape = list(x.shape)
        expanded_shape[self.dest_dim] = -1

        expand_dims = [None] * len(x.shape)
        expand_dims[self.dim_to_encode] = slice(0, None)
        expand_dims[self.dest_dim] = slice(0, None)

        def scale(d):
            return 1 / 10000 ** (d)

        parts = []
        for idx in range(2 * self.sine_pairs):
            theta = torch.arange(x.shape[self.dim_to_encode]) * scale(idx / (2 * self.sine_pairs))
            if idx % 2 == 0:
                part = torch.sin(theta)
            else:
                part = torch.cos(theta)
            parts.append(part)

        encoding = torch.stack(parts, dim=1)

        encoding = encoding[expand_dims].expand(expanded_shape)

        x = torch.cat([x, encoding.type_as(x)], dim=self.dest_dim)
        return x


# def filter_args(args, func):
#     signature = inspect.signature(func)
#     return {
#         key: value
#         for key, value in args.items()
#         if key in signature.parameters.keys()
#     }


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

    def Affine_concise(aff):
        """
        TODO: add to kwimage.Affine
        """
        import numpy as np
        params = aff.decompose()
        params['type'] = 'affine'
        if np.allclose(params['offset'], (0, 0)):
            params.pop('offset')
        if np.allclose(params['scale'], (1, 1)):
            params.pop('scale')
        if np.allclose(params['shear'], 0):
            params.pop('shear')
        if np.allclose(params['theta'], 0):
            params.pop('theta')
        return params

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
        aux = {**extra_info, **aux}

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
