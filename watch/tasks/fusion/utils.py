from torch import nn
import inspect
import torch
import numpy as np
import math
from torch import package

millnames = ['',' K',' M',' B',' T']

def millify(n):
    n = float(n)
    millidx = max(0,min(len(millnames)-1,
                        int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))

    return '{:.2f}{}'.format(n / 10**(3 * millidx), millnames[millidx])

def create_package(model, package_path, module_name="watch_tasks_fusion", model_name="model.pkl", verbose=False):
    with package.PackageExporter(package_path, verbose=verbose) as exp:
        # TODO: this is not a problem yet, but some package types will (mainly binaries) will need to be excluded also and added as mocks
        exp.extern("**", exclude=["watch.tasks.fusion.**"])
        exp.intern("watch.tasks.fusion.**")
        exp.save_pickle(module_name, model_name, model)
    
def load_model_from_package(package_path, module_name="watch_tasks_fusion", model_name="model.pkl"):
    imp = package.PackageImporter(package_path)
    return imp.load_pickle(module_name, model_name)

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
        
        index = [slice(0,None)] * len(shape)
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

        scale = lambda d: 1 / 10000 ** (d)

        encoding = torch.stack([
            torch.sin(torch.arange(x.shape[self.dim_to_encode]) * scale(idx / (2 * self.sine_pairs)))
            if idx % 2 == 0
            else torch.cos(torch.arange(x.shape[self.dim_to_encode]) * scale(idx / (2 * self.sine_pairs)))
            for idx in range(2 * self.sine_pairs)
        ], dim=1)

        encoding = encoding[expand_dims].expand(expanded_shape)

        x = torch.cat([x, encoding.type_as(x)], dim=self.dest_dim)
        return x


def filter_args(args, func):
    signature = inspect.signature(func)
    return {
        key: value
        for key, value in args.items()
        if key in signature.parameters.keys()
    }


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
    np.putmask(canvas, (target==0) & (pred==0), 0) # true-neg
    np.putmask(canvas, (target==1) & (pred==1), 1) # true-pos
    np.putmask(canvas, (target==1) & (pred==0), 2) # false-neg
    np.putmask(canvas, (target==0) & (pred==1), 3) # false-pos
    return canvas