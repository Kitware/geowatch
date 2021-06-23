from torch import nn
class Lambda(nn.Module):
    def __init__(self, lambda_):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return self.lambda_(x)
    
import torch
from torch import nn

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

        expand_dims = [None]*len(x.shape)
        expand_dims[self.dim_to_encode] = slice(0, None)
        expand_dims[self.dest_dim] = slice(0, None)
        
        scale = lambda d: 1 / 10000 ** (d)
        
        encoding = torch.stack([
            torch.sin(torch.arange(x.shape[self.dim_to_encode]) * scale(idx / (2*self.sine_pairs)))
            if idx % 2 == 0
            else torch.cos(torch.arange(x.shape[self.dim_to_encode]) * scale(idx / (2*self.sine_pairs)))
            for idx in range(2*self.sine_pairs)
        ], dim=1)
        
        encoding = encoding[expand_dims].expand(expanded_shape)

        x = torch.cat([x, encoding.type_as(x)], dim=self.dest_dim)
        return x

import inspect
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
    from os.path import join
    import kwimage
    fpath = join(dset.bundle_dpath, fname)
#     aux_height, aux_width = data.shape[0:2]
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
        'warp_aux_to_img': Affine_concise(warp_aux_to_img),
    }

    if extra_info is not None:
        assert isinstance(extra_info, dict)
        aux = {**extra_info, **aux}

    # Save the data to disk
#     kwimage.imwrite(fpath, data)

    auxiliary = img.setdefault('auxiliary', [])
    auxiliary.append(aux)
    dset._invalidate_hashid()
