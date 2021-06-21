from torch import nn
class Lambda(nn.Module):
    def __init__(self, lambda_):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return self.lambda_(x)

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
