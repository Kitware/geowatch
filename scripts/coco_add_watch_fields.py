"""
Adds fields needed by ndsampler to correctly "watch" a region.

Some of this is done hueristically. We assume images come from certain sensors.
We assume input is orthorectified.  We assume some GSD "target" gsd for video
and image processing. Note a video GSD will typically be much higher (i.e.
lower resolution) than an image GSD.
"""


def add_watch_fields(dset):
    """
    Args:
        dset (Dataset): dataset to work with

    Example:
        import sys, ubelt
        sys.path.append(ubelt.expandpath('~/code/watch/scripts'))
        from coco_add_watch_fields import *  # NOQA
        fpath = ub.expandpath('~/data/dvc-repos/smart_watch_dvc/drop0_aligned/data.kwcoco.json')
        coco_fpath = ub.expandpath('~/data/dvc-repos/smart_watch_dvc/drop0_aligned/data.kwcoco.json')
        import kwcoco
        dset = kwcoco.CocoDataset(coco_fpath)
        dset.conform(pycocotools_info=False)
    """
    import kwcoco
    import ubelt as ub
    # from os.path import join
    # root_dpath = ub.expandpath('~/data/dvc-repos/smart_watch_dvc/extern/onera_2018')
    # coco_fpath = join(root_dpath, 'onera_all.kwcoco.json')

    # Load your KW-COCO dataset (conform populates information like image size)

    for img in dset.imgs.values():
        # fpath = dset.get_image_fpath(img)
        # shape = kwimage.load_image_shape(fpath)
        # unique_shapes.add(shape)

        sensor_coarse = img.get('sensor_coarse', None)
        num_bands = img.get('num_bands', None)

        channels = img.get('channels', None)
        assert channels is None

        channels = channel_hueristic(sensor_coarse, num_bands)
        img['channels'] = channels

        # unique_chans.add(chan)
        auxillary = img.get('auxiliary', [])
        if auxillary:
            raise NotImplementedError
        # for aux in auxillary:
        #     chan = aux['channels']
        #     unique_chans.add(chan)
        #     aux['fpath']

    dset.dump(dset.fpath, newlines=True)




def channel_hueristic(sensor_coarse, num_bands):
    """
    Given a sensor and the number of bands in the image, return likely channel
    codes for the image
    """

    if sensor_coarse == 'WV':
        if num_bands == 1:
            channels = 'gray'
        elif num_bands == 3:
            channels = 'r|g|b'
        elif num_bands == 8:
            channels = 'wv1|wv2|wv3|wv4|wv4|wv6|wv7|wv8'
            # channels = 'cb|b|g|y|r|wv6|wv7|wv8'
        else:
            raise NotImplementedError
    elif sensor_coarse == 'S2':
        if num_bands == 1:
            channels = 'gray'
        elif num_bands == 3:
            channels = 'r|g|b'
        elif num_bands == 13:
            channels = 's1|s2|s3|s4|s4|s6|s7|s8|s8a|s9|s10|s11|s12'
            # channels = 'cb|b|g|r|s4|s6|s7|s8|s8a|s9|s10|s11|s12'
        else:
            raise NotImplementedError
    elif sensor_coarse == 'LC':
        if num_bands == 1:
            channels = 'gray'
        elif num_bands == 3:
            channels = 'r|g|b'
        elif num_bands == 11:
            channels = 'lc1|lc2|lc3|lc4|lc5|lc6|lc7|lc8|lc9|lc10|lc11'
            # channels = 'cb|b|g|r|lc5|lc6|lc7|pan|lc9|lc10|lc11'
        else:
            raise NotImplementedError

    return channels
