"""
Adds fields needed by ndsampler to correctly "watch" a region.

Some of this is done hueristically. We assume images come from certain sensors.
We assume input is orthorectified.  We assume some GSD "target" gsd for video
and image processing. Note a video GSD will typically be much higher (i.e.
lower resolution) than an image GSD.
"""
import kwcoco
import ubelt as ub
import kwimage

import watch
import numpy as np
from os.path import exists
from os.path import join

#### NOTE: THIS WILL LIKELY CHANGE LOCATION IN THE FUTURE
from ndsampler._transform import Affine


def populate_geo_video_stats(dset, vidid, target_gsd='max-resolution'):
    """
    Example:
        import sys, ubelt
        sys.path.append(ubelt.expandpath('~/code/watch/scripts'))
        from coco_add_watch_fields import *  # NOQA

        import kwcoco
        # TODO: make a demo dataset with some sort of gsd metadata
        dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
        print('dset = {!r}'.format(dset))

        coco_fpath = ub.expandpath('~/data/dvc-repos/smart_watch_dvc/drop0_aligned/data.kwcoco.json')
        dset = kwcoco.CocoDataset(coco_fpath)
        vidid = 1

        target_gsd = 2.8
    """
    # Compute an image-to-video transform that aligns all frames to some
    # common resolution.
    video = dset.index.videos[vidid]
    gids = dset.index.vidid_to_gids[vidid]

    import warnings

    frame_infos = {}

    for gid in gids:
        img = dset.index.imgs[gid]

        if img.get('file_name', None) is None:
            # Choose any one of the auxiliary images, we chose the biggest
            # arbitrarilly
            # TODO: auxiliary
            aux_width = [aux['width'] for aux in img['auxiliary']]
            aux_height = [aux['height'] for aux in img['auxiliary']]
            aux_idx = (np.array(aux_width) * np.array(aux_height)).argmax()
            aux_chosen = img['auxiliary'][aux_idx]
            fpath = join(dset.bundle_dpath, aux_chosen['file_name'])

            try:
                crs_info = watch.gis.geotiff.geotiff_crs_info(fpath)
                aux_to_wld = Affine.coerce(crs_info['pxl_to_wld'])
            except Exception:
                warnings.warn('no crs info for aux')
                aux_to_wld = Affine.eye()

            aux_to_img = Affine.coerce(aux_chosen['warp_aux_to_img'])
            img_to_aux = aux_to_img.inv()
            img_to_wld = aux_to_wld @ img_to_aux
        else:
            fpath = join(dset.bundle_dpath, img['file_name'])

            try:
                crs_info = watch.gis.geotiff.geotiff_crs_info(fpath)
                img_to_wld = Affine.coerce(crs_info['pxl_to_wld'])
            except Exception:
                warnings.warn('no crs info for img')
                img_to_wld = Affine.eye()

        wld_approx_gsd = np.abs(np.linalg.det(img_to_wld))

        frame_infos[gid] = {
            'img_to_wld': img_to_wld,
            # Note: division because gsd is inverted. This got me confused, but
            # I'm pretty sure this works.
            'target_gsd': target_gsd,
            'wld_approx_gsd': wld_approx_gsd,
            'width': img['width'],
            'height': img['height'],
        }

    sorted_gids = ub.argsort(frame_infos, key=lambda x: x['wld_approx_gsd'])
    min_gsd_gid = sorted_gids[0]
    max_gsd_gid = sorted_gids[-1]
    max_example = frame_infos[max_gsd_gid]
    min_example = frame_infos[min_gsd_gid]
    max_gsd = max_example['wld_approx_gsd']
    min_gsd = min_example['wld_approx_gsd']

    import numbers

    if target_gsd == 'max-resolution':
        gsd = min_gsd
    elif target_gsd == 'min-resolution':
        gsd = max_gsd
    else:
        gsd = target_gsd
        if not isinstance(target_gsd, numbers.Number):
            raise TypeError('target_gsd must be a code or number = {}'.format(type(target_gsd)))
    gsd = float(gsd)

    for info in frame_infos.values():
        info['target_gsd'] = gsd
        info['to_target_scale_factor'] = info['wld_approx_gsd'] / gsd

    available_channels = set()
    for img in dset.imgs.values():
        if img.get('fname', None) is not None:
            available_channels.add(img.get('channels', None))
        for aux in img.get('auxiliary', []):
            available_channels.add(aux.get('channels', None))

    # Align to frame closest to the target GSD
    base_gid, info = min(frame_infos.items(),
                         key=lambda kv: 1 - kv[1]['to_target_scale_factor'])
    scale = info['wld_approx_gsd'] / gsd

    # Can add an extra transform here if the video is not exactly in
    # any specific image space
    wld_to_vid = Affine.scale(scale) @ info['img_to_wld'].inv()
    video['width'] = int(np.ceil(info['width'] * scale))
    video['height'] = int(np.ceil(info['height'] * scale))
    video['target_gsd'] = gsd

    # Store metadata in the video
    video['num_frames'] = len(gids)
    video['warp_wld_to_vid'] = wld_to_vid.__json__()
    video['min_gsd'] = min_gsd
    video['max_gsd'] = max_gsd
    video['available_channels'] = sorted(available_channels)

    for gid in gids:
        img = dset.index.imgs[gid]
        img_to_wld = frame_infos[gid]['img_to_wld']
        img_to_vid = wld_to_vid @ img_to_wld
        img['warp_img_to_vid'] = img_to_vid.__json__()

        for aux in img.get('auxiliary', []):
            aux_to_vid = img_to_vid @ Affine.coerce(aux['warp_aux_to_img'])
            aux['warp_img_to_aux'] = aux_to_vid.__json__()

    if 0:
        dset.imgs[min_gsd_gid]
        dset.imgs[max_gsd_gid]
        # inspect
        print(ub.repr2(dset.images(gids).objs, nl=1))
        print(ub.repr2(dset.videos([vidid]).objs, nl=1))


def populate_geo_channels_heuristic(dset, gid, overwrite=False):
    """
    Note: this will not overwrite existing channel info unless specified

    Example:
        >>> import sys, ubelt
        >>> sys.path.append(ubelt.expandpath('~/code/watch/scripts'))
        >>> from coco_add_watch_fields import *  # NOQA
        >>> import kwcoco
        >>> ###
        >>> gid = 1
        >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
        >>> populate_geo_channels_heuristic(dset, gid, overwrite=True)
        >>> ###
        >>> gid = 1
        >>> dset = kwcoco.CocoDataset.demo('shapes8')
        >>> populate_geo_channels_heuristic(dset, gid, overwrite=True)
    """
    def sensor_channel_hueristic(sensor_coarse, num_bands):
        """
        Given a sensor and the number of bands in the image, return likely channel
        codes for the image
        """
        err = 0
        if sensor_coarse == 'WV':
            if num_bands == 1:
                channels = 'gray'
            elif num_bands == 3:
                channels = 'r|g|b'
            elif num_bands == 8:
                channels = 'wv1|wv2|wv3|wv4|wv4|wv6|wv7|wv8'
                # channels = 'cb|b|g|y|r|wv6|wv7|wv8'
            else:
                err = 1
        elif sensor_coarse == 'S2':
            if num_bands == 1:
                channels = 'gray'
            elif num_bands == 3:
                channels = 'r|g|b'
            elif num_bands == 13:
                channels = 's1|s2|s3|s4|s4|s6|s7|s8|s8a|s9|s10|s11|s12'
                # channels = 'cb|b|g|r|s4|s6|s7|s8|s8a|s9|s10|s11|s12'
            else:
                err = 1
        elif sensor_coarse == 'LC':
            if num_bands == 1:
                channels = 'gray'
            elif num_bands == 3:
                channels = 'r|g|b'
            elif num_bands == 11:
                channels = 'lc1|lc2|lc3|lc4|lc5|lc6|lc7|lc8|lc9|lc10|lc11'
                # channels = 'cb|b|g|r|lc5|lc6|lc7|pan|lc9|lc10|lc11'
            else:
                err = 1
        else:
            err = 1
        if err:
            raise NotImplementedError(f'sensor_coarse={sensor_coarse}, num_bands={num_bands}')
        return channels

    def introspect_num_bands(fpath):
        shape = kwimage.load_image_shape(fpath)
        if len(shape) == 1:
            return 1
        elif len(shape) == 3:
            return shape[2]
        else:
            raise Exception(f'unknown format, fpath={fpath}, shape={shape}')

    def num_band_hueristic(num_bands):
        if num_bands == 1:
            channels = 'gray'
        elif num_bands == 3:
            channels = 'r|g|b'
        elif num_bands == 4:
            channels = 'r|g|b|a'
        else:
            raise Exception('num_bands=f{num_bands}')
        return channels

    def populate_canvas_obj(bundle_dpath, obj):
        sensor_coarse = obj.get('sensor_coarse', None)
        num_bands = obj.get('num_bands', None)
        channels = obj.get('channels', None)
        fname = obj.get('file_name', None)
        # Can only do this for images with file names
        if fname is not None:
            if overwrite or num_bands is None:
                fpath = join(bundle_dpath, fname)
                num_bands = introspect_num_bands(fpath)
                obj['num_bands'] = num_bands

            if overwrite or channels is None:
                if sensor_coarse is not None:
                    channels = sensor_channel_hueristic(sensor_coarse, num_bands)
                elif num_bands is not None:
                    channels = num_band_hueristic(num_bands)
                else:
                    raise Exception(ub.paragraph(
                        f'''
                        no methods to introspect channels
                        sensor_coarse={sensor_coarse},
                        num_bands={num_bands}
                        for obj={obj}
                        '''))
                obj['channels'] = channels

    bundle_dpath = dset.bundle_dpath
    img = dset.imgs[gid]

    populate_canvas_obj(bundle_dpath, img)

    auxillary = img.get('auxiliary', [])
    for aux in auxillary:
        populate_canvas_obj(bundle_dpath, aux)


def populate_watch_fields(dset, target_gsd=10.0):
    """
    Aggregate populate function for fields useful to WATCH.

    Args:
        dset (Dataset): dataset to work with
        target_gsd (float): target gsd in meters

    Ignore:
        >>> import sys, ubelt
        >>> sys.path.append(ubelt.expandpath('~/code/watch/scripts'))
        >>> from coco_add_watch_fields import *  # NOQA
        >>> import kwcoco
        >>> # root_dpath = ub.expandpath('~/data/dvc-repos/smart_watch_dvc/extern/onera_2018')
        >>> # coco_fpath = join(root_dpath, 'onera_all.kwcoco.json')
        >>> fpath = ub.expandpath('~/data/dvc-repos/smart_watch_dvc/drop0_aligned/data.kwcoco.json')
        >>> dset = kwcoco.CocoDataset(fpath)
        >>> target_gsd = 10.0
        >>> add_watch_fields(dset, target_gsd)
        >>> # dset.dump(dset.fpath, newlines=True)

        >>> print('dset.index.videos = {}'.format(ub.repr2(dset.index.videos, nl=-1)))
        >>> print('dset.index.imgs[1] = ' + ub.repr2(dset.index.imgs[1], nl=1))

    Example:
        >>> import sys, ubelt
        >>> sys.path.append(ubelt.expandpath('~/code/watch/scripts'))
        >>> from coco_add_watch_fields import *  # NOQA
        >>> import kwcoco
        >>> # TODO: make a demo dataset with some sort of gsd metadata
        >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
        >>> print('dset = {!r}'.format(dset))
        >>> target_gsd = 13.0
        >>> add_watch_fields(dset, target_gsd)

        >>> print('dset.index.imgs[1] = ' + ub.repr2(dset.index.imgs[1], nl=2))
        >>> print('dset.index.videos = {}'.format(ub.repr2(dset.index.videos, nl=1)))
    """
    # Load your KW-COCO dataset (conform populates information like image size)
    dset.conform(pycocotools_info=False)

    for gid in dset.imgs.keys():
        populate_geo_channels_heuristic(dset, gid)

    for vidid in dset.index.videos.keys():
        populate_geo_video_stats(dset, vidid, target_gsd=target_gsd)
